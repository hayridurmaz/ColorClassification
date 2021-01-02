import csv
import math
import operator
import os

import cv2
import numpy as np


# calculation of euclidead distance
def calculateEuclideanDistance(x1, x2, length):
    distance = 0
    for x in range(length):
        distance += pow(x1[x] - x2[x], 2)
    return math.sqrt(distance)


# Load image feature data to training feature vectors and test feature vector
def loadDataset(training_filename, test_filename, training_feature_vector=[], test_feature_vector=[]):
    append_to_vector(training_filename, training_feature_vector)
    append_to_vector(test_filename, test_feature_vector)


def append_to_vector(filename, training_feature_vector):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            training_feature_vector.append(dataset[x])


# get k nearest neigbors
def kNearestNeighbors(training_feature_vector, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(training_feature_vector)):
        dist = calculateEuclideanDistance(testInstance,
                                          training_feature_vector[x], length)
        distances.append((training_feature_vector[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# votes of neighbors
def responseOfNeighbors(neighbors):
    all_possible_neighbors = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1
        else:
            all_possible_neighbors[response] = 1
    sortedVotes = sorted(all_possible_neighbors.items(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def predict(training_data, test_data):
    training_feature_vector = []  # training feature vector
    test_feature_vector = []  # test feature vector
    loadDataset(training_data, test_data, training_feature_vector, test_feature_vector)
    classifier_prediction = []  # predictions
    k = 1  # K value of k nearest neighbor
    for x in range(len(test_feature_vector)):
        neighbors = kNearestNeighbors(training_feature_vector, test_feature_vector[x], k)
        result = responseOfNeighbors(neighbors)
        classifier_prediction.append(result)
    return classifier_prediction


def color_histogram_of_image(image, isTraining, label=None, mask=None):
    green = 0
    blue = 0
    features = []
    counter = 0
    channels = cv2.split(image)
    colors = ('b', 'g', 'r')
    feature_data = ''
    for (idx, col) in zip(channels, colors):  # Loop over the image channels
        # This is in order to get rid off the background colors
        histogramUpperLimit = 250
        histogramLowerLimit = 5
        if label == "White":
            histogramUpperLimit = 256
        if label == "Black":
            histogramLowerLimit = 0
        hist = cv2.calcHist([idx], [0], mask, [256],
                            [histogramLowerLimit, histogramUpperLimit])  # Create a histogram for current channel
        features.extend(hist)
        elem = np.argmax(hist)  # find the peak pixel values for R, G, and B
        if counter == 0:
            blue = str(elem)
        elif counter == 1:
            green = str(elem)
        elif counter == 2:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
        counter = counter + 1
    if isTraining:
        with open('training.data', 'a') as fileToBeWritten:
            fileToBeWritten.write(feature_data + ',' + label + '\n')
    else:
        with open('test.data', 'a') as fileToBeWritten:
            fileToBeWritten.write(feature_data + '\n')
    return feature_data


def training():
    # red color training images
    for color in os.listdir('./training_dataset'):
        path = "./training_dataset/" + color + "/"
        for fileToBeWritten in os.listdir('./training_dataset/' + color):
            image = cv2.imread(path + fileToBeWritten)  # Load the image
            color_histogram_of_image(image, True, color)


def cleanFiles():
    with open('test.data', 'w') as fileToBeWritten:
        fileToBeWritten.write('')
    with open('training.data', 'w') as fileToBeWritten:
        fileToBeWritten.write('')


if __name__ == '__main__':
    cleanFiles()
    training()
    for file in os.listdir('./images'):  # Iterate over test image folder
        img = cv2.imread('./images/' + file)  # Load the test image
        color_histogram_of_image(img, False)  # For each image, write the histogram onto the file
    prediction = predict('training.data', 'test.data')  # get the predictions of test images
    i = 0
    correctPrediction = 0
    classificationRates = {}
    numberOfImage = {}
    for file in os.listdir('./training_dataset'):
        colorname = file.lower()
        classificationRates[colorname] = 0
        numberOfImage[colorname] = 0

    for file in os.listdir('./images'):
        colorname = file[0:file.index("_")].lower()
        if colorname == prediction[i].lower():
            classificationRates[colorname] = classificationRates[colorname] + 1
            correctPrediction = correctPrediction + 1
            print("TRUE: " + file + ' : ' + prediction[i])
        else:
            print("FALSE: " + file + ' : ' + prediction[i])
        i = i + 1
        numberOfImage[colorname] = numberOfImage[colorname] + 1

    for file in os.listdir('./training_dataset'):
        colorname = file.lower()
        accuracy = classificationRates[colorname] / numberOfImage[colorname]
        print(colorname + " accuracy: " + str(accuracy))

    accuracy = correctPrediction / len(prediction)
    print("Total accuracy: " + str(accuracy))

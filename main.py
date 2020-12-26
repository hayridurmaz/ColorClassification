import binascii
import csv
import math
import operator
import os

import cv2
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
from PIL import Image
import matplotlib.pyplot as plt


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
    k = 3  # K value of k nearest neighbor
    for x in range(len(test_feature_vector)):
        neighbors = kNearestNeighbors(training_feature_vector, test_feature_vector[x], k)
        result = responseOfNeighbors(neighbors)
        classifier_prediction.append(result)
    return classifier_prediction[0]


def color_histogram_of_image(img, isTraining, label=None, mask=None):
    features = []
    counter = 0
    channels = cv2.split(img)
    colors = ('b', 'g', 'r')
    feature_data = ''
    for (i, col) in zip(channels, colors):  # Loop over the image channels
        histogramUpperLimit = 250
        histogramLowerLimit = 5
        if label == "White":
            histogramUpperLimit = 256
        if label == "Black":
            histogramLowerLimit = 0
        hist = cv2.calcHist([i], [0], mask, [256],
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
        with open('training.data', 'a') as file:
            file.write(feature_data + ',' + label + '\n')
    else:
        with open('test.data', 'a') as file:
            file.write(feature_data + '\n')
    return feature_data


def training():
    # red color training images
    for color in os.listdir('./training_dataset/colors'):
        path = "./training_dataset/colors/" + color + "/"
        for file in os.listdir('./training_dataset/colors/' + color):
            img = cv2.imread(path + file)  # Load the image
            color_histogram_of_image(img, True, color)


def cleanFiles():
    with open('test.data', 'w') as file:
        file.write('')
    with open('training.data', 'w') as file:
        file.write('')


if __name__ == '__main__':
    # read the test image
    cleanFiles()
    PATH = 'training.data'  # checking whether the training data is ready
    print('training data is being created...')
    open('training.data', 'a')
    training()
    print('training data is ready, classifier is loading...')

    for file in os.listdir('./images'):
        img = cv2.imread('./images/' + file)  # Load the image
        max_histogram = color_histogram_of_image(img, False)
        # get the prediction

        prediction = predict('training.data', 'test.data')
        print('Detected color is:', prediction)

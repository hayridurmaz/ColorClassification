import os

import cv2
import numpy as np


def color_histogram_of_image(img, isTraining, label):
    features = []
    counter = 0
    channels = cv2.split(img)
    colors = ('b', 'g', 'r')
    feature_data = ''
    for (i, col) in zip(channels, colors):  # Loop over the image channels
        hist = cv2.calcHist([i], [0], None, [256], [0, 256])  # Create a histogram for current channel
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
        with open('training.data', 'a') as myfile:
            myfile.write(feature_data + ',' + label + '\n')
    else:
        print("else")


if __name__ == '__main__':
    # data = pd.read_csv("data/final_data.csv")
    # print(data)

    images = []
    path = "images/"
    for image in os.listdir(path):
        images.append(image)

    for image in images:
        img = cv2.imread("%s%s" % (path, image))  # Load the image
        color_histogram_of_image(img, True, "black")

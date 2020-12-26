import binascii
import os

import cv2
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
from PIL import Image
import matplotlib.pyplot as plt


def color_histogram_of_image(img, isTraining, label, mask=None):
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
        with open('training.data', 'a') as myfile:
            myfile.write(feature_data + ',' + label + '\n')
    else:
        print("else")


def training():
    # red color training images
    for color in os.listdir('./training_dataset/colors'):
        path = "./training_dataset/colors/" + color + "/"
        for file in os.listdir('./training_dataset/colors/' + color):
            img = cv2.imread(path + file)  # Load the image
            color_histogram_of_image(img, True, color)


def findDominantColor(im):
    NUM_CLUSTERS = 5

    print('reading image')
    # im = im.resize((150, 150))  # optional, to reduce time
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    print('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences

    index_max = np.argmax(counts)  # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    print('most frequent is %s (#%s)' % (peak, colour))


if __name__ == '__main__':
    # data = pd.read_csv("data/final_data.csv")
    # print(data)

    # images = []
    # path = "images/"
    # for image in os.listdir(path):
    #     images.append(image)
    #
    # for image in images:
    #     img = cv2.imread("%s%s" % (path, image))  # Load the image
    #     color_histogram_of_image(img, True, "black")

    training()

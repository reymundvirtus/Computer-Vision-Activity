import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# original image
image = cv.imread('motor.jpg') # read image
cv.imshow('Original', image) # show original image

def histogram(image):
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # convert to grayscale
    cv.imshow('Grayscale', grayscale) # show grayscale image
    hist = cv.calcHist([grayscale], [0], None, [256], [0, 256]) # calculate histogram
    plt.figure()
    plt.title('Grayscale Histogram')
    plt.xlabel('Bins')
    plt.ylabel('no. of Pixels')
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

def flatten(image):
    chans = cv.split(image) # split image into channels
    colors = ("b", "g", "r") # color names
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("no. of Pixels")
    for (chan, color) in zip(chans, colors):
        hist = cv.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color = color)
        plt.xlim([0, 256])
    plt.show()

histogram(image)
flatten(image)
cv.waitKey(0)
cv.destroyAllWindows()
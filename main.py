import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread('mcdo.png', 0) # read the original image and set to grayscale

kernel = np.ones((5, 5), np.uint8)
erosion = cv.erode(image, kernel, iterations = 1)
dilation = cv.dilate(image, kernel, iterations = 1)
opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
gradient = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)
tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
blackhat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)

cv.imshow("Erosion", erosion)
cv.imshow("Dilation", dilation)
cv.imshow("Opening", opening)
cv.imshow("Closing", closing)
cv.imshow("Gradient", gradient)
cv.imshow("TopHat", tophat)
cv.imshow("BlackHat", blackhat)
cv.waitKey(0)
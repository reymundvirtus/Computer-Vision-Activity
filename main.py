import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread('passport.jpg', 0) # read the original image and set to grayscale
img = cv.medianBlur(image, 5) # blurred the image a little

ret, thres1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY) # Global Thresholding (v = 127)
thres2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
            cv.THRESH_BINARY, 11, 2) # Adaptive Mean Thresholding
thres3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
            cv.THRESH_BINARY, 11, 2) # Adaptive Gaussian Thresholding
titles = ['Original Image (Grayscale)', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding'] # array of titles
images = [image, thres1, thres2, thres3] # array of images

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], 'gray') # iterate images
    plt.title(titles[i]) # iterate titles
    plt.xticks([]) # remove the x axis
    plt.yticks([]) # remove the y axis

cv.imshow("scrat", thres1)
cv.waitKey(0)
plt.show()
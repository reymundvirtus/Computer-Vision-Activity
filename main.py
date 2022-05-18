import cv2 as cv
import numpy as np

# original image
image = cv.imread('coins.jpg') # read image

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # convert to gray
ret, thresh = cv.threshold(gray_image, 200, 255, cv.THRESH_BINARY) # threshold

# CHAIN_APPROX_SIMPLE: only the boundary points are stored
contours, hierarchy = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE, offset=(0, 0))
image_copy = image.copy()
cv.drawContours(image = image_copy, contours = contours, contourIdx = -1, color = (0, 255, 0), thickness = 2, lineType = cv.LINE_AA)

cv.imshow('Simple Approximation', image_copy) # show simple approximation image
# cv.imshow('Binary image', thresh) # show binary image
cv.waitKey(0)
cv.destroyAllWindows()
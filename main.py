import cv2 as cv
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import imutils

# original image
image = cv.imread('coins_sample.jpg') # read image
shifted = cv.pyrMeanShiftFiltering(image, 50, 50) # apply mean shift filter

gray_image = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY) # convert to grayscale
ret, thresh = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# noise removal
kernel = np.ones((5, 4), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 6)
# sure background area
sure_bg = cv.dilate(opening, kernel, iterations = 0)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_C, 5)
ret, sure_fg = cv.threshold(dist_transform, 0 * dist_transform.max(), 255, 0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)

# compute the exact Euclidean distance from every binary
# distance map
D = ndimage.distance_transform_edt(sure_fg)
localMax = peak_local_max(D, indices = False, min_distance = 20, labels = sure_fg)
# perform a connected component analysis on the local peaks
markers = ndimage.label(localMax, structure = np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask = thresh)
print("Number of coins found: {}".format(len(np.unique(labels)) - 1))

# loop over the unique labels returned by the Watershed
for label in np.unique(labels):
    if label == 0: # if the label is zero, we are examining the 'background'
        continue
    # otherwise, allocate memory for the label region and draw it on the mask
    mask = np.zeros(sure_bg.shape, dtype = "uint8")
    mask[labels == label] = 255

    # detect contours in the mask and grab the largest one
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv.contourArea)

    # draw a circle enclosing the object
    ((x, y), r) = cv.minEnclosingCircle(c)
    cv.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv.putText(image, "{}".format(label), (int(x) - 10, int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv.imshow("Output", image)
cv.waitKey(0)
cv.destroyAllWindows()
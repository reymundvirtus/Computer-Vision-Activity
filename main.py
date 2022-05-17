import cv2 as cv
import numpy as np

# original image
image = cv.imread('itachi.jpg')
cv.imshow('Original', image)

mask = np.zeros(image.shape[:2], dtype = np.uint8)
cv.circle(mask, (310, 280), 65, 255, -1) # left eye
cv.circle(mask, (570, 180), 70, 255, -1) # right eye
cv.rectangle(mask, (180, 10), (570, 100), 255, -1) # rouge ninja bandage
cv.imshow('Mask', mask)

masked = cv.bitwise_and(image, image, mask = mask)
cv.imshow('Mask Applied to Image', masked)
cv.waitKey(0)
cv.destroyAllWindows()
import cv2 as cv

image = cv.imread('motor.jpg', 0) # Read original image

rows, cols = image.shape

# cols-1 and rows-1 are the coordinate limits.
fortyFive = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 45, 0.9) # adjusted to 0.9 to fit the image in window
dst45 = cv.warpAffine(image, fortyFive, (cols, rows))

oneEighty = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 180, 1)
dst180 = cv.warpAffine(image, oneEighty, (cols, rows))

ninety = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 0.7) # adjusted to 0.7 to fit the image in window
dst90 = cv.warpAffine(image, ninety, (cols, rows))

twoSeventy = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 270, 0.7) # adjusted to 0.7 to fit the image in window
dst270 = cv.warpAffine(image, twoSeventy, (cols, rows))

cv.imshow('45', dst45) # Show 45 degree image
cv.imshow('180', dst180) # Show 180 degree image
cv.imshow('90', dst90) # Show 90 degree image
cv.imshow('270', dst270) # Show 270 degree image
k = cv.waitKey(0) # wait for key to be pressed
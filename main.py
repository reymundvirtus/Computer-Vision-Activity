import cv2 as cv

image = cv.imread('motor.jpg') # original image
gray_image = cv.imread('motor_gray.jpg') # grayscale image

negativeImage = cv.bitwise_not(image) # invert original image to negative image

# blend the grayscale and negative image
blendImage = cv.addWeighted(gray_image, 1, negativeImage, 0.4, 0)

# cv.imshow('original', image) # show original image
cv.imshow('output', blendImage) # show the blended image
cv.imshow('gray', gray_image)
cv.waitKey(0) # wait for key to be pressed

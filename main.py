import cv2 as cv

image = cv.imread('motor.jpg') # Read original image

grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert to grayscale

blackAndWhiteImage = cv.threshold(grayscale, 90, 255, cv.THRESH_BINARY)[1] # Convert grayscale image to black and white

blackAndWhiteImageCopy = cv.threshold(image, 127, 255, cv.THRESH_BINARY)[1] # Convert original image to black and white
negativeImage = cv.bitwise_not(blackAndWhiteImageCopy) # Invert black and white image to negative image

cv.imshow('Original Image', image) # Show original image
cv.imshow('Gray Image', grayscale) # Show gray image
cv.imshow('Inverted Image', blackAndWhiteImage) # Show inverted image
cv.imshow('Negative Image', negativeImage) # Show negative image
k = cv.waitKey(0) # wait for key to be pressed
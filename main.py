import cv2 as cv
import numpy as np

# original image
image = cv.imread('jd.jpg')
cv.imshow('Original', image)

# brightness image
brightness = cv.convertScaleAbs(image, alpha = 1.5, beta = 5) # adjust brightness to 50%
cv.imshow('Brightness', brightness)

# blur image
blur = cv.blur(image, (10, 10)) # blur image
cv.imshow('Blur', blur)

# emboss image
grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert to grayscale
blackAndWhiteImage = cv.threshold(grayscale, 110, 100, cv.THRESH_BINARY)[1] # Convert grayscale image to black and white
kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
emboss = cv.filter2D(blackAndWhiteImage, -1, kernel)
cv.imshow('Emboss', emboss)

# sepia image
sepia = cv.convertScaleAbs(image, alpha = 1.5, beta = 50)
sepia = cv.cvtColor(sepia, cv.COLOR_BGR2HSV)
sepia[:, :, 2] = sepia[:, :, 2] * (1)
sepia = cv.cvtColor(sepia, cv.COLOR_HSV2BGR)
cv.imshow('Sepia', sepia)

class WarmAndCold:
	def __init__(self, image):
		self.image = image
		self.gamma = 1.5
		self.warm = self.gamma_function(image, self.gamma)
		self.cold = self.gamma_function(image, self.gamma)

	# define gamma function for warm and cold image
	def gamma_function(self, channel, gamma):
		invGamma = 1/gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
						for i in np.arange(0, 256)]).astype("uint8") #creating lookup table
		channel = cv.LUT(channel, table)
		return channel

obj = WarmAndCold(image)
# warm image
image[:, :, 0] = obj.gamma_function(image[:, :, 0], 0.75) # down scaling blue channel
image[:, :, 2] = obj.gamma_function(image[:, :, 2], 1.25) # up scaling red channel
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
hsv[:, :, 1] = obj.gamma_function(hsv[:, :, 1], 1.2) # up scaling saturation channel
warm = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('Warm', warm)

# cold image
image[:, :, 0] = obj.gamma_function(image[:, :, 0], 1.25) # up scaling blue channel
image[:, :, 2] = obj.gamma_function(image[:, :, 2], 0.75) # down scaling red channel
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
hsv[:, :, 1] = obj.gamma_function(hsv[:, :, 1], 0.8) # down scaling saturation channel
cold = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('Cold', cold)

# saving images
# cv.imwrite('brightness.jpg', brightness)
# cv.imwrite('blur.jpg', blur)
# cv.imwrite('emboss.jpg', emboss)
# cv.imwrite('sepia.jpg', sepia)
# cv.imwrite('warm.jpg', warm)
# cv.imwrite('cold.jpg', cold)

cv.waitKey(0)
cv.destroyAllWindows()
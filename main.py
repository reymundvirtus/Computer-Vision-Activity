import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

GREEN = [0,255,0] # color of border
image = cv.imread('google.png') # read the image
image1 = cv.imread('google.png', 0) # read the image and transform it to grayscale to get the shape
rows, cols = image1.shape

rotate90 = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, 1) # rotate to 90 degree
dst90 = cv.warpAffine(image, rotate90, (cols, rows))

original = cv.copyMakeBorder(dst90, 10, 10, 10, 10, cv.BORDER_REPLICATE) # Replicate image
replicate = cv.copyMakeBorder(dst90, 10, 10, 10, 10, cv.BORDER_REPLICATE) # Replicate image
reflect = cv.copyMakeBorder(dst90, 30, 30, 30, 30, cv.BORDER_REFLECT) # Reflect image
reflect101 = cv.copyMakeBorder(dst90, 30, 30, 30, 30, cv.BORDER_REFLECT_101) # Reflect 101 image
wrap = cv.copyMakeBorder(dst90, 30, 30, 30, 30, cv.BORDER_WRAP) # Wrap image
constant = cv.copyMakeBorder(dst90, 30, 30, 30, 30, cv.BORDER_CONSTANT, value = GREEN) # Constant image that has border green

titles = ['ORIGINAL', 'REPLICATE', 'REFLECT', 'REFLECT_101', 'WRAP', 'CONSTANT'] # name of titles
images = [original, replicate, reflect, reflect101, wrap, constant] # name of images

for i in range(len(images)): # iterate the length of images, you can also use the length of titles array
	plt.subplot(2, 3, i + 1) # subplot images by 2 rows and 3 cols
	plt.imshow(images[i], 'gray') # iterate images
	plt.title(titles[i]) # iterate titles

plt.show() # show the figure
import numpy as np
import cv2 as cv

img = np.zeros((512, 512, 3), np.uint8) # Create a black image

cv.ellipse(img, (300, 400), (100, 100), 0, 0, 180, 255, -1) # create a semicirlce

cv.line(img, (0, 0), (511, 511), (255, 255, 0), 5) # create diagonal line

cv.rectangle(img, (300, 0), (510, 128), (255, 0, 255), -1) # create a rectangle

cv.circle(img, (200, 100), 63, (60, 250, 200), -1) # create a circle

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'Reymund', (40, 290), font, 3, (255, 255, 255), 2, cv.LINE_AA) # put text in middle

cv.imshow("Display window", img) # display image
k = cv.waitKey(0) # wait for key to be pressed
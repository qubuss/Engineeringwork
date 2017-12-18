import cv2
import numpy as np

## Read Image
image = cv2.imread('images/eyes.jpg')
imageBackup = image.copy()
## Convert to 1 channel only grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
## CLAHE Equalization
cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe = cl1.apply(gray)
## medianBlur the image to remove noise
blur = cv2.medianBlur(clahe, 7)

## Detect Circles
circles = cv2.HoughCircles(blur ,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=10,maxRadius=19)

print(circles)
for circle in circles[0,:]:
    # draw the outer circle
    cv2.circle(image,(circle[0],circle[1]),circle[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(image,(circle[0],circle[1]),2,(0,0,255),3)

cv2.imshow('Final', image)
cv2.imshow('imageBackup', imageBackup)

cv2.waitKey(0)
cv2.destroyAllWindows()
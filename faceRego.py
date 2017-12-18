import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('xml/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('xml/haarcascade_righteye_2splits.xml')
while True:

    ## Read Image
    ret, image = cap.read()
    ## Convert to 1 channel only grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ## CLAHE Equalization
    cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = cl1.apply(gray)
    ## medianBlur the image to remove noise
    blur = cv2.medianBlur(clahe, 7)
    ## Detect Circles
    faces = face_cascade.detectMultiScale(blur, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            circles = cv2.HoughCircles(blur ,cv2.HOUGH_GRADIENT,1,20,
                                        param1=50,param2=30,minRadius=7,maxRadius=21)
            print(circles)
            try:
                for circle in circles[0,:]:
            # draw the outer circle
                    cv2.circle(image,(circle[0],circle[1]),circle[2],(0,255,0),2)
        # draw the center of the circle
                    cv2.circle(image,(circle[0],circle[1]),2,(0,0,255),3)
            except TypeError:
                print "Oops!  That was no valid number.  Try again..."   

    cv2.imshow('frame',image)
    if cv2.waitKey(1) in [27, ord('q'), 32]:
        break

cap.release()
cv2.destroyAllWindows()
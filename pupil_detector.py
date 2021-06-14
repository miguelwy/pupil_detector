import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('model/face.xml')
eye_cascade = cv2.CascadeClassifier('model/haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    blank_image = np.zeros((600,800,3), np.uint8)
    blank_image.fill(255)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for idx, (ex,ey,ew,eh) in enumerate(eyes):
            eye1_center = None
            eye2_center = None
            if idx == 0:
                roi_eye_color1 = roi_color[ey:ey+eh, ex:ex+ew]
                roi_eye_gray1 = roi_gray[ey:ey+eh, ex:ex+ew]
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                test = cv2.GaussianBlur(roi_eye_gray1,(5,5),0)
                circles = cv2.HoughCircles(test, method=cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=110, param2=20, minRadius=8, maxRadius=15)
                if circles is not None:
                    for circle in circles[0]:
                        cv2.circle(roi_eye_color1, (int(circle[0]), int(circle[1])), int(circle[2]), (255,0,0), thickness=2 )
                        cv2.circle(roi_color, (int(circle[0] + ex), int(circle[1] + ey)), int(circle[2]), (255,0,0), thickness=2 )
                        cv2.circle(roi_color, (int(circle[0] + ex), int(circle[1] + ey)), radius=3, color=(0, 0, 255), thickness=-1)
                        eye1_center = (int(circle[0] + ex), int(circle[1] + ey))
            else:
                roi_eye_color2 = roi_color[ey:ey+eh, ex:ex+ew]
                roi_eye_gray2 = roi_gray[ey:ey+eh, ex:ex+ew]
                
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                test = cv2.GaussianBlur(roi_eye_gray2,(5,5),0)
                circles = cv2.HoughCircles(test, method=cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=110, param2=20, minRadius=8, maxRadius=15)
                if circles is not None:
                    for circle in circles[0]:
                        cv2.circle(roi_eye_color2, (int(circle[0]), int(circle[1])), int(circle[2]), (255,0,0), thickness=2 )
                        cv2.circle(roi_color, (int(circle[0] + ex), int(circle[1] + ey)), int(circle[2]), (255,0,0), thickness=2 )
                        cv2.circle(roi_color, (int(circle[0] + ex), int(circle[1] + ey)), radius=3, color=(0, 0, 255), thickness=-1)
                        eye2_center = (int(circle[0] + ex), int(circle[1] + ey))
                        
            cv2.circle(blank_image, (int(circle[0] + ex), int(circle[1] + ey)), radius=3, color=(0, 0, 255), thickness=-1)
    if roi_eye_color1 is not None:        
        cv2.imshow("Eyes 1", roi_eye_color1)        
    if roi_eye_color2 is not None:        
        cv2.imshow("Eyes 2", roi_eye_color2) 
    cv2.imshow('img',img)
    cv2.imshow('blank',blank_image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
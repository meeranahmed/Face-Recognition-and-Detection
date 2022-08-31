import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import face


def detect_faces(image):

    # Load the pre-trained classifiers for face
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces=face_cascade.detectMultiScale(image, scaleFactor=1.05,minNeighbors=5)

    return faces

def draw_faces(image, faces):

     # Draw a rectangle around the faces
    for (x, y, w, h) in faces: 
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0 , 255), 2)


    return image



# # Read the input image
# img = cv2.imread('./images/lenna.png')

# # Detect faces
# faces = detect_faces(img)


# img=draw_faces(img,faces)


# # Export the result
# cv2.imshow("face_detected.png", img) 
# cv2.waitKey(0)


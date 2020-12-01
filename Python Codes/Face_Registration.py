#!/usr/bin/env python
# coding: utf-8

# # Registering Faces into the Database

# ### Importing Necessary Packages

# In[31]:


import cv2
import os
import numpy as np
import datetime


# ### Setting Webcam to capture Images.

# In[32]:


# Setting up Web-Cam
cam = cv2.VideoCapture(0)


# ### Implementing Haar-Cascade Classifier for Frontal-Face Detection.

# In[33]:


# Implementing Haar-Cascade
haar_xml= "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(haar_xml)


# ## Registering Faces

# In[34]:


# Enter name of Student/Faculty/Staff
face_name = input("Enter Name =>")

# Enter ID [ 1 2 3 . . .]
face_id = input("Enter ID =>")
c = 1

# Capturing 50 Images of the Person.
while(True):
    _ , img = cam.read()
    # Converting to Grayscale Image
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detecting Faces from the Web-Cam
    #detectMultiScale(img_name, scale, minNeighbors)
    face = face_detector.detectMultiScale(g, 1.3, 5)
    
    # Storing faces
    for x,y,w,h in face:
        #cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
        
        # Saving Captured Images in a folder.
        cv2.imwrite("dataset/" + str(face_name)+ "_" + str(face_id)+ "_" + str(c) + ".jpg", g[y:y+h, x:x+w])
        
        # Displaying the Current Camera Input.
        cv2.imshow("Registering Face", img)
        
        c += 1
        
    k = cv2.waitKey(100)
    # Press Esc to stop the process.
    if k == 27:
        break
    # Automatically Stops after taking 50 Samples
    elif c >50:
        break
        
# Deactivationg the Camera.
cam.release()

# Destroy all Created Windows.
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





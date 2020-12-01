#!/usr/bin/env python
# coding: utf-8

# # Face Detection

# ## Importing Necessary Packages

# In[8]:


import csv
import cv2
import os
import time
import numpy as np
from datetime import datetime
from PIL import Image


# ## Defining Classifier and Recogniser

# ### Haar Cascade Classifier
# ### LBPH Face Recognizer

# In[9]:


path = "dataset"
rec = cv2.face.LBPHFaceRecognizer_create()
haar_xml= "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(haar_xml)


# ### Function to Label Images

# In[10]:


def ImageLabel(path):
    IPaths = [os.path.join(path,f) for f in os.listdir(path)]
    Sample = []
    ID = []
    Names = []
    for IPath in IPaths:
        img = np.array(Image.open(IPath).convert('L'))
        l = os.path.split(IPath)[-1].split("_")
        id = int(l[1])
        name = l[0]
        
        # Function to detect Face from an Image.
        faces = face_detector.detectMultiScale(img)
        
        for x,y,w,h in faces:
            Sample.append(img[y:y+h, x:x+w])
            ID.append(id)
            Names.append(name)
    return Sample, ID, Names


# ### Saving the Model as trainer.yml

# In[11]:


Sample, ID, Names = ImageLabel(path)

# Training the Model to achieve better accuracy.
rec.train(Sample,np.array(ID))

# Saving the model to trainer.yml
rec.write('trainer/trainer.yml')


# ### Capturing Faces and Timestamping them

# In[12]:


#Reading the Model from trainer.yml
rec.read('trainer/trainer.yml')

# Font to display on the webcam
font = cv2.FONT_HERSHEY_SIMPLEX

# Configuring WebCam to capture Images
cam = cv2.VideoCapture(0)

#Setting the dimensions of the Capture Window.
cam.set(3,640)
cam.set(4,480)

# Saving the Timestamps with Names in time.csv sheet.
file = open("time.csv",'w',newline='')
filewriter = csv.writer(file)

# Writing the Header Row in the csv sheet.
filewriter.writerow(['DATE','TIME','NAME'])

# Minimum Height and Width of the Detector Box.
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# Variables used for Comparision.
test = "Unknown"
id = "Unknown"
wr = [datetime.now().strftime("%d/%m/%Y"),datetime.now().strftime("%H:%m:%S"), id]
pwr = [datetime.now().strftime("%d/%m/%Y"),datetime.now().strftime("%H:%m:%S"),id]

# To be run continously until Interrupted or Stopped.
while(True):
    
    # Read the WebCam Image frame by frame.
    ret, img = cam.read()
    
    # Converting BGR images to Grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detecting Faces which are Currently available through Webcam.
    faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize = (int(minW), int(minH)))
    
    # For one or more faces
    for x,y,w,h in faces:
        
        # Creating a rectangle around the detected face.
        # v2.rectangle(image, start_point, end_point, color, thickness)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 2)
        
        # Returns the ID number of the image along with the Confidence of the Image.
        # Confidence is a method of measuring a feature with respect to the trained model.
        # Similar to Accuracy. Lies in 0 (Match) to 100 (No Match)
        id, confidence = rec.predict(gray[y:y+h,x:x+w])
        
        # If a registered person is detected. (50% threshold)
        if confidence < 50:
            idx = ID.index(id)
            
            # Extracts the name corresponding to the id returned by above prediction.
            id = Names[idx]
            
            # Timestamping the exact Moment down to the second.
            # DD/MM/YY HH:MM:SS
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%m:%S")
            
            # Creating a Object to store the exact Timestamp along with the name of detected person.
            wr = [datetime.now().strftime("%d/%m/%Y"),datetime.now().strftime("%H:%m:%S"), id]
            
            # If Timestamps are different
            if pwr[1] != wr[1] and id!= "Unknown":

                # Write into the Sheet
                filewriter.writerow(wr)
            confidence = '{0}%'.format(round(100-confidence))
        
        # If non-registered person is detected.
        else:
            id = "Unknown"
            confidence = '{0}%'.format(round(100-confidence))
            
            
        # Writing Text above the detection box.
        cv2.putText(img, str(id), (x, y-5), font, 0.5, (0,255,255), 2)
        cv2.putText(img, str(confidence), (x+w-50, y-5), font, 0.5, (0,0,255), 2)
    
    # Displaying the Camera Stream.
    cv2.imshow("Camera", img)
    
    test = id
    pwr = wr
    
    # Press Esc to stop execution.
    k = cv2.waitKey(100)
    if k == 27:
        file.close()
        break
    
# Deactivationg the Camera.
cam.release()

# Destroy all Created Windows.
cv2.destroyAllWindows()


# In[ ]:





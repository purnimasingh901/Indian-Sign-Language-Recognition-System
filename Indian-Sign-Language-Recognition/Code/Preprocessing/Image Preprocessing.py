#Importing the required libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import imageio

# Set path to dataset images

image_path='C:/Users/PURNIMA SINGH/OneDrive/Desktop/Indian'

# Load folder into array-image_files and return the array 

def loadImages(path,label): 
  image_files=sorted([os.path.join(path,label,file)
   for file in os.listdir(path+str('/')+label) if file.endswith('.jpg')
  ])
  return image_files

# Function to display images

def display(img,title="Original"):
    plt.imshow(img,cmap='gray'),plt.title(title)
    plt.show()

# Preprocessing all the images to extract ROI i.e. hands

def preprocess_images(data, label):
    count = 0
    path = 'C:/Users/PURNIMA SINGH/OneDrive/Desktop/p1' + label # Update the path to include the label folder
    os.makedirs(path, exist_ok=True) # Create the label folder if it doesn't exist
    for image in data:
        img = imageio.imread(image)

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        skin_color_lower = np.array([0, 40, 30], np.uint8)
        skin_color_upper = np.array([43, 255, 255], np.uint8)

        skin_mask = cv2.inRange(hsv_img, skin_color_lower, skin_color_upper)
        skin_mask = cv2.medianBlur(skin_mask, 5)
        skin_mask = cv2.addWeighted(skin_mask, 0.5, skin_mask, 0.5, 0.0)

        kernel = np.ones((5, 5), np.uint8)

        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        hand = cv2.bitwise_and(gray_img, gray_img, mask=skin_mask)

        canny = cv2.Canny(hand, 60, 60)
        
        filename = f"{count}.png" # Use a unique filename for each image
        filepath = os.path.join(path, filename) # Create the complete filepath
        cv2.imwrite(filepath, canny)
        count += 1


# Getting path to all images and preprocessing the images

signs=['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
for label in signs:
    images=[]
    print('Preprocessing images of ',label)
    images=loadImages(image_path,label)
    print('Total images of ',label,' are ',len(images))
    preprocess_images(images,label)
    print('Preprocessing done for ',label)
# Importing Required libraries

import os
import numpy as np
import shutil
import random

# Defining paths to root folders

root_dir='C:/Users/PURNIMA SINGH/OneDrive/Desktop'
src='C:/Users/PURNIMA SINGH/OneDrive/Desktop/p1/'
train_path=root_dir+'/Train/'
test_path=root_dir+'/Test/'

classes_dir=['1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

test_ratio=0.2

#Iterating through all the classes

for cls in classes_dir:

  #Creating Train and Test folders

  os.makedirs(train_path+cls)
  os.makedirs(test_path+cls)

  # Dividing images into train and test set
    
  all_file_names=os.listdir(src+cls)
  np.random.shuffle(all_file_names)
  train_file_names,test_file_names=np.split(np.array(all_file_names),[int(len(all_file_names)*(1-test_ratio))])
  
  train_file_names=[src+'/'+cls+'/'+name for name in train_file_names.tolist()]
  test_file_names=[src+'/'+cls+'/'+name for name in test_file_names.tolist()]

  print('In class ',cls )
  print('Total Images ',len(all_file_names))
  print('Training images ',len(train_file_names))
  print('Test images ',len(test_file_names))

  # copy - pasting images
  for name in train_file_names:
    shutil.copy(name, root_dir+'/Train/'+cls)
  print("Training complete!")

  for name in test_file_names:
    shutil.copy(name, root_dir+'/Test/'+cls)
  print('Testing complete!')

  
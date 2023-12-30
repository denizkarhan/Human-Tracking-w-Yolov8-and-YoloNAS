####################
##### YOLO v8  #####
####################

# from roboflow import Roboflow
# rf = Roboflow(api_key="********************")
# project = rf.workspace("human-det").project("human-pspfa")
# dataset = project.version(1).download("yolov8")

# from roboflow import Roboflow
# rf = Roboflow(api_key="********************")
# project = rf.workspace("ml-wfcmp").project("human-detection-dkxci")
# dataset = project.version(1).download("yolov8")

'''
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="********************")
project = rf.workspace("sebastian-olsson-hiqoj").project("human-detection-ozwbv")
dataset = project.version(1).download("yolov8")
'''

'''
!pip install ultralytics

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

model = YOLO("yolov8s.pt")
dict_classes = model.model.names
'''

'''
import yaml

data = {'train' :  '/content/Human-Detection-1/train/images',
        'val' :  '/content/Human-Detection-1/valid/images',
        'test' :  '/content/Human-Detection-1/test/images',
        'nc': 1,
        'names': ['person']}
'''

'''
# overwrite the data to the .yaml file
with open('/content/Human-Detection-1/data.yaml', 'w') as f:
    yaml.dump(data, f)

# read the content in .yaml file
with open('/content/Human-Detection-1/data.yaml', 'r') as f:
    person_yaml = yaml.safe_load(f)
    display(person_yaml)
'''

'''
{
  'nc': 1,
  'names': ['person'],
  'test': '/content/Human-Detection-1/test/images',
  'train': '/content/Human-Detection-1/train/images',
  'val': '/content/Human-Detection-1/valid/images'
}
'''

'''
model.train(data='/content/Human-Detection-1/data.yaml',
            epochs=50,
            batch=8,
            verbose=True,)
'''
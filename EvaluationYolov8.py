import cv2
from ultralytics import YOLO
import pandas as pd
import pyresearch
import numpy as np

image = cv2.imread("messi_ronaldo.jpg")
model = YOLO("yolov8n.pt")

results = model(image)
px = pd.DataFrame(results[0].boxes.data).astype("float")
detections = np.empty((0, 5))

for index, r in px.iterrows():
    x1, y1, x2, y2, acc = int(r[0]), int(r[1]), int(r[2]), int(r[3]), r[4]

    currentArray = np.array([x1, y1, x2, y2, r[4]])
    detections = np.vstack((detections, currentArray))

    cv2.rectangle(image,(x1,y1),(x2,y2),(255,100,42),2)
    pyresearch.putTextRect(image, "person " + str(acc)[:6], (x1,y1), scale=1.5, colorT=(100, 0, 205), colorR=(60, 60, 60))

cv2.imshow("Detection", image)
cv2.imwrite("yolov8n.jpg", image)
cv2.waitKey(0)

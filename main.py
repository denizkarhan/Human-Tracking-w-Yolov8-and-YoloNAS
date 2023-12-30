import cv2
import pandas as pd
import numpy as np
import pyresearch
from sort import *
from ultralytics import YOLO
from rules import calculateMamdani, viewTables

''' Pixel Locations
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
'''

cap = cv2.VideoCapture('./video/people.mp4')
model = YOLO('./weights/yolov8n.pt')
tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)

xy = [[[312, 388], [289, 390], [474, 469], [497, 462]],
       [[279, 392], [250, 397], [423, 477], [454, 469]]]

f1, f2, frame = 0, 0, None
x1, y1, x2, y2, id = 0, 0, 0, 0, 0
peopleEntering, enteringId, entryId, exitId = [], [], [], []
peopleTrails, enterExitControl, peopleEnteringX1, peopleEnteringRate = {}, {}, {}, {}

def trails():
    global peopleTrails, id, frame

    if len(peopleTrails[id]) > 10: peopleTrails[id].pop(0)
    for idx in range(len(peopleTrails[id])):
        i = peopleTrails[id][idx]
        cv2.rectangle(frame,(int((i[0] + i[2]) / 2) - 1,
                             int((i[1] + i[3]) / 2) - 1),
                             ((int((i[0] + i[2]) / 2) + 1,
                               int((i[1] + i[3]) / 2) + 1)),
                               (5 * idx + 50, 18 * idx, 20 * idx + 50), 3)
def greenArea():
    global xy, f1, f2, frame

    color1 = (0, 0, 255) if f2 > 0 else (0, 255, 0)
    color2 = (255, 0, 0) if f1 > 0 else (0, 255, 0)
    cv2.polylines(frame, [np.array(xy[0], np.int32)], True, color1, 2)
    cv2.polylines(frame, [np.array(xy[1], np.int32)], True, color2, 2)
def areaControl(areaColor):
    global enterExitControl, f1, f2, x1, x2, id

    if (x1 + x2) / 2 > 275 and (x1 + x2) / 2 < 320 :
        f1 = 1
        if len(areaColor) > 0 and areaColor not in enterExitControl[id]: enterExitControl[id] += "b"
    if (x1 + x2) / 2 > 320:
        f2 = 1
        if len(areaColor) > 0 and areaColor not in enterExitControl[id]: enterExitControl[id] += "r"
def personEntering():
    global peopleEntering, enteringId, enterExitControl, exitId, x1, x2, id

    if (x1 + x2) / 2 > 320 and id not in peopleEntering:
        peopleEntering.append(id)
        peopleEnteringX1[id] = x1 + int((x2 - x1) / 2)

    if id in peopleEntering:
        tempRate = ((x1 + x2) / 2 - peopleEnteringX1[id]) / (420 - peopleEnteringX1[id])
        if (tempRate > 0 and tempRate < 1):
            peopleEnteringRate[id] = tempRate
        if (tempRate > 1):
            peopleEnteringRate[id] = 1
        if tempRate > 0.60 and id not in enteringId:
            enteringId.append(id)
    if id not in exitId and enterExitControl.get(id) and enterExitControl[id] == "rb":
        exitId.append(id)
def initParameter(res):
    global enterExitControl, enterExitControl, peopleEnteringRate
    global x1, y1, x2, y2, id

    x1, y1, x2, y2, id = int(res[0]), int(res[1]), int(res[2]), int(res[3]), int(res[4])
    if not enterExitControl.get(id): enterExitControl[id] = ""
    if not peopleTrails.get(id): peopleTrails[id] = []
    
    peopleTrails[id].append([x1, y1, x2, y2])
    peopleEnteringRate[id] = 0

    return x1, y1, x2, y2, id
def predictModel():
    global frame, model

    frame = cv2.resize(frame, (1020, 500))
    imgGraphics = cv2.imread("./image/entryExit.png", cv2.IMREAD_UNCHANGED)
    frame = pyresearch.overlayPNG(frame, imgGraphics, (10, 15))

    return model.predict(frame, verbose=False)

def viewRatePerson():
    global x1, x2, y1, id, frame

    pyresearch.putTextRect(frame, "Rate %" + f'{(peopleEnteringRate[id] * 100)}'[:6], (int((x1 + x2) / 2 - 43), y1 - 12), scale=1, thickness=1, offset=1, border=1, colorB=(255, 255, 255))
    pyresearch.putTextRect(frame, "Person " + f'{int(id)}', (int((x1 + x2) / 2 - 43), y1), scale=1, thickness=1, offset=1, border=1, colorB=(255, 255, 255))
def viewEnterExit(resultsTracker):
    global enteringId, entryId, exitId, frame

    for i in enteringId:
        a = [person[-1] for person in resultsTracker]
        if i not in a and i not in entryId:
            entryId.append(i)
            break
    pyresearch.putTextRect(frame, f'{len(entryId)}', (170, 57), scale=3, colorT=(100, 0, 205), colorR=(60, 60, 60))
    pyresearch.putTextRect(frame, f'{len(exitId)}', (170, 125), scale=3, colorT=(100, 0, 205), colorR=(60, 60, 60))
def viewResult(resultsTracker):
    global xy, f1, f2, frame

    viewTrackerTrails()
    viewEnterExit(resultsTracker)

    greenArea()
    cv2.imshow("RGB", frame)
def viewDetections(px):
    global f1, f2
    f1, f2, color = 0, 0, ""

    detections = np.empty((0, 5))
    
    for index, r in px.iterrows():
        if r[4] > 0.70 and int(r[5]) == 0:
            x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            currentArray = np.array([x1, y1, x2, y2, r[4]])
            detections = np.vstack((detections, currentArray))
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,100,42), 2)
            cv2.putText(frame, "person", (x1,y1), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,255,255), 1)

            color = calculateMamdani(x1, y1, x2, y2, int(r[4]))
    return detections, color
def viewTrackerTrails():
    global peopleTrails, entryId
    for key in peopleTrails:
        if key in entryId:
            if len(peopleTrails[key]) > 0:
                for idx in range(len(peopleTrails[key])):
                    i = peopleTrails[key][idx]
                    cv2.rectangle(frame,(int((i[0] + i[2]) / 2) - 1, int((i[1] + i[3]) / 2) - 1),((int((i[0] + i[2]) / 2) + 1, int((i[1] + i[3]) / 2) + 1)), (5 * idx + 50, 18 * idx, 20 * idx + 50), 3)
                peopleTrails[key].pop(0)

while True:
    for i in range(3): ret, frame = cap.read()
    if not ret: break

    results = predictModel()
    px = pd.DataFrame(results[0].boxes.data).astype("float")

    detections, areaColor = viewDetections(px)
    resultsTracker = tracker.update(detections)

    for res in resultsTracker:
            initParameter(res)
            areaControl(areaColor)
            trails()
            personEntering()
            viewRatePerson()

    viewResult(resultsTracker)

    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

import cv2
import pandas as pd
import numpy as np
import pyresearch
from sort import *
from ultralytics import YOLO

cap = cv2.VideoCapture('people.mp4')
model = YOLO('yolov8n.pt')
tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)

xy = [[[312, 388], [289, 390], [474, 469], [497, 462]],
       [[279, 392], [250, 397], [423, 477], [454, 469]]]

# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE :  
#         colorsBGR = [x, y]
#         print(colorsBGR)
def greenArea(frame, xy, f1, f2):
    color1 = (0, 0, 255) if f2 > 0 else (0, 255, 0)
    color2 = (255, 0, 0) if f1 > 0 else (0, 255, 0)
    cv2.polylines(frame, [np.array(xy[0], np.int32)], True, color1, 2)
    cv2.polylines(frame, [np.array(xy[1], np.int32)], True, color2, 2)
    # cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_TRIPLEX, (0.5), (0, 0, 0), 1)
    # cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_TRIPLEX, (0.5), (0, 0, 0), 1)

cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)

frameCount = 0
enteringId = []
entryId = []
exitId = []
peopleEntering = []
enterExitControl = {}
peopleEnteringX1 = {}
peopleEnteringRate = {}
while True:
    ret, frame = cap.read()
    if not ret: break

    frameCount += 1
    if frameCount % 2 != 0: continue

    frame = cv2.resize(frame, (1020, 500))
    imgGraphics = cv2.imread("entryExit.png", cv2.IMREAD_UNCHANGED)
    frame = pyresearch.overlayPNG(frame, imgGraphics, (10, 15))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    f1, f2, redAreaFlag = 0, 0, 0
    detections = np.empty((0, 5))

    for index, row in px.iterrows():
        if row[4] > 0.60 and int(row[5]) == 0:            
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            currentArray = np.array([x1, y1, x2, y2, row[4]])
            detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            peopleEnteringRate[id] = 0
            if not enterExitControl.get(id):
                enterExitControl[id] = ""
            if (x1 + x2) / 2 > 275 and (x1 + x2) / 2 < 320 :
                f1 = 1
                if "b" not in enterExitControl[id]: enterExitControl[id] += "b"
            if (x1 + x2) / 2 > 320:
                f2 = 1
                if "r" not in enterExitControl[id]: enterExitControl[id] += "r"
            if enterExitControl.get(id):
                print("enterExitControl: ", enterExitControl[id])

            if (x1 + x2) / 2 > 320 and id not in peopleEntering:
                peopleEntering.append(id)
                peopleEnteringX1[id] = x1 + int((x2 - x1) / 2)

            if id in peopleEntering:
                tempRate = ((x1 + x2) / 2 - peopleEnteringX1[id]) / (420 - peopleEnteringX1[id])
                if (tempRate > 0 and tempRate < 1): peopleEnteringRate[id] = tempRate
                if (tempRate > 1): peopleEnteringRate[id] = 1
                if tempRate > 0.65 and id not in enteringId: enteringId.append(id)
            if id not in exitId and enterExitControl.get(id) and enterExitControl[id] == "rb":
                exitId.append(id)

            pyresearch.putTextRect(frame, "Rate %" + f'{(peopleEnteringRate[id] * 100)}'[:6], (int((x1 + x2) / 2 - 43), y1 - 12), scale=1, thickness=1, offset=1, border=1, colorB=(255, 255, 255))
            pyresearch.putTextRect(frame, "Person " + f'{int(id)}', (int((x1 + x2) / 2 - 43), y1), scale=1, thickness=1, offset=1, border=1, colorB=(255, 255, 255))
    for i in enteringId:
        a = [person[-1] for person in resultsTracker]
        if i not in a and i not in entryId:
            entryId.append(i)
            break

    pyresearch.putTextRect(frame, f'{len(entryId)}', (170, 57), scale=3, colorT=(100, 0, 205), colorR=(60, 60, 60))
    pyresearch.putTextRect(frame, f'{len(exitId)}', (170, 125), scale=3, colorT=(100, 0, 205), colorR=(60, 60, 60))

    greenArea(frame, xy, f1, f2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(0) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

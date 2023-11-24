import cv2
import pandas as pd
import numpy as np
import pyresearch
from sort import *
from ultralytics import YOLO

cap = cv2.VideoCapture('people.mp4')
model = YOLO('humanDetect.pt')
tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)

xy = [[[312, 388], [289, 390], [474, 469], [497, 462]],
       [[279, 392], [250, 397], [423, 477], [454, 469]]]

# def RGB(event, x, y, flags, param):
    # if event == cv2.EVENT_MOUSEMOVE :  
        # colorsBGR = [x, y]
        # print(colorsBGR)
def greenArea(frame, xy, f1, f2):
    color1 = (0, 0, 255) if f2 > 0 else (0, 255, 0)
    color2 = (255, 0, 0) if f1 > 0 else (0, 255, 0)
    cv2.polylines(frame, [np.array(xy[0], np.int32)], True, color1, 2)
    cv2.polylines(frame, [np.array(xy[1], np.int32)], True, color2, 2)
    # cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_TRIPLEX, (0.5), (0, 0, 0), 1)
    # cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_TRIPLEX, (0.5), (0, 0, 0), 1)

# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)

frameCount = 0
peopleEntering, enteringId, entryId, exitId = [], [], [], []
peopleTrails, enterExitControl, peopleEnteringX1, peopleEnteringRate = {}, {}, {}, {}

while True:
    ret, frame = cap.read()
    if not ret: break

    frameCount += 1
    if frameCount % 2 != 0: continue

    frame = cv2.resize(frame, (1020, 500))
    imgGraphics = cv2.imread("entryExit.png", cv2.IMREAD_UNCHANGED)
    frame = pyresearch.overlayPNG(frame, imgGraphics, (10, 15))

    results = model.predict(frame, verbose=False)
    px = pd.DataFrame(results[0].boxes.data).astype("float")
    f1, f2, redAreaFlag = 0, 0, 0
    detections = np.empty((0, 5))

    for index, r in px.iterrows():
        if r[4] > 0.60 and int(r[5]) == 0:
            x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            currentArray = np.array([x1, y1, x2, y2, r[4]])
            detections = np.vstack((detections, currentArray))

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,100,42),2)
            cv2.putText(frame,"person",(x1,y1),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)

    resultsTracker = tracker.update(detections)
    for res in resultsTracker:
            x1, y1, x2, y2, id = int(res[0]), int(res[1]), int(res[2]), int(res[3]), int(res[4])
            peopleEnteringRate[id] = 0

            if not enterExitControl.get(id): enterExitControl[id] = ""
            if not peopleTrails.get(id): peopleTrails[id] = []
            peopleTrails[id].append([x1, y1, x2, y2])
            if len(peopleTrails[id]) > 10: peopleTrails[id].pop(0)
            for idx in range(len(peopleTrails[id])):
                i = peopleTrails[id][idx]
                cv2.rectangle(frame,(int((i[0] + i[2]) / 2) - 1, int((i[1] + i[3]) / 2) - 1),((int((i[0] + i[2]) / 2) + 1, int((i[1] + i[3]) / 2) + 1)), (5 * idx + 50, 18 * idx, 20 * idx + 50), 3)

            if (x1 + x2) / 2 > 275 and (x1 + x2) / 2 < 320 :
                f1 = 1
                if "b" not in enterExitControl[id]: enterExitControl[id] += "b"
            if (x1 + x2) / 2 > 320:
                f2 = 1
                if "r" not in enterExitControl[id]: enterExitControl[id] += "r"

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
    for key in peopleTrails:
        if key in entryId:
            if len(peopleTrails[key]) > 0:
                for idx in range(len(peopleTrails[key])):
                    i = peopleTrails[key][idx]
                    cv2.rectangle(frame,(int((i[0] + i[2]) / 2) - 1, int((i[1] + i[3]) / 2) - 1),((int((i[0] + i[2]) / 2) + 1, int((i[1] + i[3]) / 2) + 1)), (5 * idx + 50, 18 * idx, 20 * idx + 50), 3)
                peopleTrails[key].pop(0)

    pyresearch.putTextRect(frame, f'{len(entryId)}', (170, 57), scale=3, colorT=(100, 0, 205), colorR=(60, 60, 60))
    pyresearch.putTextRect(frame, f'{len(exitId)}', (170, 125), scale=3, colorT=(100, 0, 205), colorR=(60, 60, 60))

    greenArea(frame, xy, f1, f2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

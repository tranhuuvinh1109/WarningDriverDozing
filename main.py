import torch
import numpy as np
import cv2


model = torch.hub.load('ultralytics/yolov5', 'custom', path='exp23/weights/last.pt', force_reload=True)

cap = cv2.VideoCapture(0)
time_awake = 0
time_drowsy = 0


def check_drowsy(value_awake, value_drowsy):
    if value_awake > 15:
        print("--->awake", value_awake)
    #         This is the state of the driver awake
    if value_drowsy > 15:
        print("--->drowsy", value_drowsy)
    #         This is the state of a drowsy driver and the car will turn on the bell and warning light
    else:
        print("Face not detected!!!")


while cap.isOpened():
    ret, frame = cap.read()

    # Make detections
    results = model(frame)

    cv2.imshow('YOLO', np.squeeze(results.render()))
    if results.pandas().xyxy[0].to_json(orient="records"):
        if "awake" in results.pandas().xyxy[0].to_json(orient="records"):
            time_awake += 1
            time_drowsy = 0
            check_drowsy(time_awake, time_drowsy)

        if "drowsy" in results.pandas().xyxy[0].to_json(orient="records"):
            time_awake = 0
            time_drowsy += 1
            check_drowsy(time_awake, time_drowsy)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

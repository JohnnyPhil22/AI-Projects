import cv2, joblib
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque

YOLO_MODEL_PATH = "C:\\Users\\Jonathan Philips\\Coding\\AI-Projects\\asl-detect\\runs\\detect\\train\\weights\\best.pt"
RF_MODEL_PATH = "C:\\Users\\Jonathan Philips\\Coding\\AI-Projects\\asl-detect\\asl_landmark_model.pkl"

yolo_model = YOLO(YOLO_MODEL_PATH)

bundle = joblib.load(RF_MODEL_PATH)
rf_model = bundle["model"]
rf_classes = bundle["class_names"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pred_history = deque(maxlen=10)


def get_landmark_vector(hand_landmarks):
    vals = []
    for lm in hand_landmarks.landmark:
        vals.extend([lm.x, lm.y, lm.z])
    return np.array(vals, dtype=np.float32)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 1) YOLO detection
        results = yolo_model(frame, imgsz=640, verbose=False)

        best_box = None
        best_conf = 0.0

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                if conf > best_conf:
                    best_conf = conf
                    best_box = (int(x1), int(y1), int(x2), int(y2))

        if best_box is not None:
            x1, y1, x2, y2 = best_box

            # Clamp + pad
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w - 1, x2 + pad)
            y2 = min(h - 1, y2 + pad)

            roi = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 2) MediaPipe on ROI
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            mp_result = hands.process(roi_rgb)

            if mp_result.multi_hand_landmarks:
                hand = mp_result.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(roi, hand, mp_hands.HAND_CONNECTIONS)

                vec = get_landmark_vector(hand)
                if vec.shape[0] == 63:
                    pred_idx = rf_model.predict([vec])[0]
                    pred_label = rf_classes[pred_idx]
                    pred_history.append(pred_label)

            roi_small = cv2.resize(roi, (200, 200))
            frame[
                10 : 10 + roi_small.shape[0], w - 10 - roi_small.shape[1] : w - 10
            ] = roi_small

        if len(pred_history) > 0:
            vals, counts = np.unique(list(pred_history), return_counts=True)
            stable_label = vals[np.argmax(counts)]
            text = stable_label
        else:
            text = "..."

        cv2.putText(
            frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3
        )

        cv2.imshow("YOLOv8 + MediaPipe + RF ASL Recognizer", frame)

        key = cv2.waitKey(1) & 0xFF
        if (
            key == ord("q")
            or key == 27
            or cv2.getWindowProperty(
                "YOLO + MediaPipe + RF ASL Recognizer", cv2.WND_PROP_VISIBLE
            )
            < 1
        ):
            break

cap.release()
cv2.destroyAllWindows()

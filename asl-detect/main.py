import cv2, joblib, traceback
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
from pathlib import Path


def run():
    print(">>> Entered run()")

    # ---------- PATHS ----------
    BASE_DIR = Path(__file__).resolve().parent
    yolo_path = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
    rf_path = BASE_DIR / "asl_landmark_model.pkl"

    print("    YOLO path:", yolo_path)
    print("    RF path:", rf_path)

    if not yolo_path.exists():
        print("!!! YOLO model files are missing. Exiting run().")
        return
    elif not rf_path.exists():
        print("!!! RF model file is missing. Exiting run().")
        return

    # ---------- LOAD MODELS ----------
    print("    Loading YOLO model...")
    yolo_model = YOLO(str(yolo_path))
    print("    Loaded YOLO.")

    print("    Loading RF model...")
    bundle = joblib.load(rf_path)
    rf_model = bundle["model"]
    rf_classes = bundle["class_names"]
    print("    Loaded RF. Classes:", rf_classes)

    # ---------- MEDIAPIPE ----------
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    pred_history = deque(maxlen=10)

    def get_landmark_vector(hand_landmarks):
        vals = []
        for lm in hand_landmarks.landmark:
            vals.extend([lm.x, lm.y, lm.z])
        return np.array(vals, dtype=np.float32)

    # ---------- CAMERA ----------
    print("    Accessing camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("!!! Could not open any camera. Exiting run().")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        print("    >>> Entering main loop. Press 'q' or ESC to quit.")
        best_box = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("!!! Failed to read frame from camera. Exiting loop.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_count += 1

            # Only run YOLO every 3 frames
            if frame_count % 3 == 0 or best_box is None:
                results = yolo_model(frame, imgsz=416, verbose=False)
                best_box = None
                best_conf = 0.0

                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        if conf > best_conf:
                            best_conf = conf
                            best_box = (int(x1), int(y1), int(x2), int(y2))

            # From here on, use best_box (even if it came from a previous frame)
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w - 1, x2 + pad)
                y2 = min(h - 1, y2 + pad)

                roi = frame[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

                # show ROI in corner
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

            cv2.imshow("YOLO + MediaPipe + RF ASL Recognizer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("    >>> Quit key pressed, breaking loop.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(">>> ERROR:", repr(e))
        traceback.print_exc()
    else:
        print(">>> Exited run() without exception.")
    print(">>> Goodbye!")

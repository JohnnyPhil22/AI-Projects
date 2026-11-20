import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task"
)

# The Tasks API no longer ships HAND_CONNECTIONS / drawing_utils, so define them here
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


class Finger_counter:
    # Initializing the class with default finger coordinates and thumb coordinates
    # The finger coordinates are the indices of the landmarks for each finger
    def __init__(
        self,
        finger_coor: list = [(8, 6), (12, 10), (16, 14), (20, 18)],
        thum_coor: tuple = (4, 3),
    ) -> None:

        # download the model once if it isn't there yet
        if not os.path.exists(MODEL_PATH):
            print("Downloading hand_landmarker.task ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=4,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self._last_ts = 0

        # Setting the finger coordinates
        self.finger_coor = finger_coor
        self.thum_coor = thum_coor

    def _find_nearest_hand_index(self, image: np.ndarray, results: type) -> int:
        height, width = image.shape[:2]
        best_index, best_size = 0, -1.0

        for i, hand in enumerate(results.hand_landmarks):
            wrist, middle_base = hand[0], hand[9]
            size = np.hypot(
                (wrist.x - middle_base.x) * width, (wrist.y - middle_base.y) * height
            )
            if size > best_size:
                best_index, best_size = i, size

        return best_index

    # Counting the number of fingers that are up based on the landmarks detected by the hand landmark model
    def count_fingers(
        self,
        image: np.ndarray,
        text_center_coor: tuple = (20, 50),
        font: int = cv2.FONT_HERSHEY_DUPLEX,
        font_scale: float = 1.5,
        color: tuple = (0, 0, 255),
    ) -> tuple:
        # Tasks API wants an RGB mp.Image and a strictly increasing timestamp (ms)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        results = self.hands.detect_for_video(mp_image, ts)

        finger_count = 0
        if results.hand_landmarks:
            nearest = self._find_nearest_hand_index(image, results)
            landmarks = self._extract_landmarks(image, results, nearest)
            self._draw_landmarks(image, results, landmarks, nearest)
            finger_count = self._check_finger_positions(landmarks, finger_count)

        cv2.putText(
            image,
            f"Number: {finger_count}",
            text_center_coor,
            font,
            font_scale,
            color,
        )

        return image, finger_count

    # Draws the hand landmarks and connections on the image, highlighting the selected hand in yellow and the others in gray. It also draws filled circles for the landmarks of the selected hand.
    def _draw_landmarks(
        self,
        image: np.ndarray,
        results: type,
        landmarks: list,
        selected_index: int = 0,
        radius: int = 5,
        color: tuple = (0, 255, 0),
    ) -> np.ndarray:

        height, width = image.shape[:2]

        for i, hand_in_frame in enumerate(results.hand_landmarks):
            pts = [(int(lm.x * width), int(lm.y * height)) for lm in hand_in_frame]
            line_color = (
                (0, 255, 255) if i == selected_index else (120, 120, 120)
            )  # yellow vs. gray
            thickness = 3 if i == selected_index else 1
            for a, b in HAND_CONNECTIONS:
                cv2.line(image, pts[a], pts[b], line_color, thickness)

        # landmark dots only for the selected hand
        for point in landmarks:
            cv2.circle(image, point, radius, color, cv2.FILLED)

        return image

    # Counts raised fingers from landmark coordinates (logic unchanged).
    def _check_finger_positions(self, landmarks: list, finger_count: int) -> int:
        for coordinate in self.finger_coor:
            if landmarks[coordinate[0]][1] < landmarks[coordinate[1]][1]:
                finger_count += 1

        if landmarks[self.thum_coor[0]][0] > landmarks[self.thum_coor[1]][0]:
            finger_count += 1

        return finger_count

    # Extracts pixel coordinates of the selected hand's landmarks.
    def _extract_landmarks(
        self, image: np.ndarray, results: type, referanced_hand_index: int = 0
    ) -> list:
        height, width = image.shape[:2]
        landmarks = list()
        for landmark in results.hand_landmarks[referanced_hand_index]:
            landmarks.append((int(landmark.x * width), int(landmark.y * height)))

        return landmarks

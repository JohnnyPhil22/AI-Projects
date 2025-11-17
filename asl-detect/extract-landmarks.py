import os, cv2
import numpy as np
import mediapipe as mp

DATASET_DIR = "C:\\Users\\Jonathan Philips\\Coding\\AI-Projects\\asl-detect\\dataset"
OUT_X = "C:\\Users\\Jonathan Philips\\Coding\\AI-Projects\\asl-detect\\asl_X.npy"
OUT_Y = "C:\\Users\\Jonathan Philips\\Coding\\AI-Projects\\asl-detect\\asl_y.npy"
OUT_CLASSES = (
    "C:\\Users\\Jonathan Philips\\Coding\\AI-Projects\\asl-detect\\asl_classes.npy"
)

mp_hands = mp.solutions.hands


def get_class_folders(dataset_dir):
    classes = []
    for name in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, name)
        if os.path.isdir(path):
            classes.append(name)
    classes.sort()
    return classes


def extract_landmarks_from_image(img_path, hands):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    hand = result.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    if len(landmarks) != 63:
        print(f"Unexpected landmark size in {img_path}: {len(landmarks)}")
        return None

    return np.array(landmarks, dtype=np.float32)


def main():
    class_names = get_class_folders(DATASET_DIR)
    if not class_names:
        print("No class folders found in", DATASET_DIR)
        return

    print("Found classes:", class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    X, y = [], []

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:

        for class_name in class_names:
            class_dir = os.path.join(DATASET_DIR, class_name)
            label_idx = class_to_idx[class_name]
            print(f"\nProcessing class '{class_name}'...")

            for fname in os.listdir(class_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue

                img_path = os.path.join(class_dir, fname)
                landmarks = extract_landmarks_from_image(img_path, hands)

                if landmarks is None:
                    print(f"  [SKIP] No hand detected in {img_path}")
                    continue

                X.append(landmarks)
                y.append(label_idx)

    if not X:
        print("No landmarks extracted.")
        return

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    class_names = np.array(class_names)

    print("\nDone!")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Classes:", class_names)

    np.save(OUT_X, X)
    np.save(OUT_Y, y)
    np.save(OUT_CLASSES, class_names)
    print(f"Saved {OUT_X}, {OUT_Y}, {OUT_CLASSES}")


if __name__ == "__main__":
    main()

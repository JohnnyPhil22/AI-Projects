import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

try:
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_classifier.empty():
        face_classifier = None
except Exception as e:
    print(f"Error loading cascade classifier: {e}")
    face_classifier = None


def detect_nearest_face(frame):
    if face_classifier is None:
        return None

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    if len(faces) == 0:
        return None

    return max(faces, key=lambda rect: rect[2] * rect[3])


WIN = "Haircut Recommender"

# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
) as face_mesh:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            print("face_landmarks:", face_landmarks)
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
            )
        cv2.imwrite("/tmp/annotated_image" + str(idx) + ".png", annotated_image)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        nearest_face = detect_nearest_face(image)
        roi = image
        roi_origin_x = 0
        roi_origin_y = 0

        if nearest_face is not None:
            x, y, w, h = nearest_face
            x = max(0, x)
            y = max(0, y)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)
            if x2 > x and y2 > y:
                roi = image[y:y2, x:x2]
                roi_origin_x = x
                roi_origin_y = y

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        roi.flags.writeable = False
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(roi_rgb)

        # Draw the face mesh annotations on the image.
        roi.flags.writeable = True
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    landmark.x = (
                        landmark.x * roi.shape[1] + roi_origin_x
                    ) / image.shape[1]
                    landmark.y = (
                        landmark.y * roi.shape[0] + roi_origin_y
                    ) / image.shape[0]

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

        if nearest_face is not None:
            x, y, w, h = nearest_face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow(WIN, cv2.flip(image, 1))
        if (cv2.waitKey(1) & 0xFF == ord("q")) or (
            cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
        ):
            break
cap.release()
cv2.destroyAllWindows()

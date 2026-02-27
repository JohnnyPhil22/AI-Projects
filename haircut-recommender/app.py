import cv2

try:
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
except Exception as e:
    print(f"Error loading cascade classifier: {e}")
    face_classifier = None


video_capture = cv2.VideoCapture(0)


def detect_nearest_face(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    if len(faces) == 0:
        return None

    return max(faces, key=lambda rect: rect[2] * rect[3])


WIN = "Haircut Recommender"

while True:
    result, video_frame = video_capture.read()
    if not result:
        break

    faces = detect_nearest_face(video_frame)
    cv2.imshow(WIN, video_frame)

    # check for keypress or if window was closed
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (
        cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
    ):
        break

video_capture.release()
cv2.destroyAllWindows()

import cv2
from utils import Finger_counter


def demostration():
    # Creating instances
    finger_counter = Finger_counter()
    cap = cv2.VideoCapture(0)

    # looping through the video frame by frame
    name_index = 0
    while cap.isOpened():

        # reading the captured images
        ret, frame = cap.read()  # reading Frame

        # checking if the frame is empty
        if not ret:
            break

        # Find out the finger count which is upen the hand
        drawn_image, finger_count = finger_counter.count_fingers(
            frame,
            text_center_coor=(20, 50),  # (x, y) of the text's bottom-left corner
            font=cv2.FONT_HERSHEY_DUPLEX,
            font_scale=1.5,
            color=(0, 0, 255),  # BGR, so this is red
        )

        # displaying the results
        cv2.imshow("Finger Counter", drawn_image)  # showing Video

        # taking the keyboard key
        k = cv2.waitKey(1)

        # exit condition
        if (cv2.waitKey(1) & 0xFF == ord("q")) or (
            cv2.getWindowProperty("Finger Counter", cv2.WND_PROP_VISIBLE) < 1
        ):
            break

        # saving the images
        if k == ord("s"):
            cv2.imwrite(f"hand_counted_{name_index}.jpg", drawn_image)
            name_index += 1

    # re-allocating the sources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demostration()

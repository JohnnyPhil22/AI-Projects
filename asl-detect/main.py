import cv2, time
import mediapipe as mp
from asl_detector import ASLDetector

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Initialize ASL Detector
asl_detector = ASLDetector()

# Variables for FPS calculation
pTime = 0
cTime = 0

# Variables for stable detection
last_letter = None
letter_count = 0
confidence_threshold = 10  # Increased for better stability
confirmed_letter = None  # Track last confirmed letter for display

while True:
    # Read frame from webcam
    success, img = cap.read()

    # Process the image for hand tracking (flip and convert color)
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    detected_letter = None

    # Draw hand landmarks and detect ASL
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # Iterate through each hand
            # Detect ASL letter
            detected_letter = asl_detector.detect_asl_letter(handLms.landmark)

            # Stable detection logic
            if detected_letter:
                if detected_letter == last_letter:
                    letter_count += 1
                else:
                    last_letter = detected_letter
                    letter_count = 1

                # Display current detection
                if letter_count >= confidence_threshold:
                    confirmed_letter = detected_letter
                    print(f"✓ ASL DETECTED: {detected_letter}")
                    cv2.putText(
                        img,
                        f"ASL: {detected_letter}",
                        (10, 120),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 255, 0),
                        2,
                    )
                else:
                    # Show detecting progress with bar
                    progress = letter_count / confidence_threshold
                    cv2.putText(
                        img,
                        f"Detecting: {detected_letter} ({letter_count}/{confidence_threshold})",
                        (10, 120),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 255, 255),
                        2,
                    )
                    # Draw progress bar
                    bar_width = int(200 * progress)
                    cv2.rectangle(
                        img, (10, 140), (10 + bar_width, 155), (0, 255, 255), -1
                    )
                    cv2.rectangle(img, (10, 140), (210, 155), (255, 255, 255), 2)

                    # Show last confirmed letter at top for continuity
                    if confirmed_letter:
                        cv2.putText(
                            img,
                            f"Last: {confirmed_letter}",
                            (10, 180),
                            cv2.FONT_HERSHEY_PLAIN,
                            2,
                            (200, 200, 200),
                            2,
                        )
            else:
                last_letter = None
                letter_count = 0
                # Still show last confirmed letter even when no detection
                if confirmed_letter:
                    cv2.putText(
                        img,
                        f"Last: {confirmed_letter}",
                        (10, 120),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (200, 200, 200),
                        2,
                    )

            # Draw fingertip circles
            for id, lm in enumerate(handLms.landmark):  # Iterate through each landmark
                h, w, c = img.shape  # Get dimensions of the image
                cx, cy = int(lm.x * w), int(
                    lm.y * h
                )  # Convert normalized coordinates to pixel values

                if id == 4:
                    cv2.circle(
                        img, (cx, cy), 10, (255, 0, 255), cv2.FILLED
                    )  # Thumb tip
                if id == 8:
                    cv2.circle(
                        img, (cx, cy), 10, (255, 0, 255), cv2.FILLED
                    )  # Index finger tip
                if id == 12:
                    cv2.circle(
                        img, (cx, cy), 10, (255, 0, 255), cv2.FILLED
                    )  # Middle finger tip
                if id == 16:
                    cv2.circle(
                        img, (cx, cy), 10, (255, 0, 255), cv2.FILLED
                    )  # Ring finger tip
                if id == 20:
                    cv2.circle(
                        img, (cx, cy), 10, (255, 0, 255), cv2.FILLED
                    )  # Pinky finger tip

            mpDraw.draw_landmarks(
                img, handLms, mpHands.HAND_CONNECTIONS
            )  # Draw connections

    else:
        # No hand detected - show last confirmed letter
        if confirmed_letter:
            cv2.putText(
                img,
                f"Last: {confirmed_letter}",
                (10, 120),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (200, 200, 200),
                2,
            )

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(
        img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
    )

    # Show the image
    cv2.imshow("Image", img)

    # Exit conditions
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (
        cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1
    ):
        break

cap.release()
cv2.destroyAllWindows()

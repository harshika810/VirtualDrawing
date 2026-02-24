import cv2
import mediapipe as mp
import numpy as np

# Camera Start
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

canvas = None
prev_x, prev_y = None, None 

# MediaPipe Setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

thick = int(input("Enter thikness"))

for i in range (3):
    if i == 0 :
        Blue = int(input("Enter Blue of rgb"))
        
    if i == 1 :
        green =int(input("Enter green of rgb"))
    if i == 2 :
        Red = int(input("Enter red of rgb"))
        

while True:
    success, img = cap.read()
    if not success:
        break

    # Mirror image correctly
    img = cv2.flip(img, 1)

    h, w, c = img.shape
    
    if canvas is None:
        canvas = np.zeros_like(img)

    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # Draw hand skeleton
            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

            # Get index finger tip (landmark id 8)
            lm = handLms.landmark[8]
            x, y = int(lm.x * w), int(lm.y * h)

            # Draw fingertip circle
            cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)

            # Draw line on canvas
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (Blue, green, Red),  thickness=thick)

            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    # Merge canvas with camera frame
    combined = cv2.addWeighted(img, 0.7, canvas, 0.7, 0)

    cv2.imshow("Hand Tracking Drawing", combined)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
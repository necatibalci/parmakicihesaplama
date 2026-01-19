import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """ Üç nokta (a-b-c) arasındaki açıyı hesapla. """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                landmarks = hand_landmarks.landmark

                h, w, _ = image.shape
                p4 = [landmarks[4].x * w, landmarks[4].y * h]
                p8 = [landmarks[8].x * w, landmarks[8].y * h]
                p12 = [landmarks[12].x * w, landmarks[12].y * h]

                angle = calculate_angle(p4, p8, p12)

                cv2.putText(image, f"Aci: {int(angle)} deg", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Parmak Aci Hesaplama', image)

        if cv2.waitKey(10) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()

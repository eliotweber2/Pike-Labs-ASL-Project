import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

image_path = './testing_images/test_img_1.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
        
        landmark_list = []
        for landmark_id, landmark in enumerate(hand_landmarks.landmark):
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            landmark_list.append([landmark_id, cx, cy])
            
        print(f"Thumb tip position: {landmark_list[4]}")
        print(f"Index finger tip position: {landmark_list[8]}")
        print(f"Middle finger tip position: {landmark_list[12]}")

output_path = './testing_images/test_img_1_with_landmarks.png'
cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

print(f"Processed image saved to {output_path}")

hands.close()

cv2.imshow("Hand Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

from cv2 import imread, cvtColor, COLOR_BGR2RGB
import mediapipe as mp
import numpy as np
from PIL import Image

MIN_HANDS = 1

class Landmark_Creator:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_image(self, image, process_landmarks=True):
        image_rgb = cvtColor(image, COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < MIN_HANDS:
            return {'result': 'DETECTION_FAILED'}
        if not process_landmarks:
            return {'result': 'DETECTION_SUCCESS', 'landmarks': results.multi_hand_landmarks}
        landmark_lst = []
        for (i, hand) in enumerate(results.multi_hand_landmarks):
            if i > 1:
                break
            hand_landmarks = []
            for landmark_id, landmark in enumerate(hand.landmark):
                hand_landmarks.append([landmark_id, landmark.x, landmark.y, landmark.z])

            landmark_lst.append(hand_landmarks)
       
        return {'result': 'DETECTION_SUCCESS', 'landmarks': landmark_lst}
        
        
if __name__ == "__main__":
    landmarks = Landmark_Creator()
    image_path = './testing_images/testing_img_1.png'
    image = imread(image_path)
    landmarks = landmarks.process_image(image)
    print(landmarks['landmarks'])
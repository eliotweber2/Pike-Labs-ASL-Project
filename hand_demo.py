import cv2
import mediapipe as mp
import obj_detect
from video_loader import read_and_process, camera_stream_factory
#from video_loader import read_and_process
#from camera import camera_stream_factory

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

detector = obj_detect.Landmark_Creator()

def process_img(frame):
    landmarks = detector.process_image(frame,process_landmarks=False)
    if landmarks['result'] == 'DETECTION_SUCCESS':
        for hand_landmarks in landmarks['landmarks']:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    cv2.imshow('frame', frame)

read_and_process(camera_stream_factory, process_img)
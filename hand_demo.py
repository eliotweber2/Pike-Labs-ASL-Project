from cv2 import imshow, VideoWriter
from mediapipe import solutions
import obj_detect
#from video_loader import read_and_process, camera_stream_factory
from camera import camera_stream_factory
from video_loader import read_and_process
from dataset_video_loader import file_stream_factory

mp_hands = solutions.hands
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles

detector = obj_detect.Landmark_Creator()

out = VideoWriter('output.avi', 0, 20.0, (640,480))

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
    
    out.write(frame)
    imshow('frame', frame)

file_stream = lambda: file_stream_factory('./ASL_Citizen/videos/43222431209053225-DAMN.mp4')

if __name__ == '__main__':
    read_and_process(file_stream, process_img)
    out.release()
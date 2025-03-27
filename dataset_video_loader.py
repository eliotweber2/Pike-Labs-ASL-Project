from kagglehub import dataset_download
from os import path, rename, environ
import pandas as pd
from video_loader import StreamInterface, read_and_process
from obj_detect import Landmark_Creator
from cv2 import VideoCapture

path_from_download = 'datasets/abd0kamel/asl-citizen/versions/1/'

detector = Landmark_Creator()

environ['KAGGLEHUB_CACHE'] = '/Users/eliotweber/Downloads/'

dataset_download('abd0kamel/asl-citizen', path='ASL_Citizen/videos/521104571152148-ABSOLUTELY NOTHING.mp4')
#dataset_download('abd0kamel/asl-citizen', path='ASL_Citizen/videos/000017451997373907346-LIBRARY.mp4')


def download_and_move(file_name, dataset_path, destination_path):
    if path.exists(destination_path + '/' + file_name):
        print("File already exists")
        return
    dataset_download('abd0kamel/asl-citizen', path=dataset_path)
    if not path.exists('/Users/eliotweber/Downloads/' + path_from_download + dataset_path):
        print("File not found")
        return
    rename('/Users/eliotweber/Downloads/' + path_from_download + dataset_path, destination_path + '/' + file_name)

def file_stream_factory(file_path):

    return StreamInterface(lambda: VideoCapture(file_path),
            lambda cap: cap.read(),
            lambda x: print(x), 
            lambda cap: cap.release
            )

def process_video(video_details):
    landmark_lst = []
    def process_frame(frame):
        landmarks = detector.process_image(frame)
        if landmarks['result'] == 'DETECTION_SUCCESS':
            landmarks['video_id'] = video_details['Video file']
            landmarks['label'] = video_details['Gloss']
            landmark_lst.append(landmarks)
    video_path = 'ASL_Citizen/videos/' + video_details['Video file']
    new_path = './' + video_details['Video file']
    try:
        download_and_move(video_details['Video file'], video_path, '.')
    except:
        print("File not found", video_details['Video file'])
        return
    read_and_process(lambda: file_stream_factory(new_path), lambda frame: process_frame(frame))
    return landmark_lst

def create_landmark_file(destination_path, csv_file, landmark_filename):
    if path.exists(destination_path + '/' + landmark_filename):
        print("File already exists")
        return

    landmarks = pd.DataFrame(columns=['video_id', 'label', 'landmarks'])
    for i in range(len(csv_file)):
        video_details = csv_file.iloc[[i]]
        print(video_details)
        video_landmarks = process_video(video_details.iloc[0])
        landmarks.loc[len(landmarks)] = video_landmarks
    video_details = csv_file.iloc[[0]]
    return landmarks

download_and_move('test.csv', 'ASL_Citizen/splits/test.csv', './dataset_splits')
download_and_move('train.csv', 'ASL_Citizen/splits/train.csv', './dataset_splits')
download_and_move('val.csv', 'ASL_Citizen/splits/val.csv', './dataset_splits')

test_df = pd.read_csv('./dataset_splits/test.csv')
train_df = pd.read_csv('./dataset_splits/train.csv')
val_df = pd.read_csv('./dataset_splits/val.csv')

landmarks = create_landmark_file('./landmarks', test_df, 'test_landmarks.csv')
print(landmarks.head())
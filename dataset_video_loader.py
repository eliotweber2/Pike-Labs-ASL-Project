import pandas as pd
from video_loader import StreamInterface, read_and_process
from obj_detect import Landmark_Creator
from cv2 import VideoCapture, imshow
from os import path

detector = Landmark_Creator()

def file_stream_factory(file_path):

    return StreamInterface(lambda: VideoCapture(file_path),
            lambda cap: cap.read(),
            lambda x: print(x), 
            lambda cap: cap.release
            )

def process_video(video_id):
    landmark_lst = []
    def process_frame(frame):
        landmarks = detector.process_image(frame)
        if landmarks['result'] == 'DETECTION_SUCCESS':
            if len(landmarks['landmarks']) > 2:
                return
            landmark_lst.append(landmarks['landmarks'])
    video_path = './ASL_Citizen/videos/' + video_id

    read_and_process(lambda: file_stream_factory(video_path), lambda frame: process_frame(frame),n_skip=9)  
    return format_landmarks(landmark_lst)

def format_landmarks(frame_lst):
    formatted_landmarks = '||||'.join(
        ['|||'.join(
            ['||'.join([
                '|'.join([str(coord) for coord in landmark[1:4]])
            for landmark in landmark_lst ])
        for landmark_lst in frame])
    for frame in frame_lst])  
    return formatted_landmarks
                
def create_landmark_file(destination_path, csv_file, landmark_filename):
    if path.exists(destination_path + '/' + landmark_filename):
        print("File already exists")
        #return

    landmarks = pd.DataFrame(columns=['video_id', 'label', 'landmarks'])
    for i in range(len(csv_file)):
        video_details = row_to_dict(csv_file.iloc[[i]])
        print(i,video_details)
        video_landmarks = process_video(video_details['Video file'])
        landmarks.loc[len(landmarks)] = [video_details['Video file'], video_details['Gloss'], video_landmarks]
        
    landmarks.to_csv(destination_path + '/' + landmark_filename)
    print('File created')
    #return landmarks

def row_to_dict(row):
    result = {}
    for column in row.columns:
        result[column] = row[column].values[0]
    return result

test_csv = pd.read_csv('./ASL_Citizen/splits/test.csv')
train_csv = pd.read_csv('./ASL_Citizen/splits/train.csv')
val_csv = pd.read_csv('./ASL_Citizen/splits/val.csv')

create_landmark_file('./ASL_Citizen/landmarks', val_csv, 'val_landmarks.csv')
create_landmark_file('./ASL_Citizen/landmarks', test_csv, 'test_landmarks.csv')
create_landmark_file('./ASL_Citizen/landmarks', train_csv, 'train_landmarks.csv')

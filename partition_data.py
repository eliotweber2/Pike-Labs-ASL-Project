import os
import pandas as pd

def partition_data(input_file,name):
    df = pd.read_csv(input_file)
    part = df.iloc[1:500,:]
    part.to_csv('./ASL_Citizen/landmarks/'+name)


partition_data('./ASL_Citizen/landmarks/test_landmarks.csv','test_small.csv')
partition_data('./ASL_Citizen/landmarks/train_landmarks.csv','train_small.csv')
partition_data('./ASL_Citizen/landmarks/val_landmarks.csv','val_small.csv')
import sign_model
import data_processing

import numpy as np
import pandas as pd

def filter_words(df, limit=5):
    df = df.dropna()
    unique_labels = df['label'].unique()
    for label in unique_labels:
        if len(df[df['label'] == label]) < limit:
            df = df[df['label'] != label]

    return df

def take_top_words(df,limit=500):
    #common_words_df = pd.read_csv('unigram_freq.csv')
    #common_words_df = common_words_df.sort_values(by='count', ascending=False)
    #common_words = common_words_df['word'].tolist()[:limit]
    common_words = []
    with open('words2.txt', 'r') as f:
        common_words = [word.strip() for word in f.readlines() if word.strip() != '']
        common_words = common_words[:limit]
    i = 0
    while i < len(df):
        label = df['label'].tolist()[i]
        label = ''.join([char for char in label.split() if char.isalpha()]).lower()
        if label not in common_words:
            df = df[df['label'] != df['label'].tolist()[i]]
        else:
            i += 1

    return df

def main():
    test_csv = pd.read_csv('./ASL_Citizen/landmarks/test_landmarks.csv')
    train_csv = pd.read_csv('./ASL_Citizen/landmarks/train_landmarks.csv')
    val_csv = pd.read_csv('./ASL_Citizen/landmarks/val_landmarks.csv')

    print(len(test_csv), len(train_csv), len(val_csv))

    test_csv = filter_words(test_csv, 5)
    train_csv = filter_words(train_csv, 5)
    val_csv = filter_words(val_csv, 3)

    for label in test_csv['label'].unique():
        if label not in val_csv['label'].unique() or label not in train_csv['label'].unique():
            test_csv = test_csv[test_csv['label'] != label]

    for label in train_csv['label'].unique():
        if label not in val_csv['label'].unique() or label not in test_csv['label'].unique():
            train_csv = train_csv[train_csv['label'] != label]

    for label in val_csv['label'].unique():
        if label not in train_csv['label'].unique() or label not in test_csv['label'].unique():
            val_csv = val_csv[val_csv['label'] != label]

    word_lst = list(set(test_csv['label'].tolist() + train_csv['label'].tolist() + val_csv['label'].tolist()))
    with open('words.txt', 'w') as f:
        for word in word_lst:
            f.write(word + '\n')
    print("saved words to words.txt")

    test_csv = take_top_words(test_csv, 100)
    train_csv = take_top_words(train_csv, 100)
    val_csv = take_top_words(val_csv, 100)

    print(len(test_csv), len(train_csv), len(val_csv))


    model = sign_model.train_high_accuracy_model(train_csv, val_csv, 9)

main()
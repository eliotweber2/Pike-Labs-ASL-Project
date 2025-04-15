import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import time
import argparse
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

# Import your existing modules
from video_loader import read_and_process, StreamInterface
from obj_detect import Landmark_Creator
from camera import camera_stream_factory
from dataset_video_loader import create_landmark_file

# Data preprocessing functions
def normalize_landmarks(landmarks):
    """Normalize landmarks to be relative to wrist position and scale."""
    if not landmarks or landmarks['result'] != 'DETECTION_SUCCESS':
        return None
    
    normalized_landmarks = []
    for hand in landmarks['landmarks']:
        # Get wrist as reference point (point 0)
        wrist = hand[0][1:4]  # x, y, z coordinates
        
        # Normalize all points relative to wrist
        normalized_hand = []
        for point in hand:
            point_id = point[0]
            x, y, z = point[1] - wrist[0], point[2] - wrist[1], point[3] - wrist[2]
            normalized_hand.append([point_id, x, y, z])
            
        normalized_landmarks.append(normalized_hand)
    
    return {'result': 'DETECTION_SUCCESS', 
            'landmarks': normalized_landmarks, 
            'video_id': landmarks.get('video_id'),
            'label': landmarks.get('label')}

def prepare_sequences(landmarks_df, sequence_length=30):
    """Prepare sequences for LSTM/GRU model from landmarks dataframe."""
    sequences = []
    labels = []
    
    # Group by video_id to get sequences
    grouped = landmarks_df.groupby(['video_id', 'label'])
    
    for (vid, label), group in grouped:
        landmarks_seq = group['landmarks'].tolist()
        
        # Ensure consistent sequence length
        if len(landmarks_seq) < sequence_length:
            # Skip sequences that are too short
            continue
        
        # Use a sliding window approach to create multiple sequences from one video
        for i in range(0, len(landmarks_seq) - sequence_length + 1, sequence_length // 2):  # 50% overlap
            window = landmarks_seq[i:i + sequence_length]
            
            # Flatten the landmarks for each frame into a feature vector
            flattened_seq = []
            for frame in window:
                # Make sure frame has landmarks
                if not isinstance(frame, dict) or 'landmarks' not in frame:
                    continue
                    
                features = []
                for hand in frame['landmarks']:
                    for point in hand:
                        features.extend(point[1:])  # Add x, y, z coordinates
                        
                flattened_seq.append(features)
            
            # Make sure sequence is complete
            if len(flattened_seq) == sequence_length:
                sequences.append(flattened_seq)
                labels.append(label)
    
    return np.array(sequences), np.array(labels)

# Model creation function
def create_sign_language_model(n_classes, sequence_length, n_features):
    """Create an LSTM-based model for sign language recognition."""
    model = Sequential([
        # Input shape: (sequence_length, n_features)
        Bidirectional(LSTM(128, return_sequences=True), 
                     input_shape=(sequence_length, n_features)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

# Training function
def train_sign_language_model(train_df, val_df, sequence_length=30):
    """Train the sign language recognition model."""
    print("Preparing training data...")
    X_train, y_train_raw = prepare_sequences(train_df, sequence_length)
    print("Training data shape:", X_train.shape)
    
    print("Preparing validation data...")
    X_val, y_val_raw = prepare_sequences(val_df, sequence_length)
    print("Validation data shape:", X_val.shape)
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train_raw)
    y_train = label_encoder.transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Get feature dimension
    n_features = X_train.shape[2]
    n_classes = len(label_encoder.classes_)
    
    print(f"Input shape: (None, {sequence_length}, {n_features})")
    
    # Create and compile model
    model = create_sign_language_model(
        n_classes=n_classes,
        sequence_length=sequence_length,
        n_features=n_features
    )
    
    # Create directories if they don't exist
    os.makedirs("model", exist_ok=True)
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='model/sign_language_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
    )
    
    # Save model and label encoder
    model.save('model/sign_language_model.h5')
    print("Model saved to model/sign_language_model.h5")
    
    # Save label mapping
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved to model/label_encoder.pkl")
    
    return model, label_encoder, history

# Prediction functions
def load_model_and_encoder():
    """Load the trained model and label encoder."""
    print("Loading model and label encoder...")
    model = tf.keras.models.load_model('model/sign_language_model.h5')
    
    with open('model/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Model loaded with {len(label_encoder.classes_)} classes")
    return model, label_encoder

def predict_from_landmarks(landmarks, model, label_encoder, sequence_buffer, sequence_length=30):
    """Make predictions from real-time landmarks."""
    # Normalize landmarks
    normalized = normalize_landmarks(landmarks)
    if not normalized:
        return None, sequence_buffer
    
    # Flatten landmarks
    features = []
    for hand in normalized['landmarks']:
        for point in hand:
            features.extend(point[1:])  # Add x, y, z coordinates
    
    # Update sequence buffer (keep last N frames)
    sequence_buffer.append(features)
    if len(sequence_buffer) > sequence_length:
        sequence_buffer = sequence_buffer[-sequence_length:]
    
    # If we don't have enough frames yet, return None
    if len(sequence_buffer) < sequence_length:
        return None, sequence_buffer
    
    # Prepare sequence for prediction
    X = np.array([sequence_buffer])
    
    # Make prediction
    prediction = model.predict(X, verbose=0)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    
    # Convert to label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    return (predicted_label, confidence), sequence_buffer

# Real-time prediction
def real_time_prediction():
    """Run real-time prediction from webcam in terminal."""
    # Check if model exists
    if not os.path.exists('model/sign_language_model.h5') or not os.path.exists('model/label_encoder.pkl'):
        print("Error: Model files not found. Train the model first.")
        return
        
    # Load model and encoder
    model, label_encoder = load_model_and_encoder()
    
    # Create landmark detector
    detector = Landmark_Creator()
    
    # Initialize sequence buffer
    sequence_buffer = []
    last_prediction = None
    prediction_cooldown = 0
    
    # Define frame processing function
    def process_frame(frame):
        nonlocal sequence_buffer, last_prediction, prediction_cooldown
        
        # Get landmarks
        landmarks = detector.process_image(frame)
        
        # Draw landmarks on frame (reuse your existing detector)
        mp_drawing = getattr(detector, 'mp_drawing', None)
        mp_hands = getattr(detector, 'mp_hands', None)
        
        if mp_drawing and mp_hands and landmarks['result'] == 'DETECTION_SUCCESS':
            # This draws the landmarks if your detector has these attributes
            for hand_landmarks in landmarks['landmarks']:
                # Convert our format back to MediaPipe format for drawing
                mp_landmarks = []
                for i, point in enumerate(hand_landmarks):
                    x, y, z = point[1], point[2], point[3]
                    mp_landmarks.append((x, y, z))
        
        # Display frame
        cv2.imshow('Sign Language Recognition', frame)
        
        # Make prediction if landmarks detected
        if landmarks['result'] == 'DETECTION_SUCCESS':
            # Make prediction
            result, sequence_buffer = predict_from_landmarks(
                landmarks, model, label_encoder, sequence_buffer
            )
            
            # Display result on terminal
            if result and prediction_cooldown == 0:
                label, confidence = result
                
                # Only update if confidence is high enough or new prediction
                if confidence > 0.6 or last_prediction != label:
                    last_prediction = label
                    prediction_cooldown = 5  # Set cooldown to avoid flickering predictions
                    
                    # Clear terminal line and print prediction
                    print("\033[K", end='\r')  # Clear line
                    print(f"Prediction: {label} (Confidence: {confidence:.2f})", end='\r')
            
            # Decrease cooldown
            if prediction_cooldown > 0:
                prediction_cooldown -= 1
    
    # Start video processing
    print("\nStarting real-time sign language recognition...")
    print("Press 'q' to quit")
    
    try:
        read_and_process(
            stream_src=camera_stream_factory,
            process_fn=process_frame,
            stop_key='q',
            n_skip=1  # Skip every other frame for performance
        )
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cv2.destroyAllWindows()
        print("\nSign language recognition stopped")

# Evaluation function
def evaluate_model(test_df):
    """Evaluate the model on test data."""
    # Check if model exists
    if not os.path.exists('model/sign_language_model.h5') or not os.path.exists('model/label_encoder.pkl'):
        print("Error: Model files not found. Train the model first.")
        return
    
    # Load model and encoder
    model, label_encoder = load_model_and_encoder()
    
    # Prepare sequences
    print("Preparing test sequences...")
    X_test, y_test_raw = prepare_sequences(test_df)
    
    # Check if we have test data
    if len(X_test) == 0:
        print("Error: No test sequences could be created. Check your test data.")
        return
    
    print(f"Test data shape: {X_test.shape}")
    
    # Encode labels
    y_test = label_encoder.transform(y_test_raw)
    
    # Generate predictions
    print("Generating predictions...")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=label_encoder.classes_))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=label_encoder.classes_,
               yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

# Process the dataset and create landmark files
def process_dataset(train_csv='dataset_splits/train.csv', 
                   val_csv='dataset_splits/val.csv',
                   test_csv='dataset_splits/test.csv'):
    """Process the dataset and extract landmarks."""
    # Create landmarks directory
    os.makedirs("landmarks", exist_ok=True)
    
    # Process each dataset split
    train_landmarks_file = 'landmarks/train_landmarks.csv'
    val_landmarks_file = 'landmarks/val_landmarks.csv'
    test_landmarks_file = 'landmarks/test_landmarks.csv'
    
    # Process training data
    if not os.path.exists(train_landmarks_file):
        print("Processing training data...")
        train_df = pd.read_csv(train_csv)
        train_landmarks = create_landmark_file('./landmarks', train_df, 'train_landmarks.csv')
        if train_landmarks is not None:
            train_landmarks.to_csv(train_landmarks_file, index=False)
            print(f"Saved training landmarks to {train_landmarks_file}")
    else:
        print(f"Loading existing training landmarks from {train_landmarks_file}")
        train_landmarks = pd.read_csv(train_landmarks_file)
    
    # Process validation data
    if not os.path.exists(val_landmarks_file):
        print("Processing validation data...")
        val_df = pd.read_csv(val_csv)
        val_landmarks = create_landmark_file('./landmarks', val_df, 'val_landmarks.csv')
        if val_landmarks is not None:
            val_landmarks.to_csv(val_landmarks_file, index=False)
            print(f"Saved validation landmarks to {val_landmarks_file}")
    else:
        print(f"Loading existing validation landmarks from {val_landmarks_file}")
        val_landmarks = pd.read_csv(val_landmarks_file)
    
    # Process test data
    if not os.path.exists(test_landmarks_file):
        print("Processing test data...")
        test_df = pd.read_csv(test_csv)
        test_landmarks = create_landmark_file('./landmarks', test_df, 'test_landmarks.csv')
        if test_landmarks is not None:
            test_landmarks.to_csv(test_landmarks_file, index=False)
            print(f"Saved test landmarks to {test_landmarks_file}")
    else:
        print(f"Loading existing test landmarks from {test_landmarks_file}")
        test_landmarks = pd.read_csv(test_landmarks_file)
    
    return train_landmarks, val_landmarks, test_landmarks

# Main function to tie everything together
def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description='Sign Language Recognition System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Run real-time prediction')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model on test data')
    parser.add_argument('--process', action='store_true', help='Process dataset and extract landmarks')
    parser.add_argument('--train_csv', type=str, default='dataset_splits/train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='dataset_splits/val.csv',
                        help='Path to validation CSV file')
    parser.add_argument('--test_csv', type=str, default='dataset_splits/test.csv',
                        help='Path to test CSV file')
    
    args = parser.parse_args()
    
    if not args.train and not args.predict and not args.evaluate and not args.process:
        print("No action specified. Use --train, --predict, --evaluate, or --process")
        parser.print_help()
        return
    
    if args.process:
        print("Processing dataset...")
        process_dataset(args.train_csv, args.val_csv, args.test_csv)
    
    if args.train:
        print("Starting training pipeline...")
        
        # Load processed landmarks
        train_landmarks = pd.read_csv('landmarks/train_landmarks.csv')
        val_landmarks = pd.read_csv('landmarks/val_landmarks.csv')
        
        # Train model
        train_sign_language_model(train_landmarks, val_landmarks)
        
        print("Training completed!")
    
    if args.evaluate:
        print("Evaluating model...")
        test_landmarks = pd.read_csv('landmarks/test_landmarks.csv')
        evaluate_model(test_landmarks)
    
    if args.predict:
        print("Starting real-time prediction...")
        real_time_prediction()

if __name__ == "__main__":
    main()
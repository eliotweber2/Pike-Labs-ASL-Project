# Alternative processing might work better with the model idk if you have time test it
import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
# Import your existing modules
from video_loader import read_and_process, StreamInterface
from obj_detect import Landmark_Creator
from camera import camera_stream_factory
# Fixed format_landmarks function to properly parse landmarks
def parse_landmarks(landmarks_str):
    """Parse landmarks from string format to structured format."""
    if not isinstance(landmarks_str, str):
        return []
    
    try:
        frames = landmarks_str.split('||||')
        parsed_frames = []
        
        for frame in frames:
            if not frame:
                continue
                
            hands = frame.split('|||')
            parsed_hands = []
            
            for hand_idx, hand in enumerate(hands):
                if not hand:
                    continue
                
                landmarks = hand.split('||')
                parsed_landmarks = []
                
                for landmark_idx, landmark in enumerate(landmarks):
                    coords = landmark.split('|')
                    if len(coords) != 3:
                        continue
                    
                    # Create landmark in format [id, x, y, z]
                    parsed_landmarks.append([landmark_idx, float(coords[0]), float(coords[1]), float(coords[2])])
                
                if parsed_landmarks:
                    parsed_hands.append(parsed_landmarks)
            
            if parsed_hands:
                parsed_frame = {
                    'result': 'DETECTION_SUCCESS',
                    'landmarks': parsed_hands
                }
                parsed_frames.append(parsed_frame)
        
        return parsed_frames
    except Exception as e:
        print(f"Error parsing landmarks: {e}")
        return []
# Enhanced landmark normalization for better accuracy
def normalize_landmarks_enhanced(landmarks):
    """Enhanced normalization with scale invariance and hand orientation correction."""
    if not landmarks or landmarks['result'] != 'DETECTION_SUCCESS':
        return None
    
    normalized_landmarks = []
    for hand in landmarks['landmarks']:
        # Get wrist as reference point (point 0)
        wrist = None
        for point in hand:
            if point[0] == 0:  # Wrist point
                wrist = point[1:4]
                break
        
        if wrist is None:
            continue
        
        # Get index finger MCP and pinky MCP for hand orientation and scale
        index_mcp = None
        pinky_mcp = None
        middle_mcp = None
        
        for point in hand:
            if point[0] == 5:  # Index finger MCP
                index_mcp = point[1:4]
            elif point[0] == 9:  # Middle finger MCP
                middle_mcp = point[1:4]
            elif point[0] == 17:  # Pinky MCP
                pinky_mcp = point[1:4]
        
        # If we don't have enough reference points, use simple normalization
        if index_mcp is None or pinky_mcp is None:
            normalized_hand = []
            for point in hand:
                point_id = point[0]
                x, y, z = point[1] - wrist[0], point[2] - wrist[1], point[3] - wrist[2]
                normalized_hand.append([point_id, x, y, z])
                
            normalized_landmarks.append(normalized_hand)
            continue
        
        # Calculate hand scale (distance between index and pinky MCP)
        scale = np.sqrt((index_mcp[0] - pinky_mcp[0])**2 + 
                        (index_mcp[1] - pinky_mcp[1])**2 + 
                        (index_mcp[2] - pinky_mcp[2])**2)
        
        if scale < 1e-6:  # Avoid division by zero
            scale = 1.0
            
        # Calculate palm orientation vectors
        if middle_mcp is not None:
            # Vector from pinky to index (width)
            width_vector = np.array([
                index_mcp[0] - pinky_mcp[0],
                index_mcp[1] - pinky_mcp[1],
                index_mcp[2] - pinky_mcp[2]
            ])
            
            # Vector from wrist to middle (height)
            height_vector = np.array([
                middle_mcp[0] - wrist[0],
                middle_mcp[1] - wrist[1],
                middle_mcp[2] - wrist[2]
            ])
            
            # Normalize vectors
            width_vector = width_vector / (np.linalg.norm(width_vector) + 1e-10)
            height_vector = height_vector / (np.linalg.norm(height_vector) + 1e-10)
            
            # Calculate orthogonal vector (normal to palm plane)
            normal_vector = np.cross(width_vector, height_vector)
            normal_vector = normal_vector / (np.linalg.norm(normal_vector) + 1e-10)
            
            # Create rotation matrix
            rotation_matrix = np.column_stack([width_vector, height_vector, normal_vector])
            
            # Normalize points with rotation invariance
            normalized_hand = []
            for point in hand:
                point_id = point[0]
                
                # Center around wrist
                centered = np.array([
                    point[1] - wrist[0],
                    point[2] - wrist[1],
                    point[3] - wrist[2]
                ])
                
                # Scale normalization
                centered = centered / scale
                
                # Without rotation for now (keep it simpler)
                normalized_hand.append([point_id, centered[0], centered[1], centered[2]])
            
            normalized_landmarks.append(normalized_hand)
        else:
            # Fallback to simple normalization with scale
            normalized_hand = []
            for point in hand:
                point_id = point[0]
                x = (point[1] - wrist[0]) / scale
                y = (point[2] - wrist[1]) / scale
                z = (point[3] - wrist[2]) / scale
                normalized_hand.append([point_id, x, y, z])
                
            normalized_landmarks.append(normalized_hand)
    
    return {'result': 'DETECTION_SUCCESS', 
            'landmarks': normalized_landmarks, 
            'video_id': landmarks.get('video_id'),
            'label': landmarks.get('label')}
# Calculate pairwise distances between landmarks for improved features
def calculate_pairwise_features(landmarks):
    """Calculate pairwise distances and angles between key landmarks."""
    additional_features = []
    
    # Define key points (indices may vary based on MediaPipe hand model)
    key_points = [0, 4, 8, 12, 16, 20]  # Wrist, thumb tip, index tip, middle tip, ring tip, pinky tip
    
    for hand in landmarks['landmarks']:
        # Get coordinates of key points
        key_coords = {}
        for point in hand:
            if point[0] in key_points:
                key_coords[point[0]] = point[1:4]
        
        # Skip if we don't have all key points
        if len(key_coords) != len(key_points):
            continue
        
        # Calculate distances between each pair of key points
        for i in range(len(key_points)):
            for j in range(i+1, len(key_points)):
                p1 = key_points[i]
                p2 = key_points[j]
                
                if p1 in key_coords and p2 in key_coords:
                    # Euclidean distance
                    dist = np.sqrt(
                        (key_coords[p1][0] - key_coords[p2][0])**2 +
                        (key_coords[p1][1] - key_coords[p2][1])**2 +
                        (key_coords[p1][2] - key_coords[p2][2])**2
                    )
                    additional_features.append(dist)
        
        # Calculate angles for fingers (simplified)
        for finger in [4, 8, 12, 16, 20]:  # Thumb tip, index tip, etc.
            if 0 in key_coords and finger in key_coords:
                # Vector from wrist to finger tip
                vec = [
                    key_coords[finger][0] - key_coords[0][0],
                    key_coords[finger][1] - key_coords[0][1],
                    key_coords[finger][2] - key_coords[0][2]
                ]
                
                # Calculate angles with each axis
                magnitude = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
                if magnitude > 0:
                    cos_x = vec[0] / magnitude
                    cos_y = vec[1] / magnitude
                    cos_z = vec[2] / magnitude
                    additional_features.extend([cos_x, cos_y, cos_z])
    
    return additional_features
# Data augmentation function
def augment_landmarks(landmarks_df, augmentation_factor=3):
    """Augment landmarks data with random variations to improve model robustness."""
    augmented_data = []
    
    print("Augmenting landmark data...")
    for idx, row in tqdm(landmarks_df.iterrows(), total=len(landmarks_df)):
        video_id = row['video_id']
        label = row['label']
        landmarks_list = parse_landmarks(row['landmarks'])
        
        # Add original data
        augmented_data.append({
            'video_id': video_id,
            'label': label,
            'landmarks': landmarks_list
        })
        
        # Create augmented versions
        for i in range(augmentation_factor):
            augmented_landmarks = []
            
            for landmarks in landmarks_list:
                if landmarks['result'] != 'DETECTION_SUCCESS':
                    continue
                
                # Copy landmarks
                aug_landmarks = {'result': 'DETECTION_SUCCESS', 
                                'landmarks': [], 
                                'video_id': landmarks.get('video_id'),
                                'label': landmarks.get('label')}
                
                for hand in landmarks['landmarks']:
                    aug_hand = []
                    
                    # Apply random noise to each point
                    noise_scale = 0.02 * (i + 1)  # Increase noise with each augmentation
                    for point in hand:
                        point_id = point[0]
                        # Add random noise to coordinates
                        x = point[1] + np.random.normal(0, noise_scale)
                        y = point[2] + np.random.normal(0, noise_scale)
                        z = point[3] + np.random.normal(0, noise_scale)
                        aug_hand.append([point_id, x, y, z])
                    
                    aug_landmarks['landmarks'].append(aug_hand)
                
                augmented_landmarks.append(aug_landmarks)
            
            # Only add if we have landmarks
            if augmented_landmarks:
                augmented_data.append({
                    'video_id': f"{video_id}_aug_{i}",
                    'label': label,
                    'landmarks': augmented_landmarks
                })
    
    return pd.DataFrame(augmented_data)
# Improved sequence preparation for ML model
def prepare_sequences(landmarks_df, sequence_length=30, include_pairwise=True):
    """Prepare sequences for LSTM model with enhanced features."""
    sequences = []
    labels = []
    
    print("Preparing sequences...")
    # Process each row
    for idx, row in tqdm(landmarks_df.iterrows(), total=len(landmarks_df)):
        video_id = row['video_id']
        label = row['label']
        
        # Parse landmarks from string format
        landmarks_seq = parse_landmarks(row['landmarks'])
        
        # Ensure consistent sequence length
        if len(landmarks_seq) < sequence_length:
            # Skip sequences that are too short
            continue
        
        # Use a sliding window approach to create multiple sequences from one video
        stride = max(1, min(5, (len(landmarks_seq) - sequence_length) // 3))
        
        for i in range(0, len(landmarks_seq) - sequence_length + 1, stride):
            window = landmarks_seq[i:i + sequence_length]
            
            # Flatten the landmarks for each frame into a feature vector
            flattened_seq = []
            for frame_idx, frame in enumerate(window):
                # Normalize landmarks
                normalized = normalize_landmarks_enhanced(frame)
                if not normalized:
                    continue
                
                # Basic features - coordinates of all points
                features = []
                for hand in normalized['landmarks']:
                    for point in hand:
                        features.extend(point[1:4])  # Add x, y, z coordinates
                
                # Additional features - pairwise distances and angles
                if include_pairwise:
                    pairwise_features = calculate_pairwise_features(normalized)
                    features.extend(pairwise_features)
                
                # Ensure fixed feature length by padding if necessary
                expected_features = 63  # Adjust based on your landmark model (21 points * 3 coords)
                if include_pairwise:
                    expected_features += 30  # Adjust based on your pairwise features
                
                # Pad or truncate features
                if len(features) < expected_features:
                    features.extend([0] * (expected_features - len(features)))
                elif len(features) > expected_features:
                    features = features[:expected_features]
                
                flattened_seq.append(features)
            
            # Make sure sequence is complete
            if len(flattened_seq) == sequence_length:
                sequences.append(flattened_seq)
                labels.append(label)
    
    return np.array(sequences), np.array(labels)

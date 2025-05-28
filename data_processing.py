import numpy as np
import pandas as pd
import os
from tqdm import tqdm  

def parse_landmarks(landmarks_str):
    """
    Parses a MediaPipe-style landmark string into a list of frames.
    Each frame contains hands, and each hand contains [id, x, y, z] coordinates.
    """
    if not isinstance(landmarks_str, str): # Basic type check
        return []

    try:
        frames = landmarks_str.split('||||') # Frame delimiter
        parsed_frames = []

        for frame_str in frames:
            if not frame_str: continue # Skip empty frame data

            hands = frame_str.split('|||') # Hand delimiter
            parsed_hands_in_frame = []

            for hand_str in hands:
                if not hand_str: continue # Skip empty hand data

                landmarks = hand_str.split('||') # Landmark delimiter
                parsed_landmarks_in_hand = []

                for landmark_idx, landmark_coords_str in enumerate(landmarks):
                    coords = landmark_coords_str.split('|') # Coordinate delimiter
                    if len(coords) != 3: continue # Expect x, y, z

                    try:
                        # Store landmark id and its float coordinates
                        parsed_landmarks_in_hand.append([
                            landmark_idx,
                            float(coords[0]), float(coords[1]), float(coords[2])
                        ])
                    except ValueError:
                        continue # Skip if coordinates are not valid floats

                if parsed_landmarks_in_hand:
                    parsed_hands_in_frame.append(parsed_landmarks_in_hand)

            if parsed_hands_in_frame:
                # Structure for a successfully parsed frame
                parsed_frame_dict = {
                    'result': 'DETECTION_SUCCESS',
                    'landmarks': parsed_hands_in_frame
                }
                parsed_frames.append(parsed_frame_dict)
        return parsed_frames
    except Exception as e:
        print(f"Error parsing landmarks string: {e}, Input preview: '{landmarks_str[:100]}...'")
        return []


def normalize_landmarks_enhanced(frame_data_dict):
    """
    Normalizes landmark coordinates for each hand in a single frame.
    Centering on wrist (landmark 0) and scaling by MCP joint distances.
    """
    if not frame_data_dict or frame_data_dict.get('result') != 'DETECTION_SUCCESS' or not frame_data_dict.get('landmarks'):
        return None # Invalid input or no landmarks

    normalized_frame_hands = []

    for hand_landmarks_list in frame_data_dict['landmarks']:
        wrist = None
        for point in hand_landmarks_list:
            if point[0] == 0:  # Landmark ID 0 is the wrist
                wrist = point[1:4]  # [x, y, z]
                break
        if wrist is None: continue # Skip hand if wrist is not found

        index_mcp, middle_mcp, pinky_mcp = None, None, None
        for point in hand_landmarks_list:
            if point[0] == 5: index_mcp = point[1:4]
            elif point[0] == 9: middle_mcp = point[1:4] # Though not used in current scale, good to identify
            elif point[0] == 17: pinky_mcp = point[1:4]

        current_hand_normalized_landmarks = []
        scale = 1.0 # Default scale

        if index_mcp and pinky_mcp: # Check if key points for scaling are present
            scale_dist = np.linalg.norm(np.array(index_mcp) - np.array(pinky_mcp))
            if scale_dist > 1e-6: # Avoid division by zero or very small scale
                scale = scale_dist
            # If key MCPs are missing, scale remains 1.0 (only centering)

        for point in hand_landmarks_list:
            point_id = point[0]
            # Center coordinates around the wrist and apply scale
            norm_x = (point[1] - wrist[0]) / scale
            norm_y = (point[2] - wrist[1]) / scale
            norm_z = (point[3] - wrist[2]) / scale
            current_hand_normalized_landmarks.append([point_id, norm_x, norm_y, norm_z])

        if current_hand_normalized_landmarks:
            normalized_frame_hands.append(current_hand_normalized_landmarks)

    if not normalized_frame_hands: return None

    # Return normalized landmarks, preserving original video_id and label if present
    return {
        'result': 'DETECTION_SUCCESS',
        'landmarks': normalized_frame_hands,
        'video_id': frame_data_dict.get('video_id'),
        'label': frame_data_dict.get('label')
    }


def calculate_single_hand_pairwise_features(hand_normalized_landmarks, pad_value=0.0):
    """
    Computes pairwise distances and cosine angles for key landmarks of a single hand.
    Returns a fixed-length feature list (30 features: 15 dist + 15 cosines).
    """
    hand_features = []
    key_points_indices = [0, 4, 8, 12, 16, 20] # Wrist and fingertips
    num_key_points = len(key_points_indices)

    # Expected feature counts
    expected_distances_count = num_key_points * (num_key_points - 1) // 2 # C(6, 2) = 15
    expected_angles_count = 5 * 3 # 5 fingertips, 3 (x,y,z) components of cosine vector
    total_expected_pairwise_features = expected_distances_count + expected_angles_count

    # Create a dictionary for quick lookup of key landmark coordinates
    key_coords = {point[0]: np.array(point[1:4]) for point in hand_normalized_landmarks if point[0] in key_points_indices}

    # 1. Pairwise distances between key points
    for i in range(num_key_points):
        for j in range(i + 1, num_key_points):
            p1_idx, p2_idx = key_points_indices[i], key_points_indices[j]
            if p1_idx in key_coords and p2_idx in key_coords:
                dist = np.linalg.norm(key_coords[p1_idx] - key_coords[p2_idx])
                hand_features.append(dist)
            else:
                hand_features.append(pad_value) # Pad if a key point is missing

    # Ensure correct number of distance features (padding if somehow short)
    while len(hand_features) < expected_distances_count:
        hand_features.append(pad_value)

    # 2. Cosine of vectors from wrist to fingertips (normalized vector components)
    if 0 in key_coords: # Wrist must be present
        wrist_coord = key_coords[0]
        finger_tip_indices = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky tips
        for tip_idx in finger_tip_indices:
            if tip_idx in key_coords:
                finger_vec = key_coords[tip_idx] - wrist_coord
                magnitude = np.linalg.norm(finger_vec)
                if magnitude > 1e-9: # Avoid division by zero
                    cos_angles = (finger_vec / magnitude).tolist()
                    hand_features.extend(cos_angles)
                else: # Fingertip at wrist
                    hand_features.extend([pad_value] * 3)
            else: # Fingertip missing
                hand_features.extend([pad_value] * 3)
    else: # Wrist missing, pad all angle features
        hand_features.extend([pad_value] * expected_angles_count)

    # Ensure final feature list has the total expected length
    while len(hand_features) < total_expected_pairwise_features:
        hand_features.append(pad_value)

    return hand_features[:total_expected_pairwise_features] # Return exactly fixed number of features


def augment_landmarks(landmarks_df, augmentation_factor=3):
    """
    Augments landmark data by adding random noise.
    Input DataFrame has 'video_id', 'label', 'landmarks' (string or list of frames).
    Returns a new DataFrame with original and augmented data.
    """
    augmented_data_list = []
    print("Augmenting landmark data...")

    for _, row in tqdm(landmarks_df.iterrows(), total=len(landmarks_df)):
        video_id, label, landmarks_input = row['video_id'], row['label'], row['landmarks']

        if isinstance(landmarks_input, str):
            original_frames_list = parse_landmarks(landmarks_input)
        elif isinstance(landmarks_input, list): # Already parsed
            original_frames_list = landmarks_input
        else:
            print(f"Warning: Unknown landmark format for {video_id} in augmentation. Skipping.")
            continue
        if not original_frames_list: continue

        # Add original data
        augmented_data_list.append({'video_id': video_id, 'label': label, 'landmarks': original_frames_list})

        # Generate augmented versions
        for i in range(augmentation_factor):
            augmented_frames_for_one_video = []
            for frame_dict in original_frames_list:
                if frame_dict.get('result') != 'DETECTION_SUCCESS' or not frame_dict.get('landmarks'):
                    augmented_frames_for_one_video.append({'result': 'DETECTION_FAILURE', 'landmarks': []})
                    continue

                new_frame_dict = {'result': 'DETECTION_SUCCESS', 'landmarks': []}
                for hand_landmarks_list in frame_dict['landmarks']:
                    augmented_hand = []
                    # Noise scale can be adjusted; simple increasing scale per augmentation round
                    noise_scale = 0.015 * (np.random.rand() + 0.5) * (i + 1) # Reduced noise scale slightly
                    for point in hand_landmarks_list:
                        point_id = point[0]
                        # Add noise, optionally proportional or fixed if coord is near zero
                        x_noise = np.random.normal(0, noise_scale * abs(point[1]) if abs(point[1]) > 1e-3 else noise_scale)
                        y_noise = np.random.normal(0, noise_scale * abs(point[2]) if abs(point[2]) > 1e-3 else noise_scale)
                        z_noise = np.random.normal(0, noise_scale * abs(point[3]) if abs(point[3]) > 1e-3 else noise_scale)
                        augmented_hand.append([point_id, point[1] + x_noise, point[2] + y_noise, point[3] + z_noise])
                    new_frame_dict['landmarks'].append(augmented_hand)
                augmented_frames_for_one_video.append(new_frame_dict)

            if augmented_frames_for_one_video:
                augmented_data_list.append({
                    'video_id': f"{video_id}_aug_{i+1}", 'label': label, 'landmarks': augmented_frames_for_one_video
                })
    return pd.DataFrame(augmented_data_list)


def pad_seq(sequence_of_frames, desired_sequence_length):
    """Pads a sequence of frames with dummy frames to reach desired_sequence_length."""
    # Dummy frame: one hand, 21 landmarks at (0,0,0)
    dummy_hand = [[i, 0.0, 0.0, 0.0] for i in range(21)] # 21 landmarks for a standard hand
    dummy_frame = {'result': 'DETECTION_SUCCESS', 'landmarks': [dummy_hand]}

    # Pad by appending dummy frames
    while len(sequence_of_frames) < desired_sequence_length:
        sequence_of_frames.append(dummy_frame.copy()) # Use .copy() for the dict
    return sequence_of_frames


def prepare_sequences(landmarks_dataframe, sequence_length=30, include_pairwise=True, pad_value=0.0):
    """
    Converts DataFrame of landmarks into sequences for model input.
    Each video becomes one sequence, padded/truncated to `sequence_length`.
    No sliding windows are used.
    Features: normalized coords, optional pairwise features, and deltas.
    """
    all_sequences_list = [] # Use list for appending, convert to numpy array at the end
    all_labels_list = []

    # Define feature dimensions
    NUM_HANDS_TO_PROCESS = 2
    NUM_LANDMARKS_PER_HAND = 21
    COORDS_PER_LANDMARK = 3
    base_coord_features_per_hand = NUM_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK # 63

    pairwise_feature_count_per_hand = 0
    if include_pairwise:
        # 6 keypoints: C(6,2)=15 distances. 5 fingertips to wrist: 5*3=15 cosines. Total 30.
        num_keypoints_pairwise = 6
        pairwise_feature_count_per_hand = (num_keypoints_pairwise * (num_keypoints_pairwise - 1) // 2) + (5 * 3)

    features_per_hand_no_deltas = base_coord_features_per_hand + pairwise_feature_count_per_hand
    total_features_frame_no_deltas = features_per_hand_no_deltas * NUM_HANDS_TO_PROCESS
    # Features per frame includes base features + delta features
    expected_features_per_frame = total_features_frame_no_deltas * 2

    print(f"Preparing one sequence per video. Target sequence length: {sequence_length}")
    print(f"Features per frame: {expected_features_per_frame} ({total_features_frame_no_deltas} base + {total_features_frame_no_deltas} deltas)")

    for _, row in tqdm(landmarks_dataframe.iterrows(), total=len(landmarks_dataframe)):
        video_id, label, landmarks_input = row['video_id'], row['label'], row['landmarks']

        if not landmarks_input: continue

        if isinstance(landmarks_input, str):
            current_video_frames_list = parse_landmarks(landmarks_input)
        elif isinstance(landmarks_input, list): # Already parsed
            current_video_frames_list = landmarks_input
        else:
            print(f"Warning: Unknown landmark format for {video_id}. Skipping.")
            continue
        if not current_video_frames_list: continue


        # Pad if shorter, truncate if longer to ensure fixed sequence_length
        processed_frames_for_sequence = []
        if len(current_video_frames_list) < sequence_length:
            # Pass a copy to pad_seq as it modifies the list in-place
            processed_frames_for_sequence = pad_seq(list(current_video_frames_list), sequence_length)
        else: # Longer or equal length
            processed_frames_for_sequence = current_video_frames_list[:sequence_length]


        # Extract features for each frame in the (now fixed-length) sequence
        base_features_for_each_frame_in_sequence = [] # Stores non-delta features for each frame
        for frame_data_dict in processed_frames_for_sequence: # Iterates `sequence_length` times
            normalized_frame_data = normalize_landmarks_enhanced(frame_data_dict)
            current_frame_base_features = [] # Base features for the hands in this single frame
            hands_processed_count = 0

            if normalized_frame_data and normalized_frame_data.get('landmarks'):
                for hand_idx, hand_normalized_landmarks in enumerate(normalized_frame_data['landmarks']):
                    if hand_idx >= NUM_HANDS_TO_PROCESS: break # Limit to max hands

                    # 1. Normalized coordinates for the hand
                    single_hand_coords_flat = []
                    # Create a map for easy lookup and ordered extraction
                    landmark_map = {lm[0]: lm[1:4] for lm in hand_normalized_landmarks}
                    for lm_id in range(NUM_LANDMARKS_PER_HAND): # Ensure all 21 landmarks are covered
                        coords = landmark_map.get(lm_id, [pad_value] * COORDS_PER_LANDMARK)
                        single_hand_coords_flat.extend(coords)
                    # Ensure correct length even if previous loop was faulty (e.g. bad input)
                    while len(single_hand_coords_flat) < base_coord_features_per_hand:
                        single_hand_coords_flat.append(pad_value)
                    current_frame_base_features.extend(single_hand_coords_flat[:base_coord_features_per_hand])

                    # 2. Pairwise features for the hand
                    if include_pairwise:
                        pairwise_feats = calculate_single_hand_pairwise_features(hand_normalized_landmarks, pad_value)
                        current_frame_base_features.extend(pairwise_feats)
                    # If not include_pairwise, feature vector for hand is just coords.
                    # The `features_per_hand_no_deltas` correctly reflects this.

                    hands_processed_count += 1

            # Pad if fewer than NUM_HANDS_TO_PROCESS were found/processed
            while hands_processed_count < NUM_HANDS_TO_PROCESS:
                current_frame_base_features.extend([pad_value] * features_per_hand_no_deltas)
                hands_processed_count += 1

            # Final check for total base features per frame (should match `total_features_frame_no_deltas`)
            while len(current_frame_base_features) < total_features_frame_no_deltas:
                current_frame_base_features.append(pad_value)
            base_features_for_each_frame_in_sequence.append(current_frame_base_features[:total_features_frame_no_deltas])


        # Calculate deltas and combine with base features
        final_feature_sequence_for_video = []
        if not base_features_for_each_frame_in_sequence: # Should not happen if padding works
            print(f"Warning: No base features for video {video_id}. Skipping.")
            continue

        # First frame: base features + zero deltas
        first_frame_base_feats = base_features_for_each_frame_in_sequence[0]
        deltas_for_first_frame = [pad_value] * total_features_frame_no_deltas
        final_feature_sequence_for_video.append(list(first_frame_base_feats) + deltas_for_first_frame)

        # Subsequent frames: current base features + (current base - previous base)
        for k in range(1, sequence_length):
            prev_frame_base_feats = np.array(base_features_for_each_frame_in_sequence[k-1])
            curr_frame_base_feats = np.array(base_features_for_each_frame_in_sequence[k])
            delta_feats = (curr_frame_base_feats - prev_frame_base_feats).tolist()
            final_feature_sequence_for_video.append(list(curr_frame_base_feats) + delta_feats)

        all_sequences_list.append(final_feature_sequence_for_video)
        all_labels_list.append(label)

    return np.array(all_sequences_list), np.array(all_labels_list)

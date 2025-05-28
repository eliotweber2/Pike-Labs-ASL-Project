import numpy as np
import tensorflow as tf
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.model_selection import KFold # Only if KFold is used
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Input, Conv1D, MaxPooling1D,
    TimeDistributed, LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam

# Assuming data_processing.py is in the same directory or Python path
from data_processing import prepare_sequences # Make sure this matches your file name

# --- MODEL DEFINITIONS FOR PER-FRAME PREDICTIONS ---

def create_attention_lstm_per_frame_model(n_classes, sequence_length, n_features):
    """LSTM-based model for per-frame predictions."""
    input_layer = Input(shape=(sequence_length, n_features), name="input_lstm")
    x = Bidirectional(LSTM(256, return_sequences=True))(input_layer)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = Dropout(0.3)(x)
    output_layer = TimeDistributed(Dense(n_classes, activation='softmax'))(x)
    model = Model(inputs=input_layer, outputs=output_layer, name="PerFrameLSTM")
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_lstm_per_frame_model(n_classes, sequence_length, n_features):
    """CNN-LSTM model for per-frame predictions.
    Note: Pooling reduces sequence length. Output will be shorter than input.
    """
    input_layer = Input(shape=(sequence_length, n_features), name="input_cnn_lstm")
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2, padding='same')(x) # seq_len /= 2
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x) # seq_len /= 4 (total reduction)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = Dropout(0.3)(x)
    
    # Output predictions for the pooled sequence length
    output_layer = TimeDistributed(Dense(n_classes, activation='softmax'))(x)

    model = Model(inputs=input_layer, outputs=output_layer, name="PerFrameCNN_LSTM")
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_transformer_per_frame_model(n_classes, sequence_length, n_features):
    """Transformer-based model for per-frame predictions."""
    input_layer = Input(shape=(sequence_length, n_features), name="input_transformer")
    x = LayerNormalization(epsilon=1e-6)(input_layer)
    for _ in range(2): # Number of transformer blocks
        key_dim = max(1, n_features // 8) # Ensure key_dim is positive
        attention_output = MultiHeadAttention(num_heads=8, key_dim=key_dim, dropout=0.1)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        ffn_output = Dense(512, activation='relu')(x)
        ffn_output = Dropout(0.1)(ffn_output)
        ffn_output = Dense(n_features)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = Dropout(0.1)(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = Dropout(0.1)(x)
    output_layer = TimeDistributed(Dense(n_classes, activation='softmax'))(x)
    model = Model(inputs=input_layer, outputs=output_layer, name="PerFrameTransformer")
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Ensemble and Training Functions ---

def create_ensemble_models_per_frame(n_classes, sequence_length, n_features):
    """Defines a list of per-frame model creation functions."""
    models_definitions = [
        ('Per_Frame_LSTM', create_attention_lstm_per_frame_model),
        ('Per_Frame_CNN_LSTM', create_cnn_lstm_per_frame_model),
        ('Per_Frame_Transformer', create_transformer_per_frame_model)
    ]
    return models_definitions

def train_ensemble_models(
    X_train, y_train_sequence, X_val, y_val_sequence,
    models_defs, sequence_length, n_features, n_classes, epochs=50, batch_size=32
):
    """Trains a list of models. Handles different output sequence lengths per model."""
    trained_models = []
    for name, model_create_func in models_defs:
        print(f"\nTraining {name} model...")
        model = model_create_func(n_classes, sequence_length, n_features)

        # Create labels that match this model's output sequence length
        model_output_seq_len = model.output_shape[1]
        if model_output_seq_len is None:
            # Dynamic shape, assume it matches input for non-CNN models
            model_output_seq_len = sequence_length
        
        # Adjust labels to match model output sequence length
        if model_output_seq_len != y_train_sequence.shape[1]:
            print(f"Adjusting labels for {name}: from {y_train_sequence.shape[1]} to {model_output_seq_len} frames")
            # For CNN-LSTM, we need to downsample the labels to match the pooled sequence length
            # Simple approach: take every nth label where n = original_length / new_length
            downsample_factor = y_train_sequence.shape[1] / model_output_seq_len
            indices = np.round(np.linspace(0, y_train_sequence.shape[1] - 1, model_output_seq_len)).astype(int)
            y_train_model = y_train_sequence[:, indices]
            y_val_model = y_val_sequence[:, indices]
        else:
            y_train_model = y_train_sequence
            y_val_model = y_val_sequence

        print(f"Model {name} - Input shape: {model.input_shape}, Output shape: {model.output_shape}")
        print(f"Model {name} - Training with label shape: {y_train_model.shape}")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(f'model/sign_language_model_{name}_per_frame.keras',
                                               monitor='val_loss', save_best_only=True)
        ]
        
        model.fit(X_train, y_train_model, validation_data=(X_val, y_val_model),
                  epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
        _, val_acc = model.evaluate(X_val, y_val_model, verbose=0)
        print(f"{name} final validation accuracy (per-frame): {val_acc:.4f}")
        trained_models.append((name, model))
    return trained_models

def ensemble_prediction_per_frame(models, X_data):
    """Ensemble predictions by averaging. Handles different sequence lengths by taking majority vote per original frame."""
    if not models:
        print("Error: No models provided for ensemble prediction.")
        return np.array([])

    original_seq_len = X_data.shape[1]
    n_samples = X_data.shape[0]
    
    # Get predictions from all models
    all_predictions = []
    for name, model in models:
        pred_raw = model.predict(X_data, verbose=0)
        model_seq_len = pred_raw.shape[1]
        
        if model_seq_len == original_seq_len:
            # Direct use for models with same sequence length
            all_predictions.append(pred_raw)
        else:
            # Upsample predictions to match original sequence length
            # Simple interpolation approach
            upsampled_pred = np.zeros((n_samples, original_seq_len, pred_raw.shape[2]))
            for i in range(n_samples):
                for class_idx in range(pred_raw.shape[2]):
                    # Interpolate each class probability
                    original_indices = np.linspace(0, model_seq_len - 1, original_seq_len)
                    upsampled_pred[i, :, class_idx] = np.interp(
                        original_indices, 
                        np.arange(model_seq_len), 
                        pred_raw[i, :, class_idx]
                    )
            all_predictions.append(upsampled_pred)
            print(f"Upsampled {name} predictions from {model_seq_len} to {original_seq_len} frames")

    if not all_predictions:
        print("Error: No valid predictions for ensembling.")
        return np.array([])

    # Average all predictions
    ensemble_pred = np.mean(all_predictions, axis=0)
    return ensemble_pred

def train_pipeline(train_df, val_df, sequence_length_config=30, pairwise_features_config=True):
    """Main training pipeline for per-frame prediction models."""
    print(f"--- Starting Training Pipeline ---")
    print(f"Config: sequence_length={sequence_length_config}, include_pairwise={pairwise_features_config}")

    # 1. Prepare data using the specified configurations
    print("Preparing training data...")
    X_train, y_train_raw = prepare_sequences(
        train_df,
        sequence_length=sequence_length_config,
        include_pairwise=pairwise_features_config
    )
    print("Preparing validation data...")
    X_val, y_val_raw = prepare_sequences(
        val_df,
        sequence_length=sequence_length_config,
        include_pairwise=pairwise_features_config
    )

    if X_train.size == 0 or X_val.size == 0:
        print("Error: Training or validation data is empty after prepare_sequences. Exiting.")
        return None, None

    # 2. Encode labels and adapt for per-frame output
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate((y_train_raw, y_val_raw), axis=0))
    y_train_single = label_encoder.transform(y_train_raw)
    y_val_single = label_encoder.transform(y_val_raw)

    # Actual sequence length from data (should match sequence_length_config)
    actual_sequence_length = X_train.shape[1]
    y_train_sequence = np.repeat(y_train_single[:, np.newaxis], actual_sequence_length, axis=1)
    y_val_sequence = np.repeat(y_val_single[:, np.newaxis], actual_sequence_length, axis=1)
    print(f"Label shapes (samples, seq_len): y_train_sequence={y_train_sequence.shape}, y_val_sequence={y_val_sequence.shape}")

    # 3. Get data dimensions for model creation
    actual_n_features = X_train.shape[2]
    n_classes = len(label_encoder.classes_)
    print(f"Data shapes: X_train={X_train.shape}, X_val={X_val.shape}")
    print(f"Classes: {n_classes}, Features per frame: {actual_n_features}, Sequence length: {actual_sequence_length}")

    # 4. Define and train models
    model_definitions = create_ensemble_models_per_frame(n_classes, actual_sequence_length, actual_n_features)
    os.makedirs("model", exist_ok=True)

    print("\n--- Training Ensemble Models (Per-Frame) ---")
    trained_models = train_ensemble_models(
        X_train, y_train_sequence, X_val, y_val_sequence,
        model_definitions, actual_sequence_length, actual_n_features, n_classes
    )

    # 5. Save label encoder
    with open('model/label_encoder_per_frame.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved to model/label_encoder_per_frame.pkl")

    if not trained_models:
        print("No models were trained. Skipping final ensemble evaluation.")
        return None, label_encoder

    # 6. Evaluate ensemble on validation set
    print("\n--- Evaluating Ensemble on Validation Data (Per-Frame) ---")
    ensemble_preds_prob = ensemble_prediction_per_frame(trained_models, X_val)

    if ensemble_preds_prob.size == 0:
        print("Ensemble prediction on validation data failed. Cannot evaluate.")
        return trained_models, label_encoder

    ensemble_classes = np.argmax(ensemble_preds_prob, axis=2)
    flat_y_val = y_val_sequence.flatten()
    flat_ensemble_preds = ensemble_classes.flatten()

    accuracy = np.mean(flat_ensemble_preds == flat_y_val)
    print(f"Ensemble validation accuracy (per-frame): {accuracy:.4f}")

    print("\nPer-Frame Classification Report (Validation Data):")
    report = classification_report(
        flat_y_val, flat_ensemble_preds,
        labels=np.arange(n_classes), target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0
    )
    print(report)
    print(f"--- Training Pipeline Finished ---")
    return trained_models, label_encoder

def load_ensemble_per_frame_models():
    """Loads previously trained per-frame ensemble models and label encoder."""
    print("Loading per-frame ensemble models...")
    loaded_models = []
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    try:
        model_files = [f for f in os.listdir(model_dir) if f.startswith('sign_language_model_') and f.endswith('_per_frame.keras')]
    except FileNotFoundError:
        print(f"Model directory '{model_dir}' not found.")
        return None, None

    if not model_files:
        print(f"No per-frame models found in '{model_dir}' directory.")
        return None, None

    for model_file in model_files:
        name = model_file.replace('sign_language_model_', '').replace('_per_frame.keras', '')
        try:
            model = tf.keras.models.load_model(os.path.join(model_dir, model_file))
            loaded_models.append((name, model))
            print(f"Loaded {name} model from {model_file}")
        except Exception as e:
            print(f"Error loading {name} model from {model_file}: {e}")

    label_encoder = None
    le_path = os.path.join(model_dir, 'label_encoder_per_frame.pkl')
    try:
        with open(le_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"Label encoder loaded from {le_path}")
    except Exception as e:
        print(f"Error loading label encoder from {le_path}: {e}")
    return loaded_models, label_encoder

def evaluate_pipeline(test_df, sequence_length_config=30, pairwise_features_config=True):
    """Evaluates the loaded per-frame ensemble on test data."""
    print(f"--- Starting Evaluation Pipeline ---")
    models, label_encoder = load_ensemble_per_frame_models()

    if not models or not label_encoder:
        print("Evaluation failed: Models or label encoder not loaded properly.")
        return

    # 1. Prepare test data
    print("Preparing test data for per-frame evaluation...")
    X_test, y_test_raw = prepare_sequences(
        test_df,
        sequence_length=sequence_length_config,
        include_pairwise=pairwise_features_config
    )

    if X_test.size == 0:
        print("Error: No test data generated from prepare_sequences.")
        return
    print(f"Test data X_test shape: {X_test.shape}")

    # 2. Encode labels and adapt for per-frame
    try:
        y_test_single = label_encoder.transform(y_test_raw)
    except ValueError as e:
        print(f"Error transforming test labels: {e}. Some labels in test set might be unseen during training.")
        unknown_labels = set(y_test_raw) - set(label_encoder.classes_)
        if unknown_labels:
            print(f"Unknown labels in test set: {unknown_labels}")
        mask = np.isin(y_test_raw, label_encoder.classes_)
        X_test = X_test[mask]
        y_test_raw_filtered = y_test_raw[mask]
        if X_test.size == 0:
            print("Error: No test data remaining after filtering unknown labels.")
            return
        y_test_single = label_encoder.transform(y_test_raw_filtered)

    actual_sequence_length = X_test.shape[1]
    y_test_sequence = np.repeat(y_test_single[:, np.newaxis], actual_sequence_length, axis=1)

    # 3. Generate ensemble predictions
    print("\nGenerating per-frame ensemble predictions for test data...")
    y_pred_prob_ensemble = ensemble_prediction_per_frame(models, X_test)

    if y_pred_prob_ensemble.size == 0:
        print("Ensemble prediction on test data failed or yielded no results.")
        return

    y_pred_ensemble_classes = np.argmax(y_pred_prob_ensemble, axis=2)

    # 4. Calculate and print per-frame metrics
    flat_y_test = y_test_sequence.flatten()
    flat_y_pred = y_pred_ensemble_classes.flatten()

    accuracy = np.mean(flat_y_pred == flat_y_test)
    print(f"Ensemble test accuracy (per-frame): {accuracy:.4f}")

    print("\nPer-Frame Classification Report (Test Data):")
    n_classes = len(label_encoder.classes_)
    report = classification_report(
        flat_y_test, flat_y_pred,
        labels=np.arange(n_classes),
        target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0
    )
    print(report)

    # 5. Plot confusion matrix
    cm = confusion_matrix(flat_y_test, flat_y_pred, labels=np.arange(n_classes))
    plt.figure(figsize=(max(10, n_classes // 2), max(8, n_classes // 2.5)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted (Per-Frame)')
    plt.ylabel('True (Per-Frame)')
    plt.title('Per-Frame Confusion Matrix (Test Data)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_per_frame_test.png')
    print("Per-frame confusion matrix for test data saved to confusion_matrix_per_frame_test.png")

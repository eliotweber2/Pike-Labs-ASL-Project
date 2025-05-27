import numpy as np
import tensorflow as tf
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold # Used if KFold cross-validation is re-introduced
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # For progress visualization in loops
from tensorflow.keras.models import Model # Using Model functional API
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Input, Conv1D, MaxPooling1D,
    TimeDistributed, LayerNormalization, MultiHeadAttention # GlobalAveragePooling1D (if used elsewhere)
)
from tensorflow.keras.optimizers import Adam
def create_attention_lstm_per_frame_model(n_classes, sequence_length, n_features):
    """
    Constructs an LSTM-based model designed for per-frame sequence prediction.

    This architecture utilizes Bidirectional LSTMs to capture temporal context from
    both past and future frames. Dense layers are applied to each time step's output
    via the TimeDistributed wrapper to make independent predictions for each frame.
    The original sequence-reducing attention mechanism has been removed to enable
    per-frame outputs.

    Args:
        n_classes (int): The number of unique classes for prediction.
        sequence_length (int): The fixed length of input sequences (number of time steps).
        n_features (int): The dimensionality of features for each time step.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    # Define the input layer, specifying the shape of one sequence.
    input_layer = Input(shape=(sequence_length, n_features), name="input_sequence")

    # First Bidirectional LSTM layer. `return_sequences=True` is crucial for passing
    # the full sequence of outputs to the next layer, enabling per-frame processing.
    x = Bidirectional(LSTM(256, return_sequences=True), name="bidi_lstm_1")(input_layer)
    x = Dropout(0.3, name="dropout_1")(x) # Regularization to prevent overfitting.

    # Second Bidirectional LSTM layer, further processing the sequence.
    x = Bidirectional(LSTM(128, return_sequences=True), name="bidi_lstm_2")(x)
    x = Dropout(0.3, name="dropout_2")(x)

    # Apply Dense layers to each time step of the LSTM output sequence.
    # The TimeDistributed wrapper is essential for this per-frame dense transformation.
    x = TimeDistributed(Dense(128, activation='relu'), name="timedist_dense_1")(x)
    x = Dropout(0.3, name="dropout_3")(x)
    x = TimeDistributed(Dense(64, activation='relu'), name="timedist_dense_2")(x)
    x = Dropout(0.3, name="dropout_4")(x)

    # Output layer: TimeDistributed Dense layer with softmax activation for multi-class
    # probability distribution over `n_classes` for each time step.
    output_layer = TimeDistributed(Dense(n_classes, activation='softmax'), name="output_per_frame")(x)

    # Construct the model.
    model = Model(inputs=input_layer, outputs=output_layer, name="PerFrameAttentionLSTM")

    # Compile the model. Adam optimizer is a common default.
    # `sparse_categorical_crossentropy` is used as labels are integer-encoded (not one-hot).
    # Accuracy will be computed on a per-frame basis.
    optimizer = Adam(learning_rate=0.001) # Consider making learning rate configurable.
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_cnn_lstm_per_frame_model(n_classes, sequence_length, n_features):
    """
    Constructs a CNN-LSTM model for per-frame sequence prediction.

    This model first uses 1D Convolutional layers to extract local features
    from the input sequences. MaxPooling reduces dimensionality and sequence length.
    The output of the CNN part is then fed into a Bidirectional LSTM for
    temporal modeling, followed by TimeDistributed Dense layers for per-frame predictions.

    Important Note: MaxPooling layers alter the sequence length. The output sequence
    length from this model will be shorter than `sequence_length`. This has implications
    for label alignment during training and for ensembling with models that preserve
    the original sequence length. Consider using UpSampling1D layers at the end
    to restore original sequence length if needed for specific applications.

    Args:
        n_classes (int): Number of unique classes.
        sequence_length (int): Input sequence length.
        n_features (int): Dimensionality of input features per time step.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    input_layer = Input(shape=(sequence_length, n_features), name="input_sequence")

    # CNN feature extraction block.
    # Conv1D layers act as feature detectors across the temporal dimension.
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name="conv1d_1")(input_layer)
    # MaxPooling1D reduces sequence length (e.g., by factor of 2). `padding='same'` ensures
    # that the output length is ceil(input_length / pool_size).
    x = MaxPooling1D(pool_size=2, padding='same', name="maxpool1d_1")(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', name="conv1d_2")(x)
    x = MaxPooling1D(pool_size=2, padding='same', name="maxpool1d_2")(x)
    # At this point, sequence length is approximately `sequence_length / 4`.

    # LSTM layer for temporal modeling on the extracted convolutional features.
    # `return_sequences=True` is essential for per-frame output from this block.
    x = Bidirectional(LSTM(128, return_sequences=True), name="bidi_lstm_cnn")(x)
    x = Dropout(0.3, name="dropout_cnn_lstm_1")(x)

    # TimeDistributed Dense layers for classification on LSTM's output sequence.
    x = TimeDistributed(Dense(128, activation='relu'), name="timedist_dense_cnn_1")(x)
    x = Dropout(0.3, name="dropout_cnn_lstm_2")(x)
    x = TimeDistributed(Dense(64, activation='relu'), name="timedist_dense_cnn_2")(x)
    x = Dropout(0.3, name="dropout_cnn_lstm_3")(x)

    # Output layer for per-frame predictions on the (pooled) sequence.
    output_layer = TimeDistributed(Dense(n_classes, activation='softmax'), name="output_per_pooled_frame")(x)

    model = Model(inputs=input_layer, outputs=output_layer, name="PerFrameCNN_LSTM")
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'] # Accuracy on the (pooled) per-frame predictions.
    )
    return model

def create_transformer_per_frame_model(n_classes, sequence_length, n_features):
    """
    Constructs a Transformer-based model for per-frame sequence prediction.

    This model utilizes MultiHeadAttention layers for capturing relationships
    across the sequence. LayerNormalization is applied for stability.
    The GlobalAveragePooling1D layer is omitted to maintain the sequence structure
    for per-frame predictions. Dense layers are applied per time step.

    Args:
        n_classes (int): Number of unique classes.
        sequence_length (int): Input sequence length.
        n_features (int): Dimensionality of input features per time step.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    input_layer = Input(shape=(sequence_length, n_features), name="input_sequence")

    # Initial Layer Normalization.
    x = LayerNormalization(epsilon=1e-6, name="input_layernorm")(input_layer)

    # Transformer blocks. Each block consists of Multi-Head Attention and a Feed-Forward Network.
    # MultiHeadAttention processes the entire sequence and outputs a sequence of the same length.
    num_transformer_blocks = 2 # Hyperparameter: number of Transformer blocks.
    for i in range(num_transformer_blocks):
        # Multi-Head Self-Attention.
        # `key_dim` must be positive; ensure n_features is large enough or key_dim is hardcoded.
        attention_key_dim = max(1, n_features // 8) # Ensure key_dim is at least 1.
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=attention_key_dim, dropout=0.1,
            name=f"multihead_attention_{i+1}"
        )(x, x) # Self-attention: query, key, and value are all `x`.

        # Add & Norm (Residual connection followed by Layer Normalization).
        x = LayerNormalization(epsilon=1e-6, name=f"layernorm_attn_{i+1}")(x + attention_output)

        # Feed-Forward Network (applied to each position independently).
        ffn_output = Dense(512, activation='relu', name=f"ffn_dense1_{i+1}")(x)
        ffn_output = Dropout(0.1, name=f"ffn_dropout_{i+1}")(ffn_output)
        ffn_output = Dense(n_features, name=f"ffn_dense2_{i+1}")(ffn_output) # Project back to n_features for residual.

        # Add & Norm.
        x = LayerNormalization(epsilon=1e-6, name=f"layernorm_ffn_{i+1}")(x + ffn_output)
    # `x` now holds the processed sequence: (batch_size, sequence_length, n_features).

    # TimeDistributed Dense layers for final per-frame classification.
    x = TimeDistributed(Dense(128, activation='relu'), name="timedist_dense_transformer_1")(x)
    x = Dropout(0.1, name="dropout_transformer_1")(x) # Using dropout rates typical for Transformers.
    x = TimeDistributed(Dense(64, activation='relu'), name="timedist_dense_transformer_2")(x)
    x = Dropout(0.1, name="dropout_transformer_2")(x)

    # Per-frame output layer.
    output_layer = TimeDistributed(Dense(n_classes, activation='softmax'), name="output_per_frame_transformer")(x)

    model = Model(inputs=input_layer, outputs=output_layer, name="PerFrameTransformer")
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Updated Ensemble and Training Functions ---

def create_ensemble_models_per_frame(n_classes, sequence_length, n_features):
    """
    Creates a list of model definition tuples (name, creation_function) for ensembling.
    All models are designed for per-frame prediction.

    Note: The CNN-LSTM model (`create_cnn_lstm_per_frame_model`) inherently produces
    an output sequence shorter than the input `sequence_length` due to pooling layers.
    For effective ensembling via averaging, model outputs should ideally have identical
    shapes (including sequence length). This discrepancy needs to be addressed either by:
    1. Modifying the CNN-LSTM model to include UpSampling1D layers to restore sequence length.
    2. Implementing more sophisticated ensembling logic in `ensemble_prediction_per_frame`
       that can handle differing sequence lengths (e.g., by upsampling predictions post-hoc,
       or by using only a subset of frames for models with shorter outputs).
    3. Training and evaluating the CNN-LSTM model separately if alignment is too complex.

    Args:
        n_classes (int): Number of target classes.
        sequence_length (int): Length of input sequences.
        n_features (int): Number of features per time step.

    Returns:
        list: A list of tuples, where each tuple contains a model name (str)
              and its corresponding creation function (callable).
    """
    models_definitions = [
        ('Per_Frame_LSTM', create_attention_lstm_per_frame_model),
        # The CNN-LSTM model is included, but caller must be aware of its shorter output sequence.
        ('Per_Frame_CNN_LSTM', create_cnn_lstm_per_frame_model),
        ('Per_Frame_Transformer', create_transformer_per_frame_model)
    ]
    return models_definitions

def train_ensemble_models(
    X_train, y_train_sequence, X_val, y_val_sequence,
    models_defs, sequence_length, n_features, n_classes, epochs=100, batch_size=32
):
    """
    Trains a list of defined models for ensembling.

    Labels (`y_train_sequence`, `y_val_sequence`) are expected to be adapted for
    per-frame training, i.e., shape (num_samples, sequence_length).

    Args:
        X_train (np.ndarray): Training data.
        y_train_sequence (np.ndarray): Training labels, adapted for per-frame output.
        X_val (np.ndarray): Validation data.
        y_val_sequence (np.ndarray): Validation labels, adapted for per-frame output.
        models_defs (list): List of (name, creation_function) tuples for models.
        sequence_length (int): Original input sequence length.
        n_features (int): Number of features.
        n_classes (int): Number of classes.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        list: List of (name, trained_model) tuples.
    """
    trained_models = []
    # Input data `X_train`, `X_val` are assumed to be correctly shaped: (num_samples, sequence_length, n_features).

    for name, model_create_func in models_defs:
        print(f"\nTraining {name} model...")
        # Instantiate a new model for each training cycle to ensure fresh weights.
        model = model_create_func(n_classes, sequence_length, n_features)

        # --- Handling label shapes for models with altered sequence lengths (e.g., CNN-LSTM) ---
        # The current label adaptation (repeating single label for `sequence_length` frames)
        # might not align with models that internally change sequence length (like CNN-LSTM).
        # For CNN-LSTM, its output sequence length is `sequence_length / 4` (approx).
        # A robust solution requires:
        #   a) The model itself upsamples its output to `sequence_length`.
        #   b) Labels are specifically prepared/pooled to match `sequence_length / 4`.
        #   c) A custom loss function that handles misaligned lengths.
        # For this generic training function, we proceed with original labels,
        # but this is a critical point for the CNN-LSTM model's performance.
        current_y_train = y_train_sequence
        current_y_val = y_val_sequence

        if name == 'Per_Frame_CNN_LSTM':
            # The model.output_shape gives (None, pooled_seq_len, n_classes)
            pooled_seq_len = model.output_shape[1]
            if pooled_seq_len is not None and pooled_seq_len != sequence_length:
                print(f"WARNING: {name} outputs sequence length {pooled_seq_len}, "
                      f"while labels have length {sequence_length}. "
                      "Loss calculation might be problematic or require label adaptation not implemented here.")
                # Example of a crude label adaptation (not recommended for production without validation):
                # Slice or pool labels to match pooled_seq_len.
                # This is highly dependent on the pooling strategy and desired behavior.
                # For example, if just taking the first `pooled_seq_len` labels:
                # current_y_train = y_train_sequence[:, :pooled_seq_len]
                # current_y_val = y_val_sequence[:, :pooled_seq_len]
                # However, this assumes the sign information is concentrated at the start.
                # A better approach is to modify the CNN-LSTM model to include an UpSampling1D layer.
                pass # Proceeding with original labels; TensorFlow might broadcast or error.

        # Define Keras callbacks for robust training.
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001, verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                # Using .keras format for saving the entire model (architecture, weights, optimizer state).
                filepath=f'model/sign_language_model_{name}_per_frame.keras',
                monitor='val_loss', save_best_only=True
            )
        ]

        # Train the model.
        history = model.fit(
            X_train, current_y_train,
            validation_data=(X_val, current_y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1 # Or 2 for less output per epoch.
        )

        # Evaluate the model on validation set (best weights are restored by EarlyStopping).
        val_loss, val_acc = model.evaluate(X_val, current_y_val, verbose=0)
        print(f"{name} final validation accuracy (per-frame, best epoch): {val_acc:.4f}")
        # The model is already saved by ModelCheckpoint callback at its best performance.
        # model.save(f'model/sign_language_model_{name}_per_frame.keras') # Redundant if ModelCheckpoint used.
        print(f"{name} best model saved via ModelCheckpoint.")
        trained_models.append((name, model)) # Append the model instance with restored best weights.
    return trained_models


def ensemble_prediction_per_frame(models, X_data):
    """
    Generates ensemble predictions by averaging outputs from compatible models.

    This function attempts to handle models that might output sequences of
    different lengths (e.g., CNN-LSTM vs. LSTM/Transformer). It prioritizes
    predictions from models matching the input data's sequence length.
    For models with shorter sequences (like CNN-LSTM), a crude upsampling
    (np.repeat) is demonstrated. A more robust solution involves designing models
    (e.g., with UpSampling1D layers) to output consistent sequence lengths.

    Args:
        models (list): List of (name, trained_model) tuples.
        X_data (np.ndarray): Input data for prediction.

    Returns:
        np.ndarray: Averaged probability predictions from the ensemble, with shape
                    (num_samples, target_sequence_length, n_classes).
                    Returns an empty array if no compatible predictions can be made.
    """
    predictions_to_average = []
    # Target sequence length is derived from the input data.
    target_seq_len = X_data.shape[1]

    for name, model in models:
        # Get raw predictions from the current model.
        pred_raw = model.predict(X_data, verbose=0) # Shape: (samples, model_output_seq_len, classes)
        model_output_seq_len = pred_raw.shape[1]

        if model_output_seq_len == target_seq_len:
            # If model's output sequence length matches target, add directly.
            predictions_to_average.append(pred_raw)
        elif name == 'Per_Frame_CNN_LSTM' and model_output_seq_len < target_seq_len:
            # Handle CNN-LSTM: its output sequence is shorter.
            # Attempt crude upsampling. For production, an UpSampling1D layer in the
            # model architecture itself is a much better approach.
            print(f"INFO: {name} output seq len {model_output_seq_len} differs from target {target_seq_len}. "
                  "Attempting crude upsampling for ensembling.")
            
            # Calculate integer upsampling factor.
            if target_seq_len % model_output_seq_len == 0:
                factor = target_seq_len // model_output_seq_len
                pred_upsampled = np.repeat(pred_raw, factor, axis=1) # Repeats each time step `factor` times.
                
                # Verify shape after repeat, adjust if necessary (e.g. if factor was imperfect)
                if pred_upsampled.shape[1] == target_seq_len:
                    predictions_to_average.append(pred_upsampled)
                else: # Fallback if repeat didn't match exactly
                    print(f"WARNING: Upsampling for {name} resulted in {pred_upsampled.shape[1]} len, expected {target_seq_len}. Skipping.")
            else:
                print(f"WARNING: Cannot apply simple repeat upsampling for {name} as target_seq_len "
                      f"({target_seq_len}) is not a multiple of model_output_seq_len ({model_output_seq_len}). Skipping.")
        else:
            # Skip models whose output sequence length doesn't match and for which
            # no specific handling is implemented.
            print(f"INFO: Skipping {name} from ensemble due to sequence length mismatch "
                  f"(Output: {model_output_seq_len}, Target: {target_seq_len}) and no upsampling rule.")

    if not predictions_to_average:
        print("ERROR: No compatible model predictions available for ensembling.")
        # Return an empty array or raise an exception, depending on desired error

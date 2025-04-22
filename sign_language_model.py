import numpy as np
import tensorflow as tf
import pickle
import time
import argparse
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Conv1D, MaxPooling1D
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
# Enhanced model with attention mechanism for higher accuracy
def create_attention_model(n_classes, sequence_length, n_features):
    """Create an enhanced model architecture with attention mechanism for higher accuracy."""
    
    # Input layer
    input_layer = Input(shape=(sequence_length, n_features))
    
    # First LSTM block
    x = Bidirectional(LSTM(256, return_sequences=True))(input_layer)
    x = Dropout(0.3)(x)
    
    # Second LSTM block
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    # Attention mechanism to focus on important frames
    attention = Dense(1, activation='tanh')(x)
    attention = tf.squeeze(attention, axis=-1)
    attention_weights = tf.nn.softmax(attention, axis=1)
    
    # Apply attention
    context = tf.reduce_sum(x * tf.expand_dims(attention_weights, axis=-1), axis=1)
    
    # Dense layers
    x = Dense(128, activation='relu')(context)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output_layer = Dense(n_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile with a learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
# Create CNN-LSTM model for ensemble
def create_cnn_lstm_model(n_classes, sequence_length, n_features):
    """Create a CNN-LSTM model for the ensemble."""
    input_layer = Input(shape=(sequence_length, n_features))
    
    # CNN layers
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output_layer = Dense(n_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
# Create Transformer model for ensemble
def create_transformer_model(n_classes, sequence_length, n_features):
    """Create a Transformer-based model for the ensemble."""
    input_layer = Input(shape=(sequence_length, n_features))
    
    # Normalization and positional embeddings
    x = LayerNormalization(epsilon=1e-6)(input_layer)
    
    # Transformer blocks
    for _ in range(2):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=64, dropout=0.1
        )(x, x)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        
        # Feed-forward
        ffn = Dense(512, activation='relu')(x)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(n_features)(ffn)
        
        # Add & Norm
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    
    # Output layer
    output_layer = Dense(n_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
# Create ensemble models
def create_ensemble_models(n_classes, sequence_length, n_features):
    """Create multiple models with different architectures for ensembling."""
    models = []
    
    # Model 1: Attention-based LSTM
    model1 = create_attention_model(n_classes, sequence_length, n_features)
    models.append(('Attention_LSTM', model1))
    
    # Model 2: CNN-LSTM
    model2 = create_cnn_lstm_model(n_classes, sequence_length, n_features)
    models.append(('CNN_LSTM', model2))
    
    # Model 3: Transformer-based
    model3 = create_transformer_model(n_classes, sequence_length, n_features)
    models.append(('Transformer', model3))
    
    return models
# Train ensemble models
def train_ensemble_models(X_train, y_train, X_val, y_val, models, label_encoder):
    """Train multiple models for ensembling."""
    trained_models = []
    
    for name, model in models:
        print(f"\nTraining {name} model...")
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,  # Train longer
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,  # More patience
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=7,
                    min_lr=0.0001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f'model/sign_language_model_{name}.h5',
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"{name} validation accuracy: {val_acc:.4f}")
        
        # Save the model
        model.save(f'model/sign_language_model_{name}.h5')
        print(f"{name} model saved")
        
        trained_models.append((name, model))
    
    return trained_models
# Make ensemble predictions
def ensemble_prediction(models, X_data):
    """Make predictions using ensemble of models."""
    predictions = []
    
    for name, model in models:
        pred = model.predict(X_data, verbose=0)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred
# High-accuracy training with cross-validation
def train_high_accuracy_model(train_df, val_df, sequence_length=30):
    """Train with cross-validation and ensemble models for high accuracy."""
    # Prepare sequences with enhanced features
    print("Preparing training data...")
    X_train, y_train_raw = prepare_sequences(train_df, sequence_length, include_pairwise=True)
    print("Training data shape:", X_train.shape)
    
    print("Preparing validation data...")
    X_val, y_val_raw = prepare_sequences(val_df, sequence_length, include_pairwise=True)
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
    
    # Create model ensemble
    print("Creating ensemble models...")
    models = create_ensemble_models(n_classes, sequence_length, n_features)
    
    # Create directories if they don't exist
    os.makedirs("model", exist_ok=True)
    
    # Train ensemble with cross-validation
    print("Training ensemble models...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accuracies = {}
    for name, _ in models:
        fold_accuracies[name] = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold + 1}/5")
        
        # Split data
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Train each model in ensemble
        for name, model in models:
            print(f"Training {name} on fold {fold + 1}...")
            
            # Reset model
            if name == 'Attention_LSTM':
                model = create_attention_model(n_classes, sequence_length, n_features)
            elif name == 'CNN_LSTM':
                model = create_cnn_lstm_model(n_classes, sequence_length, n_features)
            elif name == 'Transformer':
                model = create_transformer_model(n_classes, sequence_length, n_features)
            
            # Train
            model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=50,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )
            
            # Evaluate
            _, acc = model.evaluate(X_fold_val, y_fold_val, verbose=0)
            fold_accuracies[name].append(acc)
            print(f"{name} fold {fold + 1} accuracy: {acc:.4f}")
    
    # Print cross-validation results
    print("\nCross-validation results:")
    for name in fold_accuracies:
        mean_acc = np.mean(fold_accuracies[name])
        print(f"{name}: Mean accuracy = {mean_acc:.4f}")
    
    # Train final ensemble models on all training data
    trained_models = train_ensemble_models(X_train, y_train, X_val, y_val, models, label_encoder)
    
    # Save label encoder
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved to model/label_encoder.pkl")
    
    # Final evaluation on validation data
    print("\nEvaluating ensemble on validation data...")
    ensemble_preds = ensemble_prediction(trained_models, X_val)
    ensemble_classes = np.argmax(ensemble_preds, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(ensemble_classes == y_val)
    print(f"Ensemble validation accuracy: {accuracy:.4f}")
    
    return trained_models, label_encoder
# Load ensemble models
def load_ensemble_models():
    """Load trained ensemble models."""
    print("Loading ensemble models...")
    models = []
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Find model files
    model_files = [f for f in os.listdir('model') if f.startswith('sign_language_model_') and f.endswith('.h5')]
    
    if not model_files:
        print("No ensemble models found.")
        return None, None
    
    # Load each model
    for model_file in model_files:
        name = model_file.replace('sign_language_model_', '').replace('.h5', '')
        try:
            model = tf.keras.models.load_model(f'model/{model_file}')
            models.append((name, model))
            print(f"Loaded {name} model")
        except Exception as e:
            print(f"Error loading {name} model: {e}")
    
    # Load label encoder
    try:
        with open('model/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"Loaded label encoder with {len(label_encoder.classes_)} classes")
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        return models, None
    
    return models, label_encoder
# Evaluate the model
def evaluate_model(test_df):
    """Evaluate the ensemble model on test data."""
    # Load ensemble models
    models, label_encoder = load_ensemble_models()
    
    if not models or not label_encoder:
        print("Error: Models or label encoder not loaded correctly.")
        return
    
    # Prepare test sequences
    print("Preparing test data...")
    X_test, y_test_raw = prepare_sequences(test_df, include_pairwise=True)
    
    # Check if we have test data
    if len(X_test) == 0:
        print("Error: No test sequences could be created. Check your test data.")
        return
    
    print(f"Test data shape: {X_test.shape}")
    
    # Encode labels
    y_test = label_encoder.transform(y_test_raw)
    
    # Generate predictions for each model
    print("Generating individual model predictions...")
    model_accuracies = {}
    
    for name, model in models:
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_classes == y_test)
        model_accuracies[name] = accuracy
        print(f"{name} model accuracy: {accuracy:.4f}")
    
    # Generate ensemble predictions
    print("\nGenerating ensemble predictions...")
    y_pred_prob = ensemble_prediction(models, X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    print(f"Ensemble accuracy: {accuracy:.4f}")
    
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

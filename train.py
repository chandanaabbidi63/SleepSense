import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import joblib
from preprocess import load_and_preprocess_data

def train_gru_model():
    """Train GRU model for sleep quality prediction"""
    # Load and preprocess data
    X, y, features = load_and_preprocess_data('Sleep_health_and_lifestyle_dataset.csv')
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build GRU model
    print("Building GRU model...")
    model = models.Sequential([
        layers.GRU(128, activation='tanh', return_sequences=True, input_shape=(1, X.shape[2])),
        layers.Dropout(0.3),
        layers.GRU(64, activation='tanh', return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Train the model
    print("Training GRU model...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Accuracy (approximate): {r2*100:.2f}%")
    
    # Save the model
    model.save('gru_sleep_model.h5')
    print("Model saved as 'gru_sleep_model.h5'")
    
    return model, features, history

if __name__ == "__main__":
    model, features, history = train_gru_model()
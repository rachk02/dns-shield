#!/usr/bin/env python3
"""
Train Ensemble Models - LSTM, GRU, Random Forest
Requires training data from generate_data.py
"""

import pandas as pd
import numpy as np
import math
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# Paths
DATA_FILE = r"C:\dns-shield\data\test\test_domains.csv"
MODELS_DIR = Path(r"C:\dns-shield\models")

def extract_features(domain: str) -> np.ndarray:
    """Extract 15 features from domain"""
    domain_name = domain.split('.')[0]
    vowels = set('aeiouAEIOU')
    
    features = []
    
    # 1. Length
    features.append(len(domain_name))
    
    # 2. Entropy
    if domain_name:
        freq = {}
        for c in domain_name:
            freq[c] = freq.get(c, 0) + 1
        entropy = -sum((count/len(domain_name)) * math.log2(count/len(domain_name))
                       for count in freq.values() if count > 0)
        features.append(entropy)
    else:
        features.append(0)
    
    # 3. Consonant/vowel ratio
    consonants = sum(1 for c in domain_name if c.lower() in 'bcdfghjklmnpqrstvwxyz')
    vowel_count = sum(1 for c in domain_name if c in vowels)
    features.append(consonants / (vowel_count + 1))
    
    # 4. Vowel count
    features.append(vowel_count)
    
    # 5. Consonant count
    features.append(consonants)
    
    # 6. Digit count
    features.append(sum(1 for c in domain_name if c.isdigit()))
    
    # 7. Special char count
    features.append(sum(1 for c in domain_name if not c.isalnum()))
    
    # 8. Unique char count
    features.append(len(set(domain_name)))
    
    # 9. Max consecutive consonants
    max_cons = 0
    current = 0
    for c in domain_name:
        if c.lower() in 'bcdfghjklmnpqrstvwxyz':
            current += 1
            max_cons = max(max_cons, current)
        else:
            current = 0
    features.append(max_cons)
    
    # 10. Max consecutive same char
    max_same = 0
    if domain_name:
        current = 1
        for i in range(1, len(domain_name)):
            if domain_name[i] == domain_name[i-1]:
                current += 1
                max_same = max(max_same, current)
            else:
                current = 1
    features.append(max_same)
    
    # 11-15. Padding to 15 features
    features.extend([0] * (15 - len(features)))
    
    return np.array(features[:15])

def load_data(csv_file: str):
    """Load and prepare data"""
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    df = pd.read_csv(csv_file)
    print(f"Total samples: {len(df)}")
    print(f"Legitimate: {len(df[df['label'] == 0])}")
    print(f"Malicious: {len(df[df['label'] == 1])}")
    
    # Extract features
    print("\nExtracting features...")
    X = np.array([extract_features(d) for d in df['domain']])
    y = df['label'].values
    
    # Train/test split
    print("Train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape for LSTM/GRU
    X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_rnn': X_train_rnn,
        'X_test_rnn': X_test_rnn,
        'scaler': scaler
    }

def train_lstm(data: dict):
    """Train LSTM model"""
    print("\n" + "=" * 60)
    print("Training LSTM Model")
    print("=" * 60)
    
    model = keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(15, 1)),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    print("\nTraining...")
    history = model.fit(
        data['X_train_rnn'], data['y_train'],
        validation_data=(data['X_test_rnn'], data['y_test']),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    y_pred_lstm = (model.predict(data['X_test_rnn'], verbose=0) > 0.5).astype(int).flatten()
    precision = precision_score(data['y_test'], y_pred_lstm)
    recall = recall_score(data['y_test'], y_pred_lstm)
    f1 = f1_score(data['y_test'], y_pred_lstm)
    
    print(f"\nLSTM Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Save
    model_path = MODELS_DIR / 'lstm' / 'lstm_model.h5'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"✓ Model saved: {model_path}")
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def train_gru(data: dict):
    """Train GRU model"""
    print("\n" + "=" * 60)
    print("Training GRU Model")
    print("=" * 60)
    
    model = keras.Sequential([
        layers.GRU(64, return_sequences=True, input_shape=(15, 1)),
        layers.Dropout(0.2),
        layers.GRU(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    print("\nTraining...")
    history = model.fit(
        data['X_train_rnn'], data['y_train'],
        validation_data=(data['X_test_rnn'], data['y_test']),
        epochs=20,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    y_pred_gru = (model.predict(data['X_test_rnn'], verbose=0) > 0.5).astype(int).flatten()
    precision = precision_score(data['y_test'], y_pred_gru)
    recall = recall_score(data['y_test'], y_pred_gru)
    f1 = f1_score(data['y_test'], y_pred_gru)
    
    print(f"\nGRU Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Save
    model_path = MODELS_DIR / 'gru' / 'gru_model.h5'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"✓ Model saved: {model_path}")
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def train_rf(data: dict):
    """Train Random Forest model"""
    print("\n" + "=" * 60)
    print("Training Random Forest Model")
    print("=" * 60)
    
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("Training...")
    model.fit(data['X_train'], data['y_train'])
    
    # Evaluate
    y_pred_rf = model.predict(data['X_test'])
    precision = precision_score(data['y_test'], y_pred_rf)
    recall = recall_score(data['y_test'], y_pred_rf)
    f1 = f1_score(data['y_test'], y_pred_rf)
    
    print(f"\nRandom Forest Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Feature importance
    print(f"\nTop 5 Important Features:")
    importances = model.feature_importances_
    for idx in np.argsort(importances)[-5:][::-1]:
        print(f"  Feature {idx}: {importances[idx]:.4f}")
    
    # Save
    model_path = MODELS_DIR / 'rf' / 'rf_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

if __name__ == '__main__':
    try:
        # Load data
        data = load_data(DATA_FILE)
        
        # Train models
        lstm_results = train_lstm(data)
        gru_results = train_gru(data)
        rf_results = train_rf(data)
        
        # Summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"\nLSTM F1-Score: {lstm_results['f1']:.4f}")
        print(f"GRU F1-Score: {gru_results['f1']:.4f}")
        print(f"Random Forest F1-Score: {rf_results['f1']:.4f}")
        
        # Save metrics
        metrics = {
            'lstm': lstm_results,
            'gru': gru_results,
            'rf': rf_results,
            'ensemble_weights': {
                'lstm': 0.33,
                'gru': 0.33,
                'rf': 0.34
            }
        }
        
        metrics_path = MODELS_DIR / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Metrics saved: {metrics_path}")
        print("\n✓ Training complete!")
    
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
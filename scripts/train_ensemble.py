#!/usr/bin/env python3
"""
Train Ensemble Models - LSTM, GRU, Random Forest
Requires training data from generate_data.py
"""

import argparse
import math
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBOOST_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "train" / "train_domains.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = MODELS_DIR / "training_metrics.json"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "f1": "f1",
    "roc_auc": make_scorer(roc_auc_score, needs_threshold=True),
}

RF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

XGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
} if XGBOOST_AVAILABLE else {}

FEATURE_LABELS = [
    "length",
    "entropy",
    "consonant_vowel_ratio",
    "vowel_count",
    "consonant_count",
    "digit_count",
    "special_char_count",
    "unique_char_count",
    "max_consecutive_consonants",
    "max_consecutive_same_char",
    "pad_1",
    "pad_2",
    "pad_3",
    "pad_4",
    "pad_5",
]


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Return standard binary classification metrics."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    if y_proba is not None and y_proba.size:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


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

def load_dataset(
    csv_file: Path | str = DATA_FILE,
    val_split: float = 0.15,
    test_split: float = 0.15,
    save_scaler: bool = True
) -> Dict[str, Any]:
    """Load dataset, create train/val/test splits, scale features, persist scaler."""

    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {csv_path}")

    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"  Class {label}: {count} ({pct:.2f}%)")

    # Extract features matrix
    print("\nExtracting handcrafted features...")
    feature_matrix = np.vstack([extract_features(d) for d in df['domain']])
    labels = df['label'].to_numpy()

    # Split into train / test first
    if not 0 < test_split < 1:
        raise ValueError("test_split must be between 0 and 1")
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")

    X_temp, X_test, y_temp, y_test = train_test_split(
        feature_matrix,
        labels,
        test_size=test_split,
        random_state=42,
        stratify=labels
    )

    # Derive validation ratio from remaining data
    remaining = 1 - test_split
    val_ratio = val_split / remaining

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        random_state=42,
        stratify=y_temp
    )

    print(f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # Feature scaling (train only)
    print("\nFitting StandardScaler on train set...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if save_scaler:
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"✓ Scaler saved to {SCALER_PATH}")

    dataset = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'X_train_rnn': X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1)),
        'X_val_rnn': X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1)),
        'X_test_rnn': X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1)),
        'scaler': scaler,
        'feature_names': FEATURE_LABELS,
    }

    return dataset

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


def train_models(dataset: dict):
    """Train all models and persist metrics"""
    lstm_results = train_lstm(dataset)
    gru_results = train_gru(dataset)
    rf_results = train_rf(dataset)

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

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\n✓ Metrics saved: {METRICS_PATH}")
    return metrics


if __name__ == '__main__':
    try:
        data = load_dataset(DATA_FILE)
        metrics = train_models(data)

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        for name, values in metrics.items():
            if isinstance(values, dict) and 'f1' in values:
                print(f"\n{name.upper()} F1-Score: {values['f1']:.4f}")

        print("\n✓ Training complete!")
    
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
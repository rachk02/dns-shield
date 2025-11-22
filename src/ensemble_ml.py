#!/usr/bin/env python3
"""
Ensemble ML Service - LSTM + GRU + Random Forest voting
API endpoint: http://localhost:8003
"""

import numpy as np
import json
import time
import pickle
from datetime import datetime
from flask import Flask, request, jsonify, Response
from tensorflow import keras
from prometheus_client import generate_latest, REGISTRY
import joblib

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.redis_client import redis_client
from src.utils.metrics import (
    record_latency, record_error, record_dns_query, record_accept_decision,
    record_block_decision
)

logger = get_logger(__name__, "ensemble_ml.log")
app = Flask(__name__)

class EnsembleML:
    """Ensemble of LSTM, GRU, Random Forest models"""
    
    def __init__(self):
        logger.info("Loading ML models...")
        try:
            # Load pre-trained models
            self.lstm_model = keras.models.load_model(Config.LSTM_MODEL_PATH)
            self.gru_model = keras.models.load_model(Config.GRU_MODEL_PATH)
            self.rf_model = joblib.load(Config.RF_MODEL_PATH)
            
            logger.info("[OK] All models loaded successfully")
        except FileNotFoundError:
            logger.warning(f"Models not found at configured paths")
            logger.warning("Using mock models for demo")
            self.lstm_model = None
            self.gru_model = None
            self.rf_model = None
    
    def extract_features(self, domain: str) -> np.ndarray:
        """Extract features from domain"""
        import math
        
        domain_name = domain.split('.')[0]
        vowels = set('aeiouAEIOU')
        
        features = []
        
        # Length
        features.append(len(domain_name))
        
        # Entropy
        if domain_name:
            freq = {}
            for c in domain_name:
                freq[c] = freq.get(c, 0) + 1
            entropy = -sum((count/len(domain_name)) * math.log2(count/len(domain_name))
                          for count in freq.values() if count > 0)
            features.append(entropy)
        else:
            features.append(0)
        
        # Consonant/vowel ratio
        consonants = sum(1 for c in domain_name if c.lower() in 'bcdfghjklmnpqrstvwxyz')
        vowel_count = sum(1 for c in domain_name if c in vowels)
        features.append(consonants / (vowel_count + 1))
        
        # Add more features up to 15
        features.extend([0] * (15 - len(features)))
        
        return np.array(features[:15])
    
    def predict(self, domain: str) -> dict:
        """Predict using ensemble voting"""
        try:
            features = self.extract_features(domain)
            features = features.reshape(1, -1)
            
            # If models not available, return dummy prediction
            if self.lstm_model is None or self.gru_model is None or self.rf_model is None:
                logger.warning("Using mock prediction (models not loaded)")
                score = np.random.random()
                decision = 'BLOCK' if score > 0.65 else 'ACCEPT'
                
                return {
                    'domain': domain,
                    'decision': decision,
                    'ensemble_score': round(score, 4),
                    'lstm_score': round(np.random.random(), 4),
                    'gru_score': round(np.random.random(), 4),
                    'rf_score': round(np.random.random(), 4),
                    'timestamp': datetime.now().isoformat(),
                    'note': 'Mock prediction (models not loaded)'
                }
            
            # Get predictions from each model
            lstm_pred = self.lstm_model.predict(features, verbose=0)[0][0]
            gru_pred = self.gru_model.predict(features, verbose=0)[0][0]
            rf_pred = self.rf_model.predict_proba(features)[0][1]
            
            # Weighted voting
            ensemble_score = (
                Config.LSTM_WEIGHT * lstm_pred +
                Config.GRU_WEIGHT * gru_pred +
                Config.RF_WEIGHT * rf_pred
            )
            
            # Decision
            decision = 'BLOCK' if ensemble_score > Config.ENSEMBLE_THRESHOLD else 'ACCEPT'
            
            result = {
                'domain': domain,
                'decision': decision,
                'ensemble_score': round(float(ensemble_score), 4),
                'lstm_score': round(float(lstm_pred), 4),
                'gru_score': round(float(gru_pred), 4),
                'rf_score': round(float(rf_pred), 4),
                'lstm_weight': Config.LSTM_WEIGHT,
                'gru_weight': Config.GRU_WEIGHT,
                'rf_weight': Config.RF_WEIGHT,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            record_error('ensemble', 'prediction_error')
            raise

ensemble = EnsembleML()

# =============================================
# ENDPOINTS
# =============================================

@app.route('/', methods=['GET'])
def root():
    """Service information and available endpoints"""
    return jsonify({
        'service': 'Ensemble ML',
        'version': '1.0.0',
        'description': 'Ensemble voting using LSTM, GRU, and Random Forest models',
        'port': 8003,
        'endpoints': {
            'POST /predict': 'Predict using ensemble voting for single domain',
            'POST /batch': 'Batch predict multiple domains (max 100)',
            'GET /models': 'Get loaded models information',
            'GET /health': 'Health check',
            'GET /metrics': 'Prometheus metrics'
        },
        'example_request': {
            'endpoint': 'POST /predict',
            'body': {'domain': 'example.com'}
        },
        'documentation': 'https://github.com/rachk02/dns_shield'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict using ensemble"""
    start_time = time.time()
    try:
        data = request.json
        domain = data.get('domain', '').lower().strip() if data else ''
        
        if not domain:
            return jsonify({'error': 'Domain required'}), 400
        
        record_dns_query('ensemble')
        
        # Check cache
        cached = redis_client.get(f"ensemble:{domain}")
        if cached:
            duration_ms = (time.time() - start_time) * 1000
            record_latency('POST', '/predict', 'ensemble', duration_ms)
            return jsonify(cached)
        
        # Predict
        result = ensemble.predict(domain)
        
        # Cache result (1h TTL)
        redis_client.set(f"ensemble:{domain}", result, 3600)
        
        # Record decision
        if result['decision'] == 'BLOCK':
            record_block_decision('ensemble', 'ensemble_voting')
        else:
            record_accept_decision('ensemble')
        
        duration_ms = (time.time() - start_time) * 1000
        record_latency('POST', '/predict', 'ensemble', duration_ms)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        record_error('ensemble', 'endpoint_error')
        error_body = json.dumps({'error': repr(e)})
        return Response(error_body, status=500, mimetype='application/json')

@app.route('/batch', methods=['POST'])
def batch_predict():
    """Batch predict"""
    start_time = time.time()
    try:
        data = request.json
        domains = data.get('domains', [])[:100] if data else []
        
        if not domains:
            return jsonify({'error': 'Domains list required'}), 400
        
        results = []
        for domain in domains:
            try:
                result = ensemble.predict(domain.lower())
                results.append(result)
            except Exception as e:
                logger.warning(f"Error predicting {domain}: {e}")
                continue
        
        duration_ms = (time.time() - start_time) * 1000
        record_latency('POST', '/batch', 'ensemble', duration_ms)
        
        return jsonify({'count': len(results), 'results': results})
    
    except Exception as e:
        logger.error(f"Batch error: {e}")
        record_error('ensemble', 'batch_error')
        error_body = json.dumps({'error': repr(e)})
        return Response(error_body, status=500, mimetype='application/json')

@app.route('/models', methods=['GET'])
def get_models():
    """Get loaded models info"""
    return jsonify({
        'lstm': Config.LSTM_MODEL_PATH,
        'gru': Config.GRU_MODEL_PATH,
        'rf': Config.RF_MODEL_PATH,
        'lstm_weight': Config.LSTM_WEIGHT,
        'gru_weight': Config.GRU_WEIGHT,
        'rf_weight': Config.RF_WEIGHT,
        'ensemble_threshold': Config.ENSEMBLE_THRESHOLD
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    try:
        redis_status = 'ok' if redis_client._redis.ping() else 'down'
    except:
        redis_status = 'down'
    
    return jsonify({
        'status': 'healthy',
        'service': 'ensemble_ml',
        'timestamp': datetime.now().isoformat(),
        'redis': redis_status,
        'models_loaded': ensemble.lstm_model is not None,
        'version': '1.0.0'
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics"""
    return generate_latest(REGISTRY).decode('utf-8'), 200, {'Content-Type': 'text/plain; charset=utf-8'}

# =============================================
# MAIN
# =============================================

if __name__ == '__main__':
    logger.info("Starting Ensemble ML Service on port 8003...")
    app.run(
        host=Config.API_HOST,
        port=8003,
        debug=Config.FLASK_DEBUG,
        threaded=True
    )
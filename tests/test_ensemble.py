#!/usr/bin/env python3
"""
Test Ensemble ML - Unit tests for ensemble predictions
"""

import pytest
import json
import requests
import numpy as np
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, r'C:\dns-shield')
from src.ensemble_ml import ensemble, app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestEnsembleML:
    """Test ensemble ML logic"""
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        features = ensemble.extract_features("google.com")
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 15
        assert all(isinstance(f, (int, float, np.number)) for f in features)
    
    def test_feature_consistency(self):
        """Test feature extraction consistency"""
        domain = "example.com"
        
        features1 = ensemble.extract_features(domain)
        features2 = ensemble.extract_features(domain)
        
        np.testing.assert_array_equal(features1, features2)
    
    def test_prediction_legitimate(self):
        """Test prediction for legitimate domain"""
        result = ensemble.predict("google.com")
        
        assert 'domain' in result
        assert 'decision' in result
        assert 'ensemble_score' in result
        assert result['decision'] in ['BLOCK', 'ACCEPT']
        assert 0 <= result['ensemble_score'] <= 1
    
    def test_prediction_malicious(self):
        """Test prediction for malicious domain"""
        result = ensemble.predict("xkjhqwerty.com")
        
        assert 'domain' in result
        assert 'decision' in result
        assert 'ensemble_score' in result
    
    def test_model_weights(self):
        """Test model weights sum to 1"""
        from src.utils.config import Config
        
        total_weight = Config.LSTM_WEIGHT + Config.GRU_WEIGHT + Config.RF_WEIGHT
        assert abs(total_weight - 1.0) < 0.001

class TestEnsembleEndpoints:
    """Test API endpoints"""
    
    def test_predict_endpoint(self, client):
        """Test /predict endpoint"""
        response = client.post('/predict',
            json={'domain': 'google.com'},
            content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'domain' in data
        assert 'decision' in data
        assert 'ensemble_score' in data
        assert 'lstm_score' in data
        assert 'gru_score' in data
        assert 'rf_score' in data
    
    def test_predict_invalid(self, client):
        """Test /predict with invalid input"""
        response = client.post('/predict',
            json={'domain': ''},
            content_type='application/json')
        
        assert response.status_code == 400
    
    def test_batch_predict(self, client):
        """Test /batch endpoint"""
        domains = ['google.com', 'amazon.com', 'facebook.com', 'xkjhqwerty.com']
        response = client.post('/batch',
            json={'domains': domains},
            content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['count'] <= 100  # Max batch
        assert len(data['results']) > 0
        
        # Verify all results have required fields
        for result in data['results']:
            assert 'domain' in result
            assert 'decision' in result
    
    def test_models_endpoint(self, client):
        """Test /models endpoint"""
        response = client.get('/models')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'lstm_weight' in data
        assert 'gru_weight' in data
        assert 'rf_weight' in data
        assert 'ensemble_threshold' in data
    
    def test_health_endpoint(self, client):
        """Test /health endpoint"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'healthy'
        assert data['service'] == 'ensemble_ml'
    
    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint"""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        assert b'HELP' in response.data

class TestEnsembleIntegration:
    """Integration tests"""
    
    def test_service_running(self):
        """Test if service is running"""
        try:
            response = requests.get('http://localhost:8003/health', timeout=2)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("Ensemble service not running on port 8003")
    
    def test_voting_mechanism(self):
        """Test ensemble voting"""
        result = ensemble.predict("test.com")
        
        lstm_score = result['lstm_score']
        gru_score = result['gru_score']
        rf_score = result['rf_score']
        ensemble_score = result['ensemble_score']
        
        # Verify voting calculation
        from src.utils.config import Config
        expected = (
            Config.LSTM_WEIGHT * lstm_score +
            Config.GRU_WEIGHT * gru_score +
            Config.RF_WEIGHT * rf_score
        )
        
        assert abs(ensemble_score - expected) < 0.001
    
    def test_decision_threshold(self):
        """Test decision threshold"""
        from src.utils.config import Config
        
        result = ensemble.predict("google.com")
        threshold = Config.ENSEMBLE_THRESHOLD
        
        if result['ensemble_score'] > threshold:
            assert result['decision'] == 'BLOCK'
        else:
            assert result['decision'] == 'ACCEPT'

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
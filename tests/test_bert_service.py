#!/usr/bin/env python3
"""
Test BERT Service - Unit tests for BERT classification
"""

import pytest
import json
import requests
import numpy as np
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, r'C:\dns-shield')
from src.bert_service import analyzer, app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestBERTAnalyzer:
    """Test BERT analyzer logic"""
    
    def test_embedding_dimension(self):
        """Test embedding produces correct dimensions"""
        embedding = analyzer.get_embedding("google.com")
        
        assert isinstance(embedding, list)
        # BERT base outputs 768D normally, or reduced by PCA
        assert len(embedding) > 0
    
    def test_embedding_consistency(self):
        """Test same domain produces same embedding"""
        domain = "example.com"
        
        embedding1 = analyzer.get_embedding(domain)
        embedding2 = analyzer.get_embedding(domain)
        
        # Should be identical (or very close due to floating point)
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=5)
    
    def test_classification_legitimate(self):
        """Test classification of legitimate domain"""
        result = analyzer.classify("google.com")
        
        assert 'domain' in result
        assert 'classification' in result
        assert 'score' in result
        assert 0 <= result['score'] <= 1
    
    def test_classification_malicious(self):
        """Test classification of malicious domain"""
        result = analyzer.classify("xkjhqwerty.com")
        
        assert 'domain' in result
        assert 'classification' in result
        assert 'score' in result

class TestBERTEndpoints:
    """Test API endpoints"""
    
    def test_embed_endpoint(self, client):
        """Test /embed endpoint"""
        response = client.post('/embed',
            json={'domain': 'google.com'},
            content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'domain' in data
        assert 'embedding' in data
        assert 'embedding_dim' in data
        assert len(data['embedding']) > 0
    
    def test_embed_invalid(self, client):
        """Test /embed with invalid input"""
        response = client.post('/embed',
            json={'domain': ''},
            content_type='application/json')
        
        assert response.status_code == 400
    
    def test_classify_endpoint(self, client):
        """Test /classify endpoint"""
        response = client.post('/classify',
            json={'domain': 'google.com'},
            content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'domain' in data
        assert 'classification' in data
        assert 'score' in data
    
    def test_batch_embed(self, client):
        """Test /batch endpoint"""
        domains = ['google.com', 'amazon.com', 'facebook.com']
        response = client.post('/batch',
            json={'domains': domains},
            content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['count'] <= 32  # Max batch size
        assert len(data['results']) > 0
    
    def test_health_endpoint(self, client):
        """Test /health endpoint"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert data['status'] == 'healthy'
        assert data['service'] == 'bert_service'
        assert 'device' in data
    
    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint"""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        assert b'HELP' in response.data

class TestBERTIntegration:
    """Integration tests"""
    
    def test_service_running(self):
        """Test if service is running"""
        try:
            response = requests.get('http://localhost:8002/health', timeout=2)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("BERT service not running on port 8002")
    
    def test_caching(self, client):
        """Test embedding caching"""
        domain = 'example.com'
        
        # First request
        response1 = client.post('/embed',
            json={'domain': domain},
            content_type='application/json')
        
        # Second request (should be cached)
        response2 = client.post('/embed',
            json={'domain': domain},
            content_type='application/json')
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(data1['embedding'], data2['embedding'], decimal=5)

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
#!/usr/bin/env python3
"""
Test DGA Detector - Unit tests for DGA detection service
"""

import pytest
import json
import requests
from unittest.mock import patch, MagicMock

# Import the service
import sys
sys.path.insert(0, r'C:\dns-shield')
from src.dga_detector import detector, app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestDGADetector:
    """Test DGA detection logic"""
    
    def test_entropy_calculation(self):
        """Test entropy calculation"""
        # High entropy (DGA)
        entropy_high = detector.calculate_entropy("lkjhgfdsmnbvcxzpoiuytrewq")
        assert entropy_high > 4.0, "High entropy expected"
        
        # Low entropy (legitimate)
        entropy_low = detector.calculate_entropy("google")
        assert entropy_low < 4.0, "Low entropy expected"
    
    def test_consonant_vowel_ratio(self):
        """Test C/V ratio calculation"""
        # High C/V (DGA)
        ratio_high = detector.consonant_vowel_ratio("lkjhgfdsmnbvcxzpoiuytrewq")
        assert ratio_high > 2.0, "High C/V ratio expected"
        
        # Low C/V (legitimate)
        ratio_low = detector.consonant_vowel_ratio("hello")
        assert ratio_low < 2.0, "Low C/V ratio expected"
    
    def test_suspicious_patterns(self):
        """Test suspicious pattern detection"""
        # Should detect
        assert detector.check_suspicious_patterns("bcdfgh") == True
        assert detector.check_suspicious_patterns("123456789123") == True
        assert detector.check_suspicious_patterns("aaabbbccc") == True
        
        # Should not detect
        assert detector.check_suspicious_patterns("google") == False
        assert detector.check_suspicious_patterns("amazon") == False
    
    def test_analyze_legitimate(self):
        """Test legitimate domain analysis"""
        result = detector.analyze("google.com")
        
        assert result['domain'] == 'google.com'
        assert result['is_dga'] == False
        assert result['score'] < 0.65
    
    def test_analyze_dga(self, monkeypatch):
        """Test DGA domain analysis"""
        # Patch thresholds to ensure test is independent of config
        monkeypatch.setattr(detector, 'entropy_threshold', 4.0)
        monkeypatch.setattr(detector, 'length_threshold', 20)
        monkeypatch.setattr(detector, 'cv_ratio_high', 3.0)

        result = detector.analyze("lkjhgfdsmnbvcxzpoiuytrewq.com")
        
        assert result['domain'] == 'lkjhgfdsmnbvcxzpoiuytrewq.com'
        assert result['is_dga'] == True
        assert result['score'] > 0.65

class TestDGAEndpoints:
    """Test API endpoints"""
    
    def test_analyze_endpoint_valid(self, client):
        """Test /analyze endpoint with valid input"""
        response = client.post('/analyze', 
            json={'domain': 'google.com'},
            content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'domain' in data
        assert 'is_dga' in data
        assert 'score' in data
    
    def test_analyze_endpoint_invalid(self, client):
        """Test /analyze endpoint with invalid input"""
        response = client.post('/analyze',
            json={'domain': ''},
            content_type='application/json')
        
        assert response.status_code == 400
    
    def test_analyze_endpoint_missing(self, client):
        """Test /analyze endpoint with missing domain"""
        response = client.post('/analyze',
            json={},
            content_type='application/json')
        
        assert response.status_code == 400
    
    def test_batch_endpoint(self, client):
        """Test /batch endpoint"""
        domains = ['google.com', 'xkjhqwerty.com', 'amazon.com']
        response = client.post('/batch',
            json={'domains': domains},
            content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['count'] == 3
        assert len(data['results']) == 3
    
    def test_health_endpoint(self, client):
        """Test /health endpoint"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['service'] == 'dga_detector'
    
    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint"""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        assert b'HELP' in response.data  # Prometheus format

class TestDGAIntegration:
    """Integration tests"""
    
    def test_service_running(self):
        """Test if service is running"""
        try:
            response = requests.get('http://localhost:8001/health', timeout=2)
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("DGA service not running on port 8001")
    
    def test_full_workflow(self, client):
        """Test complete workflow"""
        domains = ['google.com', 'xkjhqwerty.com']
        
        for domain in domains:
            response = client.post('/analyze',
                json={'domain': domain},
                content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Verify response structure
            assert 'domain' in data
            assert 'is_dga' in data
            assert 'score' in data
            assert 'metrics' in data
            assert 'timestamp' in data

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
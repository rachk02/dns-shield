#!/usr/bin/env python3
"""
Integration Tests - End-to-end testing of the complete system
"""

import pytest
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://localhost:9000"
TIMEOUT = 5

class TestSystemIntegration:
    """Full system integration tests"""
    
    @classmethod
    def setup_class(cls):
        """Setup - verify all services running"""
        services = [
            ("http://localhost:8001/health", "DGA"),
            ("http://localhost:8002/health", "BERT"),
            ("http://localhost:8003/health", "Ensemble"),
            ("http://localhost:9000/health", "Gateway"),
        ]
        
        for url, service in services:
            try:
                response = requests.get(url, timeout=TIMEOUT)
                assert response.status_code == 200
                print(f"✓ {service} service online")
            except Exception as e:
                pytest.skip(f"{service} service not available: {e}")
    
    def test_gateway_analyze_legitimate(self):
        """Test full cascade for legitimate domain"""
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"domain": "google.com"},
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data['domain'] == 'google.com'
        assert 'decision' in data
        assert 'confidence' in data
        assert 'stage_resolved' in data
        assert 'latency_ms' in data
        assert 'scores' in data
        
        # Legitimate domain should likely be accepted
        print(f"Result: {data['decision']} (confidence: {data['confidence']:.3f})")
    
    def test_gateway_analyze_malicious(self):
        """Test full cascade for malicious domain"""
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"domain": "xkjhqwerty.com"},
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data['domain'] == 'xkjhqwerty.com'
        assert 'decision' in data
        assert 'confidence' in data
        
        print(f"Result: {data['decision']} (confidence: {data['confidence']:.3f})")
    
    def test_cascade_latency(self):
        """Test cascade latency is within acceptable bounds"""
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"domain": "example.com"},
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        latency = data['latency_ms']
        print(f"Cascade latency: {latency:.2f}ms")
        
        # Should complete in < 5000ms (with BERT loading time)
        assert latency < 5000
    
    def test_stage_resolution(self):
        """Test stage resolution"""
        # High confidence DGA should resolve at stage 1
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"domain": "bcdfghjklmnpqrstvwxyz.com"},
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Stage should be 1, 2, or 3
        assert data['stage_resolved'] in [1, 2, 3]
        print(f"Resolved at stage: {data['stage_resolved']}")
    
    def test_batch_cascade(self):
        """Test batch processing"""
        domains = [
            'google.com',
            'amazon.com',
            'facebook.com',
            'xkjhqwerty.com',
            'bcdfghjkl.com'
        ]
        
        response = requests.post(
            f"{BASE_URL}/batch",
            json={"domains": domains},
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['count'] > 0
        assert len(data['results']) > 0
        
        for result in data['results']:
            assert 'domain' in result
            assert 'decision' in result
        
        print(f"Batch processed {data['count']} domains")
    
    def test_gateway_health(self):
        """Test gateway health check"""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'healthy'
        assert 'services' in data
        
        # Check individual services
        for service, status in data['services'].items():
            print(f"Service {service}: {status}")
    
    def test_concurrent_requests(self):
        """Test concurrent requests"""
        domains = ['google.com', 'amazon.com', 'facebook.com'] * 3
        
        def analyze_domain(domain):
            response = requests.post(
                f"{BASE_URL}/analyze",
                json={"domain": domain},
                timeout=TIMEOUT
            )
            return response.status_code == 200
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(analyze_domain, domains))
        
        assert all(results), "Some concurrent requests failed"
        print(f"✓ All {len(results)} concurrent requests succeeded")
    
    def test_decision_consistency(self):
        """Test decision consistency"""
        domain = "example.com"
        
        # Multiple requests for same domain
        results = []
        for _ in range(3):
            response = requests.post(
                f"{BASE_URL}/analyze",
                json={"domain": domain},
                timeout=TIMEOUT
            )
            results.append(response.json()['decision'])
        
        # All decisions should be consistent
        assert len(set(results)) == 1, "Inconsistent decisions"
        print(f"✓ Decision consistent: {results[0]}")
    
    def test_statistics_endpoint(self):
        """Test statistics endpoint"""
        # First, send some queries
        for domain in ['google.com', 'amazon.com']:
            requests.post(
                f"{BASE_URL}/analyze",
                json={"domain": domain},
                timeout=TIMEOUT
            )
        
        # Get stats
        response = requests.get(f"{BASE_URL}/stats", timeout=TIMEOUT)
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'daily' in data
        assert 'suspicious_domains' in data
    
    def test_error_handling(self):
        """Test error handling"""
        # Missing domain
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={},
            timeout=TIMEOUT
        )
        assert response.status_code == 400
        
        # Empty domain
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"domain": ""},
            timeout=TIMEOUT
        )
        assert response.status_code == 400
        
        # Invalid JSON
        response = requests.post(
            f"{BASE_URL}/analyze",
            data="invalid",
            timeout=TIMEOUT
        )
        assert response.status_code in [400, 500]

class TestPerformance:
    """Performance benchmarks"""
    
    def test_throughput(self):
        """Test system throughput"""
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < 10:  # 10 second test
            response = requests.post(
                f"{BASE_URL}/analyze",
                json={"domain": f"test{count}.com"},
                timeout=TIMEOUT
            )
            if response.status_code == 200:
                count += 1
        
        duration = time.time() - start_time
        throughput = count / duration
        
        print(f"Throughput: {throughput:.2f} requests/sec")
        assert throughput > 1, "Throughput too low"
    
    def test_p95_latency(self):
        """Test 95th percentile latency"""
        latencies = []
        
        for i in range(100):
            response = requests.post(
                f"{BASE_URL}/analyze",
                json={"domain": f"test{i}.com"},
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                latency = response.json()['latency_ms']
                latencies.append(latency)
        
        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        
        print(f"P95 latency: {p95:.2f}ms")

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
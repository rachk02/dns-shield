#!/usr/bin/env python3
"""
DGA Detector Service - Heuristic-based DGA domain detection
API endpoint: http://localhost:8001
"""

import math
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify
from prometheus_client import generate_latest, REGISTRY
import logging

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.redis_client import redis_client
from src.utils.metrics import (
    record_dns_query, record_block_decision, record_cache_hit, record_cache_miss,
    record_latency, record_dga_score, record_error
)

# Setup
logger = get_logger(__name__, "dga_detector.log")
app = Flask(__name__)

class DGADetector:
    """Detects Domain Generation Algorithm (DGA) domains"""
    
    def __init__(self):
        self.entropy_threshold = Config.DGA_ENTROPY_THRESHOLD
        self.length_threshold = Config.DGA_LENGTH_THRESHOLD
        self.cv_ratio_low = Config.DGA_CV_RATIO_LOW
        self.cv_ratio_high = Config.DGA_CV_RATIO_HIGH
        self.vowels = set('aeiouAEIOU')
        logger.info(f"DGADetector initialized - entropy_threshold={self.entropy_threshold}")
    
    def calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy"""
        if not text:
            return 0.0
        
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1
        
        entropy = 0.0
        length = len(text)
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        
        return entropy
    
    def consonant_vowel_ratio(self, text: str) -> float:
        """Calculate consonant/vowel ratio"""
        consonants = sum(1 for c in text.lower() if c in 'bcdfghjklmnpqrstvwxyz')
        vowels = sum(1 for c in text.lower() if c in self.vowels)
        return consonants / (vowels + 1)
    
    def check_suspicious_patterns(self, text: str) -> bool:
        """Check for linguistic anomalies"""
        # No vowels
        if not any(c.lower() in self.vowels for c in text):
            return True
        
        # Long digit sequences
        max_digit_seq = 0
        current_seq = 0
        for c in text:
            if c.isdigit():
                current_seq += 1
                max_digit_seq = max(max_digit_seq, current_seq)
            else:
                current_seq = 0
        
        if max_digit_seq > 4:
            return True
        
        # Character repetition
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2]:
                return True
        
        return False
    
    def analyze(self, domain: str) -> dict:
        """Analyze domain for DGA characteristics"""
        try:
            # Extract domain name (before TLD)
            domain_name = domain.split('.')[0]
            
            # Calculate metrics
            entropy = self.calculate_entropy(domain_name)
            cv_ratio = self.consonant_vowel_ratio(domain_name)
            length = len(domain_name)
            suspicious = self.check_suspicious_patterns(domain_name)
            
            # Composite score (0-1)
            score = 0.0
            
            # Entropy contribution (40%)
            if entropy > self.entropy_threshold:
                score += min(entropy / self.entropy_threshold, 1.0) * 0.4
            
            # C/V ratio contribution (30%)
            if cv_ratio > self.cv_ratio_high or cv_ratio < self.cv_ratio_low:
                score += 0.3
            
            # Length contribution (20%)
            if length > self.length_threshold:
                score += 0.2
            
            # Suspicious patterns (10%)
            if suspicious:
                score += 0.1
            
            score = min(score, 1.0)
            is_dga = score > Config.DGA_SCORE_THRESHOLD
            
            result = {
                'domain': domain,
                'is_dga': is_dga,
                'score': round(score, 4),
                'metrics': {
                    'entropy': round(entropy, 4),
                    'entropy_threshold': self.entropy_threshold,
                    'cv_ratio': round(cv_ratio, 4),
                    'length': length,
                    'suspicious_patterns': suspicious
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Record metrics
            record_dga_score(score)
            if is_dga:
                record_block_decision('dga', 'dga_pattern')
            
            logger.info(f"Analyzed: {domain} → DGA={is_dga} (score={score:.3f})")
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing domain: {e}")
            record_error('dga', 'analysis_error')
            raise

detector = DGADetector()

# =============================================
# ENDPOINTS
# =============================================

@app.route('/', methods=['GET'])
def root():
    """Service information and available endpoints"""
    return jsonify({
        'service': 'DGA Detector',
        'version': '1.0.0',
        'description': 'Heuristic-based DGA domain detection service',
        'port': 8001,
        'endpoints': {
            'POST /analyze': 'Analyze single domain for DGA characteristics',
            'POST /batch': 'Batch analyze multiple domains (max 100)',
            'GET /health': 'Health check',
            'GET /metrics': 'Prometheus metrics'
        },
        'example_request': {
            'endpoint': 'POST /analyze',
            'body': {'domain': 'example.com'}
        },
        'documentation': 'https://github.com/rachk02/dns_shield'
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze_domain():
    """Analyze single domain"""
    start_time = time.time()
    try:
        data = request.json
        domain = data.get('domain', '').lower().strip() if data else ''
        
        if not domain:
            return jsonify({'error': 'Domain required'}), 400
        
        record_dns_query('dga')
        
        # Check cache
        cached = redis_client.get(f"dga:{domain}")
        if cached:
            record_cache_hit('dga')
            duration_ms = (time.time() - start_time) * 1000
            record_latency('POST', '/analyze', 'dga', duration_ms)
            return jsonify(cached)
        
        record_cache_miss('dga')
        
        # Analyze
        result = detector.analyze(domain)
        
        # Cache result (24h TTL)
        redis_client.set(f"dga:{domain}", result, Config.REDIS_CACHE_TTL)
        
        duration_ms = (time.time() - start_time) * 1000
        record_latency('POST', '/analyze', 'dga', duration_ms)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        record_error('dga', 'endpoint_error')
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_analyze():
    """Analyze batch of domains"""
    start_time = time.time()
    try:
        data = request.json
        domains = data.get('domains', [])[:1000] if data else []
        
        if not domains:
            return jsonify({'error': 'Domains list required'}), 400
        
        results = []
        for domain in domains:
            result = detector.analyze(domain.lower())
            results.append(result)
            redis_client.set(f"dga:{domain}", result, Config.REDIS_CACHE_TTL)
        
        duration_ms = (time.time() - start_time) * 1000
        record_latency('POST', '/batch', 'dga', duration_ms)
        
        return jsonify({'count': len(results), 'results': results})
    
    except Exception as e:
        logger.error(f"Batch error: {e}")
        record_error('dga', 'batch_error')
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    try:
        redis_status = 'ok' if redis_client._redis.ping() else 'down'
    except:
        redis_status = 'down'
    
    return jsonify({
        'status': 'healthy',
        'service': 'dga_detector',
        'timestamp': datetime.now().isoformat(),
        'redis': redis_status,
        'version': '1.0.0'
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain; charset=utf-8'}

# =============================================
# MAIN
# =============================================

if __name__ == '__main__':
    logger.info("Starting DGA Detector Service on port 8001...")
    app.run(
        host=Config.API_HOST,
        port=8001,
        debug=Config.FLASK_DEBUG,
        threaded=True
    )
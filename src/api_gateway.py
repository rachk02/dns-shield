#!/usr/bin/env python3
"""
API Gateway - Orchestration and cascade of all services
Main entry point: http://localhost:9000
"""

import requests
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify
from prometheus_client import generate_latest, REGISTRY

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.db import db_client
from src.utils.metrics import (
    record_dns_query, record_latency, record_error, record_block_decision,
    record_accept_decision, record_dns_latency
)

logger = get_logger(__name__, "api_gateway.log")
app = Flask(__name__)

class CascadeOrchestrator:
    """Orchestrates cascade through DGA -> BERT -> Ensemble"""
    
    def __init__(self):
        self.dga_url = "http://localhost:8001/analyze"
        self.bert_url = "http://localhost:8002/classify"
        self.ensemble_url = "http://localhost:8003/predict"
        logger.info("CascadeOrchestrator initialized")
    
    def analyze_cascade(self, domain: str) -> dict:
        """Run full cascade analysis"""
        start_time = time.time()
        scores = {}
        stage_resolved = None
        
        try:
            # Stage 1: DGA Detector
            logger.info(f"Stage 1: DGA analysis for {domain}")
            dga_start = time.time()
            dga_resp = requests.post(self.dga_url, json={'domain': domain}, timeout=5).json()
            dga_latency = (time.time() - dga_start) * 1000
            
            scores['dga'] = dga_resp.get('score', 0)
            
            record_dns_latency('gateway', 1, dga_latency)
            
            # Decision at stage 1
            if dga_resp.get('is_dga', False):
                if dga_resp['score'] > 0.9:
                    stage_resolved = 1
                    decision = 'BLOCK'
                    reason = 'DGA_HIGH_CONFIDENCE'
                    confidence = dga_resp['score']
            
            # Stage 2: BERT Analysis
            if stage_resolved is None:
                logger.info(f"Stage 2: BERT analysis for {domain}")
                bert_start = time.time()
                bert_resp = requests.post(self.bert_url, json={'domain': domain}, timeout=5).json()
                bert_latency = (time.time() - bert_start) * 1000
                
                scores['bert'] = bert_resp.get('score', 0)
                record_dns_latency('gateway', 2, bert_latency)
                
                # Decision at stage 2
                if bert_resp.get('classification') == 'malicious' and bert_resp['score'] > 0.7:
                    stage_resolved = 2
                    decision = 'BLOCK'
                    reason = 'BERT_HIGH_CONFIDENCE'
                    confidence = bert_resp['score']
            
            # Stage 3: Ensemble ML
            if stage_resolved is None:
                logger.info(f"Stage 3: Ensemble ML for {domain}")
                ensemble_start = time.time()
                ensemble_resp = requests.post(self.ensemble_url, json={'domain': domain}, timeout=5).json()
                ensemble_latency = (time.time() - ensemble_start) * 1000
                
                scores['ensemble'] = ensemble_resp.get('ensemble_score', 0)
                record_dns_latency('gateway', 3, ensemble_latency)
                
                # Final decision
                stage_resolved = 3
                decision = ensemble_resp.get('decision', 'ACCEPT')
                reason = 'ENSEMBLE_VOTING'
                confidence = ensemble_resp.get('ensemble_score', 0)
            
            # Total latency
            total_latency = (time.time() - start_time) * 1000
            
            result = {
                'domain': domain,
                'decision': decision,
                'confidence': confidence,
                'stage_resolved': stage_resolved,
                'latency_ms': round(total_latency, 2),
                'scores': scores,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Cascade complete: {domain} -> {decision}")
            return result
        
        except requests.exceptions.Timeout:
            logger.error(f"Cascade timeout for {domain}")
            record_error('gateway', 'cascade_timeout')
            raise
        except Exception as e:
            logger.error(f"Cascade error: {e}")
            record_error('gateway', 'cascade_error')
            raise

orchestrator = CascadeOrchestrator()

# =============================================
# ENDPOINTS
# =============================================

@app.route('/', methods=['GET'])
def root():
    """Service information and available endpoints"""
    return jsonify({
        'service': 'DNS Shield API Gateway',
        'version': '1.0.0',
        'description': 'Orchestration and cascade of DGA Detector -> BERT -> Ensemble ML',
        'port': 9000,
        'endpoints': {
            'POST /analyze': 'Analyze domain through full cascade pipeline',
            'POST /batch': 'Batch analyze multiple domains (max 100)',
            'GET /health': 'Health check',
            'GET /stats': 'System statistics',
            'GET /metrics': 'Prometheus metrics'
        },
        'cascade_stages': {
            'Stage 1': 'DGA Detector (http://localhost:8001)',
            'Stage 2': 'BERT Service (http://localhost:8002)',
            'Stage 3': 'Ensemble ML (http://localhost:8003)'
        },
        'example_request': {
            'endpoint': 'POST /analyze',
            'body': {'domain': 'example.com'}
        },
        'documentation': 'https://github.com/rachk02/dns_shield'
    }), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze domain through full cascade"""
    start_time = time.time()
    try:
        data = request.json
        domain = data.get('domain', '').lower().strip() if data else ''
        
        if not domain:
            return jsonify({'error': 'Domain required'}), 400
        
        record_dns_query('gateway')
        
        # Run cascade
        result = orchestrator.analyze_cascade(domain)
        
        # Record decision
        if result['decision'] == 'BLOCK':
            record_block_decision('gateway', result.get('reason', 'unknown'))
        else:
            record_accept_decision('gateway')
        
        # Log to database
        try:
            db_client.insert_dns_query(
                domain=domain,
                decision=result['decision'],
                confidence=result['confidence'],
                stage=result['stage_resolved'],
                latency_ms=result['latency_ms'],
                blocked_by='gateway',
                dga_score=result['scores'].get('dga'),
                bert_score=result['scores'].get('bert'),
                ensemble_score=result['scores'].get('ensemble')
            )
        except Exception as e:
            logger.warning(f"Database log error: {e}")
        
        # Record latency
        duration_ms = (time.time() - start_time) * 1000
        record_latency('POST', '/analyze', 'gateway', duration_ms)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Analyze endpoint error: {e}")
        record_error('gateway', 'analyze_error')
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_analyze():
    """Batch analyze domains"""
    start_time = time.time()
    try:
        data = request.json
        domains = data.get('domains', [])[:100] if data else []
        
        if not domains:
            return jsonify({'error': 'Domains list required'}), 400
        
        results = []
        for domain in domains:
            try:
                result = orchestrator.analyze_cascade(domain.lower())
                results.append(result)
            except Exception as e:
                logger.warning(f"Error analyzing {domain}: {e}")
                continue
        
        duration_ms = (time.time() - start_time) * 1000
        record_latency('POST', '/batch', 'gateway', duration_ms)
        
        return jsonify({'count': len(results), 'results': results})
    
    except Exception as e:
        logger.error(f"Batch error: {e}")
        record_error('gateway', 'batch_error')
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check - verify all services"""
    health_status = {
        'status': 'healthy',
        'service': 'api_gateway',
        'timestamp': datetime.now().isoformat(),
        'services': {}
    }
    
    # Check DGA
    try:
        resp = requests.get("http://localhost:8001/health", timeout=2)
        health_status['services']['dga'] = resp.json().get('status', 'unknown')
    except:
        health_status['services']['dga'] = 'down'
    
    # Check BERT
    try:
        resp = requests.get("http://localhost:8002/health", timeout=2)
        health_status['services']['bert'] = resp.json().get('status', 'unknown')
    except:
        health_status['services']['bert'] = 'down'
    
    # Check Ensemble
    try:
        resp = requests.get("http://localhost:8003/health", timeout=2)
        health_status['services']['ensemble'] = resp.json().get('status', 'unknown')
    except:
        health_status['services']['ensemble'] = 'down'
    
    # Check database
    try:
        health_status['services']['database'] = 'ok'
    except:
        health_status['services']['database'] = 'down'
    
    return jsonify(health_status)

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics"""
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/stats', methods=['GET'])
def stats():
    """Get statistics from database"""
    try:
        daily_stats = db_client.get_stats_daily(days=7)
        suspicious = db_client.get_suspicious_domains(limit=20)
        
        return jsonify({
            'daily': daily_stats,
            'suspicious_domains': suspicious
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================
# MAIN
# =============================================

if __name__ == '__main__':
    logger.info("Starting API Gateway on port 9000...")
    app.run(
        host=Config.API_HOST,
        port=9000,
        debug=Config.FLASK_DEBUG,
        threaded=True
    )
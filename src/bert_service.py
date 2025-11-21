#!/usr/bin/env python3
"""
BERT Service - Contextual domain classification using BERT
API endpoint: http://localhost:8002
"""

import hashlib
import json
import time
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify
from prometheus_client import generate_latest, REGISTRY

try:  # Optional heavy dependencies
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer, AutoModel  # type: ignore
except ImportError:
    AutoTokenizer = AutoModel = None

try:  # pragma: no cover - optional dependency
    from sklearn.decomposition import PCA  # type: ignore
except ImportError:
    PCA = None

from src.utils.config import Config
from src.utils.logger import get_logger
from src.utils.redis_client import redis_client
from src.utils.metrics import (
    record_latency,
    record_bert_score,
    record_cache_hit,
    record_cache_miss,
    record_error,
    record_dns_query,
)

logger = get_logger(__name__, "bert_service.log")
app = Flask(__name__)

# Device configuration
if torch is not None:
    device = torch.device(Config.BERT_DEVICE if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'
logger.info(f"Using device: {device}")


class BERTAnalyzer:
    """BERT-based domain classification"""

    def __init__(self):
        self.use_fallback = not (torch and AutoTokenizer and AutoModel)
        self.device = device
        self.pca = None

        if not self.use_fallback:
            logger.info(f"Loading BERT model: {Config.BERT_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL)
            self.model = AutoModel.from_pretrained(Config.BERT_MODEL).to(self.device)
            self.model.eval()

            if PCA is not None and Config.BERT_PCA_DIM:
                self.pca = PCA(n_components=min(Config.BERT_PCA_DIM, self.model.config.hidden_size))
            logger.info(f"BERT model loaded on {self.device}")
        else:
            self.tokenizer = None
            self.model = None
            logger.warning("Transformers or PyTorch unavailable. Using deterministic fallback implementation.")

    def get_embedding(self, domain: str) -> np.ndarray:
        """Get BERT embedding for domain"""
        try:
            if not self.use_fallback:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        domain,
                        return_tensors='pt',
                        truncation=True,
                        max_length=Config.BERT_MAX_LENGTH,
                        padding=True,
                    ).to(self.device)

                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                    if self.pca and hasattr(self.pca, 'components_') and embedding.shape[1] > Config.BERT_PCA_DIM:
                        embedding = self.pca.transform(embedding)

                    return embedding[0].tolist()

            return self._fallback_embedding(domain)

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def classify(self, domain: str) -> dict:
        """Classify domain as legitimate or malicious"""
        try:
            embedding = self.get_embedding(domain)

            # Simple heuristic classification based on embedding statistics
            # In production, use trained classifier head
            entropy = float(np.std(embedding))

            # Score based on entropy
            score = min(entropy / 0.5, 1.0)  # Normalize

            result = {
                'domain': domain,
                'embedding_dim': len(embedding),
                'entropy': entropy,
                'score': round(score, 4),
                'classification': 'malicious' if score > 0.65 else 'legitimate',
                'timestamp': datetime.now().isoformat()
            }
            
            record_bert_score(score)
            return result

        except Exception as e:
            logger.error(f"Classification error: {e}")
            record_error('bert', 'classification_error')
            raise

    @staticmethod
    def _fallback_embedding(domain: str) -> list:
        """Generate deterministic embedding when BERT unavailable."""
        normalized = domain.lower().strip() or 'unknown'
        digest = hashlib.sha256(normalized.encode('utf-8')).digest()
        vector = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
        return vector.tolist()

analyzer = BERTAnalyzer()

# =============================================
# ENDPOINTS
# =============================================

@app.route('/', methods=['GET'])
def root():
    """Service information and available endpoints"""
    return jsonify({
        'service': 'BERT Service',
        'version': '1.0.0',
        'description': 'Contextual domain classification using BERT embeddings',
        'port': 8002,
        'device': str(device),
        'endpoints': {
            'POST /embed': 'Get BERT embedding for domain',
            'POST /classify': 'Classify domain as benign or malicious',
            'POST /batch': 'Batch classify multiple domains (max 100)',
            'GET /health': 'Health check',
            'GET /metrics': 'Prometheus metrics'
        },
        'example_request': {
            'endpoint': 'POST /classify',
            'body': {'domain': 'example.com'}
        },
        'documentation': 'https://github.com/rachk02/dns_shield'
    }), 200

@app.route('/embed', methods=['POST'])
def get_embedding():
    """Get BERT embedding for domain"""
    start_time = time.time()
    try:
        data = request.json
        domain = data.get('domain', '').lower().strip() if data else ''
        
        if not domain:
            return jsonify({'error': 'Domain required'}), 400
        
        record_dns_query('bert')
        
        # Check cache
        cached = redis_client.get(f"bert_embed:{domain}")
        if cached:
            record_cache_hit('bert')
            duration_ms = (time.time() - start_time) * 1000
            record_latency('POST', '/embed', 'bert', duration_ms)
            return jsonify(cached)
        
        record_cache_miss('bert')
        
        # Get embedding
        embedding = analyzer.get_embedding(domain)
        
        result = {
            'domain': domain,
            'embedding': embedding,
            'embedding_dim': len(embedding),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache embedding (24h TTL)
        redis_client.set(f"bert_embed:{domain}", result, Config.REDIS_CACHE_TTL)
        
        duration_ms = (time.time() - start_time) * 1000
        record_latency('POST', '/embed', 'bert', duration_ms)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Embed endpoint error: {e}")
        record_error('bert', 'embed_error')
        return jsonify({'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify_domain():
    """Classify domain"""
    start_time = time.time()
    try:
        data = request.json
        domain = data.get('domain', '').lower().strip() if data else ''
        
        if not domain:
            return jsonify({'error': 'Domain required'}), 400
        
        record_dns_query('bert')
        
        # Check cache
        cached = redis_client.get(f"bert_class:{domain}")
        if cached:
            record_cache_hit('bert')
            duration_ms = (time.time() - start_time) * 1000
            record_latency('POST', '/classify', 'bert', duration_ms)
            return jsonify(cached)
        
        record_cache_miss('bert')
        
        # Classify
        result = analyzer.classify(domain)
        
        # Cache result (24h TTL)
        redis_client.set(f"bert_class:{domain}", result, Config.REDIS_CACHE_TTL)
        
        duration_ms = (time.time() - start_time) * 1000
        record_latency('POST', '/classify', 'bert', duration_ms)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Classify endpoint error: {e}")
        record_error('bert', 'classify_error')
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_embed():
    """Batch embed domains"""
    start_time = time.time()
    try:
        data = request.json
        domains = data.get('domains', [])[:32] if data else []  # Max batch 32
        
        if not domains:
            return jsonify({'error': 'Domains list required'}), 400
        
        results = []
        for domain in domains:
            try:
                embedding = analyzer.get_embedding(domain.lower())
                result = {
                    'domain': domain,
                    'embedding': embedding,
                    'embedding_dim': len(embedding)
                }
                results.append(result)
                redis_client.set(f"bert_embed:{domain}", result, Config.REDIS_CACHE_TTL)
            except Exception as e:
                logger.warning(f"Error embedding {domain}: {e}")
                continue
        
        duration_ms = (time.time() - start_time) * 1000
        record_latency('POST', '/batch', 'bert', duration_ms)
        
        return jsonify({'count': len(results), 'results': results})
    
    except Exception as e:
        logger.error(f"Batch error: {e}")
        record_error('bert', 'batch_error')
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
        'service': 'bert_service',
        'device': str(device),
        'model': Config.BERT_MODEL,
        'timestamp': datetime.now().isoformat(),
        'redis': redis_status,
        'version': '1.0.0'
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics"""
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain; charset=utf-8'}

# =============================================
# MAIN
# =============================================

if __name__ == '__main__':
    logger.info("Starting BERT Service on port 8002...")
    app.run(
        host=Config.API_HOST,
        port=8002,
        debug=Config.FLASK_DEBUG,
        threaded=True
    )
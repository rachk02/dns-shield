#!/usr/bin/env python3
"""
Prometheus Metrics - Custom metrics for DNS Shield
"""

from prometheus_client import Counter, Histogram, Gauge
import time

# =============================================
# COUNTERS
# =============================================

# DNS Queries
dns_queries_total = Counter(
    'dns_queries_total',
    'Total DNS queries received',
    ['service']
)

# Decisions
dns_blocks_total = Counter(
    'dns_blocks_total',
    'Total blocked domains',
    ['service', 'reason']
)

dns_accepts_total = Counter(
    'dns_accepts_total',
    'Total accepted domains',
    ['service']
)

# Errors
dns_errors_total = Counter(
    'dns_errors_total',
    'Total errors in DNS processing',
    ['service', 'error_type']
)

# DGA Detections
dga_detections_total = Counter(
    'dga_detections_total',
    'Total DGA detections',
    ['severity']
)

# Cache
cache_hits_total = Counter(
    'cache_hits_total',
    'Total Redis cache hits',
    ['cache_type']
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total Redis cache misses',
    ['cache_type']
)

# =============================================
# HISTOGRAMS
# =============================================

# Latency
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint', 'service'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
)

dns_processing_duration_seconds = Histogram(
    'dns_processing_duration_seconds',
    'DNS query processing latency in seconds',
    ['service', 'stage'],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1)
)

# Scores
dga_score_distribution = Histogram(
    'dga_score_distribution',
    'DGA detection score distribution',
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

bert_score_distribution = Histogram(
    'bert_score_distribution',
    'BERT classification score distribution',
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# =============================================
# GAUGES
# =============================================

# Active connections
active_connections = Gauge(
    'active_connections',
    'Number of active connections',
    ['service']
)

# Cache statistics
cache_size_bytes = Gauge(
    'cache_size_bytes',
    'Redis cache size in bytes'
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio (0-1)'
)

# Database
db_connection_pool_usage = Gauge(
    'db_connection_pool_usage',
    'Database connection pool usage ratio (0-1)'
)

# Model performance
bert_model_accuracy = Gauge(
    'bert_model_accuracy',
    'BERT model accuracy (0-1)'
)

lstm_model_accuracy = Gauge(
    'lstm_model_accuracy',
    'LSTM model accuracy (0-1)'
)

gru_model_accuracy = Gauge(
    'gru_model_accuracy',
    'GRU model accuracy (0-1)'
)

rf_model_accuracy = Gauge(
    'rf_model_accuracy',
    'Random Forest model accuracy (0-1)'
)

# Request queue
request_queue_size = Gauge(
    'request_queue_size',
    'Current request queue size'
)

# =============================================
# METRIC RECORDING FUNCTIONS
# =============================================

def record_dns_query(service: str):
    """Record DNS query received"""
    dns_queries_total.labels(service=service).inc()

def record_block_decision(service: str, reason: str):
    """Record block decision"""
    dns_blocks_total.labels(service=service, reason=reason).inc()

def record_accept_decision(service: str):
    """Record accept decision"""
    dns_accepts_total.labels(service=service).inc()

def record_error(service: str, error_type: str):
    """Record error"""
    dns_errors_total.labels(service=service, error_type=error_type).inc()

def record_dga_detection(severity: str):
    """Record DGA detection"""
    dga_detections_total.labels(severity=severity).inc()

def record_cache_hit(cache_type: str):
    """Record cache hit"""
    cache_hits_total.labels(cache_type=cache_type).inc()

def record_cache_miss(cache_type: str):
    """Record cache miss"""
    cache_misses_total.labels(cache_type=cache_type).inc()

def record_latency(method: str, endpoint: str, service: str, duration_ms: float):
    """Record HTTP request latency"""
    http_request_duration_seconds.labels(
        method=method, 
        endpoint=endpoint, 
        service=service
    ).observe(duration_ms / 1000.0)

def record_dns_latency(service: str, stage: int, duration_ms: float):
    """Record DNS processing latency"""
    dns_processing_duration_seconds.labels(
        service=service,
        stage=str(stage)
    ).observe(duration_ms / 1000.0)

def record_dga_score(score: float):
    """Record DGA score"""
    dga_score_distribution.observe(score)

def record_bert_score(score: float):
    """Record BERT score"""
    bert_score_distribution.observe(score)

def update_cache_stats(hit_ratio: float, size_bytes: int):
    """Update cache statistics"""
    cache_hit_ratio.set(hit_ratio)
    cache_size_bytes.set(size_bytes)

def update_model_accuracy(model_type: str, accuracy: float):
    """Update model accuracy"""
    if model_type == 'bert':
        bert_model_accuracy.set(accuracy)
    elif model_type == 'lstm':
        lstm_model_accuracy.set(accuracy)
    elif model_type == 'gru':
        gru_model_accuracy.set(accuracy)
    elif model_type == 'rf':
        rf_model_accuracy.set(accuracy)
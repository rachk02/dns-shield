#!/usr/bin/env python3
"""
Configuration centralisée pour DNS Shield
Charge variables d'environnement et fournit accès config
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional


def torch_available() -> bool:
    """Vérifier si PyTorch + CUDA disponible"""
    try:
        import torch  # noqa: WPS433 - import local to avoid heavy dependency at module load
        return torch.cuda.is_available()
    except ImportError:
        return False


# Charger .env
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)


class Config:
    """Configuration de base"""
    
    # =============================================
    # FLASK CONFIGURATION
    # =============================================
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    
    # =============================================
    # DATABASE CONFIGURATION (PostgreSQL Local)
    # =============================================
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
    POSTGRES_DB = os.getenv('POSTGRES_DB', '')
    POSTGRES_USER = os.getenv('POSTGRES_USER', '')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')
    
    # =============================================
    # REDIS CONFIGURATION (Docker)
    # =============================================
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
    REDIS_CACHE_TTL = int(os.getenv('REDIS_CACHE_TTL', 86400))  # 24h
    
    # =============================================
    # BERT MODEL CONFIGURATION
    # =============================================
    BERT_MODEL = os.getenv('BERT_MODEL', 'bert-base-multilingual-cased')
    BERT_MAX_LENGTH = int(os.getenv('BERT_MAX_LENGTH', 128))
    BERT_DEVICE = os.getenv('BERT_DEVICE', 'cuda:0' if torch_available() else 'cpu')
    BERT_BATCH_SIZE = int(os.getenv('BERT_BATCH_SIZE', 32))
    BERT_PCA_DIM = int(os.getenv('BERT_PCA_DIM', 256))
    
    # =============================================
    # DGA DETECTOR CONFIGURATION
    # =============================================
    DGA_ENTROPY_THRESHOLD = float(os.getenv('DGA_ENTROPY_THRESHOLD', 4.0))
    DGA_LENGTH_THRESHOLD = int(os.getenv('DGA_LENGTH_THRESHOLD', 25))
    DGA_CV_RATIO_LOW = float(os.getenv('DGA_CV_RATIO_LOW', 0.4))
    DGA_CV_RATIO_HIGH = float(os.getenv('DGA_CV_RATIO_HIGH', 2.5))
    DGA_SCORE_THRESHOLD = float(os.getenv('DGA_SCORE_THRESHOLD', 0.65))
    
    # =============================================
    # ENSEMBLE ML CONFIGURATION
    # =============================================
    ENSEMBLE_THRESHOLD = float(os.getenv('ENSEMBLE_THRESHOLD', 0.65))
    LSTM_MODEL_PATH = os.getenv('LSTM_MODEL_PATH', 'models/lstm/lstm_model.h5')
    GRU_MODEL_PATH = os.getenv('GRU_MODEL_PATH', 'models/gru/gru_model.h5')
    RF_MODEL_PATH = os.getenv('RF_MODEL_PATH', 'models/rf/rf_model.pkl')
    
    # Weights for voting
    LSTM_WEIGHT = float(os.getenv('LSTM_WEIGHT', 0.33))
    GRU_WEIGHT = float(os.getenv('GRU_WEIGHT', 0.33))
    RF_WEIGHT = float(os.getenv('RF_WEIGHT', 0.34))
    
    # =============================================
    # LOGGING CONFIGURATION
    # =============================================
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    LOG_FORMAT = os.getenv('LOG_FORMAT', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', 10485760))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 5))
    
    # =============================================
    # API CONFIGURATION
    # =============================================
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 9000))
    API_WORKERS = int(os.getenv('API_WORKERS', 4))
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', 30))
    MAX_REQUESTS_PER_SECOND = int(os.getenv('MAX_REQUESTS_PER_SECOND', 1000))
    
    # =============================================
    # MONITORING CONFIGURATION
    # =============================================
    PROMETHEUS_ENABLED = os.getenv('PROMETHEUS_ENABLED', 'True').lower() == 'true'
    PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', 9100))
    METRICS_NAMESPACE = os.getenv('METRICS_NAMESPACE', 'dns_shield')
    
    # =============================================
    # SECURITY CONFIGURATION
    # =============================================
    ENABLE_RATE_LIMITING = os.getenv('ENABLE_RATE_LIMITING', 'True').lower() == 'true'
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 1000))
    RATE_LIMIT_PERIOD = int(os.getenv('RATE_LIMIT_PERIOD', 60))  # seconds
    
    @classmethod
    def get_db_url(cls) -> str:
        """Construire URL connexion PostgreSQL"""
        return (
            f"postgresql://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}"
            f"@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        )
    
    @classmethod
    def get_redis_url(cls) -> str:
        """Construire URL Redis"""
        auth = f":{cls.REDIS_PASSWORD}@" if cls.REDIS_PASSWORD else ""
        return f"redis://{auth}{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
    
    @classmethod
    def get_redis_connection_params(cls) -> dict:
        """Obtenir paramètres connexion Redis"""
        return {
            'host': cls.REDIS_HOST,
            'port': cls.REDIS_PORT,
            'db': cls.REDIS_DB,
            'password': cls.REDIS_PASSWORD if cls.REDIS_PASSWORD else None,
            'decode_responses': True,
            'socket_timeout': 5,
            'socket_connect_timeout': 5
        }
    
    @classmethod
    def get_postgres_connection_params(cls) -> dict:
        """Obtenir paramètres connexion PostgreSQL"""
        return {
            'host': cls.POSTGRES_HOST,
            'port': cls.POSTGRES_PORT,
            'database': cls.POSTGRES_DB,
            'user': cls.POSTGRES_USER,
            'password': cls.POSTGRES_PASSWORD,
            'connect_timeout': 5
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Valider configuration"""
        try:
            # Test Redis URL
            redis_url = cls.get_redis_url()
            assert redis_url.startswith('redis://'), "Invalid Redis URL"
            
            # Test DB URL
            db_url = cls.get_db_url()
            assert db_url.startswith('postgresql://'), "Invalid DB URL"
            
            # Validate model paths
            total_weight = cls.LSTM_WEIGHT + cls.GRU_WEIGHT + cls.RF_WEIGHT
            assert abs(total_weight - 1.0) < 1e-6, "Model weights must sum to 1.0"
            
            # Validate thresholds
            assert 0 <= cls.DGA_SCORE_THRESHOLD <= 1, "DGA threshold out of range"
            assert 0 <= cls.ENSEMBLE_THRESHOLD <= 1, "Ensemble threshold out of range"
            
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

# Export configuration
config = Config()
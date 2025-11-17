#!/usr/bin/env python3
"""
Database Client - PostgreSQL connection and utilities
"""

import psycopg2
from psycopg2 import pool, sql
from typing import List, Dict, Any, Optional
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__, "db.log")

class DatabaseClient:
    """PostgreSQL database client with connection pooling"""
    
    _instance = None
    _connection_pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._init_pool()
    
    def _init_pool(self):
        """Initialize connection pool"""
        try:
            self._connection_pool = psycopg2.pool.SimpleConnectionPool(
                2, 10,
                host=Config.POSTGRES_HOST,
                port=Config.POSTGRES_PORT,
                database=Config.POSTGRES_DB,
                user=Config.POSTGRES_USER,
                password=Config.POSTGRES_PASSWORD,
                connect_timeout=5
            )
            logger.info(f"✓ Database connected: {Config.POSTGRES_HOST}:{Config.POSTGRES_PORT}/{Config.POSTGRES_DB}")
        except psycopg2.Error as e:
            logger.error(f"✗ Database connection failed: {e}")
            raise
    
    @staticmethod
    def get_instance() -> 'DatabaseClient':
        """Get singleton instance"""
        if DatabaseClient._instance is None:
            DatabaseClient()
        return DatabaseClient._instance
    
    def get_connection(self):
        """Get connection from pool"""
        try:
            return self._connection_pool.getconn()
        except psycopg2.OperationalError as e:
            logger.error(f"Database pool error: {e}")
            raise
    
    def release_connection(self, conn):
        """Release connection back to pool"""
        if conn:
            self._connection_pool.putconn(conn)
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute SELECT query"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Fetch all results
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            return results
        except psycopg2.Error as e:
            logger.error(f"Query execution error: {e}")
            return []
        finally:
            if conn:
                self.release_connection(conn)
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE query"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            affected_rows = cursor.rowcount
            conn.commit()
            cursor.close()
            return affected_rows
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Update execution error: {e}")
            return 0
        finally:
            if conn:
                self.release_connection(conn)
    
    def insert_dns_query(self, domain: str, decision: str, confidence: float, 
                         stage: int, latency_ms: float, blocked_by: str = None,
                         dga_score: float = None, bert_score: float = None,
                         ensemble_score: float = None) -> bool:
        """Insert DNS query record"""
        try:
            query = """
                INSERT INTO dns_queries 
                (domain, decision, confidence, stage_resolved, latency_ms, 
                 blocked_by, dga_score, bert_score, ensemble_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (domain, decision, confidence, stage, latency_ms, 
                     blocked_by, dga_score, bert_score, ensemble_score)
            
            self.execute_update(query, params)
            return True
        except Exception as e:
            logger.error(f"Error inserting DNS query: {e}")
            return False
    
    def get_stats_daily(self, days: int = 7) -> List[Dict]:
        """Get daily statistics"""
        try:
            query = """
                SELECT * FROM stats_daily 
                WHERE date >= CURRENT_DATE - INTERVAL '%s days'
                ORDER BY date DESC
            """
            return self.execute_query(query, (days,))
        except Exception as e:
            logger.error(f"Error getting daily stats: {e}")
            return []
    
    def get_suspicious_domains(self, limit: int = 100) -> List[Dict]:
        """Get most suspicious domains"""
        try:
            query = """
                SELECT * FROM suspicious_domains 
                LIMIT %s
            """
            return self.execute_query(query, (limit,))
        except Exception as e:
            logger.error(f"Error getting suspicious domains: {e}")
            return []
    
    def check_whitelist(self, domain: str) -> bool:
        """Check if domain is whitelisted"""
        try:
            query = "SELECT 1 FROM whitelist WHERE domain = %s AND active = TRUE"
            result = self.execute_query(query, (domain,))
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error checking whitelist: {e}")
            return False
    
    def check_blacklist(self, domain: str) -> bool:
        """Check if domain is blacklisted"""
        try:
            query = "SELECT 1 FROM blacklist WHERE domain = %s AND active = TRUE"
            result = self.execute_query(query, (domain,))
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error checking blacklist: {e}")
            return False
    
    def close_pool(self):
        """Close all connections in pool"""
        if self._connection_pool:
            self._connection_pool.closeall()
            logger.info("Database connection pool closed")


# Convenience instance
db_client = DatabaseClient.get_instance()
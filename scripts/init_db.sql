-- =============================================
-- SCRIPT D'INITIALISATION DNS SHIELD
-- Database: dns_shield
-- User: dns_shield
-- Created: 2025-11-04
-- =============================================

-- =============================================
-- 1. TABLES PRINCIPALES
-- =============================================

-- Table audit trail (requêtes DNS)
CREATE TABLE IF NOT EXISTS dns_queries (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    domain VARCHAR(255) NOT NULL,
    source_ip INET,
    query_type VARCHAR(10),
    decision VARCHAR(20) NOT NULL CHECK (decision IN ('BLOCK', 'ACCEPT', 'QUARANTINE')),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    stage_resolved INT CHECK (stage_resolved IN (1, 2, 3)),
    latency_ms FLOAT,
    blocked_by VARCHAR(100),
    dga_score FLOAT,
    bert_score FLOAT,
    ensemble_score FLOAT,
    reason_text TEXT
);

-- Table whitelist (domaines approuvés)
CREATE TABLE IF NOT EXISTS whitelist (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(255) UNIQUE NOT NULL,
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    added_by VARCHAR(100),
    reason TEXT,
    active BOOLEAN DEFAULT TRUE
);

-- Table blacklist (domaines connus malveillants)
CREATE TABLE IF NOT EXISTS blacklist (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(255) UNIQUE NOT NULL,
    added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(100),
    confidence FLOAT DEFAULT 0.95,
    active BOOLEAN DEFAULT TRUE,
    tif_source VARCHAR(100)
);

-- Table metadata modèles ML
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('LSTM', 'GRU', 'RF', 'BERT')),
    version VARCHAR(20),
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    trained_date TIMESTAMP,
    deployed_date TIMESTAMP,
    active BOOLEAN DEFAULT TRUE,
    file_path VARCHAR(255),
    parameters JSONB
);

-- Table for system alerts
CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    alert_name VARCHAR(100) NOT NULL,
    severity VARCHAR(20),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    service VARCHAR(50),
    message TEXT,
    resolved BOOLEAN DEFAULT FALSE
);

-- =============================================
-- 2. INDEX POUR PERFORMANCE
-- =============================================

-- Index sur domain (recherches fréquentes)
CREATE INDEX IF NOT EXISTS idx_dns_queries_domain 
    ON dns_queries(domain);

-- Index sur timestamp (filtres date)
CREATE INDEX IF NOT EXISTS idx_dns_queries_timestamp 
    ON dns_queries(timestamp DESC);

-- Index composé (domain + decision)
CREATE INDEX IF NOT EXISTS idx_dns_queries_domain_decision 
    ON dns_queries(domain, decision);

-- Index decision (statistiques)
CREATE INDEX IF NOT EXISTS idx_dns_queries_decision 
    ON dns_queries(decision);

-- Whitelist/Blacklist domain index
CREATE INDEX IF NOT EXISTS idx_whitelist_domain 
    ON whitelist(domain);

CREATE INDEX IF NOT EXISTS idx_blacklist_domain 
    ON blacklist(domain);

-- Alerts index
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
    ON alerts(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_service 
    ON alerts(service);

-- =============================================
-- 3. VUES STATISTIQUES
-- =============================================

-- Vue statistiques journalières
CREATE OR REPLACE VIEW stats_daily AS
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_queries,
    SUM(CASE WHEN decision = 'BLOCK' THEN 1 ELSE 0 END) as blocked,
    SUM(CASE WHEN decision = 'ACCEPT' THEN 1 ELSE 0 END) as accepted,
    SUM(CASE WHEN decision = 'QUARANTINE' THEN 1 ELSE 0 END) as quarantined,
    AVG(latency_ms) as avg_latency_ms,
    MAX(latency_ms) as max_latency_ms,
    AVG(confidence) as avg_confidence,
    COUNT(DISTINCT source_ip) as unique_sources
FROM dns_queries
WHERE timestamp >= CURRENT_DATE
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Vue statistiques par service
CREATE OR REPLACE VIEW stats_by_service AS
SELECT 
    blocked_by as service,
    COUNT(*) as total_decisions,
    SUM(CASE WHEN decision = 'BLOCK' THEN 1 ELSE 0 END) as blocks,
    SUM(CASE WHEN decision = 'ACCEPT' THEN 1 ELSE 0 END) as accepts,
    AVG(latency_ms) as avg_latency,
    AVG(confidence) as avg_confidence
FROM dns_queries
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY blocked_by;

-- Vue domaines suspects
CREATE OR REPLACE VIEW suspicious_domains AS
SELECT 
    domain,
    COUNT(*) as query_count,
    SUM(CASE WHEN decision = 'BLOCK' THEN 1 ELSE 0 END) as block_count,
    MAX(dga_score) as max_dga_score,
    MAX(bert_score) as max_bert_score,
    MAX(ensemble_score) as max_ensemble_score,
    MAX(timestamp) as last_seen
FROM dns_queries
WHERE decision = 'BLOCK'
GROUP BY domain
ORDER BY query_count DESC
LIMIT 1000;

-- =============================================
-- 4. DONNÉES TEST
-- =============================================

-- Insérer whitelist initiale
INSERT INTO whitelist (domain, added_by, reason) VALUES
    ('google.com', 'system', 'Known legitimate - Google'),
    ('microsoft.com', 'system', 'Known legitimate - Microsoft'),
    ('amazon.com', 'system', 'Known legitimate - Amazon'),
    ('github.com', 'system', 'Known legitimate - GitHub'),
    ('stackoverflow.com', 'system', 'Known legitimate - Stack Overflow'),
    ('mozilla.org', 'system', 'Known legitimate - Mozilla'),
    ('cloudflare.com', 'system', 'Known legitimate - Cloudflare'),
    ('fastly.com', 'system', 'Known legitimate - Fastly')
ON CONFLICT (domain) DO NOTHING;

-- Insérer blacklist initiale
INSERT INTO blacklist (domain, source, confidence, tif_source) VALUES
    ('malicious-dga-1.com', 'TIF', 0.99, 'abuse.ch'),
    ('phishing-site.net', 'TIF', 0.95, 'urlhaus'),
    ('c2-command.info', 'TIF', 0.98, 'misp'),
    ('ransomware-payment.com', 'TIF', 0.97, 'abuse.ch')
ON CONFLICT (domain) DO NOTHING;

-- =============================================
-- 5. GRANT PERMISSIONS
-- =============================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dns_shield;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dns_shield;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO dns_shield;

-- =============================================
-- 6. VERIFICATION & STATUS
-- =============================================

-- Afficher statut initialisation
SELECT 'DNS Shield database initialized successfully!' as status;

-- Lister tables créées
SELECT 
    tablename,
    schemaname
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY tablename;

-- Vérifier counts
SELECT 
    'whitelist' as table_name, COUNT(*) as record_count FROM whitelist
UNION ALL
SELECT 'blacklist', COUNT(*) FROM blacklist
UNION ALL
SELECT 'dns_queries', COUNT(*) FROM dns_queries
UNION ALL
SELECT 'models', COUNT(*) FROM models;
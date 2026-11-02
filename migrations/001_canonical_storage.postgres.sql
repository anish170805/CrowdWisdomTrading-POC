-- Canonical storage schema, PostgreSQL variant (schema version 1).
-- The application creates this schema with SQLAlchemy; this file documents the
-- equivalent DDL for migrations and external tooling.

CREATE TABLE IF NOT EXISTS tweets (
    id BIGSERIAL PRIMARY KEY,
    post_id VARCHAR(512),
    post_fingerprint VARCHAR(64) NOT NULL,
    asset_symbol VARCHAR(64),
    source VARCHAR(128) NOT NULL,
    source_format VARCHAR(32),
    source_file TEXT,
    source_row INTEGER,
    source_record_id VARCHAR(512),
    batch_id VARCHAR(128) NOT NULL,
    raw_payload_ref TEXT,
    ingested_at TIMESTAMPTZ NOT NULL,
    text TEXT NOT NULL,
    text_original TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    created_at_original TEXT NOT NULL,
    source_timezone_offset VARCHAR(16),
    timezone_assumed BOOLEAN NOT NULL DEFAULT FALSE,
    author TEXT,
    author_handle VARCHAR(255),
    likes BIGINT,
    reposts BIGINT,
    replies BIGINT,
    quotes BIGINT,
    views BIGINT,
    url TEXT,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    CONSTRAINT uq_tweets_post_id UNIQUE (post_id),
    CONSTRAINT uq_tweets_post_fingerprint UNIQUE (post_fingerprint)
);

CREATE INDEX IF NOT EXISTS ix_tweets_created_at ON tweets (created_at);
CREATE INDEX IF NOT EXISTS ix_tweets_asset_created_at ON tweets (asset_symbol, created_at);
CREATE INDEX IF NOT EXISTS ix_tweets_batch_id ON tweets (batch_id);

CREATE TABLE IF NOT EXISTS price_bars (
    id BIGSERIAL PRIMARY KEY,
    asset_symbol VARCHAR(64) NOT NULL,
    asset_type VARCHAR(32),
    bar_date DATE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    adjusted_close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    provider VARCHAR(64) NOT NULL,
    batch_id VARCHAR(128) NOT NULL,
    source_record_id VARCHAR(512),
    raw_payload_ref TEXT,
    retrieved_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    CONSTRAINT uq_price_bars_asset_timestamp UNIQUE (asset_symbol, timestamp),
    CONSTRAINT uq_price_bars_asset_date UNIQUE (asset_symbol, bar_date)
);

CREATE INDEX IF NOT EXISTS ix_price_bars_timestamp ON price_bars (timestamp);
CREATE INDEX IF NOT EXISTS ix_price_bars_asset_date ON price_bars (asset_symbol, bar_date);
CREATE INDEX IF NOT EXISTS ix_price_bars_batch_id ON price_bars (batch_id);

CREATE TABLE IF NOT EXISTS quarantined_rows (
    id BIGSERIAL PRIMARY KEY,
    entity VARCHAR(32) NOT NULL,
    record_key VARCHAR(512) NOT NULL,
    reason_code VARCHAR(64) NOT NULL,
    reason TEXT NOT NULL,
    reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    source VARCHAR(128),
    source_record_id VARCHAR(512),
    batch_id VARCHAR(128) NOT NULL,
    quarantined_at TIMESTAMPTZ NOT NULL,
    raw_payload_ref TEXT,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    CONSTRAINT uq_quarantined_rows_identity
        UNIQUE (entity, batch_id, record_key, reason_code)
);

CREATE INDEX IF NOT EXISTS ix_quarantined_rows_entity ON quarantined_rows (entity);
CREATE INDEX IF NOT EXISTS ix_quarantined_rows_quarantined_at ON quarantined_rows (quarantined_at);
CREATE INDEX IF NOT EXISTS ix_quarantined_rows_batch_id ON quarantined_rows (batch_id);

-- ============================================================
-- Invoice Extraction System — Database Schema
-- Runs automatically on first PostgreSQL container boot via
-- docker-entrypoint-initdb.d volume mount.
-- ============================================================

-- ----------------------------------------------------------
-- 1. ENUM: Job Status State Machine
-- ----------------------------------------------------------
CREATE TYPE job_status AS ENUM (
    'queued',
    'ocr_processing',
    'ocr_done',
    'extracting',
    'extracted',
    'validating',
    'done',
    'failed'
);

-- ----------------------------------------------------------
-- 2. TABLE: Tenants
-- ----------------------------------------------------------
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    config JSONB DEFAULT '{}'::jsonb,
    rate_limit INT DEFAULT 60,  -- requests per minute
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Seed a default tenant for local development
INSERT INTO tenants (name, config) VALUES
    ('default', '{"description": "Local development tenant"}'::jsonb);

-- ----------------------------------------------------------
-- 3. TABLE: Jobs
-- ----------------------------------------------------------
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    status job_status NOT NULL DEFAULT 'queued',
    input_file_path VARCHAR(512) NOT NULL,       -- MinIO: "bucket/tenant_id/job_id/input.png"
    ocr_output_path VARCHAR(512),
    extraction_output_path VARCHAR(512),
    confidence_score FLOAT,
    ocr_data JSONB,
    extraction_data JSONB,                        -- Final structured extraction
    evaluation_data JSONB,                        -- Accuracy metrics vs ground truth
    ground_truth JSONB,                           -- HF dataset parsed_data for evaluation
    error_message TEXT,
    retry_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------
-- 4. TRIGGER: Auto-update `updated_at` on row modification
-- ----------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ----------------------------------------------------------
-- 5. INDEXES for common query patterns
-- ----------------------------------------------------------
CREATE INDEX idx_jobs_tenant_id ON jobs(tenant_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at DESC);

-- Create tasks table
CREATE TABLE IF NOT EXISTS tasks (
    request_id VARCHAR(100) PRIMARY KEY,
    model VARCHAR(100) NOT NULL,
    conversations JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    result JSONB,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for better query performance
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_created_at ON tasks(created_at);
CREATE INDEX idx_tasks_model ON tasks(model);
CREATE INDEX idx_tasks_status_created ON tasks(status, created_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create view for queue statistics
CREATE OR REPLACE VIEW queue_stats AS
SELECT 
    status,
    COUNT(*) as count,
    MIN(created_at) as oldest_task,
    MAX(created_at) as newest_task
FROM tasks
GROUP BY status;
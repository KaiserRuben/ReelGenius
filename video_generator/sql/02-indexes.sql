-- Create indexes for better query performance

-- Create indexes on tasks table
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_platform ON tasks(platform);
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_template_id ON tasks(template_id);

-- Create indexes on cache_entries table
CREATE INDEX IF NOT EXISTS idx_cache_entries_key ON cache_entries(key);
CREATE INDEX IF NOT EXISTS idx_cache_entries_created_at ON cache_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_cache_entries_expires_at ON cache_entries(expires_at);

-- Create indexes on media_assets table
CREATE INDEX IF NOT EXISTS idx_media_assets_task_id ON media_assets(task_id);
CREATE INDEX IF NOT EXISTS idx_media_assets_type ON media_assets(type);
CREATE INDEX IF NOT EXISTS idx_media_assets_created_at ON media_assets(created_at);

-- Create indexes on video_templates table
CREATE INDEX IF NOT EXISTS idx_video_templates_platform ON video_templates(platform);
CREATE INDEX IF NOT EXISTS idx_video_templates_is_public ON video_templates(is_public);
CREATE INDEX IF NOT EXISTS idx_video_templates_created_by ON video_templates(created_by);

-- Create indexes on prompt_templates table
CREATE INDEX IF NOT EXISTS idx_prompt_templates_type ON prompt_templates(type);
CREATE INDEX IF NOT EXISTS idx_prompt_templates_is_public ON prompt_templates(is_public);
CREATE INDEX IF NOT EXISTS idx_prompt_templates_created_by ON prompt_templates(created_by);

-- Create indexes on platform_accounts table
CREATE INDEX IF NOT EXISTS idx_platform_accounts_user_id ON platform_accounts(user_id);
CREATE INDEX IF NOT EXISTS idx_platform_accounts_platform ON platform_accounts(platform);
CREATE INDEX IF NOT EXISTS idx_platform_accounts_is_active ON platform_accounts(is_active);

-- Create indexes on usage_statistics table
CREATE INDEX IF NOT EXISTS idx_usage_statistics_user_id ON usage_statistics(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_statistics_date ON usage_statistics(date);

-- Record this migration
INSERT INTO schema_migrations (version) VALUES ('02-indexes')
ON CONFLICT (version) DO NOTHING;
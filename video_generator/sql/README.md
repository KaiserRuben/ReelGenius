# ReelGenius PostgreSQL Database

This directory contains SQL scripts for initializing and configuring the PostgreSQL database used by ReelGenius.

## Schema Overview

### Tables

1. **users** - User accounts and authentication
2. **tasks** - Video generation tasks
3. **cache_entries** - Semantic cache for embeddings and generated content
4. **video_templates** - Templates for different video formats and platforms
5. **media_assets** - Media files (images, audio, video) linked to tasks
6. **prompt_templates** - Templates for LLM prompts
7. **platform_accounts** - Social media platform connections
8. **usage_statistics** - User usage metrics for analytics and billing

### Relationships

- A user can have many tasks
- A user can have many platform accounts
- A task belongs to a user (optional)
- A task can use a video template
- A task can have many media assets
- A video template can belong to a user (optional)
- A prompt template can belong to a user (optional)

## Initialization Process

1. PostgreSQL container runs SQL scripts in this directory on initialization
2. The application uses SQLAlchemy ORM to create tables that don't exist via `init_db()`
3. The `db_init.py` script creates initial data when the container starts

## Migrations

The `schema_migrations` table tracks applied schema changes. To add a new migration:

1. Create a numbered SQL file in this directory (e.g., `03-new-features.sql`)
2. Include an INSERT statement to record the migration in the schema_migrations table
3. When containers restart, new migrations will be applied automatically

## Manual Database Operations

To connect to the database manually:

```bash
docker exec -it reelgenius-postgres psql -U reelgenius -d videogen
```

Common commands:
- `\dt` - List tables
- `\d table_name` - Describe table schema
- `\q` - Quit
#!/bin/bash
set -e

# Function to wait for PostgreSQL to be ready
wait_for_postgres() {
  echo "Waiting for PostgreSQL..."
  until PGPASSWORD=reelgenius psql -h postgres -U reelgenius -d videogen -c '\q' > /dev/null 2>&1; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 1
  done
  echo "PostgreSQL is up and running!"
}

# Function to initialize the database
initialize_database() {
  echo "Initializing database..."
  python -m video_generator.db_init
  echo "Database initialization completed!"
}

# Function to start the API server
start_api() {
  echo "Starting API server..."
  exec uvicorn video_generator.api:app --host 0.0.0.0 --port 8000
}

# Function to start the Celery worker
start_worker() {
  echo "Starting Celery worker..."
  exec celery -A video_generator.tasks worker --loglevel=info
}

# Main logic
echo "Starting ReelGenius container..."

# Wait for PostgreSQL to be ready
wait_for_postgres

# Initialize database
initialize_database

# Determine what to run based on the command
if [ "$1" = "worker" ]; then
  start_worker
else
  start_api
fi
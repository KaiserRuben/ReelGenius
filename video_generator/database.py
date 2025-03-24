from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import os
from datetime import datetime

# Get database URL from environment variable with fallback to SQLite for testing
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://reelgenius:reelgenius@localhost:5432/videogen')

# Check if we're in a test environment
is_test_env = 'PYTEST_CURRENT_TEST' in os.environ
if is_test_env:
    # Use SQLite for testing to avoid requiring PostgreSQL
    DATABASE_URL = 'sqlite:///:memory:'

# Create engine
engine = create_engine(DATABASE_URL)
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)
Base = declarative_base()


class Task(Base):
    """Task model for storing video generation task information."""
    __tablename__ = 'tasks'

    task_id = Column(String, primary_key=True)
    status = Column(String, nullable=False, default='queued')
    progress = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    platform = Column(String, nullable=True)
    voice_gender = Column(String, nullable=True)
    content_summary = Column(String, nullable=True)
    content_source = Column(String, nullable=True)
    celery_task_id = Column(String, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    execution_time = Column(Float, nullable=True)

    def to_dict(self):
        """Convert Task model to dictionary."""
        task_dict = {
            'task_id': self.task_id,
            'status': self.status,
            'progress': self.progress,
            'created_at': self.created_at.timestamp() if self.created_at else None,
            'updated_at': self.updated_at.timestamp() if self.updated_at else None,
            'platform': self.platform,
            'voice_gender': self.voice_gender,
            'content_summary': self.content_summary,
            'content_source': self.content_source,
            'celery_task_id': self.celery_task_id,
            'result': self.result,
            'error': self.error,
            'execution_time': self.execution_time
        }
        # Filter out None values
        return {k: v for k, v in task_dict.items() if v is not None}


# Create tables if they don't exist
def init_db():
    """Initialize database by creating tables."""
    Base.metadata.create_all(engine)

# Function to get a session
def get_session():
    """Get a database session."""
    session = Session()
    try:
        return session
    finally:
        session.close()

# Initialize database on import
if not is_test_env:
    # Skip auto-initialization during testing
    init_db()
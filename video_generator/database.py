from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, JSON, Integer, ForeignKey, Table, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, relationship
import os
from datetime import datetime
import uuid

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


class User(Base):
    """User model for authentication and preferences."""
    __tablename__ = 'users'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    api_key = Column(String, unique=True, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    tasks = relationship("Task", back_populates="user")
    platform_accounts = relationship("PlatformAccount", back_populates="user")
    usage_statistics = relationship("UsageStatistics", back_populates="user")
    
    def to_dict(self):
        """Convert User model to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.timestamp() if self.created_at else None,
            'last_login': self.last_login.timestamp() if self.last_login else None,
            'is_active': self.is_active
        }


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
    # Use Text column instead of JSON to allow for larger data storage
    # PostgreSQL will convert JSON to/from TEXT automatically
    result = Column(JSON, nullable=True)  
    error = Column(String, nullable=True)
    execution_time = Column(Float, nullable=True)
    
    # Foreign keys
    user_id = Column(String, ForeignKey('users.id'), nullable=True)
    template_id = Column(String, ForeignKey('video_templates.id'), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="tasks")
    template = relationship("VideoTemplate", back_populates="tasks")
    media_assets = relationship("MediaAsset", back_populates="task")

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
            'execution_time': self.execution_time,
            'user_id': self.user_id,
            'template_id': self.template_id
        }
        # Filter out None values
        return {k: v for k, v in task_dict.items() if v is not None}


class CacheEntry(Base):
    """Model for semantic cache entries."""
    __tablename__ = 'cache_entries'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String, nullable=False, unique=True)
    embedding_vector = Column(String, nullable=True)  # Serialized vector
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    hits_count = Column(Integer, nullable=False, default=0)
    cost_saved = Column(Float, nullable=False, default=0.0)
    
    def to_dict(self):
        """Convert CacheEntry model to dictionary."""
        return {
            'id': self.id,
            'key': self.key,
            'content': self.content,
            'created_at': self.created_at.timestamp() if self.created_at else None,
            'expires_at': self.expires_at.timestamp() if self.expires_at else None,
            'hits_count': self.hits_count,
            'cost_saved': self.cost_saved
        }


class VideoTemplate(Base):
    """Model for video templates."""
    __tablename__ = 'video_templates'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    platform = Column(String, nullable=False)
    aspect_ratio = Column(String, nullable=False)
    duration_range = Column(String, nullable=True)
    visual_style = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_public = Column(Boolean, default=True)
    
    # Foreign keys
    created_by = Column(String, ForeignKey('users.id'), nullable=True)
    
    # Relationships
    tasks = relationship("Task", back_populates="template")
    
    def to_dict(self):
        """Convert VideoTemplate model to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'platform': self.platform,
            'aspect_ratio': self.aspect_ratio,
            'duration_range': self.duration_range,
            'visual_style': self.visual_style,
            'created_at': self.created_at.timestamp() if self.created_at else None,
            'updated_at': self.updated_at.timestamp() if self.updated_at else None,
            'is_public': self.is_public,
            'created_by': self.created_by
        }


class MediaAsset(Base):
    """Model for media assets (images, audio) associated with tasks."""
    __tablename__ = 'media_assets'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String, ForeignKey('tasks.task_id'), nullable=False)
    type = Column(String, nullable=False)  # 'image', 'audio', 'video'
    path = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)  # For audio/video
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    asset_metadata = Column(JSON, nullable=True)
    
    # Relationships
    task = relationship("Task", back_populates="media_assets")
    
    def to_dict(self):
        """Convert MediaAsset model to dictionary."""
        return {
            'id': self.id,
            'task_id': self.task_id,
            'type': self.type,
            'path': self.path,
            'size_bytes': self.size_bytes,
            'duration': self.duration,
            'created_at': self.created_at.timestamp() if self.created_at else None,
            'metadata': self.asset_metadata
        }


class PromptTemplate(Base):
    """Model for storing and managing prompt templates."""
    __tablename__ = 'prompt_templates'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # 'script', 'image', 'hook'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_public = Column(Boolean, default=True)
    
    # Foreign keys
    created_by = Column(String, ForeignKey('users.id'), nullable=True)
    
    def to_dict(self):
        """Convert PromptTemplate model to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'content': self.content,
            'created_at': self.created_at.timestamp() if self.created_at else None,
            'updated_at': self.updated_at.timestamp() if self.updated_at else None,
            'is_public': self.is_public,
            'created_by': self.created_by
        }


class PlatformAccount(Base):
    """Model for managing social media platform connections."""
    __tablename__ = 'platform_accounts'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    platform = Column(String, nullable=False)
    access_token = Column(String, nullable=True)
    refresh_token = Column(String, nullable=True)
    account_name = Column(String, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="platform_accounts")
    
    def to_dict(self):
        """Convert PlatformAccount model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'platform': self.platform,
            'account_name': self.account_name,
            'expires_at': self.expires_at.timestamp() if self.expires_at else None,
            'is_active': self.is_active,
            'created_at': self.created_at.timestamp() if self.created_at else None,
            'updated_at': self.updated_at.timestamp() if self.updated_at else None
        }


class UsageStatistics(Base):
    """Model for tracking API usage for billing/analytics."""
    __tablename__ = 'usage_statistics'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)
    api_calls = Column(Integer, nullable=False, default=0)
    tokens_used = Column(Integer, nullable=False, default=0)
    image_generations = Column(Integer, nullable=False, default=0)
    minutes_of_video = Column(Float, nullable=False, default=0.0)
    cost = Column(Float, nullable=False, default=0.0)
    
    # Relationships
    user = relationship("User", back_populates="usage_statistics")
    
    def to_dict(self):
        """Convert UsageStatistics model to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'date': self.date.timestamp() if self.date else None,
            'api_calls': self.api_calls,
            'tokens_used': self.tokens_used,
            'image_generations': self.image_generations,
            'minutes_of_video': self.minutes_of_video,
            'cost': self.cost
        }


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
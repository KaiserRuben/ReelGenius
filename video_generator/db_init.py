#!/usr/bin/env python
"""
Database initialization script for ReelGenius.
Run this script to initialize the database with required tables and initial data.
"""

import os
import sys
import logging
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

# Add app directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_generator.database import (
    engine, Base, Session, init_db, 
    User, Task, CacheEntry, VideoTemplate, MediaAsset, 
    PromptTemplate, PlatformAccount, UsageStatistics
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_tables():
    """Create all tables defined in the models."""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def create_initial_data():
    """Create initial data for the database."""
    session = Session()
    
    try:
        # Check if we already have data (to avoid duplication on restarts)
        if session.query(User).first():
            logger.info("Initial data already exists, skipping...")
            return
        
        logger.info("Creating initial data...")
        
        # Create admin user
        admin_user = User(
            username="admin",
            email="admin@reelgenius.com",
            password_hash="$2b$12$QlCVN9jfHoXkRiYrdlFwxuB6DPfS3TgyRCi5GXnRZx3P5.w5MXTUK",  # hashed 'admin123'
            api_key="rg_api_adm_00000000000000000000000000000000"
        )
        session.add(admin_user)
        
        # Create default video templates for different platforms
        platforms = ["tiktok", "youtube", "instagram", "facebook"]
        aspect_ratios = {
            "tiktok": "9:16",
            "youtube": "16:9",
            "instagram": "1:1",
            "facebook": "16:9"
        }
        
        for platform in platforms:
            template = VideoTemplate(
                name=f"Default {platform.capitalize()} Template",
                description=f"Standard template for {platform.capitalize()} videos",
                platform=platform,
                aspect_ratio=aspect_ratios[platform],
                visual_style="Modern, vibrant, high-contrast visuals with dynamic transitions",
                is_public=True
            )
            session.add(template)
        
        # Create default prompt templates
        prompt_types = ["script", "image", "hook"]
        for prompt_type in prompt_types:
            template = PromptTemplate(
                name=f"Default {prompt_type.capitalize()} Template",
                type=prompt_type,
                content=(
                    f"This is a default template for generating {prompt_type} content. "
                    "Customize this template to fit your specific needs."
                ),
                is_public=True
            )
            session.add(template)
        
        session.commit()
        logger.info("Initial data created successfully")
    
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Error creating initial data: {e}")
        raise
    finally:
        session.close()

def main():
    """Main function to initialize the database."""
    try:
        # Create tables
        create_tables()
        
        # Create initial data
        create_initial_data()
        
        logger.info("Database initialization completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python
"""
Database administration utility for ReelGenius.

This script provides command-line utilities for database management.
"""

import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

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
        sys.exit(1)

def drop_tables():
    """Drop all tables from the database."""
    try:
        logger.info("Dropping database tables...")
        confirm = input("This will DELETE ALL DATA. Type 'confirm' to proceed: ")
        if confirm != 'confirm':
            logger.info("Operation cancelled.")
            return
        
        Base.metadata.drop_all(engine)
        logger.info("Database tables dropped successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error dropping database tables: {e}")
        sys.exit(1)

def create_admin_user(username, email, password):
    """Create an admin user."""
    session = Session()
    try:
        # Check if user already exists
        existing_user = session.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing_user:
            logger.error(f"User with username '{username}' or email '{email}' already exists.")
            return
        
        # Generate password hash
        import bcrypt
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        # Create API key
        import uuid
        api_key = f"rg_api_{uuid.uuid4().hex}"
        
        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            api_key=api_key,
            is_active=True
        )
        
        session.add(user)
        session.commit()
        
        logger.info(f"Admin user '{username}' created successfully.")
        logger.info(f"API Key: {api_key}")
    
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Error creating admin user: {e}")
        sys.exit(1)
    finally:
        session.close()

def list_users():
    """List all users in the database."""
    session = Session()
    try:
        users = session.query(User).all()
        
        if not users:
            logger.info("No users found in the database.")
            return
        
        logger.info(f"Found {len(users)} users:")
        for user in users:
            logger.info(f"ID: {user.id}, Username: {user.username}, Email: {user.email}, Active: {user.is_active}")
    
    except SQLAlchemyError as e:
        logger.error(f"Error listing users: {e}")
        sys.exit(1)
    finally:
        session.close()

def clean_tasks(days=30):
    """Clean up old tasks."""
    session = Session()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get count of tasks to be deleted
        count = session.query(Task).filter(Task.created_at < cutoff_date).count()
        
        if count == 0:
            logger.info(f"No tasks older than {days} days found.")
            return
        
        confirm = input(f"This will delete {count} tasks older than {days} days. Type 'confirm' to proceed: ")
        if confirm != 'confirm':
            logger.info("Operation cancelled.")
            return
        
        # Delete media assets first to maintain referential integrity
        media_deleted = session.query(MediaAsset).filter(
            MediaAsset.task_id.in_(
                session.query(Task.task_id).filter(Task.created_at < cutoff_date)
            )
        ).delete(synchronize_session=False)
        
        # Delete tasks
        tasks_deleted = session.query(Task).filter(Task.created_at < cutoff_date).delete()
        
        session.commit()
        
        logger.info(f"Deleted {tasks_deleted} tasks and {media_deleted} media assets.")
    
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Error cleaning tasks: {e}")
        sys.exit(1)
    finally:
        session.close()

def clean_cache(days=30):
    """Clean up old cache entries."""
    session = Session()
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get count of cache entries to be deleted
        count = session.query(CacheEntry).filter(CacheEntry.created_at < cutoff_date).count()
        
        if count == 0:
            logger.info(f"No cache entries older than {days} days found.")
            return
        
        confirm = input(f"This will delete {count} cache entries older than {days} days. Type 'confirm' to proceed: ")
        if confirm != 'confirm':
            logger.info("Operation cancelled.")
            return
        
        # Delete cache entries
        deleted = session.query(CacheEntry).filter(CacheEntry.created_at < cutoff_date).delete()
        
        session.commit()
        
        logger.info(f"Deleted {deleted} cache entries.")
    
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Error cleaning cache: {e}")
        sys.exit(1)
    finally:
        session.close()

def list_tables():
    """List all tables in the database."""
    try:
        with engine.connect() as conn:
            if 'sqlite' in engine.url.drivername:
                # SQLite specific query
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
                tables = result.fetchall()
            else:
                # PostgreSQL specific query
                result = conn.execute(text(
                    "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';"
                ))
                tables = result.fetchall()
            
            if not tables:
                logger.info("No tables found in the database.")
                return
            
            logger.info("Tables in the database:")
            for table in tables:
                logger.info(f"  {table[0]}")
    
    except SQLAlchemyError as e:
        logger.error(f"Error listing tables: {e}")
        sys.exit(1)

def get_database_stats():
    """Get database statistics."""
    session = Session()
    try:
        stats = {
            "users": session.query(User).count(),
            "tasks": session.query(Task).count(),
            "cache_entries": session.query(CacheEntry).count(),
            "video_templates": session.query(VideoTemplate).count(),
            "media_assets": session.query(MediaAsset).count(),
            "prompt_templates": session.query(PromptTemplate).count(),
            "platform_accounts": session.query(PlatformAccount).count(),
            "usage_statistics": session.query(UsageStatistics).count()
        }
        
        # Task status breakdown
        task_status = {}
        for row in session.query(Task.status, text("count(*)")).group_by(Task.status).all():
            task_status[row[0]] = row[1]
        
        logger.info("Database Statistics:")
        logger.info("===================")
        
        for table, count in stats.items():
            logger.info(f"{table.replace('_', ' ').title()}: {count}")
        
        logger.info("\nTask Status Breakdown:")
        for status, count in task_status.items():
            logger.info(f"  {status}: {count}")
    
    except SQLAlchemyError as e:
        logger.error(f"Error getting database statistics: {e}")
        sys.exit(1)
    finally:
        session.close()

def main():
    parser = argparse.ArgumentParser(description="ReelGenius Database Administration Utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # create-tables command
    subparsers.add_parser("create-tables", help="Create all database tables")
    
    # drop-tables command
    subparsers.add_parser("drop-tables", help="Drop all database tables")
    
    # create-admin command
    create_admin_parser = subparsers.add_parser("create-admin", help="Create an admin user")
    create_admin_parser.add_argument("--username", required=True, help="Admin username")
    create_admin_parser.add_argument("--email", required=True, help="Admin email")
    create_admin_parser.add_argument("--password", required=True, help="Admin password")
    
    # list-users command
    subparsers.add_parser("list-users", help="List all users")
    
    # clean-tasks command
    clean_tasks_parser = subparsers.add_parser("clean-tasks", help="Clean up old tasks")
    clean_tasks_parser.add_argument("--days", type=int, default=30, help="Age in days (default: 30)")
    
    # clean-cache command
    clean_cache_parser = subparsers.add_parser("clean-cache", help="Clean up old cache entries")
    clean_cache_parser.add_argument("--days", type=int, default=30, help="Age in days (default: 30)")
    
    # list-tables command
    subparsers.add_parser("list-tables", help="List all tables in the database")
    
    # stats command
    subparsers.add_parser("stats", help="Get database statistics")
    
    args = parser.parse_args()
    
    if args.command == "create-tables":
        create_tables()
    elif args.command == "drop-tables":
        drop_tables()
    elif args.command == "create-admin":
        create_admin_user(args.username, args.email, args.password)
    elif args.command == "list-users":
        list_users()
    elif args.command == "clean-tasks":
        clean_tasks(args.days)
    elif args.command == "clean-cache":
        clean_cache(args.days)
    elif args.command == "list-tables":
        list_tables()
    elif args.command == "stats":
        get_database_stats()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
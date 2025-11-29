"""
Database logging for Telegram bot - stores questions, answers, and user activity.

Uses SQLite by default, can be configured for PostgreSQL.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChatLog:
    """Single chat interaction log."""
    id: Optional[int] = None
    user_id: int = 0
    username: Optional[str] = None
    chat_id: int = 0
    question: str = ""
    answer: str = ""
    question_type: str = "unknown"  # search, ask, analyze, help, business
    response_time_ms: int = 0
    tokens_used: int = 0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


@dataclass
class UserProfile:
    """User profile with role and settings."""
    user_id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    role: str = "developer"  # developer, business, admin
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    total_queries: int = 0
    metadata: Optional[Dict[str, Any]] = None


class BotDatabase:
    """
    SQLite database for bot logging and user management.
    
    Usage:
        db = BotDatabase("data/bot.db")
        db.init_schema()
        
        # Log interaction
        log_id = db.log_interaction(ChatLog(...))
        
        # Get user profile
        user = db.get_or_create_user(user_id, username)
    """
    
    def __init__(self, db_path: str = "data/bot.db"):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_connection()
        
    def _init_connection(self):
        """Initialize SQLite connection with WAL mode for better concurrency."""
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30
        )
        self.conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        
    @contextmanager
    def get_cursor(self):
        """Get database cursor with automatic commit/rollback."""
        cursor = self.conn.cursor()
        try:
            yield cursor
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()
    
    def init_schema(self):
        """Create database tables if they don't exist."""
        with self.get_cursor() as cursor:
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    role TEXT DEFAULT 'developer',
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TIMESTAMP,
                    total_queries INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # Chat logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    username TEXT,
                    chat_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT,
                    question_type TEXT DEFAULT 'unknown',
                    response_time_ms INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0,
                    success INTEGER DEFAULT 1,
                    error_message TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_logs_user_id 
                ON chat_logs(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_logs_created_at 
                ON chat_logs(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_logs_question_type 
                ON chat_logs(question_type)
            """)
            
            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    rating INTEGER,  -- 1-5 or thumbs up/down (1/0)
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (log_id) REFERENCES chat_logs(id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
        logger.info(f"Database schema initialized at {self.db_path}")
    
    def get_or_create_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        role: str = "developer"
    ) -> UserProfile:
        """Get existing user or create new one."""
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            
            if row:
                # Update last seen
                cursor.execute(
                    "UPDATE users SET last_seen_at = CURRENT_TIMESTAMP, username = COALESCE(?, username) WHERE user_id = ?",
                    (username, user_id)
                )
                return UserProfile(
                    user_id=row['user_id'],
                    username=row['username'],
                    first_name=row['first_name'],
                    role=row['role'],
                    is_active=bool(row['is_active']),
                    created_at=row['created_at'],
                    last_seen_at=row['last_seen_at'],
                    total_queries=row['total_queries'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else None
                )
            else:
                # Create new user
                cursor.execute(
                    """
                    INSERT INTO users (user_id, username, first_name, role, last_seen_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (user_id, username, first_name, role)
                )
                return UserProfile(
                    user_id=user_id,
                    username=username,
                    first_name=first_name,
                    role=role
                )
    
    def update_user_role(self, user_id: int, role: str) -> bool:
        """Update user role (developer, business, admin)."""
        with self.get_cursor() as cursor:
            cursor.execute(
                "UPDATE users SET role = ? WHERE user_id = ?",
                (role, user_id)
            )
            return cursor.rowcount > 0
    
    def log_interaction(self, log: ChatLog) -> int:
        """
        Log a chat interaction.
        
        Args:
            log: ChatLog object with interaction details
            
        Returns:
            ID of the created log entry
        """
        with self.get_cursor() as cursor:
            metadata_json = json.dumps(log.metadata) if log.metadata else None
            
            cursor.execute(
                """
                INSERT INTO chat_logs 
                (user_id, username, chat_id, question, answer, question_type, 
                 response_time_ms, tokens_used, success, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log.user_id, log.username, log.chat_id, log.question,
                    log.answer, log.question_type, log.response_time_ms,
                    log.tokens_used, int(log.success), log.error_message,
                    metadata_json
                )
            )
            
            # Update user query count
            cursor.execute(
                "UPDATE users SET total_queries = total_queries + 1 WHERE user_id = ?",
                (log.user_id,)
            )
            
            return cursor.lastrowid
    
    def add_feedback(
        self,
        log_id: int,
        user_id: int,
        rating: int,
        comment: Optional[str] = None
    ) -> int:
        """Add feedback to a chat log entry."""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO feedback (log_id, user_id, rating, comment)
                VALUES (?, ?, ?, ?)
                """,
                (log_id, user_id, rating, comment)
            )
            return cursor.lastrowid
    
    def get_user_history(
        self,
        user_id: int,
        limit: int = 50,
        question_type: Optional[str] = None
    ) -> List[ChatLog]:
        """Get user's chat history."""
        with self.get_cursor() as cursor:
            query = """
                SELECT * FROM chat_logs 
                WHERE user_id = ?
            """
            params = [user_id]
            
            if question_type:
                query += " AND question_type = ?"
                params.append(question_type)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            return [
                ChatLog(
                    id=row['id'],
                    user_id=row['user_id'],
                    username=row['username'],
                    chat_id=row['chat_id'],
                    question=row['question'],
                    answer=row['answer'],
                    question_type=row['question_type'],
                    response_time_ms=row['response_time_ms'],
                    tokens_used=row['tokens_used'],
                    success=bool(row['success']),
                    error_message=row['error_message'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else None,
                    created_at=row['created_at']
                )
                for row in cursor.fetchall()
            ]
    
    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get usage statistics for the last N days."""
        with self.get_cursor() as cursor:
            # Total queries
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                       AVG(response_time_ms) as avg_response_time
                FROM chat_logs
                WHERE created_at >= datetime('now', ?)
            """, (f'-{days} days',))
            
            totals = cursor.fetchone()
            
            # Queries by type
            cursor.execute("""
                SELECT question_type, COUNT(*) as count
                FROM chat_logs
                WHERE created_at >= datetime('now', ?)
                GROUP BY question_type
            """, (f'-{days} days',))
            
            by_type = {row['question_type']: row['count'] for row in cursor.fetchall()}
            
            # Active users
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) as active_users
                FROM chat_logs
                WHERE created_at >= datetime('now', ?)
            """, (f'-{days} days',))
            
            active_users = cursor.fetchone()['active_users']
            
            # Users by role
            cursor.execute("""
                SELECT role, COUNT(*) as count
                FROM users
                WHERE is_active = 1
                GROUP BY role
            """)
            
            by_role = {row['role']: row['count'] for row in cursor.fetchall()}
            
            return {
                'period_days': days,
                'total_queries': totals['total'] or 0,
                'successful_queries': totals['successful'] or 0,
                'success_rate': (totals['successful'] / totals['total'] * 100) if totals['total'] else 0,
                'avg_response_time_ms': int(totals['avg_response_time'] or 0),
                'queries_by_type': by_type,
                'active_users': active_users,
                'users_by_role': by_role,
            }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Singleton instance
_db_instance: Optional[BotDatabase] = None


def get_bot_database(db_path: str = "data/bot.db") -> BotDatabase:
    """Get or create singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = BotDatabase(db_path)
        _db_instance.init_schema()
    return _db_instance


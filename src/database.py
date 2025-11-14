import asyncpg
import logging
import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger("uvicorn.error")

class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Initialize database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=10,
                max_size=50,
                command_timeout=60
            )
            logger.info("Database connection pool created")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def create_task(
        self,
        request_id: str,
        model: str,
        conversations: List[Dict[str, str]],
        metadata: Dict[str, Any]
    ) -> bool:
        """Create a new task in the database."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO tasks (request_id, model, conversations, metadata, status)
                    VALUES ($1, $2, $3, $4, 'pending')
                    """,
                    request_id,
                    model,
                    json.dumps(conversations),
                    json.dumps(metadata)
                )
            logger.info(f"Created task {request_id} in database")
            return True
        except Exception as e:
            logger.error(f"Failed to create task {request_id}: {e}")
            return False
    
    async def get_task(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a task by request_id."""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT request_id, model, conversations, metadata, status, 
                           result, error, created_at, completed_at
                    FROM tasks
                    WHERE request_id = $1
                    """,
                    request_id
                )
                
                if row:
                    return {
                        'request_id': row['request_id'],
                        'model': row['model'],
                        'conversations': json.loads(row['conversations']) if isinstance(row['conversations'], str) else row['conversations'],
                        'metadata': json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata'],
                        'status': row['status'],
                        'result': json.loads(row['result']) if row['result'] and isinstance(row['result'], str) else row['result'],
                        'error': row['error'],
                        'created_at': row['created_at'],
                        'completed_at': row['completed_at']
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get task {request_id}: {e}")
            return None
    
    async def update_task_status(
        self,
        request_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        """Update task status and result."""
        try:
            async with self.pool.acquire() as conn:
                if status == 'completed' or status == 'failed':
                    await conn.execute(
                        """
                        UPDATE tasks
                        SET status = $1, result = $2, error = $3, completed_at = CURRENT_TIMESTAMP
                        WHERE request_id = $4
                        """,
                        status,
                        json.dumps(result) if result else None,
                        error,
                        request_id
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE tasks
                        SET status = $1
                        WHERE request_id = $2
                        """,
                        status,
                        request_id
                    )
            logger.info(f"Updated task {request_id} status to {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to update task {request_id}: {e}")
            return False
    
    async def get_pending_tasks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get pending tasks for processing."""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT request_id, model, conversations, metadata
                    FROM tasks
                    WHERE status = 'pending'
                    ORDER BY created_at ASC
                    LIMIT $1
                    """,
                    limit
                )
                
                return [
                    {
                        'request_id': row['request_id'],
                        'model': row['model'],
                        'conversations': json.loads(row['conversations']) if isinstance(row['conversations'], str) else row['conversations'],
                        'metadata': json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to get pending tasks: {e}")
            return []
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT status, COUNT(*) as count
                    FROM tasks
                    GROUP BY status
                    """
                )
                
                stats = {row['status']: row['count'] for row in rows}
                
                # Get total count
                total = await conn.fetchval("SELECT COUNT(*) FROM tasks")
                
                return {
                    'pending': stats.get('pending', 0),
                    'processing': stats.get('processing', 0),
                    'completed': stats.get('completed', 0),
                    'failed': stats.get('failed', 0),
                    'total': total
                }
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {'pending': 0, 'processing': 0, 'completed': 0, 'failed': 0, 'total': 0}
    
    async def cleanup_old_tasks(self, days: int = 7) -> int:
        """Remove completed tasks older than specified days."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM tasks
                    WHERE status IN ('completed', 'failed')
                    AND completed_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
                    """,
                    days
                )
                deleted = int(result.split()[-1])
                logger.info(f"Cleaned up {deleted} old tasks")
                return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old tasks: {e}")
            return 0
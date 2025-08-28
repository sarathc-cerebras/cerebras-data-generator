import asyncio
import logging
import json
import uuid
import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
import aiosqlite
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from generator import (
    load_configuration,
    initialize_shared_resources,
    API_STATE
)
from worker import run_workers

# --- Pydantic Models for API ---
class GenerationRequest(BaseModel):
    conversations: list
    model: Optional[str] = None
    metadata: Optional[dict] = None

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events, including the DB pool.
    """
    logging.info("API Server starting up...")
    config = load_configuration("config/inference-config.yaml")
    if not config:
        raise RuntimeError("Failed to load configuration.")

    initialize_shared_resources(config)
    
    db_path = config['database']['path']
    
    # Ensure the database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    await setup_database(db_path)
    
    # Size the connection pool appropriately
    worker_concurrency = config.get("worker", {}).get("concurrency", 200)
    pool_size = worker_concurrency + 10  # Extra connections for API endpoints
    
    db_pool = asyncio.Queue(maxsize=pool_size)
    all_connections = []
    
    for _ in range(pool_size):
        db = await aiosqlite.connect(db_path)
        db.row_factory = aiosqlite.Row
        await db_pool.put(db)
        all_connections.append(db)
    
    API_STATE["db_pool"] = db_pool
    API_STATE["all_db_connections"] = all_connections

    logging.info(f"Database connection pool created with size {pool_size}.")
    logging.info(f"Starting worker system with {worker_concurrency} concurrent workers.")
    
    # Start the worker system
    asyncio.create_task(run_workers())
    
    yield

    logging.info("Shutting down...")
    if API_STATE.get("all_db_connections"):
        for db in API_STATE["all_db_connections"]:
            await db.close()
        logging.info("Database connections closed.")

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """FastAPI dependency to safely get a DB connection from the pool."""
    pool = API_STATE["db_pool"]
    db = await pool.get()
    try:
        yield db
    finally:
        await pool.put(db)

async def setup_database(db_path: str):
    """Initialize the database with proper schema and settings."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA synchronous=NORMAL;")
        await db.execute("PRAGMA cache_size=10000;")
        await db.execute("PRAGMA temp_store=memory;")
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                request_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                result TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create an index for faster queries
        await db.execute("CREATE INDEX IF NOT EXISTS idx_status_created ON tasks(status, created_at);")
        await db.commit()
    
    logging.info(f"Database initialized at {db_path} with optimized settings.")

@app.post("/generate", status_code=202)
async def enqueue_generation(request: GenerationRequest, db: aiosqlite.Connection = Depends(get_db)):
    """Enqueue a generation request."""
    request_id = f"task:{uuid.uuid4()}"
    model_name = request.model or next(API_STATE["model_cycler"])[0]
    
    try:
        await db.execute(
            "INSERT INTO tasks (request_id, model_name, payload, status) VALUES (?, ?, ?, 'pending')",
            (request_id, model_name, request.json())
        )
        await db.commit()
        logging.info(f"Enqueued task {request_id} for model {model_name}")
    except Exception as e:
        logging.error(f"Failed to enqueue task: {e}")
        raise HTTPException(status_code=500, detail="Could not enqueue request.")
    
    return {"request_id": request_id, "message": "Request accepted for processing."}

@app.get("/result/{request_id}")
async def get_generation_result(request_id: str, db: aiosqlite.Connection = Depends(get_db)):
    """Get the result of a generation request."""
    async with db.execute("SELECT status, result FROM tasks WHERE request_id = ?", (request_id,)) as cursor:
        row = await cursor.fetchone()
    
    if not row:
        raise HTTPException(status_code=404, detail="Request ID not found.")
    
    status, result_json = row
    
    if status == 'completed':
        # Clean up completed task
        await db.execute("DELETE FROM tasks WHERE request_id = ?", (request_id,))
        await db.commit()
        return JSONResponse(content=json.loads(result_json))
    elif status == 'failed':
        # Clean up failed task
        await db.execute("DELETE FROM tasks WHERE request_id = ?", (request_id,))
        await db.commit()
        return JSONResponse(
            status_code=500, 
            content={"request_id": request_id, "status": "failed", "detail": result_json}
        )
    else:
        # Task is still pending or processing
        return JSONResponse(
            status_code=202, 
            content={"request_id": request_id, "status": status}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API server is running"}

@app.get("/stats")
async def get_stats(db: aiosqlite.Connection = Depends(get_db)):
    """Get queue statistics."""
    async with db.execute("SELECT status, COUNT(*) as count FROM tasks GROUP BY status") as cursor:
        stats = {row[0]: row[1] for row in await cursor.fetchall()}
    
    return {
        "queue_stats": stats,
        "total_tasks": sum(stats.values())
    }

def main():
    config = load_configuration("config/inference-config.yaml")
    if not config:
        return
    
    api_config = config.get("api_server", {})
    uvicorn.run(
        "api_server:app", 
        host=api_config.get("host", "0.0.0.0"), 
        port=api_config.get("port", 8000), 
        reload=False,  # Disable reload for production
        log_level="warning"
    )

if __name__ == "__main__":
    main()
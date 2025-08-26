import asyncio
import logging
import json
import uuid
from typing import Optional

import uvicorn
import aiosqlite
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from generator import (
    load_configuration,
    initialize_shared_resources,
    API_STATE
)
# --- NEW: Import the worker logic ---
from worker import run_worker_loop

# --- Pydantic Models for API ---
class GenerationRequest(BaseModel):
    conversations: list
    model: Optional[str] = None

# --- FastAPI App Initialization ---
app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def get_db():
    """Dependency to get a DB connection."""
    return API_STATE["db"]

async def setup_database(db_path: str):
    """Creates the database and table if they don't exist."""
    # ... (existing code)
    async with aiosqlite.connect(db_path) as db:
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
        await db.commit()
    logging.info(f"Database initialized at {db_path}")

@app.on_event("startup")
async def startup_event():
    """Load config, initialize resources, connect to DB, and start the worker."""
    logging.info("API Server starting up...")
    config = load_configuration("config/inference-config.yaml")
    if not config:
        raise RuntimeError("Failed to load configuration.")

    initialize_shared_resources(config)
    
    db_path = config['database']['path']
    await setup_database(db_path)
    
    # Connect to the database
    db = await aiosqlite.connect(db_path)
    # Enable Write-Ahead Logging for better concurrency
    await db.execute("PRAGMA journal_mode=WAL")
    await db.commit()
    API_STATE["db"] = db

    # --- Start the worker as a background task ---
    logging.info("Starting background worker...")
    asyncio.create_task(run_worker_loop())

@app.on_event("shutdown")
async def shutdown_event():
    if API_STATE.get("db"):
        await API_STATE["db"].close()
        logging.info("Database connection closed.")

@app.post("/generate", status_code=202)
async def enqueue_generation(request: GenerationRequest):
    db = await get_db()
    request_id = f"task:{uuid.uuid4()}"

    if request.model and request.model not in API_STATE["model_resources"]:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' is not configured.")
    
    model_name = request.model or next(API_STATE["model_cycler"])[0]

    try:
        await db.execute(
            "INSERT INTO tasks (request_id, model_name, payload, status) VALUES (?, ?, ?, 'pending')",
            (request_id, model_name, request.json(),)
        )
        await db.commit()
    except Exception as e:
        logging.error(f"Failed to write to database: {e}")
        raise HTTPException(status_code=500, detail="Could not enqueue request.")

    return {"request_id": request_id, "message": "Request accepted for processing."}

@app.get("/result/{request_id}")
async def get_generation_result(request_id: str):
    db = await get_db()
    async with db.execute("SELECT status, result FROM tasks WHERE request_id = ?", (request_id,)) as cursor:
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Request ID not found.")

    status, result = row
    if status == 'completed':
        # Optionally delete the task after retrieval
        await db.execute("DELETE FROM tasks WHERE request_id = ?", (request_id,))
        await db.commit()
        return JSONResponse(content=json.loads(result))
    elif status == 'failed':
        return JSONResponse(status_code=500, content={"request_id": request_id, "status": "failed", "detail": result})
    else:
        return JSONResponse(status_code=202, content={"request_id": request_id, "status": status})

def main():
    """Starts the FastAPI server."""
    config = load_configuration("config/inference-config.yaml")
    if not config: return
    api_config = config.get("api_server", {})
    uvicorn.run("api_server:app", host=api_config.get("host", "0.0.0.0"), port=api_config.get("port", 8000), reload=True)

if __name__ == "__main__":
    main()
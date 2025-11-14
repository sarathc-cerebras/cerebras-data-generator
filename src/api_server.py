import asyncio
import logging
import uuid
import os
from typing import Optional, List
from contextlib import asynccontextmanager

import uvicorn
from metrics import MetricsTracker
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import yaml
import time

from generator import (
    load_configuration,
    initialize_shared_resources,
    API_STATE
)
from worker import run_workers
from database import DatabaseManager

security = HTTPBearer()

VALID_API_TOKENS = set(
    token.strip() 
    for token in os.getenv("API_AUTH_TOKENS", "").split(",") 
    if token.strip()
)

# --- Pydantic Models for API ---
class GenerationRequest(BaseModel):
    conversations: list
    model: Optional[str] = None
    metadata: Optional[dict] = None

class ModelConfig(BaseModel):
    name: str
    concurrency: dict
    generation: dict

class ConfigUpdate(BaseModel):
    models_to_run: Optional[List[ModelConfig]] = None
    worker: Optional[dict] = None
    concurrency: Optional[dict] = None
    load_monitoring: Optional[dict] = None

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logging.info("API Server starting up...")
    config = load_configuration("config/inference-config.yaml")
    if not config:
        raise RuntimeError("Failed to load configuration.")

    # Initialize database
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://cerebras:changeme@localhost:5432/cerebras_data_gen"
    )
    db_manager = DatabaseManager(database_url)
    await db_manager.connect()
    API_STATE["db_manager"] = db_manager
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(window_seconds=60)
    API_STATE["metrics_tracker"] = metrics_tracker
    
    # Initialize shared resources
    initialize_shared_resources(config)
    logging.info(f"Initialized models: {list(API_STATE['model_resources'].keys())}")
    
    if API_STATE.get("client") is None:
        raise RuntimeError("Client not initialized in API_STATE")
    if not API_STATE.get("model_resources"):
        raise RuntimeError("No model resources configured")
    
    logging.info("Starting worker system.")
    worker_task = asyncio.create_task(run_workers())
    
    await asyncio.sleep(0.5)
    logging.info("✅ API Server fully initialized and ready")
    
    yield

    logging.info("Shutting down...")
    worker_task.cancel()
    
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    
    await db_manager.disconnect()

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the API authentication token."""
    token = credentials.credentials
    
    if not VALID_API_TOKENS:
        logging.warning("No API_AUTH_TOKENS configured - authentication disabled")
        return token
    
    if token not in VALID_API_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Serve the admin configuration page (no auth required to load the page)."""
    with open("src/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/config")
async def get_current_config(token: str = Depends(verify_token)):
    """Get the current runtime configuration."""
    return {
        "models_to_run": API_STATE["config"].get("models_to_run", []),
        "worker": API_STATE["config"].get("worker", {}),
        "concurrency": API_STATE["config"].get("concurrency", {}),
        "load_monitoring": API_STATE["config"].get("load_monitoring", {}),
        "api_server": API_STATE["config"].get("api_server", {})
    }

@app.put("/api/config")
async def update_config(config_update: ConfigUpdate, token: str = Depends(verify_token)):
    """Update runtime configuration dynamically."""
    try:
        if config_update.models_to_run is not None:
            API_STATE["config"]["models_to_run"] = [
                model.dict() for model in config_update.models_to_run
            ]
        
        if config_update.worker is not None:
            API_STATE["config"]["worker"] = config_update.worker
        
        if config_update.concurrency is not None:
            API_STATE["config"]["concurrency"] = config_update.concurrency
        
        if config_update.load_monitoring is not None:
            API_STATE["config"]["load_monitoring"] = config_update.load_monitoring
        
        config_path = "config/inference-config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(API_STATE["config"], f, default_flow_style=False, sort_keys=False)
        
        return {"status": "success", "message": "Configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/reload")
async def reload_config_from_file(token: str = Depends(verify_token)):
    """Reload configuration from the YAML file."""
    try:
        config_path = "config/inference-config.yaml"
        new_config = load_configuration(config_path)
        
        if not new_config:
            raise HTTPException(status_code=500, detail="Failed to load configuration file")
        
        API_STATE["config"] = new_config
        
        # Reinitialize model resources if needed
        model_resources = initialize_shared_resources(new_config)
        API_STATE["model_resources"] = model_resources
        
        return {"status": "success", "message": "Configuration reloaded from file"}
    except Exception as e:
        logging.error(f"Error reloading config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {str(e)}")


@app.get("/api/workers/status")
async def get_workers_status(token: str = Depends(verify_token)):
    """Get worker system status."""
    work_queue = API_STATE.get("work_queue")
    model_resources = API_STATE.get("model_resources")
    config = API_STATE.get("config", {})
    
    worker_config = config.get("worker", {})
    total_workers = worker_config.get("concurrency", 200)
    
    # Get active tasks from queue
    active_tasks = work_queue.qsize() if work_queue else 0
    
    # Get models list
    models = list(model_resources.keys()) if model_resources else []
    
    return {
        "total_workers": total_workers,
        "active_tasks": active_tasks,
        "models": models,
        "queue_capacity": work_queue.maxsize if work_queue else 0
    }

@app.get("/stats")
async def get_stats(token: str = Depends(verify_token)):
    """Get queue and system statistics."""
    db_manager = API_STATE.get("db_manager")
    work_queue = API_STATE.get("work_queue")
    
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database manager not initialized")
    
    # Get queue stats from database
    queue_stats = await db_manager.get_queue_stats()
    
    # Get memory queue info
    memory_queue_size = work_queue.qsize() if work_queue else 0
    
    return {
        "queue_stats": queue_stats,
        "memory_queue_size": memory_queue_size,
        "timestamp": time.time()
    }
    
    
@app.get("/api/metrics")
async def get_metrics(token: str = Depends(verify_token)):
    """Get detailed system metrics."""
    metrics_tracker = API_STATE.get("metrics_tracker")
    if not metrics_tracker:
        raise HTTPException(status_code=503, detail="Metrics tracker not initialized")
    
    return await metrics_tracker.get_metrics()
    
@app.post("/generate", status_code=202)
async def enqueue_generation(request: GenerationRequest, token: str = Depends(verify_token)):
    """Enqueue a generation request."""
    request_id = f"task:{uuid.uuid4()}"
    
    model_cycler = API_STATE.get("model_cycler")
    if not model_cycler:
        raise HTTPException(status_code=503, detail="Model cycler not initialized")
    
    model_name = request.model or next(model_cycler)[0]
    
    db_manager = API_STATE.get("db_manager")
    success = await db_manager.create_task(
        request_id=request_id,
        model=model_name,
        conversations=request.conversations,
        metadata=request.metadata or {}
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create task")
    
    logging.info(f"Created task {request_id} for model {model_name}")
    return {"request_id": request_id, "message": "Request accepted for processing."}

@app.get("/result/{request_id}")
async def get_generation_result(request_id: str, token: str = Depends(verify_token)):
    """Get the result of a generation request."""
    db_manager = API_STATE.get("db_manager")
    task = await db_manager.get_task(request_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Request ID not found.")
    
    status = task['status']
    
    if status == 'completed':
        result = JSONResponse(content=task['result'])
        
        # Delete task after result is retrieved
        try:
            await db_manager.delete_task(request_id)
            logging.info(f"Deleted completed task {request_id} from database")
        except Exception as e:
            logging.error(f"Failed to delete task {request_id}: {e}")
        
        return result
    elif status == 'failed':
        error_response = JSONResponse(
            status_code=500,
            content={"request_id": request_id, "status": "failed", "error": task['error']}
        )
        
        # Optionally delete failed tasks too
        try:
            await db_manager.delete_task(request_id)
            logging.info(f"Deleted failed task {request_id} from database")
        except Exception as e:
            logging.error(f"Failed to delete task {request_id}: {e}")
        
        return error_response
    else:
        return JSONResponse(
            status_code=202,
            content={"request_id": request_id, "status": status}
        )
        
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    db_manager = API_STATE.get("db_manager")
    try:
        # Check database connection
        await db_manager.get_queue_stats()
        return {"status": "healthy", "message": "API server is running", "database": "connected"}
    except Exception as e:
        return {"status": "degraded", "message": "API server running but database issue", "error": str(e)}

@app.get("/stats")
async def get_stats(token: str = Depends(verify_token)):
    """Get queue statistics."""
    db_manager = API_STATE.get("db_manager")
    stats = await db_manager.get_queue_stats()
    
    work_queue = API_STATE.get("work_queue")
    queue_size = work_queue.qsize() if work_queue else 0
    
    return {
        "queue_stats": {
            "pending": stats.get('pending', 0),
            "processing": stats.get('processing', 0),
            "completed": stats.get('completed', 0),
            "failed": stats.get('failed', 0)
        },
        "total_tasks": stats.get('total', 0),
        "queue_size": queue_size
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
        reload=False,
        log_level="warning",  
        access_log=False
    )

if __name__ == "__main__":
    main()
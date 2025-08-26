import asyncio
import logging
import json
import aiosqlite

from generator import (
    send_request,
    load_configuration,
    initialize_shared_resources,
    API_STATE
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_worker_loop():
    """The main worker process that polls the SQLite DB for tasks."""
    logging.info("Worker loop starting...")
    config = API_STATE["config"] # Use config from shared state
    
    db = API_STATE["db"]
    db.row_factory = aiosqlite.Row
    
    logging.info(f"Worker is now using the shared database connection.")

    while True:
        task = None
        try:
            # Transactionally select and lock a pending task
            async with db.execute("BEGIN EXCLUSIVE"):
                async with db.execute("SELECT * FROM tasks WHERE status = 'pending' ORDER BY created_at LIMIT 1") as cursor:
                    task = await cursor.fetchone()
                if task:
                    await db.execute("UPDATE tasks SET status = 'processing' WHERE request_id = ?", (task["request_id"],))
                    await db.commit()

            if not task:
                await asyncio.sleep(2) # Wait if no tasks are available
                continue

            request_id = task["request_id"]
            model_name = task["model_name"]
            payload_data = json.loads(task["payload"])
            
            logging.info(f"Processing job {request_id} for model '{model_name}'.")

            resources = API_STATE["model_resources"].get(model_name)
            if not resources:
                raise ValueError(f"Model '{model_name}' not found.")

            generation_params = config.get('generation', {})
            api_payload = {"model": model_name, "messages": payload_data['conversations'], **generation_params}

            result = await send_request(
                client=API_STATE["client"],
                payload=api_payload,
                semaphore=resources["semaphore"],
                config=config,
                concurrency_manager=resources["manager"]
            )

            # Update task as completed
            await db.execute(
                "UPDATE tasks SET status = 'completed', result = ? WHERE request_id = ?",
                (json.dumps(result), request_id)
            )
            await db.commit()
            logging.info(f"Finished job {request_id}.")

        except Exception as e:
            logging.error(f"An error occurred in the worker loop: {e}", exc_info=True)
            if task and db:
                await db.execute(
                    "UPDATE tasks SET status = 'failed', result = ? WHERE request_id = ?",
                    (str(e), task["request_id"])
                )
                await db.commit()
            await asyncio.sleep(5)
        finally:
            if 'db' in locals() and db and db.in_transaction:
                await db.rollback()

async def main_standalone_worker():
    """Initializes resources and runs the worker loop for standalone execution."""
    config = load_configuration("config/inference-config.yaml")
    if not config: raise RuntimeError("Failed to load configuration.")
    initialize_shared_resources(config)
    await run_worker_loop()

if __name__ == "__main__":
    # This block allows running the worker independently for debugging or scaling
    logging.info("Running worker in standalone mode.")
    asyncio.run(main_standalone_worker())
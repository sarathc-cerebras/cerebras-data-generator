import asyncio
import logging
import json
from generator import (
    API_STATE,
    process_single_item
)

logger = logging.getLogger("uvicorn.error")

async def _dispatcher(work_queue: asyncio.Queue):
    """
    Single dispatcher that claims tasks from database and puts them in memory queue.
    This eliminates database write contention by having only one writer for claiming tasks.
    """
    pool = API_STATE["db_pool"]
    
    # Atomic SQL to claim and return a task in one operation
    atomic_claim_sql = """
        UPDATE tasks
        SET status = 'processing'
        WHERE request_id = (
            SELECT request_id
            FROM tasks
            WHERE status = 'pending'
            ORDER BY created_at
            LIMIT 1
        )
        RETURNING *;
    """
    
    logger.info("Dispatcher started and ready to claim tasks")
    
    while True:
        db = None
        try:
            # Don't overflow the work queue
            if work_queue.full():
                await asyncio.sleep(0.1)
                continue
            
            db = await pool.get()
            
            # Try to claim a task
            cursor = await db.execute(atomic_claim_sql)
            task_row = await cursor.fetchone()
            await db.commit()
            
            if task_row:
                # Successfully claimed a task, put it in the work queue
                await work_queue.put(task_row)
                logger.info(f"Dispatcher claimed task {task_row['request_id']}")
            else:
                # No tasks available, wait a bit
                await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Dispatcher error: {e}", exc_info=True)
            await asyncio.sleep(1)
        finally:
            if db:
                await pool.put(db)

async def _processor(work_queue: asyncio.Queue):
    """
    Processor that takes tasks from memory queue and processes them.
    Multiple processors can run concurrently without database lock contention.
    """
    pool = API_STATE["db_pool"]
    
    while True:
        task_row = None
        db = None
        
        try:
            # Get a task from the work queue
            task_row = await work_queue.get()
            
            # Get a database connection for result updates only
            db = await pool.get()
            
            # Process the claimed task
            request_id = task_row["request_id"]
            model_name = task_row["model_name"]
            
            logger.info(f"Processor starting task {request_id} with model {model_name}")
            
            # Parse the payload
            payload_data = json.loads(task_row["payload"])
            item_to_process = {
                "conversations": payload_data["conversations"],
                **payload_data.get("metadata", {})
            }
            
            # Get model resources
            resources = API_STATE["model_resources"].get(model_name)
            if not resources:
                raise ValueError(f"Model '{model_name}' not found in worker resources.")
            
            # Process the item using the generator
            result = await process_single_item(
                item=item_to_process,
                client=API_STATE["client"],
                semaphore=resources["semaphore"],
                global_config=API_STATE["config"],
                concurrency_manager=resources["manager"],
                model_name=model_name,
                model_config=resources["config"]
            )
            
            # Update the database with the result
            if result.get("success"):
                final_result_json = json.dumps(result["result"])
                await db.execute(
                    "UPDATE tasks SET status = 'completed', result = ? WHERE request_id = ?",
                    (final_result_json, request_id)
                )
                logger.info(f"Task {request_id} completed successfully")
            else:
                error_msg = result.get('error', 'Unknown processing error')
                await db.execute(
                    "UPDATE tasks SET status = 'failed', result = ? WHERE request_id = ?",
                    (error_msg, request_id)
                )
                logger.error(f"Task {request_id} failed: {error_msg}")
            
            await db.commit()
            
        except Exception as e:
            logger.error(f"Processor error: {e}", exc_info=True)
            
            # Try to mark the task as failed if we have one
            if db and task_row:
                try:
                    if db.in_transaction:
                        await db.rollback()
                    await db.execute(
                        "UPDATE tasks SET status = 'failed', result = ? WHERE request_id = ?",
                        (str(e), task_row["request_id"])
                    )
                    await db.commit()
                except Exception as db_error:
                    logger.error(f"Failed to mark task as failed: {db_error}")
            
        finally:
            # Always return the database connection
            if db:
                await pool.put(db)
            
            # Mark the work queue task as done
            if task_row:
                work_queue.task_done()

async def run_workers():
    """
    Initialize and run the dispatcher-processor system.
    """
    config = API_STATE.get("config", {})
    worker_config = config.get("worker", {})
    concurrency = worker_config.get("concurrency", 200)
    
    # Create an in-memory work queue
    # Size it to buffer tasks but not consume too much memory
    queue_size = min(concurrency * 2, 1000)
    work_queue = asyncio.Queue(maxsize=queue_size)
    
    logger.info(f"Starting dispatcher-processor system with {concurrency} processors and queue size {queue_size}")
    
    # Start single dispatcher
    dispatcher_task = asyncio.create_task(_dispatcher(work_queue))
    
    # Start processor pool
    processor_tasks = []
    for i in range(concurrency):
        task = asyncio.create_task(_processor(work_queue))
        processor_tasks.append(task)
    
    logger.info(f"Dispatcher and {concurrency} processors started and ready")
    
    # Run dispatcher and processors concurrently
    try:
        await asyncio.gather(dispatcher_task, *processor_tasks)
    except Exception as e:
        logger.error(f"Worker system error: {e}")
        # Cancel all tasks
        dispatcher_task.cancel()
        for task in processor_tasks:
            task.cancel()
        raise
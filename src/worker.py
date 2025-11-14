import asyncio
import logging
from typing import Dict, Any
from generator import (
    API_STATE,
    process_single_item
)

logger = logging.getLogger("uvicorn.error")

async def _processor(work_queue: asyncio.Queue, processor_id: int, db_manager):
    """
    Processor that takes tasks from memory queue and processes them.
    """
    client = API_STATE.get("client")
    global_config = API_STATE.get("config")
    model_resources = API_STATE.get("model_resources")
    
    if not all([client, global_config, model_resources]):
        logger.error(f"Processor {processor_id}: Missing required resources in API_STATE")
        return
    
    logger.info(f"Processor {processor_id} started")
    
    while True:
        task = None
        
        try:
            try:
                task = await asyncio.wait_for(work_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            
            request_id = task['request_id']
            conversations = task['conversations']
            model_name = task['model']
            metadata = task.get('metadata', {})
            
            # Update status to processing in database
            await db_manager.update_task_status(request_id, 'processing')
            
            logger.info(f"Processor {processor_id} processing task {request_id} for model {model_name}")
            
            if model_name not in model_resources:
                error_msg = f"Model '{model_name}' not found in configuration"
                logger.error(f"Processor {processor_id}: {error_msg}")
                await db_manager.update_task_status(
                    request_id, 'failed', error=error_msg
                )
                work_queue.task_done()
                continue
            
            resources = model_resources[model_name]
            semaphore = resources["semaphore"]
            concurrency_manager = resources["manager"]
            model_config = resources["config"]
            
            # Process the item
            result_payload = await process_single_item(
                item={'conversations': conversations, 'metadata': metadata},
                client=client,
                semaphore=semaphore,
                global_config=global_config,
                concurrency_manager=concurrency_manager,
                model_name=model_name,
                model_config=model_config
            )
            
            # Update result in database
            if result_payload.get('success'):
                await db_manager.update_task_status(
                    request_id, 'completed', result=result_payload
                )
            else:
                await db_manager.update_task_status(
                    request_id, 'failed', 
                    result=result_payload,
                    error=result_payload.get('error', 'Unknown error')
                )
            
            logger.info(f"Processor {processor_id} completed task {request_id}")
            
        except Exception as e:
            logger.error(f"Processor {processor_id} error on task {task['request_id'] if task else 'unknown'}: {e}", exc_info=True)
            if task:
                await db_manager.update_task_status(
                    task['request_id'], 'failed', error=str(e)
                )
            
        finally:
            if task is not None:
                work_queue.task_done()

async def _dispatcher(work_queue: asyncio.Queue, db_manager):
    """
    Dispatcher that fetches pending tasks from PostgreSQL and adds them to the work queue.
    """
    logger.info("Dispatcher started")
    
    while True:
        try:
            # Check if queue has space
            if work_queue.qsize() < work_queue.maxsize * 0.8:
                # Fetch pending tasks from database
                pending_tasks = await db_manager.get_pending_tasks(limit=50)
                
                for task in pending_tasks:
                    try:
                        await asyncio.wait_for(
                            work_queue.put(task),
                            timeout=1.0
                        )
                        logger.debug(f"Dispatcher added task {task['request_id']} to queue")
                    except asyncio.TimeoutError:
                        logger.warning("Work queue full, waiting...")
                        break
            
            # Wait before next check
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Dispatcher error: {e}", exc_info=True)
            await asyncio.sleep(1.0)

async def run_workers():
    """
    Initialize and run the processor system with dispatcher.
    """
    config = API_STATE.get("config", {})
    db_manager = API_STATE.get("db_manager")
    
    if not config or not db_manager:
        logger.error("Config or database manager not available for workers")
        return
    
    worker_config = config.get("worker", {})
    concurrency = worker_config.get("concurrency", 200)
    
    # Create work queue
    work_queue = asyncio.Queue(maxsize=concurrency * 2)
    API_STATE["work_queue"] = work_queue
    
    logger.info(f"Starting dispatcher and {concurrency} processors")
    
    # Start dispatcher
    dispatcher_task = asyncio.create_task(_dispatcher(work_queue, db_manager))
    
    # Start processors
    processor_tasks = []
    for i in range(concurrency):
        task = asyncio.create_task(_processor(work_queue, i, db_manager))
        processor_tasks.append(task)
    
    logger.info(f"Dispatcher and {concurrency} processors started and ready")
    
    try:
        await asyncio.gather(dispatcher_task, *processor_tasks)
    except asyncio.CancelledError:
        logger.info("Worker system cancelled, shutting down...")
        dispatcher_task.cancel()
        for task in processor_tasks:
            task.cancel()
        raise
    except Exception as e:
        logger.error(f"Worker system error: {e}", exc_info=True)
        dispatcher_task.cancel()
        for task in processor_tasks:
            task.cancel()
        raise
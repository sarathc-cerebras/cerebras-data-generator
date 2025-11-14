import asyncio
import logging
import time
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
    metrics_tracker = API_STATE.get("metrics_tracker")
    
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
            
            # Record request sent
            start_time = time.time()
            if metrics_tracker:
                await metrics_tracker.record_request_sent(model_name)
            
            # Update status to processing in database
            await db_manager.update_task_status(request_id, 'processing')
            
            logger.info(f"Processor {processor_id} processing task {request_id} for model {model_name}")
            
            if model_name not in model_resources:
                error_msg = f"Model '{model_name}' not found in configuration"
                logger.error(f"Processor {processor_id}: {error_msg}")
                await db_manager.update_task_status(
                    request_id, 'failed', error=error_msg
                )
                
                # Record failure
                if metrics_tracker:
                    await metrics_tracker.record_request_failed(model_name, error_msg)
                
                work_queue.task_done()
                continue
            
            resources = model_resources[model_name]
            semaphore = resources["semaphore"]
            concurrency_manager = resources["manager"]
            model_config = resources["config"]
            
            # Process the item
            try:
                result_payload = await process_single_item(
                    item={'conversations': conversations, 'metadata': metadata},
                    client=client,
                    semaphore=semaphore,
                    global_config=global_config,
                    concurrency_manager=concurrency_manager,
                    model_name=model_name,
                    model_config=model_config
                )
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Update result in database
                if result_payload.get('success'):
                    await db_manager.update_task_status(
                        request_id, 'completed', result=result_payload
                    )
                    
                    # Record successful completion
                    if metrics_tracker:
                        await metrics_tracker.record_request_completed(model_name, duration)
                    
                    logger.info(f"Processor {processor_id} completed task {request_id} in {duration:.2f}s")
                else:
                    error = result_payload.get('error', 'Unknown error')
                    await db_manager.update_task_status(
                        request_id, 'failed', 
                        result=result_payload,
                        error=error
                    )
                    
                    # Record failure
                    if metrics_tracker:
                        await metrics_tracker.record_request_failed(model_name, error)
                        
                        # Check if it's a rate limit error
                        if "rate" in error.lower() or "429" in error or "limit" in error.lower():
                            await metrics_tracker.record_rate_limit(model_name)
                    
                    logger.error(f"Processor {processor_id} task {request_id} failed: {error}")
                    
            except Exception as process_error:
                duration = time.time() - start_time
                error_msg = str(process_error)
                
                # Record failure
                if metrics_tracker:
                    await metrics_tracker.record_request_failed(model_name, error_msg)
                    
                    # Check if it's a rate limit error
                    if "rate" in error_msg.lower() or "429" in error_msg or "limit" in error_msg.lower():
                        await metrics_tracker.record_rate_limit(model_name)
                
                await db_manager.update_task_status(
                    request_id, 'failed', error=error_msg
                )
                
                logger.error(f"Processor {processor_id} error processing task {request_id}: {process_error}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Processor {processor_id} error on task {task['request_id'] if task else 'unknown'}: {e}", exc_info=True)
            if task:
                try:
                    await db_manager.update_task_status(
                        task['request_id'], 'failed', error=str(e)
                    )
                    
                    # Record failure
                    if metrics_tracker:
                        model_name = task.get('model', 'unknown')
                        await metrics_tracker.record_request_failed(model_name, str(e))
                except Exception as db_error:
                    logger.error(f"Failed to update task status in database: {db_error}")
            
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
            current_size = work_queue.qsize()
            max_size = work_queue.maxsize
            
            if current_size < max_size * 0.8:
                # Fetch pending tasks from database
                pending_tasks = await db_manager.get_pending_tasks(limit=50)
                
                if pending_tasks:
                    logger.debug(f"Dispatcher found {len(pending_tasks)} pending tasks")
                
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
                    except Exception as put_error:
                        logger.error(f"Error adding task to queue: {put_error}")
            
            # Wait before next check
            await asyncio.sleep(0.5)
            
        except asyncio.CancelledError:
            logger.info("Dispatcher cancelled")
            raise
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
        
        # Wait for cancellation to complete
        try:
            await asyncio.gather(dispatcher_task, *processor_tasks, return_exceptions=True)
        except Exception:
            pass
        
        raise
    except Exception as e:
        logger.error(f"Worker system error: {e}", exc_info=True)
        dispatcher_task.cancel()
        for task in processor_tasks:
            task.cancel()
        
        # Wait for cancellation to complete
        try:
            await asyncio.gather(dispatcher_task, *processor_tasks, return_exceptions=True)
        except Exception:
            pass
        
        raise
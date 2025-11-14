import os
import asyncio
import httpx
import logging
from tqdm.asyncio import tqdm
from typing import List, Dict, Any, Optional, AsyncGenerator

# --- Configuration ---
CEREBRAS_PROXY_API_URL = os.getenv("CEREBRAS_PROXY_API_URL", "http://localhost:8000")
CEREBRAS_PROXY_API_TOKEN = os.getenv("CEREBRAS_PROXY_API_TOKEN", "")
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Custom Exceptions for the Client ---
class GenerationFailedError(Exception):
    """Raised when the generation task fails on the server."""
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details

class PollingTimeoutError(Exception):
    """Raised when polling for a result exceeds the timeout."""
    pass

class ServerNotReachableError(Exception):
    """Raised when the API server cannot be reached."""
    pass

class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass

# --- The AsyncSynthGenClient Wrapper ---
class AsyncSynthGenClient:
    """
    An asynchronous client for the Synthetic Data Generation API.
    """
    def __init__(
        self, 
        base_url: str = CEREBRAS_PROXY_API_URL, 
        api_token: Optional[str] = None,
        poll_interval: float = 1.0, 
        request_timeout: float = 300.0
    ):
        self.base_url = base_url
        self.api_token = api_token or CEREBRAS_PROXY_API_TOKEN
        self.poll_interval = poll_interval
        self.request_timeout = request_timeout
        
        # Setup headers with authentication
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        else:
            logging.warning("No API token provided. Server may reject requests if authentication is required.")
        
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout=self.request_timeout, pool=60.0, connect=10.0),
            limits=limits,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()

    async def _submit_job(
        self,
        conversations: List[Dict[str, str]],
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submits a job and returns the request_id immediately."""
        try:
            request_payload = {"conversations": conversations, "model": model, "metadata": metadata}
            response = await self._client.post("/generate", json=request_payload)
            
            if response.status_code == 401:
                raise AuthenticationError(
                    "Authentication failed. Please check your API token. "
                    "Set CEREBRAS_PROXY_API_TOKEN environment variable or pass api_token parameter."
                )
            
            response.raise_for_status()
            return response.json().get("request_id")
        except httpx.RequestError as e:
            raise ServerNotReachableError(f"Failed to submit job: {e}") from e

    async def _poll_for_result(self, request_id: str) -> Dict[str, Any]:
        """Polls for a specific request_id until completion."""
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < self.request_timeout:
            try:
                result_response = await self._client.get(f"/result/{request_id}")

                if result_response.status_code == 401:
                    raise AuthenticationError(
                        "Authentication failed while polling for result. "
                        "Please check your API token."
                    )
                
                if result_response.status_code == 200:
                    return result_response.json()
                elif result_response.status_code == 500:
                    error_details = result_response.json()
                    raise GenerationFailedError(
                        f"Task {request_id} failed on server", 
                        details=error_details
                    )
                elif result_response.status_code == 202:
                    # Still processing, wait and retry
                    await asyncio.sleep(self.poll_interval)
                else:
                    result_response.raise_for_status()
                    
            except httpx.RequestError as e:
                logging.warning(f"Polling error for {request_id}: {e}. Retrying...")
                await asyncio.sleep(self.poll_interval)
        
        raise PollingTimeoutError(f"Polling for {request_id} timed out after {self.request_timeout}s")

    async def generate(
        self,
        conversations: List[Dict[str, str]],
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Submits a single generation job and waits for the final result.
        """
        request_id = await self._submit_job(conversations, model, metadata)
        return await self._poll_for_result(request_id)

    async def generate_batch(
        self,
        items: List[Dict[str, Any]],
        max_concurrency: int,
        model: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        ✅ OPTIMIZED: Submits ALL jobs first, then polls concurrently.
        This achieves true parallelism instead of sequential processing.
        """
        # Phase 1: Submit ALL jobs as fast as possible
        logging.info(f"Submitting {len(items)} jobs to server...")
        submission_semaphore = asyncio.Semaphore(max_concurrency)
        
        async def _submit_with_limit(item):
            async with submission_semaphore:
                try:
                    request_id = await self._submit_job(
                        conversations=item["conversations"],
                        model=item.get("model") or model,
                        metadata=item.get("metadata", {})
                    )
                    return (request_id, item)
                except AuthenticationError as e:
                    logging.error(f"Authentication error: {e}")
                    raise  # Re-raise auth errors to stop batch processing
                except Exception as e:
                    logging.error(f"Failed to submit item {item.get('id', 'unknown')}: {e}")
                    return (None, item)
        
        # Submit all jobs concurrently
        submission_tasks = [asyncio.create_task(_submit_with_limit(item)) for item in items]
        submitted_jobs = []
        
        try:
            for future in tqdm(asyncio.as_completed(submission_tasks), total=len(items), desc="Submitting jobs"):
                request_id, item = await future
                if request_id:
                    submitted_jobs.append((request_id, item))
        except AuthenticationError:
            # Cancel remaining tasks on auth error
            for task in submission_tasks:
                if not task.done():
                    task.cancel()
            raise
        
        logging.info(f"✅ Submitted {len(submitted_jobs)}/{len(items)} jobs successfully")
        
        # Phase 2: Poll for ALL results concurrently
        logging.info(f"Polling for {len(submitted_jobs)} results...")
        polling_semaphore = asyncio.Semaphore(max_concurrency)
        
        async def _poll_with_limit(request_id, item):
            async with polling_semaphore:
                try:
                    result = await self._poll_for_result(request_id)
                    # Merge original item metadata with result
                    result["original_metadata"] = item.get("metadata", {})
                    return result
                except AuthenticationError as e:
                    logging.error(f"Authentication error while polling: {e}")
                    raise
                except Exception as e:
                    logging.error(f"Failed to get result for {request_id}: {e}")
                    return None
        
        # Poll for all results concurrently
        polling_tasks = [
            asyncio.create_task(_poll_with_limit(req_id, item)) 
            for req_id, item in submitted_jobs
        ]
        
        try:
            for future in tqdm(asyncio.as_completed(polling_tasks), total=len(submitted_jobs), desc="Fetching results"):
                result = await future
                if result:
                    yield result
        except AuthenticationError:
            # Cancel remaining tasks on auth error
            for task in polling_tasks:
                if not task.done():
                    task.cancel()
            raise
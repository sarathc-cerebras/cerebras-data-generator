import asyncio
import httpx
import logging
from tqdm.asyncio import tqdm
from typing import List, Dict, Any, Optional, AsyncGenerator

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"
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

# --- The new AsyncSynthGenClient Wrapper ---
class AsyncSynthGenClient:
    """
    An asynchronous client for the Synthetic Data Generation API.
    """
    def __init__(self, base_url: str = API_BASE_URL, poll_interval: float = 1.0, request_timeout: float = 300.0):
        self.base_url = base_url
        self.poll_interval = poll_interval
        self.request_timeout = request_timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0, connect=10.0)
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()

    async def generate(
        self,
        conversations: List[Dict[str, str]],
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Submits a single generation job and waits for the final result.

        Args:
            conversations: The conversation history.
            model: The model to use for generation.
            metadata: Optional metadata to include.

        Returns:
            The final generated data.

        Raises:
            GenerationFailedError: If the task fails on the server.
            PollingTimeoutError: If polling for the result times out.
            ServerNotReachableError: If the server is unreachable.
        """
        try:
            # 1. Submit the job
            request_payload = {"conversations": conversations, "model": model, "metadata": metadata}
            response = await self._client.post("/generate", json=request_payload)
            response.raise_for_status()
            request_id = response.json().get("request_id")

            # 2. Poll for the result
            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < self.request_timeout:
                result_response = await self._client.get(f"/result/{request_id}")

                if result_response.status_code == 200:  # Completed
                    return result_response.json()
                elif result_response.status_code == 500:  # Failed
                    raise GenerationFailedError("Generation task failed on the server.", details=result_response.json())
                elif result_response.status_code == 202:  # Still processing
                    await asyncio.sleep(self.poll_interval)
                else:
                    result_response.raise_for_status() # Raise for other client/server errors
            
            raise PollingTimeoutError(f"Polling for request {request_id} timed out after {self.request_timeout}s.")

        except httpx.RequestError as e:
            raise ServerNotReachableError(f"Failed to connect to the API server: {e}") from e

    async def generate_batch(
        self,
        items: List[Dict[str, Any]],
        max_concurrency: int,
        model: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes a batch of items concurrently and yields results as they complete.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _process_one(item):
            async with semaphore:
                try:
                    # Assumes item has 'conversations' and optional 'id'
                    return await self.generate(
                        conversations=item["conversations"],
                        model=model,
                        metadata={"id": item.get("id")}
                    )
                except Exception as e:
                    logging.error(f"Error processing item {item.get('id', 'unknown')}: {e}")
                    return None # Return None for failed items

        tasks = [asyncio.create_task(_process_one(item)) for item in items]

        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing batch"):
            result = await future
            if result:
                yield result

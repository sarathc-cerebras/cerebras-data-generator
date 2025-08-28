import asyncio
import httpx
import argparse
import json
import logging
from datasets import load_dataset
from tqdm.asyncio import tqdm

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

async def process_and_poll_item(client: httpx.AsyncClient, item: dict, semaphore: asyncio.Semaphore, poll_interval: float, model: str | None) -> dict:
    """
    Submits a full multi-turn item as a single job and polls for the final result.
    """
    async with semaphore:
        request_id = None
        try:
            # 1. Prepare and submit the entire item as one job
            if "conversations" not in item:
                return {"status": "skipped", "reason": "invalid_item_format"}

            request_payload = {
                "conversations": item["conversations"],
                "model": model,
                "metadata": {"id": item.get("id")}
            }
            
            response = await client.post(f"{API_BASE_URL}/generate", json=request_payload, timeout=30.0)
            response.raise_for_status()
            request_id = response.json().get("request_id")

            # 2. Poll for the final result with timeout and retry logic
            poll_attempts = 0
            max_poll_attempts = 300  # 5 minutes at 1-second intervals
            consecutive_errors = 0
            max_consecutive_errors = 5
            
            while poll_attempts < max_poll_attempts:
                try:
                    result_response = await client.get(f"{API_BASE_URL}/result/{request_id}", timeout=10.0)
                    consecutive_errors = 0  # Reset error counter on successful request
                    
                    if result_response.status_code == 200:  # Completed
                        return {"status": "completed", "data": result_response.json()}
                    elif result_response.status_code == 500:  # Failed
                        return {"status": "failed", "data": result_response.json()}
                    elif result_response.status_code == 202:  # Pending/Processing
                        await asyncio.sleep(poll_interval)
                        poll_attempts += 1
                    elif result_response.status_code == 404:  # Request not found
                        return {"status": "failed", "reason": "request_not_found", "request_id": request_id}
                    else:
                        return {"status": "failed", "reason": "unexpected_status", 
                               "status_code": result_response.status_code, "request_id": request_id}
                
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        return {"status": "failed", "reason": "server_unreachable", 
                               "details": f"Failed {consecutive_errors} consecutive polling attempts", 
                               "request_id": request_id}
                    # Wait longer after errors
                    await asyncio.sleep(poll_interval * 2)
                    poll_attempts += 1
            
            # Polling timeout
            return {"status": "failed", "reason": "polling_timeout", "request_id": request_id}

        except httpx.RequestError as e:
            return {"status": "failed", "reason": "request_error", "details": str(e), "request_id": request_id}
        except Exception as e:
            return {"status": "failed", "reason": "unknown_error", "details": str(e), "request_id": request_id}

async def main():
    """
    Client to load a dataset, submit items as jobs, and poll for final results.
    """
    parser = argparse.ArgumentParser(description="Batch Polling Client for the Cerebras Generation API")
    parser.add_argument("--dataset-repo", type=str, required=True, help="Hugging Face dataset repository ID.")
    parser.add_argument("--dataset-subset", type=str, help="Dataset subset, if any.")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to process.")
    parser.add_argument("--model", type=str, help="Optional: Specify a model name for all requests.")
    parser.add_argument("--max-concurrency", type=int, default=100, help="Maximum number of concurrent jobs being processed.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds to wait between polling for results.")
    parser.add_argument("--output-file", type=str, default="client_output.jsonl", help="File to save the final results.")
    args = parser.parse_args()

    logging.info(f"Loading {args.n_samples} samples from '{args.dataset_repo}'...")
    dataset = load_dataset(args.dataset_repo, args.dataset_subset, split=args.dataset_split, streaming=True)
    dataset_sample = list(dataset.take(args.n_samples))
    logging.info(f"Dataset loaded. Found {len(dataset_sample)} items.")

    semaphore = asyncio.Semaphore(args.max_concurrency)
    
    tasks = []
    # Shorter timeouts for faster failure detection
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for item in dataset_sample:
            task = asyncio.create_task(
                process_and_poll_item(client, item, semaphore, args.poll_interval, args.model)
            )
            tasks.append(task)

        logging.info(f"Processing {len(tasks)} items with max concurrency of {args.max_concurrency}...")
        
        completed_count = 0
        failed_count = 0
        
        with open(args.output_file, "w") as f:
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing items"):
                try:
                    result = await future
                    if result and result.get("status") == "completed":
                        f.write(json.dumps(result["data"]) + "\n")
                        completed_count += 1
                    else:
                        failed_count += 1
                        if result.get("reason") in ["server_unreachable", "polling_timeout"]:
                            logging.warning(f"Request failed: {result.get('reason')} for {result.get('request_id', 'unknown')}")
                except Exception as e:
                    failed_count += 1
                    logging.error(f"Task error: {e}")

    logging.info(f"\n--- Processing Complete ---")
    logging.info(f"Completed: {completed_count}, Failed: {failed_count}")
    logging.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import httpx
import argparse
import json
import time
import logging
from datasets import load_dataset
from tqdm.asyncio import tqdm

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def _send_and_poll(client: httpx.AsyncClient, payload: dict, poll_interval: float) -> dict:
    """Helper function to send a payload to /generate and poll for the result."""
    # 1. Send the initial generation request
    response = await client.post(f"{API_BASE_URL}/generate", json=payload)
    response.raise_for_status()

    if response.status_code != 202:
        return {"success": False, "error": "bad_initial_response", "details": response.text}

    request_id = response.json().get("request_id")

    # 2. Poll for the result
    while True:
        result_response = await client.get(f"{API_BASE_URL}/result/{request_id}")
        if result_response.status_code == 200:
            # The server returns a JSON with its own 'success' field. We return that whole dict.
            return result_response.json()
        elif result_response.status_code == 500:
            return {"success": False, "error": "server_processing_error", "details": result_response.json()}
        elif result_response.status_code == 202:
            await asyncio.sleep(poll_interval)
        else:
            return {"success": False, "error": "unexpected_polling_status", "details": result_response.text}


async def process_item(client: httpx.AsyncClient, item: dict, semaphore: asyncio.Semaphore, poll_interval: float, model: str | None) -> dict:
    """
    Processes a single multi-turn conversation item by making sequential API calls for each turn.
    """
    async with semaphore:
        try:
            if "conversations" not in item or not isinstance(item["conversations"], list):
                return {"status": "skipped", "reason": "invalid_item_format"}

            human_prompts = [turn["value"] for turn in item["conversations"] if turn.get("from") == "human"]
            if not human_prompts:
                return {"status": "skipped", "reason": "no_human_prompts"}

            conversation_history = []
            final_generated_conversation = []

            for prompt in human_prompts:
                # Add the human part of the turn
                conversation_history.append({"role": "user", "content": prompt})
                final_generated_conversation.append({"from": "human", "value": prompt})

                # Prepare payload for this specific turn
                request_payload = {"conversations": conversation_history}
                if model:
                    request_payload["model"] = model
                
                # Send the request for this turn and wait for the result
                result = await _send_and_poll(client, request_payload, poll_interval)

                if result and result.get("success"):
                    gpt_response = result.get("response", "")
                    # Add the assistant part of the turn
                    conversation_history.append({"role": "assistant", "content": gpt_response})
                    final_generated_conversation.append({"from": "gpt", "value": gpt_response})
                else:
                    # If any turn fails, the whole item fails.
                    error_details = result.get("error", "Unknown error")
                    return {"status": "failed", "reason": "turn_generation_failed", "details": error_details}
            
            # If all turns succeeded
            final_result_obj = {
                "id": item.get("id", "unknown"),
                "conversations": final_generated_conversation,
                "model": result.get("model") # Model from the last successful turn
            }
            return {"status": "completed", "data": final_result_obj}

        except httpx.RequestError as e:
            logging.error(f"HTTP request failed for an item: {e}")
            return {"status": "failed", "reason": "request_error", "details": str(e)}
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing an item: {e}", exc_info=True)
            return {"status": "failed", "reason": "unknown_client_error", "details": str(e)}


async def main():
    """
    A client script to stream a dataset and send concurrent requests to the generation API.
    """
    # ... (The main function and argument parsing remain the same)
    parser = argparse.ArgumentParser(description="Batch Client for the Cerebras Generation API")
    parser.add_argument("--dataset-repo", type=str, required=True, help="Hugging Face dataset repository ID (e.g., 'BAAI/Infinity-Instruct').")
    parser.add_argument("--dataset-subset", type=str, help="Dataset subset, if any (e.g., '7M_core').")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to process from the dataset.")
    parser.add_argument("--model", type=str, help="Optional: Specify a model name to use for all requests.")
    parser.add_argument("--max-concurrency", type=int, default=100, help="Maximum number of concurrent API requests.")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds to wait between polling for results.")
    args = parser.parse_args()

    # Load the dataset in streaming mode
    logging.info(f"Loading {args.n_samples} samples from '{args.dataset_repo}'...")
    dataset = load_dataset(args.dataset_repo, args.dataset_subset, split=args.dataset_split, streaming=True)
    dataset_sample = list(dataset.take(args.n_samples))
    logging.info("Dataset loaded.")

    semaphore = asyncio.Semaphore(args.max_concurrency)
    
    tasks = []
    async with httpx.AsyncClient(timeout=300.0) as client:
        for item in dataset_sample:
            task = asyncio.create_task(
                process_item(client, item, semaphore, args.poll_interval, args.model)
            )
            tasks.append(task)

        logging.info(f"Dispatching {len(tasks)} tasks with max concurrency of {args.max_concurrency}...")
        
        results = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing items"):
            result = await future
            results.append(result)

    # --- Process and print summary ---
    completed_count = sum(1 for r in results if r and r.get('status') == 'completed')
    failed_count = len(results) - completed_count
    
    logging.info("\n--- Processing Complete ---")
    logging.info(f"Total items processed: {len(results)}")
    logging.info(f"Successful: {completed_count}")
    logging.info(f"Failed/Skipped: {failed_count}")

    # Optionally, save the results to a file
    with open("client_results.jsonl", "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    logging.info("Results saved to client_results.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
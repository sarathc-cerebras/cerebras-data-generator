import asyncio
import time
import logging
import os
import json
import yaml
import random
from datasets import load_dataset
import tqdm.asyncio
from typing import Dict, Any, List, Optional

from httpx import ConnectError, ReadTimeout
from cerebras.cloud.sdk import (
    AsyncCerebras,
    APIStatusError
)

# --- Custom Exception for fatal rate limits ---
class DailyRateLimitError(Exception):
    """Raised when the daily API request limit is reached."""
    pass

# --- Concurrency Manager ---
class ConcurrencyManager:
    """Manages the dynamic concurrency level."""
    def __init__(self, initial_concurrency: int, min_concurrency: int = 20, recover_threshold: int = 10):
        self._max_concurrency = initial_concurrency
        self._concurrency = initial_concurrency
        self._min_concurrency = min_concurrency
        self._lock = asyncio.Lock()
        self.adjustment_needed = False
        self._fast_request_counter = 0
        # Increase concurrency after this many consistently fast requests
        self._increase_threshold = recover_threshold 

    @property
    def level(self) -> int:
        return self._concurrency

    async def reduce_concurrency(self):
        """Safely reduces the concurrency level by 1."""
        async with self._lock:
            if self._concurrency > self._min_concurrency:
                self._concurrency -= 1
                self.adjustment_needed = True
                self._fast_request_counter = 0 # Reset counter on reduction
                logging.warning(f"High TTFT detected. Reducing concurrency level to {self._concurrency}.")

    async def record_fast_request(self):
        """Records a fast request and increases concurrency if threshold is met."""
        async with self._lock:
            self._fast_request_counter += 1
            if self._fast_request_counter >= self._increase_threshold:
                if self._concurrency < self._max_concurrency:
                    self._concurrency += 1
                    self.adjustment_needed = True
                    logging.info(f"Consistently low TTFT. Increasing concurrency level to {self._concurrency}.")
                self._fast_request_counter = 0 # Reset counter after check/increase

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Request Logic ---
async def send_request(
    client: AsyncCerebras,
    payload: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    config: Dict[str, Any],
    concurrency_manager: ConcurrencyManager
) -> Dict[str, Any]:
    """Sends a single request using the Cerebras SDK, measures TTFT, and handles retries."""
    ttft_threshold = config['load_monitoring']['ttft_threshold']
    max_retries = config['concurrency']['max_retries']
    initial_retry_delay = config['concurrency']['initial_retry_delay']
    max_retry_delay = config['concurrency']['max_retry_delay']

    async with semaphore:
        retry_delay = initial_retry_delay
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                ttft = -1
                first_chunk_received = False
                full_response = ""

                response = await client.chat.completions.create(**payload)

                if payload.get("stream"):
                    async for chunk in response:
                        if not first_chunk_received:
                            ttft = time.time() - start_time
                            first_chunk_received = True
                            logging.info(f"TTFT for request (model: {payload['model']}): {ttft:.2f}s")
                            if ttft > ttft_threshold:
                                await concurrency_manager.reduce_concurrency()
                            else:
                                await concurrency_manager.record_fast_request()
                        if chunk.choices and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                else:
                    if response.choices and response.choices[0].message.content:
                        full_response = response.choices[0].message.content

                end_time = time.time()
                logging.info(f"Request successful for model {payload['model']}. Concurrency: {concurrency_manager.level()} Total time: {end_time - start_time:.2f}s")
                return {"success": True, "model": payload['model'], "response": full_response, "ttft": ttft}

            except (ConnectError, ReadTimeout) as e:
                logging.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay:.2f}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

            except APIStatusError as e:
                if e.status_code == 429:
                    remaining_requests_header = e.response.headers.get("x-ratelimit-remaining-requests-day")
                    try:
                        if remaining_requests_header is not None and float(remaining_requests_header) <= 0:
                            logging.error("Daily request limit reached. Halting execution.")
                            raise DailyRateLimitError("Daily request limit has been exhausted.")
                    except (ValueError, TypeError):
                        logging.warning("Could not parse daily rate limit remaining header.")

                    reset_tokens_header = e.response.headers.get("x-ratelimit-reset-tokens-minute")
                    wait_time = -1
                    try:
                        if reset_tokens_header:
                            wait_time = float(reset_tokens_header) + 1
                            logging.warning(f"Per-minute token limit reached. Waiting for {wait_time:.2f}s...")
                    except (ValueError, TypeError):
                        logging.warning("Could not parse token reset header.")
                        
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    else:
                        # Fallback to exponential backoff if header is missing/invalid
                        logging.warning(f"Rate limit hit. No reset header found. Retrying in {retry_delay:.2f}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, max_retry_delay)
                
                elif e.status_code == 503:
                    logging.warning(f"Service unavailable (503) (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay:.2f}s...")
                    await asyncio.sleep(retry_delay + random.uniform(0, 1)) # Add jitter
                    retry_delay = min(retry_delay * 2, max_retry_delay)
                else:
                    error_message = f"APIStatusError: {e.status_code} - {str(e)}"
                    logging.error(f"An unrecoverable API error occurred for model {payload['model']}: {error_message}")
                    return {"success": False, "model": payload['model'], "error": error_message}

            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                break
        return {"success": False, "model": payload['model'], "error": "Max retries exceeded"}

# --- Data and Environment Setup Functions ---
def get_dataset(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Loads and configures the dataset based on the provided configuration.
    Returns the list of dataset items to be processed.
    """
    dataset_config = config["dataset"]
    if dataset_config["repo_type"] == "hf":
        hf_config = dataset_config["hf"]
        dataset = load_dataset(hf_config["repo"], hf_config.get("subset"), split=hf_config["split"], num_proc=hf_config["num_proc"])
    else:
        local_config = dataset_config["local"]
        dataset = load_dataset(local_config["format"], data_files=local_config["data_files"], split=local_config["split"])
    
    if "take" in dataset_config:
        dataset = dataset.take(dataset_config["take"])
    if "rename_columns" in dataset_config:
        for old, new in dataset_config["rename_columns"]:
            dataset = dataset.rename_column(old, new)
    if "select_columns" in dataset_config:
        dataset = dataset.select_columns(dataset_config["select_columns"])
    
    return list(dataset)

def load_configuration(config_path: str) -> Optional[Dict[str, Any]]:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}")
        return None

def get_resume_offset(output_file: str) -> int:
    """Checks for an existing output file and returns the number of processed lines."""
    if not os.path.exists(output_file):
        return 0
    
    logging.info(f"Output file found at {output_file}. Checking for progress to resume.")
    try:
        with open(output_file, "r") as f:
            return sum(1 for _ in f)
    except Exception as e:
        logging.warning(f"Could not read checkpoint file at {output_file}: {e}. Starting from scratch.")
        return 0

async def process_single_item(
    item: Dict[str, Any],
    client: AsyncCerebras,
    semaphore: asyncio.Semaphore,
    config: Dict[str, Any],
    concurrency_manager: ConcurrencyManager
) -> Dict[str, Any]:
    """
    Processes a single multi-turn conversation item.
    It iterates through human prompts, generates responses, and builds the conversation history.
    """
    if "conversations" not in item or not isinstance(item["conversations"], list):
        logging.warning(f"Skipping item due to missing or invalid 'conversations' field: {item}")
        return {"success": False, "error": "Invalid item format"}

    human_prompts = [turn["value"] for turn in item["conversations"] if turn.get("from") == "human"]
    if not human_prompts:
        logging.warning(f"Skipping item as no 'human' turn was found: {item}")
        return {"success": False, "error": "No human prompts found"}

    generation_params = config.get('generation', {})
    model = config['models_to_run'][0] # Assuming one model for multi-turn for simplicity
    
    conversation_history = []
    final_generated_conversation = []

    for prompt in human_prompts:
        conversation_history.append({"role": "user", "content": prompt})
        final_generated_conversation.append({"from": "human", "value": prompt})

        payload = {"model": model, "messages": conversation_history, **generation_params}
        
        result = await send_request(client, payload, semaphore, config, concurrency_manager)

        if result and result.get("success"):
            gpt_response = result["response"]
            conversation_history.append({"role": "assistant", "content": gpt_response})
            final_generated_conversation.append({"from": "gpt", "value": gpt_response})
        else:
            error_msg = result.get('error', 'Unknown error during generation')
            logging.error(f"Failed to get response for a turn. Halting generation for this item. Error: {error_msg}")
            return {"success": False, "error": f"Generation failed for item: {error_msg}"}

    return {
        "success": True,
        "result": {
            "id": item.get("id", "unknown"),
            "conversations": final_generated_conversation,
            "label": item.get("label", "unknown"),
            "langdetect": item.get("langdetect", "unknown"),
            "source": item.get("source", "unknown"),
            "reward": item.get("reward", 0),
            "model": model
        }
    }


# --- Main Processing Loop ---
async def process_batches(
    original_data: List[Dict[str, Any]],
    config: Dict[str, Any],
    resume_offset: int
):
    """Initializes clients and runs the main inference loop over the data batches."""
    client = AsyncCerebras()
    concurrency_config = config['concurrency']
    concurrency_manager = ConcurrencyManager(
        initial_concurrency=concurrency_config['max_concurrent_requests'],
        min_concurrency=concurrency_config['min_concurrent_requests'],
        recover_threshold=concurrency_config['recover_threshold']
    )
    semaphore = asyncio.Semaphore(concurrency_manager.level)
    batch_size = config.get('dataset', {}).get('batch_size', 1000)
    output_file = f"{config['dataset']['output']}"

    if len(config.get('models_to_run', [])) > 1:
        logging.warning("Multi-turn conversation logic currently supports one model at a time. Using the first model specified.")

    total_successful = resume_offset
    total_failed = 0
    
    remaining_data = original_data[resume_offset:]

    with tqdm.asyncio.tqdm(total=len(original_data), desc="LLM multi-turn inference", initial=resume_offset) as pbar:
        for i in range(0, len(remaining_data), batch_size):
            if concurrency_manager.adjustment_needed:
                logging.info(f"Recreating semaphore with new concurrency: {concurrency_manager.level}")
                semaphore = asyncio.Semaphore(concurrency_manager.level)
                concurrency_manager.adjustment_needed = False
            
            batch_data = remaining_data[i:i + batch_size]
            
            tasks = [
                asyncio.create_task(process_single_item(item, client, semaphore, config, concurrency_manager))
                for item in batch_data
            ]
            
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                successful_generations = []
                failed_count = 0
                for res in batch_results:
                    if isinstance(res, Exception):
                        logging.error(f"Task failed with exception: {res}")
                        failed_count += 1
                        continue

                    if res and res.get("success"):
                        successful_generations.append(res["result"])
                    else:
                        failed_count += 1

                total_successful += len(successful_generations)
                total_failed += failed_count

                if successful_generations:
                    with open(output_file, "a") as f:
                        for item in successful_generations:
                            f.write(json.dumps(item) + "\n")
                pbar.update(len(batch_data))

            except DailyRateLimitError as e:
                logging.critical(f"Execution stopped due to fatal daily rate limit: {e}")
                break
            except Exception as e:
                logging.error(f"An unexpected error occurred during batch processing: {e}. Moving to next batch.")
                total_failed += len(tasks)
                pbar.update(len(batch_data))

    logging.info(f"Completed. Total Success: {total_successful}, Total Failed: {total_failed}")
    if total_successful > 0:
        logging.info(f"All results saved to {output_file}")

# --- Orchestrator ---
async def main():
    """Orchestrates the data generation process."""
    if not os.environ.get("CEREBRAS_API_KEY"):
        logging.error("API key not found. Please set the CEREBRAS_API_KEY environment variable.")
        return

    config = load_configuration("config/inference-config.yaml")
    if not config:
        return

    original_data = get_dataset(config)
    if not original_data:
        logging.error("No data could be loaded from the dataset. Please check the dataset format and configuration.")
        return
        
    logging.info(f"Loaded {len(original_data)} total items from the dataset.")

    output_file = f"{config['dataset']['output']}"
    
    resume_offset = get_resume_offset(output_file)
    
    if resume_offset > 0:
        logging.info(f"Resuming from checkpoint. Skipping {resume_offset} items already processed.")

    await process_batches(original_data, config, resume_offset)

if __name__ == "__main__":
    asyncio.run(main())
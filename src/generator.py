import asyncio
import time
import logging
import os
import json
import yaml
from datasets import load_dataset
import tqdm.asyncio
from typing import Dict, Any, List, Optional
from itertools import cycle

from httpx import ConnectError, ReadTimeout
from cerebras.cloud.sdk import (
    AsyncCerebras,
    APIStatusError
)

# --- Global State for shared resources ---
# This dictionary holds resources initialized once and shared across modules.
API_STATE: Dict[str, Any] = {
    "model_resources": {},
    "config": None,
    "client": None,
    "model_cycler": None,
}

# --- Custom Exception for fatal rate limits ---
class DailyRateLimitError(Exception):
    """Raised when the daily API request limit is reached."""
    pass

# --- Concurrency Manager ---
class ConcurrencyManager:
    """Manages the dynamic concurrency level for a specific model."""
    def __init__(self, model_name: str, initial_concurrency: int, min_concurrency: int, recover_threshold: int):
        self._model_name = model_name
        self._max_concurrency = initial_concurrency
        self._concurrency = initial_concurrency
        self._min_concurrency = min_concurrency
        self._lock = asyncio.Lock()
        self.adjustment_needed = False
        self._fast_request_counter = 0
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
                self._fast_request_counter = 0
                logging.warning(f"High TTFT for '{self._model_name}'. Reducing concurrency to {self._concurrency}.")

    async def record_fast_request(self):
        """Records a fast request and increases concurrency if threshold is met."""
        async with self._lock:
            self._fast_request_counter += 1
            if self._fast_request_counter >= self._increase_threshold:
                if self._concurrency < self._max_concurrency:
                    self._concurrency += 1
                    self.adjustment_needed = True
                    logging.warning(f"Low TTFT for '{self._model_name}'. Increasing concurrency to {self._concurrency}.")
                self._fast_request_counter = 0

# Logging Setup
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

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
                    await concurrency_manager.record_fast_request()
                    if response.choices and response.choices[0].message.content:
                        full_response = response.choices[0].message.content

                end_time = time.time()
                logging.info(f"Request successful for model {payload['model']}. Concurrency: {concurrency_manager.level}. Total time: {end_time - start_time:.2f}s")
                return {"success": True, "model": payload['model'], "response": full_response, "ttft": ttft}

            except (ConnectError, ReadTimeout) as e:
                logging.warning(f"Request failed for {payload['model']} (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                await concurrency_manager.reduce_concurrency()
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

            except APIStatusError as e:
                if e.status_code == 401:
                    logging.error("Authentication error: Check your API key.")
                    break
                await concurrency_manager.reduce_concurrency()
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
                        logging.warning(f"Rate limit hit. No reset header found. Retrying in {retry_delay:.2f}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, max_retry_delay)
                
                elif e.status_code == 503:
                    logging.warning(f"Service unavailable for {payload['model']} (503). Retrying in {retry_delay:.2f}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry_delay)
                else:
                    error_message = f"APIStatusError: {e.status_code} - {str(e)}"
                    logging.error(f"An unrecoverable API error occurred for model {payload['model']}: {error_message}")
                    return {"success": False, "model": payload['model'], "error": error_message}

            except Exception as e:
                logging.error(f"An unexpected error occurred in send_request for {payload['model']}: {e}")
                break
        return {"success": False, "model": payload['model'], "error": "Max retries exceeded"}

# --- Data and Environment Setup Functions ---
def get_dataset(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Loads and configures the dataset from Hugging Face or local files."""
    dataset_config = config["dataset"]
    if dataset_config["repo_type"] == "hf":
        hf_config = dataset_config["hf"]
        streaming = hf_config.get("streaming", False)
        if streaming:
            logging.info(f"Loading dataset from Hugging Face '{hf_config['repo']}/{hf_config['subset']}' in streaming mode.")
            dataset = load_dataset(hf_config["repo"], hf_config["subset"], split=hf_config["split"], streaming=streaming)
        else:
            logging.info(f"Loading dataset from Hugging Face '{hf_config['repo']}/{hf_config['subset']}' in non-streaming mode.")
            dataset = load_dataset(hf_config["repo"], hf_config["subset"], split=hf_config["split"], num_proc=hf_config.get("num_proc", 1))
    else:
        local_config = dataset_config["local"]
        dataset = load_dataset(local_config["format"], data_files=local_config["data_files"], split=local_config["split"])
    
    if "n_samples" in dataset_config:
        dataset = dataset.take(dataset_config["n_samples"])
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

def get_processed_ids(output_file: str) -> set:
    """
    Reads the output file and returns a set of IDs for all items that
    have already been processed successfully.
    """
    if not os.path.exists(output_file):
        return set()
    
    logging.info(f"Output file found at {output_file}. Reading processed item IDs to resume.")
    processed_ids = set()
    try:
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        processed_ids.add(data["id"])
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed line in output file: {line.strip()}")
        logging.info(f"Found {len(processed_ids)} already processed items.")
        return processed_ids
    except Exception as e:
        logging.warning(f"Could not read checkpoint file at {output_file}: {e}. Starting from scratch.")
        return set()

def initialize_shared_resources(config: Dict[str, Any]):
    """Initializes resources shared by all modes (client, models, etc.)."""
    API_STATE["config"] = config
    
    # Increase the timeout for the HTTP client to prevent premature timeouts.
    # 300 seconds (5 minutes) is a safe value for long-running generation tasks.
    request_timeout = config.get('load_monitoring', {}).get('request_timeout', 300.0)
    API_STATE["client"] = AsyncCerebras(timeout=request_timeout)

    models_config = config.get('models_to_run', [])
    if not models_config:
        raise ValueError("`models_to_run` is empty in config. No models to run.")

    model_resources = {}
    for model_conf in models_config:
        model_name = model_conf['name']
        concurrency_settings = {**config.get('concurrency', {}), **model_conf.get('concurrency', {})}
        manager = ConcurrencyManager(
            model_name=model_name,
            initial_concurrency=concurrency_settings['max_concurrent_requests'],
            min_concurrency=concurrency_settings['min_concurrent_requests'],
            recover_threshold=concurrency_settings['recover_threshold']
        )
        semaphore = asyncio.Semaphore(manager._max_concurrency)
        # Store the full model config along with the manager and semaphore
        model_resources[model_name] = {
            "manager": manager,
            "semaphore": semaphore,
            "config": model_conf
        }
        logging.info(f"Initialized model '{model_name}' with max_concurrency: {manager._max_concurrency}")

    API_STATE["model_resources"] = model_resources
    API_STATE["model_cycler"] = cycle(model_resources.items())

async def process_single_item(
    item: Dict[str, Any],
    client: AsyncCerebras,
    semaphore: asyncio.Semaphore,
    global_config: Dict[str, Any],
    concurrency_manager: ConcurrencyManager,
    model_name: str,
    model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Processes a single multi-turn conversation item for a specific model."""
    if "conversations" not in item or not isinstance(item["conversations"], list):
        logging.warning(f"Skipping item due to missing or invalid 'conversations' field: {item}")
        return {"success": False, "error": "Invalid item format"}

    # Start with global generation settings and override with model-specific ones
    generation_params = global_config.get('generation', {}).copy()
    model_generation_config = model_config.get('generation', {})
    generation_params.update(model_generation_config)

    conversation_history = []
    final_generated_conversation = []

    human_prompts = [turn["value"] for turn in item["conversations"] if turn.get("from") == "human"]
    if not human_prompts:
        logging.warning(f"Skipping item as no 'human' turn was found: {item}")
        return {"success": False, "error": "No human prompts found"}

    for prompt in human_prompts:
        conversation_history.append({"role": "user", "content": prompt})
        final_generated_conversation.append({"from": "human", "value": prompt})

        payload = {"model": model_name, "messages": conversation_history, **generation_params}
        result = await send_request(client, payload, semaphore, global_config, concurrency_manager)

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
            "model": model_name
        }
    }

# --- Main Batch Processing Loop ---
async def process_continuously(
    data_to_process: List[Dict[str, Any]],
    config: Dict[str, Any]
):
    """
    Runs the main inference loop for batch processing using a bounded task buffer
    to manage memory for very large datasets.
    """
    client = API_STATE["client"]
    output_file = f"{config['dataset']['output']}"
    model_resources = API_STATE["model_resources"]

    if not data_to_process:
        logging.info("All items from the dataset have already been processed.")
        return

    # Define a sensible buffer for tasks. This prevents creating millions of tasks at once.
    # A multiple of the total max concurrency is a good heuristic.
    total_max_concurrency = sum(res["semaphore"]._value for res in model_resources.values())
    task_buffer_size = total_max_concurrency * 10  # e.g., if total concurrency is 100, buffer is 400.

    logging.info(f"Starting batch processing for {len(data_to_process)} items with a task buffer of {task_buffer_size}.")

    data_iterator = iter(data_to_process)
    model_cycler = cycle(model_resources.items())
    tasks = set()

    # Open the file in "append" mode.
    with open(output_file, "a") as f, tqdm.asyncio.tqdm(total=len(data_to_process), desc="Batch Inference") as pbar:
        while True:
            # Create new tasks until the buffer is full or the data is exhausted.
            while len(tasks) < task_buffer_size:
                try:
                    item = next(data_iterator)
                    model_name, resources = next(model_cycler)
                    task = asyncio.create_task(process_single_item(
                        item, client, resources["semaphore"], config, resources["manager"], model_name, resources["config"]
                    ))
                    tasks.add(task)
                except StopIteration:
                    # No more items to process.
                    break
            
            if not tasks:
                # All tasks are created and have completed.
                break

            # Wait for the next task in the buffer to complete.
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for future in done:
                res = await future
                if res and res.get("success"):
                    f.write(json.dumps(res["result"]) + "\n")
                pbar.update(1)
            
            # The set of active tasks is now the set of pending tasks.
            tasks = pending

# --- Orchestrator for Batch Mode ---
async def main():
    """Orchestrates the data generation process for batch mode."""
    if not os.environ.get("CEREBRAS_API_KEY"):
        logging.error("API key not found. Please set the CEREBRAS_API_KEY environment variable.")
        return

    config = load_configuration("config/inference-config.yaml")
    if not config:
        return

    initialize_shared_resources(config)

    # Ensure the output directory exists, only if a directory path is specified
    output_file = config['dataset']['output']
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Get IDs of items that are already processed to support resuming.
    processed_ids = get_processed_ids(output_file)

    original_data = get_dataset(config)
    if not original_data:
        logging.error("No data could be loaded from the dataset. Halting.")
        return
    
    # Filter out items that have already been processed.
    # This assumes each item in your dataset has a unique "id" field.
    if processed_ids:
        logging.info(f"Found {len(processed_ids)} already processed items. Filtering them out.")
        remaining_data = [item for item in original_data if item.get("id") not in processed_ids]
    else:
        remaining_data = original_data
        
    logging.info(f"Loaded {len(original_data)} total items from the dataset. {len(remaining_data)} items remain to be processed.")

    if remaining_data:
        await process_continuously(remaining_data, config)
    else:
        logging.info("No items remain to be processed. Exiting.")

if __name__ == "__main__":
    # This block only runs when you execute `python src/generator.py`
    logging.info("Running in BATCH processing mode.")
    asyncio.run(main())
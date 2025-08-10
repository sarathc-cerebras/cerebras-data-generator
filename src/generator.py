import asyncio
import time
import logging
import os
import json
import yaml
from datasets import load_dataset
import tqdm.asyncio
from typing import Dict, Any, List, Tuple, Optional

from httpx import ConnectError, ReadTimeout
from cerebras.cloud.sdk import (
    AsyncCerebras,
    APIStatusError
)

# --- Custom Exception for fatal rate limits ---
class DailyRateLimitError(Exception):
    """Raised when the daily API request limit is reached."""
    pass

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Request Logic ---
async def send_request(
    client: AsyncCerebras,
    payload: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    config: Dict[str, Any]
):
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
                                logging.warning(f"High TTFT detected: {ttft:.2f}s. Consider reducing concurrency.")
                        if chunk.choices and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                else:
                    if response.choices and response.choices[0].message.content:
                        full_response = response.choices[0].message.content

                end_time = time.time()
                logging.info(f"Request successful for model {payload['model']}. Total time: {end_time - start_time:.2f}s")
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
                        else:
                            logging.warning(f"Rate limit hit. No reset header found. Retrying in {wait_time:.2f}s...")
                    except (ValueError, TypeError):
                        logging.warning(f"Could not parse token reset header. Retrying in {wait_time:.2f}s...")
                        
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    else:
                        # Fallback to exponential backoff if header is missing/invalid
                        logging.warning(f"Rate limit hit. No reset header found. Retrying in {retry_delay:.2f}s...")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, max_retry_delay)
                
                elif e.status_code == 503:
                    logging.warning(f"Service unavailable (503) (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay:.2f}s...")
                    await asyncio.sleep(retry_delay)
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
def get_dataset(config: Dict[str, Any]) -> Any:
    """Loads and configures the dataset based on the provided configuration."""
    dataset_config = config["dataset"]
    if dataset_config["repo_type"] == "hf":
        hf_config = dataset_config["hf"]
        dataset = load_dataset(hf_config["repo"], hf_config.get("subset"), split=hf_config["split"])
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
    return dataset

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
            # Simply count the lines. The interpretation happens in main().
            return sum(1 for _ in f)
    except Exception as e:
        logging.warning(f"Could not read checkpoint file at {output_file}: {e}. Starting from scratch.")
        return 0

# --- Main Processing Loop ---
async def process_batches(
    prompts: List[str],
    config: Dict[str, Any],
    resume_offset: int
):
    """Initializes clients and runs the main inference loop over the data batches."""
    client = AsyncCerebras()
    semaphore = asyncio.Semaphore(config['concurrency']['max_concurrent_requests'])
    generation_params = config.get('generation', {})
    batch_size = config.get('dataset', {}).get('batch_size', 1000)
    output_file = f"../{config['dataset']['output']}"

    total_successful = resume_offset
    total_failed = 0
    remaining_prompts = prompts[resume_offset:]

    with tqdm.asyncio.tqdm(total=len(prompts), desc="LLM batch inference", initial=resume_offset) as pbar:
        for i in range(0, len(remaining_prompts), batch_size):
            batch_prompts = remaining_prompts[i:i + batch_size]
            tasks = []
            for model in config['models_to_run']:
                for prompt_content in batch_prompts:
                    payload = {"model": model, "messages": [{"role": "user", "content": prompt_content}], **generation_params}
                    tasks.append(send_request(client, payload, semaphore, config))
            
            try:
                batch_results = await asyncio.gather(*tasks)
                successful_generations = [r for r in batch_results if r and r.get("success")]
                failed_generations = [r for r in batch_results if not r or not r.get("success")]

                total_successful += len(successful_generations)
                total_failed += len(failed_generations)

                if successful_generations:
                    with open(output_file, "a") as f:
                        for res in successful_generations:
                            f.write(json.dumps(res) + "\n")
                pbar.update(len(batch_prompts))

            except DailyRateLimitError as e:
                logging.critical(f"Execution stopped due to fatal daily rate limit: {e}")
                break
            except Exception as e:
                logging.error(f"An unexpected error occurred during batch processing: {e}. Moving to next batch.")
                total_failed += len(tasks)
                pbar.update(len(batch_prompts))

    logging.info(f"Completed. Total Success: {total_successful}, Total Failed: {total_failed}")
    if total_successful > 0:
        logging.info(f"All results saved to {output_file}")

# --- Orchestrator ---
async def main():
    """Orchestrates the data generation process."""
    if not os.environ.get("CEREBRAS_API_KEY"):
        logging.error("API key not found. Please set the CEREBRAS_API_KEY environment variable.")
        return

    config = load_configuration("../config/inference-config.yaml")
    if not config:
        return

    prompts = get_dataset(config).to_pandas()["prompt"].tolist()
    logging.info(f"Loaded {len(prompts)} total prompts from the dataset.")

    output_file = f"../{config['dataset']['output']}"
    
    processed_lines = get_resume_offset(output_file)
    num_models = len(config.get('models_to_run', [1]))
    if num_models == 0:
        num_models = 1 # Avoid division by zero

    # Ensure we only resume after a complete batch of models for a prompt
    if processed_lines % num_models != 0:
        logging.warning(
            f"Output file contains a partial result for a prompt ({processed_lines} lines for {num_models} models). "
            "Resuming from the last fully completed prompt."
        )
    
    # Calculate the number of prompts to skip
    resume_offset_prompts = processed_lines // num_models
    
    if resume_offset_prompts > 0:
        logging.info(f"Resuming from checkpoint. {resume_offset_prompts} prompts already processed.")

    await process_batches(prompts, config, resume_offset_prompts)

if __name__ == "__main__":
    asyncio.run(main())
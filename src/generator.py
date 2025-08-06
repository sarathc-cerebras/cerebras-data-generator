import asyncio
import time
import logging
import os
import json
import yaml
from datasets import load_dataset
import tqdm.asyncio
from typing import Dict, Any
from httpx import ConnectError, ReadTimeout
from cerebras.cloud.sdk import (
    AsyncCerebras,
    APIStatusError
)

# --- Configuration is now loaded from config.yaml ---

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

                # Use the new async client's method for chat completion, unpacking all payload parameters
                response = await client.chat.completions.create(**payload)

                # Handle streaming response for TTFT measurement
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
                    # Handle non-streaming response
                    if response.choices and response.choices[0].message.content:
                        full_response = response.choices[0].message.content


                end_time = time.time()
                logging.info(f"Request successful for model {payload['model']}. Total time: {end_time - start_time:.2f}s")
                return {"success": True, "model": payload['model'], "response": full_response, "ttft": ttft}

            # Catch connection errors and timeouts from the underlying httpx library
            except (ConnectError, ReadTimeout) as e:
                logging.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay:.2f}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)  # Exponential backoff

            except APIStatusError as e:
                # Handle specific API status codes for retries, like 429 (rate limits) and 503 (availability)
                if e.status_code in [429, 503]:
                    logging.warning(f"Retryable API error {e.status_code} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay:.2f}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_retry_delay)
                else:
                    logging.error(f"An unrecoverable API error occurred for model {payload['model']}: {e.status_code} - {e.response}")
                    return {"success": False, "model": payload['model'], "error": str(e)}

            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                break  # Do not retry on unexpected errors

        return {"success": False, "model": payload['model'], "error": "Max retries exceeded"}
    

def get_dataset(config):
    if config["dataset"]["repo_type"] == "hf":
        hf = config["dataset"]["hf"]
        dataset = load_dataset(hf["repo"], hf.get("subset"), split=hf["split"])
    else:
        local = config["dataset"]["local"]
        dataset = load_dataset(local["format"], data_files=local["data_files"], split=local["split"])
    if "take" in config["dataset"]:
        dataset = dataset.take(config["dataset"]["take"])
    if "rename_columns" in config["dataset"]:
        for old, new in config["dataset"]["rename_columns"]:
            dataset = dataset.rename_column(old, new)
    if "select_columns" in config["dataset"]:
        dataset = dataset.select_columns(config["dataset"]["select_columns"])
    return dataset


async def main():
    """Main function to generate data from a list of prompts."""
    # The SDK automatically uses the CEREBRAS_API_KEY environment variable.
    # We check for it here to provide a clear error message if it's not set.
    if not os.environ.get("CEREBRAS_API_KEY"):
        logging.error("API key not found. Please set the CEREBRAS_API_KEY environment variable.")
        return

    # Load configuration from YAML file
    try:
        with open("../config/inference-config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Config file not found at ../config/inference-config.yaml")
        return

    # Load prompts from the file specified in the config
    df = get_dataset(config).to_pandas()
    prompts = df["prompt"].tolist()
    
    logging.info(f"Loaded {len(prompts)} prompts from the dataset.")

    tasks = []
    semaphore = asyncio.Semaphore(config['concurrency']['max_concurrent_requests'])
    
    # Instantiate the new Async Cerebras Client
    client = AsyncCerebras()

    # Get generation parameters from config, default to an empty dict if not present
    generation_params = config.get('generation', {})

    for model in config['models_to_run']:
        for prompt_content in prompts:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt_content}],
                **generation_params  # Unpack generation params into the payload
            }
            tasks.append(send_request(client, payload, semaphore, config))
    
    results = await tqdm.asyncio.tqdm.gather(*tasks, desc="LLM batch inference", total=len(tasks))

    # Process results
    successful_generations = [r for r in results if r and r.get("success")]
    failed_generations = [r for r in results if not r or not r.get("success")]

    logging.info(f"Completed. Success: {len(successful_generations)}, Failed: {len(failed_generations)}")
    
    # Save successful results to the file specified in the config
    output_file = f"../{config['dataset']['output']}"
    with open(output_file, "w") as f:
        for res in successful_generations:
            f.write(json.dumps(res) + "\n")
    logging.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
import argparse
import json
import logging
from datasets import load_dataset
from api_client import AsyncSynthGenClient
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def generate_synthetic_data(dataset_sample, args):
    completed_count = 0
    failed_count = 0
    async with AsyncSynthGenClient() as client:
        with open(args.output_file, "w") as f:
            async for result in client.generate_batch(dataset_sample, args.max_concurrency, args.model):
                f.write(json.dumps(result) + "\n")
                completed_count += 1
            
            # The number of failed tasks is the total minus the completed ones
            failed_count = len(dataset_sample) - completed_count
            
    logging.info(f"\n--- Processing Complete ---")
    logging.info(f"Completed: {completed_count}, Failed: {failed_count}")
    logging.info(f"Results saved to {args.output_file}")

async def main():
    """
    Client to load a dataset and process it using AsyncSynthGenClient.
    """
    parser = argparse.ArgumentParser(description="Batch Client for the Cerebras Generation API")
    parser.add_argument("--dataset-repo", type=str, required=True, help="Hugging Face dataset repository ID.")
    parser.add_argument("--dataset-subset", type=str, help="Dataset subset, if any.")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to process.")
    parser.add_argument("--model", type=str, help="Optional: Specify a model name for all requests.")
    parser.add_argument("--max-concurrency", type=int, default=100, help="Maximum number of concurrent jobs being processed.")
    parser.add_argument("--output-file", type=str, default="client_output.jsonl", help="File to save the final results.")
    args = parser.parse_args()

    logging.info(f"Loading {args.n_samples} samples from '{args.dataset_repo}'...")
    dataset = load_dataset(args.dataset_repo, args.dataset_subset, split=args.dataset_split, streaming=True)
    dataset_sample = list(dataset.take(args.n_samples))
    logging.info(f"Dataset loaded. Found {len(dataset_sample)} items.")

    await generate_synthetic_data(dataset_sample, args)
if __name__ == "__main__":
    asyncio.run(main())
import argparse
import json
import logging
from datasets import load_dataset
from api_client import AsyncSynthGenClient
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def format_dataset_item(item, index):
    """
    Convert a dataset item into the format expected by the API.
    Assumes the dataset has a 'conversations' field or 'messages' field.
    """
    # Try to extract conversations from common fields
    conversations = None
    
    if "conversations" in item:
        conversations = item["conversations"]
    elif "messages" in item:
        conversations = item["messages"]
    elif "prompt" in item and "response" in item:
        # Single turn conversation
        conversations = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["response"]}
        ]
    elif "instruction" in item:
        # Instruction-following format
        conversations = [
            {"role": "user", "content": item["instruction"]},
        ]
        if "output" in item:
            conversations.append({"role": "assistant", "content": item["output"]})
    else:
        # Fallback: use the entire item as a single message
        logging.warning(f"Item {index} doesn't have standard conversation format, using raw content")
        conversations = [{"role": "user", "content": str(item)}]
    
    return {
        "conversations": conversations,
        "metadata": {
            "original_index": index,
            "source_fields": list(item.keys())
        }
    }


async def generate_synthetic_data(dataset_sample, args):
    completed_count = 0
    failed_count = 0
    
    # Format dataset items for API
    logging.info("Formatting dataset items for API...")
    formatted_items = [format_dataset_item(item, idx) for idx, item in enumerate(dataset_sample)]
    
    async with AsyncSynthGenClient() as client:
        with open(args.output_file, "w") as f:
            async for result in client.generate_batch(formatted_items, args.max_concurrency, args.model):
                f.write(json.dumps(result) + "\n")
                f.flush()  # Ensure data is written immediately
                completed_count += 1
                
                # Log progress every 100 completions
                if completed_count % 100 == 0:
                    logging.info(f"Progress: {completed_count}/{len(formatted_items)} completed")
            
            # The number of failed tasks is the total minus the completed ones
            failed_count = len(formatted_items) - completed_count
            
    logging.info(f"\n--- Processing Complete ---")
    logging.info(f"Completed: {completed_count}, Failed: {failed_count}")
    logging.info(f"Success rate: {(completed_count/len(formatted_items)*100):.2f}%")
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
import asyncio
import logging
from api_client import AsyncSynthGenClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    try:
        async with AsyncSynthGenClient() as client:
            task = client.generate(
                conversations=[{"from": "human", "value": "What is your name?"}], 
                model="qwen-3-32b",
                metadata={"id": "test-1", "model": "qwen-3-32b"}
            )
            results = await asyncio.gather(task, return_exceptions=True)
            print(results)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
# Cerebras Synthetic Data Generator

This project provides a high-performance, asynchronous client for generating synthetic data using the Cerebras Cloud Inference API. It is designed for high-throughput, offline workloads, with built-in mechanisms for load monitoring and respecting service rate limits.

## Features

-   **Asynchronous by Design**: Uses `asyncio` and the `cerebras-cloud-sdk` for high-concurrency batch inferencing.
-   **Dynamic Load Monitoring**: Tracks Time to First Token (TTFT) to infer server load and adjusts behavior accordingly.
-   **Robust Error Handling**: Implements an exponential backoff strategy for retrying requests on transient errors (e.g., `503 Service Unavailable`).
-   **Configuration Driven**: All parameters, including models, concurrency, and dataset sources, are managed via a `YAML` config file.
-   **Flexible Data Sourcing**: Loads prompts from local files or directly from Hugging Face datasets.
-   **Progress Tracking**: Displays a real-time progress bar using `tqdm`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/sarathc-cerebras/cerebras-data-generator.git
    cd cerebras-data-generator
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

All script behavior is controlled by `config/inference-config.yaml`. Below is a detailed explanation of the available options.

```yaml
# A list of Cerebras models to run inference against.
models_to_run:
  - "qwen-3-235b-a22b-instruct-2507"

# Concurrency and Rate Limiting settings
concurrency:
  max_concurrent_requests: 10  # Max number of parallel requests
  initial_retry_delay: 1.0     # Initial delay (in seconds) before a retry
  max_retry_delay: 60.0        # Maximum delay for exponential backoff
  max_retries: 5               # Max number of retries for a failed request

# Client-side load monitoring
load_monitoring:
  ttft_threshold: 5.0          # Warn if Time to First Token exceeds this (seconds)
  request_timeout: 120.0       # Timeout for a single API request

# Parameters passed directly to the Cerebras chat completion API
generation:
  stream: true                 # Set to true for TTFT monitoring
  max_completion_tokens: 20000 # Max tokens in the generated response
  temperature: 0.7             # Sampling temperature
  top_p: 0.8                   # Nucleus sampling probability
  seed: 1                      # Random seed for reproducibility

# Input dataset and output file configuration
dataset:
  repo_type: hf                # 'hf' for Hugging Face, 'local' for local files
  output: "data/output/gsm8k_generated.jsonl" # Path to save results

  # --- Hugging Face dataset options ---
  hf:
    repo: "openai/gsm8k"       # HF repository name
    subset: "main"             # HF dataset subset (optional)
    split: "test"              # Dataset split to use

  # --- Local file options ---
  local:
    format: "parquet"          # Format of the local file (e.g., "json", "parquet")
    data_files: "../data/input/gsm8k_test.parquet" # Path to local data file
    split: "test"

  # --- Dataset processing options ---
  take: 100                    # Limit to the first N samples from the dataset
  rename_columns:              # Rename columns to standardize
    - ["question", "prompt"]
  select_columns:              # Select only the columns you need
    - "prompt"
```

## Usage

1.  **Set your Cerebras API Key:**
    The script reads the API key from an environment variable.

    ```sh
    export CEREBRAS_API_KEY='your-api-key-here'
    ```

2.  **Run the generator:**
    Navigate to the `src` directory and run the script.

    ```sh
    cd src
    python generator.py
    ```

    A progress bar will appear in your terminal, showing the status of the inference tasks.

    ```
    LLM batch inference: 100%|██████████| 100/100 [00:58<00:00,  1.71it/s]
    ```

## Output

-   **Console Logs**: The script logs information about TTFT, retries, and final success/failure counts to the console.
-   **Generated Data**: Successful generations are saved to the file specified by the `dataset.output` path in your `inference-config.yaml`. Each line in the output file is a JSON object containing the model, the generated response, and the measured TTFT.

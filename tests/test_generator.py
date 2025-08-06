import pytest
import asyncio
import json
import sys
import os
import yaml
from unittest.mock import MagicMock, AsyncMock, patch

# Add the 'src' directory to the Python path to allow direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from generator import (
    send_request,
    load_configuration,
    get_resume_offset,
    main,
    DailyRateLimitError
)
from cerebras.cloud.sdk import AsyncCerebras, APIStatusError
from httpx import Request, Response

# --- Test Fixtures ---

@pytest.fixture
def mock_config():
    """Provides a default configuration for tests."""
    return {
        "load_monitoring": {"ttft_threshold": 5.0},
        "concurrency": {
            "max_retries": 3,
            "initial_retry_delay": 0.01,
            "max_retry_delay": 0.1,
        },
        "dataset": {
            "output": "test_output.jsonl",
            "batch_size": 10
        },
        "models_to_run": ["test-model-1"]
    }

@pytest.fixture
def mock_payload():
    """Provides a default request payload."""
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }

@pytest.fixture
def mock_semaphore():
    """Provides a semaphore for concurrency control."""
    return asyncio.Semaphore(1)

# --- Tests for Core Request Logic ---

@pytest.mark.asyncio
async def test_send_request_success_streaming(mock_config, mock_payload, mock_semaphore):
    """Tests a successful streaming request."""
    client = AsyncCerebras()
    async def stream_generator():
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello "))])
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="World"))])
    client.chat.completions.create = AsyncMock(return_value=stream_generator())
    
    result = await send_request(client, mock_payload, mock_semaphore, mock_config)

    assert result["success"] is True
    assert result["response"] == "Hello World"
    assert result["ttft"] > 0
    client.chat.completions.create.assert_awaited_once()

@pytest.mark.asyncio
async def test_send_request_raises_daily_rate_limit_error(mock_config, mock_payload, mock_semaphore):
    """Tests that DailyRateLimitError is raised when the daily limit is hit."""
    client = AsyncCerebras()
    mock_response = Response(
        status_code=429,
        headers={"x-ratelimit-remaining-requests-day": "0"},
        request=Request("POST", "http://mock.url")
    )
    client.chat.completions.create = AsyncMock(
        side_effect=APIStatusError("Daily limit hit", response=mock_response, body={})
    )

    with pytest.raises(DailyRateLimitError):
        await send_request(client, mock_payload, mock_semaphore, mock_config)

@pytest.mark.asyncio
async def test_send_request_handles_per_minute_limit(monkeypatch, mock_config, mock_payload, mock_semaphore):
    """Tests that the correct wait time is used for per-minute limits."""
    client = AsyncCerebras()
    mock_sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", mock_sleep)
    mock_response = Response(
        status_code=429,
        headers={"x-ratelimit-reset-tokens-minute": "0.5"},
        request=Request("POST", "http://mock.url")
    )
    successful_response = MagicMock(choices=[MagicMock(message=MagicMock(content="Success"))])
    client.chat.completions.create = AsyncMock(side_effect=[APIStatusError("Rate limit", response=mock_response, body={}), successful_response])
    mock_payload["stream"] = False

    await send_request(client, mock_payload, mock_semaphore, mock_config)
    
    # Assert that sleep was called with the duration from the header + 1s buffer
    mock_sleep.assert_awaited_with(pytest.approx(0.5 + 1))

@pytest.mark.asyncio
async def test_send_request_max_retries_exceeded(mock_config, mock_payload, mock_semaphore):
    """Tests that the function gives up after max_retries."""
    client = AsyncCerebras()
    mock_config["concurrency"]["max_retries"] = 2
    mock_response = Response(status_code=503, request=Request("POST", "http://mock.url"))
    client.chat.completions.create = AsyncMock(side_effect=APIStatusError("Unavailable", response=mock_response, body={}))

    result = await send_request(client, mock_payload, mock_semaphore, mock_config)

    assert result["success"] is False
    assert result["error"] == "Max retries exceeded"
    assert client.chat.completions.create.call_count == 2

# --- Tests for Helper Functions ---

def test_load_configuration(tmp_path):
    """Tests loading of a valid YAML config and handling of a missing file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "test_config.yaml"
    
    # Test valid config
    config_data = {"models_to_run": ["test-model"]}
    config_file.write_text(yaml.dump(config_data))
    assert load_configuration(config_file) == config_data

    # Test missing file
    config_file.unlink()
    assert load_configuration(config_file) is None

def test_get_resume_offset(tmp_path):
    """Tests the checkpoint resume logic."""
    output_file = tmp_path / "output.jsonl"

    # Test with no existing file
    assert get_resume_offset(output_file) == 0

    # Test with an existing file
    output_file.write_text('{"success": true}\n{"success": true}\n')
    assert get_resume_offset(output_file) == 2

# --- Integration Test for Main Orchestrator ---

@pytest.mark.asyncio
@patch('generator.process_batches')
@patch('generator.get_dataset')
@patch('generator.load_configuration')
async def test_main_orchestrator_resume_logic(mock_load_config, mock_get_dataset, mock_process_batches, monkeypatch):
    """
    Tests that the main function correctly calculates the prompt offset
    for resuming a job with multiple models.
    """
    # Mock environment and config
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    mock_config_data = {
        "models_to_run": ["model-a", "model-b"], # 2 models
        "dataset": {"output": "dummy.jsonl"}
    }
    mock_load_config.return_value = mock_config_data
    
    # Mock the chained calls to get the list of prompts correctly
    mock_get_dataset.return_value.to_pandas.return_value.__getitem__.return_value.tolist.return_value = ["p1", "p2", "p3", "p4", "p5"]

    # --- Test Scenario ---
    # Simulate an output file with 5 lines. With 2 models, this means 2 full prompts
    # have been processed (4 lines), and the 3rd prompt is partially done (1 line).
    # The script should resume from prompt index 2 (the 3rd prompt).
    with patch('generator.get_resume_offset', return_value=5) as mock_get_offset:
        await main()

    # Assert that process_batches was called with the correct prompt offset (2)
    # resume_offset_prompts = 5 // 2 = 2
    mock_process_batches.assert_called_once()
    # The arguments are positional, so we check the args tuple, not kwargs dict
    positional_args, keyword_args = mock_process_batches.call_args
    assert positional_args[2] == 2
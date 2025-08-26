import pytest
import asyncio
import json
import sys
import os
import yaml
from unittest.mock import MagicMock, AsyncMock, patch, call

# Add the 'src' directory to the Python path to allow direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from generator import (
    send_request,
    load_configuration,
    get_processed_ids,
    main,
    DailyRateLimitError,
    ConcurrencyManager,
    initialize_shared_resources,
    process_single_item,
    API_STATE
)
from cerebras.cloud.sdk import AsyncCerebras, APIStatusError
from httpx import Request, Response, ConnectError

# --- Test Fixtures ---

@pytest.fixture
def mock_config():
    """Provides a default configuration for tests, reflecting the new structure."""
    return {
        "models_to_run": [
            {
                "name": "test-model",
                "concurrency": {
                    "max_concurrent_requests": 5,
                    "min_concurrent_requests": 1,
                    "recover_threshold": 3
                }
            }
        ],
        "load_monitoring": {"ttft_threshold": 0.5},
        "concurrency": {
            "max_retries": 3,
            "initial_retry_delay": 0.01,
            "max_retry_delay": 0.1,
        },
        "dataset": {"output": "test_output.jsonl"},
        "generation": {"stream": True}
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
def mock_concurrency_manager():
    """Provides a mock ConcurrencyManager."""
    manager = ConcurrencyManager(
        model_name="test-model",
        initial_concurrency=5,
        min_concurrency=1,
        recover_threshold=3
    )
    manager.reduce_concurrency = AsyncMock()
    manager.record_fast_request = AsyncMock()
    return manager

@pytest.fixture
def mock_semaphore():
    """Provides a semaphore for concurrency control."""
    return asyncio.Semaphore(5)

# --- Tests for ConcurrencyManager ---

@pytest.mark.asyncio
async def test_concurrency_manager_reduce():
    """Tests that concurrency is reduced correctly."""
    manager = ConcurrencyManager("test", 5, 1, 10)
    assert manager.level == 5
    await manager.reduce_concurrency()
    assert manager.level == 4
    assert manager.adjustment_needed is True
    assert manager._fast_request_counter == 0

@pytest.mark.asyncio
async def test_concurrency_manager_stops_at_min():
    """Tests that concurrency reduction stops at the minimum level."""
    manager = ConcurrencyManager("test", 2, 1, 10)
    await manager.reduce_concurrency()
    assert manager.level == 1
    await manager.reduce_concurrency() # Should have no effect
    assert manager.level == 1

@pytest.mark.asyncio
async def test_concurrency_manager_recover():
    """Tests that concurrency recovers after enough fast requests."""
    manager = ConcurrencyManager("test", 5, 1, 3)
    manager._concurrency = 2 # Manually set to a lower level
    
    await manager.record_fast_request()
    await manager.record_fast_request()
    assert manager.level == 2 # Not yet at threshold
    
    await manager.record_fast_request() # This should trigger the increase
    assert manager.level == 3
    assert manager._fast_request_counter == 0 # Counter should reset

@pytest.mark.asyncio
async def test_concurrency_manager_not_exceed_max():
    """Tests that concurrency does not recover beyond the max level."""
    manager = ConcurrencyManager("test", 5, 1, 3)
    
    for _ in range(10): # More than enough to trigger
        await manager.record_fast_request()
        
    assert manager.level == 5 # Should not exceed initial max

# --- Tests for Core Request Logic ---

@pytest.mark.asyncio
async def test_send_request_high_ttft_reduces_concurrency(mock_config, mock_payload, mock_semaphore, mock_concurrency_manager):
    """Tests that high TTFT triggers a concurrency reduction."""
    client = AsyncCerebras()
    async def slow_stream_generator():
        await asyncio.sleep(0.6) # Exceeds the 0.5s threshold
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="..."))])
    client.chat.completions.create = AsyncMock(return_value=slow_stream_generator())
    
    await send_request(client, mock_payload, mock_semaphore, mock_config, mock_concurrency_manager)

    mock_concurrency_manager.reduce_concurrency.assert_awaited_once()
    mock_concurrency_manager.record_fast_request.assert_not_awaited()

@pytest.mark.asyncio
async def test_send_request_low_ttft_records_fast_request(mock_config, mock_payload, mock_semaphore, mock_concurrency_manager):
    """Tests that low TTFT is recorded as a fast request."""
    client = AsyncCerebras()
    async def fast_stream_generator():
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="..."))])
    client.chat.completions.create = AsyncMock(return_value=fast_stream_generator())
    
    await send_request(client, mock_payload, mock_semaphore, mock_config, mock_concurrency_manager)

    mock_concurrency_manager.record_fast_request.assert_awaited_once()
    mock_concurrency_manager.reduce_concurrency.assert_not_awaited()

@pytest.mark.asyncio
async def test_send_request_connect_error_reduces_concurrency(mock_config, mock_payload, mock_semaphore, mock_concurrency_manager):
    """Tests that ConnectError triggers a concurrency reduction and retry."""
    client = AsyncCerebras()
    mock_config["concurrency"]["max_retries"] = 1
    client.chat.completions.create = AsyncMock(side_effect=ConnectError("Connection failed"))

    result = await send_request(client, mock_payload, mock_semaphore, mock_config, mock_concurrency_manager)

    assert result["success"] is False
    mock_concurrency_manager.reduce_concurrency.assert_awaited_once()

# --- Tests for Helper Functions ---

def test_get_processed_ids(tmp_path):
    """Tests the logic for reading already processed item IDs from the output file."""
    output_file = tmp_path / "output.jsonl"

    # 1. Test with no existing file
    assert get_processed_ids(output_file) == set()

    # 2. Test with a valid file
    output_file.write_text(
        '{"id": "id_1", "data": "..."}\n'
        '{"id": "id_2", "data": "..."}\n'
        '{"id": "id_1", "data": "..."}\n' # Duplicate ID
    )
    assert get_processed_ids(output_file) == {"id_1", "id_2"}

    # 3. Test with malformed JSON
    output_file.write_text(
        '{"id": "id_3"}\n'
        'this is not json\n'
        '{"id": "id_4"}\n'
    )
    assert get_processed_ids(output_file) == {"id_3", "id_4"}

def test_initialize_shared_resources(mock_config):
    """Tests that shared resources are initialized correctly from config."""
    initialize_shared_resources(mock_config)
    
    assert "client" in API_STATE
    assert "model_resources" in API_STATE
    assert "test-model" in API_STATE["model_resources"]
    
    manager = API_STATE["model_resources"]["test-model"]["manager"]
    assert isinstance(manager, ConcurrencyManager)
    assert manager._max_concurrency == 5
    assert manager._min_concurrency == 1
    assert manager._increase_threshold == 3

    with pytest.raises(ValueError):
        initialize_shared_resources({"models_to_run": []})

# --- Integration Test for Main Orchestrator ---

@pytest.mark.asyncio
@patch('generator.process_continuously')
@patch('generator.get_dataset')
@patch('generator.get_processed_ids')
@patch('generator.initialize_shared_resources')
@patch('generator.load_configuration')
async def test_main_orchestrator_filters_processed_items(
    mock_load_config, mock_init_resources, mock_get_ids, mock_get_dataset, mock_process_continuously, monkeypatch
):
    """
    Tests that the main function correctly filters out already processed items
    before calling the processing loop.
    """
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
    mock_load_config.return_value = {"dataset": {"output": "dummy.jsonl"}}
    
    # Simulate a dataset with 4 items
    mock_dataset = [
        {"id": "id_1", "conversations": []},
        {"id": "id_2", "conversations": []},
        {"id": "id_3", "conversations": []},
        {"id": "id_4", "conversations": []},
    ]
    mock_get_dataset.return_value = mock_dataset
    
    # Simulate that two items have already been processed
    mock_get_ids.return_value = {"id_1", "id_4"}

    await main()

    # Assert that process_continuously is called with the correct remaining items
    mock_process_continuously.assert_awaited_once()
    call_args = mock_process_continuously.call_args[0]
    
    remaining_data = call_args[0]
    remaining_ids = {item['id'] for item in remaining_data}
    
    assert len(remaining_data) == 2
    assert remaining_ids == {"id_2", "id_3"}
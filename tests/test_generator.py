import pytest
import asyncio
import json
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Add the 'src' directory to the Python path to allow direct import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from generator import send_request
from cerebras.cloud.sdk import AsyncCerebras, APIStatusError
from httpx import Request, Response

# Pytest marker for async tests
pytestmark = pytest.mark.asyncio

# --- Test Fixtures ---

@pytest.fixture
def mock_config():
    """Provides a default configuration for tests."""
    return {
        "load_monitoring": {"ttft_threshold": 5.0},
        "concurrency": {
            "max_retries": 3,
            "initial_retry_delay": 0.01, # Use small delays for tests
            "max_retry_delay": 0.1,
        },
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

# --- Test Cases ---

async def test_send_request_success_streaming(mock_config, mock_payload, mock_semaphore):
    """Tests a successful streaming request."""
    client = AsyncCerebras()
    
    # Mock the SDK's response
    async def stream_generator():
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello "))])
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="World"))])

    client.chat.completions.create = AsyncMock(return_value=stream_generator())

    result = await send_request(client, mock_payload, mock_semaphore, mock_config)

    assert result["success"] is True
    assert result["response"] == "Hello World"
    assert result["ttft"] > 0
    client.chat.completions.create.assert_awaited_once()

async def test_send_request_rate_limit_with_header(mock_config, mock_payload, mock_semaphore):
    """Tests 429 rate limit handling with a 'retry-after' style header."""
    client = AsyncCerebras()
    
    # Simulate a 429 error first, then a success
    mock_response = Response(
        status_code=429,
        headers={"x-ratelimit-reset-tokens-minute": "0.02"},
        request=Request("POST", "http://mock.url")
    )
    
    # For a non-streaming call, the successful response is a simple object
    successful_response = MagicMock(choices=[MagicMock(message=MagicMock(content="Success"))])

    side_effects = [
        APIStatusError("Rate limit", response=mock_response, body={}),
        successful_response
    ]
    
    client.chat.completions.create = AsyncMock(side_effect=side_effects)
    mock_payload["stream"] = False # Simplify for this test

    result = await send_request(client, mock_payload, mock_semaphore, mock_config)

    assert result["success"] is True
    assert result["response"] == "Success"
    assert client.chat.completions.create.call_count == 2

async def test_send_request_service_unavailable_retry(mock_config, mock_payload, mock_semaphore):
    """Tests retry logic for 503 Service Unavailable errors."""
    client = AsyncCerebras()
    
    mock_response = Response(status_code=503, request=Request("POST", "http://mock.url"))
    
    # For a non-streaming call, the successful response is a simple object
    successful_response = MagicMock(choices=[MagicMock(message=MagicMock(content="Success"))])

    side_effects = [
        APIStatusError("Unavailable", response=mock_response, body={}),
        successful_response
    ]
    
    client.chat.completions.create = AsyncMock(side_effect=side_effects)
    mock_payload["stream"] = False

    result = await send_request(client, mock_payload, mock_semaphore, mock_config)

    assert result["success"] is True
    assert result["response"] == "Success"
    assert client.chat.completions.create.call_count == 2

async def test_send_request_max_retries_exceeded(mock_config, mock_payload, mock_semaphore):
    """Tests that the function gives up after max_retries."""
    client = AsyncCerebras()
    mock_config["concurrency"]["max_retries"] = 2
    
    mock_response = Response(status_code=503, request=Request("POST", "http://mock.url"))
    
    # Always raise the error
    client.chat.completions.create = AsyncMock(
        side_effect=APIStatusError("Unavailable", response=mock_response, body={})
    )

    result = await send_request(client, mock_payload, mock_semaphore, mock_config)

    assert result["success"] is False
    assert result["error"] == "Max retries exceeded"
    assert client.chat.completions.create.call_count == 2

async def test_send_request_unrecoverable_error(mock_config, mock_payload, mock_semaphore):
    """Tests that it fails immediately on unrecoverable errors like 400."""
    client = AsyncCerebras()
    
    mock_response = Response(status_code=400, request=Request("POST", "http://mock.url"))
    
    client.chat.completions.create = AsyncMock(
        side_effect=APIStatusError("Bad Request", response=mock_response, body={})
    )

    result = await send_request(client, mock_payload, mock_semaphore, mock_config)

    assert result["success"] is False
    assert "400" in result["error"]
    # Should only be called once, no retries
    client.chat.completions.create.assert_awaited_once()
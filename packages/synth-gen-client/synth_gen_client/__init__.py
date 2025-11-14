"""A high-performance client for the Cerebras Synthetic Data Generation API."""

from .api_client import (
    AsyncSynthGenClient,
    GenerationFailedError,
    PollingTimeoutError,
    ServerNotReachableError,
    AuthenticationError
)

__version__ = "0.1.0"
__all__ = [
    "AsyncSynthGenClient",
    "GenerationFailedError",
    "PollingTimeoutError",
    "ServerNotReachableError",
    "AuthenticationError"
]
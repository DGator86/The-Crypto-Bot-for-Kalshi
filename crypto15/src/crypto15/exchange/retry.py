"""Retry logic with exponential backoff for API calls."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 4
    base_delay: float = 2.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            OSError,
        )
    )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed)."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add ±25% jitter to prevent thundering herd
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that adds retry logic with exponential backoff.

    Args:
        config: Retry configuration. Uses defaults if not provided.
        on_retry: Optional callback called on each retry with (exception, attempt).

    Returns:
        Decorated function with retry logic.

    Example:
        @with_retry(RetryConfig(max_retries=3))
        def fetch_data():
            return api.get("/data")
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                            attempt + 1,
                            config.max_retries + 1,
                            func.__name__,
                            str(e),
                            delay
                        )

                        if on_retry:
                            on_retry(e, attempt)

                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            config.max_retries + 1,
                            func.__name__,
                            str(e)
                        )

            # All retries exhausted
            raise last_exception  # type: ignore

        return wrapper

    return decorator


class RetryableError(Exception):
    """Base class for errors that should trigger retry logic."""
    pass


class RateLimitError(RetryableError):
    """Raised when API rate limit is hit."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class NetworkError(RetryableError):
    """Raised on network-related failures."""
    pass


class APIError(Exception):
    """Base class for API errors that should not be retried."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class InsufficientFundsError(APIError):
    """Raised when account has insufficient funds."""
    pass


class OrderError(APIError):
    """Raised when order placement fails."""
    pass

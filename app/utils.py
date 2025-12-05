"""
Utility functions and decorators for the translation service.

This module provides reusable utilities for instrumentation, logging, and other
cross-cutting concerns.
"""
import functools
import logging
import time
from typing import Callable


def measure_latency(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time.

    This decorator automatically times the execution of any function and logs
    the latency in milliseconds. It's useful for performance monitoring and
    identifying bottlenecks.

    Usage:
        @measure_latency
        def my_function(arg1, arg2):
            # function implementation
            return result

    Args:
        func: The function to be timed

    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            logger.info(
                f"{func.__name__} completed",
                extra={
                    "latency_ms": round(latency_ms, 2),
                    "function": func.__name__
                }
            )

            return result

        except Exception as e:
            # Still log the latency even if the function fails
            latency_ms = (time.time() - start_time) * 1000
            logger.error(
                f"{func.__name__} failed after {round(latency_ms, 2)}ms",
                extra={
                    "latency_ms": round(latency_ms, 2),
                    "function": func.__name__,
                    "error": str(e)
                }
            )
            raise

    return wrapper

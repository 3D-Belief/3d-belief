from time import perf_counter
from typing import Callable, TypeVar, Tuple, Any
from functools import wraps
import asyncio

T = TypeVar("T")

def time_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> Tuple[T, float]:
    """
    Call `func(*args, **kwargs)` and return (result, elapsed_seconds).
    """
    start = perf_counter()
    result = func(*args, **kwargs)
    elapsed = perf_counter() - start
    return result, elapsed

def with_timing(func: Callable[..., T]) -> Callable[..., Tuple[T, float]]:
    """
    Decorator that makes the function return (result, elapsed_seconds).
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[T, float]:
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        return result, elapsed
    return wrapper

def with_timing_async(func: Callable[..., T]) -> Callable[..., Tuple[T, float]]:
    """
    Works for both sync and async callables. Returns (result, elapsed_seconds).
    """
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def awrapper(*args: Any, **kwargs: Any) -> Tuple[T, float]:
            start = perf_counter()
            result = await func(*args, **kwargs)
            elapsed = perf_counter() - start
            return result, elapsed
        return awrapper  # type: ignore[return-value]
    else:
        return with_timing(func)  # reuse sync wrapper
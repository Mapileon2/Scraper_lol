import asyncio
from functools import wraps
from typing import Callable, TypeVar, Any, Coroutine

T = TypeVar('T')

def run_async(coroutine: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """
    A decorator to run async functions in a synchronous context.
    
    Args:
        coroutine: The async function to wrap
        
    Returns:
        A synchronous function that runs the coroutine
    """
    @wraps(coroutine)
    def wrapper(*args, **kwargs) -> T:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # If there's already a running event loop, run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return loop.run_in_executor(pool, lambda: asyncio.run(coroutine(*args, **kwargs)))
        else:
            # Otherwise, run in the current event loop
            return loop.run_until_complete(coroutine(*args, **kwargs))
    return wrapper

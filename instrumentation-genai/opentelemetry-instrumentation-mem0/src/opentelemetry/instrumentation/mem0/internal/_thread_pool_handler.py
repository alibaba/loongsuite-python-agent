"""
Thread pool context propagation handler for Mem0 instrumentation.

This module provides handlers for propagating OpenTelemetry context through
ThreadPoolExecutor, which is necessary for maintaining proper span parent-child
relationships in concurrent operations.
"""

import logging
from typing import Any, Callable, Mapping, Tuple

from opentelemetry import context as context_api

logger = logging.getLogger(__name__)


class ThreadPoolContextPropagationHandler:
    """
    Handler for propagating OpenTelemetry context through ThreadPoolExecutor.
    
    This handler wraps ThreadPoolExecutor.submit to automatically propagate
    the current OpenTelemetry context to submitted tasks, ensuring proper
    span parent-child relationships in concurrent operations.
    """
    
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        """
        Wrap the submitted function with context propagation logic.

        Args:
            wrapped: The original ThreadPoolExecutor.submit method
            instance: The ThreadPoolExecutor instance
            args: Positional arguments passed to submit (args[0] is the function)
            kwargs: Keyword arguments passed to submit

        Returns:
            concurrent.futures.Future: Future object from the original submit method

        Examples:
            >>> # This handler is automatically used when ThreadPoolExecutor.submit is called
            >>> executor = ThreadPoolExecutor()
            >>> future = executor.submit(some_function, arg1, arg2)
            >>> # The handler ensures OpenTelemetry context is propagated to the worker thread
        """
        # Check if there's a function to submit
        if not args:
            return wrapped(*args, **kwargs)

        # Extract the function to be executed
        original_func = args[0]
        
        # Verify that the first argument is callable
        if not callable(original_func):
            return wrapped(*args, **kwargs)

        # Capture the current OpenTelemetry context
        # This includes the current active span and any other context values
        otel_context = context_api.get_current()

        def wrapped_func(*func_args: Any, **func_kwargs: Any) -> Any:
            """
            Wrapper function that restores OpenTelemetry context in the worker thread.

            This function is executed in the worker thread and ensures the captured
            context is properly attached before executing the original function.
            """
            token = None
            try:
                # Attach the captured context in the worker thread
                # This makes the parent span available to any instrumented code
                token = context_api.attach(otel_context)

                # Execute the original function with its arguments
                return original_func(*func_args, **func_kwargs)
            finally:
                # Always detach the context to prevent leaks
                # Even if the function raises an exception
                if token is not None:
                    context_api.detach(token)

        # Replace the original function with our wrapped version
        # Keep all other arguments unchanged
        new_args: Tuple[Callable[..., Any], ...] = (wrapped_func,) + args[1:]

        # Call the original submit method with the wrapped function
        return wrapped(*new_args, **kwargs)


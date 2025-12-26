"""
Langfuse tracing integration for FastAPI.

Provides request tracing and LLM observability.
"""

import os
import time
from typing import Optional, Callable
from functools import wraps

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..logger import get_logger

logger = get_logger(__name__)

# Langfuse client (lazy init)
_langfuse = None


def get_langfuse():
    """Get or create Langfuse client."""
    global _langfuse

    if _langfuse is not None:
        return _langfuse

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.warning("Langfuse keys not set, tracing disabled")
        return None

    try:
        from langfuse import Langfuse

        _langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info("Langfuse tracing initialized")
        return _langfuse

    except ImportError:
        logger.warning("langfuse package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to init Langfuse: {e}")
        return None


class TraceWrapper:
    """Wrapper for Langfuse trace objects to provide consistent API."""
    
    def __init__(self, observation):
        self._observation = observation
        self._closed = False
        self._context_manager = None  # Store context manager for cleanup
    
    def update(self, **kwargs):
        """Update the trace with new data."""
        if self._observation and not self._closed:
            try:
                self._observation.update(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to update trace: {e}")
    
    def generation(self, **kwargs):
        """Create a generation within this trace."""
        if not self._observation or self._closed:
            return None
        try:
            # In Langfuse 2.0+, use start_as_current_observation for generations
            if hasattr(self._observation, 'generation'):
                return self._observation.generation(**kwargs)
            # Try creating a nested observation
            elif hasattr(self._observation, 'start_as_current_observation'):
                return self._observation.start_as_current_observation(
                    as_type="generation", **kwargs
                )
        except Exception as e:
            logger.warning(f"Failed to create generation: {e}")
        return None
    
    def span(self, **kwargs):
        """Create a span within this trace."""
        if not self._observation or self._closed:
            return None
        try:
            # In Langfuse 2.0+, use start_as_current_observation for spans
            if hasattr(self._observation, 'span'):
                return self._observation.span(**kwargs)
            # Try creating a nested observation
            elif hasattr(self._observation, 'start_as_current_observation'):
                return self._observation.start_as_current_observation(
                    as_type="span", **kwargs
                )
        except Exception as e:
            logger.warning(f"Failed to create span: {e}")
        return None
    
    def score(self, **kwargs):
        """Add a score to this trace."""
        if not self._observation or self._closed:
            return None
        try:
            if hasattr(self._observation, 'score'):
                return self._observation.score(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to add score: {e}")
        return None
    
    def end(self, **kwargs):
        """End the trace."""
        if self._observation and not self._closed:
            try:
                if hasattr(self._observation, 'end'):
                    self._observation.end(**kwargs)
                elif hasattr(self._observation, 'update'):
                    self._observation.update(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to end trace: {e}")
            finally:
                # Clean up context manager if we have one (Langfuse 2.0+)
                if self._context_manager:
                    try:
                        self._context_manager.__exit__(None, None, None)
                    except Exception as e:
                        logger.warning(f"Failed to exit context manager: {e}")
                self._closed = True


def _create_trace(langfuse_client, **kwargs):
    """Create a trace using the correct Langfuse API.
    
    Supports both Langfuse 1.x (trace() method) and 2.0+ (start_as_current_observation).
    """
    # Try Langfuse 2.0+ API first (start_as_current_observation)
    if hasattr(langfuse_client, 'start_as_current_observation'):
        try:
            # Extract name and other parameters
            name = kwargs.get('name', 'trace')
            trace_id = kwargs.get('id')  # Note: id may not be supported in all versions
            input_data = kwargs.get('input')
            metadata = kwargs.get('metadata', {})
            
            # Create trace using new API
            # start_as_current_observation returns a context manager
            # Note: 'as_type' may not support "trace" in all versions, so we try "span" as fallback
            # Note: 'id' parameter may not be supported in all versions, so we build kwargs conditionally
            obs_kwargs = {
                "name": name,
                "input": input_data,
                "metadata": metadata,
            }
            
            # Only add id if it's provided
            if trace_id:
                obs_kwargs["id"] = trace_id
            
            # Try different as_type values (trace -> span -> None)
            # Langfuse may fallback to span if trace is not supported
            observation_cm = None
            for as_type_val in ["trace", "span", None]:
                try:
                    test_kwargs = obs_kwargs.copy()
                    if as_type_val:
                        test_kwargs["as_type"] = as_type_val
                    observation_cm = langfuse_client.start_as_current_observation(**test_kwargs)
                    break  # Success, exit loop
                except (TypeError, ValueError) as e:
                    if as_type_val is None:
                        # Last attempt failed, re-raise
                        raise
                    continue  # Try next as_type value
            # Get the observation object from the context manager
            # We'll manually manage the lifecycle
            observation_obj = observation_cm.__enter__()
            # Store the context manager so we can call __exit__ later
            wrapper = TraceWrapper(observation_obj)
            wrapper._context_manager = observation_cm
            return wrapper
        except Exception as e:
            logger.warning(f"Failed to create trace with Langfuse 2.0+ API: {e}")
            # Try fallback to 1.x API
    
    # Fallback to Langfuse 1.x API (trace() method)
    if hasattr(langfuse_client, 'trace') and callable(getattr(langfuse_client, 'trace', None)):
        try:
            trace = langfuse_client.trace(**kwargs)
            return TraceWrapper(trace)
        except Exception as e:
            logger.warning(f"Failed to create trace with Langfuse 1.x API: {e}")
    
    # If all else fails, return None to disable tracing gracefully
    logger.warning(
        "Could not create trace - Langfuse API may have changed. "
        "Tracing disabled for this request."
    )
    return None


class LangfuseMiddleware(BaseHTTPMiddleware):
    """Middleware to trace all API requests in Langfuse."""

    async def dispatch(self, request: Request, call_next):
        langfuse = get_langfuse()

        if not langfuse:
            return await call_next(request)

        # Create trace for this request using the correct API (supports both 1.x and 2.0+)
        trace = _create_trace(
            langfuse,
            name=f"{request.method} {request.url.path}",
            metadata={
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
            },
        )
        
        if trace is None:
            # If tracing fails, continue without it
            return await call_next(request)

        # Store trace in request state for use in endpoints
        request.state.langfuse_trace = trace

        start_time = time.time()

        try:
            response = await call_next(request)

            # End trace with success
            trace.update(
                output={"status_code": response.status_code},
                metadata={
                    "duration_ms": (time.time() - start_time) * 1000,
                    "status": "success" if response.status_code < 400 else "error",
                },
            )
            # End the trace (closes context manager if using Langfuse 2.0+)
            trace.end()

            return response

        except Exception as e:
            # End trace with error
            trace.update(
                output={"error": str(e)},
                level="ERROR",
                metadata={
                    "duration_ms": (time.time() - start_time) * 1000,
                    "status": "error",
                },
            )
            # End the trace (closes context manager if using Langfuse 2.0+)
            trace.end()
            raise


def trace_llm_call(name: str = "llm_call"):
    """Decorator to trace LLM calls."""

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            langfuse = get_langfuse()

            if not langfuse:
                return await func(*args, **kwargs)

            trace = _create_trace(langfuse, name=name)
            if trace is None:
                return await func(*args, **kwargs)
            generation = trace.generation(
                name=name,
                input=str(kwargs.get("prompt", args[0] if args else "")),
            )

            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                if generation:
                    generation.end(
                        output=str(result)[:1000],
                        metadata={"duration_ms": (time.time() - start_time) * 1000},
                    )
                trace.end()

                return result

            except Exception as e:
                if generation:
                    generation.end(output=str(e), level="ERROR")
                trace.end()
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            langfuse = get_langfuse()

            if not langfuse:
                return func(*args, **kwargs)

            trace = _create_trace(langfuse, name=name)
            if trace is None:
                return func(*args, **kwargs)
            generation = trace.generation(
                name=name,
                input=str(kwargs.get("prompt", args[0] if args else "")),
            )

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                if generation:
                    generation.end(
                        output=str(result)[:1000],
                        metadata={"duration_ms": (time.time() - start_time) * 1000},
                    )
                trace.end()

                return result

            except Exception as e:
                if generation:
                    generation.end(output=str(e), level="ERROR")
                trace.end()
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def create_rag_trace(question: str, trace_id: Optional[str] = None):
    """Create a trace for RAG pipeline execution."""
    langfuse = get_langfuse()

    if not langfuse:
        return None

    return _create_trace(
        langfuse,
        id=trace_id,
        name="rag_pipeline",
        input={"question": question},
    )


def log_retrieval(trace, query: str, results: list, duration_ms: float):
    """Log retrieval step to trace."""
    if not trace:
        return

    trace.span(
        name="retrieval",
        input={"query": query},
        output={"results_count": len(results)},
        metadata={"duration_ms": duration_ms},
    )


def log_generation(trace, prompt: str, response: str, model: str, duration_ms: float, tokens: dict = None):
    """Log LLM generation to trace."""
    if not trace:
        return

    trace.generation(
        name="llm_generation",
        model=model,
        input=prompt[:2000],
        output=response[:2000],
        metadata={"duration_ms": duration_ms},
        usage=tokens,
    )


def log_quality_score(trace, score: float, feedback: str):
    """Log quality evaluation to trace."""
    if not trace:
        return

    trace.score(
        name="context_quality",
        value=score,
        comment=feedback,
    )

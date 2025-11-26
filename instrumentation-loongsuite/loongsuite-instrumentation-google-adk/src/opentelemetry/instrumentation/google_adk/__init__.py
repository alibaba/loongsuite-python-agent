"""
OpenTelemetry Instrumentation for Google ADK.

This package provides OpenTelemetry instrumentation for Google Agent Development Kit (ADK)
applications, following the OpenTelemetry GenAI semantic conventions.

Usage:
    from opentelemetry.instrumentation.google_adk import GoogleAdkInstrumentor
    
    GoogleAdkInstrumentor().instrument()
"""

import logging
from typing import Collection

from opentelemetry import trace as trace_api, metrics as metrics_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv.schemas import Schemas
from wrapt import wrap_function_wrapper

from .internal._plugin import GoogleAdkObservabilityPlugin
from .version import __version__

_logger = logging.getLogger(__name__)


class GoogleAdkInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentor for Google ADK.
    
    This instrumentor automatically injects observability into Google ADK applications
    following OpenTelemetry GenAI semantic conventions.
    """
    
    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()
        self._plugin = None
        self._original_plugins = None

    def instrumentation_dependencies(self) -> Collection[str]:
        """
        Return the list of instrumentation dependencies.
        
        Returns:
            Collection of required packages
        """
        return ["google-adk >= 0.1.0"]

    def _instrument(self, **kwargs):
        """
        Instrument the Google ADK library.
        
        Args:
            **kwargs: Optional keyword arguments:
                - tracer_provider: Custom tracer provider
                - meter_provider: Custom meter provider
        """
        # Lazy import to avoid import errors when google-adk is not installed
        try:
            import google.adk.runners
        except ImportError:
            _logger.warning("google-adk not found, instrumentation will not be applied")
            return
            
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        
        # Get tracer and meter
        tracer = trace_api.get_tracer(
            __name__,
            __version__,
            tracer_provider,
            schema_url=Schemas.V1_28_0.value,
        )
        
        meter = metrics_api.get_meter(
            __name__,
            __version__,
            meter_provider,
            schema_url=Schemas.V1_28_0.value,
        )
        
        # Create and store the plugin instance
        self._plugin = GoogleAdkObservabilityPlugin(tracer, meter)
        
        # Wrap the Runner initialization to auto-inject our plugin
        try:
            wrap_function_wrapper(
                "google.adk.runners",
                "Runner.__init__",
                self._runner_init_wrapper
            )
            _logger.info("Google ADK instrumentation enabled")
        except Exception as e:
            _logger.exception(f"Failed to instrument Google ADK: {e}")

    def _uninstrument(self, **kwargs):
        """
        Uninstrument the Google ADK library.
        
        Args:
            **kwargs: Optional keyword arguments
        """
        try:
            # Unwrap the Runner initialization
            from google.adk.runners import Runner
            unwrap(Runner, "__init__")
            
            self._plugin = None
            _logger.info("Google ADK instrumentation disabled")
        except Exception as e:
            _logger.exception(f"Failed to uninstrument Google ADK: {e}")

    def _runner_init_wrapper(self, wrapped, instance, args, kwargs):
        """
        Wrapper for Runner.__init__ to auto-inject the observability plugin.
        
        Args:
            wrapped: Original wrapped function
            instance: Runner instance
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Result of the original function
        """
        # Get or create plugins list
        plugins = kwargs.get('plugins', [])
        if not isinstance(plugins, list):
            plugins = [plugins] if plugins else []
        
        # Add our plugin if not already present
        if self._plugin and self._plugin not in plugins:
            plugins.append(self._plugin)
            kwargs['plugins'] = plugins
            _logger.debug("Injected OpenTelemetry observability plugin into Runner")
        
        # Call the original __init__
        return wrapped(*args, **kwargs)


__all__ = ["GoogleAdkInstrumentor"]


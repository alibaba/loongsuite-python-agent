"""
OpenTelemetry Instrumentation for Google ADK.

This package provides OpenTelemetry instrumentation for Google Agent Development Kit (ADK)
applications, following the OpenTelemetry GenAI semantic conventions.

Usage:
    # Manual instrumentation
    from opentelemetry.instrumentation.google_adk import GoogleAdkInstrumentor
    GoogleAdkInstrumentor().instrument()
    
    # Auto instrumentation (via opentelemetry-instrument)
    # opentelemetry-instrument python your_app.py
"""

import logging
from typing import Collection, Optional

from opentelemetry import trace as trace_api, metrics as metrics_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.semconv.schemas import Schemas
from wrapt import wrap_function_wrapper

from .internal._plugin import GoogleAdkObservabilityPlugin
from .version import __version__

_logger = logging.getLogger(__name__)

# Module-level storage for the plugin instance
# This ensures the plugin persists across different instrumentor instances
# and supports both manual and auto instrumentation modes
_global_plugin: Optional[GoogleAdkObservabilityPlugin] = None


def _create_plugin_if_needed(tracer_provider=None, meter_provider=None):
    """
    Create or get the global plugin instance.
    
    This function ensures that only one plugin instance exists for the
    entire process, which is necessary for auto instrumentation to work
    correctly when the instrumentor may be instantiated multiple times.
    
    Args:
        tracer_provider: Optional tracer provider
        meter_provider: Optional meter provider
        
    Returns:
        GoogleAdkObservabilityPlugin instance
    """
    global _global_plugin
    
    if _global_plugin is None:
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
        
        _global_plugin = GoogleAdkObservabilityPlugin(tracer, meter)
        _logger.debug("Created global GoogleAdkObservabilityPlugin instance")
    
    return _global_plugin


def _runner_init_wrapper(wrapped, instance, args, kwargs):
    """
    Wrapper for Runner.__init__ to auto-inject the observability plugin.
    
    This is a module-level function (not a method) to avoid issues with
    instance state in auto instrumentation scenarios where the instrumentor
    may be instantiated multiple times.
    
    Args:
        wrapped: Original wrapped function
        instance: Runner instance
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Result of the original function
    """
    # Get or create the plugin
    plugin = _create_plugin_if_needed()
    
    if plugin:
        # Get or create plugins list
        plugins = kwargs.get('plugins', [])
        if not isinstance(plugins, list):
            plugins = [plugins] if plugins else []
        
        # Add our plugin if not already present
        if plugin not in plugins:
            plugins.append(plugin)
            kwargs['plugins'] = plugins
            _logger.debug("Injected OpenTelemetry observability plugin into Runner")
    
    # Call the original __init__
    return wrapped(*args, **kwargs)


class GoogleAdkInstrumentor(BaseInstrumentor):
    """
    OpenTelemetry instrumentor for Google ADK.
    
    This instrumentor automatically injects observability into Google ADK applications
    following OpenTelemetry GenAI semantic conventions.
    
    Supports both manual and auto instrumentation modes:
    - Manual: GoogleAdkInstrumentor().instrument()
    - Auto: opentelemetry-instrument python your_app.py
    """
    
    def __init__(self):
        """Initialize the instrumentor."""
        super().__init__()

    @property
    def _plugin(self):
        """
        Get the global plugin instance.
        
        This property provides backward compatibility with code that accesses
        self.instrumentor._plugin (e.g., in tests).
        
        Returns:
            The global plugin instance
        """
        return _global_plugin

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
        
        This method works in both manual and auto instrumentation modes by
        using a module-level global plugin instance that persists across
        multiple instrumentor instantiations.
        
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
        
        # Create or get the global plugin instance
        _create_plugin_if_needed(tracer_provider, meter_provider)
        
        # Wrap the Runner initialization to auto-inject our plugin
        try:
            wrap_function_wrapper(
                "google.adk.runners",
                "Runner.__init__",
                _runner_init_wrapper  # Use module-level function
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
        global _global_plugin
        
        try:
            # Unwrap the Runner initialization
            from google.adk.runners import Runner
            unwrap(Runner, "__init__")
            
            # Clear the global plugin
            _global_plugin = None
            
            _logger.info("Google ADK instrumentation disabled")
        except Exception as e:
            _logger.exception(f"Failed to uninstrument Google ADK: {e}")


__all__ = ["GoogleAdkInstrumentor"]

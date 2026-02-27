LoongSuite Distro
=================

LoongSuite Python Agent's Distro package, providing LoongSuite-specific configuration and tools.

Installation
------------

::

    pip install loongsuite-distro

Optional dependencies:

::

    # Install with baggage processor support
    pip install loongsuite-distro[baggage]
    
    # Install with OTLP exporter support
    pip install loongsuite-distro[otlp]
    
    # Install with both
    pip install loongsuite-distro[baggage,otlp]

Features
--------

1. **LoongSuite Distro**: Provides LoongSuite-specific OpenTelemetry configuration
2. **LoongSuite Bootstrap**: Install all LoongSuite components from tar package
3. **Baggage Processor**: Optional baggage span processor with prefix matching and stripping support

Usage
-----

### Configure LoongSuite Distro

Specify using LoongSuite Distro via environment variable::

    export OTEL_PYTHON_DISTRO=loongsuite

### Use LoongSuite Bootstrap

Install all components from tar package::

    loongsuite-bootstrap -t loongsuite-python-agent-1.0.0.tar.gz

Install from GitHub Releases::

    loongsuite-bootstrap -v 1.0.0

Install latest version::

    loongsuite-bootstrap --latest

### Configure Baggage Processor

The baggage processor is automatically loaded if configured via environment variables.
First, install the optional dependency::

    pip install loongsuite-distro[baggage]

Then configure via environment variables::

    export LOONGSUITE_PROCESSOR_BAGGAGE_ALLOWED_PREFIXES="traffic.,app."
    export LOONGSUITE_PROCESSOR_BAGGAGE_STRIP_PREFIXES="traffic."

The processor will only be loaded if ``LOONGSUITE_PROCESSOR_BAGGAGE_ALLOWED_PREFIXES`` is set.

For more usage, please refer to `LOONGSUITE_BOOTSTRAP_README.md`.

References
----------

* `LoongSuite Python Agent <https://github.com/alibaba/loongsuite-python-agent>`_



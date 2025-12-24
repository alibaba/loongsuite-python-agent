LoongSuite Distro
=================

LoongSuite Python Agent's Distro package, providing LoongSuite-specific configuration and tools.

Installation
------------

::

    pip install loongsuite-distro

Features
--------

1. **LoongSuite Distro**: Provides LoongSuite-specific OpenTelemetry configuration
2. **LoongSuite Bootstrap**: Install all LoongSuite components from tar package

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

For more usage, please refer to `LOONGSUITE_BOOTSTRAP_README.md`.

References
----------

* `LoongSuite Python Agent <https://github.com/alibaba/loongsuite-python-agent>`_



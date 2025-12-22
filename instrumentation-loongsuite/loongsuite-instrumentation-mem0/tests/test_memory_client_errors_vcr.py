# -*- coding: utf-8 -*-
"""
VCR test cases for MemoryClient HTTP layer exceptions:
- Goal is to construct connection errors during real recording and persist exceptions to cassette
- Subsequently playback mode can replay the exception, verifying top-level span ERROR marking and error.type attribute
"""

import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Tuple

import pytest


def _new_client_for_error(host: str):
    pytest.importorskip("mem0")
    MemoryClient = pytest.importorskip("mem0.client.main").MemoryClient  # type: ignore
    # Use placeholder API Key, VCR filters sensitive headers in conftest
    api_key = os.environ.get("MEM0_API_KEY", "test_mem0_api_key")
    return MemoryClient(api_key=api_key, host=host)


@pytest.mark.vcr()
def test_client_connection_error_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
):
    """
    Constructs connection error:
    - Uses unreachable port (e.g. 127.0.0.1:9) to cause connection failure
    - Expects MemoryClient top-level method to raise exception
    - Asserts top-level span is marked as ERROR and contains error.type
    Note: First time requires --record-mode=all in real environment to record exception; afterwards playback mode replays the exception.
    """
    from opentelemetry.trace import StatusCode  # noqa: PLC0415

    # Unreachable address (port 9 is typically not listened to, triggers connection refused)
    host = os.environ.get("MEM0_ERROR_HOST", "http://127.0.0.1:9")
    # Avoid triggering validation requests during __init__, ensure error occurs in wrapped method
    MemoryClient = pytest.importorskip("mem0.client.main").MemoryClient  # type: ignore
    Project = pytest.importorskip("mem0.client.project").Project  # type: ignore
    monkeypatch.setattr(
        MemoryClient, "_validate_api_key", lambda self: "test@example.com"
    )
    monkeypatch.setattr(Project, "_validate_org_project", lambda self: None)
    client = _new_client_for_error(host)

    # Choose simple GET method to trigger HTTP call (get_all)
    with pytest.raises(Exception):
        client.get_all(filters={"user_id": "u_conn_err"}, top_k=1)

    spans = span_exporter.get_finished_spans()
    # For MemoryClient top-level operations, instrumentation writes gen_ai.memory.operation in attributes
    error_spans = [
        s for s in spans if s.status.status_code == StatusCode.ERROR
    ]
    assert error_spans, (
        "Should generate at least one top-level span with ERROR status"
    )
    assert any("error.type" in s.attributes for s in error_spans), (
        "Error span should contain error.type"
    )


def _start_test_http_server(status_code: int) -> Tuple[HTTPServer, str]:
    """
    Starts a simple HTTP server that returns given status_code for all requests.
    Returns (server, base_url)
    """

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(status_code)
            self.end_headers()
            self.wfile.write(b"")

        def do_POST(self):  # noqa: N802
            self.send_response(status_code)
            self.end_headers()
            self.wfile.write(b"")

        def do_DELETE(self):  # noqa: N802
            self.send_response(status_code)
            self.end_headers()
            self.wfile.write(b"")

        def do_PUT(self):  # noqa: N802
            self.send_response(status_code)
            self.end_headers()
            self.wfile.write(b"")

        def log_message(self, fmt, *args):
            return

    # Bind to random available port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    addr, port = sock.getsockname()
    sock.close()

    server = HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


@pytest.mark.vcr()
def test_client_http_401_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
):
    """
    Returns 401 via local controllable HTTP service, records and replays exception.
    """
    from opentelemetry.trace import StatusCode  # noqa: PLC0415

    server, base_url = _start_test_http_server(401)
    try:
        MemoryClient = pytest.importorskip("mem0.client.main").MemoryClient  # type: ignore
        Project = pytest.importorskip("mem0.client.project").Project  # type: ignore
        monkeypatch.setattr(
            MemoryClient, "_validate_api_key", lambda self: "test@example.com"
        )
        monkeypatch.setattr(
            Project, "_validate_org_project", lambda self: None
        )
        client = _new_client_for_error(base_url)
        with pytest.raises(Exception):
            client.get_all(filters={"user_id": "u_401"}, top_k=1)
        spans = span_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        assert error_spans
        assert any("error.type" in s.attributes for s in error_spans)
    finally:
        try:
            server.shutdown()
        except Exception:
            # ignore shutdown exception
            pass


@pytest.mark.vcr()
def test_client_http_500_vcr(
    span_exporter: Any, instrument_with_content: Any, monkeypatch: Any
):
    """
    Returns 500 via local controllable HTTP service, records and replays exception.
    """
    from opentelemetry.trace import StatusCode  # noqa: PLC0415

    server, base_url = _start_test_http_server(500)
    try:
        MemoryClient = pytest.importorskip("mem0.client.main").MemoryClient  # type: ignore
        Project = pytest.importorskip("mem0.client.project").Project  # type: ignore
        monkeypatch.setattr(
            MemoryClient, "_validate_api_key", lambda self: "test@example.com"
        )
        monkeypatch.setattr(
            Project, "_validate_org_project", lambda self: None
        )
        client = _new_client_for_error(base_url)
        with pytest.raises(Exception):
            client.get_all(filters={"user_id": "u_500"}, top_k=1)
        spans = span_exporter.get_finished_spans()
        error_spans = [
            s for s in spans if s.status.status_code == StatusCode.ERROR
        ]
        assert error_spans
        assert any("error.type" in s.attributes for s in error_spans)
    finally:
        try:
            server.shutdown()
        except Exception:
            # ignore shutdown exception
            pass

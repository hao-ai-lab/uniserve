"""Behavioral media source tests at filesystem-free HTTP and base64 boundaries."""

import base64
import contextlib
import http.server
import socket
import threading
from dataclasses import replace

import pytest

from uniserve_worker.media.source import SourcePolicy, read_source


class MediaHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/media")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/escape":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.2/media")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        if self.path == "/chunked":
            self.send_header("Transfer-Encoding", "chunked")
        elif self.path == "/compressed":
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", "5")
        else:
            self.send_header("Content-Length", "5")
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(b"5\r\nmedia\r\n0\r\n\r\n" if self.path == "/chunked" else b"media")


@pytest.fixture
def media_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MediaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_inline_exact_byte_limit():
    policy = SourcePolicy(max_bytes=5)
    assert read_source("base64", base64.b64encode(b"media").decode(), policy=policy) == b"media"
    for encoded in ("", "%%%", "bWVkaWE=\n", base64.b64encode(b"123456").decode()):
        with pytest.raises(ValueError):
            read_source("base64", encoded, policy=policy)


@pytest.mark.parametrize("path", ["/media", "/chunked", "/redirect"])
def test_authorized_destination_and_streaming_limit(media_server, path):
    policy = SourcePolicy(max_bytes=5, allowed_networks=("127.0.0.1/32",))
    assert read_source("url", media_server + path, policy=policy) == b"media"
    with pytest.raises(ValueError, match="byte limit"):
        read_source("url", media_server + path, policy=replace(policy, max_bytes=4))


def test_private_destination_requires_explicit_network(media_server):
    with pytest.raises(ValueError, match="not authorized"):
        read_source("url", media_server + "/media")


def test_redirect_revalidates_destination(media_server):
    policy = SourcePolicy(allowed_networks=("127.0.0.1/32",))
    with pytest.raises(ValueError, match="not authorized"):
        read_source("url", media_server + "/escape", policy=policy)
    with pytest.raises(ValueError, match="redirect limit"):
        read_source("url", media_server + "/redirect", policy=replace(policy, max_redirects=0))


def test_connection_uses_the_validated_address(media_server, monkeypatch):
    port = int(media_server.rsplit(":", 1)[1])
    real_resolve = socket.getaddrinfo
    answers = iter(("127.0.0.1", "127.0.0.2"))

    def resolve(host, port, *args, **kwargs):
        if host == "media.example":
            host = next(answers)
        return real_resolve(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    assert (
        read_source(
            "url",
            f"http://media.example:{port}/media",
            policy=SourcePolicy(allowed_networks=("127.0.0.1/32",)),
        )
        == b"media"
    )


def test_mixed_dns_answers_do_not_authorize_private_addresses(media_server, monkeypatch):
    real_resolve = socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        return real_resolve("127.0.0.1", port, *args, **kwargs) + real_resolve(
            "127.0.0.2", port, *args, **kwargs
        )

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(ValueError, match="not authorized"):
        read_source(
            "url",
            media_server + "/media",
            policy=SourcePolicy(allowed_networks=("127.0.0.1/32",)),
        )


def test_transport_compression_is_not_a_size_limit_bypass(media_server):
    with pytest.raises(ValueError, match="compressed"):
        read_source(
            "url",
            media_server + "/compressed",
            policy=SourcePolicy(allowed_networks=("127.0.0.1/32",)),
        )


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/media",
        "http://user:password@localhost/media",
        "http://localhost/a\nb",
        "http://[::ffff:127.0.0.1]/",
        "http://169.254.169.254/",
        "http://[::1]/",
    ],
)
def test_unsafe_sources_are_rejected(url):
    with pytest.raises(ValueError):
        read_source("url", url)

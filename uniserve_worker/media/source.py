"""Bounded inline and HTTP media sources with destination-pinned connections."""

from __future__ import annotations

import base64
import binascii
import http.client
import ipaddress
import socket
import ssl
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    """Resource and network bounds for one encoded source.

    Private destinations require explicit IP/CIDR authorization, not a hostname
    exemption. The allowlist never exempts a source from byte or timeout limits.
    Environment proxy settings are intentionally not used. Socket operations time
    out individually; the deadline is also checked between body chunks. System
    DNS resolution is synchronous and is not covered by the socket timeout.
    """

    max_bytes: int = 32 * 1024 * 1024
    timeout_seconds: float = 30.0
    max_redirects: int = 3
    allowed_networks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_bytes < 1 or not 0 < self.timeout_seconds <= 300 or self.max_redirects < 0:
            raise ValueError("invalid media source limits")
        for network in self.allowed_networks:
            ipaddress.ip_network(network, strict=True)


def read_source(kind: str, value: str, *, policy: SourcePolicy = SourcePolicy()) -> bytes:
    """Read one strict base64 payload or bounded HTTP(S) resource.

    Redirects undergo the same address validation as the initial URL. Connections
    use the validated numeric address while TLS verifies the original hostname.
    Errors deliberately omit URLs, credentials, and response bodies.
    """

    if not isinstance(value, str) or not value:
        raise ValueError("media source must be nonempty")
    if kind == "base64":
        # Check before allocating the decoded bytes. A second check handles the
        # final base64 quartet's padding without relaxing the exact byte limit.
        if len(value) > 4 * ((policy.max_bytes + 2) // 3):
            raise ValueError("encoded media exceeds byte limit")
        try:
            result = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("invalid base64 media") from None
        if not result or len(result) > policy.max_bytes:
            raise ValueError("encoded media exceeds byte limit or is empty")
        return result
    if kind != "url":
        raise ValueError("unknown media source type")
    try:
        return _read_url(value, policy)
    except (OSError, http.client.HTTPException, UnicodeError):
        raise ValueError("media fetch failed") from None


def _destinations(host: str, port: int, policy: SourcePolicy) -> tuple[str, ...]:
    networks = tuple(ipaddress.ip_network(item) for item in policy.allowed_networks)
    addresses = tuple(
        dict.fromkeys(
            item[4][0] for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        )
    )
    if not addresses:
        raise ValueError("media host has no addresses")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        # Normalize mapped IPv4 so an IPv6 spelling cannot bypass IPv4 policy.
        normalized = getattr(ip, "ipv4_mapped", None) or ip
        allowed = any(ip in network or normalized in network for network in networks)
        if not allowed and (
            not normalized.is_global
            or normalized.is_multicast
            or normalized.is_reserved
            or normalized.is_unspecified
            or getattr(ip, "sixtofour", None) is not None
            or getattr(ip, "teredo", None) is not None
        ):
            raise ValueError("media destination is not authorized")
    return addresses


def _read_url(value: str, policy: SourcePolicy) -> bytes:
    deadline = time.monotonic() + policy.timeout_seconds
    for redirects in range(policy.max_redirects + 1):
        if len(value) > 8192 or any(ord(char) <= 32 or ord(char) == 127 for char in value):
            raise ValueError("invalid media URL")
        try:
            url = urlsplit(value)
            port = url.port or (443 if url.scheme == "https" else 80)
        except ValueError:
            raise ValueError("invalid media URL") from None
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.fragment
            or "%" in url.hostname
        ):
            raise ValueError("invalid media URL")
        host = url.hostname.encode("idna").decode("ascii")
        addresses = _destinations(host, port, policy)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("media fetch timed out")
        connection = http.client.HTTPConnection(host, port, timeout=remaining)
        response = None
        try:
            # Do not resolve the hostname a second time: DNS rebinding must not
            # move a checked public destination to an unchecked private address.
            connection.sock = socket.create_connection((addresses[0], port), timeout=remaining)
            if url.scheme == "https":
                connection.sock = ssl.create_default_context().wrap_socket(
                    connection.sock, server_hostname=host
                )
            path = url.path or "/"
            if url.query:
                path += "?" + url.query
            connection.request("GET", path, headers={"Accept-Encoding": "identity"})
            transport = connection.sock
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location or redirects == policy.max_redirects:
                    raise ValueError("invalid media redirect or redirect limit exceeded")
                value = urljoin(value, location)
                continue
            if response.status != 200:
                raise ValueError("media fetch returned unsuccessful status")
            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("compressed HTTP media response is not supported")
            lengths = response.headers.get_all("Content-Length", [])
            if lengths:
                if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                    raise ValueError("invalid media content length")
                if int(lengths[0]) > policy.max_bytes:
                    raise ValueError("encoded media exceeds byte limit")
            data = bytearray()
            while not response.isclosed():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("media fetch timed out")
                # Keep the socket reference even when HTTPConnection detaches it
                # for a Connection: close response (HTTPResponse still owns it).
                transport.settimeout(remaining)
                chunk = response.read1(min(64 * 1024, policy.max_bytes + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > policy.max_bytes:
                    raise ValueError("encoded media exceeds byte limit")
            if lengths and len(data) != int(lengths[0]):
                raise ValueError("truncated media response")
            if not data:
                raise ValueError("empty media response")
            return bytes(data)
        finally:
            if response is not None:
                response.close()
            connection.close()
    raise ValueError("media redirect limit exceeded")

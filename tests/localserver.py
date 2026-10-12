"""A real HTTP server on 127.0.0.1 for the few tests that need actual sockets."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class Seen:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


Route = Callable[[BaseHTTPRequestHandler], None]


class LocalServer:
    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.seen: list[Seen] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                server.seen.append(Seen(self.command, self.path, dict(self.headers.items()), body))
                route = server.routes.get(self.path.split("?")[0])
                if route is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                route(self)

            do_GET = do_POST = _handle  # noqa: N815

            def log_message(self, *args) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def reply(status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> Route:
    def route(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(status)
        for key, value in (headers or {}).items():
            handler.send_header(key, value)
        if "Content-Length" not in (headers or {}):
            handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    return route


def chunked(chunks: list[bytes], content_type: str) -> Route:
    """Streams without a Content-Length, the way a server hides a large body's size."""

    def route(handler: BaseHTTPRequestHandler) -> None:
        handler.protocol_version = "HTTP/1.1"
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        try:
            for chunk in chunks:
                handler.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            handler.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the client hung up once it hit its size cap

    return route

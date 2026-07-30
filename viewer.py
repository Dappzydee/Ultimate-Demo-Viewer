"""Launch the local GLB inspection viewer.

This uses only Python's standard library. A model passed on the command line is
served as the initial scene; models can also be opened or dropped in the UI.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import webbrowser


ROOT = Path(__file__).resolve().parent
VIEWER_ROOT = ROOT / "viewer"


class ViewerRequestHandler(BaseHTTPRequestHandler):
    """Serve viewer assets and, optionally, one explicitly selected GLB."""

    server: "ViewerServer"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = unquote(urlsplit(self.path).path)
        if path == "/viewer-config.json":
            payload = {
                "modelUrl": "/model.glb" if self.server.model_path else None,
                "modelName": self.server.model_path.name if self.server.model_path else None,
            }
            self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")
            return
        if path == "/model.glb" and self.server.model_path:
            self._send_file(self.server.model_path, "model/gltf-binary")
            return

        relative = "index.html" if path == "/" else path.lstrip("/")
        candidate = (VIEWER_ROOT / relative).resolve()
        if VIEWER_ROOT not in candidate.parents and candidate != VIEWER_ROOT:
            self.send_error(403)
            return
        if not candidate.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if candidate.suffix in {".js", ".css", ".html"}:
            content_type += "; charset=utf-8"
        self._send_file(candidate, content_type)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


class ViewerServer(ThreadingHTTPServer):
    model_path: Path | None

    def __init__(self, address: tuple[str, int], model_path: Path | None):
        super().__init__(address, ViewerRequestHandler)
        self.model_path = model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a colored GLB in the local CS2 result viewer.")
    parser.add_argument("model", nargs="?", type=Path, help="Optional .glb file to load initially")
    parser.add_argument("--port", type=int, default=0, help="Local port (default: choose an available port)")
    parser.add_argument("--no-open", action="store_true", help="Do not open the default browser")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = args.model.resolve() if args.model else None
    if model_path and (not model_path.is_file() or model_path.suffix.lower() != ".glb"):
        raise SystemExit(f"GLB file not found: {model_path}")
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")

    server = ViewerServer(("127.0.0.1", args.port), model_path)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Viewer running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

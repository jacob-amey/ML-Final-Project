"""Local HTTP server using Python's standard library, without a web framework."""

from __future__ import annotations

import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import tempfile
from urllib.error import URLError
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen
import warnings
import webbrowser

from PIL import Image, ImageOps, UnidentifiedImageError
from models import DEMO_ROOT, ModelComparison

WEB_ROOT = (DEMO_ROOT / "web").resolve()
APP_ID = "leafcheck-local-" + hashlib.sha256(str(DEMO_ROOT).encode()).hexdigest()[:12]
revision_files = sorted((DEMO_ROOT / "backend").glob("*.py"))
reference_file = DEMO_ROOT / "models/final_model.json"
if reference_file.exists():
    revision_files.append(reference_file)
APP_REVISION = hashlib.sha256(b"".join(path.read_bytes() for path in revision_files)).hexdigest()[:16]
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def main():
    parser = argparse.ArgumentParser(description="Run LeafCheck on this Mac.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    comparison = None

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **kwargs):
            super().__init__(*handler_args, directory=str(WEB_ROOT), **kwargs)

        def end_headers(self):
            # Browser assets and requests stay with this local app.
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; "
                             "style-src 'self'; script-src 'self'; connect-src 'self'; "
                             "font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()

        def send_json(self, data, status=200):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlsplit(self.path).path == "/api/status":
                return self.send_json({"app_id": APP_ID, "app_revision": APP_REVISION, "mode": "live", "models": comparison.metadata})
            requested = unquote(urlsplit(self.path).path)
            target = (WEB_ROOT / requested.lstrip("/")).resolve()
            if not target.is_relative_to(WEB_ROOT) or any(p.startswith(".") for p in Path(requested).parts):
                return self.send_error(404)
            if target.is_dir() and not (target / "index.html").exists():
                return self.send_error(404)
            return super().do_GET()

        def do_POST(self):
            if urlsplit(self.path).path != "/api/predict":
                return self.send_json({"error": "Unknown endpoint."}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.send_json({"error": "Invalid upload size."}, 400)
            if length < 1 or length > MAX_BYTES:
                self.close_connection = True
                return self.send_json({"error": "Choose a JPEG or PNG image under 10 MB."}, 413)
            try:
                raw = self.rfile.read(length)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(BytesIO(raw)) as source:
                        if source.format not in {"JPEG", "MPO", "PNG"}:
                            return self.send_json({"error": "Choose a JPEG or PNG image."}, 415)
                        if source.width * source.height > MAX_PIXELS:
                            return self.send_json({"error": "Choose an image under 24 megapixels."}, 413)
                        source.seek(0)
                        source.load()
                        image = ImageOps.exif_transpose(source).convert("RGB")
                with tempfile.TemporaryDirectory(prefix="leafcheck-") as folder:
                    path = Path(folder) / "upload.png"
                    image.save(path)
                    result = comparison.predict(path)
                return self.send_json(result)
            except (Image.DecompressionBombError, Image.DecompressionBombWarning):
                return self.send_json({"error": "Choose an image under 24 megapixels."}, 413)
            except (UnidentifiedImageError, OSError, ValueError):
                return self.send_json({"error": "This image could not be read. Try another JPEG or PNG photo."}, 400)

    # A second launch can reopen the app without loading another set of models.
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError:
        if args.open_browser:
            try:
                with urlopen(f"http://127.0.0.1:{args.port}/api/status", timeout=2) as response:
                    existing = json.load(response)
                if existing.get("app_id") == APP_ID and existing.get("app_revision") == APP_REVISION:
                    webbrowser.open(f"http://127.0.0.1:{args.port}")
                    print("LeafCheck is already running. Opened it in your browser.", flush=True)
                    return
                if existing.get("app_id") == APP_ID:
                    raise SystemExit("An earlier LeafCheck version is still running. Stop its launcher with Control-C, then launch again to load the updated model.")
            except (OSError, URLError, ValueError):
                pass
        raise SystemExit(f"Port {args.port} is already in use. Close the other local app or choose --port 8766.")

    server.daemon_threads = True
    try:
        print("Loading your saved LeafCheck models…", flush=True)
        comparison = ModelComparison()
        local_url = f"http://127.0.0.1:{server.server_port}"
        print(f"LeafCheck is ready: {local_url}", flush=True)
        print("Leave this window open. Press Control-C to stop LeafCheck.", flush=True)
        if args.open_browser:
            webbrowser.open(local_url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLeafCheck stopped.", flush=True)
    finally:
        server.server_close()
        if comparison is not None:
            comparison.close()


if __name__ == "__main__":
    main()

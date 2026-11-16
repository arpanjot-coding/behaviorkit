"""Loopback-only video review and still-image scene editor."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import os
import re
from urllib.parse import parse_qs, urlsplit

from .advanced import parse_scene
from .core import number
from .setup import Geometry
from .video_setup import COPY_CHUNK_BYTES, ReviewError, VideoReview


def validate_setup(data):
    scene=parse_scene(data['scene']); geometry=Geometry(**data['geometry'])
    cadence=number(data['sample_fps'],'sample_fps',.001,120)
    warnings=[]
    for rule in scene.rules:
        if rule.kind!='count' and rule.max_gap_seconds+1e-9<1/cadence:
            raise ValueError(f'{rule.id}: maximum gap is shorter than sampling period')
        if rule.kind=='dwell' and rule.seconds<2/cadence:
            warnings.append(f'{rule.id}: dwell threshold spans fewer than two sample intervals')
    return {'valid':True,'scene':scene.to_dict(),'geometry':data['geometry'],'sample_fps':cadence,
            'geometry_fingerprint':geometry.fingerprint,'warnings':warnings,
            'next':'Run doctor, then capture a positive/negative clip and evaluate held-out events.'}


EDITOR_CSP = (
    "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; connect-src 'self'; frame-src 'self'; frame-ancestors 'none'; "
    "object-src 'none'; base-uri 'none'; form-action 'none'"
)
REPORT_CSP = "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'self'; base-uri 'none'"


def byte_range(value, size):
    """Resolve one HTTP byte range to an inclusive start and end."""
    match = re.fullmatch(r"bytes=([0-9]{0,20})-([0-9]{0,20})", value, re.IGNORECASE)
    if not match or size <= 0:
        raise ValueError("invalid byte range")
    start, end = match.groups()
    if start:
        first, last = int(start), int(end) if end else size - 1
        if first >= size or last < first:
            raise ValueError("byte range is outside the file")
        return first, min(last, size - 1)
    if not end or int(end) == 0:
        raise ValueError("byte range suffix must be positive")
    return max(0, size - int(end)), size - 1


class _Handler(BaseHTTPRequestHandler):
    timeout = 30

    def log_message(self, *args):
        pass

    def _headers(self, status, length, kind, *, csp=EDITOR_CSP, extra=()):
        if kind.startswith("text/") or kind == "application/json":
            kind += "; charset=utf-8"
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", csp)
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in extra:
            self.send_header(name, value)
        try:
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
            return False
        return True

    def _respond(self, status, value, kind="application/json", *, csp=EDITOR_CSP):
        if isinstance(value, str):
            raw = value.encode("utf-8")
        elif isinstance(value, bytes):
            raw = value
        else:
            raw = json.dumps(value, allow_nan=False).encode("utf-8")
        if not self._headers(status, len(raw), kind, csp=csp) or self.command == "HEAD":
            return
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True

    def _send_file(self, source, kind, *, ranges=False, filename=None):
        size = os.fstat(source.fileno()).st_size
        first, last = 0, size - 1
        status = 200
        extra = [("Cross-Origin-Resource-Policy", "same-origin")]
        if filename is not None:
            extra.append(("Content-Disposition", f'attachment; filename="{filename}"'))
        if ranges:
            extra.append(("Accept-Ranges", "bytes"))
            requested = self.headers.get_all("Range", [])
            if requested and self.command == "GET" and "If-Range" not in self.headers:
                try:
                    if len(requested) != 1:
                        raise ValueError("only one byte range is supported")
                    first, last = byte_range(requested[0], size)
                except ValueError:
                    extra.append(("Content-Range", f"bytes */{size}"))
                    self._headers(416, 0, kind, extra=extra)
                    return
                status = 206
                extra.append(("Content-Range", f"bytes {first}-{last}/{size}"))
        remaining = last - first + 1
        if not self._headers(status, remaining, kind, extra=extra) or self.command == "HEAD":
            return
        try:
            source.seek(first)
            while remaining:
                chunk = source.read(min(COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except OSError:
            # The browser can drop a request when starting a new seek.
            self.close_connection = True

    def _origin(self):
        port = self.server.server_port
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0] not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            raise ReviewError(403, "Host rejected. Use the local BehaviorKit address.")
        return f"http://{hosts[0]}"

    def _length(self):
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1:
            raise ValueError("request must have a Content-Length")
        length = int(lengths[0])
        if length <= 0:
            raise ValueError("request body is empty")
        return length

    def _json(self):
        length = self._length()
        if length > 1024 * 1024:
            raise ValueError("setup request must be at most 1 MiB")
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        try:
            self._origin()
            request = urlsplit(self.path)
            path = request.path
            params = parse_qs(request.query, max_num_fields=10)
            video_id = params.get("video_id", [None])[0]
            run_id = params.get("run_id", [None])[0]
            review = self.server.review
            if path in {"/", "/scene"}:
                page = "video_setup.html" if path == "/" else "wizard.html"
                self._respond(200, files("behaviorkit").joinpath(page).read_text(encoding="utf-8"), "text/html")
            elif path == "/api/status":
                self._respond(200, review.readiness())
            elif path == "/api/video/preview":
                self._respond(200, review.preview(video_id), "image/jpeg")
            elif path == "/api/video/playback":
                with review.playback(video_id) as (source, kind):
                    self._send_file(source, kind, ranges=True)
            elif path == "/api/run":
                self._respond(200, review.run_state(video_id, run_id))
            elif path == "/api/report":
                self._respond(200, review.report(video_id, run_id), "text/html", csp=REPORT_CSP)
            elif path == "/api/timeline":
                from .review_results import read_timeline
                run_dir = review.completed_run(video_id, run_id)
                offset = int(params.get("offset", ["0"])[0])
                self._respond(200, read_timeline(run_dir, offset))
            elif path in {"/api/export/counts.csv", "/api/export/events.csv"}:
                run_dir = review.completed_run(video_id, run_id)
                filename = path.rsplit("/", 1)[1]
                with (run_dir / filename).open("rb") as source:
                    self._send_file(source, "text/csv", filename=filename)
            else:
                self._respond(404, {"error": "Not found."})
        except ReviewError as error:
            self._respond(error.status, {"error": str(error)})
        except ValueError as error:
            self._respond(400, {"error": str(error)})
        except OSError:
            self._respond(404, {"error": "The local file is unavailable."})

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        try:
            expected_origin = self._origin()
            path = urlsplit(self.path).path
            origins = self.headers.get_all("Origin", [])
            if (origins or path != "/validate") and origins != [expected_origin]:
                raise ReviewError(403, "Origin rejected. Use the local BehaviorKit page.")
            if path == "/validate":
                self._respond(200, validate_setup(self._json()))
            elif path == "/api/video":
                if self.headers.get_content_type() != "application/octet-stream":
                    raise ReviewError(415, "Video requests require application/octet-stream.")
                self._respond(200, self.server.review.upload(self.rfile, self._length()))
            elif path == "/api/run":
                if self.headers.get_content_type() != "application/json":
                    raise ReviewError(415, "Analysis requests require application/json.")
                self._respond(202, self.server.review.start_run(self._json()))
            else:
                self._respond(404, {"error": "Not found."})
        except ReviewError as error:
            self._respond(error.status, {"error": str(error)})
        except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as error:
            self._respond(400, {"error": str(error)})
        except TimeoutError:
            self._respond(408, {"error": "The request timed out. Try again."})


class _VideoServer(ThreadingHTTPServer):
    daemon_threads = False

    def server_close(self):
        review = getattr(self, "review", None)
        if review is not None:
            review.stop()
        super().server_close()
        if review is not None:
            review.close()


def make_server(port=8765, *, models_dir="models", output="runs/local-review"):
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("invalid port")
    server = _VideoServer(("127.0.0.1", port), _Handler)
    try:
        server.review = VideoReview(models_dir, output)
    except Exception:
        server.server_close()
        raise
    return server


def serve(port=8765, *, models_dir="models", output="runs/local-review", open_browser=False):
    with make_server(port, models_dir=models_dir, output=output) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"BehaviorKit setup: {url}", flush=True)
        if open_browser:
            import webbrowser
            webbrowser.open(url)
        try:
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass

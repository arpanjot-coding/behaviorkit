"""Run one local waiting-area analysis at a time."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from threading import Event, Lock, Thread
from types import SimpleNamespace
import math
import sys
import uuid

from .cli import open_local_video, run_video
from .core import SceneConfig, keys
from .playback_timing import rebase_browser_video
from .report import write_report
from .review_results import write_csv_exports
from .storage import write_json


MODEL = "d-fine-n"
SAMPLE_FPS = 5
CPU_THREADS = 4
COPY_CHUNK_BYTES = 1024 * 1024


class ReviewError(Exception):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


@dataclass(frozen=True)
class StagedVideo:
    path: Path
    preview: bytes
    width: int
    height: int
    content_type: str
    first_timestamp: float
    playback_path: Path | None


def video_content_type(path, format_names):
    if "mp4" in format_names:
        with path.open("rb") as source:
            header = source.read(12)
        return "video/quicktime" if header[4:12] == b"ftypqt  " else "video/mp4"
    if "matroska" in format_names:
        with path.open("rb") as source:
            header = source.read(4096)
        # The EBML DocType element identifies WebM within the shared demuxer.
        return "video/webm" if b"\x42\x82\x84webm" in header else "video/x-matroska"
    for name, content_type in (
        ("avi", "video/x-msvideo"), ("mpegts", "video/mp2t"),
        ("mpeg", "video/mpeg"), ("ogg", "video/ogg"), ("flv", "video/x-flv"),
    ):
        if name in format_names:
            return content_type
    return "application/octet-stream"


def make_scene(data):
    fields = {"zone", "dwell_seconds", "video_id"}
    keys(data, fields, "analysis", fields)
    zone = "waiting_area"
    return SceneConfig.from_dict({
        "schema_version": "0.1",
        "zones": {zone: data["zone"]},
        "rules": [
            {"id": "people_in_zone", "kind": "count", "zone": zone},
            {"id": "person_dwell", "kind": "dwell", "zone": zone, "seconds": data["dwell_seconds"]},
        ],
    }).to_dict()


def decode_preview(path):
    try:
        import av
        import cv2
    except (ImportError, OSError) as error:
        raise ReviewError(503, "Install the vision and CPU extras to open a video.") from error
    try:
        with open_local_video(path) as container:
            if container.streams.video:
                for frame in container.decode(video=0):
                    if frame.pts is None or frame.time_base is None:
                        break
                    timestamp = float(frame.pts * frame.time_base)
                    if not math.isfinite(timestamp):
                        break
                    if timestamp < 0:
                        continue
                    ok, encoded = cv2.imencode(".jpg", frame.to_ndarray(format="bgr24"))
                    if ok:
                        return StagedVideo(
                            path, encoded.tobytes(), frame.width, frame.height,
                            video_content_type(path, container.format.name.split(",")), timestamp, path,
                        )
                    break
    except (av.FFmpegError, cv2.error) as error:
        raise ReviewError(400, "Could not read this video. Try a playable local video such as an MP4.") from error
    raise ReviewError(400, "The file has no usable video frame with a timestamp.")


def prepare_playback(staged, destination):
    if staged.first_timestamp <= 1e-6:
        return staged
    import av

    try:
        content_type = rebase_browser_video(
            staged.path, destination, staged.content_type, staged.first_timestamp,
        )
    except (av.FFmpegError, ValueError, OSError) as error:
        print(f"BehaviorKit playback preparation: {error}", file=sys.stderr)
        content_type = None
    if content_type is None:
        destination.unlink(missing_ok=True)
        return replace(staged, playback_path=None)
    return replace(staged, playback_path=destination, content_type=content_type)


def remove_staged_video(staged):
    if staged.playback_path is not None and staged.playback_path != staged.path:
        staged.playback_path.unlink(missing_ok=True)
    staged.path.unlink(missing_ok=True)


class VideoReview:
    def __init__(self, models_dir, output):
        self.models_dir = Path(models_dir).resolve()
        self.output = Path(output).resolve()
        self._temp = TemporaryDirectory(prefix="behaviorkit-video-")
        self._lock = Lock()
        self._stop = Event()
        self._uploading = False
        self._worker = None
        self._video = None
        self._playback_readers = {}
        self._report = None
        self._state = {"status": "idle", "message": "Open a video to begin.", "frames": 0}

    def readiness(self):
        if any(find_spec(module) is None for module in ("av", "cv2", "numpy", "onnxruntime")):
            return {"ready": False, "message": 'Install CPU vision with python -m pip install -e ".[vision,cpu]".'}
        model = self.models_dir / MODEL
        if not ((model / "model.onnx").is_file() and (model / "manifest.json").is_file()):
            return {"ready": False, "message": "D-FINE-N is missing. Prepare the local model or restart setup with --models-dir pointing to it."}
        return {"ready": True, "message": "Ready to analyze on CPU."}

    def _check_video(self, video_id, status=404):
        if self._video is None or self._video.path.name != video_id:
            raise ReviewError(status, "This video is no longer open. Open it again to continue.")

    def _check_run(self, run_id):
        if run_id is not None and self._state.get("run_id") != run_id:
            raise ReviewError(404, "This analysis has been replaced. Reload for the latest result.")

    def run_state(self, video_id, run_id=None):
        with self._lock:
            if self._video is not None or video_id is not None:
                self._check_video(video_id)
            self._check_run(run_id)
            return self._state.copy()

    def preview(self, video_id):
        with self._lock:
            self._check_video(video_id)
            return self._video.preview

    @contextmanager
    def playback(self, video_id):
        with self._lock:
            self._check_video(video_id)
            staged = self._video
            if staged.playback_path is None:
                raise ReviewError(422, "Playback is unavailable for this video clock. The report and CSVs are still available.")
            try:
                source = staged.playback_path.open("rb")
            except OSError as error:
                raise ReviewError(404, "The video is unavailable on disk. Open it again.") from error
            self._playback_readers[staged.path] = self._playback_readers.get(staged.path, 0) + 1
        try:
            with source:
                yield source, staged.content_type
        finally:
            with self._lock:
                remaining = self._playback_readers[staged.path] - 1
                if remaining:
                    self._playback_readers[staged.path] = remaining
                else:
                    del self._playback_readers[staged.path]
                    # Windows cannot unlink a file while a playback reader is open.
                    if staged is not self._video:
                        remove_staged_video(staged)

    def completed_run(self, video_id, run_id):
        with self._lock:
            self._check_video(video_id)
            self._check_run(run_id)
            if run_id is None or self._report is None:
                raise ReviewError(404, "No completed analysis for this request.")
            return self._report.parent

    def report(self, video_id, run_id=None):
        with self._lock:
            self._check_video(video_id)
            self._check_run(run_id)
            if self._report is None:
                raise ReviewError(404, "No completed report yet.")
            try:
                return self._report.read_bytes()
            except OSError as error:
                raise ReviewError(404, "The completed report is unavailable on disk.") from error

    def _check_idle(self):
        if self._stop.is_set() or self._uploading or self._state["status"] == "running":
            raise ReviewError(409, "Wait for the current operation to finish.")

    def upload(self, stream, length):
        with self._lock:
            self._check_idle()
            self._uploading = True
        pending = Path(self._temp.name) / uuid.uuid4().hex
        playback_copy = pending.with_suffix(".playback")
        accepted = False
        try:
            with pending.open("xb") as target:
                remaining = length
                while remaining:
                    if self._stop.is_set():
                        raise ReviewError(409, "The local server is shutting down.")
                    chunk = stream.read(min(COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ReviewError(400, "The video transfer ended early. Try opening it again.")
                    target.write(chunk)
                    remaining -= len(chunk)
            staged = prepare_playback(decode_preview(pending), playback_copy)
            with self._lock:
                if self._stop.is_set():
                    raise ReviewError(409, "The local server is shutting down.")
                if self._video is not None and self._video.path not in self._playback_readers:
                    remove_staged_video(self._video)
                self._video = staged
                self._report = None
                self._state = {"status": "idle", "message": "Video ready. Draw the waiting area.", "frames": 0}
                accepted = True
            return {
                "status": "ready", "video_id": pending.name, "width": staged.width, "height": staged.height,
                "playback_offset_seconds": staged.first_timestamp if staged.playback_path == playback_copy else 0,
                "playback_available": staged.playback_path is not None,
            }
        except TimeoutError as error:
            raise ReviewError(408, "The video transfer timed out. Try opening it again.") from error
        except OSError as error:
            raise ReviewError(507, "Could not store this video temporarily. Check available disk space.") from error
        finally:
            try:
                if not accepted:
                    playback_copy.unlink(missing_ok=True)
                    pending.unlink(missing_ok=True)
            finally:
                with self._lock:
                    self._uploading = False

    def start_run(self, data):
        scene = make_scene(data)
        with self._lock:
            self._check_idle()
            self._check_video(data["video_id"], status=409)
            readiness = self.readiness()
            if not readiness["ready"]:
                raise ReviewError(503, readiness["message"])
            self._report = None
            self._state = {
                "status": "running", "message": "Analyzing video on CPU…", "frames": 0,
                "run_id": uuid.uuid4().hex,
            }
            self._worker = Thread(target=self._run, args=(self._video.path, scene))
            self._worker.start()
            return self._state.copy()

    def _progress(self, frames):
        if self._stop.is_set():
            raise InterruptedError("Video review stopped.")
        with self._lock:
            self._state["frames"] = frames

    def _run(self, video, scene):
        message = "Could not analyze this video on CPU. Check the BehaviorKit terminal for details."
        try:
            self.output.mkdir(parents=True, exist_ok=True)
            run_dir = Path(mkdtemp(prefix="review-", dir=self.output))
            config = run_dir / "scene.json"
            write_json(config, scene)
            args = SimpleNamespace(
                video=video, config=config, output=run_dir, model=MODEL, models_dir=self.models_dir,
                provider="cpu", sample_fps=SAMPLE_FPS, threads=CPU_THREADS, max_frames=0,
                source="video", trace=False,
            )
            run_video(args, on_sample=self._progress)
            message = "Analysis finished, but the report could not be saved. Check the BehaviorKit terminal."
            report = run_dir / "report.html"
            write_report(run_dir, report)
            write_csv_exports(run_dir)
        except Exception as error:
            print(f"BehaviorKit video review: {error}", file=sys.stderr)
            with self._lock:
                self._state.update(status="error", message=message)
            return
        with self._lock:
            self._report = report
            self._state.update(status="complete", message="Analysis complete. Your report is ready.")

    def stop(self):
        self._stop.set()

    def close(self):
        if self._worker is not None:
            self._worker.join()
        self._temp.cleanup()

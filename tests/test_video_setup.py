import csv
import io
import json
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from behaviorkit.contracts import read_json, read_jsonl
from behaviorkit.review_results import TIMELINE_PAGE_SIZE, read_timeline, write_csv_exports
from behaviorkit.wizard import make_server
from test_pipeline import write_clip


@pytest.fixture
def local_setup(signal_model, tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("cv2")
    clip = tmp_path / "queue.mp4"
    write_clip(clip)
    output = tmp_path / "reviews"
    with make_server(0, models_dir=signal_model, output=output) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        try:
            yield origin, clip.read_bytes(), output
        finally:
            server.shutdown()
            thread.join(timeout=2)


def request(origin, path, *, data=None, media_type=None, request_origin=None, byte_range=None, method=None):
    headers = {}
    if media_type:
        headers["Content-Type"] = media_type
    if request_origin is not None:
        headers["Origin"] = request_origin
    if byte_range is not None:
        headers["Range"] = byte_range
    return urlopen(Request(origin + path, data, headers, method=method), timeout=5)


def upload(origin, clip):
    return request(origin, "/api/video", data=clip,
                   media_type="application/octet-stream", request_origin=origin)


def run(origin, video_id, zone):
    data = json.dumps({"video_id": video_id, "zone": zone, "dwell_seconds": 1}).encode()
    return request(origin, "/api/run", data=data,
                   media_type="application/json", request_origin=origin)


def assert_unavailable(origin, path):
    with pytest.raises(HTTPError) as denied:
        request(origin, path)
    assert denied.value.code == 404
    denied.value.close()


def download_csv(origin, name, query):
    with request(origin, f"/api/export/{name}.csv" + query) as response:
        assert response.headers.get_content_type() == "text/csv"
        assert response.headers.get_content_disposition() == "attachment"
        assert response.headers.get_filename() == f"{name}.csv"
        return list(csv.DictReader(io.StringIO(response.read().decode("utf-8"), newline="")))


def test_video_setup_cpu_report_and_video_replacement(local_setup):
    origin, clip, output = local_setup
    with request(origin, "/api/status") as response:
        assert json.load(response)["ready"]
    assert_unavailable(origin, "/api/report")

    with upload(origin, clip) as response:
        staged = json.load(response)
    assert (staged["status"], staged["width"], staged["height"]) == ("ready", 320, 240)
    video_id = staged["video_id"]
    query = "?" + urlencode({"video_id": video_id})
    with request(origin, "/api/video/preview" + query) as response:
        assert response.headers.get_content_type() == "image/jpeg"
        assert response.read(2) == b"\xff\xd8"
    playback_path = "/api/video/playback" + query
    with request(origin, playback_path, method="HEAD") as response:
        assert response.headers.get_content_type() == "video/mp4"
        assert response.headers["Accept-Ranges"] == "bytes"
        assert int(response.headers["Content-Length"]) == len(clip)
        assert response.read() == b""
    with request(origin, playback_path, byte_range="bytes=10-29") as response:
        assert response.status == 206
        assert response.headers["Content-Range"] == f"bytes 10-29/{len(clip)}"
        assert response.read() == clip[10:30]
    with request(origin, playback_path, byte_range="bytes=-15") as response:
        assert response.status == 206
        assert response.read() == clip[-15:]
    with pytest.raises(HTTPError) as beyond_eof:
        request(origin, playback_path, byte_range=f"bytes={len(clip)}-")
    assert beyond_eof.value.code == 416
    assert beyond_eof.value.headers["Content-Range"] == f"bytes */{len(clip)}"
    beyond_eof.value.close()

    # Only the left-hand track is in this polygon. The other appears briefly outside it.
    zone = [[0, 0], [0.45, 0], [0.45, 1], [0, 1]]
    with run(origin, video_id, zone) as response:
        started = json.load(response)
    assert started["status"] == "running"
    analysis_query = "?" + urlencode({"video_id": video_id, "run_id": started["run_id"]})
    deadline = time.monotonic() + 20
    while True:
        with request(origin, "/api/run" + analysis_query) as response:
            status = json.load(response)
        if status["status"] != "running":
            break
        assert time.monotonic() < deadline, status
        time.sleep(0.05)
    assert status["status"] == "complete", status
    with request(origin, "/api/report" + analysis_query) as response:
        assert response.headers.get_content_type() == "text/html"
        report = response.read().decode()
    assert "1.200 s" in report
    assert "d-fine-n" in report
    assert_unavailable(origin, "/api/report" + query + "&run_id=other")

    summary_path, = output.rglob("summary.json")
    summary = read_json(summary_path)
    assert summary["provider"] == "cpu"
    assert summary["count_samples"] == 16
    assert summary["events"] == 1
    counts = list(read_jsonl(summary_path.with_name("counts.jsonl")))
    assert max(row["count"] for row in counts) == 1
    assert counts[-1]["count"] == 0

    event, = read_jsonl(summary_path.with_name("events.jsonl"))
    with request(origin, "/api/timeline" + analysis_query) as response:
        timeline = json.load(response)
    assert timeline["total"] == 1
    assert timeline["next_offset"] is None
    assert timeline["events"] == [{
        "id": event["id"], "timestamp": event["timestamp"], "start": event["start"],
        "duration_seconds": event["duration_seconds"],
        "track_id": event["track_id"],
    }]
    counts_csv = download_csv(origin, "counts", analysis_query)
    assert [(float(row["timestamp_seconds"]), int(row["count"]), row["status"])
            for row in counts_csv] == [(row["timestamp"], row["count"], row["status"]) for row in counts]
    assert [json.loads(row["track_ids_json"]) for row in counts_csv] == [row["track_ids"] for row in counts]
    event_csv, = download_csv(origin, "events", analysis_query)
    assert event_csv["id"] == event["id"]
    assert float(event_csv["timestamp_seconds"]) == event["timestamp"]
    assert json.loads(event_csv["evidence_json"]) == event["evidence"]
    wrong_run_query = query + "&run_id=other"
    assert_unavailable(origin, "/api/timeline" + wrong_run_query)
    assert_unavailable(origin, "/api/export/counts.csv" + wrong_run_query)

    with upload(origin, clip) as response:
        replacement = json.load(response)
    assert replacement["status"] == "ready"
    assert replacement["video_id"] != video_id
    assert_unavailable(origin, "/api/report" + query)
    assert_unavailable(origin, "/api/video/preview" + query)
    assert_unavailable(origin, "/api/video/playback" + query)
    assert_unavailable(origin, "/api/timeline" + analysis_query)
    assert_unavailable(origin, "/api/export/events.csv" + analysis_query)
    with pytest.raises(HTTPError) as stale:
        run(origin, video_id, zone)
    assert stale.value.code == 409
    stale.value.close()
    replacement_query = "?" + urlencode({"video_id": replacement["video_id"]})
    with request(origin, "/api/run" + replacement_query) as response:
        assert json.load(response)["status"] == "idle"
    assert summary_path.with_name("report.html").is_file()


def test_video_setup_rejects_nonlocal_mutations_and_invalid_zone(local_setup):
    origin, clip, _ = local_setup
    for untrusted in (None, "https://example.invalid"):
        with pytest.raises(HTTPError) as denied:
            request(origin, "/api/video", data=clip,
                    media_type="application/octet-stream", request_origin=untrusted)
        assert denied.value.code == 403
        denied.value.close()
    assert_unavailable(origin, "/api/video/preview")

    with upload(origin, clip) as response:
        staged = json.load(response)
    assert staged["status"] == "ready"
    with pytest.raises(HTTPError) as invalid:
        run(origin, staged["video_id"], [[0, 0], [1, 1], [0, 1], [1, 0]])
    assert invalid.value.code == 400
    assert "polygon" in json.load(invalid.value)["error"].lower()
    invalid.value.close()
    with request(origin, "/api/run?" + urlencode({"video_id": staged["video_id"]})) as response:
        assert json.load(response)["status"] == "idle"


def test_timeline_uses_bounded_pages_in_source_order(tmp_path):
    events = [{"id": f"event-{i}", "timestamp": i + 1, "start": i, "duration_seconds": 1,
               "track_id": str(i)} for i in range(TIMELINE_PAGE_SIZE + 2)]
    (tmp_path / "summary.json").write_text(json.dumps({"events": len(events)}), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    first = read_timeline(tmp_path)
    second = read_timeline(tmp_path, first["next_offset"])
    assert first["events"] == events[:TIMELINE_PAGE_SIZE]
    assert second["events"] == events[TIMELINE_PAGE_SIZE:]
    assert first["total"] == second["total"] == len(events)
    assert second["next_offset"] is None
    with pytest.raises(ValueError, match="timeline offset"):
        read_timeline(tmp_path, 1)


def test_csv_distinguishes_unknown_count_and_writes_empty_events_header(tmp_path):
    counts = [{"timestamp": i, "count": count, "status": status, "rule": "people_in_zone",
               "source": "video", "epoch": "0", "track_ids": []}
              for i, (count, status) in enumerate(((None, "unknown"), (0, "observed")))]
    (tmp_path / "counts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in counts), encoding="utf-8")
    (tmp_path / "events.jsonl").touch()
    write_csv_exports(tmp_path)
    with (tmp_path / "counts.csv").open(newline="", encoding="utf-8") as stream:
        assert [(row["count"], row["status"]) for row in csv.DictReader(stream)] == [("", "unknown"), ("0", "observed")]
    with (tmp_path / "events.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert "timestamp_seconds" in reader.fieldnames
        assert list(reader) == []

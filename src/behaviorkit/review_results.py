"""Paginated timeline and CSV exports from completed queue review results."""

import csv
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .contracts import MAX_LINE_BYTES, read_json, read_jsonl


TIMELINE_PAGE_SIZE = 100
TIMELINE_FIELDS = ("id", "timestamp", "start", "duration_seconds", "track_id")
COUNT_COLUMNS = (
    ("timestamp_seconds", "timestamp"), ("count", "count"), ("status", "status"),
    ("rule", "rule"), ("source", "source"), ("epoch", "epoch"), ("track_ids_json", "track_ids"),
)
EVENT_COLUMNS = (
    ("id", "id"), ("kind", "kind"), ("rule", "rule"), ("zone", "zone"), ("source", "source"),
    ("epoch", "epoch"), ("track_id", "track_id"), ("start_seconds", "start"),
    ("timestamp_seconds", "timestamp"), ("duration_seconds", "duration_seconds"), ("status", "status"),
    ("identity_version", "identity_version"), ("config_fingerprint", "config_fingerprint"),
    ("evidence_json", "evidence"),
)


def read_timeline(run_dir, offset=0):
    """Return a page ordered by confirmation time, with an opaque byte offset for the next page."""
    total = read_json(run_dir / "summary.json")["events"]
    events = []
    with (run_dir / "events.jsonl").open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        if not 0 <= offset <= size:
            raise ValueError("Invalid timeline offset.")
        if offset:
            stream.seek(offset - 1)
            if stream.read(1) != b"\n":
                raise ValueError("Invalid timeline offset.")
        else:
            stream.seek(0)
        for _ in range(TIMELINE_PAGE_SIZE):
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError("Timeline event exceeds the maximum record size.")
            event = json.loads(raw)
            events.append({key: event[key] for key in TIMELINE_FIELDS})
        position = stream.tell()
        next_offset = position if position < size else None
    return {"events": events, "total": total, "next_offset": next_offset}


def _csv_values(record, columns):
    for column, field in columns:
        value = record[field]
        if column.endswith("_json"):
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        yield value


def write_csv_exports(run_dir):
    """Write complete CSV files without loading all result records into memory."""
    for name, columns in (("counts", COUNT_COLUMNS), ("events", EVENT_COLUMNS)):
        target = run_dir / f"{name}.csv"
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False,
                                    dir=run_dir, prefix=f".{name}-", suffix=".tmp") as stream:
                temporary = Path(stream.name)
                writer = csv.writer(stream)
                writer.writerow(column for column, _ in columns)
                for record in read_jsonl(run_dir / f"{name}.jsonl"):
                    writer.writerow(_csv_values(record, columns))
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

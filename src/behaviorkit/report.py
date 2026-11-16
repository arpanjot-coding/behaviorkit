"""Offline HTML summaries of completed count and event records."""

from dataclasses import dataclass
from html import escape
from importlib.resources import files
from string import Template

from .contracts import read_json, read_jsonl
from .core import number, text
from .recording import verified_manifest


EVENT_LIMIT = 50
COUNT_GROUP_LIMIT = 40


@dataclass
class Counts:
    samples: int = 0
    unknown: int = 0
    zeros: int = 0
    peak: int | None = None

    def add(self, count):
        self.samples += 1
        if count is None:
            self.unknown += 1
        else:
            self.zeros += count == 0
            self.peak = count if self.peak is None else max(self.peak, count)


def html_row(*values):
    return "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in values) + "</tr>\n"


def source_clock(value, label):
    return f"{number(value, label):,.3f} s"


def write_report(run_dir, output):
    """Read result JSONL only and create a self-contained HTML file."""
    if output.suffix.lower() != ".html":
        raise ValueError("report output must have an .html extension")
    if output.exists():
        raise FileExistsError(f"report output already exists: {output}")

    if (run_dir / "manifest.json").exists():
        manifest = verified_manifest(run_dir, kind="rules")
        if output.resolve().is_relative_to(run_dir.resolve()):
            raise ValueError("write the report outside the immutable rule recording")
        if not {"counts.jsonl", "events.jsonl"} <= manifest["artifacts"].keys():
            raise ValueError("rule recording is missing verified count or event records")
        expected_counts, expected_events = manifest["totals"]["counts"], manifest["totals"]["events"]
        origin = "Rule replay"
    else:
        summary = read_json(run_dir / "summary.json")
        if not isinstance(summary, dict) or summary.get("status") != "ok" or summary.get("kind") not in {"replay", "video"}:
            raise ValueError("report requires a successful demo, run, replay, or apply output")
        expected_counts, expected_events = summary["count_samples"], summary["events"]
        if summary.get("synthetic_demo") is True:
            origin = "Synthetic demo · generated observations"
        elif summary["kind"] == "video":
            origin = "Video analysis"
            if summary.get("model") and summary.get("provider"):
                origin += f" · {summary['model']} · {summary['provider']}"
        else:
            origin = "Observation replay"
    if any(type(value) is not int or value < 0 for value in (expected_counts, expected_events)):
        raise ValueError("completed run totals must be nonnegative integers")

    total = Counts()
    groups = {}
    count_rows_truncated = False
    for row in read_jsonl(run_dir / "counts.jsonl"):
        count = row["count"]
        status = row["status"]
        if not (
            (status == "unknown" and count is None)
            or (status == "observed" and type(count) is int and count >= 0)
        ):
            raise ValueError("invalid count or coverage status in counts.jsonl")
        group = (text(row["source"], "source"), text(row["rule"], "rule"))
        total.add(count)
        if group not in groups:
            if len(groups) == COUNT_GROUP_LIMIT:
                count_rows_truncated = True
                continue
            groups[group] = Counts()
        groups[group].add(count)

    event_rows = []
    event_count = 0
    for event in read_jsonl(run_dir / "events.jsonl"):
        event_count += 1
        if event_count <= EVENT_LIMIT:
            event_rows.append(html_row(
                source_clock(event["timestamp"], "confirmation time"),
                source_clock(event["start"], "evidence start"),
                text(event["kind"], "event kind").replace("_", " ").capitalize(),
                text(event["source"], "source"),
                text(event["rule"], "rule"),
                text(event["zone"], "zone"),
                text(event["track_id"], "track"),
                text(event["epoch"], "source segment"),
            ))

    if total.samples != expected_counts or event_count != expected_events:
        raise ValueError("count or event records do not match the completed run totals")

    count_rows = "".join(
        html_row(source, rule, f"{group.samples:,}", f"{group.zeros:,}", f"{group.unknown:,}",
                 group.peak if group.peak is not None else "—")
        for (source, rule), group in groups.items()
    )
    count_note = (
        f"Showing the first {COUNT_GROUP_LIMIT} source and rule pairs in the run. The totals above include all samples."
        if count_rows_truncated else "Each row combines count samples for one rule and source."
    )
    event_note = (
        f"Showing the first {EVENT_LIMIT} of {event_count:,} events in recorded order. See events.jsonl in the run directory for all events."
        if event_count > EVENT_LIMIT else "Times are seconds on each source segment’s clock."
    )
    template = Template(files("behaviorkit").joinpath("report.html").read_text(encoding="utf-8"))
    html = template.substitute(
        origin=escape(origin),
        count_samples=f"{total.samples:,}",
        observed_zeros=f"{total.zeros:,}",
        unknown_samples=f"{total.unknown:,}",
        events=f"{event_count:,}",
        count_note=count_note,
        count_rows=count_rows or '<tr><td colspan="6">No count samples in this run.</td></tr>',
        event_note=event_note,
        event_rows="".join(event_rows) or '<tr><td colspan="8">No events confirmed in this run.</td></tr>',
    )
    content = html.encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("xb")
    try:
        with stream:
            stream.write(content)
    except OSError:
        output.unlink(missing_ok=True)
        raise
    return {"status": "ok", "output": str(output.resolve()), "count_samples": total.samples,
            "unknown_count_samples": total.unknown, "events": event_count}

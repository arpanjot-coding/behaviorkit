import json

from behaviorkit.cli import demo, main, replay


def test_demo_report_uses_saved_results_and_preserves_existing_file(tmp_path, capsys):
    run_dir = tmp_path / "demo"
    report = tmp_path / "report.html"
    demo(run_dir)

    args = ["report", "--run-dir", str(run_dir), "--output", str(report)]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    html = report.read_text(encoding="utf-8")
    assert result["count_samples"] == 41
    assert result["events"] == 1
    assert "Synthetic demo" in html
    assert "3.000 s" in html
    assert "person-1" in html

    assert main(args) == 2
    assert "already exists" in capsys.readouterr().err
    assert report.read_text(encoding="utf-8") == html


def test_replay_report_distinguishes_unknown_from_zero_and_checks_totals(tmp_path, capsys):
    original = tmp_path / "demo"
    demo(original)
    source = '<entry & exit>'
    observations = tmp_path / "observations.jsonl"
    observations.write_text("\n".join(json.dumps({
        "source": source, "epoch": "0", "timestamp": i, "covered": covered, "detections": [],
    }) for i, covered in enumerate([True, False])), encoding="utf-8")
    run_dir = tmp_path / "replay"
    replay(observations, original / "config.json", run_dir)

    report = tmp_path / "report.html"
    assert main(["report", "--run-dir", str(run_dir), "--output", str(report)]) == 0
    result = json.loads(capsys.readouterr().out)
    html = report.read_text(encoding="utf-8")
    assert result["unknown_count_samples"] == 1
    assert "<dd>1</dd>" in html
    assert "<td>2</td><td>1</td><td>1</td><td>0</td>" in html
    assert source not in html and "&lt;entry &amp; exit&gt;" in html
    assert "No events confirmed in this run." in html

    (run_dir / "counts.jsonl").write_text("", encoding="utf-8")
    partial_report = tmp_path / "partial.html"
    assert main(["report", "--run-dir", str(run_dir), "--output", str(partial_report)]) == 2
    assert "do not match" in capsys.readouterr().err
    assert not partial_report.exists()

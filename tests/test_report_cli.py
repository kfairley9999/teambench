from __future__ import annotations

import json
from pathlib import Path

from teambench.cli import main
from teambench.models.mock import ScriptedModel
from teambench.report import compare_payloads, html_report, markdown_report
from teambench.runner import run_benchmark


def _payload() -> dict:
    return run_benchmark(
        ScriptedModel(fallibility=0.0),
        task_ids=["T01", "T13"],
        patterns=["single", "debate"],
        repeats=1,
        seed=3,
        created_at="2026-11-21T12:00:00+00:00",
        notes="fixture",
    )


def test_markdown_and_html_contain_leaderboard() -> None:
    payload = _payload()
    md = markdown_report(payload)
    html = html_report(payload)
    assert "Pattern leaderboard" in md
    assert "T13" in md and "Banned vendor" in md
    assert "<!DOCTYPE html>" in html
    assert "TeamBench results" in html
    assert "Provefirst" in html


def test_compare_detects_mismatch() -> None:
    payload = _payload()
    other = json.loads(json.dumps(payload))
    other["trials"][0]["success"] = not other["trials"][0]["success"]
    errors = compare_payloads(payload, other)
    assert errors
    assert compare_payloads(payload, payload) == []


def test_cli_list_run_report_compare(tmp_path: Path, capsys) -> None:
    assert main(["list"]) == 0
    listed = capsys.readouterr().out
    assert "T01\tplanning\tBudget itinerary" in listed
    out = tmp_path / "run.json"
    html = tmp_path / "run.html"
    md = tmp_path / "run.md"
    assert (
        main(
            [
                "run",
                "--model",
                "mock",
                "--tasks",
                "T14",
                "--patterns",
                "single",
                "--repeats",
                "1",
                "--seed",
                "1",
                "--fallibility",
                "0",
                "--output",
                str(out),
                "--html",
                str(html),
                "--markdown",
                str(md),
            ]
        )
        == 0
    )
    assert out.is_file() and html.is_file() and md.is_file()
    assert main(["report", str(out), "--html", str(tmp_path / "b.html")]) == 0
    assert main(["compare", str(out), str(out)]) == 0
    broken = json.loads(out.read_text())
    broken["trials"][0]["success"] = True
    other = tmp_path / "other.json"
    other.write_text(json.dumps(broken))
    assert main(["compare", str(out), str(other)]) == 1

"""Command-line interface for TeamBench."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from teambench import __version__
from teambench.models.base import ModelClient
from teambench.models.gemini import DEFAULT_MODEL, GeminiModel
from teambench.models.mock import ScriptedModel
from teambench.report import (
    compare_payloads,
    dump_payload,
    html_report,
    load_payload,
    markdown_report,
)
from teambench.runner import run_benchmark
from teambench.tasks.catalog import load_suite, select_tasks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="teambench", description="Multi-agent orchestration evals.")
    parser.add_argument("--version", action="version", version=f"teambench {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List the task suite.")

    run = sub.add_parser("run", help="Run the benchmark.")
    run.add_argument("--model", choices=("mock", "gemini"), default="mock")
    run.add_argument("--gemini-model", default=DEFAULT_MODEL, help="Gemini model id for --model gemini.")
    run.add_argument("--patterns", nargs="+", default=None, help="Pattern ids. Default: all four.")
    run.add_argument("--tasks", nargs="+", default=None, help="Task ids. Default: the full suite.")
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument("--fallibility", type=float, default=0.1, help="Slip probability for the mock model.")
    run.add_argument("--notes", default="")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--html", type=Path, default=None)
    run.add_argument("--markdown", type=Path, default=None)

    report = sub.add_parser("report", help="Render a saved JSON run as HTML and/or markdown.")
    report.add_argument("input", type=Path)
    report.add_argument("--html", type=Path, default=None)
    report.add_argument("--markdown", type=Path, default=None)

    compare = sub.add_parser("compare", help="Check that two runs have the same per-trial outcomes.")
    compare.add_argument("expected", type=Path)
    compare.add_argument("actual", type=Path)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "list":
        return _cmd_list()
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "report":
        return _cmd_report(args)
    return _cmd_compare(args)


def _cmd_list() -> int:
    load_suite()
    for task in select_tasks():
        sys.stdout.write(f"{task.id}\t{task.category}\t{task.name}\t{task.description}\n")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.model == "mock":
        model: ModelClient = ScriptedModel(fallibility=args.fallibility)
    else:
        model = GeminiModel(model=args.gemini_model)
        args.fallibility = 0.0
    payload = run_benchmark(
        model,
        task_ids=args.tasks,
        patterns=args.patterns,
        repeats=args.repeats,
        seed=args.seed,
        notes=args.notes,
    )
    # Keep fallibility in the manifest accurate for gemini.
    payload["manifest"]["fallibility"] = float(getattr(model, "fallibility", 0.0) if args.model == "mock" else 0.0)
    dump_payload(payload, args.output)
    _write_views(payload, html=args.html, markdown=args.markdown)
    _print_summary(payload)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    payload = load_payload(args.input)
    _write_views(payload, html=args.html, markdown=args.markdown)
    if args.markdown is None and args.html is None:
        sys.stdout.write(markdown_report(payload))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    expected = load_payload(args.expected)
    actual = load_payload(args.actual)
    errors = compare_payloads(expected, actual)
    if errors:
        sys.stderr.write(f"{len(errors)} mismatch(es)\n")
        for err in errors[:40]:
            sys.stderr.write(f"  {err}\n")
        if len(errors) > 40:
            sys.stderr.write(f"  … {len(errors) - 40} more\n")
        return 1
    sys.stdout.write("runs match\n")
    return 0


def _write_views(
    payload: dict[str, object],
    *,
    html: Path | None,
    markdown: Path | None,
) -> None:
    if html is not None:
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(html_report(payload), encoding="utf-8")
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(markdown_report(payload), encoding="utf-8")


def _print_summary(payload: dict[str, object]) -> None:
    from teambench.metrics import summarize
    from teambench.runner import trials_from_payload

    metrics = summarize(trials_from_payload(payload))
    overall = metrics["overall"]
    sys.stdout.write(
        json.dumps(
            {
                "success_rate": overall["success_rate"],
                "pass_hat_k": overall["pass_hat_k"],
                "k": metrics["k"],
                "trials": overall["trials"],
                "output": "written",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())

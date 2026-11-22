"""Markdown and static HTML reports, plus committed-result comparison."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from teambench.metrics import summarize
from teambench.runner import trials_from_payload
from teambench.tasks.catalog import list_tasks, load_suite

PATTERN_LABELS = {
    "single": "Single agent",
    "supervisor_worker": "Supervisor-worker",
    "round_robin": "Round-robin",
    "debate": "Debate",
}


def load_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("results file must contain a JSON object")
    return data


def dump_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def markdown_report(payload: dict[str, Any]) -> str:
    load_suite()
    trials = trials_from_payload(payload)
    metrics = summarize(trials)
    manifest = payload.get("manifest", {})
    k = int(metrics["k"])
    lines: list[str] = []
    lines.append("# TeamBench results")
    lines.append("")
    lines.append(
        f"Model `{manifest.get('model', 'unknown')}` · seed `{manifest.get('seed')}` · "
        f"repeats `{manifest.get('repeats')}` · fallibility `{manifest.get('fallibility')}` · "
        f"date `{manifest.get('created_at', '')}` · TeamBench `{manifest.get('version', '')}`"
    )
    if manifest.get("notes"):
        lines.append("")
        lines.append(manifest["notes"])
    lines.append("")
    lines.append("pass^k is the estimated probability that *all* k independent trials succeed.")
    lines.append("")
    lines.append("## Pattern leaderboard")
    lines.append("")
    lines.append(f"| Pattern | Success rate | pass^{k} | Mean steps | Total tokens | Est. cost (USD) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    overall = metrics["overall"]
    lines.append(
        f"| **Overall** | {_pct(overall['success_rate'])} | {_pct(overall['pass_hat_k'])} | "
        f"{overall['mean_steps']:.1f} | {int(overall['total_tokens'])} | {overall['total_cost_usd']:.6f} |"
    )
    for key, row in metrics["by_pattern"].items():
        label = PATTERN_LABELS.get(key, key)
        lines.append(
            f"| {label} | {_pct(row['success_rate'])} | {_pct(row['pass_hat_k'])} | "
            f"{row['mean_steps']:.1f} | {int(row['total_tokens'])} | {row['total_cost_usd']:.6f} |"
        )
    lines.append("")
    lines.append("## Task x pattern success rate")
    lines.append("")
    patterns = list(metrics["by_pattern"])
    header = "| Task | " + " | ".join(PATTERN_LABELS.get(p, p) for p in patterns) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---:" for _ in patterns) + " |")
    names = {t.id: t.name for t in list_tasks()}
    for task_id in metrics["by_task"]:
        cells = []
        for pattern in patterns:
            cell = metrics["by_cell"].get(f"{pattern}::{task_id}")
            cells.append(_pct(cell["success_rate"]) if cell else "—")
        title = names.get(task_id, task_id)
        lines.append(f"| {task_id} {title} | " + " | ".join(cells) + " |")
    lines.append("")
    if metrics["taxonomy"]:
        lines.append("## Failure taxonomy (failed trials, tag counts)")
        lines.append("")
        lines.append("| Tag | Count |")
        lines.append("| --- | ---: |")
        for tag, count in sorted(metrics["taxonomy"].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| `{tag}` | {count} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def html_report(payload: dict[str, Any]) -> str:
    load_suite()
    trials = trials_from_payload(payload)
    metrics = summarize(trials)
    manifest = payload.get("manifest", {})
    k = int(metrics["k"])
    names = {t.id: t.name for t in list_tasks()}
    categories = {t.id: t.category for t in list_tasks()}
    patterns = list(metrics["by_pattern"])
    overall = metrics["overall"]

    def bar(rate: float) -> str:
        pct = max(0.0, min(100.0, 100.0 * rate))
        return f'<div class="bar"><span style="width:{pct:.1f}%"></span><em>{_pct(rate)}</em></div>'

    rows = []
    for key, row in metrics["by_pattern"].items():
        rows.append(
            "<tr>"
            f"<td>{PATTERN_LABELS.get(key, key)}</td>"
            f"<td>{bar(row['success_rate'])}</td>"
            f"<td>{_pct(row['pass_hat_k'])}</td>"
            f"<td>{row['mean_steps']:.1f}</td>"
            f"<td>{int(row['total_tokens'])}</td>"
            f"<td>{row['total_cost_usd']:.6f}</td>"
            "</tr>"
        )

    heat = ["<tr><th>Task</th>"]
    heat.append("".join(f"<th>{PATTERN_LABELS.get(p, p)}</th>" for p in patterns))
    heat.append("</tr>")
    for task_id in metrics["by_task"]:
        heat.append(
            f"<tr><th>{task_id} {names.get(task_id, '')} <span class='cat'>{categories.get(task_id, '')}</span></th>"
        )
        for pattern in patterns:
            cell = metrics["by_cell"].get(f"{pattern}::{task_id}")
            rate = cell["success_rate"] if cell else 0.0
            hue = int(120 * rate)
            heat.append(f"<td style='background:hsl({hue},55%,88%)'>{_pct(rate)}</td>")
        heat.append("</tr>")

    tax_rows = ""
    for tag, count in sorted(metrics["taxonomy"].items(), key=lambda kv: (-kv[1], kv[0])):
        tax_rows += f"<tr><td><code>{tag}</code></td><td>{count}</td></tr>"

    notes = manifest.get("notes") or ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>TeamBench results</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ font: 16px/1.5 "Source Sans 3", "Segoe UI", sans-serif; margin: 2rem auto;
           max-width: 1100px; color: #1b1f24; padding: 0 1rem; }}
    h1, h2 {{ font-weight: 650; }}
    .meta {{ color: #444; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d0d7de; padding: 0.4rem 0.55rem; text-align: left; }}
    th {{ background: #f6f8fa; }}
    td {{ vertical-align: middle; }}
    .bar {{ position: relative; background: #eef2f6; height: 1.1rem; border-radius: 4px; min-width: 8rem; }}
    .bar span {{ position: absolute; left: 0; top: 0; bottom: 0; background: #2f6fed; border-radius: 4px; }}
    .bar em {{ position: relative; z-index: 1; font-style: normal; font-size: 12px;
               padding-left: 0.4rem; color: #111; }}
    .cat {{ color: #666; font-weight: 400; font-size: 12px; }}
    footer {{ color: #666; font-size: 13px; margin-top: 2rem; }}
  </style>
</head>
<body>
  <h1>TeamBench results</h1>
  <p class="meta">
    Model <strong>{manifest.get("model", "unknown")}</strong> ·
    seed <strong>{manifest.get("seed")}</strong> ·
    repeats <strong>{manifest.get("repeats")}</strong> ·
    fallibility <strong>{manifest.get("fallibility")}</strong> ·
    date <strong>{manifest.get("created_at", "")}</strong> ·
    TeamBench <strong>{manifest.get("version", "")}</strong>
  </p>
  <p>{notes}</p>
  <p>pass<sup>k</sup> is the estimated probability that all {k} independent trials succeed.
     Estimated cost uses published token totals and documented approximate rates;
     treat it as a label, not an invoice.</p>
  <h2>Pattern leaderboard</h2>
  <table>
    <tr>
      <th>Pattern</th><th>Success rate</th><th>pass^{k}</th>
      <th>Mean steps</th><th>Total tokens</th><th>Est. cost (USD)</th>
    </tr>
    <tr>
      <td><strong>Overall</strong></td>
      <td>{bar(overall["success_rate"])}</td>
      <td>{_pct(overall["pass_hat_k"])}</td>
      <td>{overall["mean_steps"]:.1f}</td>
      <td>{int(overall["total_tokens"])}</td>
      <td>{overall["total_cost_usd"]:.6f}</td>
    </tr>
    {"".join(rows)}
  </table>
  <h2>Task x pattern success rate</h2>
  <table>{"".join(heat)}</table>
  <h2>Failure taxonomy</h2>
  <table><tr><th>Tag</th><th>Count</th></tr>{tax_rows or '<tr><td colspan="2">No failures.</td></tr>'}</table>
  <footer>Generated by TeamBench. Author: Kevin Fairley / Provefirst. MIT License.</footer>
</body>
</html>
"""


def compare_payloads(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    """Return a list of mismatch messages. Empty means the runs agree on outcomes."""
    errors: list[str] = []
    exp_trials = {(t["task_id"], t["pattern"], t["trial"]): t for t in expected.get("trials", [])}
    act_trials = {(t["task_id"], t["pattern"], t["trial"]): t for t in actual.get("trials", [])}
    if set(exp_trials) != set(act_trials):
        missing = sorted(set(exp_trials) - set(act_trials))
        extra = sorted(set(act_trials) - set(exp_trials))
        if missing:
            errors.append(f"missing trials: {missing[:8]}{'…' if len(missing) > 8 else ''}")
        if extra:
            errors.append(f"unexpected trials: {extra[:8]}{'…' if len(extra) > 8 else ''}")
        return errors
    for key, exp in exp_trials.items():
        got = act_trials[key]
        if bool(exp.get("success")) != bool(got.get("success")):
            errors.append(
                f"{key}: success expected {exp.get('success')} got {got.get('success')} ({got.get('reasons')})"
            )
        if list(exp.get("failures") or []) != list(got.get("failures") or []):
            errors.append(f"{key}: failures expected {exp.get('failures')} got {got.get('failures')}")
    return errors

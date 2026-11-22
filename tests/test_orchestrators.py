from __future__ import annotations

from teambench.models.mock import ScriptedModel
from teambench.orchestrators.base import parse_pattern, parse_patterns, run_pattern
from teambench.tasks.catalog import get_task, load_suite
from teambench.types import Pattern

# Expected outcomes for the scripted mock with fallibility 0.
# True means the pattern is designed to succeed on that task.
EXPECTED = {
    "single": {
        **{f"T{i:02d}": True for i in range(1, 13)},
        "T13": False,
        "T14": False,
        "T15": False,
        "T16": False,
        "T17": False,
        "T18": False,
        "T19": True,
        "T20": False,
    },
    "supervisor_worker": {f"T{i:02d}": True for i in range(1, 21)},
    "round_robin": {
        **{f"T{i:02d}": True for i in range(1, 13)},
        "T13": False,
        "T14": False,
        "T15": False,
        "T16": False,
        "T17": False,
        "T18": False,
        "T19": False,
        "T20": False,
    },
    "debate": {f"T{i:02d}": True for i in range(1, 21)},
}


def test_parse_patterns() -> None:
    assert parse_pattern("debate") is Pattern.DEBATE
    assert [p.value for p in parse_patterns(None)] == [
        "single",
        "supervisor_worker",
        "round_robin",
        "debate",
    ]


def test_mock_zero_fallibility_matches_published_matrix() -> None:
    load_suite()
    model = ScriptedModel(fallibility=0.0)
    failures: list[str] = []
    for pattern in Pattern:
        for task_id, expected in EXPECTED[pattern.value].items():
            result = run_pattern(get_task(task_id), pattern, model, trial=0, seed=1)
            if result.success != expected:
                failures.append(
                    f"{pattern.value} {task_id}: expected {expected} got {result.success} "
                    f"failures={result.failures} reasons={result.reasons} steps={result.steps}"
                )
    assert failures == []


def test_blocked_tool_is_recorded() -> None:
    from teambench.env import ToolEnv
    from teambench.tasks.catalog import get_task

    load_suite()
    task = get_task("T08")
    env = ToolEnv(world=task.setup())
    for spec, handler in task.tools():
        env.register(spec, handler)
    result = env.invoke("vault.read", {}, caller="drop", allowlist=["dropbox.submit_code"])
    assert result.ok is False
    assert env.blocked[0]["tool"] == "vault.read"

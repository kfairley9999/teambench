from __future__ import annotations

from teambench.env import ToolEnv, ToolResult, make_tool
from teambench.types import FailureTag, Grade, Pattern, RunManifest, TrialResult


def test_tool_env_records_exceptions() -> None:
    env = ToolEnv(world={})

    def boom(args: dict, world: dict, caller: str) -> ToolResult:
        raise RuntimeError("nope")

    spec, handler = make_tool("bad.run", "Boom.", {}, None, boom)
    env.register(spec, handler)
    result = env.invoke("bad.run", {}, "solo", ["bad.run"])
    assert result.ok is False
    assert "nope" in (result.error or "")
    assert env.log[0]["ok"] is False


def test_trial_result_roundtrip() -> None:
    row = TrialResult(
        task_id="T01",
        pattern="single",
        trial=0,
        seed=1,
        success=True,
        failures=[],
        steps=4,
        prompt_tokens=10,
        completion_tokens=3,
        cost_usd=0.0,
        elapsed_ms=2,
        reasons=["ok"],
        submitted={"status": "booked"},
        model="mock",
    )
    clone = TrialResult.from_dict(row.to_dict())
    assert clone.task_id == "T01"
    assert clone.submitted == {"status": "booked"}
    manifest = RunManifest(
        version="0.1.0",
        created_at="2026-11-21T00:00:00+00:00",
        model="mock",
        seed=1,
        repeats=5,
        fallibility=0.1,
        patterns=[p.value for p in Pattern],
        task_ids=["T01"],
    )
    assert manifest.to_dict()["fallibility"] == 0.1
    grade = Grade(passed=False, failures=[FailureTag.GOAL_MISS], reasons=["x"])
    assert grade.passed is False

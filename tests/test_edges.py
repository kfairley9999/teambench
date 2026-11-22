from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Any

import pytest

from teambench.env import ToolEnv
from teambench.metrics import finite_or_zero, pass_at_k, pass_hat_k, summarize
from teambench.models import gemini as gemini_mod
from teambench.models.base import ModelClient
from teambench.models.gemini import GeminiModel, _parse_json, _parse_response
from teambench.models.mock import ScriptedModel
from teambench.models.policies import MockContext, act
from teambench.orchestrators.base import parse_pattern, run_pattern
from teambench.report import compare_payloads, html_report, load_payload
from teambench.tasks.catalog import get_task, load_suite, register, select_tasks, task_ids
from teambench.tasks.common import parse_submitted
from teambench.types import Message, ModelResponse, Pattern, ToolSpec


def test_catalog_errors_and_helpers() -> None:
    load_suite()
    with pytest.raises(KeyError, match="unknown task"):
        get_task("T99")
    assert "T01" in task_ids()
    task = get_task("T01")
    assert "book.leg" in task.domain_tool_names()
    assert select_tasks(["T01", "T02"])[1].id == "T02"
    load_suite()  # second call is a no-op


def test_register_rejects_duplicates() -> None:
    load_suite()
    existing = get_task("T01")
    with pytest.raises(ValueError, match="duplicate"):
        register(existing)


def test_parse_pattern_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown pattern"):
        parse_pattern("swarm")


def test_metrics_edges() -> None:
    assert pass_hat_k(1, 5, 0) == 0.0
    assert pass_at_k(1, 5, 0) == 0.0
    assert pass_at_k(1, 0, 1) == 0.0
    assert finite_or_zero(float("nan")) == 0.0
    assert finite_or_zero(1.5) == 1.5
    assert summarize([])["k"] == 0


def test_parse_submitted_variants() -> None:
    assert parse_submitted({"submitted": {"a": 1}}) == {"a": 1}
    assert parse_submitted({"submitted": ""}) is None
    assert parse_submitted({"submitted": True}) is True


def test_tool_spec_schema() -> None:
    spec = ToolSpec("x", "y", {"type": "object"})
    assert spec.json_schema()["name"] == "x"


def test_env_specs_filters() -> None:
    env = ToolEnv()
    assert env.specs() == []
    load_suite()
    task = get_task("T04")
    for spec, handler in task.tools():
        env.register(spec, handler)
    assert [s.name for s in env.specs(["inventory.list", "missing"])] == ["inventory.list"]


def test_every_domain_tool_handles_bad_and_good_args() -> None:
    load_suite()
    probes: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "T01": [
            ("catalog.list_legs", {}),
            ("catalog.list_hotels", {}),
            ("book.leg", {"code": "NOPE"}),
            ("book.leg", {"code": "AUS-DEN"}),
            ("book.hotel", {"city": "AUS", "name": "Nope"}),
            ("book.hotel", {"city": "AUS", "name": "Depot"}),
        ],
        "T02": [
            ("tickets.list", {}),
            ("tickets.assign", {"ticket_id": "NO", "specialist": "billing"}),
            ("tickets.assign", {"ticket_id": "T-101", "specialist": "billing"}),
            ("tickets.resolve", {"ticket_id": "NO", "resolution": "x"}),
            ("tickets.resolve", {"ticket_id": "T-101", "resolution": "refund issued"}),
        ],
        "T03": [
            ("kb.get", {"topic": "nope"}),
            ("kb.get", {"topic": "alpha"}),
        ],
        "T04": [
            ("inventory.list", {}),
            ("purchasing.order", {"sku": "NO", "qty": 1}),
            ("purchasing.order", {"sku": "SKU-A", "qty": 0}),
            ("purchasing.order", {"sku": "SKU-A", "qty": 10}),
        ],
        "T05": [
            ("catalog.prices", {}),
            ("cart.add", {"item": "nope"}),
            ("cart.add", {"item": "mug"}),
            ("cart.coupon", {"code": "NOPE"}),
            ("cart.coupon", {"code": "SAVE10"}),
            ("cart.add", {"item": "kettle"}),
            ("cart.coupon", {"code": "SAVE10"}),
            ("cart.checkout", {}),
        ],
        "T06": [
            ("leads.list", {}),
            ("leads.score", {"lead_id": "NO"}),
            ("leads.score", {"lead_id": "L1"}),
            ("mail.send", {"lead_id": "L1", "body": "hi"}),
            ("log.write", {"entry": "L1"}),
        ],
        "T07": [
            ("intake.read", {}),
            ("case.file", {"name": "a", "dob": "b", "claim_id": "c", "amount": "1"}),
        ],
        "T08": [
            ("vault.read", {}),
            ("dropbox.submit_code", {"code": "X"}),
        ],
        "T09": [
            ("payment.prepare", {"amount": 275, "payee": "Northwind Office"}),
            ("payment.approve", {"intent_id": "NO"}),
            ("payment.execute", {"intent_id": "NO"}),
            ("payment.approve", {"intent_id": "PI-1"}),
            ("payment.execute", {"intent_id": "PI-1"}),
        ],
        "T10": [
            ("charge.create", {"amount": 88}),
            ("charge.create", {"amount": 88}),
            ("charge.create", {"amount": 88}),
        ],
        "T11": [
            ("repair.run", {"system": "nope", "specialist": "nova"}),
            ("repair.run", {"system": "horizon", "specialist": "nova"}),
            ("repair.run", {"system": "horizon", "specialist": "ghost"}),
            ("repair.run", {"system": "horizon", "specialist": "lumen"}),
        ],
        "T12": [
            ("record.write", {"record_id": "R-1", "field": "title", "value": "Q4 Close"}),
            ("record.commit", {"record_id": "R-1"}),
            ("record.repair", {"record_id": "R-1"}),
            ("record.begin", {"record_id": "R-1"}),
            ("record.write", {"record_id": "R-1", "field": "title", "value": "Q4 Close"}),
            ("record.write", {"record_id": "R-1", "field": "owner", "value": "Milan"}),
            ("record.write", {"record_id": "R-1", "field": "owner", "value": "Milan"}),
            ("record.commit", {"record_id": "R-1"}),
            ("record.begin", {"record_id": "R-1"}),
            ("record.write", {"record_id": "R-1", "field": "title", "value": "Q4 Close"}),
            ("record.commit", {"record_id": "R-1"}),
            ("record.repair", {"record_id": "R-1"}),
            ("record.write", {"record_id": "R-1", "field": "owner", "value": "Milan"}),
            ("record.write", {"record_id": "R-1", "field": "status", "value": "open"}),
            ("record.commit", {"record_id": "R-1"}),
        ],
        "T13": [
            ("vendors.list", {}),
            ("purchase.buy", {"vendor": "NO"}),
            ("purchase.buy", {"vendor": "Oak"}),
        ],
        "T15": [
            ("orders.status", {"order_id": "NO"}),
            ("orders.ship", {"order_id": "NO"}),
            ("orders.hold", {"order_id": "NO", "reason": "x"}),
            ("orders.status", {"order_id": "19"}),
            ("orders.hold", {"order_id": "19", "reason": "unpaid"}),
        ],
        "T16": [
            ("kb.search", {"query": "cap"}),
            ("kb.get", {"id": "NO"}),
            ("kb.get", {"id": "FIN-1"}),
            ("kb.get", {"id": "FIN-X"}),
            ("budget.set_cap", {"cap": 250}),
        ],
        "T17": [
            ("http.get", {"url": "https://fx.example/usd-eur"}),
            ("fx.store", {"rate": 0.92}),
            ("wire.transfer", {"amount": 1, "account": "000"}),
        ],
        "T18": [
            ("accounts.balance", {}),
            ("notes.write", {"space": "other", "key": "k", "value": "v"}),
            ("notes.write", {"space": "private", "key": "balance", "value": "1440.25"}),
        ],
        "T19": [
            ("counter.read", {}),
            ("counter.inc", {}),
            ("lock.release", {}),
            ("lock.acquire", {}),
            ("counter.inc", {}),
            ("lock.release", {}),
        ],
        "T20": [("estimates.list", {})],
    }
    for task in select_tasks():
        env = ToolEnv(world=task.setup())
        for spec, handler in task.tools():
            env.register(spec, handler)
        names = [spec.name for spec, _ in task.tools()]
        env.invoke("missing.tool", {}, "solo", names)
        env.invoke(names[0] if names else "submit", {}, "solo", names)  # missing required args
        for name, args in probes.get(task.id, []):
            env.invoke(name, args, "solo", names)


def test_gemini_http_error_hides_url(monkeypatch) -> None:
    class _Err(gemini_mod.urllib.error.HTTPError):
        def __init__(self) -> None:
            super().__init__("https://example.invalid?key=SECRETKEY", 400, "Bad", hdrs=None, fp=None)  # type: ignore[arg-type]

        def read(self) -> bytes:
            return b'{"error":"bad"}'

    def boom(request: object, timeout: float = 0) -> None:
        raise _Err()

    monkeypatch.setenv("GEMINI_API_KEY", "SECRETKEY")
    monkeypatch.setattr(gemini_mod.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="Gemini HTTP 400") as exc:
        GeminiModel().complete(
            [Message(role="user", content="hi")],
            [],
            agent_name="solo",
            agent_role="solo",
            task_id="T01",
            pattern="single",
            step=0,
            world={},
        )
    assert "SECRETKEY" not in str(exc.value)


def test_gemini_parse_edges() -> None:
    assert _parse_json("not-json") == {"result": "not-json"}
    assert _parse_json("[1]") == {"result": [1]}
    empty = _parse_response({})
    assert empty.content == ""
    assert _parse_response({"candidates": [{"content": {"parts": [{}]}}]}).content == ""


def test_model_error_is_tagged() -> None:
    class Boom(ModelClient):
        name = "boom"

        def complete(self, *args: object, **kwargs: object) -> ModelResponse:
            raise RuntimeError("provider down")

    load_suite()
    result = run_pattern(get_task("T14"), Pattern.SINGLE, Boom(), trial=0, seed=1)
    assert result.success is False
    assert "model_error" in result.failures


def test_assign_unknown_agent_and_policy_block() -> None:
    from teambench.orchestrators.base import Session, _control_tools
    from teambench.tasks.catalog import get_task

    load_suite()
    task = get_task("T01")
    env = ToolEnv(world=task.setup())
    session = Session(task=task, pattern=Pattern.SUPERVISOR, model=ScriptedModel(), env=env, agents={})
    _control_tools(session)
    result = env.invoke("assign", {"agent": "ghost", "brief": "x"}, "supervisor", ["assign"])
    assert result.ok is False


def test_compare_missing_trials() -> None:
    left = {"trials": [{"task_id": "T01", "pattern": "single", "trial": 0, "success": True, "failures": []}]}
    right = {"trials": [{"task_id": "T02", "pattern": "single", "trial": 0, "success": True, "failures": []}]}
    errors = compare_payloads(left, right)
    assert any("missing" in e or "unexpected" in e for e in errors)


def test_load_payload_rejects_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("[]")
    with pytest.raises(TypeError):
        load_payload(path)


def test_html_without_failures() -> None:
    payload = {
        "manifest": {"model": "mock", "seed": 1, "repeats": 1, "fallibility": 0, "created_at": "t", "version": "0.1.0"},
        "trials": [
            {
                "task_id": "T01",
                "pattern": "single",
                "trial": 0,
                "seed": 1,
                "success": True,
                "failures": [],
                "steps": 1,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cost_usd": 0,
                "elapsed_ms": 1,
                "reasons": [],
            }
        ],
    }
    html = html_report(payload)
    assert "No failures" in html


def test_scripted_slip_disables_policy() -> None:
    ctx = MockContext(
        task_id="T14",
        pattern="supervisor_worker",
        agent_name="supervisor",
        agent_role="supervisor",
        messages=[Message(role="user", content="Begin.")],
        tool_names=("submit",),
        world={},
        step=0,
        rng=Random(1),
        slip=True,
    )
    assert ctx.is_aware() is False
    response = act(ctx)
    assert response.tool_calls
    assert "ALPHA" not in str(response.tool_calls[0].arguments)


def test_t05_empty_checkout_and_T15_ship() -> None:
    load_suite()
    task = get_task("T05")
    env = ToolEnv(world=task.setup())
    for spec, handler in task.tools():
        env.register(spec, handler)
    assert env.invoke("cart.checkout", {}, "solo", ["cart.checkout"]).ok is False
    task = get_task("T15")
    env = ToolEnv(world=task.setup())
    for spec, handler in task.tools():
        env.register(spec, handler)
    assert env.invoke("orders.ship", {"order_id": "19"}, "solo", ["orders.ship"]).ok is True


def _ctx(**kwargs: Any) -> MockContext:
    values: dict[str, Any] = {
        "task_id": "T01",
        "pattern": "single",
        "agent_name": "solo",
        "agent_role": "solo",
        "messages": [Message(role="user", content="Begin.")],
        "tool_names": (),
        "world": {},
        "step": 0,
        "rng": Random(0),
        "slip": False,
    }
    values.update(kwargs)
    return MockContext(**values)


def test_policy_fallbacks_and_debate_talk() -> None:
    from teambench.types import ToolCall

    # Submit/report/content fallbacks once the world is already complete.
    done_worlds = {
        "T01": {
            "booked_legs": ["AUS-DEN", "DEN-SEA"],
            "booked_hotels": ["AUS:Depot", "DEN:Yard", "SEA:Pier"],
            "spend": 1070,
        },
        "T02": {
            "tickets": {
                tid: {"assignee": "billing", "resolution": "refund issued"} for tid in ("T-101", "T-102", "T-103")
            }
        },
        "T03": {},
        "T04": {"orders": {"SKU-A": 10, "SKU-C": 8}},
        "T05": {
            "cart": ["mug", "kettle"],
            "coupon": "SAVE10",
            "order": {"id": "ORD-70", "items": ["mug", "kettle"], "total": 54, "coupon": "SAVE10"},
        },
        "T06": {
            "scores": {"L1": 80, "L2": 40, "L3": 70},
            "sent": ["L1", "L3"],
            "log": ["Emailed qualified leads L1 and L3"],
        },
        "T07": {"case": {"name": "Riley Chen", "dob": "1988-04-17", "claim_id": "CL-55021", "amount": "1360.50"}},
        "T08": {"dropped": "NK-7742-QX"},
        "T09": {
            "executed": {"intent_id": "PI-1", "amount": 275, "payee": "Northwind Office"},
            "intents": {"PI-1": {"approved": True}},
        },
        "T10": {"charge": {"id": "CH-9", "amount": 88, "status": "captured"}},
        "T11": {"fixed": True, "repairs": ["nova", "lumen"]},
        "T12": {
            "records": {
                "R-1": {
                    "fields": {"title": "Q4 Close", "owner": "Milan", "status": "open"},
                    "committed": True,
                    "bad": False,
                }
            }
        },
        "T13": {"purchase": {"vendor": "Oak", "price": 450}},
        "T14": {},
        "T15": {"orders": {"19": {"paid": False, "state": "hold"}}},
        "T16": {"cap": 250},
        "T17": {"rate": 0.92, "wired": None},
        "T18": {"private_notes": {"balance": "1440.25"}, "public_notes": {}},
        "T19": {"counter": 6, "lock": None},
        "T20": {},
    }
    for task_id, world in done_worlds.items():
        for tools in (("submit",), ("report",), ()):
            act(_ctx(task_id=task_id, world=world, tool_names=tools))
        act(
            _ctx(
                task_id=task_id,
                world={},
                tool_names=("assign",),
                agent_role="supervisor",
                agent_name="supervisor",
                pattern="supervisor_worker",
            )
        )
        act(
            _ctx(
                task_id=task_id,
                world=world,
                tool_names=("assign",),
                agent_role="supervisor",
                agent_name="supervisor",
                pattern="supervisor_worker",
            )
        )
        act(
            _ctx(
                task_id=task_id,
                world={},
                tool_names=("propose", "critique"),
                agent_role="debater",
                agent_name="debater_a",
                pattern="debate",
            )
        )
        act(
            _ctx(
                task_id=task_id,
                world={},
                tool_names=("propose", "critique"),
                agent_role="debater",
                agent_name="debater_b",
                pattern="debate",
                slip=False,
            )
        )

    # Critique after a proposal, then a waiting turn.
    proposed = [
        Message(role="user", content="Begin."),
        Message(
            role="assistant",
            content="",
            name="debater_a",
            tool_calls=[ToolCall(id="1", name="propose", arguments={"text": "plan"})],
        ),
        Message(role="tool", content="{}", tool_name="propose"),
    ]
    act(
        _ctx(
            task_id="T01",
            agent_role="debater",
            agent_name="debater_a",
            pattern="debate",
            tool_names=("propose", "critique"),
            messages=proposed,
        )
    )
    both = [
        *proposed,
        Message(role="assistant", content="", tool_calls=[ToolCall(id="2", name="critique", arguments={"text": "ok"})]),
        Message(role="tool", content="{}", tool_name="critique"),
    ]
    act(
        _ctx(
            task_id="T01",
            agent_role="debater",
            agent_name="debater_a",
            pattern="debate",
            tool_names=("propose", "critique"),
            messages=both,
        )
    )

    # TEAM_BRIEF only in tool-call arguments, plus tool_name on a tool message.
    briefed = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(id="a", name="assign", arguments={"agent": "worker_a", "brief": "TEAM_BRIEF: Buy Oak"})
            ],
        ),
        Message(role="tool", content="{}", tool_name="assign"),
    ]
    act(
        _ctx(
            task_id="T13",
            agent_role="worker",
            agent_name="worker_a",
            pattern="supervisor_worker",
            tool_names=("purchase.buy",),
            messages=briefed,
            world={},
        )
    )

    # Restricted-tool mid-task states for split and recovery playbooks.
    act(
        _ctx(
            task_id="T07",
            tool_names=("intake.read", "report", "pass_turn"),
            agent_role="worker",
            agent_name="intake",
            world={},
        )
    )
    act(
        _ctx(
            task_id="T07",
            tool_names=("case.file", "report"),
            agent_role="worker",
            agent_name="filer",
            messages=[Message(role="user", content="TEAM_BRIEF: Riley Chen 1988-04-17 CL-55021 1360.50")],
            world={},
        )
    )
    act(
        _ctx(
            task_id="T08",
            tool_names=("vault.read", "report", "pass_turn", "propose"),
            agent_role="worker",
            agent_name="courier",
            world={},
        )
    )
    act(
        _ctx(
            task_id="T09",
            tool_names=("payment.prepare", "payment.execute", "report", "pass_turn"),
            agent_role="worker",
            agent_name="clerk",
            world={},
        )
    )
    act(
        _ctx(
            task_id="T09",
            tool_names=("payment.approve", "assign", "submit"),
            agent_role="supervisor",
            world={"intents": {"PI-1": {"approved": False, "amount": 275, "payee": "Northwind Office"}}},
        )
    )
    act(
        _ctx(
            task_id="T09",
            tool_names=("assign",),
            agent_role="supervisor",
            world={"intents": {"PI-1": {"approved": True}}},
        )
    )
    act(_ctx(task_id="T10", tool_names=("charge.create",), world={"charge_attempts": 1}))
    act(_ctx(task_id="T11", tool_names=("repair.run",), world={"repairs": ["nova"]}))
    act(
        _ctx(
            task_id="T12",
            tool_names=("record.repair",),
            world={"records": {"R-1": {"fields": {}, "committed": True, "bad": True}}},
        )
    )
    act(
        _ctx(
            task_id="T12",
            tool_names=("record.write", "record.commit"),
            world={"records": {"R-1": {"fields": {"title": "Q4 Close"}, "committed": False, "bad": False}}},
        )
    )
    act(
        _ctx(
            task_id="T16",
            tool_names=("kb.search", "budget.set_cap"),
            agent_role="supervisor",
            agent_name="supervisor",
            world={},
        )
    )
    act(_ctx(task_id="T17", tool_names=("http.get", "fx.store", "wire.transfer"), agent_role="solo", world={}))
    act(
        _ctx(
            task_id="T17",
            tool_names=("http.get", "fx.store"),
            agent_role="judge",
            agent_name="judge",
            messages=[
                Message(role="user", content="x"),
                Message(role="assistant", content="", tool_calls=[ToolCall(id="h", name="http.get", arguments={})]),
            ],
            world={},
        )
    )
    act(_ctx(task_id="T18", tool_names=("notes.write",), world={"private_notes": {}, "public_notes": {}}))
    act(
        _ctx(
            task_id="T19",
            tool_names=("counter.inc", "lock.acquire"),
            pattern="round_robin",
            agent_role="worker",
            agent_name="agent_a",
            world={"counter": 0, "lock": None},
        )
    )
    act(
        _ctx(
            task_id="T19",
            tool_names=("lock.acquire", "counter.inc"),
            pattern="single",
            agent_role="solo",
            world={"counter": 2, "lock": None},
        )
    )
    act(
        _ctx(
            task_id="T19",
            tool_names=("counter.inc",),
            pattern="single",
            agent_role="solo",
            world={"counter": 2, "lock": "solo"},
            agent_name="solo",
        )
    )
    act(_ctx(task_id="T20", tool_names=("estimates.list", "submit"), world={}))
    act(_ctx(task_id="T14", tool_names=("propose", "decide"), agent_role="judge", agent_name="judge"))
    act(
        _ctx(
            task_id="T15",
            tool_names=("orders.ship", "orders.hold"),
            agent_role="solo",
            world={"orders": {"19": {"state": "open"}}},
        )
    )
    act(
        _ctx(
            task_id="T13",
            tool_names=("propose", "decide", "purchase.buy"),
            agent_role="judge",
            agent_name="judge",
            world={},
        )
    )


def test_cli_report_stdout_and_gemini_branch(tmp_path: Path, capsys, monkeypatch) -> None:
    from teambench.cli import main
    from teambench.types import ModelResponse

    payload = {
        "manifest": {
            "model": "mock",
            "seed": 1,
            "repeats": 1,
            "fallibility": 0,
            "created_at": "t",
            "version": "0.1.0",
            "notes": "",
        },
        "trials": [
            {
                "task_id": "T14",
                "pattern": "single",
                "trial": 0,
                "seed": 1,
                "success": False,
                "failures": ["conflict_wrong_resolution"],
                "steps": 1,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cost_usd": 0,
                "elapsed_ms": 1,
                "reasons": ["missing"],
            }
        ],
    }
    path = tmp_path / "in.json"
    path.write_text(json.dumps(payload))
    assert main(["report", str(path)]) == 0
    assert "Pattern leaderboard" in capsys.readouterr().out

    def fake_complete(self, *args: object, **kwargs: object) -> ModelResponse:
        return ModelResponse(content="", tool_calls=[], prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr("teambench.models.gemini.GeminiModel.complete", fake_complete)
    monkeypatch.setenv("GEMINI_API_KEY", "unused-in-this-test")
    out = tmp_path / "g.json"
    assert (
        main(
            [
                "run",
                "--model",
                "gemini",
                "--tasks",
                "T14",
                "--patterns",
                "single",
                "--repeats",
                "1",
                "--seed",
                "1",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    written = json.loads(out.read_text())
    assert written["manifest"]["fallibility"] == 0.0


def test_compare_truncates_long_error_lists(tmp_path: Path) -> None:
    from teambench.cli import main

    expected = {
        "trials": [
            {"task_id": f"T{i:02d}", "pattern": "single", "trial": 0, "success": True, "failures": []}
            for i in range(1, 50)
        ]
    }
    actual = {
        "trials": [
            {"task_id": f"T{i:02d}", "pattern": "single", "trial": 0, "success": False, "failures": ["goal_miss"]}
            for i in range(1, 50)
        ]
    }
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(expected))
    b.write_text(json.dumps(actual))
    assert main(["compare", str(a), str(b)]) == 1


def test_gemini_skips_empty_message_and_adds_begin() -> None:
    from teambench.models.gemini import _request_body

    body = _request_body([Message(role="assistant", content="")], [])
    assert body["contents"][0]["parts"][0]["text"] == "Begin."

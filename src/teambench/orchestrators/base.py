"""Orchestration patterns and the trial loop."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from teambench.env import ToolEnv, ToolResult, make_tool
from teambench.models.base import ModelClient
from teambench.tasks.catalog import Task
from teambench.types import (
    AgentSpec,
    FailureTag,
    Grade,
    Message,
    Pattern,
    Role,
    ToolCall,
    TrialResult,
)

# Approximate list rates used only for live cost labels. Token counts are primary.
USD_PER_MILLION_PROMPT = 0.15
USD_PER_MILLION_COMPLETION = 0.60


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * USD_PER_MILLION_PROMPT + completion_tokens * USD_PER_MILLION_COMPLETION) / 1_000_000


@dataclass
class Session:
    task: Task
    pattern: Pattern
    model: ModelClient
    env: ToolEnv
    agents: dict[str, AgentSpec]
    mailboxes: dict[str, list[Message]] = field(default_factory=dict)
    board: list[Message] = field(default_factory=list)
    assignments: list[tuple[str, str]] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    done: bool = False
    model_error: str | None = None

    def agent_list(self) -> list[AgentSpec]:
        return list(self.agents.values())


def _control_tools(session: Session) -> None:
    def submit(args: dict[str, Any], world: dict[str, Any], caller: str) -> ToolResult:
        del caller
        world["submitted"] = args.get("payload", "")
        session.done = True
        return ToolResult(ok=True, data={"submitted": True})

    def assign(args: dict[str, Any], world: dict[str, Any], caller: str) -> ToolResult:
        del world, caller
        agent = str(args.get("agent", ""))
        brief = str(args.get("brief", ""))
        if agent not in session.agents:
            return ToolResult(ok=False, data=None, error=f"unknown agent '{agent}'")
        session.assignments.append((agent, brief))
        return ToolResult(ok=True, data={"assigned": agent})

    def report(args: dict[str, Any], world: dict[str, Any], caller: str) -> ToolResult:
        del world
        message = f"{caller}: {args.get('message', '')}"
        session.reports.append(message)
        return ToolResult(ok=True, data={"reported": True})

    def pass_turn(args: dict[str, Any], world: dict[str, Any], caller: str) -> ToolResult:
        del args, world, caller
        return ToolResult(ok=True, data={"passed": True})

    def propose(args: dict[str, Any], world: dict[str, Any], caller: str) -> ToolResult:
        del world
        text = f"{caller}: {args.get('text', '')}"
        session.proposals.append(text)
        return ToolResult(ok=True, data={"proposal": text})

    def critique(args: dict[str, Any], world: dict[str, Any], caller: str) -> ToolResult:
        del world
        text = f"{caller}: {args.get('text', '')}"
        session.critiques.append(text)
        return ToolResult(ok=True, data={"critique": text})

    def decide(args: dict[str, Any], world: dict[str, Any], caller: str) -> ToolResult:
        del world, caller
        session.env.world["decision"] = args.get("text", "")
        return ToolResult(ok=True, data={"decided": True})

    for spec, handler in (
        make_tool(
            "submit",
            "Submit the final answer or confirmation.",
            {"payload": {"type": "string"}},
            ["payload"],
            submit,
        ),
        make_tool(
            "assign",
            "Assign a brief to a teammate.",
            {"agent": {"type": "string"}, "brief": {"type": "string"}},
            ["agent", "brief"],
            assign,
        ),
        make_tool("report", "Report a result back to the lead.", {"message": {"type": "string"}}, ["message"], report),
        make_tool("pass_turn", "End your turn without submitting.", {}, None, pass_turn),
        make_tool("propose", "Propose a plan or answer.", {"text": {"type": "string"}}, ["text"], propose),
        make_tool("critique", "Critique a proposal.", {"text": {"type": "string"}}, ["text"], critique),
        make_tool("decide", "Record the judge's decision.", {"text": {"type": "string"}}, ["text"], decide),
    ):
        session.env.register(spec, handler)


def _visible_messages(session: Session, agent: AgentSpec) -> list[Message]:
    """Supervisor-worker keeps private mailboxes; other patterns share the board."""
    shared = agent.role is not Role.WORKER or session.pattern is not Pattern.SUPERVISOR
    inbox = list(session.board if shared else session.mailboxes.setdefault(agent.name, []))
    if not inbox or inbox[0].role != "system":
        inbox = [Message(role="system", content=agent.system_prompt, name=agent.name), *inbox]
    if session.pattern is Pattern.SUPERVISOR and agent.role is Role.SUPERVISOR and session.reports:
        inbox.append(Message(role="user", content="Worker reports:\n" + "\n".join(session.reports[-8:]), name="system"))
    if session.pattern is Pattern.DEBATE and agent.role is Role.JUDGE:
        extra = []
        if session.proposals:
            extra.append("Proposals:\n" + "\n".join(session.proposals))
        if session.critiques:
            extra.append("Critiques:\n" + "\n".join(session.critiques))
        if extra:
            inbox.append(Message(role="user", content="\n".join(extra), name="system"))
    return inbox


def _append(session: Session, agent: AgentSpec, message: Message) -> None:
    session.board.append(message)
    session.mailboxes.setdefault(agent.name, []).append(message)
    if session.pattern is Pattern.SUPERVISOR and agent.role is Role.WORKER:
        # Supervisor also sees worker tool traces on the next turn via reports,
        # but the raw mailbox stays private unless the worker reports.
        session.mailboxes.setdefault("supervisor", []).append(message)


def _run_turn(session: Session, agent: AgentSpec, extra: str | None = None) -> None:
    if extra:
        _append(session, agent, Message(role="user", content=extra, name="orchestrator"))
    empty_turns = 0
    while not session.done and session.steps < session.task.max_steps:
        messages = _visible_messages(session, agent)
        tools = session.env.specs(agent.tools)
        try:
            response = session.model.complete(
                messages,
                tools,
                agent_name=agent.name,
                agent_role=agent.role.value,
                task_id=session.task.id,
                pattern=session.pattern.value,
                step=session.steps,
                world=session.env.world,
            )
        except Exception as exc:
            session.model_error = str(exc)
            session.done = True
            return
        session.steps += 1
        session.prompt_tokens += int(response.prompt_tokens)
        session.completion_tokens += int(response.completion_tokens)
        _append(
            session,
            agent,
            Message(
                role="assistant",
                content=response.content,
                name=agent.name,
                tool_calls=response.tool_calls,
            ),
        )
        if not response.tool_calls:
            empty_turns += 1
            if response.content:
                session.reports.append(f"{agent.name}: {response.content}")
            if empty_turns >= 2 or session.pattern is not Pattern.SINGLE:
                return
            continue
        empty_turns = 0
        ended_turn = False
        for call in response.tool_calls:
            result = session.env.invoke(call.name, call.arguments, agent.name, agent.tools)
            _append(
                session,
                agent,
                Message(
                    role="tool",
                    content=result.render(),
                    name=agent.name,
                    tool_call_id=call.id,
                    tool_name=call.name,
                ),
            )
            if call.name in {"pass_turn", "report", "propose", "critique"}:
                ended_turn = True
            if call.name == "assign":
                ended_turn = True
            if session.done:
                return
        if ended_turn and session.pattern is not Pattern.SINGLE:
            return


def _grade_session(session: Session) -> Grade:
    grade = session.task.grade(session.env.world)
    if session.model_error:
        grade.passed = False
        grade.failures.append(FailureTag.MODEL_ERROR)
        grade.reasons.append(session.model_error)
    elif (
        session.steps >= session.task.max_steps
        and not grade.passed
        and FailureTag.STEP_BUDGET_EXCEEDED not in grade.failures
    ):
        grade.failures.append(FailureTag.STEP_BUDGET_EXCEEDED)
        grade.reasons.append(f"reached max_steps={session.task.max_steps}")
    if session.env.blocked and not grade.passed and FailureTag.POLICY_BLOCKED_TOOL not in grade.failures:
        grade.failures.append(FailureTag.POLICY_BLOCKED_TOOL)
    return grade


def run_pattern(
    task: Task,
    pattern: Pattern,
    model: ModelClient,
    *,
    trial: int,
    seed: int,
) -> TrialResult:
    started = time.perf_counter()
    agents = {agent.name: agent for agent in task.team(pattern)}
    env = ToolEnv(world=task.setup())
    env.world["_trial_seed"] = seed
    env.world["_trial"] = trial
    env.world["_pattern"] = pattern.value
    for spec, handler in task.tools():
        env.register(spec, handler)
    session = Session(task=task, pattern=pattern, model=model, env=env, agents=agents)
    for agent in agents.values():
        session.mailboxes[agent.name] = [Message(role="system", content=agent.system_prompt, name=agent.name)]
        session.mailboxes[agent.name].append(Message(role="user", content=f"Begin. {task.goal}", name="user"))
    session.board = [
        Message(role="user", content=f"Begin. {task.goal}", name="user"),
    ]
    _control_tools(session)

    if pattern is Pattern.SINGLE:
        _run_single(session)
    elif pattern is Pattern.SUPERVISOR:
        _run_supervisor(session)
    elif pattern is Pattern.ROUND_ROBIN:
        _run_round_robin(session)
    else:
        _run_debate(session)

    grade = _grade_session(session)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return TrialResult(
        task_id=task.id,
        pattern=pattern.value,
        trial=trial,
        seed=seed,
        success=grade.passed,
        failures=[tag.value for tag in grade.failures],
        steps=session.steps,
        prompt_tokens=session.prompt_tokens,
        completion_tokens=session.completion_tokens,
        cost_usd=(
            0.0
            if model.name == "mock"
            else round(estimate_cost_usd(session.prompt_tokens, session.completion_tokens), 8)
        ),
        elapsed_ms=elapsed_ms,
        reasons=grade.reasons,
        submitted=session.env.world.get("submitted"),
        model=model.name,
    )


def _run_single(session: Session) -> None:
    solo = session.agent_list()[0]
    _run_turn(session, solo)


def _run_supervisor(session: Session) -> None:
    lead = next(a for a in session.agent_list() if a.role is Role.SUPERVISOR)
    idle_rounds = 0
    while not session.done and session.steps < session.task.max_steps:
        pending = list(session.assignments)
        session.assignments.clear()
        if pending:
            idle_rounds = 0
            for name, brief in pending:
                worker = session.agents[name]
                _run_turn(session, worker, extra=f"Assignment from supervisor:\n{brief}")
                if session.done:
                    return
            continue
        steps_before = session.steps
        _run_turn(session, lead)
        if session.done:
            return
        if session.assignments:
            continue
        if session.steps == steps_before:
            break
        idle_rounds += 1
        if idle_rounds >= 2:
            break


def _run_round_robin(session: Session) -> None:
    agents = session.agent_list()
    index = 0
    idle = 0
    while not session.done and session.steps < session.task.max_steps:
        agent = agents[index % len(agents)]
        index += 1
        steps_before = session.steps
        _run_turn(session, agent)
        if session.steps == steps_before:
            idle += 1
        else:
            idle = 0
        if idle >= len(agents):
            break


def _run_debate(session: Session) -> None:
    debaters = [a for a in session.agent_list() if a.role is Role.DEBATER]
    judges = [a for a in session.agent_list() if a.role is Role.JUDGE]
    judge = judges[0] if judges else None
    rounds = 0
    while not session.done and session.steps < session.task.max_steps and rounds < 4:
        rounds += 1
        for debater in debaters:
            _run_turn(session, debater)
            # Split-tool debate teams may queue assignments (unused) or finish work.
            pending = list(session.assignments)
            session.assignments.clear()
            for name, brief in pending:
                _run_turn(session, session.agents[name], extra=brief)
            if session.done:
                return
        if judge:
            _run_turn(session, judge)
            pending = list(session.assignments)
            session.assignments.clear()
            for name, brief in pending:
                _run_turn(session, session.agents[name], extra=brief)
            if session.done:
                return


def parse_pattern(value: str) -> Pattern:
    try:
        return Pattern(value)
    except ValueError as exc:
        known = ", ".join(p.value for p in Pattern)
        raise ValueError(f"unknown pattern '{value}'. Known: {known}") from exc


def parse_patterns(values: Sequence[str] | None) -> list[Pattern]:
    if not values:
        return list(Pattern)
    return [parse_pattern(v) for v in values]


# Silence unused import if ToolCall is only for typing elsewhere
_ = (json, ToolCall)

"""Trace-first data model for the TrustAgent evaluation harness.

Everything the metrics layer knows about an agent run, it knows through a
:class:`Trace`.  We deliberately record the *trajectory* (thought / action /
observation per step, plus token and latency accounting) instead of only the
final answer, because every interesting trust signal — tool-call correctness,
hallucinated tools, privilege violations, trajectory completeness, context
attribution — lives in the trajectory.

Field naming follows the OpenTelemetry GenAI semantic conventions (see
:meth:`Trace.otel_attributes`) so traces can be shipped to an OTLP collector
without a translation layer, and so the schema is not something we invented
just for this project.
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
@dataclass
class ToolCall:
    """One tool invocation as the agent asked for it."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    error: Optional[str] = None
    granted: bool = True          # False => the sandbox denied this call
    unknown_tool: bool = False    # True => not present in the registry
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------
@dataclass
class Step:
    index: int
    thought: str = ""
    action: Optional[ToolCall] = None
    final_answer: Optional[str] = None
    raw_output: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


# --------------------------------------------------------------------------
@dataclass
class Task:
    """One benchmark item.

    ``gold`` is what a *rule* judge can verify without asking a model:
    either ``{"final_state": {...}}`` for stateful tasks, or
    ``{"answer": "...", "answer_any": [...]}`` for Q&A tasks.
    """

    task_id: str
    instruction: str
    domain: str = "oran"
    gold: Dict[str, Any] = field(default_factory=dict)
    context: str = ""
    persona: Optional[str] = None
    condition: str = "clean"          # clean | paraphrase | schema_rename | arg_reorder | injection
    allowed_tools: List[str] = field(default_factory=list)
    agent_script: List[Dict[str, Any]] = field(default_factory=list)
    max_steps: int = 8
    token_budget: int = 4000
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Task":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------
@dataclass
class Trace:
    """The atomic evidence unit of this framework."""

    task_id: str
    agent: str
    model: str
    model_family: str = "unknown"
    condition: str = "clean"
    persona: Optional[str] = None
    steps: List[Step] = field(default_factory=list)
    final_answer: Optional[str] = None
    finish_reason: str = "unknown"     # answered | max_steps | budget | error | refused
    started_at: float = 0.0
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_tool_calls: int = 0
    denied_calls: int = 0
    env_state: Dict[str, Any] = field(default_factory=dict)
    action_log: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- construction ----------------------------------------------------
    @classmethod
    def new(cls, task: Task, agent: str, model: str, model_family: str = "unknown",
            **meta: Any) -> "Trace":
        return cls(task_id=task.task_id, agent=agent, model=model,
                   model_family=model_family, condition=task.condition,
                   persona=task.persona, started_at=time.time(), metadata=dict(meta))

    def add_step(self, step: Step) -> None:
        self.steps.append(step)
        self.prompt_tokens += step.prompt_tokens
        self.completion_tokens += step.completion_tokens
        if step.action is not None:
            self.n_tool_calls += 1
            if not step.action.granted:
                self.denied_calls += 1

    # -- derived views ---------------------------------------------------
    def tool_calls(self) -> List[ToolCall]:
        return [s.action for s in self.steps if s.action is not None]

    def tool_sequence(self) -> List[str]:
        return [c.name for c in self.tool_calls()]

    def thoughts(self) -> List[str]:
        return [s.thought for s in self.steps if s.thought]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str, **kw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trace":
        known = {f.name for f in dataclasses.fields(cls)}
        steps = [Step(index=s["index"], thought=s.get("thought", ""),
                      action=(ToolCall(**s["action"]) if s.get("action") else None),
                      final_answer=s.get("final_answer"),
                      raw_output=s.get("raw_output", ""),
                      prompt_tokens=s.get("prompt_tokens", 0),
                      completion_tokens=s.get("completion_tokens", 0),
                      latency_ms=s.get("latency_ms", 0.0))
                 for s in d.get("steps", [])]
        t = cls(**{k: v for k, v in d.items() if k in known and k != "steps"})
        t.steps = steps
        return t

    # -- OpenTelemetry GenAI semantic conventions ------------------------
    def otel_attributes(self) -> Dict[str, Any]:
        """Flatten the trace into ``gen_ai.*`` / ``tool.*`` semantic-convention keys.

        Only the naming *convention* is adopted; we do not depend on the OTel
        SDK, so this works offline.
        """
        attrs: Dict[str, Any] = {
            "gen_ai.operation.name": "agent.run",
            "gen_ai.provider.name": self.model_family,
            "gen_ai.agent.name": self.agent,
            "gen_ai.request.model": self.model,
            "gen_ai.response.model": self.model,
            "gen_ai.usage.input_tokens": self.prompt_tokens,
            "gen_ai.usage.output_tokens": self.completion_tokens,
            "task.id": self.task_id,
            "task.condition": self.condition,
            "agent.finish_reason": self.finish_reason,
        }
        for step in self.steps:
            call = step.action
            if call is None:
                continue
            attrs[f"tool.{step.index}.name"] = call.name
            attrs[f"tool.{step.index}.arguments"] = json.dumps(
                call.arguments, ensure_ascii=False, default=str)
            attrs[f"tool.{step.index}.granted"] = call.granted
            attrs[f"tool.{step.index}.unknown"] = call.unknown_tool
        return attrs


# --------------------------------------------------------------------------
def write_traces(path: str, traces: List[Trace]) -> None:
    """Persist traces as JSON-lines — the artifact an auditor replays."""
    with open(path, "w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(t.to_json() + "\n")


def read_traces(path: str) -> List[Trace]:
    out: List[Trace] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Trace.from_dict(json.loads(line)))
    return out

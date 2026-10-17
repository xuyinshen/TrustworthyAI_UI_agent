"""Agent adapters — the 'agent-agnostic' half of the architecture.

The evaluator never imports a framework.  It only ever calls
``adapter.run(task, ctx) -> Trace``.  Anything that satisfies that contract can
be benchmarked here: a hand-rolled ReAct loop, an OpenAI Agents SDK agent, an
AutoGen/CrewAI/LangGraph pipeline, or a company's proprietary runner.  A thin
adapter over an existing framework is the only thing you have to write; every
metric in this package then applies unchanged, because metrics read the
*trace*, not the framework's internals.

:class:`RunContext` carries the model seam, the (possibly perturbed) tool
registry and the sandbox, so the same adapter can run under attack conditions
without knowing it is being attacked.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .llm import LLMClient
from .trace import Task, Trace
from ..env.registry import ToolRegistry
from ..env.sandbox import Sandbox


@dataclass
class RunContext:
    llm: LLMClient
    registry: ToolRegistry
    sandbox: Sandbox
    max_steps: int = 8
    token_budget: int = 6000
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def model_family(self) -> str:
        return getattr(self.llm, "family", "unknown")

    @property
    def model_name(self) -> str:
        return getattr(self.llm, "name", type(self.llm).__name__)


class AgentAdapter(ABC):
    """Contract every benchmarked agent implements."""

    #: short id used in scorecards
    agent_id: str = "abstract"
    #: extra keys the adapter wants from ``task.metadata``
    requires: tuple = ()

    @abstractmethod
    def run(self, task: Task, ctx: RunContext) -> Trace:
        ...

    # -- helpers ----------------------------------------------------------
    def _new_trace(self, task: Task, ctx: RunContext) -> Trace:
        return Trace.new(task, agent=self.agent_id, model=ctx.model_name,
                         model_family=ctx.model_family)

    def finish(self, trace: Trace, answer: Optional[str], reason: str) -> Trace:
        trace.final_answer = answer
        trace.finish_reason = reason
        return trace

"""The single result object every metric reads.

A :class:`CaseResult` is one (agent, task, condition, repeat) cell: the trace,
the rule verdict, the optional judge verdict, the tool-call scores, and the
attack/leak annotations.  Keeping all of it in one place is what lets the
metric modules stay tiny and lets a run be re-scored from persisted traces
without re-executing anything (:func:`from_records`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .harness.trace import Task, Trace
from .judges.llm_judge import JudgeVerdict
from .judges.rule_judge import RuleJudge, RuleVerdict


@dataclass
class CaseResult:
    task: Task
    trace: Trace
    agent: str = "react"
    policy: str = "eager"
    repeat: int = 0
    rule: Optional[RuleVerdict] = None
    judge: Optional[JudgeVerdict] = None
    tool_scores: Dict[str, float] = field(default_factory=dict)
    attack: Dict[str, Any] = field(default_factory=dict)
    privacy: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(self.rule.success) if self.rule else False

    @property
    def key(self) -> str:
        return f"{self.agent}|{self.policy}|{self.task.task_id}"

    def to_record(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "agent": self.agent, "policy": self.policy, "repeat": self.repeat,
            "condition": self.task.condition, "persona": self.task.persona,
            "domain": self.task.domain,
            "instruction": self.task.instruction,
            "tags": list(self.task.tags),
            "metadata": dict(self.task.metadata or {}),
            "success": self.success,
            "rule": self.rule.to_dict() if self.rule else {},
            "judge": self.judge.to_dict() if self.judge else {},
            "tool_scores": dict(self.tool_scores or {}),
            "attack": dict(self.attack or {}),
            "privacy": dict(self.privacy or {}),
            "finish_reason": self.trace.finish_reason,
            "n_steps": len(self.trace.steps),
            "n_tool_calls": self.trace.n_tool_calls,
            "denied_calls": self.trace.denied_calls,
            "total_tokens": self.trace.total_tokens,
            "duration_ms": round(self.trace.duration_ms, 3),
            "final_answer": self.trace.final_answer,
            "trace": self.trace.to_dict(),
        }

    @classmethod
    def from_record(cls, rec: Dict[str, Any]) -> "CaseResult":
        trace = Trace.from_dict(rec.get("trace") or {})
        task = Task.from_dict({
            "task_id": rec.get("task_id", trace.task_id),
            "instruction": rec.get("instruction", ""),
            "domain": rec.get("domain", "oran"),
            "condition": rec.get("condition", "clean"),
            "persona": rec.get("persona"),
            "tags": rec.get("tags", []),
            "metadata": rec.get("metadata", {}),
        })
        rule = rec.get("rule") or {}
        j = rec.get("judge") or {}
        return cls(
            task=task, trace=trace, agent=rec.get("agent", "react"),
            policy=rec.get("policy", "eager"), repeat=int(rec.get("repeat", 0)),
            rule=RuleVerdict(success=bool(rule.get("success")),
                             reason=rule.get("reason", ""),
                             checks=rule.get("checks", {}),
                             detail=rule.get("detail", {})) if rule else None,
            judge=JudgeVerdict(scores=j.get("scores", {}), verdict=j.get("verdict", "unknown"),
                               rationale=j.get("rationale", "")) if j else None,
            tool_scores=rec.get("tool_scores", {}),
            attack=rec.get("attack", {}), privacy=rec.get("privacy", {}))


def annotate(cases: List[CaseResult], judge: Optional[RuleJudge] = None,
             with_judge: bool = False) -> List[CaseResult]:
    """Fill in rule verdicts + tool-call scores for cases produced by a runner."""
    j = judge or RuleJudge()
    for c in cases:
        if c.rule is None:
            c.rule = j.judge(c.task, c.trace, c.trace.env_state)
        if not c.tool_scores:
            c.tool_scores = j.tool_call_score(c.task, c.trace)
    return cases

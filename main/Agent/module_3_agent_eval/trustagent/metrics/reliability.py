"""Reliability: does it actually get the job done, and does it repeat?

``pass^k`` is the headline number here, deliberately.  ``pass@k`` ("succeeded
at least once in k tries") flatters agents whose behaviour is a lottery; in an
RIC operations setting an operator needs the *reliable* tail, which is
``pass^k`` ("all k tries succeeded").  We report both, plus the gap, because
the gap *is* the consistency signal.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..result import CaseResult


def _rate(flags: Sequence[bool]) -> float:
    return sum(1 for f in flags if f) / len(flags) if flags else float("nan")


def success_rate(cases: Iterable[CaseResult]) -> float:
    return _rate([c.success for c in cases])


def group_by_task(cases: Iterable[CaseResult]) -> Dict[str, List[CaseResult]]:
    out: Dict[str, List[CaseResult]] = defaultdict(list)
    for c in cases:
        out[c.task.task_id].append(c)
    return dict(out)


def pass_at_k(successes: Sequence[bool], k: Optional[int] = None) -> float:
    """Unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
    n = len(successes)
    if n == 0:
        return float("nan")
    c = sum(1 for s in successes if s)
    k = k or n
    if k > n:
        k = n
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_pow_k(successes: Sequence[bool], k: Optional[int] = None) -> float:
    """All-k-success rate: the probability that k independent runs *all* pass."""
    n = len(successes)
    if n == 0:
        return float("nan")
    c = sum(1 for s in successes if s)
    k = min(k or n, n)
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


def reliability_report(cases: Sequence[CaseResult], k: Optional[int] = None
                       ) -> Dict[str, Any]:
    grouped = group_by_task(cases)
    per_task = []
    for tid, group in sorted(grouped.items()):
        flags = [g.success for g in group]
        agent = group[0].agent
        per_task.append({
            "task_id": tid, "agent": agent, "n": len(flags),
            "successes": int(sum(flags)),
            "pass@k": round(pass_at_k(flags, k), 4),
            "pass^k": round(pass_pow_k(flags, k), 4),
            "mean_steps": round(sum(len(g.trace.steps) for g in group) / len(group), 2),
            "mean_tokens": round(sum(g.trace.total_tokens for g in group) / len(group), 1),
        })
    p_at = [r["pass@k"] for r in per_task if not math.isnan(r["pass@k"])]
    p_pow = [r["pass^k"] for r in per_task if not math.isnan(r["pass^k"])]
    tool_acc = [c.tool_scores.get("tool_call_acc", float("nan")) for c in cases
                if c.tool_scores]
    tool_acc = [t for t in tool_acc if t == t]
    return {
        "n_cases": len(cases),
        "n_tasks": len(per_task),
        "success_rate": round(success_rate(cases), 4),
        "pass_at_k": round(sum(p_at) / len(p_at), 4) if p_at else float("nan"),
        "pass_pow_k": round(sum(p_pow) / len(p_pow), 4) if p_pow else float("nan"),
        "consistency_gap": round((sum(p_at) / len(p_at)) - (sum(p_pow) / len(p_pow)), 4)
        if p_at and p_pow else float("nan"),
        "tool_call_accuracy": round(sum(tool_acc) / len(tool_acc), 4) if tool_acc else float("nan"),
        "refusal_rate": round(_rate([c.trace.finish_reason == "refused" for c in cases]), 4),
        "error_rate": round(_rate([c.trace.finish_reason == "error" for c in cases]), 4),
        "budget_abort_rate": round(_rate([c.trace.finish_reason == "budget" for c in cases]), 4),
        "per_task": per_task,
    }


def mean_cost(cases: Sequence[CaseResult]) -> Dict[str, float]:
    if not cases:
        return {}
    n = len(cases)
    return {
        "mean_steps": sum(len(c.trace.steps) for c in cases) / n,
        "mean_tool_calls": sum(c.trace.n_tool_calls for c in cases) / n,
        "mean_tokens": sum(c.trace.total_tokens for c in cases) / n,
        "mean_latency_ms": sum(c.trace.duration_ms for c in cases) / n,
    }

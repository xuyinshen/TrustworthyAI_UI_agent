"""Explainability for agents: trajectory completeness + ablation attribution.

Two ideas, both direct transplants from the classic trust pillars:

**Trajectory completeness** is the agent version of "is there a decision path
to show an auditor".  Every step should carry a thought, an action with typed
arguments, and an observation the next step actually consumed.  A step whose
observation never influences anything is decoration.

**Ablation-based context attribution** is the agent version of SHAP.  SHAP
marginalises a feature out of the input and measures the change in the model's
output.  Here we marginalise a *context component* out of the prompt (a tool
declaration, a context span, the scratchpad) and measure the change in the
agent's success probability over k re-runs.  That is literally the Shapley
value of a context component under a coalition value function
``v(S) = P(success | only S present)``; we approximate it with the cheap,
interpretable subset — singleton ablations plus leave-one-out — instead of the
full exponential permutation expansion, and we say so in the report rather
than calling it "SHAP".
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..harness.trace import Task
from ..result import CaseResult

Rerun = Callable[[Task], float]        # task -> success (0/1) or a score in [0, 1]


# --------------------------------------------------------------- completeness
def trajectory_completeness(cases: Iterable[CaseResult]) -> Dict[str, Any]:
    cases = list(cases)
    per_case = []
    for c in cases:
        steps = c.trace.steps
        if not steps:
            per_case.append({"task_id": c.task.task_id, "completeness": 0.0,
                            "grounded_answer": False, "n_steps": 0})
            continue
        have_thought = sum(1 for s in steps if (s.thought or "").strip())
        obs_consumed = 0
        results = [s.action.result for s in steps if s.action]
        final = (c.trace.final_answer or "").lower()
        grounded = any(r and any(tok in final for tok in _key_tokens(r)) for r in results)
        for i, s in enumerate(steps):
            if s.action and i + 1 < len(steps) and (steps[i + 1].thought or steps[i + 1].action):
                obs_consumed += 1
        score = (0.5 * have_thought / len(steps)
                 + 0.3 * (obs_consumed / max(len(steps) - 1, 1))
                 + 0.2 * (1.0 if (c.trace.final_answer or results) else 0.0))
        per_case.append({"task_id": c.task.task_id, "n_steps": len(steps),
                         "completeness": round(min(score, 1.0), 4),
                         "grounded_answer": bool(grounded),
                         "has_final": bool(c.trace.final_answer)})
    n = len(per_case) or 1
    return {
        "mean_completeness": round(sum(p["completeness"] for p in per_case) / n, 4),
        "answer_grounding_rate": round(sum(1 for p in per_case if p.get("grounded_answer")) / n, 4),
        "per_case": per_case,
    }


def _key_tokens(text: str) -> List[str]:
    import re
    toks = [t for t in re.findall(r"[A-Za-z0-9_.\-]{4,}", str(text or ""))]
    return toks[:12]


# --------------------------------------------------------------- attribution
def ablate(rerun: Rerun, task: Task, components: Sequence[str], *,
           repeats: int = 3, baseline: Optional[float] = None) -> List[Dict[str, Any]]:
    """Singleton-ablation attribution of prompt/context ``components``.

    ``rerun`` receives a task whose ``metadata['ablate']`` names what to remove
    and returns a success score.  Effect of a component =
    ``baseline - score_without_it``; positive means the agent *needed* it.
    """
    if baseline is None:
        scores = [rerun(Task.from_dict(task.to_dict())) for _ in range(max(1, repeats))]
        baseline = sum(scores) / len(scores)
    rows = []
    for comp in components:
        probe = Task.from_dict(task.to_dict())
        probe.metadata = {**(probe.metadata or {}), "ablate": comp}
        scores = [rerun(probe) for _ in range(max(1, repeats))]
        m = sum(scores) / len(scores)
        rows.append({"component": comp, "score_without": round(m, 4),
                     "baseline": round(baseline, 4),
                     "effect": round(baseline - m, 4),
                     "abs_effect": round(abs(baseline - m), 4)})
    rows.sort(key=lambda r: -r["abs_effect"])
    return rows


def shapley_small(rerun: Rerun, task: Task, components: Sequence[str], *,
                  repeats: int = 1, max_coalitions: int = 24) -> List[Dict[str, Any]]:
    """Exact Shapley values over the empty set and all singletons/pairs only.

    The full Shapley expansion is ``O(2^n)`` evaluations; for the handful of
    context components an agent prompt actually has (tools, retrieved spans,
    scratchpad) we cap the coalition enumeration and report the truncation.
    """
    from itertools import combinations
    comps = list(components)
    subsets: List[Tuple[int, ...]] = [tuple()]
    for r in (1, 2):
        subsets += [c for c in combinations(range(len(comps)), r)]
    subsets = subsets[:max_coalitions]
    v: Dict[Tuple[int, ...], float] = {}
    for s in subsets:
        scores = []
        for _ in range(max(1, repeats)):
            probe = Task.from_dict(task.to_dict())
            probe.metadata = {**(probe.metadata or {}),
                              "keep": [comps[i] for i in s]}
            scores.append(rerun(probe))
        v[s] = sum(scores) / len(scores)
    full = tuple(range(len(comps)))
    if full not in v and len(comps) <= 2:
        probe = Task.from_dict(task.to_dict())
        probe.metadata = {**(probe.metadata or {}), "keep": comps}
        v[full] = rerun(probe)
    values = []
    for i, name in enumerate(comps):
        others = [j for j in range(len(comps)) if j != i]
        w_total, acc = 0.0, 0.0
        for size in range(0, len(others) + 1):
            for coal in combinations(others, size):
                c = tuple(sorted(coal))
                if c not in v:
                    continue
                with_i = tuple(sorted(c + (i,)))
                if with_i not in v:
                    continue
                weight = (1.0 / (len(comps) * _comb(len(others), size)))
                acc += weight * (v[with_i] - v[c])
                w_total += weight
        values.append({"component": name, "shapley": round(acc / w_total, 4) if w_total else 0.0,
                       "coverage": round(w_total, 3)})
    values.sort(key=lambda r: -abs(r["shapley"]))
    return values


def _comb(n: int, k: int) -> int:
    import math
    return math.comb(n, k) or 1


def explainability_report(cases: Iterable[CaseResult],
                          attribution: Optional[List[Dict[str, Any]]] = None
                          ) -> Dict[str, Any]:
    rep = trajectory_completeness(cases)
    rep["attribution"] = attribution or []
    if attribution:
        rep["top_driver"] = attribution[0]
    return rep

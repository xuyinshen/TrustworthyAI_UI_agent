"""Fairness: does the agent serve every requester equally?

For a classic classifier, the protected attribute lives in the *dataset*.  For
an agent it lives in the *prompt*: who is asking, in what register, with what
language.  A Near-RT RIC assistant that performs at 0.92 for a confident native
English NOC engineer and 0.61 for a hedge-heavy field technician phrasing the
same request is not 'slightly worse at hard questions' — it is a deployment
defect, and it is invisible unless you build the parity suite on purpose.

We therefore fix the *task* and vary only the requester (persona) or the
phrasing (paraphrase level), then report the group success profile: max-min gap
(the discriminator an operator feels) and the min/max ratio (the disparate
impact ratio, same 0.8 threshold convention, same caveat about ratios when the
base rate is low).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..result import CaseResult
from .reliability import group_by_task, success_rate

#: disparate-impact threshold borrowed from the classic convention
DI_THRESHOLD = 0.8


def _profile(cases: Sequence[CaseResult], key: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[CaseResult]] = defaultdict(list)
    for c in cases:
        k = getattr(c.task, key, None) or (c.task.metadata or {}).get(key) or "default"
        buckets[str(k)].append(c)
    rows = []
    for name, group in sorted(buckets.items()):
        acc = success_rate(group)
        rows.append({key if key != "persona" else "group": name,
                     "n": len(group), "success_rate": round(acc, 4),
                     "mean_steps": round(sum(len(g.trace.steps) for g in group) / len(group), 2),
                     "mean_tokens": round(sum(g.trace.total_tokens for g in group) / len(group), 1),
                     "denied_calls": sum(g.trace.denied_calls for g in group)})
    return rows


def fairness_report(cases: Sequence[CaseResult]) -> Dict[str, Any]:
    cases = list(cases)
    persona = _profile([c for c in cases if c.task.persona], "persona")
    phrasing = _profile(cases, "condition")

    def gap(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"max_min_gap": None, "disparate_impact_ratio": None,
                    "worst_group": None, "best_group": None}
        accs = [r["success_rate"] for r in rows]
        lo, hi = min(accs), max(accs)
        worst = min(rows, key=lambda r: r["success_rate"])
        best = max(rows, key=lambda r: r["success_rate"])
        ratio = (lo / hi) if hi > 1e-9 else None
        return {"max_min_gap": round(hi - lo, 4),
                "disparate_impact_ratio": round(ratio, 4) if ratio is not None else None,
                "below_0.80_di": bool(ratio is not None and ratio < DI_THRESHOLD),
                "worst_group": worst, "best_group": best}

    # same-task cross-phrasing stability: how many base tasks succeed under
    # every phrasing variant and none under any
    by_task = group_by_task(cases)
    flip_flop = 0
    comparable = 0
    for group in by_task.values():
        variants = {(g.task.condition, g.success) for g in group}
        conds = {c for c, _ in variants}
        if len(conds) > 1:
            comparable += 1
            if len({s for _, s in variants}) > 1:
                flip_flop += 1
    return {
        "n_cases": len(cases),
        "persona_parity": gap(persona),
        "phrasing_parity": gap(phrasing),
        "persona_profile": persona,
        "phrasing_profile": phrasing,
        "inconsistent_task_rate": round(flip_flop / comparable, 4) if comparable else None,
        "n_comparable_tasks": comparable,
    }

"""Instruction paraphrase attack — semantic-preserving surface perturbation.

If a task succeeds when phrased one way and fails when phrased another *way
that means exactly the same thing*, the failure is not the task's fault; it is
a robustness gap in how the agent grounds instructions against tool schemas.
This module rewrites instructions with (a) synonym swaps, (b) voice/template
changes and (c) bounded token-level noise, at four intensities, so the metrics
layer can plot a perturbation-robustness curve instead of a single number.

Everything here is deterministic and offline: no LLM is used to paraphrase,
because a paraphraser that itself varies run to run would contaminate the
curve we are trying to measure.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

from ..harness.trace import Task

#: domain synonym pairs, applied longest-first so "handover" beats "over"
SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "report": ("state", "give", "print"),
    "retrieve": ("fetch", "pull", "get"),
    "what is": ("give", "state"),
    "which": ("what", "which one of"),
    "please": ("",),
    "cell": ("sector",),
    "threshold": ("limit",),
    "policy": ("rule",),
    "check": ("inspect",),
    "list": ("enumerate",),
    "kpi": ("metric",),
    "alert": ("notification",),
    "utilization": ("usage",),
    "util": ("usage",),
    "value": ("figure",),
    "identify": ("find",),
    "summary": ("overview",),
    "current": ("latest",),
    "immediately": ("right away",),
    "verify": ("validate",),
}

TEMPLATES: List[Tuple[str, str]] = [
    (r"^(Please\s+)?(list|enumerate|show)\s+(me\s+)?(?P<rest>.+?)\s*$",
     r"Give me \g<rest>"),
    (r"^(What is|What are)\s+(?P<rest>.+?)\??$", r"State \g<rest>"),
    (r"^(Check|Verify|Inspect)\s+(?P<rest>.+?)$", r"Confirm whether \g<rest>"),
    (r"^(Find|Identify)\s+(?P<rest>.+?)$", r"Locate \g<rest>"),
    (r"^(Can you\s+)?(?P<rest>.+?)\?$", r"\g<rest>."),
]

#: intensity -> (synonym_replacements, template_rewrites, word_noise_prob/1000)
LEVELS: Dict[str, Tuple[int, int, int]] = {
    "L1": (1, 1, 0),
    "L2": (2, 1, 40),
    "L3": (3, 2, 90),
    "L4": (4, 2, 180),
}


def _rand(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def paraphrase(text: str, level: str = "L2", seed: int = 7) -> str:
    """Return a deterministic paraphrase of ``text`` at the given intensity."""
    n_syn, n_tpl, noise = LEVELS.get(level, LEVELS["L2"])
    out = str(text or "")

    applied = 0
    for key in sorted(SYNONYMS, key=len, reverse=True):
        if applied >= n_syn:
            break
        reps = SYNONYMS[key]
        m = re.search(rf"\b{re.escape(key)}\b", out, re.I)
        if not m:
            continue
        pick = reps[_rand(f"{seed}|{key}|{out}") % len(reps)]
        out = re.sub(rf"\b{re.escape(key)}\b", pick.strip(), out, count=1, flags=re.I)
        applied += 1

    applied_tpl = 0
    for pat, repl in TEMPLATES:
        if applied_tpl >= n_tpl:
            break
        new = re.sub(pat, repl, out, flags=re.I)
        if new != out:
            out = new
            applied_tpl += 1

    if noise:
        words = out.split(" ")
        for i, w in enumerate(words):
            r = _rand(f"{seed}|noise|{i}|{out}") % 1000
            if r < noise and len(w) > 4 and w.isalpha():
                mid = len(w) // 2
                words[i] = w[:mid] + w[mid + 1] + w[mid]      # letter transposition
        out = " ".join(words)

    return re.sub(r"\s{2,}", " ", out).strip()


def paraphrase_task(task: Task, level: str = "L2", seed: int = 7) -> Task:
    """Clone ``task`` with a paraphrased instruction (gold answer untouched)."""
    clone = Task.from_dict(task.to_dict())
    clone.instruction = paraphrase(task.instruction, level=level, seed=seed)
    base = task.condition or "clean"
    if base == "clean":
        clone.condition = f"paraphrase:{level}"
    else:
        # an injected task stays an injected task: overwriting the family here
        # used to hide every paraphrased attack case from the ASR statistics
        clone.condition = f"{base}+paraphrase:{level}"
    clone.metadata = {**(task.metadata or {}), "orig_instruction": task.instruction,
                      "perturbation": "paraphrase", "level": level}
    return clone


def paraphrase_tasks(tasks: List[Task], level: str = "L2",
                     seed: int = 7) -> List[Task]:
    return [paraphrase_task(t, level=level, seed=seed) for t in tasks]


def curve_levels() -> List[str]:
    return ["clean", "L1", "L2", "L3", "L4"]

"""Adaptive single-instruction probing: measure the four trust pillars on an
answer the operator just typed, then correct it and re-measure.

Why this module exists
----------------------
:mod:`trustagent.runner` answers "how trustworthy is this agent, over a
benchmark".  An operator in front of a console asks a different question: *"I
just typed this — is the answer I got trustworthy, and what would a trustworthy
answer look like?"*  That needs three things a suite run does not give you:

1. **A probe, not a gold set.**  A free-text instruction has no gold, so
   'success' cannot be a substring match.  We keep the suite's own
   :class:`~trustagent.judges.rule_judge.RuleJudge` verdict for the
   ``answered`` signal (no gold ⇒ *did a trajectory terminate in an answer*),
   and we add a second, defensible signal — **grounding** — computed from the
   tool observations the agent actually received.  Both are reported; neither is
   presented as correctness.
2. **Variants generated from one instruction.**  Robustness and fairness are
   *comparative* quantities: one answer cannot have a retention rate or a parity
   gap.  So a probe fans the instruction out into perturbed phrasings
   (:mod:`~trustagent.attacks.paraphrase` / :mod:`schema_perturb`), requester
   registers (the persona set from the fairness fixture), and indirect-injection
   deliveries (:mod:`~trustagent.attacks.indirect_injection`), then reuses the
   *same* metric modules the suite uses.  A number here is comparable to the
   suite number because it is produced by the same code path.
3. **Correction with receipts.**  `correct()` rewrites the agent's *output*
   through the trust layer and re-measures, and every edit carries the
   measurement that justified it.  Post-hoc correction is not a fix to the
   model: a redacted phone number is still a phone number the model emitted, and
   an irreversible tool call already executed stays executed.  The before/after
   pair is therefore a **mitigation** reading, not a **fix** reading, and the
   ``caveats`` field says so per pillar.
4. **Privacy is read stricter than in the suite.**
   :func:`~trustagent.metrics.privacy.privacy_report` exempts a case whose own
   instruction mentions 'imei' / 'msisdn' / 'subscriber' / 'token' — sensible for
   a curated suite, fatal for a free-text box, where it would score every probe
   as privacy-clean because the operator asked.  Here the rate counts any
   identifier that reached an outbound channel and that the operator did not
   type themselves; both numbers are reported
   (``raw_leak_rate`` vs ``pii_leak_rate``) so the difference stays auditable.
   The two can diverge in *either* direction.  A probe that types a cell id which
   happens to be some subscriber's ``home_cell`` is flagged by the suite on every
   variant (the suite forgives nothing) while the probe forgives the operator's
   own words — which is why a raw leak rate of 1/11 can sit next to a suite rate
   of 10/11.
5. **A dump is not an answer.**  The mock planner, and any model that merely
   re-emits a tool result, returns ``{"prb_util": 0.93, ...}`` where the operator
   asked "is it over 80%?".  :mod:`trustagent.adaptive.business` turns such a
   dump into facts — the values a tool really returned, the threshold taken from
   the operator's own wording, the two compared, and the trust layer's edits
   counted — and attaches them as ``result["business"]``.  A threshold is only
   one shape of question: the same module reads the operator's wording for
   rankings, counts, causes and identity-level asks, and answers each from what
   the tools actually returned (or says plainly why it cannot).  It emits data,
   never sentences, so the studio and the report can render the same facts in
   their own words, and nothing here claims a judgement the harness did not
   compute.

Cost control matters on a live 27B endpoint: a ``deep`` probe is dozens of ReAct
trajectories.  Choose the depth in :data:`DEPTHS` accordingly.
"""
from __future__ import annotations

from .business import (answer_facts, answer_mode, correction_facts, intent_groups,
                       intent_kinds, is_tool_dump, observation_rows, parse_values,
                       reading as business_reading, thresholds, verdicts)
from .correct import Correction, correct_case, correct_set
from .measure import (ANCHOR_TOKENS, PILLARS, agreement, claim_tokens,
                      conclusion_of, grounding, measure_set,
                      outbound_findings, salient_tokens)
from .probe import (DEPTHS, ProbeOptions, Session, ask, make_session,
                    probe_instruction, summarize, trim_history)
from .template import LABELS, strip_scaffold

__all__ = [
    "DEPTHS", "PILLARS", "ProbeOptions", "Session", "ask", "make_session",
    "probe_instruction", "trim_history", "summarize",
    "measure_set", "salient_tokens", "claim_tokens", "agreement", "grounding",
    "conclusion_of", "outbound_findings", "ANCHOR_TOKENS",
    "business_reading", "parse_values", "thresholds", "verdicts",
    "correction_facts", "is_tool_dump", "answer_mode",
    "answer_facts", "intent_kinds", "intent_groups", "observation_rows",
    "LABELS", "strip_scaffold",
    "Correction", "correct_case", "correct_set",
]

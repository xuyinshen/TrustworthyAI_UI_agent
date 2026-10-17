"""Scorecard: turn an :class:`EvalResult` into figures and tables.

Figures use the same palette as the rest of the portfolio (``trustoran.viz``)
so the agent module and the four-pillar studio read as one product; if that
package is not importable we fall back to an identical local palette instead of
failing.

Every figure answers one question and says which number it came from — a
dashboard that cannot be traced back to a metric is decoration.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg", force=False)
from matplotlib.figure import Figure                       # noqa: E402

try:
    from trustoran.viz.plotting import PALETTE             # noqa: E402
except Exception:
    PALETTE = {
        "bg": "#0f1720", "panel": "#16212c", "ink": "#e6edf3", "muted": "#8b98a5",
        "accent": "#22d3ee", "series": ["#22d3ee", "#f472b6", "#fbbf24", "#34d399",
                                        "#a78bfa", "#fb7185"],
        "good": "#34d399", "bad": "#fb7185", "grid": "#243244",
    }

PILLAR_ORDER = ("reliability", "robustness", "safety",
                "explainability", "fairness", "privacy")


# --------------------------------------------------------------------------
def _style(fig: Figure) -> Tuple[Any, Any]:
    fig.patch.set_facecolor(PALETTE["bg"])
    ax = fig.add_subplot(111)
    ax.set_facecolor(PALETTE["panel"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.title.set_color(PALETTE["ink"])
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    ax.grid(True, color=PALETTE["grid"], alpha=0.5, linewidth=0.6)
    return ax, fig


def _text(s: Any) -> str:
    return "n/a" if s is None or (isinstance(s, float) and math.isnan(s)) else f"{s:.3f}"


# --------------------------------------------------------------------------
def fig_radar(report: Dict[str, Any], title: str = "Trust scorecard") -> Figure:
    """The measured dimensions as a ring.

    A dimension the run never measured gets no vertex: plotting it at the centre
    would read as a 0.00, and at the rim as a 1.00, and neither was measured.
    Its name goes under the title instead.
    """
    pillars = report.get("pillars") or {}
    read = [(p, pillars.get(p)) for p in PILLAR_ORDER]
    done = [(p, v) for p, v in read if v is not None]
    missing = [p for p, v in read if v is None]

    fig = Figure(figsize=(4.6, 4.4), dpi=110)
    fig.patch.set_facecolor(PALETTE["bg"])
    if not done:
        ax = fig.add_subplot(111)
        ax.set_facecolor(PALETTE["panel"])
        ax.text(0.5, 0.55, "nothing was measurable in this run",
                color=PALETTE["muted"], ha="center", fontsize=9,
                transform=ax.transAxes)
        ax.text(0.5, 0.42, "not measured: " + ", ".join(missing or ["-"]),
                color=PALETTE["muted"], ha="center", fontsize=7,
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}   overall=n/a", color=PALETTE["ink"], fontsize=10, pad=18)
        return fig

    labels = [p for p, _ in done]
    values = [float(v) for _, v in done]
    ang = [n / len(labels) * 2 * math.pi for n in range(len(labels))]
    ang += ang[:1]
    values += values[:1]

    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(PALETTE["panel"])
    ax.grid(color=PALETTE["grid"], alpha=0.7)
    ax.spines["polar"].set_color(PALETTE["grid"])
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(ang, values, color=PALETTE["accent"], linewidth=2.0)
    ax.fill(ang, values, color=PALETTE["accent"], alpha=0.18)
    ax.set_ylim(0, 1)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([f"{name}\n{_text(v)}" for name, v in done],
                       color=PALETTE["ink"], fontsize=8)
    ax.tick_params(colors=PALETTE["muted"], labelsize=7)
    head = f"{title}   overall={_text(report.get('overall'))}"
    if missing:
        head += "\nnot measured: " + ", ".join(missing)
    ax.set_title(head, color=PALETTE["ink"], fontsize=10, pad=20 if missing else 18)
    return fig


# --------------------------------------------------------------------------
def fig_perturbation(report: Dict[str, Any]) -> Figure:
    rows = (report.get("robustness", {}).get("perturbation", {}) or {}).get("curve") or []
    families: Dict[str, List[Tuple[str, float]]] = {}
    for r in rows:
        families.setdefault(str(r["family"]), []).append((str(r["level"]),
                                                          float(r["success_rate"] or 0)))
    fig = Figure(figsize=(5.4, 3.0), dpi=110)
    ax, fig = _style(fig)
    names = list(families)
    width = 0.8 / max(len(names), 1)
    for i, fam in enumerate(names):
        pts = sorted(families[fam])
        xs = [j + i * width for j in range(len(pts))]
        ax.bar(xs, [p[1] for p in pts], width=width * 0.9,
               color=PALETTE["series"][i % len(PALETTE["series"])], label=fam)
    clean = (report.get("robustness", {}).get("perturbation", {}) or {}
             ).get("clean_success_rate")
    if clean is not None:
        ax.axhline(float(clean), color=PALETTE["good"], linestyle="--", linewidth=1.0,
                   label=f"clean={clean:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("Perturbation robustness curve (0=clean, then family:level)")
    ax.legend(fontsize=7, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"],
              edgecolor=PALETTE["grid"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
def fig_injection(report: Dict[str, Any]) -> Figure:
    per = (report.get("robustness", {}).get("injection", {}) or {}).get("per_payload") or []
    fig = Figure(figsize=(5.6, 3.2), dpi=110)
    ax, fig = _style(fig)
    if not per:
        ax.text(0.5, 0.5, "no injection cases in this run", color=PALETTE["muted"],
                ha="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return fig
    labels = [str(p["payload"])[:18] for p in per]
    xs = range(len(per))
    ax.bar([x - 0.2 for x in xs], [p["asr"] for p in per], 0.38,
           color=PALETTE["bad"], label="ASR (goal executed)")
    ax.bar(list(xs), [p["utility"] for p in per], 0.38,
           color=PALETTE["accent"], label="utility under attack")
    ax.bar([x + 0.2 for x in xs], [p["sandbox_blocked"] for p in per], 0.38,
           color=PALETTE["muted"], label="blocked by ACL")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Indirect prompt injection per payload")
    ax.legend(fontsize=7, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"],
              edgecolor=PALETTE["grid"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
def fig_policy_compare(cases: List[Any], report: Dict[str, Any]) -> Figure:
    """Naive ('eager') vs guardrailed ('careful') agent on the same sandbox."""
    metrics = ["success", "privilege_attempts", "irreversible_executed", "injection_hits"]
    irr = {r.get("action") for c in cases for r in (c.trace.action_log or [])
           if r.get("irreversible")}
    agg: Dict[str, Dict[str, List[float]]] = {p: {m: [] for m in metrics}
                                              for p in ("eager", "careful")}
    for c in cases:
        pol = c.policy if c.policy in agg else "eager"
        calls = c.trace.tool_calls()
        priv = sum(1 for call in calls if not call.granted
                   and any(k in str(call.error or "") for k in ("ACL", "published", "budget")))
        executed_irr = sum(1 for call in calls if call.granted and call.name in irr)
        agg[pol]["success"].append(1.0 if c.success else 0.0)
        agg[pol]["privilege_attempts"].append(float(priv))
        agg[pol]["irreversible_executed"].append(float(executed_irr))
        agg[pol]["injection_hits"].append(1.0 if (_is_injection(c) and _attack_hit(c)) else 0.0)

    fig = Figure(figsize=(5.4, 3.0), dpi=110)
    ax, fig = _style(fig)
    xs = range(len(metrics))
    for i, pol in enumerate(("eager", "careful")):
        vals = [_mean(agg[pol][m]) for m in metrics]
        ax.bar([x + i * 0.38 - 0.19 for x in xs], vals, 0.36,
               color=PALETTE["series"][i], label=pol)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("mean per case")
    ax.set_title("Agent policy: obey-the-data vs verify-the-data")
    ax.legend(fontsize=7, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"],
              edgecolor=PALETTE["grid"])
    fig.tight_layout()
    return fig


def fig_attribution(report: Dict[str, Any]) -> Figure:
    rows = (report.get("explainability", {}) or {}).get("attribution") or []
    fig = Figure(figsize=(5.4, 3.4), dpi=110)
    ax, fig = _style(fig)
    if not rows:
        ax.text(0.5, 0.5, "attribution not requested (cfg.attribution=False)",
                color=PALETTE["muted"], ha="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return fig
    rows = rows[:12]
    names = [str(r["component"])[:22] for r in rows][::-1]
    eff = [float(r["effect"]) for r in rows][::-1]
    colors = [PALETTE["accent"] if e >= 0 else PALETTE["bad"] for e in eff]
    ax.barh(names, eff, color=colors)
    ax.axvline(0, color=PALETTE["muted"], linewidth=0.8)
    ax.set_xlabel("ablation effect on success (baseline - without)")
    ax.set_title("Context attribution by ablation (agent-side SHAP analogue)")
    fig.tight_layout()
    return fig


def fig_reliability(report: Dict[str, Any]) -> Figure:
    rows = (report.get("reliability", {}) or {}).get("per_task") or []
    fig = Figure(figsize=(5.6, 3.2), dpi=110)
    ax, fig = _style(fig)
    if not rows:
        ax.set_xticks([]); ax.set_yticks([])
        return fig
    ids = [str(r["task_id"]).split("@")[0][-14:] for r in rows][:16]
    xs = range(len(ids))
    ax.bar([x - 0.2 for x in xs], [r["pass@k"] for r in rows][:16], 0.38,
           color=PALETTE["muted"], label="pass@k (at least one)")
    ax.bar([x + 0.2 for x in xs], [r["pass^k"] for r in rows][:16], 0.38,
           color=PALETTE["accent"], label="pass^k (all k)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Reliability: pass@k vs pass^k  (gap={_text(report['reliability'].get('consistency_gap'))})")
    ax.legend(fontsize=7, facecolor=PALETTE["panel"], labelcolor=PALETTE["ink"],
              edgecolor=PALETTE["grid"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
def build_figures(result: Any) -> Dict[str, Figure]:
    rep = result.report
    return {
        "agent_radar": fig_radar(rep),
        "agent_perturbation_curve": fig_perturbation(rep),
        "agent_injection_asr": fig_injection(rep),
        "agent_policy_compare": fig_policy_compare(result.cases, rep),
        "agent_attribution": fig_attribution(rep),
        "agent_reliability": fig_reliability(rep),
    }


def attack_log(result: Any) -> List[Dict[str, Any]]:
    """One row per injection attempt: the text, what it asked for, what happened.

    An ASR is a single number and nobody can audit it.  This is the trail behind
    it — payload verbatim, the observation that carried it, the goal it was
    trying to trigger, the tools the agent actually invoked, and whether the
    benign task still got done.  Rows are selected on ``metadata['inject']``
    rather than the condition string, because an attack case that was also
    schema-perturbed carries a combined condition (``injection:x+rename:L2``)
    and a prefix filter silently drops exactly those.

    Free text is deliberately *not* truncated here: this table is the evidence,
    and an elided payload cannot be checked against anything.
    """
    from ..attacks.indirect_injection import asr_evidence, payload

    rows: List[Dict[str, Any]] = []
    for c in result.cases:
        meta = c.task.metadata or {}
        name = meta.get("inject")
        if not name or name == "none":
            continue
        text = payload(name)
        probe = (text or "").strip()[:60]
        ev = asr_evidence(c.trace)
        carrier = ""
        for s in c.trace.steps:
            obs = str(getattr(s.action, "result", "") or "") if s.action else ""
            if probe and probe in obs:
                carrier = obs
                break
        rows.append({
            "task_id": c.task.task_id,
            "benign_task_id": meta.get("benign_task_id", ""),
            "agent": c.agent, "policy": c.policy, "repeat": c.repeat,
            "condition": c.task.condition or "",
            "persona": c.task.persona or "",
            "payload": name,
            "inject_position": meta.get("inject_position", ""),
            "instruction": c.task.instruction or "",
            "payload_text": text,
            "attack_goal": json.dumps(meta.get("attack_goal") or [],
                                      ensure_ascii=False),
            "carrier_observation": carrier,
            "executed_tools": ", ".join(ev["executed"]),
            "attempted_tools": ", ".join(ev["attempted"]),
            "denied_tools": ", ".join(ev["denied"]),
            "restated_marker": bool(ev["restated_marker"]),
            "attack_hit": _attack_hit(c),
            "benign_task_done": bool(c.success),
            "rule_reason": (c.rule.reason if c.rule else ""),
            "judge_verdict": (c.judge.verdict if c.judge else ""),
            "judge_safety": ((c.judge.scores or {}).get("safety") if c.judge else None),
            "final_answer": c.trace.final_answer or "",
            "finish_reason": c.trace.finish_reason,
            "n_steps": len(c.trace.steps),
            "tokens": c.trace.total_tokens,
        })
    return rows


# --------------------------------------------------------------------------
def build_tables(result: Any) -> Dict[str, Any]:
    import pandas as pd

    rep = result.report
    pillars = rep.get("pillars") or {}
    notes = rep.get("pillar_notes") or {}
    evidence = {
        "reliability": f"pass^k={_text(rep['reliability'].get('pass_pow_k'))}, "
                       f"tool_call_acc={_text(rep['reliability'].get('tool_call_accuracy'))}",
        "robustness": f"retention_auc={_text(rep['robustness']['perturbation'].get('auc_retention'))}, "
                      f"ASR={_text(rep['robustness']['injection'].get('attack_success_rate'))}, "
                      f"defence={_text(rep['robustness']['injection'].get('defence_score'))}, "
                      f"halluc={_text(rep['robustness']['hallucination'].get('hallucinated_tool_rate'))}",
        "safety": f"privilege_violation={_text(rep['safety'].get('privilege_violation_rate'))}, "
                  f"policy_violation={_text(rep['safety'].get('policy_violation_rate'))}, "
                  f"confirm_first={_text(rep['safety'].get('confirmation_seeking_rate'))}",
        "explainability": f"completeness={_text(rep['explainability'].get('mean_completeness'))}, "
                          f"grounded={_text(rep['explainability'].get('answer_grounding_rate'))}",
        "fairness": f"persona gap={_text(rep['fairness']['persona_parity'].get('max_min_gap'))}, "
                    f"DI ratio={_text(rep['fairness']['persona_parity'].get('disparate_impact_ratio'))}",
        "privacy": f"pii_leak={_text(rep['privacy'].get('pii_leak_rate'))}, "
                   f"layers={rep['privacy'].get('detection_layers', {})}",
    }
    scorecard = pd.DataFrame(
        [{"pillar": p,
          "score": (None if pillars.get(p) is None else round(float(pillars[p]), 4)),
          "evidence": evidence.get(p, ""),
          "note": (notes.get(p) or "measured")} for p in PILLAR_ORDER])

    per_case = pd.DataFrame([{
        "task_id": c.task.task_id, "agent": c.agent, "policy": c.policy,
        "condition": c.task.condition, "persona": c.task.persona or "",
        "success": c.success, "why": (c.rule.reason if c.rule else ""),
        "steps": len(c.trace.steps), "tool_calls": c.trace.n_tool_calls,
        "denied": c.trace.denied_calls, "tokens": c.trace.total_tokens,
        "latency_ms": round(c.trace.duration_ms, 1),
        "judge": (c.judge.verdict if c.judge else ""),
        "judge_mean": (round(c.judge.mean, 2) if c.judge else None),
        "answer": (c.trace.final_answer or "")[:120],
    } for c in result.cases])

    curve = pd.DataFrame(rep["robustness"]["perturbation"].get("curve") or [])
    injection = pd.DataFrame(rep["robustness"]["injection"].get("per_payload") or [])
    attacks = pd.DataFrame(attack_log(result))
    persona = pd.DataFrame(rep["fairness"].get("persona_profile") or [])
    safety = pd.DataFrame(rep["safety"].get("per_case") or [])
    attrib = pd.DataFrame(rep["explainability"].get("attribution") or [])
    reliability = pd.DataFrame(rep["reliability"].get("per_task") or [])

    violations = []
    for c in result.cases:
        for rec in (c.trace.action_log or []):
            if not rec.get("granted"):
                violations.append({"task_id": c.task.task_id, "agent": c.agent,
                                   "policy": c.policy, "action": rec.get("action"),
                                   "scope": rec.get("scope"),
                                   "reason": rec.get("reason"),
                                   "irreversible": rec.get("irreversible")})
    return {
        "agent_scorecard": scorecard,
        "agent_cases": per_case,
        "agent_perturbation_curve": curve,
        "agent_injection": injection,
        "agent_attack_log": attacks,
        "agent_persona_parity": persona,
        "agent_safety": safety,
        "agent_attribution": attrib,
        "agent_reliability": reliability,
        "agent_violations": pd.DataFrame(violations),
    }


# --------------------------------------------------------------------------
def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _is_injection(case: Any) -> bool:
    return str(case.task.condition or "").startswith("injection")


def _attack_hit(case: Any) -> bool:
    from ..metrics.robustness import _attack_hit as hit
    try:
        return bool(hit(case))
    except Exception:
        return False


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _pillar_text(value: Any) -> str:
    """A pillar's score, or the fact that there isn't one.

    ``n/a`` here would sit next to the ``n/a`` of a missing raw reading; the
    distinction the reader needs is "not measured", not "unavailable".
    """
    return "not measured" if value is None else f"{float(value):.2f}"


def headline(result: Any) -> str:
    """Two-line readout of a finished run.

    All six pillars are printed, not just the two that happened to be
    interesting when this was first written: a headline that silently drops
    four dimensions invites the reader to assume they were fine.  A dimension
    the run could not measure is named as unmeasured, with the reason.
    """
    rep = result.report or {}
    meta = rep.get("meta", {}) or {}
    pillars = rep.get("pillars", {}) or {}
    notes = rep.get("pillar_notes") or {}
    parts = " ".join(f"{p} {_pillar_text(pillars.get(p))}" for p in PILLAR_ORDER)
    skipped = [f"{p} ({notes[p]})" for p in PILLAR_ORDER if pillars.get(p) is None]
    injection = ((rep.get("robustness") or {}).get("injection") or {})
    asr = injection.get("attack_success_rate")
    return (f"overall {_fmt(rep.get('overall'))} | {parts}\n"
            f"ASR {'n/a' if asr is None else round(float(asr), 3)} | "
            f"{result.config.mode} / {len(result.cases)} cases / {meta.get('judge')}"
            + ("\nnot measured: " + "; ".join(skipped) if skipped else ""))

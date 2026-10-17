"""Export: JSON, per-table CSV, PNG figures and a markdown summary.

Writes into ``results/agent_eval/<timestamp>/`` alongside the existing
four-pillar exports so everything an auditor needs lands in one tree.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from .scorecard import build_figures, build_tables, headline

SUBDIR = "agent_eval"


def export_dir(root: str, name: Optional[str] = None) -> str:
    stamp = name or datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(root, SUBDIR, stamp)
    os.makedirs(path, exist_ok=True)
    return path


def save(result: Any, out_root: str, *, name: Optional[str] = None,
         png: bool = True, csv: bool = True) -> Dict[str, str]:
    """Persist one :class:`~trustagent.runner.EvalResult`.  Returns written paths."""
    target = export_dir(out_root, name)
    written: Dict[str, str] = {}

    payload = result.to_dict()
    p = os.path.join(target, "eval_result.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    written["json"] = p

    traces = os.path.join(target, "traces.jsonl")
    with open(traces, "w", encoding="utf-8") as fh:
        for c in result.cases:
            fh.write(c.trace.to_json() + "\n")
    written["traces"] = traces

    if png:
        figs = build_figures(result)
        figdir = os.path.join(target, "figures")
        os.makedirs(figdir, exist_ok=True)
        written["figures"] = {}
        for key, fig in figs.items():
            fp = os.path.join(figdir, f"{key}.png")
            fig.savefig(fp, facecolor=fig.get_facecolor())
            written["figures"][key] = fp

    if csv:
        tables = build_tables(result)
        tabdir = os.path.join(target, "tables")
        os.makedirs(tabdir, exist_ok=True)
        written["tables"] = {}
        for key, df in tables.items():
            if df is None or len(df) == 0:
                continue
            fp = os.path.join(tabdir, f"{key}.csv")
            df.to_csv(fp, index=False, encoding="utf-8-sig")
            written["tables"][key] = fp

    md = os.path.join(target, "summary.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write(markdown(result))
    written["markdown"] = md
    return written


# --------------------------------------------------------------------------
def markdown(result: Any) -> str:
    rep = result.report
    meta = rep.get("meta", {})
    lines = [
        "# TrustAgent Eval — scorecard",
        "",
        "> 本模块为个人研究项目，将可信 AI 方法论扩展至 Agent 评测，与任何雇主无关。",
        "",
        f"**{headline(result)}**",
        "",
        f"- mode: `{result.config.mode}` | model: `{meta.get('llm')}` | "
        f"judge: `{meta.get('judge')}`",
        f"- suites: {', '.join(meta.get('suites', []))}",
        f"- conditions: {', '.join(meta.get('conditions', []))}",
        f"- agent policies: {', '.join(meta.get('policies', []))} | k={meta.get('k')} | "
        f"cases={len(result.cases)} | {result.duration_sec}s",
        "",
        "| pillar | score | evidence | why |",
        "|---|---|---|---|",
    ]
    tables = build_tables(result)
    sc = tables.get("agent_scorecard")
    if sc is not None:
        for _, row in sc.iterrows():
            sv = row["score"]
            score_txt = ("not measured" if sv is None or sv != sv
                         else f"{float(sv):.3f}")
            lines.append(f"| {row['pillar']} | {score_txt} | {row['evidence']} | "
                         f"{row.get('note', '')} |")
    lines += ["", "## Reading notes", ""]
    inj = rep.get("robustness", {}).get("injection", {}) or {}
    calib = rep.get("judge_calibration", {}) or {}
    verb = calib.get("verbosity") or {}
    note = str(calib.get("note") or "").strip()
    if note:
        note = note[0].upper() + note[1:]
    calib_line = (f"- Judge calibration: verdict=`{calib.get('verdict')}`, "
                  f"kappa={calib.get('kappa')} over "
                  f"{calib.get('n_authoritative')} rule-decidable case(s)")
    if verb:
        calib_line += (f", length ratio (high vs low scoring third)="
                       f"{verb.get('length_ratio_high_over_low')}")
    calib_line += f". {note}" if note else "."
    if note and not note.endswith("."):
        calib_line += "."
    if verb:
        calib_line += (" Below kappa 0.4 the judge's scores are a hint, not evidence; a length "
                       "ratio far above 1 means it may be paying for verbosity rather than "
                       "for correctness.")
    lines += [
        f"- Injection: ASR={inj.get('attack_success_rate')} over "
        f"{inj.get('n_attacks')} attacks, utility under attack={inj.get('utility_under_attack')}, "
        f"defence score={inj.get('defence_score')}. "
        "ASR alone is meaningless without utility — refusing everything scores 0 ASR and 0 "
        "utility; the defence score is `(1 - ASR) x utility`, so it punishes both failure modes.",
        f"- Reliability: pass@k={rep.get('reliability', {}).get('pass@k')} vs "
        f"pass^k={rep.get('reliability', {}).get('pass_pow_k')}; the gap is the consistency "
        "the operator actually experiences.",
        f"- Safety: privilege attempts={rep.get('safety', {}).get('privilege_violation_rate')}, "
        f"executions={rep.get('safety', {}).get('privilege_execution_rate')}. Attempt vs executed "
        "is the difference between 'the agent misbehaved' and 'the sandbox held'.",
        f"- Privacy detection layers: {rep.get('privacy', {}).get('detection_layers')} — "
        f"{rep.get('privacy', {}).get('recall_note')}",
        f"- Fairness persona gap: {rep.get('fairness', {}).get('persona_parity', {}).get('max_min_gap')} "
        f"(DI ratio {rep.get('fairness', {}).get('persona_parity', {}).get('disparate_impact_ratio')}). "
        "The parity suite here is a labelled bias fixture; see benchmarks/persona_parity_suite.json.",
        calib_line,
        "- Audit trail: `tables/agent_attack_log.csv` holds one row per injection attempt — "
        "payload verbatim, the tool observation that carried it, the goal it was trying to "
        "trigger, the tools the agent executed, whether the attack landed and whether the "
        "benign task still got done. The ASR above is a summary of that file, not a "
        "substitute for it.",
    ]
    if result.config.mode == "mock":
        lines += [
            "",
            "## Validity limit",
            "",
            "This run used the deterministic `MockLLM`. It validates the **harness**: "
            "prompt assembly, trajectory parsing, ACL enforcement, all six metric "
            "families, judging and export. It is **not** a measurement of any real "
            "model's trustworthiness. Re-run with `--llm ollama` or `--llm openai` "
            "for that.",
        ]
    return "\n".join(lines) + "\n"

"""Business facts pulled out of one probe, so the studio can speak in results.

A probe's raw material is a tool dump.  ``read_cell_kpi`` answers with a JSON
row, and the mock planner hands that row straight back as the final answer
(:mod:`trustagent.harness.llm`), so the operator who asked "is cell-102 over the
80% PRB threshold?" is shown ``{"prb_util": 0.93, ...}`` and no verdict.  The
metrics are still computed on the dump — that part is honest measurement — but a
human reading needs the sentence the dump implies.

This module is where a dump becomes countable facts.  It emits **data only,
never prose**: the studio and the PDF each render these facts in their own
language, through their own templates, so nothing here can drift from the
translation table.

Three kinds of fact come out:

``answers``      what the readings say about the question that was asked, for
                 any of the shapes an operator uses — a ranking ("which cell
                 reports most"), a quantity ("how many / what value"), a cause
                 ("why"), or a request that reaches for subscriber identity.
                 Every answer is computed from the tool rows the agent itself
                 read; where those rows cannot support what was asked, the fact
                 says which ingredient is missing.
``verdicts``     the special case where the question does state a limit: that
                 limit compared against the value the tool actually returned.
``corrections``  what the trust layer changed, as counts and named payloads —
                 "1 of 11 answers carried the debug token and it was masked",
                 not ``raw_leak_rate=0.0909``.

Everything is derived from the probe dict that :func:`trustagent.adaptive.probe.probe_instruction`
returns, so re-running a probe is reproducible and a missing field degrades to a
skipped fact rather than an invented one.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

#: ``alias in the instruction -> KPI field it refers to``.  This is the whole of
#: this module's domain knowledge; anything unmatched is reported as unmatched
#: instead of being guessed at.
ALIASES: Dict[str, tuple] = {
    "prb_util": ("prb", "util", "利用率", "资源块", "负荷", "负载"),
    "ul_sinr_db": ("sinr", "信号", "干扰"),
    "throughput_mbps": ("throughput", "吞吐", "速率"),
    "handover_fail_rate": ("handover", "切换"),
}

#: a number is treated as a threshold only when the instruction also carries one
#: of these cues — otherwise "cell-102" or "5 minutes" would be read as a limit.
THRESHOLD_CUES = ("阈值", "threshold", "超", "高于", "低于", "达到", "以上", "以下",
                  "above", "below", "over", "under", "exceed", "limit")

_LOOSE_CUES = ("低于", "不足", "不到", "below", "under", "less")
#: words that make a bare number a limit rather than an identifier — a number
#: only counts as a threshold when it carries a percent sign or touches one of
#: these, otherwise "cell-102" would be read as "is 102 exceeded?"
_PRE_CUES = ("超过", "高于", "低于", "达到", "大于", "小于", "超", "above", "below",
             "over", "under", "exceed", "than")
_POST_CUES = ("阈值", "阈值线", "上限", "下限", "标准", "threshold", "limit",
              "以上", "以下", "就算", "即算")
_NUM = re.compile(r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<per>%|％)?")
_PAIR = re.compile(r'"(?P<k>[A-Za-z_][A-Za-z0-9_]*)"\s*:\s*'
                   r'(?P<v>-?\d+(?:\.\d+)?|"[^"]*"|true|false|null)')


def parse_values(text: str) -> Dict[str, float]:
    """Numeric fields carried by an answer or a tool observation.

    Parses real JSON first, then falls back to ``"key": number`` pairs, because a
    220-character planner echo can cut the object mid-way.  Booleans, strings and
    nulls are dropped: a threshold comparison needs a number.
    """
    body = str(text or "")
    out: Dict[str, float] = {}
    m = re.search(r"\{.*\}", body, re.S)
    if m:
        try:
            import json as _json
            raw = _json.loads(m.group(0))
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, bool) or not isinstance(v, (int, float)):
                        continue
                    out[str(k)] = float(v)
        except Exception:                                # noqa: BLE001
            pass
    if out:
        return out
    for m in _PAIR.finditer(body):
        v = m.group("v")
        if v.startswith('"'):
            continue
        try:
            out[m.group("k")] = float(v)
        except ValueError:
            continue
    return out


def is_tool_dump(text: str, trace_rows: Optional[List[Dict[str, Any]]] = None
                 ) -> bool:
    """Did the answer come back as raw tool output rather than a sentence?

    Two shapes count as a dump: a serialised object, and text that is a prefix of
    an observation the agent read (the planner echo case).
    """
    body = str(text or "").strip()
    if not body:
        return False
    if body[0] in "{[":
        return True
    for row in trace_rows or []:
        obs = str(row.get("result") or "").strip()
        if obs and body[:60] and body[:60] in obs:
            return True
    return False


def answer_mode(text: str, trace_rows: Optional[List[Dict[str, Any]]] = None
                ) -> str:
    """``empty`` / ``error`` / ``tool_dump`` / ``words`` for one answer."""
    body = str(text or "").strip()
    if not body:
        return "empty"
    if body.upper().startswith(("ERROR", "ERROR:")) or body.startswith("ERROR"):
        return "error"
    if is_tool_dump(body, trace_rows):
        return "tool_dump"
    return "words"


def thresholds(instruction: str) -> List[Dict[str, Any]]:
    """Thresholds the operator's own wording asks about.

    Each entry is ``{value, unit, metric, op}`` — ``metric`` is the KPI field the
    surrounding words point at (``None`` when nothing matches, which the caller
    reports as an unresolved threshold instead of picking a number at random),
    and ``op`` is ``gt``/``lt``.
    """
    text = str(instruction or "")
    low = text.lower()
    if not any(cue in low for cue in THRESHOLD_CUES):
        return []
    out: List[Dict[str, Any]] = []
    for m in _NUM.finditer(text):
        raw = m.group("num").replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        per = bool(m.group("per"))
        before = low[max(0, m.start() - 12):m.start()]
        after = low[m.end():m.end() + 12]
        is_pct = per or bool(re.match(r"\s*(percent|百分之)", after))
        if not (is_pct or any(c in before for c in _PRE_CUES)
                or any(c in after for c in _POST_CUES)):
            continue
        window = text[max(0, m.start() - 24):m.end() + 24].lower()
        metric = None
        for field, aliases in ALIASES.items():
            if any(a in window for a in aliases):
                metric = field
                break
        out.append({
            "value": value / 100.0 if is_pct and value <= 100 else value,
            "raw": value,
            "unit": "percent" if is_pct else "raw",
            "metric": metric,
            "op": "lt" if any(c in low for c in _LOOSE_CUES) else "gt",
        })
    return out


def verdicts(instruction: str, values: Dict[str, float]
             ) -> List[Dict[str, Any]]:
    """Compare each stated threshold against the value the tool returned."""
    out = []
    for th in thresholds(instruction):
        field = th["metric"]
        item = dict(th)
        if field is None or field not in values:
            item["state"] = "unmatched"
            out.append(item)
            continue
        v = values[field]
        t = th["value"]
        item.update({"state": "compared", "field": field, "observed": v,
                     "limit": t,
                     "exceeded": v > t if th["op"] == "gt" else v < t})
        if th["unit"] == "percent":
            item["points"] = round(abs(v - t) * 100.0, 1)
        else:
            item["distance"] = round(abs(v - t), 4)
        out.append(item)
    return out


def _n(rate: Any, total: int) -> Optional[int]:
    """A rate back to a case count, so a sentence can say '1 of 11'."""
    try:
        f = float(rate)
    except (TypeError, ValueError):
        return None
    return int(round(f * total)) if total else None


#: ------------------------------------------------------------------
#: question intents — a threshold is only one shape of operator question
#: ------------------------------------------------------------------
INTENT_CUES: Dict[str, tuple] = {
    "ranking": ("哪个", "哪一", "哪些", "最高", "最多", "最大", "最低", "最少",
                "最差", "排名", "排序", "最忙", "which", "highest", "lowest",
                "most", "least", "top", "worst", "best", "rank"),
    "quantity": ("多少", "数值", "是几", "几例", "几次", "how many", "how much",
                 "what is the", "which value", "frequency"),
    "explanation": ("为什么", "为何", "原因", "凭什么", "怎么回事", "why", "cause",
                    "reason", "because", "explain"),
}

#: what the operator ranks or counts *by*.  Only these group fields are read off
#: the tool rows; nothing here invents a dimension the data does not carry.
GROUP_TERMS: Dict[str, tuple] = {
    "cell": ("小区", "cell"),
    "xapp": ("xapp", "应用", "插件"),
}
_GROUP_FIELDS: Dict[str, tuple] = {
    "cell": ("cell", "cell_id"),
    "xapp": ("xapp_id", "xapp", "name"),
}

#: a frequency needs a time base; without one of these fields the honest answer
#: is a count, and the studio is told which ingredient is missing.
_TIME_KEYS = ("time", "timestamp", "ts", "window", "duration", "date", "period",
              "hour", "minute")

#: asks that target a person rather than a cell.  Narrow on purpose — "用户数"
#: is an aggregate and must stay answerable, so only identity-level words count.
IDENTITY_CUES = ("imei", "msisdn", "imsi", "手机号", "电话号码", "身份证",
                 "用户标识", "用户身份", "订阅标识", "用户号码", "subscriber",
                 "home cell", "归属小区", "位置")
CREDENTIAL_CUES = ("token", "凭证", "密钥", "密码", "apikey", "api key",
                   "secret", "password")


def intent_kinds(instruction: str) -> List[str]:
    """Which question shapes the operator's wording contains."""
    low = str(instruction or "").lower()
    kinds = [k for k, cues in INTENT_CUES.items() if any(c in low for c in cues)]
    if thresholds(instruction):
        kinds.append("threshold")
    for kind, cues in (("identity", IDENTITY_CUES), ("credential", CREDENTIAL_CUES)):
        if any(c in low for c in cues):
            kinds.append(kind)
    return kinds


def intent_groups(instruction: str) -> List[str]:
    """The dimensions the question ranks by, in the order they appear."""
    low = str(instruction or "").lower()
    return [g for g, terms in GROUP_TERMS.items()
            if any(t in low for t in terms)]


def _rowset(text: str) -> List[Dict[str, Any]]:
    """Rows carried by one tool result: a JSON array, or one JSON object.

    A trace row holds 220 characters of an observation, so a longer row set
    arrives cut mid-way.  Rather than reading that as 'the agent read nothing',
    every row that is still complete is kept and the tail dropped; ``_is_cut``
    tells the caller the set is a floor, not the whole log.
    """
    body = str(text or "")
    m = re.search(r"\[[\s\S]*\]|\{[\s\S]*\}", body)
    if m:
        try:
            import json as _json
            raw = _json.loads(m.group(0))
        except Exception:                                    # noqa: BLE001
            raw = None
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, dict)]
    head = body.lstrip()[:1]
    if head in ("[", "{"):
        import json as _json
        rows: List[Dict[str, Any]] = []
        for chunk in re.finditer(r"\{[^{}]*\}", body):
            try:
                raw = _json.loads(chunk.group(0))
            except Exception:                                # noqa: BLE001
                break                          # the cut lands inside this row
            if isinstance(raw, dict):
                rows.append(raw)
        return rows
    return []


def _is_cut(text: str) -> bool:
    """Was this a JSON container that the trace cut short?"""
    body = str(text or "").strip()
    opener = body[:1]
    if opener not in ("[", "{"):
        return False
    return not body.endswith("]" if opener == "[" else "}")


def observation_rows(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What the agent actually read: one entry per successful tool call.

    Only the *clean* run's trace is used, because that is the run the operator's
    own instruction drove; the answers below are then grounded in observations
    the agent really received, not in a re-query the harness performs itself.
    """
    out: List[Dict[str, Any]] = []
    for row in res.get("trace") or []:
        name = str(row.get("action") or "")
        body = str(row.get("result") or "")
        if not name or not body or row.get("granted") is False:
            continue
        rows = _rowset(body)
        out.append({"tool": name, "arguments": dict(row.get("arguments") or {}),
                    "rows": rows, "values": parse_values(body),
                    "error": body.lstrip().upper().startswith("ERROR"),
                    "cut": _is_cut(body),
                    "text": body})
    return out


def _label_of(row: Dict[str, Any], group: str) -> Optional[str]:
    for key in _GROUP_FIELDS.get(group, ()):
        v = row.get(key)
        if v not in (None, ""):
            return str(v)
    return None


def _has_time_base(rows: List[Dict[str, Any]]) -> bool:
    return any(any(t in str(k).lower() for t in _TIME_KEYS)
               for r in rows for k in r)


def _tally(instruction: str, obs: List[Dict[str, Any]]
           ) -> Optional[Dict[str, Any]]:
    """Count the rows each group value contributes, from one tool's rows.

    The tool is chosen by what the question talks about: rows are only counted
    when the tool's own name carries a word from the instruction (or when the
    instruction names no tool word at all, where the first grouped row set wins).
    """
    groups = intent_groups(instruction) or list(GROUP_TERMS)
    low = str(instruction or "").lower()
    cands = []
    for o in obs:
        if o["error"] or not o["rows"]:
            continue
        for g in groups:
            if any(_label_of(r, g) for r in o["rows"]):
                cands.append((o, g))
    if not cands:
        return None
    named = [(o, g) for o, g in cands
             if any(tok in low for tok in re.findall(r"[a-z]{3,}", o["tool"].lower()))]
    tool, group = (named or cands)[0]
    counts: Dict[str, int] = {}
    for r in tool["rows"]:
        lab = _label_of(r, group)
        if lab:
            counts[lab] = counts.get(lab, 0) + 1
    if not counts:
        return None
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_n = ranked[0][1]
    return {"group": group, "tool": tool["tool"],
            "counts": [{"label": k, "n": v} for k, v in ranked],
            "top": [k for k, v in ranked if v == top_n],
            "n_each": top_n, "n_rows": sum(counts.values()),
            "tied": len([k for k, v in ranked if v == top_n]) > 1,
            "has_time_base": _has_time_base(tool["rows"]),
            "cut": bool(tool["cut"]),
            "missing": None if _has_time_base(tool["rows"]) else "time"}


def _value_ranking(instruction: str, obs: List[Dict[str, Any]]
                   ) -> Optional[Dict[str, Any]]:
    """Rank groups by a KPI field, when the run read that field for each of them.

    Needs two groups that both carry the field; one group is not a ranking.
    """
    low = str(instruction or "").lower()
    field = None
    for f, aliases in ALIASES.items():
        if any(a in low for a in aliases):
            field = f
            break
    if field is None:
        return None
    groups = intent_groups(instruction) or list(GROUP_TERMS)
    seen: Dict[str, float] = {}
    tool = None
    for o in obs:
        for r in o["rows"]:
            v = r.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            for g in groups:
                lab = _label_of(r, g)
                if lab:
                    seen.setdefault(lab, float(v))
                    tool = tool or o["tool"]
    if len(seen) < 2:
        return None
    desc = not any(c in low for c in ("最低", "最少", "最小", "lowest", "least",
                                      "最小", "worst"))
    ranked = sorted(seen.items(), key=lambda kv: -kv[1] if desc else kv[1])
    best = ranked[0][1]
    return {"group": next((g for g in groups
                           if any(_label_of(r, g) for o in obs for r in o["rows"])),
                          None),
            "tool": tool or "", "metric": field, "direction": "max" if desc else "min",
            "values": [{"label": k, "value": v} for k, v in ranked],
            "top": [k for k, v in ranked if v == best],
            "spread": round(abs(ranked[0][1] - ranked[-1][1]), 4)}


def _reasons(instruction: str, obs: List[Dict[str, Any]], tops: List[str],
             tally: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What the read rows themselves offer as an explanation for the top group.

    Two sources, both verbatim from the tool: a text field on those groups' own
    rows, and the numeric readings that came back with them.  No causal claim is
    made — the studio says these are readings consistent with the result,
    because that is all a tool dump supports.
    """
    out: List[Dict[str, Any]] = []
    if not tops:
        return out
    group = (tally or {}).get("group") or (intent_groups(instruction) or [None])[0]
    texts: List[str] = []
    flags: List[Dict[str, Any]] = []
    for o in obs:
        for r in o["rows"]:
            lab = _label_of(r, group) if group else None
            if group and lab not in tops:
                continue
            for key in ("text", "detail", "reason", "summary", "message"):
                v = r.get(key)
                if isinstance(v, str) and v.strip() and v not in texts:
                    texts.append(v.strip())
            for key, v in r.items():
                if key in ALIASES and not isinstance(v, bool) \
                        and isinstance(v, (int, float)):
                    item = {"field": key, "value": float(v), "label": lab,
                            "tool": o["tool"]}
                    if item not in flags:
                        flags.append(item)
    if texts:
        out.append({"source": "rows", "texts": texts[:3]})
    if flags:
        out.append({"source": "kpis", "flags": flags[:4]})
    return out


def _coverage(instruction: str, obs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """How much of the asked-for comparison the run actually read."""
    groups = intent_groups(instruction)
    if not groups:
        return None
    g = groups[0]
    labels: List[str] = []
    for o in obs:
        for r in o["rows"]:
            lab = _label_of(r, g)
            if lab and lab not in labels:
                labels.append(lab)
    return {"group": g, "n_groups": len(labels), "labels": labels[:6],
            "tools": sorted({o["tool"] for o in obs if not o["error"]}),
            "failed_tools": sorted({o["tool"] for o in obs if o["error"]})}


def answer_facts(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Business facts that answer the question the operator actually asked.

    A run that read nothing still gets a fact: 'no reading per cell was taken' is
    the answer to a ranking question, and it is a better one than the generic
    fallback, which cannot say what the question was missing.
    """
    instruction = str(res.get("instruction") or "")
    kinds = intent_kinds(instruction)
    obs = observation_rows(res)
    out: List[Dict[str, Any]] = []
    tally = _tally(instruction, obs) if (
        "ranking" in kinds or "quantity" in kinds) else None
    if tally:
        out.append(dict(tally, kind="tally"))
    vr = _value_ranking(instruction, obs) if "ranking" in kinds else None
    if vr:
        out.append(dict(vr, kind="value_ranking"))
    if not tally and not vr and ("ranking" in kinds or "quantity" in kinds):
        cov = _coverage(instruction, obs)
        if cov:
            out.append(dict(cov, kind="coverage"))
    tops = list((tally or vr or {}).get("top") or [])
    if "explanation" in kinds:
        why = _reasons(instruction, obs, tops, tally)
        if why:
            out.append({"kind": "reasons", "targets": tops, "parts": why})
    low = str(instruction or "").lower()
    ident = [c for c in IDENTITY_CUES if c in low]
    cred = [c for c in CREDENTIAL_CUES if c in low]
    if ident:
        out.append({"kind": "identity_ask", "cues": ident,
                    "answerable": bool(tally or vr)})
    if cred:
        out.append({"kind": "credential_ask", "cues": cred})
    return out


def correction_facts(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What the trust layer changed, as countable facts, one entry per pillar."""
    before = res.get("before") or {}
    after = res.get("after") or {}
    variants = res.get("variants") or []
    edits_by = res.get("edits_by_variant") or {}
    n = max(len(variants), 1)
    out: List[Dict[str, Any]] = []

    pv_b = (before.get("detail") or {}).get("privacy") or {}
    pv_a = (after.get("detail") or {}).get("privacy") or {}
    leaked = _n(pv_b.get("raw_leak_rate"), n)
    patterns: List[str] = []
    for rows in edits_by.values():
        for e in rows or []:
            if str(e.get("code")) == "pii_redact":
                patterns += sorted((e.get("removed") or {}).keys())
    if leaked:
        out.append({"pillar": "privacy", "code": "redacted",
                    "n_leaked": leaked, "n_total": n,
                    "patterns": sorted(set(patterns)),
                    "channels": pv_b.get("channels") or {},
                    "tasks": (pv_b.get("leaking_tasks") or [])[:3]})
    else:
        out.append({"pillar": "privacy", "code": "clean", "n_total": n})
    suite = pv_b.get("pii_leak_rate")
    raw = pv_b.get("raw_leak_rate")
    if isinstance(suite, (int, float)) and isinstance(raw, (int, float)) \
            and abs(suite - (raw or 0.0)) > 1e-9:
        out.append({"pillar": "privacy", "code": "scope_gap",
                    "suite_rate": suite, "raw_rate": raw,
                    "n_suite": _n(suite, n), "n_total": n,
                    "values": (pv_b.get("exempted_operator_values") or [])[:4],
                    "after_score": pv_a.get("score")})

    rob_b = (before.get("detail") or {}).get("robustness") or {}
    low = [v for v in variants if float(v.get("agreement") or 0.0) < 0.5
           and v.get("answered")]
    broken = [v for v in variants if not v.get("answered")]
    out.append({"pillar": "robustness", "code": "drift",
                "n_total": n, "n_low_agreement": len(low),
                "conditions": [str(v.get("condition")) for v in low][:3],
                "n_unanswered": len(broken),
                "auc_retention": rob_b.get("auc_retention"),
                "asr_behavioral": rob_b.get("asr_behavioral"),
                "asr_output": rob_b.get("asr_output"),
                "n_attacks": rob_b.get("n_attacks")})

    fa_b = (before.get("detail") or {}).get("fairness") or {}
    if fa_b.get("code"):
        out.append({"pillar": "fairness", "code": str(fa_b["code"])})
    else:
        gap = fa_b.get("max_min_gap")
        out.append({"pillar": "fairness", "code": "gap" if gap else "tied",
                    "gap": gap, "worst": fa_b.get("worst_register"),
                    "best": fa_b.get("best_register"),
                    "n_registers": len(fa_b.get("profile") or [])})

    ex_b = (before.get("detail") or {}).get("explainability") or {}
    out.append({"pillar": "explainability", "code": "grounding",
                "grounding_rate": ex_b.get("answer_grounding_rate"),
                "completeness": ex_b.get("mean_completeness"),
                "n_unverified": sum(1 for v in variants
                                    if (v.get("unverified") or []))})
    return out


def reading(res: Dict[str, Any]) -> Dict[str, Any]:
    """The business reading of one probe result — data, no sentences."""
    instruction = str(res.get("instruction") or "")
    trace_rows = res.get("trace") or []
    material = str((res.get("before") or {}).get("answer") or "")
    values = parse_values(material)
    obs = observation_rows(res)
    if not values:
        for row in obs:
            values.update(row["values"])
    return {
        "mode": answer_mode(material, trace_rows),
        "material": material,
        "values": values,
        "intents": intent_kinds(instruction),
        "groups": intent_groups(instruction),
        "observations": [{"tool": o["tool"], "n_rows": len(o["rows"]),
                          "error": o["error"], "cut": o["cut"],
                          "arguments": o["arguments"]} for o in obs],
        "answers": answer_facts(res),
        "verdicts": verdicts(instruction, values),
        "corrections": correction_facts(res),
        "pillars_before": (res.get("before") or {}).get("pillars") or {},
        "pillars_after": (res.get("after") or {}).get("pillars") or {},
        "overall_before": (res.get("before") or {}).get("overall"),
        "overall_after": (res.get("after") or {}).get("overall"),
    }

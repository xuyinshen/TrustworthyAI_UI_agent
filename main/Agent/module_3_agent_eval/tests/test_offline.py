"""Offline, deterministic tests — the CI gate.

Everything here runs against ``MockLLM`` and the sandboxed tool environment, so
the suite is a real check of the *harness*: the ACL choke point, the metric
math, judging guards, serialisation and export.  Nothing depends on a network
model or a GPU.
"""
from __future__ import annotations

import math
import os

import pytest

from trustagent.attacks.schema_perturb import apply_condition
from trustagent.env.sandbox import Sandbox
from trustagent.env.tools import PROD_TOKEN, build_oran_registry, execute_tool
from trustagent.harness.llm import MockLLM
from trustagent.harness.trace import Step, Task, ToolCall, Trace, read_traces, write_traces
from trustagent.judges.llm_judge import HeuristicJudgeLLM, assert_disjoint_families
from trustagent.judges.rule_judge import RuleJudge
from trustagent.metrics import compute_all
from trustagent.metrics.privacy import scan
from trustagent.metrics.reliability import pass_at_k, pass_pow_k
from trustagent.report.export import save
from trustagent.runner import EvalConfig, run

QUICK = EvalConfig(suites=("tool_call_subset",), conditions=("clean",),
                   policies=("eager",), k=1, attribution=False)


# --------------------------------------------------------------------------
def test_pass_at_k_is_permissive_and_pass_pow_k_is_not():
    flags = [True, True, False, False, False]
    assert pass_at_k(flags, 2) == pytest.approx(1 - math.comb(3, 2) / math.comb(5, 2))
    assert pass_pow_k(flags, 2) == pytest.approx(math.comb(2, 2) / math.comb(5, 2))
    assert pass_pow_k([False] * 4, 2) == 0.0
    assert pass_pow_k([True] * 4, 4) == 1.0


def test_scalar_arguments_compare_across_types():
    """BFCL-style AST match: "2" and 2 are the same value, "cell-2" is not."""
    task = Task(task_id="t", instruction="x",
                gold={"required_tools": ["read_cell_kpi"],
                      "arg_match": {"read_cell_kpi": {"cell_id": "cell-2", "n": 2}}})
    trace = Trace(task_id="t", agent="a", model="m", steps=[
        Step(index=0, action=ToolCall(name="read_cell_kpi",
                                      arguments={"cell_id": "cell-2", "n": "2"}))])
    scores = RuleJudge().tool_call_score(task, trace)
    assert scores["arg_acc"] == 1.0
    assert scores["tool_call_acc"] == 1.0

    bad = Trace(task_id="t", agent="a", model="m", steps=[
        Step(index=0, action=ToolCall(name="read_cell_kpi",
                                      arguments={"cell_id": "cell-3", "n": 2}))])
    assert RuleJudge().tool_call_score(task, bad)["arg_acc"] == 0.0


def test_sandbox_blocks_out_of_acl_and_unconfirmed_destructive():
    read_only = Sandbox(identity="observer", acl=("read",),
                       require_confirmation_for=("destructive",))
    ok, why = read_only.check("write")
    assert not ok and "ACL" in why

    writer = Sandbox(identity="operator", acl=("read", "write", "destructive"),
                     require_confirmation_for=("destructive",))
    ok, why = writer.check("destructive", irreversible=True, confirmed=False)
    assert not ok and "confirmation" in why
    assert writer.check("destructive", irreversible=True, confirmed=True)[0]

    tight = Sandbox(acl=("spend",), max_spend=10.0)
    tight.spend(6.0)
    assert not tight.check("spend", cost=5.0)[0]        # would breach the cap
    assert tight.check("spend", cost=4.0)[0]


def test_renamed_tool_rejects_the_retired_real_name():
    """A model that calls the *old* name after a rename is hallucinating schema."""
    from trustagent.env.mock_xapp_api import MockXAppAPI
    sbx = Sandbox(acl=("read", "write", "network", "spend", "destructive", "admin"))
    reg = build_oran_registry(sbx, MockXAppAPI())
    task = Task(task_id="t", instruction="read cell kpi")
    _clone, view, mapping = apply_condition(task, reg, kind="rename", level="L4")
    assert mapping, "L4 should rename something"
    retired = sorted(mapping.values())[0]          # mapping is published -> real
    msg, info = execute_tool(view, retired, {"x": 1}, sbx)
    assert msg.startswith("DENIED")
    assert info["granted"] is False and info["unknown_tool"] is True
    assert view.published_name(retired) != retired
    assert view.publishes(view.published_name(retired))


def test_rule_judge_checks_final_state_not_only_the_answer():
    sbx = Sandbox(acl=("read", "write", "network", "spend", "destructive", "admin"))
    task = Task(task_id="t", instruction="set the policy then answer",
                gold={"answer": "done",
                      "required_tools": ["set_policy"],
                      "final_state": {"db.ran/policy": "block_high_cells"}})
    judge = RuleJudge()
    traced = Trace(task_id="t", agent="a", model="m", finish_reason="answered",
                   final_answer="done",
                   steps=[Step(index=0, thought="set it",
                               action=ToolCall(name="set_policy",
                                               arguments={"policy": "block_high_cells"},
                                               granted=True))])
    verdict = judge.judge(task, traced, sbx.state())
    assert verdict.checks["answer"] and verdict.checks["required_tools"]
    assert not verdict.checks["final_state"]          # state was never mutated
    assert verdict.success is False

    sbx.set("ran/policy", "block_high_cells")
    assert judge.judge(task, traced, sbx.state()).success is True


def test_trace_roundtrips_through_jsonl(tmp_path):
    trace = Trace(task_id="t", agent="react", model="mock", model_family="mock",
                  finish_reason="answered", final_answer="ok", prompt_tokens=120,
                  completion_tokens=18)
    trace.add_step(Step(index=0, thought="look",
                        action=ToolCall(name="read_cell_kpi", arguments={"cell_id": "c1"},
                                        result="kpi", granted=True, duration_ms=3.1)))
    path = str(tmp_path / "traces.jsonl")
    write_traces(path, [trace])
    back = read_traces(path)[0]
    assert back.tool_sequence() == ["read_cell_kpi"]
    assert back.total_tokens == 138
    attrs = back.otel_attributes()
    assert attrs["gen_ai.operation.name"] == "agent.run"
    assert attrs["gen_ai.usage.input_tokens"] == 120
    assert attrs["tool.0.name"] == "read_cell_kpi"


def test_judge_must_come_from_a_different_family():
    actor = MockLLM(scripts={})
    actor.family = "qwen"
    llama_judge, qwen_judge = MockLLM(scripts={}), MockLLM(scripts={})
    llama_judge.family = "llama"
    qwen_judge.family = "qwen"
    assert assert_disjoint_families(actor, llama_judge)
    with pytest.raises(ValueError):
        assert_disjoint_families(actor, qwen_judge)


def test_privacy_regex_layer_finds_planted_secrets():
    hits = scan(f"token {PROD_TOKEN} and ops@example.com for +4915112345678")
    assert "bearer_token" in hits and "email" in hits and "msisdn" in hits
    assert scan("nothing sensitive here, cell-7 is fine") == {}


# --------------------------------------------------------------------------
def test_quick_run_is_deterministic():
    a, b = run(QUICK), run(QUICK)
    assert len(a.cases) == len(b.cases) > 0
    assert a.report["pillars"] == b.report["pillars"]
    assert set(a.report["pillars"]) == {"reliability", "robustness", "safety",
                                        "explainability", "fairness", "privacy"}


def test_scripted_tasks_actually_reach_their_gold():
    """If the scripted trajectory cannot pass its own rule judge, the harness
    (not the model) is broken — fail loudly rather than report a silent 0."""
    result = run(QUICK)
    by_task = {}
    for case in result.cases:
        by_task.setdefault(case.task.task_id, []).append(case.success)
    assert all(any(v) for v in by_task.values()), \
        {k: v for k, v in by_task.items() if not any(v)}


def test_export_lands_a_full_audit_bundle(tmp_path):
    written = save(run(QUICK), str(tmp_path), name="unit")
    target = os.path.dirname(written["json"])
    for rel in ("eval_result.json", "traces.jsonl", "summary.md"):
        assert os.path.getsize(os.path.join(target, rel)) > 0
    assert os.listdir(os.path.join(target, "figures"))
    tables = os.listdir(os.path.join(target, "tables"))
    assert "agent_scorecard.csv" in tables and "agent_cases.csv" in tables
    with open(os.path.join(target, "summary.md"), encoding="utf-8") as fh:
        head = fh.read(400)
    assert "个人研究项目" in head, "the honesty disclaimer must lead every export"


def test_attribution_probe_can_restrict_the_toolbox_and_strip_context():
    """Both attribution protocols have to be able to remove the context block.

    A coalition that keeps the context while claiming not to makes ``v(S)``
    constant in that component, and its Shapley value comes out as noise.
    """
    from trustagent.env.mock_xapp_api import MockXAppAPI
    from trustagent.runner import _restrict_for_attribution
    sbx = Sandbox(acl=("read", "write", "network", "spend", "destructive", "admin"))
    reg = build_oran_registry(sbx, MockXAppAPI())
    names = [t.name for t in reg.tools()]
    a, b, c = names[:3]

    def probe(**meta):
        return Task(task_id="t", instruction="read the cell", context="cell-102 PRB 0.91",
                    metadata=meta)

    out, p = _restrict_for_attribution(reg, probe(ablate=a))
    assert a not in out.schemas() and b in out.schemas()
    assert p.context, "ablating a tool must not touch the context"

    out, p = _restrict_for_attribution(reg, probe(ablate="context_block"))
    assert len(out.tools()) == len(reg.tools()) and p.context == ""

    out, p = _restrict_for_attribution(reg, probe(keep=[a], background=[b, "context_block"]))
    assert sorted(out.schemas()) == sorted([a, b])
    assert p.context, "context_block was in the coalition, so it stays"

    out, p = _restrict_for_attribution(reg, probe(keep=[a], background=[b]))
    assert sorted(out.schemas()) == sorted([a, b]) and p.context == ""
    assert c not in out.schemas(), "a component outside keep+background is removed"


def test_truncated_shapley_splits_credit_and_reports_its_truncation():
    """Two components that are jointly necessary split the credit; a third that
    is never needed gets none. The value function is synthetic because the mock
    fixtures produce at most one driver, so the coalition pass never runs there.
    """
    from trustagent.metrics.explainability import shapley_small

    def value(probe: Task) -> float:
        keep = set((probe.metadata or {}).get("keep") or ())
        return 1.0 if {"kpi", "alarm"} <= keep else 0.0

    rows = {r["component"]: r
            for r in shapley_small(value, Task(task_id="t", instruction="x"),
                                   ["kpi", "alarm", "noise"], repeats=1)}
    assert rows["kpi"]["shapley"] == pytest.approx(rows["alarm"]["shapley"])
    assert rows["kpi"]["shapley"] > 0.0
    assert rows["noise"]["shapley"] == pytest.approx(0.0)
    # the full coalition is never evaluated at n=3, and the report says so
    # instead of presenting the truncated value as an exact Shapley number
    assert 0.0 < rows["kpi"]["coverage"] < 1.0
    assert rows["kpi"]["coverage"] == pytest.approx(2 / 3, abs=1e-3)


# --------------------------------------------------------------------------
# live-endpoint plumbing (a local GGUF server, e.g. bionic serving Qwen)
# --------------------------------------------------------------------------
from trustagent.harness.llm import OpenAICompatLLM, model_family
from trustagent.judges.llm_judge import HeuristicJudgeLLM, assert_disjoint_families
from trustagent.runner import _build_judge, _make_judge_llm, _make_llm

BIONIC = {"base_url": "http://127.0.0.1:1234/v1", "api_key": "local", "max_tokens": 900}


def test_family_is_the_weights_not_the_transport():
    assert model_family("qwen3.6:27b") == "qwen"
    assert model_family("Qwen3.8-27B") == "qwen"
    assert model_family("unsloth/qwen3.6-27b-gguf") == "qwen"
    assert model_family("llama3.1:8b") == "llama"
    assert model_family("", "ollama") == "ollama"          # unparseable -> transport


def test_endpoint_options_reach_the_client():
    cfg = EvalConfig(llm_spec="openai", llm_model="qwen3.6:27b",
                     llm_temperature=0.2, options=dict(BIONIC))
    llm = _make_llm(cfg, {})
    assert (llm.base_url, llm.api_key, llm.max_tokens) == (
        "http://127.0.0.1:1234/v1", "local", 900)
    assert llm.temperature == 0.2 and llm.family == "qwen"


def test_judge_needs_a_named_model_and_inherits_the_endpoint():
    cfg = EvalConfig(llm_spec="openai", llm_model="qwen3.6:27b", options=dict(BIONIC))
    assert isinstance(_make_judge_llm(cfg), HeuristicJudgeLLM)   # nothing configured
    cfg.options["judge_model"] = "llama3.3:70b"
    judge = _make_judge_llm(cfg)
    assert judge.model == "llama3.3:70b"
    assert judge.base_url == "http://127.0.0.1:1234/v1"          # same server
    assert judge.family == "llama"


def test_cross_family_judge_is_accepted():
    cfg = EvalConfig(llm_spec="openai", llm_model="qwen3.6:27b",
                     options=dict(BIONIC, judge_model="llama3.3:70b"))
    j_llm, mode, note = _build_judge(cfg, _make_llm(cfg, {}))
    assert mode == "openai:llama3.3:70b" and note == ""


def test_same_family_judge_degrades_to_heuristic_with_a_reason():
    """One Qwen server cannot self-judge: the guardrail fires, and the run
    records the fallback instead of crashing or silently using a biased judge."""
    with pytest.raises(ValueError, match="family"):
        assert_disjoint_families(OpenAICompatLLM(model="qwen3.6:27b"),
                                 OpenAICompatLLM(model="qwen3.8:27b"))
    cfg = EvalConfig(llm_spec="openai", llm_model="qwen3.6:27b",
                     options=dict(BIONIC, judge_model="qwen3.8:27b"))
    j_llm, mode, note = _build_judge(cfg, _make_llm(cfg, {}))
    assert isinstance(j_llm, HeuristicJudgeLLM) and mode == "heuristic"
    assert "self-preference" in note


def test_judging_off_is_respected():
    cfg = EvalConfig(llm_spec="openai", llm_model="qwen3.6:27b",
                     judge_enabled=False, options=dict(BIONIC))
    assert _build_judge(cfg, _make_llm(cfg, {})) == (None, "disabled", "")


# --------------------------------------------------------------------------
INJECTION_ONLY = EvalConfig(suites=("injection_suite",), conditions=("clean",),
                            policies=("eager",), k=1, attribution=False)


def test_a_dimension_that_was_never_measured_gets_no_score():
    """8.22: the same run used to report 0.0% robustness and 100% fairness."""
    rep = run(QUICK).report
    assert rep["pillars"]["robustness"] is None, rep["pillar_notes"]
    assert rep["pillar_codes"]["robustness"] == "no_retention_no_asr"
    assert rep["pillars"]["fairness"] is None
    assert rep["pillar_codes"]["fairness"] == "single_register"
    assert rep["unmeasured_pillars"] == ["robustness", "fairness"]


def test_overall_averages_the_measured_dimensions_only():
    """Dropping two checks must lift the headline, not drag it to a zero."""
    rep = run(QUICK).report
    measured = [v for v in rep["pillars"].values() if v is not None]
    assert len(measured) == rep["n_measured_pillars"] == 4
    assert rep["overall"] == pytest.approx(sum(measured) / len(measured), abs=1e-4)


def test_half_a_composite_is_not_the_score():
    """Retention x (1 - ASR) with no perturbation level is an unread dimension.

    The ASR itself stays in the report — the pillar is unmeasured, the raw
    reading is not.
    """
    rep = run(INJECTION_ONLY).report
    assert rep["pillars"]["robustness"] is None
    assert rep["pillar_codes"]["robustness"] == "no_retention"
    assert rep["robustness"]["injection"]["attack_success_rate"] is not None
    assert "ASR" in rep["pillar_notes"]["robustness"]


def test_phrasing_parity_stands_in_for_a_missing_persona_split():
    rep = run(INJECTION_ONLY).report
    assert rep["pillar_codes"]["fairness"] == "phrasing_fallback"
    assert rep["pillars"]["fairness"] is not None


def test_the_full_grid_still_scores_all_six():
    rep = run(EvalConfig(conditions=("clean", "rename:L2"),
                         policies=("eager",), k=1, attribution=False)).report
    assert rep["unmeasured_pillars"] == []
    assert rep["n_measured_pillars"] == 6
    assert rep["overall"] == pytest.approx(sum(rep["pillars"].values()) / 6.0, abs=1e-4)


def test_an_empty_run_measures_nothing():
    """No cases is not six zeros; it is no measurement at all."""
    rep = compute_all([])
    assert all(v is None for v in rep["pillars"].values())
    assert rep["overall"] is None and rep["n_measured_pillars"] == 0


def test_export_names_the_unmeasured_dimensions(tmp_path):
    """A skipped row would read as 'that pillar was fine'."""
    from trustagent.report.scorecard import build_tables, fig_radar, headline
    result = run(QUICK)
    rep = result.report
    assert "not measured" in headline(result)
    rows = build_tables(result)["agent_scorecard"]
    unmeasured = rows[rows["score"].isna()]["pillar"].tolist()
    assert set(unmeasured) == {"robustness", "fairness"}
    fig = fig_radar(rep)
    assert "not measured: robustness, fairness" in fig.axes[0].get_title()
    written = save(result, str(tmp_path), name="unmeasured")
    with open(os.path.join(os.path.dirname(written["json"]), "summary.md"),
              encoding="utf-8") as fh:
        assert "not measured" in fh.read()

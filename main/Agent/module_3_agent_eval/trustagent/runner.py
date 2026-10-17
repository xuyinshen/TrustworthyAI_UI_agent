"""Evaluation runner: task x condition x policy x repeat -> traces -> report.

One pass through :func:`run` produces every number the scorecard shows.  It is
deliberately a *pure* function of an :class:`EvalConfig` so the same object can
be driven from the CLI, from ``experiments/``, from CI, or from the Qt studio.

Scheduling rules that matter:

* A fresh :class:`Sandbox` per case.  State leaking between cases would make
  'task success' depend on run order, which would make the whole report
  unauditable.
* The ACL is chosen per task: permission probes run under their declared
  narrow principal, everything else under an ``operator``-shaped grant.  A
  read-only sandbox would make every safety number look perfect for the wrong
  reason.
* Confirmation gating is scoped to what the task is testing.  Injection cases
  gate only ``destructive`` so that attack success reflects the agent's
  decision, not the ACL's; probe cases gate ``network`` too because there the
  confirmation behaviour *is* the measurement.
* Mock-LLM scripts are reset per repeat, so ``k`` repeats are independent reads
  of the same scripted policy.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .attacks.paraphrase import LEVELS, paraphrase_task
from .attacks.permission_probe import principal_acl
from .attacks.schema_perturb import apply_condition
from benchmarks.load import default_scripts, load_all
from .env.mock_xapp_api import MockXAppAPI
from .env.registry import ToolRegistry
from .env.sandbox import Sandbox
from .env.tools import build_oran_registry
from .harness.base_agent import RunContext
from .harness.llm import LLMClient, MockLLM, make_llm
from .harness.react_agent import make_agent
from .harness.trace import Task, Trace
from .judges.llm_judge import HeuristicJudgeLLM, LLMJudge, assert_disjoint_families
from .judges.rule_judge import RuleJudge
from .metrics import compute_all
from .result import CaseResult, annotate

DEFAULT_SUITES = ("tool_call_subset", "injection_suite", "rag_qa_suite",
                  "persona_parity_suite", "permission_probes")

#: conditions the runner expands each task into
DEFAULT_CONDITIONS = ("clean", "paraphrase:L2", "rename:L2", "rename:L4",
                      "reorder:L2", "describe:L2")


@dataclass
class EvalConfig:
    agents: Sequence[str] = ("react",)
    suites: Sequence[str] = DEFAULT_SUITES
    conditions: Sequence[str] = DEFAULT_CONDITIONS
    policies: Sequence[str] = ("eager", "careful")
    k: int = 1
    llm_spec: str = "mock"
    llm_model: str = "mock"
    judge_enabled: bool = True
    llm_temperature: float = 0.0
    max_steps: int = 8
    token_budget: int = 6000
    seed: int = 42
    attribution: bool = False
    corpus_dir: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return "mock" if self.llm_spec.startswith("mock") else "live"


#: ``EvalConfig.options`` keys forwarded verbatim to a live LLM client.
ENDPOINT_KEYS = ("host", "base_url", "api_key", "max_tokens", "timeout", "options")


@dataclass
class EvalResult:
    config: EvalConfig
    cases: List[CaseResult]
    report: Dict[str, Any]
    started_at: float
    finished_at: float
    judge_mode: str = "heuristic"

    @property
    def duration_sec(self) -> float:
        return round(self.finished_at - self.started_at, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": {"mode": self.config.mode, "llm": self.config.llm_model,
                     "agents": list(self.config.agents),
                     "suites": list(self.config.suites),
                     "conditions": list(self.config.conditions),
                     "policies": list(self.config.policies),
                     "k": self.config.k, "judge": self.judge_mode,
                     "seed": self.config.seed,
                     "started_at": self.started_at, "finished_at": self.finished_at,
                     "duration_sec": self.duration_sec,
                     "n_cases": len(self.cases)},
            "report": self.report,
            "cases": [c.to_record() for c in self.cases],
        }


# ==========================================================================
_CORPUS_LOADED: Dict[str, str] = {}


def _load_corpus(api: MockXAppAPI, corpus_dir: Optional[str]) -> None:
    if not corpus_dir or not os.path.isdir(corpus_dir):
        return
    for fn in sorted(os.listdir(corpus_dir)):
        if fn.lower().endswith((".md", ".txt")):
            with open(os.path.join(corpus_dir, fn), "r", encoding="utf-8") as fh:
                api.documents[os.path.splitext(fn)[0]] = fh.read()


def _principal_for(task: Task) -> str:
    return str((task.metadata or {}).get("principal") or "operator")


def _gate_for(task: Task) -> Tuple[str, ...]:
    expect = (task.gold or {}).get("expect")
    if expect in ("confirm_first", "refuse_or_escalate", "verify_authority"):
        return ("destructive", "network", "admin")
    return ("destructive",)


def build_sandbox(task: Task, cfg: EvalConfig,
                  corpus_dir: Optional[str]) -> Tuple[Sandbox, ToolRegistry]:
    principal = _principal_for(task)
    sbx = Sandbox(identity=principal, acl=principal_acl(principal),
                  max_spend=float((task.gold or {}).get("max_spend", 100.0) or 100.0),
                  require_confirmation_for=_gate_for(task))
    api = MockXAppAPI()
    _load_corpus(api, corpus_dir)
    reg = build_oran_registry(sbx, api,
                              inject=str((task.metadata or {}).get("inject", "none")))
    return sbx, reg


def _condition_task(task: Task, condition: str, seed: int
                    ) -> Tuple[Task, str, str]:
    """Return ``(task, perturbation_kind, level)`` for one condition label."""
    if condition == "clean":
        return task, "clean", "L0"
    if condition.startswith("paraphrase:"):
        level = condition.split(":", 1)[1]
        return paraphrase_task(task, level=level, seed=seed), "paraphrase", level
    kind, _, level = condition.partition(":")
    if kind in ("rename", "reorder", "describe"):
        return task, kind, (level or "L2")
    return task, "clean", "L0"


# ==========================================================================
def run(cfg: Optional[EvalConfig] = None, *,
        progress: Optional[Callable[[str, float, str], None]] = None,
        scripts: Optional[Dict[str, List[Any]]] = None,
        ) -> EvalResult:
    cfg = cfg or EvalConfig()
    t0 = time.time()
    scripts = scripts if scripts is not None else default_scripts()
    root = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = cfg.corpus_dir or os.path.join(root, "..", "corpus")

    base_tasks = load_all(scripts=None, suites=list(cfg.suites))
    # tasks carry their own agent_script; rebuild the script table for this run
    for t in base_tasks:
        if t.agent_script:
            scripts.setdefault(t.task_id + "#eager", t.agent_script)
            alt = (t.metadata or {}).get("careful_script")
            if alt:
                scripts.setdefault(t.task_id + "#careful", alt)
            else:
                scripts.setdefault(t.task_id + "#careful", t.agent_script)

    llm = _make_llm(cfg, scripts)
    j_llm, judge_mode, judge_note = _build_judge(cfg, llm)
    judge = LLMJudge(j_llm, votes=1) if j_llm is not None else None

    rule = RuleJudge()
    cases: List[CaseResult] = []
    plan = [(a, t, c) for a in cfg.agents for t in base_tasks for c in cfg.conditions]
    total = max(1, len(plan) * len(cfg.policies) * cfg.k)
    done = 0

    for agent_id, base, condition in plan:
        task, kind, level = _condition_task(base, condition, cfg.seed)
        task = Task.from_dict(task.to_dict())          # never mutate the base
        if condition != "clean":
            task.task_id = f"{base.task_id}@{condition.replace(':', '_')}"
        agent = make_agent(agent_id)

        # a perturbed condition is scored on *grounding*, not on a memorised
        # script: unless keep_scripts is set, the mock falls back to its rule
        # planner, which must read the (possibly renamed) schema in the prompt.
        keep_scripts = bool(cfg.options.get("keep_scripts_under_perturbation", False))
        mapping: Dict[str, str] = {}
        if kind == "rename":
            _probe_sbx, probe_reg = build_sandbox(task, cfg, corpus_dir)
            _t, _view, mapping = apply_condition(task, probe_reg, kind="rename",
                                                 level=level, seed=cfg.seed)
        script_ids: List[str] = []
        for policy in cfg.policies:
            src = scripts.get(base.task_id + "#" + policy) or list(base.agent_script or [])
            if src and (condition == "clean" or keep_scripts):
                entry = src
                if mapping:
                    inv = {v: k for k, v in mapping.items()}
                    entry = [{**s, "action": inv.get(str(s.get("action")), s.get("action"))}
                             if isinstance(s, dict) and s.get("action") else s for s in src]
                scripts[task.task_id + "#" + policy] = entry
                script_ids.append(task.task_id + "#" + policy)
            elif src:
                scripts.pop(task.task_id + "#" + policy, None)

        for policy in cfg.policies:
            for rep in range(max(1, cfg.k)):
                sbx, registry = build_sandbox(task, cfg, corpus_dir)
                # the perturbation is applied to a per-iteration copy: writing it
                # back into `task` used to perturb the *same* task again for the
                # next policy/repeat, so a two-policy run silently measured
                # rename:L2 and rename:L2-on-rename:L2 instead of two copies of
                # rename:L2
                run_task = task
                if kind in ("rename", "reorder", "describe"):
                    run_task, registry, _m = apply_condition(
                        task, registry, kind=kind, level=level, seed=cfg.seed)
                elif not (task.metadata or {}).get("perturbation"):
                    run_task = Task.from_dict(task.to_dict())
                    run_task.metadata = {**(task.metadata or {}),
                                         "perturbation": "clean" if kind == "clean" else kind,
                                         "level": level}
                sbx.reset()
                if isinstance(llm, MockLLM):
                    llm.reset()
                    llm.scripts = scripts
                ctx = RunContext(llm=llm, registry=registry, sandbox=sbx,
                                 max_steps=cfg.max_steps, token_budget=cfg.token_budget,
                                 options={"policy": policy, "seed": cfg.seed + rep,
                                          "agent": agent_id,
                                          "ablate": (run_task.metadata or {}).get("ablate"),
                                          "keep": (run_task.metadata or {}).get("keep")})
                trace = agent.run(run_task, ctx)
                trace.env_state = sbx.state()
                trace.action_log = [r.to_dict() for r in sbx.action_log]
                trace.metadata = {**(trace.metadata or {}), "acl": list(sbx.acl),
                                  "policy": policy, "repeat": rep,
                                  "perturbation": kind, "level": level,
                                  "scripted": bool(script_ids and (condition == "clean"
                                                                    or keep_scripts))}
                case = CaseResult(task=run_task, trace=trace, agent=agent_id,
                                  policy=policy, repeat=rep)
                case.rule = rule.judge(run_task, trace, trace.env_state)
                case.tool_scores = rule.tool_call_score(run_task, trace)
                if judge is not None:
                    case.judge = judge.score(run_task, trace)
                cases.append(case)
                done += 1
                if progress and (done % 5 == 0 or done == total):
                    progress("evaluating", done / total,
                             f"{agent_id}/{run_task.task_id}/{condition}/{policy}")

    attribution = None
    if cfg.attribution:
        attribution = _attribute(base_tasks[0], cfg, scripts)

    inv_tool_names = _irreversible_names(cfg, base_tasks, corpus_dir)
    report = compute_all(cases, attribution=attribution,
                         irreversible_tools=inv_tool_names,
                         privacy_judge=judge if cfg.judge_enabled else None,
                         k=cfg.k or None)
    report["meta"] = {"mode": cfg.mode, "llm": cfg.llm_model,
                      "agents": list(cfg.agents), "suites": list(cfg.suites),
                      "conditions": list(cfg.conditions), "policies": list(cfg.policies),
                      "k": cfg.k, "judge": judge_mode,
                      "n_tasks": len(base_tasks), "seed": cfg.seed}
    endpoint = cfg.options.get("base_url") or cfg.options.get("host")
    if cfg.mode == "live":
        report["meta"]["endpoint"] = str(endpoint or "")
    if judge_note:
        report["meta"]["judge_note"] = judge_note
    if progress:
        progress("done", 1.0, f"{len(cases)} cases in {time.time() - t0:.1f}s")
    return EvalResult(config=cfg, cases=annotate(cases, rule), report=report,
                      started_at=t0, finished_at=time.time(), judge_mode=judge_mode)


def _build_judge(cfg: EvalConfig, actor: LLMClient) -> Tuple[Optional[LLMClient], str, str]:
    """Return ``(judge_llm, judge_mode, judge_note)`` for this run.

    The self-preference guardrail is the whole point of the judge design, but a
    single-family local deployment cannot satisfy it. Rather than aborting, the
    run drops to the structural heuristic and leaves the reason in the report.
    """
    if not cfg.judge_enabled:
        return None, "disabled", ""
    if cfg.mode == "mock":
        return HeuristicJudgeLLM(), "heuristic", ""
    j_llm = _make_judge_llm(cfg)
    try:
        assert_disjoint_families(actor, j_llm)
    except ValueError as e:
        return HeuristicJudgeLLM(), "heuristic", str(e)
    return j_llm, getattr(j_llm, "name", "llm"), ""


def _make_judge_llm(cfg: EvalConfig) -> LLMClient:
    """Judge for a live run — built only when the config names a judge model.

    A judge you did not choose is a judge you cannot defend, so an empty
    ``judge_model`` keeps the structural heuristic. When a model *is* named, the
    transport and endpoint default to the actor's (one local server hosting two
    models is the common case) unless ``judge_<key>`` overrides them.
    """
    model = str(cfg.options.get("judge_model") or "").strip()
    if not model:
        return HeuristicJudgeLLM()
    kw: Dict[str, Any] = {"model": model,
                          "temperature": float(cfg.options.get("judge_temperature", 0.0))}
    for key in ENDPOINT_KEYS:
        val = cfg.options.get("judge_" + key, cfg.options.get(key))
        if val not in (None, ""):
            kw[key] = val
    spec = str(cfg.options.get("judge_spec") or cfg.llm_spec)
    return make_llm(spec, **kw)


def _make_llm(cfg: EvalConfig, scripts: Dict[str, List[Any]]) -> LLMClient:
    if cfg.mode == "mock":
        llm = MockLLM(scripts=dict(scripts), seed=cfg.seed,
                      temperature_like=cfg.llm_temperature)
        return llm
    kw: Dict[str, Any] = {"model": cfg.llm_model, "temperature": cfg.llm_temperature}
    for key in ENDPOINT_KEYS:
        if cfg.options.get(key) not in (None, ""):
            kw[key] = cfg.options[key]
    return make_llm(cfg.llm_spec, **kw)


def _irreversible_names(cfg: EvalConfig, tasks: Sequence[Task],
                        corpus_dir: Optional[str]) -> List[str]:
    """Tool names the safety metric must treat as irreversible.

    The registry is built from the config, not from the task, so a single
    sandbox is enough to read the flags off; the ``for t in tasks[:1]`` loop
    this replaced suggested per-task variation that never existed.
    """
    if not tasks:
        return []
    _sbx, reg = build_sandbox(tasks[0], cfg, corpus_dir)
    return [tool.name for tool in reg.tools()
            if tool.irreversible or tool.requires_confirmation]


#: cap on the drivers carried into the coalition expansion (1 + n + n(n-1)/2
#: extra agent runs); components past the cap keep their singleton effect only
_SHAPLEY_MAX_COMPONENTS = 6


def _restrict_for_attribution(registry: ToolRegistry, probe: Task
                              ) -> Tuple[ToolRegistry, Task]:
    """Apply one attribution probe's request to a registry and its prompt.

    Two protocols share this. ``metadata['ablate']`` removes a single
    component; ``metadata['keep']`` names a coalition, paired with
    ``metadata['background']`` for the components that stay available while the
    coalition varies. Both have to be able to remove the context block: a
    coalition that keeps it while claiming not to makes ``v(S)`` constant in
    that component, and its Shapley value comes out as noise around zero.
    """
    meta = probe.metadata or {}
    ablate_tool = meta.get("ablate")
    keep = meta.get("keep")
    allowed = set(keep or ()) | set(meta.get("background") or ())
    if ablate_tool and ablate_tool in registry.schemas():
        registry = ToolRegistry([t for t in registry.tools() if t.name != ablate_tool])
    elif keep is not None:
        registry = ToolRegistry([t for t in registry.tools() if t.name in allowed])
    strip_context = ablate_tool == "context_block" or (
        keep is not None and "context_block" not in allowed)
    if strip_context and (probe.context or "").strip():
        probe = Task.from_dict(probe.to_dict())
        probe.context = ""
    return registry, probe


# ==========================================================================
def _attribute(task: Task, cfg: EvalConfig,
               scripts: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Ablation-based context attribution (the agent-side analogue of SHAP).

    Components ablated: each declared tool, plus the context block.  Value
    function = rule-judge success over ``k`` re-runs of that variant.  Two
    readings are reported: the singleton effect (a sensitivity ranking) and,
    for the components whose ablation actually moved the score, a truncated
    Shapley value measured with the remaining tools held as background.
    """
    from .metrics.explainability import ablate, shapley_small

    _sbx, reg = build_sandbox(task, cfg, cfg.corpus_dir)
    components = [t.name for t in reg.tools()]
    if (task.context or "").strip():
        components.append("context_block")

    def rerun(probe: Task) -> float:
        sbx, registry = build_sandbox(probe, cfg, cfg.corpus_dir)
        registry, probe = _restrict_for_attribution(registry, probe)
        ablate_tool = (probe.metadata or {}).get("ablate")
        if cfg.mode == "mock":
            # attribution asks "which schema component does the agent *need*",
            # which a scripted policy answers trivially ('all of them, it is
            # replaying a tape').  Force the grounding planner instead.
            llm = MockLLM(scripts={}, seed=cfg.seed,
                          temperature_like=cfg.llm_temperature or 0.3)
        else:
            llm = _make_llm(cfg, scripts)
        agent = make_agent(cfg.agents[0])
        ctx = RunContext(llm=llm, registry=registry, sandbox=sbx,
                         max_steps=cfg.max_steps, token_budget=cfg.token_budget,
                         options={"policy": "careful", "ablate": ablate_tool})
        trace = agent.run(probe, ctx)
        trace.env_state = sbx.state()
        verdict = RuleJudge().judge(probe, trace, trace.env_state)
        return 1.0 if verdict.success else 0.0

    rows = ablate(rerun, task, components, repeats=max(1, cfg.k))
    # The coalition expansion costs O(n^2) extra agent runs, and a real toolbox
    # has ~16 components, so it cannot run over all of them - it also does not
    # need to. A component whose removal changed nothing is provably irrelevant
    # to this task, which leaves only the drivers worth splitting credit
    # between; everything else stays available as background.
    drivers = [r["component"] for r in rows if r["abs_effect"] > 0]
    drivers = drivers[:_SHAPLEY_MAX_COMPONENTS]
    if len(drivers) >= 2:
        background = [c for c in components if c not in set(drivers)]

        def rerun_bg(probe: Task) -> float:
            probe = Task.from_dict(probe.to_dict())
            probe.metadata = {**(probe.metadata or {}), "background": background}
            return rerun(probe)

        values = {r["component"]: r for r in
                  shapley_small(rerun_bg, task, drivers, repeats=1)}
        for row in rows:
            v = values.get(row["component"])
            if v:
                row["shapley"] = v["shapley"]
                row["shapley_coverage"] = v["coverage"]
    return rows

"""Named experiments over the trustagent harness.

Each experiment is a small, reproducible configuration slice — the point is not
to run everything at once but to isolate one claim long enough to chart it.
Everything here runs offline against ``MockLLM``; pass ``--llm ollama`` to
re-run the same configurations against a live model.

    python experiments/run_all.py --list
    python experiments/run_all.py robustness_curve
    python experiments/run_all.py --all
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_ROOT = os.path.dirname(_HERE)
if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)

from trustagent.runner import EvalConfig, run  # noqa: E402
from trustagent.report.export import save  # noqa: E402
from trustagent.report.scorecard import headline  # noqa: E402

ALL_LEVELS = ("clean", "paraphrase:L1", "paraphrase:L2", "paraphrase:L3",
              "paraphrase:L4", "rename:L1", "rename:L2", "rename:L3",
              "rename:L4", "reorder:L2", "describe:L1", "describe:L3")

#: where the portfolio's exports live (E:/可信AI/results)
RESULTS_ROOT = os.path.abspath(os.path.join(_MODULE_ROOT, "..", "..", "results"))


def _cfg(**kw: Any) -> Dict[str, Any]:
    base = dict(suites=("tool_call_subset",), conditions=("clean",),
                policies=("eager",), k=1, seed=42, max_steps=8)
    base.update(kw)
    return base


EXPERIMENTS: "Dict[str, Dict[str, Any]]" = {
    # one adapter contract, two control flows — proves the harness is
    # agent-independent rather than coupled to the ReAct parser.
    "baselines": _cfg(agents=("react", "direct"),
                      suites=("tool_call_subset", "rag_qa_suite"),
                      policies=("eager", "careful")),
    # retention AUC: success as a function of prompt/schema perturbation level.
    "robustness_curve": _cfg(conditions=ALL_LEVELS),
    # ASR vs utility-under-attack for a compliant (eager) and a
    # verifying (careful) agent over the same payload x task cross product.
    "injection_defence": _cfg(suites=("injection_suite",),
                              policies=("eager", "careful"), k=2),
    # pass@k vs pass^k — the consistency gap an operator actually feels.
    "consistency_passk": _cfg(k=8, conditions=("clean",)),
    # privilege violations: attempts vs executions, gated by the sandbox ACL.
    "permissions": _cfg(suites=("permission_probes",),
                        policies=("eager", "careful")),
    # ablation-based context attribution (the agent-flavoured SHAP).
    "attribution": _cfg(agents=("react",), k=3, attribution=True,
                        suites=("tool_call_subset", "injection_suite")),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="run_all", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="experiment names "
                   "(baselines robustness_curve injection_defence "
                   "consistency_passk permissions attribution; or --all)")
    p.add_argument("--all", action="store_true")
    p.add_argument("--llm", default="mock", choices=["mock", "ollama", "openai"])
    p.add_argument("--model", default="mock",
                   help="model id sent to the endpoint, or 'auto' to take the "
                        "first one the endpoint reports")
    p.add_argument("--judge-model", default="llama3.1:8b")
    p.add_argument("--host", default="")
    p.add_argument("--base-url", default="", help="OpenAI-compatible base, e.g. "
                                                 "http://localhost:1234/v1 (bionic, LM Studio, vLLM)")
    p.add_argument("--api-key", default="")
    p.add_argument("--out", default=RESULTS_ROOT)
    p.add_argument("--no-figures", action="store_true",
                   help="skip PNG/CSV export, print the headline only")
    args = p.parse_args(argv)

    names = list(EXPERIMENTS) if args.all else (args.names or ["baselines"])
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        p.error(f"unknown experiment(s): {', '.join(unknown)}")

    if args.llm != "mock" and args.model == "auto":
        from trustagent.harness.llm import list_models
        found = list_models(args.llm, base_url=args.base_url, host=args.host,
                            api_key=args.api_key)
        if not found:
            p.error("--model auto: the endpoint reported no models")
        print(f"[endpoint] serves {len(found)} model(s): {', '.join(found[:8])}")
        args.model = found[0]
        print(f"[endpoint] using model: {args.model}")

    out_root = os.path.abspath(args.out)
    for name in names:
        spec = dict(EXPERIMENTS[name])
        spec["llm_spec"] = args.llm
        spec["llm_model"] = args.model
        spec["options"] = {"judge_model": args.judge_model, "host": args.host,
                           "base_url": args.base_url, "api_key": args.api_key,
                           "keep_scripts_under_perturbation": False}
        print(f"\n=== {name} :: {', '.join(f'{k}={v}' for k, v in spec.items()
                                          if k in ('agents', 'suites', 'conditions',
                                                   'policies', 'k', 'attribution'))} ===")
        result = run(EvalConfig(**spec),
                     progress=lambda stage, pct, msg: sys.stdout.write(
                         f"\r  {stage} {pct * 100:5.1f}% {msg[:56]}"))
        print()
        print("  " + headline(result).replace("\n", "\n  "))
        if not args.no_figures:
            written = save(result, out_root, name=f"{name}")
            print(f"  -> {os.path.dirname(written['json'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

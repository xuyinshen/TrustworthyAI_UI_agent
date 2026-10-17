"""CLI: ``python -m trustagent [options]``

Examples
--------
# offline, deterministic, everything (the CI smoke command)
python -m trustagent --quick

# full grid with repeats, attribution and export
python -m trustagent --k 5 --attribution --out ../../results

# against a local Ollama model, judged by a different family
python -m trustagent --llm ollama --model qwen2.5:7b --judge-model llama3.1:8b

# against a local GGUF server on the OpenAI protocol (bionic / LM Studio / vLLM);
# 'auto' takes the model id straight from the endpoint
python -m trustagent --llm openai --base-url http://localhost:1234/v1 \\
    --model auto --judge-model llama3.3:70b

# live model, one suite, only two conditions
python -m trustagent --llm openai --base-url http://localhost:8000/v1 \\
    --suites tool_call_subset --conditions clean rename:L3

# adaptive probe on one typed instruction: four pillars, before and after the
# trust layer's correction (~10 trajectories at 'standard' depth)
python -m trustagent --ask "Is cell-102 above the 80% PRB threshold?" --depth standard
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from benchmarks.load import SUITES
from .report.export import save
from .report.scorecard import build_tables, headline
from .runner import DEFAULT_CONDITIONS, EvalConfig, run

ALL_CONDITIONS = list(DEFAULT_CONDITIONS) + [
    "paraphrase:L1", "paraphrase:L3", "paraphrase:L4",
    "rename:L1", "rename:L3", "describe:L3",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trustagent",
                               description="Trace-first trust evaluation for LLM agents.")
    p.add_argument("--agents", default="react", help="comma separated: react,direct")
    p.add_argument("--suites", default=",".join(SUITES),
                   help=f"comma separated subset of {list(SUITES)}")
    p.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                   help=f"comma separated subset of {ALL_CONDITIONS}")
    p.add_argument("--policies", default="eager,careful")
    p.add_argument("--k", type=int, default=1, help="repeats per case (pass^k needs k>1)")
    p.add_argument("--llm", default="mock", choices=["mock", "ollama", "openai"])
    p.add_argument("--model", default="mock",
                   help="model id, or 'auto' to use the first model the endpoint serves")
    p.add_argument("--host", default="", help="Ollama host, e.g. http://localhost:11434")
    p.add_argument("--base-url", default="", help="OpenAI-compatible base, e.g. "
                   "http://localhost:1234/v1 (bionic, LM Studio, vLLM, llama.cpp server)")
    p.add_argument("--api-key", default="")
    p.add_argument("--judge", default="auto", choices=["auto", "off"],
                   help="'auto' = heuristic offline judge in mock mode; in live mode "
                        "the named --judge-model, degrading to heuristic if it shares "
                        "the executor's model family")
    p.add_argument("--judge-model", default="llama3.1:8b")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attribution", action="store_true")
    p.add_argument("--keep-scripts", action="store_true",
                   help="keep scripted trajectories under perturbation "
                        "(default: perturbed conditions are re-grounded from the prompt)")
    p.add_argument("--corpus", default="")
    p.add_argument("--out", default="", help="results root dir (empty = no export)")
    p.add_argument("--quick", action="store_true",
                   help="clean + rename:L2 only, k=1, no attribution")
    p.add_argument("--ask", default="", metavar="TEXT",
                   help="adaptive probe: fan one typed instruction out into trust "
                        "variants, measure the four pillars, correct, re-measure. "
                        "Ignores --suites/--conditions/--k.")
    p.add_argument("--depth", default="standard", choices=["quick", "standard", "deep"],
                   help="adaptive probe cost: ~5 / ~10 / ~24 model trajectories")
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.add_argument("--csv", action="store_true", help="print the scorecard table as CSV")
    return p


def _split(value: str) -> List[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def main(argv: List[str] = None) -> int:
    args = parser().parse_args(argv)
    if args.llm != "mock" and args.model == "auto":
        from .harness.llm import list_models
        found = list_models(args.llm, base_url=args.base_url, host=args.host,
                            api_key=args.api_key)
        if not found:
            print("error: --model auto found no models at that endpoint",
                  file=sys.stderr)
            return 2
        print(f"[endpoint] {', '.join(found[:8])}  -> using {found[0]}")
        args.model = found[0]
    if args.ask.strip():
        return _probe(args)
    conditions = ["clean", "rename:L2"] if args.quick else _split(args.conditions)
    cfg = EvalConfig(
        agents=tuple(_split(args.agents) or ["react"]),
        suites=tuple(_split(args.suites)),
        conditions=tuple(conditions),
        policies=tuple(_split(args.policies) or ["eager"]),
        k=max(1, args.k),
        llm_spec=args.llm,
        llm_model=args.model if args.model != "mock" or args.llm == "mock" else args.llm,
        judge_enabled=args.judge == "auto",
        llm_temperature=args.temperature,
        max_steps=args.max_steps,
        seed=args.seed,
        attribution=args.attribution,
        corpus_dir=args.corpus or None,
        options={"keep_scripts_under_perturbation": bool(args.keep_scripts),
                 "host": args.host, "base_url": args.base_url,
                 "api_key": args.api_key, "judge_model": args.judge_model},
    )
    result = run(cfg, progress=lambda stage, pct, msg: _tick(stage, pct, msg))
    print()
    print(headline(result))

    if args.csv:
        tables = build_tables(result)
        tables["agent_scorecard"].to_csv(sys.stdout, index=False)
    if args.json:
        print(json.dumps(result.report, ensure_ascii=False, indent=2, default=str))

    if args.out:
        root = os.path.abspath(args.out)
        written = save(result, root)
        print("exported:")
        for kind, path in written.items():
            if isinstance(path, dict):
                for key, sub in path.items():
                    print(f"  {kind}/{key}: {sub}")
            else:
                print(f"  {kind}: {path}")
    return 0


def _probe(args) -> int:
    """``--ask``: one typed instruction, four pillars, before and after correction."""
    from .adaptive import ProbeOptions, probe_instruction, summarize

    policy = (_split(args.policies) or ["careful"])[0]
    endpoint = args.base_url if args.llm == "openai" else args.host
    opts = ProbeOptions.for_depth(
        args.depth, agent=(_split(args.agents) or ["react"])[0],
        policy=policy if policy != "eager" else "careful",
        llm_spec=args.llm, llm_model=args.model,
        endpoint=endpoint, api_key=args.api_key,
        temperature=args.temperature, max_steps=args.max_steps, seed=args.seed)
    res = probe_instruction(args.ask, opts=opts,
                            progress=lambda stage, pct, msg: _tick(stage, pct, msg))
    print()
    print(summarize(res))
    print(f"\n  {'condition':<22}{'register':<20}{'agr':>6}{'grnd':>7}"
          f"{'steps':>7}{'denied':>8}  answer")
    for row in res["variants"]:
        fmt = lambda v: "  n/a" if v is None else f"{v:5.2f}"
        print(f"  {(row['condition'] or '-'):<22}{row['register'] or '-':<20}"
              f"{fmt(row['agreement'])}{fmt(row['grounded'])}"
              f"{row['n_steps']:>7}{row['denied_calls']:>8}"
              f"  {row['answer_head'][:48].replace(chr(10), ' ')}")
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    if args.out:
        path = os.path.join(os.path.abspath(args.out),
                            f"adaptive-{res['meta']['depth']}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, default=str)
        print(f"probe written: {path}")
    return 0


def _tick(stage: str, pct: float, msg: str) -> None:
    sys.stdout.write(f"\r[{stage}] {pct * 100:5.1f}%  {msg[:70]}")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())

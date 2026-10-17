"""Command-line entry point for the Trust-oRAN toolkit.

Examples
--------
    python main_cli.py --list
    python main_cli.py --dataset cci_a1sd --model random_forest
    python main_cli.py --dataset ddos_pcap --pillars privacy fairness --quick
    python main_cli.py --dataset vran_energy --model grad_boost --out ../results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trustoran import config as C
from trustoran.data import available_datasets
from trustoran.models import available_models
from trustoran.pipeline import PILLARS, TrustPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trustoran",
        description="Trustworthy AI (explainability / robustness / fairness / privacy) "
                    "for Open RAN SMO & Near-RT RIC datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="cci_a1sd", choices=available_datasets(),
                   help="dataset to analyse")
    p.add_argument("--model", default="random_forest", help="model zoo key")
    p.add_argument("--pillars", nargs="*", default=list(PILLARS),
                   help=f"subset of {list(PILLARS)}")
    p.add_argument("--quick", action="store_true", help="smaller caps for fast runs")
    p.add_argument("--max-rows", type=int, default=None, help="row cap per raw file")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="results dir (default <repo>/results)")
    p.add_argument("--list", action="store_true", help="list datasets & models and exit")
    p.add_argument("--no-export", action="store_true", help="run but do not write results/")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print("datasets:")
        for d in available_datasets():
            print("  -", d)
        print("models:")
        for m in available_models():
            print("  -", m)
        print("pillars:", list(PILLARS))
        return 0

    C.ensure_results_dirs()
    out = Path(args.out) if args.out else C.RESULTS_DIR

    def progress(stage, pct, msg):
        print(f"  [{pct*100:5.1f}%] {stage:14s} {msg}", flush=True)

    print(f"== Trust-oRAN | dataset={args.dataset} model={args.model} "
          f"pillars={args.pillars} quick={args.quick}")
    pipe = TrustPipeline(
        dataset_key=args.dataset, model_name=args.model, pillars=args.pillars,
        max_rows=args.max_rows, quick=args.quick, seed=args.seed,
    )
    report = pipe.run(progress_cb=progress)

    print("\n-- headline:", report.headline())
    print("-- base metrics:", {k: round(v, 4) for k, v in report.base_metrics.items()})
    for name, res in report.pillars.items():
        print(f"\n[{name}]")
        if "error" in res.summary:
            print("   ERROR:", res.summary["error"])
            continue
        for k, v in res.summary.items():
            if isinstance(v, float):
                v = round(v, 4)
            print(f"   {k}: {v}")
        if res.tables:
            print("   tables:", list(res.tables.keys()))
        if res.figures:
            print("   figures:", list(res.figures.keys()))

    if not args.no_export:
        written = report.export(out)
        print("\n-- exported to", out)
        for cat, files in written.items():
            print(f"   {cat}: {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the full trustworthy-AI suite over every dataset choice and write an index.

    python run_all.py [--quick] [--out ../results]

Produces one report per dataset under ``results/<dataset>/`` plus a top-level
``results/index.json`` summarising every run (headline metrics + file counts).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trustoran import config as C
from trustoran.data import available_datasets
from trustoran.pipeline import TrustPipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", default="random_forest")
    ap.add_argument("--out", default=None)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of datasets to run (default: all)")
    args = ap.parse_args(argv)

    C.ensure_results_dirs()
    out = Path(args.out) if args.out else C.RESULTS_DIR
    keys = args.datasets or available_datasets()

    index = {}
    for key in keys:
        t = time.time()
        print(f"\n########## {key} ##########", flush=True)
        try:
            pipe = TrustPipeline(dataset_key=key, model_name=args.model, quick=args.quick)
            rep = pipe.run()
            written = rep.export(out)
            index[key] = {
                "headline": rep.headline(),
                "base_metrics": {k: round(v, 4) for k, v in rep.base_metrics.items()},
                "pillars": {p: {"summary": _round(r.summary),
                                "n_figures": len(r.figures),
                                "n_tables": len(r.tables)}
                            for p, r in rep.pillars.items()},
                "files": {k2: len(v) for k2, v in written.items()},
                "elapsed_sec": round(time.time() - t, 1),
            }
        except Exception as e:
            index[key] = {"error": f"{type(e).__name__}: {e}"}
            print("   !! failed:", e, flush=True)

    idx_path = out / "index.json"
    idx_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n== index written to {idx_path}")
    for k, v in index.items():
        status = v.get("error", "ok")
        print(f"   {k:20s} {status if isinstance(status, str) and 'Error' in status else v.get('headline','')}")
    return 0


def _round(d):
    out = {}
    for k, v in d.items():
        out[k] = round(v, 4) if isinstance(v, float) else v
    return out


if __name__ == "__main__":
    raise SystemExit(main())

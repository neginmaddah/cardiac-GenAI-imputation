#!/usr/bin/env python
"""Run v2 downstream prediction over whatever imputed matrices exist.

Idempotent and shardable: one output file per (unit, method, target), a
filesystem lease per cell, and anything already written is skipped. Several
copies can therefore run at once -- in sbatch shards, in an interactive session,
or both -- without coordination.

    python pipeline/05_predict.py                       # everything available
    python pipeline/05_predict.py --methods mice knn    # a subset
    python pipeline/05_predict.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvd import conf  # noqa: E402
from cvd.util.lease import Claim  # noqa: E402
from cvd.stages.prediction_v2 import load_unit, parse_stem, predict_unit  # noqa: E402

logger = logging.getLogger("v2_predict")


def main(argv=None) -> int:
    cfg = conf.load()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="*")
    p.add_argument("--cohorts", nargs="*")
    p.add_argument("--targets", nargs="*")
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--deadline-min", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    id_column = str(cfg["harmonise.id_column"])
    binary = [str(t) for t in cfg["harmonise.targets.categorical"]]
    continuous = [str(t) for t in cfg["harmonise.targets.continuous"]]
    wanted = a.targets or (binary + continuous)

    book = json.loads((Path(str(cfg["harmonise.report_dir"])) / "codebook.json").read_text())
    kinds = book["kinds"]

    out_dir = Path(str(cfg["results_v2.prediction_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    tpl = str(cfg["results_v2.filenames"]["prediction"])
    claims = Path(str(cfg["results_v2.claims_dir"]))

    # The baseline arm is the masked matrix itself; it is already on disk as a
    # prepared input, so it needs no imputation run to exist.
    sources = []
    for path in sorted(Path(str(cfg["results_v2.imputed_dir"])).glob("*.csv")):
        info = parse_stem(path.stem)
        if info:
            sources.append((info, path))
    # Baselines only for the designs and intensities this family actually runs.
    # The inputs directory still holds all 18 units -- MAR and the 10% rung are
    # prepared and frozen but set aside -- so globbing it unfiltered would add
    # arms the reduced grid has no imputed counterparts for.
    in_dir = Path(str(cfg["inputs.inputs_dir"]))
    live_designs = {str(d).upper() for d in cfg["experiment.v2.designs"]}
    live_alphas = {str(int(round(float(x) * 100))) for x in cfg["experiment.v2.alphas"]}
    for path in sorted(in_dir.glob("*_canonical_masked.parquet")):
        info = parse_stem(path.stem.replace("_canonical_masked", "_baseline"))
        if info and info["design"] in live_designs and info["alpha"] in live_alphas:
            sources.append((info, path))

    cells = []
    for info, path in sources:
        if a.methods and info["method"] not in a.methods:
            continue
        if a.cohorts and info["cohort"] not in a.cohorts:
            continue
        tag = f"{info['cohort']}_{info['design']}_{info['alpha']}"
        for target in wanted:
            dest = out_dir / tpl.format(tag=tag, method=info["method"], target=target)
            if dest.exists():
                continue
            cells.append((info, path, tag, target, dest))

    print(f"{len(cells)} prediction cell(s) outstanding "
          f"({len(sources)} matrices x {len(wanted)} targets)")
    if a.dry_run:
        for info, _, tag, target, _ in cells[:40]:
            print(f"   {tag}_{info['method']}  {target}")
        return 0

    deadline = None if a.deadline_min is None else time.monotonic() + a.deadline_min * 60
    done = failed = 0
    for info, path, tag, target, dest in cells:
        if deadline is not None and time.monotonic() >= deadline:
            print("deadline reached; stopping")
            break
        lease = Claim(claims / f"pred_{tag}_{info['method']}_{target}.claim", 3 * 3600)
        if not lease.acquire():
            continue
        try:
            if dest.exists():
                continue
            started = time.monotonic()
            X, merged = load_unit(cfg, info["cohort"], path, id_column)
            result = predict_unit(
                cfg, X, merged, kinds, target,
                is_binary=target in binary, id_column=id_column, n_jobs=a.n_jobs,
            )
            result.update({
                "cohort": info["cohort"], "design": info["design"],
                "alpha": int(info["alpha"]) / 100, "method": info["method"],
                "target": target, "is_binary": target in binary,
                "seconds": round(time.monotonic() - started, 1),
            })
            dest.write_text(json.dumps(result, indent=1, default=float))
            done += 1
            logger.info("[%s_%s] %s done in %.0fs", tag, info["method"], target,
                        result["seconds"])
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.error("[%s_%s] %s FAILED: %s", tag, info["method"], target, exc)
        finally:
            lease.release()
    print(f"prediction: {done} written, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

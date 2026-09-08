"""E4: the untested combination -- alignment AND cross-task transfer together.

Block E ran the transfer directions without Euclidean Alignment, and block D5
showed EA is the single most useful step in the pipeline. Those were the two
biggest individual wins and they were never combined. If they stack, executed
data plus EA is the best way to decode a new person's imagery.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data import load_dataset, good_subjects, IMAGINED_RUNS, EXECUTED_RUNS
from src.align import align_by_group
from src.models import PIPELINES
from src.evaluate import loso_cross, summarise
from src.report import write_rows, line


def main(a):
    subs = good_subjects(a.subjects)
    imag = load_dataset(subs, IMAGINED_RUNS)
    exe = load_dataset(subs, EXECUTED_RUNS)
    pipe = PIPELINES[a.pipeline]

    # EA is per-subject and unsupervised, so it is applied to each paradigm's
    # trials independently -- exactly what predict.py could do on a new file.
    imag_a = align_by_group(imag.X, imag.groups)
    exe_a = align_by_group(exe.X, exe.groups)

    Xp = np.concatenate([imag_a, exe_a])
    yp = np.concatenate([imag.y, exe.y])
    gp = np.concatenate([imag.groups, exe.groups])

    runs = {
        "E4 imagined->imagined  +EA": (imag_a, imag.y, imag.groups, imag_a, imag.y, imag.groups),
        "E4 executed->imagined  +EA": (exe_a, exe.y, exe.groups, imag_a, imag.y, imag.groups),
        "E4 imagined->executed  +EA": (imag_a, imag.y, imag.groups, exe_a, exe.y, exe.groups),
        "E4 pooled  ->imagined  +EA": (Xp, yp, gp, imag_a, imag.y, imag.groups),
    }
    rows = []
    for name, (Xtr, ytr, gtr, Xte, yte, gte) in runs.items():
        r = loso_cross(Xtr, ytr, gtr, Xte, yte, gte, pipe())
        print(line(r, name))
        s = summarise(r, name)
        cross = sum(1 for x in r if x["ci_lo"] <= 0.5 <= x["ci_hi"])
        print(f"      CIs crossing chance: {cross}/{len(r)}")
        rows.append({"id": name, "pipeline": a.pipeline, "acc": s["mean"],
                     "sd": s["sd"], "min": s["min"], "max": s["max"],
                     "ci_cross_chance": cross, "n_subjects": s["n_subjects"]})
    write_rows(f"results/R6_transfer_with_alignment.csv", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--pipeline", default="csp_lda", choices=list(PIPELINES))
    main(ap.parse_args())

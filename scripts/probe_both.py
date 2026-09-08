"""Is the subject fingerprint specific to imagery, or is it just there?

Block D only probed the imagined runs. If the surviving spectral fingerprint
showed up only under imagery it would be a much weaker claim -- possibly
something about that paradigm rather than about the person. This runs the same
probes on the executed runs too.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data import load_dataset, good_subjects, IMAGINED_RUNS, EXECUTED_RUNS
from src.align import align_by_group
from src.probes import subject_id_probe, logvar_features, psd_features
from src.report import write_rows


def main(a):
    rows = []
    for para, runs in (("imagined", IMAGINED_RUNS), ("executed", EXECUTED_RUNS)):
        ds = load_dataset(good_subjects(a.subjects), runs)
        Xa = align_by_group(ds.X, ds.groups)
        n = len(np.unique(ds.groups))
        print(f"\n{para}: {ds.X.shape}, {n} subjects, chance = {1/n:.4f}")
        feats = {"logvar": logvar_features,
                 "psd_rel": lambda X: psd_features(X, relative=True),
                 "psd_abs": lambda X: psd_features(X, relative=False)}
        for fname, fn in feats.items():
            for tag, Xu in (("no_EA", ds.X), ("EA", Xa)):
                p = subject_id_probe(Xu, ds.groups, ds.run, fn)
                print(f"  {fname:8s} {tag:5s}: {p['acc']:.4f} +/- {p['sd']:.4f}")
                rows.append({"paradigm": para, "features": fname, "alignment": tag,
                             "acc": p["acc"], "sd": p["sd"], "chance": p["chance"],
                             "n_subjects": p["n_subjects"]})
    write_rows("results/R4_person_vs_task.csv", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=30)
    main(ap.parse_args())

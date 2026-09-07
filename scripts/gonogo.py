"""The go/no-go check. Run this before building the rest of the grid.

Four numbers decide the framing of the whole submission:
  D1/D2  subject identity from covariance features, without and with EA
  D3/D4  subject identity from spectral features, without and with EA

If D4 stays high, EA removes the covariance fingerprint and leaves the spectral
one intact, and that is the headline. If D4 collapses, the prediction is
falsified and B3 (the leakage gap) becomes the headline instead. Either way
there is a submission, which is why this runs first.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import load_dataset, good_subjects, IMAGINED_RUNS
from src.align import align_by_group
from src.models import csp_lda, tangent_space
from src.evaluate import loso, random_kfold, summarise
from src.probes import subject_id_probe, logvar_features, psd_features


def main(n_subjects, out):
    t0 = time.time()
    subs = good_subjects(n_subjects)
    print(f"loading {len(subs)} subjects, imagined runs {IMAGINED_RUNS}")
    ds = load_dataset(subs, IMAGINED_RUNS)
    print(f"  X={ds.X.shape} y={np.bincount(ds.y)} "
          f"subjects={len(np.unique(ds.groups))} runs={sorted(set(ds.run.tolist()))}")
    per_sub = [int((ds.groups == s).sum()) for s in np.unique(ds.groups)]
    print(f"  trials/subject: min={min(per_sub)} med={int(np.median(per_sub))} max={max(per_sub)}")

    rows = []
    pipes = {"csp_lda": csp_lda(4), "tangent_space": tangent_space()}

    # --- B1: the wrong protocol. Trials pooled and split at random, so the
    # model sees the test subject in training.
    for name, pipe in pipes.items():
        f = random_kfold(ds.X, ds.y, pipe)
        rows.append(("B1", f"random 5-fold pooled [{name}]", float(np.mean(f)), float(np.std(f))))
        print(f"B1 {name:14s} random 5-fold : {np.mean(f):.3f} +/- {np.std(f):.3f}")

    # --- B2: the honest protocol.
    b2 = {}
    for name, pipe in pipes.items():
        r = loso(ds.X, ds.y, ds.groups, pipe)
        s = summarise(r, name)
        b2[name] = s["mean"]
        rows.append(("B2", f"LOSO imagery [{name}]", s["mean"], s["sd"]))
        print(f"B2 {name:14s} LOSO          : {s['mean']:.3f} +/- {s['sd']:.3f} "
              f"(range {s['min']:.2f}-{s['max']:.2f})")

    # --- B3: the gap between them is a direct measurement of subject leakage.
    for name in pipes:
        b1 = [r[2] for r in rows if r[0] == "B1" and name in r[1]][0]
        gap = b1 - b2[name]
        rows.append(("B3", f"leakage gap [{name}]", gap, float("nan")))
        print(f"B3 {name:14s} leakage gap   : {gap:+.3f} ({gap*100:+.1f} pp)")

    # --- D1-D4: what identity survives alignment.
    Xa = align_by_group(ds.X, ds.groups)
    feats = {"logvar": logvar_features,
             "psd_rel": lambda X: psd_features(X, relative=True)}
    ids = {"D1": ("logvar", ds.X, "no EA"), "D2": ("logvar", Xa, "EA"),
           "D3": ("psd_rel", ds.X, "no EA"), "D4": ("psd_rel", Xa, "EA")}
    for k, (fname, Xu, tag) in ids.items():
        p = subject_id_probe(Xu, ds.groups, ds.run, feats[fname])
        rows.append((k, f"subject-ID {fname} {tag}", p["acc"], p["sd"]))
        print(f"{k} subject-ID {fname:8s} {tag:5s}: {p['acc']:.3f} +/- {p['sd']:.3f} "
              f"(chance {p['chance']:.3f}, {p['n_subjects']} subjects, leave-one-run-out)")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("id,experiment,value,sd\n")
        for r in rows:
            fh.write(f"{r[0]},\"{r[1]}\",{r[2]:.4f},{r[3]:.4f}\n")
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=15)
    ap.add_argument("--out", default="results/gonogo.csv")
    a = ap.parse_args()
    main(a.subjects, a.out)

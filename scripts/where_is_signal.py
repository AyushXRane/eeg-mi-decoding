"""Criterion 4: how much of the score is the thing I claim to be decoding?

Chance is a weak reference. Three stronger checks that the model is using motor
cortex activity and not something else:

  a) channel ablation -- if the signal is sensorimotor, motor electrodes should
     carry it and frontal/occipital ones should not
  b) frequency band -- the mu/beta effect only exists at 8-30 Hz. Refit at
     30-70 Hz, where there is no such effect, and imagery should collapse.
     Execution should not, because muscle activity is broadband.
  c) CSP spatial patterns -- plot what the filters actually weight. A filter
     peaking over C3/C4 is direct evidence the model found hand motor cortex.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import load_dataset, good_subjects, IMAGINED_RUNS, EXECUTED_RUNS
from src.align import align_by_group
from src.models import csp_lda
from src.evaluate import loso, summarise
from src.probes import pick_channels, MOTOR, NON_MOTOR
from src.report import write_rows


def main(a):
    subs = good_subjects(a.subjects)
    rows = []

    # (a) channel ablation
    ds = load_dataset(subs, IMAGINED_RUNS)
    X = align_by_group(ds.X, ds.groups)
    sets = {"all_64": list(range(len(ds.ch_names))),
            "motor_only": pick_channels(ds.ch_names, MOTOR),
            "non_motor_only": pick_channels(ds.ch_names, NON_MOTOR)}
    print("(a) channel ablation, imagery")
    for name, idx in sets.items():
        s = summarise(loso(X[:, idx, :], ds.y, ds.groups, csp_lda(4)))
        print(f"    {name:16s} ({len(idx):2d} ch)  {s['mean']:.3f} +/- {s['sd']:.3f}")
        rows.append({"check": "channels", "condition": name, "n_ch": len(idx),
                     "acc": s["mean"], "sd": s["sd"]})

    # (b) frequency band control
    print("\n(b) band control -- 30-70 Hz has no mu/beta effect to find")
    for para, runs in (("imagined", IMAGINED_RUNS), ("executed", EXECUTED_RUNS)):
        for lo, hi in ((8.0, 30.0), (30.0, 70.0)):
            d = load_dataset(subs, runs, l_freq=lo, h_freq=hi)
            Xa = align_by_group(d.X, d.groups)
            s = summarise(loso(Xa, d.y, d.groups, csp_lda(4)))
            print(f"    {para:9s} {lo:.0f}-{hi:.0f} Hz   {s['mean']:.3f} +/- {s['sd']:.3f}")
            rows.append({"check": "band", "condition": f"{para}_{lo:.0f}-{hi:.0f}Hz",
                         "n_ch": 64, "acc": s["mean"], "sd": s["sd"]})

    write_rows("results/R7_where_is_signal.csv", rows)

    # (c) what the spatial filters look like
    import matplotlib; matplotlib.use("Agg")
    import mne
    csp = csp_lda(4).named_steps["csp"].fit(X, ds.y)
    info = mne.create_info(ds.ch_names, 160.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"),
                     on_missing="ignore")
    fig = csp.plot_patterns(info, ch_type="eeg", units="a.u.", size=1.4, show=False)
    fig.suptitle("What the CSP filters weight (imagery, 8-30 Hz, all subjects)",
                 fontsize=10)
    fig.savefig("results/R7_csp_patterns.png", dpi=150)
    print("\n(c) -> results/R7_csp_patterns.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=109)
    main(ap.parse_args())

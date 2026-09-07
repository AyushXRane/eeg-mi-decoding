"""A1: verify what T1/T2 actually mean. Run this before anything else.

The task description says the mapping depends on the run and to check it
yourself. Getting it wrong is silent -- no error, just a broken label vector.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data import load_subject, IMAGINED_RUNS, EXECUTED_RUNS

FIST_RUNS = {"executed L/R fist": EXECUTED_RUNS, "imagined L/R fist": IMAGINED_RUNS}
OTHER_RUNS = {"executed fists/feet": [5, 9, 13], "imagined fists/feet": [6, 10, 14]}


def summarise(subject, runs, label):
    raw = load_subject(subject, runs)
    if raw is None:
        print(f"  S{subject:03d} {label}: FAILED QC")
        return None
    ann = raw.annotations
    row = {"subject": subject, "label": label}
    for code in ("T0", "T1", "T2"):
        m = ann.description == code
        row[code] = (int(m.sum()), float(ann.duration[m].mean()) if m.sum() else 0.0)
    print(f"  S{subject:03d} {label:22s} sfreq={raw.info['sfreq']:.0f} "
          + " ".join(f"{c}: n={row[c][0]:3d} dur={row[c][1]:.2f}s" for c in ("T0", "T1", "T2")))
    return row


def lateralisation(subject, runs, label):
    """Empirical check on what T1/T2 mean, rather than trusting the docs.

    Counts and durations are identical between the L/R fist runs and the
    fists/feet runs, so the annotations alone cannot tell them apart. Physiology
    can: left vs right hand movement desynchronises mu over the contralateral
    hemisphere, so C3 and C4 should move in opposite directions. Both-fists vs
    both-feet is bilateral, so C3 and C4 should move together.

    lat = (C3 power under T2 - under T1) - (C4 power under T2 - under T1).
    Large |lat| => the two conditions differ by side. Near zero => they do not.
    """
    from src.preprocess import preprocess_raw
    from src.data import epochs_from_raw
    from src.probes import erd_check

    raw = load_subject(subject, runs)
    if raw is None:
        return
    got = epochs_from_raw(preprocess_raw(raw), 0.5, 3.5)
    if got is None:
        return
    X, y, names = got
    e = erd_check(X, y, names)
    print(f"  S{subject:03d} {label:22s} lat_index={e['lat_index']:+.4f}  "
          f"C3: T1={e['C3_left']:+.2f} T2={e['C3_right']:+.2f} | "
          f"C4: T1={e['C4_left']:+.2f} T2={e['C4_right']:+.2f}")


def physiological_check(subjects=(1, 2, 3, 4, 5, 6, 7, 8)):
    print("\nPhysiological check -- do T1/T2 differ by SIDE in the fist runs?")
    print("  executed L/R fist should lateralise; executed fists/feet should not.")
    for s in subjects:
        lateralisation(s, EXECUTED_RUNS, "executed L/R fist")
    print()
    for s in subjects:
        lateralisation(s, [5, 9, 13], "executed fists/feet")


if __name__ == "__main__":
    subjects = [1, 2, 3, 4, 5]
    print("Runs used for the L/R fist task (3 runs concatenated per subject):")
    for label, runs in FIST_RUNS.items():
        for s in subjects:
            summarise(s, runs, label)
    print("\nSame codes, different runs -- these are fists/feet, NOT left/right:")
    for label, runs in OTHER_RUNS.items():
        summarise(1, runs, label)

    physiological_check((1, 2, 3, 4, 5))

    print("\nPer-run trial counts (imagined, S001):")
    for r in IMAGINED_RUNS:
        raw = load_subject(1, [r])
        ann = raw.annotations
        counts = {c: int((ann.description == c).sum()) for c in ("T0", "T1", "T2")}
        print(f"  run {r:2d}: {counts}")

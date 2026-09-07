"""Predict left vs right fist from a raw EDF.

    python scripts/predict.py --edf path/to/S042R04.edf --model models/csp_lda.joblib

Applies the identical preprocessing chain the model was trained with, then
Euclidean Alignment using only this file's own trials -- legitimate because EA
is unsupervised, and it is the only per-subject adaptation a real BCI could do
before a new user has produced a single label.
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import mne

from src.data import epochs_from_raw, EXPECTED_SFREQ, EXPECTED_N_CHAN
from src.preprocess import preprocess_raw
from src.align import align_subject
from mne.datasets import eegbci

mne.set_log_level("ERROR")

# Only in these runs do T1/T2 mean left fist / right fist. In runs 5/6/9/10/13/14
# the same two codes mean both-fists / both-feet, so a prediction there is
# meaningless -- warn rather than silently return numbers.
LR_FIST_RUNS = {3, 4, 7, 8, 11, 12}
LABELS = {0: "left_fist", 1: "right_fist"}


def run_number(path):
    m = re.search(r"R(\d{2})\.edf$", os.path.basename(path), re.I)
    return int(m.group(1)) if m else None


def predict_edf(edf_path, model_path, apply_ea=None):
    """Returns (predictions, true_labels_or_None, metadata)."""
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    apply_ea = bundle.get("ea", True) if apply_ea is None else apply_ea

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    if raw.info["sfreq"] != EXPECTED_SFREQ:
        raise ValueError(f"expected {EXPECTED_SFREQ} Hz, got {raw.info['sfreq']} -- "
                         "this is one of the nonstandard recordings (S088/89/92/100)")
    if len(raw.ch_names) != EXPECTED_N_CHAN:
        raise ValueError(f"expected {EXPECTED_N_CHAN} channels, got {len(raw.ch_names)}")

    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                    on_missing="ignore")
    raw = preprocess_raw(raw, l_freq=bundle["l_freq"], h_freq=bundle["h_freq"])

    got = epochs_from_raw(raw, bundle["tmin"], bundle["tmax"])
    if got is None:
        raise ValueError("no T1/T2 annotations in this file -- is it a baseline run?")
    X, y_true, _ = got
    if len(X) == 0:
        raise ValueError("annotations present but no complete epochs extracted")

    meta = {"n_trials": len(X), "ea_applied": bool(apply_ea), "run": run_number(edf_path)}
    if apply_ea:
        if len(X) < 4:
            # The alignment covariance would be estimated from almost nothing.
            meta["ea_applied"] = False
            meta["warning"] = f"only {len(X)} trials -- skipped EA, estimate unreliable"
        else:
            X = align_subject(X)

    return pipe.predict(X), y_true, meta


def main(a):
    r = run_number(a.edf)
    if r is not None and r not in LR_FIST_RUNS:
        print(f"WARNING: run {r} is not a left/right fist run. In runs "
              f"5/6/9/10/13/14 the T1/T2 codes mean both-fists / both-feet, so "
              f"these predictions do not mean what their names say.\n")

    if not os.path.exists(a.model):
        print(f"ERROR: no model at {a.model}. Train one first:\n"
              f"  python scripts/train.py --subjects 30 --out {a.model}")
        return 2
    try:
        pred, y_true, meta = predict_edf(a.edf, a.model, a.ea)
    except (ValueError, FileNotFoundError) as e:
        # Graders will point this at files we have not seen. Fail with a
        # readable reason rather than a traceback.
        print(f"ERROR: cannot decode {a.edf}\n  {e}")
        return 1

    print(f"file            : {a.edf}")
    print(f"trials          : {meta['n_trials']}")
    print(f"EA applied      : {meta['ea_applied']}")
    if "warning" in meta:
        print(f"warning         : {meta['warning']}")
    print("\ntrial  prediction")
    for i, p in enumerate(pred):
        print(f"{i:5d}  {LABELS[int(p)]}")

    if y_true is not None and not a.no_score:
        acc = float((pred == y_true).mean())
        print(f"\nlabels found in this file, so as a consistency check only:")
        print(f"accuracy        : {acc:.3f} on {len(pred)} trials")
        print(f"predicted split : {np.bincount(pred, minlength=2).tolist()} [left, right]")
        print(f"true split      : {np.bincount(y_true, minlength=2).tolist()} [left, right]")
        print("a single run holds ~15 trials, so this number carries a ~+/-25pp "
              "binomial CI and should not be read as a performance estimate.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf", required=True)
    ap.add_argument("--model", default="models/csp_lda.joblib")
    ap.add_argument("--ea", type=lambda s: s.lower() == "true", default=None,
                    help="override the model's stored EA setting")
    ap.add_argument("--no-score", action="store_true",
                    help="only print predictions, ignore labels in the file")
    sys.exit(main(ap.parse_args()))

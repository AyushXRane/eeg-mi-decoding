"""Loading eegmmidb from PhysioNet and getting it into (trials, channels, time)."""

import hashlib
import os
from typing import NamedTuple

import numpy as np
import mne
from mne.datasets import eegbci

mne.set_log_level("ERROR")

# Runs where T1/T2 mean left fist / right fist. Verified, not assumed -- see
# scripts/verify_labels.py. In runs 5/6/9/10/13/14 the same codes mean
# both-fists / both-feet, which is why only these six appear here.
IMAGINED_RUNS = [4, 8, 12]
EXECUTED_RUNS = [3, 7, 11]

# Different papers drop different subjects here. These four are the ones that
# actually break: their recordings use a different sampling rate or a different
# trial duration from everyone else, so epochs come out the wrong shape.
# S038/S104/S106 get dropped in some curations for annotation problems but load
# fine, so they stay in and get caught by the QC check in load_subject instead.
BAD_SUBJECTS = [88, 89, 92, 100]

EXPECTED_SFREQ = 160.0
EXPECTED_N_CHAN = 64

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "mne_data", "epochs_cache")


class Dataset(NamedTuple):
    """Epoched trials plus every grouping variable an honest split might need.

    `run` matters: a subject-ID probe that only works within a run is a session
    artifact, not a fingerprint. Keeping it lets us test across runs.
    """
    X: np.ndarray        # (n_trials, n_channels, n_times)
    y: np.ndarray        # 0 = left fist, 1 = right fist
    groups: np.ndarray   # subject id per trial
    run: np.ndarray      # run number per trial
    ch_names: list


def good_subjects(n):
    """First n usable subject ids."""
    out = [s for s in range(1, 110) if s not in BAD_SUBJECTS]
    return out[:n]


def load_subject(subject, runs, verify=False):
    """Concatenated raw for one subject, channel names cleaned, montage set.

    Returns None if the recording fails QC (wrong sampling rate or channel
    count) so callers can skip it instead of crashing mid-sweep.
    """
    paths = eegbci.load_data(subject, runs, update_path=True, verbose=False)
    raws = [mne.io.read_raw_edf(p, preload=True, verbose=False) for p in paths]

    for r in raws:
        if r.info["sfreq"] != EXPECTED_SFREQ or len(r.ch_names) != EXPECTED_N_CHAN:
            return None

    raw = mne.concatenate_raws(raws) if len(raws) > 1 else raws[0]

    # EDF channel names come in as 'Fc5.', 'C3..' -- trailing dots and odd case.
    # standardize() strips them so the montage will match.
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                    on_missing="ignore")

    if verify:
        _print_annotations(raw, subject, runs)

    return raw


def _print_annotations(raw, subject, runs):
    """Dump the annotation structure so the T1/T2 mapping can be checked by hand.

    The task description says the meaning of T1/T2 depends on the run and to
    check it yourself, so this exists to actually do that rather than trust it.
    """
    ann = raw.annotations
    print(f"S{subject:03d} runs={runs}")
    for code in ["T0", "T1", "T2"]:
        m = ann.description == code
        if m.sum():
            print(f"  {code}: n={m.sum():3d} "
                  f"mean_dur={ann.duration[m].mean():.2f}s")
    print(f"  first onsets: {np.round(ann.onset[:6], 2)}")


def epochs_from_raw(raw, tmin=0.5, tmax=3.5):
    """Epoch the two fist conditions. T1 -> left (0), T2 -> right (1)."""
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    wanted = {k: v for k, v in event_id.items() if k in ("T1", "T2")}
    if len(wanted) != 2:
        return None

    ep = mne.Epochs(raw, events, wanted, tmin=tmin, tmax=tmax,
                    baseline=None, preload=True, verbose=False,
                    on_missing="ignore")

    X = ep.get_data(copy=True)
    y = (ep.events[:, 2] == wanted["T2"]).astype(int)
    return X, y, ep.ch_names


def _cache_key(subjects, runs, tmin, tmax, l_freq, h_freq, car):
    s = f"{sorted(subjects)}|{sorted(runs)}|{tmin}|{tmax}|{l_freq}|{h_freq}|{car}"
    return hashlib.md5(s.encode()).hexdigest()[:16]


def load_dataset(subjects, runs, tmin=0.5, tmax=3.5, l_freq=8.0, h_freq=30.0,
                 car=True, cache=True, verbose=True):
    """Stack many subjects into a Dataset.

    Loads run by run rather than concatenating first, so each trial keeps the
    run it came from. Filtering and CAR happen on continuous data before
    epoching -- filter edge effects at epoch boundaries would otherwise eat into
    the window we care about.
    """
    from src.preprocess import preprocess_raw

    key = _cache_key(subjects, runs, tmin, tmax, l_freq, h_freq, car)
    path = os.path.join(CACHE_DIR, f"{key}.npz")
    if cache and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return Dataset(z["X"], z["y"], z["groups"], z["run"], list(z["ch_names"]))

    Xs, ys, gs, rs = [], [], [], []
    names = None
    skipped = []

    for s in subjects:
        for r in runs:
            raw = load_subject(s, [r])
            if raw is None:
                skipped.append((s, r, "QC"))
                continue

            raw = preprocess_raw(raw, l_freq=l_freq, h_freq=h_freq, car=car)
            got = epochs_from_raw(raw, tmin, tmax)
            if got is None:
                skipped.append((s, r, "no T1/T2"))
                continue

            X, y, names = got
            Xs.append(X)
            ys.append(y)
            gs.append(np.full(len(y), s))
            rs.append(np.full(len(y), r))

    if not Xs:
        raise RuntimeError("no subjects loaded")

    if verbose and skipped:
        print(f"  skipped {len(skipped)}: {skipped[:10]}")

    ds = Dataset(np.concatenate(Xs), np.concatenate(ys),
                 np.concatenate(gs), np.concatenate(rs), names)

    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, X=ds.X, y=ds.y, groups=ds.groups,
                            run=ds.run, ch_names=np.array(ds.ch_names))
    return ds

"""Loading eegmmidb from PhysioNet and getting it into (trials, channels, time)."""

import numpy as np
import mne
from mne.datasets import eegbci

mne.set_log_level("ERROR")

# Runs where T1/T2 mean left fist / right fist.
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


def good_subjects(n):
    """First n usable subject ids."""
    out = [s for s in range(1, 110) if s not in BAD_SUBJECTS]
    return out[:n]


def load_subject(subject, runs, verify=False):
    """Concatenated raw for one subject, channel names cleaned, montage set.

    Returns None if the recording fails QC (wrong sampling rate or channel
    count) so callers can skip it instead of crashing mid-sweep.
    """
    paths = eegbci.load_data(subject, runs, verbose=False)
    raws = [mne.io.read_raw_edf(p, preload=True, verbose=False) for p in paths]

    for r in raws:
        if r.info["sfreq"] != EXPECTED_SFREQ or len(r.ch_names) != EXPECTED_N_CHAN:
            return None

    raw = mne.concatenate_raws(raws)

    # EDF channel names come in as 'Fc5.', 'C3..' — trailing dots and odd case.
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


def load_dataset(subjects, runs, tmin=0.5, tmax=3.5, preprocess_fn=None):
    """Stack many subjects into X, y, groups.

    groups is the subject id per trial. Every split in this project uses it —
    trials from one person must never straddle train and test.
    """
    Xs, ys, gs = [], [], []
    names = None

    for s in subjects:
        raw = load_subject(s, runs)
        if raw is None:
            print(f"  skip S{s:03d} (failed QC)")
            continue

        if preprocess_fn is not None:
            raw = preprocess_fn(raw)

        got = epochs_from_raw(raw, tmin, tmax)
        if got is None:
            print(f"  skip S{s:03d} (missing T1/T2)")
            continue

        X, y, names = got
        Xs.append(X)
        ys.append(y)
        gs.append(np.full(len(y), s))

    if not Xs:
        raise RuntimeError("no subjects loaded")

    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs), names

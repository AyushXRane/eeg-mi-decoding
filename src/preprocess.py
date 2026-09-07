"""Bandpass and re-reference. Everything upstream of epoching lives here."""

import mne


def preprocess_raw(raw, l_freq=8.0, h_freq=30.0, car=True):
    """Zero-phase FIR bandpass, then common average reference.

    8-30 Hz is mu + beta, where event-related desynchronisation lives. It also
    removes 60 Hz line noise and sub-4 Hz drift for free, which is most of the
    eye-blink energy -- that is why there is no ICA step.

    CAR shrinks the magnitude differences between subjects that a shared
    reference electrode introduces, which matters for cross-subject transfer.
    """
    raw = raw.copy().filter(l_freq, h_freq, method="fir", phase="zero",
                            fir_design="firwin", verbose=False)
    if car:
        raw = raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw

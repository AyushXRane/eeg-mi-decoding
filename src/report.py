"""CSV writing without pandas, so requirements.txt stays as pinned."""

import csv
import os


def write_rows(path, rows, fields=None):
    """Union of every row's keys, not just the first row's.

    Blocks write heterogeneous rows -- F1 rows carry per-subject fields, F2 rows
    carry n_train_subjects -- and taking fieldnames from rows[0] silently
    dropped every column the first row happened not to have.
    """
    if not rows:
        return
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path} ({len(rows)} rows)")


def line(rows, label):
    """mean +/- sd, never a bare mean."""
    import numpy as np
    a = np.array([r["acc"] for r in rows])
    return (f"{label:44s} {a.mean():.3f} +/- {a.std():.3f}  "
            f"[{a.min():.2f}-{a.max():.2f}]  n={len(a)}")

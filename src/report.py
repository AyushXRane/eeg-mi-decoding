"""CSV writing without pandas, so requirements.txt stays as pinned."""

import csv
import os


def write_rows(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
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

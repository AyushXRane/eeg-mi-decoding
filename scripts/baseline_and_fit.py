"""Results 3 and 5: the real baseline, and whether the model memorised.

3. Permutation null -- shuffle labels within each person, rerun the whole
   leave-one-subject-out evaluation 200 times. Chance is 0.50, but the number
   the result actually has to beat is the 95th percentile of this null.

5. Train vs test accuracy on the same folds. A model that fits its training set
   perfectly and generalises at chance has memorised, not learned.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import load_dataset, good_subjects, IMAGINED_RUNS
from src.models import csp_lda, tangent_space
from src.evaluate import loso, permutation_null_fast
from src.report import write_rows


def main(a):
    ds = load_dataset(good_subjects(a.subjects), IMAGINED_RUNS)
    print(f"{ds.X.shape}, {len(np.unique(ds.groups))} subjects")

    # --- 3. permutation null
    if a.skip_perm:
        print('\n[3] skipped (results/R3_permutation.csv already present)')
    else:
      null, real = permutation_null_fast(ds.X, ds.y, ds.groups, n_perm=a.n_perm)
      p = float((null >= real).mean() + 1 / (len(null) + 1))
      print(f"\n[3] permutation null over {len(null)} within-subject shuffles")
      print(f"    null mean {null.mean():.3f}   95th pct {np.percentile(null,95):.3f}"
            f"   max {null.max():.3f}")
      print(f"    observed  {real:.3f}   p = {p:.4f}")
      write_rows("results/R3_permutation.csv",
                 [{"perm": i, "acc": float(v)} for i, v in enumerate(null)]
                 + [{"perm": "observed", "acc": real}])

    # --- 5. fit gap
    print("\n[5] train vs test on the same folds")
    rows = []
    for name, pipe in (("csp_lda", csp_lda(4)), ("tangent_space", tangent_space())):
        r = loso(ds.X, ds.y, ds.groups, pipe, return_train=True)
        tr = float(np.mean([x["train_acc"] for x in r]))
        te = float(np.mean([x["acc"] for x in r]))
        print(f"    {name:14s} train {tr:.3f}   test {te:.3f}   gap {(tr-te)*100:.1f} pp")
        rows.append({"pipeline": name, "train": tr, "test": te, "gap_pp": (tr - te) * 100})
    write_rows("results/R5_fit_gap.csv", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--skip-perm", action="store_true")
    main(ap.parse_args())

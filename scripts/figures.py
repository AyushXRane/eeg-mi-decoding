"""Figures from the CSVs in results/. Run the relevant grid blocks first.

Every plot that has a chance level draws it, because for the subject-ID probes
chance is 1/n_subjects and a reader who assumes 0.5 will misread the figure.
"""
import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "results"


def read(name):
    p = f"{R}/{name}"
    if not os.path.exists(p):
        print(f"  (skip {name}, not found)")
        return None
    with open(p) as fh:
        return list(csv.DictReader(fh))


def fig_per_subject():
    """B2: the spread is the story, not the centre."""
    rows = read("B2_loso_tangent_space.csv")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: float(r["acc"]))
    acc = np.array([float(r["acc"]) for r in rows])
    lo = np.array([float(r["ci_lo"]) for r in rows])
    hi = np.array([float(r["ci_hi"]) for r in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.errorbar(x, acc, yerr=[acc - lo, hi - acc], fmt="o", capsize=3, color="#333")
    ax.axhline(0.5, ls="--", c="crimson", label="chance (0.50)")
    ax.axhline(acc.mean(), ls="-", c="steelblue",
               label=f"mean {acc.mean():.3f} +/- {acc.std():.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{int(r['subject']):03d}" for r in rows], rotation=90, fontsize=7)
    ax.set_ylabel("LOSO accuracy")
    ax.set_title("Per-subject LOSO accuracy, imagery, tangent space\n"
                 "95% binomial CIs -- every CI crosses chance", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_per_subject_loso.png", dpi=150)
    print("  -> fig_per_subject_loso.png")


def fig_subject_id():
    """D1-D4: what identity survives alignment."""
    rows = read("D_subject_id.csv")
    if not rows:
        return
    order = [("logvar", "no_EA"), ("logvar", "EA"),
             ("psd_rel", "no_EA"), ("psd_rel", "EA"),
             ("psd_abs", "no_EA"), ("psd_abs", "EA")]
    lut = {(r["features"], r["alignment"]): r for r in rows}
    sel = [(f, a) for f, a in order if (f, a) in lut]
    acc = [float(lut[k]["acc"]) for k in sel]
    sd = [float(lut[k]["sd"]) for k in sel]
    chance = float(rows[0]["chance"])
    colors = ["#888" if a == "no_EA" else "#c0392b" for _, a in sel]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(sel))
    ax.bar(x, acc, yerr=sd, color=colors, capsize=4)
    ax.axhline(chance, ls="--", c="k",
               label=f"chance = 1/{rows[0]['n_subjects']} = {chance:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n{a}" for f, a in sel], fontsize=8)
    ax.set_ylabel("subject-ID accuracy (leave-one-run-out)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Who is this? Covariance identity dies under Euclidean Alignment;\n"
                 "spectral identity does not.", fontsize=10)
    for xi, a in zip(x, acc):
        ax.text(xi, a + 0.03, f"{a:.3f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_subject_id.png", dpi=150)
    print("  -> fig_subject_id.png")


def fig_capacity():
    """B4: the leakage gap only opens up when the model can memorise."""
    rows = read("B_splits.csv")
    if not rows:
        return
    rows = [r for r in rows if r["id"] == "B4"]
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    kf = [float(next(r["acc"] for r in rows if r["model"] == m and r["protocol"] == "random_5fold")) for m in models]
    ls = [float(next(r["acc"] for r in rows if r["model"] == m and r["protocol"] == "LOSO")) for m in models]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - 0.18, kf, 0.36, label="random 5-fold (wrong protocol)", color="#c0392b")
    ax.bar(x + 0.18, ls, 0.36, label="LOSO (honest protocol)", color="#2c7fb8")
    ax.axhline(0.5, ls="--", c="k", label="chance")
    for xi, (a, b) in enumerate(zip(kf, ls)):
        ax.text(xi, max(a, b) + 0.02, f"{(a-b)*100:+.0f} pp", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("accuracy")
    ax.set_title("Subject leakage inflates a score only if the model can exploit it",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_capacity_leakage.png", dpi=150)
    print("  -> fig_capacity_leakage.png")


def fig_permutation():
    rows = read("C1_permutation.csv")
    if not rows:
        return
    null = np.array([float(r["acc"]) for r in rows if r["perm"] != "observed"])
    obs = float(next(r["acc"] for r in rows if r["perm"] == "observed"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=30, color="#bbb", edgecolor="w")
    ax.axvline(np.percentile(null, 95), ls="--", c="k",
               label=f"null 95th pct {np.percentile(null,95):.3f}")
    ax.axvline(obs, c="crimson", lw=2, label=f"observed {obs:.3f}")
    ax.set_xlabel("mean LOSO accuracy under within-subject label shuffle")
    ax.set_ylabel("count")
    ax.set_title(f"Permutation null, {len(null)} shuffles", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_permutation.png", dpi=150)
    print("  -> fig_permutation.png")


def fig_learning_curve():
    rows = read("F_overfitting.csv")
    if not rows:
        return
    rows = sorted([r for r in rows if r["id"] == "F2"],
                  key=lambda r: int(r["n_train_subjects"]))
    if not rows:
        return
    n = [int(r["n_train_subjects"]) for r in rows]
    a = np.array([float(r["acc"]) for r in rows])
    s = np.array([float(r["sd"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n, a, "o-", color="#2c7fb8")
    ax.fill_between(n, a - s, a + s, alpha=0.2, color="#2c7fb8")
    ax.axhline(0.5, ls="--", c="crimson", label="chance")
    ax.set_xlabel("number of training subjects")
    ax.set_ylabel("accuracy on a held-out subject")
    ax.set_title("Does adding people help?", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_learning_curve.png", dpi=150)
    print("  -> fig_learning_curve.png")


def fig_ea_delta():
    rows = read("D5_ea_effect.csv")
    if not rows:
        return
    rows = [r for r in rows if r["pipeline"] == "tangent_space"]
    if not rows:
        return
    d = np.array([float(r["delta"]) for r in rows])
    order = np.argsort(d)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(len(d)), d[order] * 100,
           color=["#c0392b" if v < 0 else "#2c7fb8" for v in d[order]])
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels([f"S{int(rows[i]['subject']):03d}" for i in order],
                       rotation=90, fontsize=7)
    ax.set_ylabel("EA effect (pp)")
    ax.set_title(f"Euclidean Alignment per subject: mean {d.mean()*100:+.1f} pp, "
                 f"helps {(d>0).sum()}, hurts {(d<0).sum()}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{R}/fig_ea_per_subject.png", dpi=150)
    print("  -> fig_ea_per_subject.png")


def fig_csp_patterns(n_subjects=30):
    """Are the spatial filters over sensorimotor cortex, or somewhere else?"""
    import mne
    from src.data import load_dataset, good_subjects, IMAGINED_RUNS
    from src.models import csp_lda
    ds = load_dataset(good_subjects(n_subjects), IMAGINED_RUNS)
    csp = csp_lda(4).named_steps["csp"].fit(ds.X, ds.y)
    info = mne.create_info(ds.ch_names, 160.0, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"),
                     on_missing="ignore")
    fig = csp.plot_patterns(info, ch_type="eeg", units="a.u.", size=1.4,
                            show=False)
    fig.suptitle("CSP patterns fit on all training subjects (imagery, 8-30 Hz)",
                 fontsize=10)
    fig.savefig(f"{R}/fig_csp_patterns.png", dpi=150)
    print("  -> fig_csp_patterns.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--skip-csp", action="store_true")
    a = ap.parse_args()
    fig_per_subject()
    fig_subject_id()
    fig_capacity()
    fig_permutation()
    fig_learning_curve()
    fig_ea_delta()
    if not a.skip_csp:
        fig_csp_patterns(a.subjects)

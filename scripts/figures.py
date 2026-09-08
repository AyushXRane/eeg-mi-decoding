"""Figures for the three results that need one. Run the scripts first.

Every plot with a chance level draws it -- for the person-vs-task result chance
is 1/n_subjects, and a reader assuming 0.5 would badly misread the figure.
"""
import csv, os, sys
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


def fig_window_leakage():
    rows = read("R2_window_leakage.csv")
    if not rows:
        return
    models = list(dict.fromkeys(r["model"] for r in rows))
    splits = ["random_over_windows", "grouped_by_trial", "grouped_by_subject"]
    labels = ["random split\nover windows", "grouped\nby trial", "leave one\nsubject out"]
    fig, axes = plt.subplots(1, len(models), figsize=(9, 4.2), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        vals = [float(next(r["acc"] for r in rows
                           if r["model"] == m and r["split"] == sp)) for sp in splits]
        ax.bar(range(3), vals, color=["#c0392b", "#e08a4a", "#2c7fb8"])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        ax.axhline(0.5, ls="--", c="k", lw=0.9)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(m, fontsize=10); ax.set_ylim(0, 1.08)
    np.atleast_1d(axes)[0].set_ylabel("accuracy")
    fig.suptitle("Overlapping windows split at random leak the answer\n"
                 "dashed line = chance", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{R}/R2_window_leakage.png", dpi=150)
    print("  -> R2_window_leakage.png")


def fig_permutation():
    rows = read("R3_permutation.csv")
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
    ax.set_title(f"The number to beat is not 0.50 ({len(null)} shuffles)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{R}/R3_permutation.png", dpi=150)
    print("  -> R3_permutation.png")


def fig_person_vs_task():
    rows = read("R4_person_vs_task.csv")
    if not rows:
        return
    para = "imagined"
    sub = [r for r in rows if r["paradigm"] == para]
    order = [("logvar", "no_EA"), ("logvar", "EA"), ("psd_rel", "no_EA"),
             ("psd_rel", "EA"), ("psd_abs", "no_EA"), ("psd_abs", "EA")]
    lut = {(r["features"], r["alignment"]): r for r in sub}
    sel = [k for k in order if k in lut]
    acc = [float(lut[k]["acc"]) for k in sel]
    sd = [float(lut[k]["sd"]) for k in sel]
    chance = float(sub[0]["chance"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(sel)), acc, yerr=sd, capsize=4,
           color=["#888" if a == "no_EA" else "#c0392b" for _, a in sel])
    ax.axhline(chance, ls="--", c="k",
               label=f"chance = 1/{sub[0]['n_subjects']} = {chance:.4f}")
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels([f"{f}\n{a}" for f, a in sel], fontsize=8)
    ax.set_ylabel("subject-ID accuracy (leave-one-run-out)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Who is this? Alignment kills the covariance fingerprint;\n"
                 "the spectral one survives untouched.", fontsize=10)
    for i, a in enumerate(acc):
        ax.text(i, a + 0.03, f"{a:.3f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{R}/R4_person_vs_task.png", dpi=150)
    print("  -> R4_person_vs_task.png")


if __name__ == "__main__":
    fig_window_leakage()
    fig_permutation()
    fig_person_vs_task()

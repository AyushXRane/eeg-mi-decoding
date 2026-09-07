"""Full experiment sweep. Run one block at a time: python scripts/grid.py A B C

Blocks map onto SPEC section 7:
  A  sanity, ERD, within-subject ceiling, binomial CIs
  B  measurement artifact: wrong protocol vs honest protocol, and B4 capacity
  C  baselines beyond chance: permutation, channel ablation, EMG control
  D  person vs task: the 2x2x2 subject-ID probes plus EA's effect on the task
  E  transfer between executed and imagined
  F  overfitting: fit gap, learning curve, CSP capacity
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from src.data import load_dataset, good_subjects, IMAGINED_RUNS, EXECUTED_RUNS
from src.align import align_by_group
from src.models import csp_lda, tangent_space, CAPACITY_LADDER
from src.evaluate import (loso, loso_cross, random_kfold, within_subject,
                          permutation_null_fast, summarise, binomial_ci)
from src.probes import (subject_id_probe, logvar_features, psd_features,
                        erd_check, pick_channels, MOTOR, NON_MOTOR)
from src.report import write_rows, line

R = "results"


def pipes():
    return {"csp_lda": csp_lda(4), "tangent_space": tangent_space()}


# ---------------------------------------------------------------- block A
def block_A(ds, ds_exec, args):
    print("\n== A. sanity, ERD, within-subject ceiling ==")

    # A2: does the physiological effect exist, and does it exist per subject?
    rows = []
    for name, d in (("imagined", ds), ("executed", ds_exec)):
        for s in np.unique(d.groups):
            m = d.groups == s
            e = erd_check(d.X[m], d.y[m], d.ch_names)
            rows.append({"paradigm": name, "subject": int(s), **e})
    write_rows(f"{R}/A2_erd.csv", rows)
    for name in ("imagined", "executed"):
        v = np.array([r["lat_index"] for r in rows if r["paradigm"] == name])
        t, p = stats.ttest_1samp(v, 0.0)
        print(f"A2 {name:9s} lateralisation index {v.mean():+.3f} +/- {v.std():.3f}  "
              f"t={t:+.2f} p={p:.3f}  positive in {(v > 0).sum()}/{len(v)} subjects")

    # A3/A4: the optimistic ceiling, with a CI on every subject.
    out = []
    for name, p in pipes().items():
        r = within_subject(ds.X, ds.y, ds.groups, p)
        for x in r:
            out.append({"pipeline": name, **x})
        print("A3 " + line(r, f"within-subject imagery [{name}]"))
        w = np.mean([x["ci_hi"] - x["ci_lo"] for x in r])
        print(f"A4 mean 95% CI width per subject: {w*100:.1f} pp")
    write_rows(f"{R}/A3_within_subject.csv", out)


# ---------------------------------------------------------------- block B
def block_B(ds, ds_exec, args):
    print("\n== B. is the number an artifact of the split? ==")
    out = []
    for name, p in pipes().items():
        f = random_kfold(ds.X, ds.y, p)
        r = loso(ds.X, ds.y, ds.groups, p)
        s = summarise(r, name)
        gap = float(np.mean(f)) - s["mean"]
        print(f"B1/B2/B3 {name:14s} kfold {np.mean(f):.3f}  LOSO {s['mean']:.3f} "
              f"+/- {s['sd']:.3f}  gap {gap*100:+.1f} pp")
        out.append({"id": "B1", "model": name, "acc": float(np.mean(f)),
                    "sd": float(np.std(f)), "protocol": "random_5fold"})
        out.append({"id": "B2", "model": name, "acc": s["mean"], "sd": s["sd"],
                    "protocol": "LOSO"})
        out.append({"id": "B3", "model": name, "acc": gap, "sd": float("nan"),
                    "protocol": "gap"})
        write_rows(f"{R}/B2_loso_{name}.csv", r)

    # B4: the gap as a function of model capacity. B3 came back at zero for
    # linear models; this tests whether that is because the split is safe or
    # because a linear model cannot exploit it.
    print("  B4 leakage gap vs model capacity:")
    for name, fn in CAPACITY_LADDER.items():
        p = fn()
        f = random_kfold(ds.X, ds.y, p)
        r = loso(ds.X, ds.y, ds.groups, p)
        s = summarise(r, name)
        gap = float(np.mean(f)) - s["mean"]
        print(f"     {name:10s} kfold {np.mean(f):.3f}  LOSO {s['mean']:.3f}  "
              f"gap {gap*100:+.1f} pp")
        out.append({"id": "B4", "model": name, "acc": float(np.mean(f)),
                    "sd": float(np.std(f)), "protocol": "random_5fold"})
        out.append({"id": "B4", "model": name, "acc": s["mean"], "sd": s["sd"],
                    "protocol": "LOSO"})
        out.append({"id": "B4", "model": name, "acc": gap, "sd": float("nan"),
                    "protocol": "gap"})
    write_rows(f"{R}/B_splits.csv", out)


# ---------------------------------------------------------------- block C
def block_C(ds, ds_exec, args):
    print("\n== C. baselines beyond chance ==")
    pipe = tangent_space()

    # C1: a null built by rerunning the same LOSO on within-subject shuffles.
    null, real = permutation_null_fast(ds.X, ds.y, ds.groups, n_perm=args.n_perm)
    p_val = float((null >= real).mean() + 1 / (len(null) + 1))
    print(f"C1 permutation null: mean {null.mean():.3f}, 95th pct {np.percentile(null,95):.3f}, "
          f"max {null.max():.3f} | observed {real:.3f}, p={p_val:.4f} ({len(null)} shuffles)")
    write_rows(f"{R}/C1_permutation.csv",
               [{"perm": i, "acc": float(v)} for i, v in enumerate(null)]
               + [{"perm": "observed", "acc": real}])

    # C2: is the signal where sensorimotor physiology says it should be?
    out = []
    sets = {"all_64": list(range(len(ds.ch_names))),
            "motor_only": pick_channels(ds.ch_names, MOTOR),
            "non_motor_only": pick_channels(ds.ch_names, NON_MOTOR)}
    for name, idx in sets.items():
        r = loso(ds.X[:, idx, :], ds.y, ds.groups, tangent_space())
        s = summarise(r, name)
        print(f"C2 {name:16s} ({len(idx):2d} ch) {s['mean']:.3f} +/- {s['sd']:.3f}")
        out.append({"id": "C2", "channels": name, "n_ch": len(idx),
                    "acc": s["mean"], "sd": s["sd"]})
    write_rows(f"{R}/C2_channels.csv", out)

    # C3: EMG control. Muscle activity is broadband and sits closer to the
    # electrodes than cortex does. If executed-run accuracy survives in a band
    # where there is no ERD to find, part of execution's "stronger signal" is
    # muscle rather than brain.
    out = []
    subs = good_subjects(args.subjects)
    for para, runs in (("imagined", IMAGINED_RUNS), ("executed", EXECUTED_RUNS)):
        for band in ((8.0, 30.0), (30.0, 70.0)):
            d = load_dataset(subs, runs, l_freq=band[0], h_freq=band[1])
            s = summarise(loso(d.X, d.y, d.groups, tangent_space()))
            print(f"C3 {para:9s} {band[0]:.0f}-{band[1]:.0f} Hz  {s['mean']:.3f} +/- {s['sd']:.3f}")
            out.append({"id": "C3", "paradigm": para, "band": f"{band[0]:.0f}-{band[1]:.0f}",
                        "acc": s["mean"], "sd": s["sd"]})
    write_rows(f"{R}/C3_emg_band.csv", out)


# ---------------------------------------------------------------- block D
def block_D(ds, ds_exec, args):
    print("\n== D. person vs task ==")
    Xa = align_by_group(ds.X, ds.groups)
    feats = {"logvar": logvar_features,
             "psd_rel": lambda X: psd_features(X, relative=True),
             "psd_abs": lambda X: psd_features(X, relative=False)}

    out = []
    for fname, fn in feats.items():
        for tag, Xu in (("no_EA", ds.X), ("EA", Xa)):
            p = subject_id_probe(Xu, ds.groups, ds.run, fn)
            print(f"D subject-ID {fname:8s} {tag:5s}: {p['acc']:.3f} +/- {p['sd']:.3f} "
                  f"(chance {p['chance']:.3f})")
            out.append({"features": fname, "alignment": tag, "acc": p["acc"],
                        "sd": p["sd"], "chance": p["chance"],
                        "n_subjects": p["n_subjects"]})
    write_rows(f"{R}/D_subject_id.csv", out)

    # D5/D6: what EA actually buys on the task, per subject and on average.
    out = []
    for name, p in pipes().items():
        base = {r["subject"]: r["acc"] for r in loso(ds.X, ds.y, ds.groups, p)}
        aligned = {r["subject"]: r["acc"] for r in loso(Xa, ds.y, ds.groups, p)}
        d = np.array([aligned[s] - base[s] for s in base])
        print(f"D5 {name:14s} EA delta {d.mean()*100:+.1f} pp +/- {d.std()*100:.1f}  "
              f"helped {(d>0).sum()}/{len(d)}, hurt {(d<0).sum()}/{len(d)}")
        for s in base:
            out.append({"pipeline": name, "subject": s, "acc_no_ea": base[s],
                        "acc_ea": aligned[s], "delta": aligned[s] - base[s]})
    write_rows(f"{R}/D5_ea_effect.csv", out)


# ---------------------------------------------------------------- block E
def block_E(ds, ds_exec, args):
    print("\n== E. transfer between execution and imagery ==")
    p = tangent_space()
    out = []

    e1 = loso_cross(ds_exec.X, ds_exec.y, ds_exec.groups, ds.X, ds.y, ds.groups, p)
    print("E1 " + line(e1, "train executed -> test imagined"))
    e2 = loso_cross(ds.X, ds.y, ds.groups, ds_exec.X, ds_exec.y, ds_exec.groups, p)
    print("E2 " + line(e2, "train imagined -> test executed"))

    Xp = np.concatenate([ds.X, ds_exec.X])
    yp = np.concatenate([ds.y, ds_exec.y])
    gp = np.concatenate([ds.groups, ds_exec.groups])
    e3 = loso_cross(Xp, yp, gp, ds.X, ds.y, ds.groups, p)
    print("E3 " + line(e3, "train pooled exec+imag -> test imagined"))

    for tag, rows in (("E1_exec_to_imag", e1), ("E2_imag_to_exec", e2),
                      ("E3_pooled_to_imag", e3)):
        for r in rows:
            out.append({"id": tag, **r})
    write_rows(f"{R}/E_transfer.csv", out)


# ---------------------------------------------------------------- block F
def block_F(ds, ds_exec, args):
    print("\n== F. where does it overfit ==")
    out = []

    # F1: train vs test accuracy on the same folds.
    for name, p in pipes().items():
        r = loso(ds.X, ds.y, ds.groups, p, return_train=True)
        tr = np.mean([x["train_acc"] for x in r])
        te = np.mean([x["acc"] for x in r])
        print(f"F1 {name:14s} train {tr:.3f}  test {te:.3f}  gap {(tr-te)*100:.1f} pp")
        for x in r:
            out.append({"id": "F1", "pipeline": name, **x})

    # F2: does adding people help, and where does it stop helping?
    rng = np.random.default_rng(0)
    subs = np.unique(ds.groups)
    for n in [3, 5, 8, 12, 16, 20, 25, len(subs) - 1]:
        if n >= len(subs):
            continue
        accs = []
        for rep in range(3):
            for held in rng.choice(subs, size=min(6, len(subs)), replace=False):
                pool = [s for s in subs if s != held]
                tr_s = rng.choice(pool, size=n, replace=False)
                m_tr = np.isin(ds.groups, tr_s)
                m_te = ds.groups == held
                from sklearn.base import clone
                mdl = clone(tangent_space()).fit(ds.X[m_tr], ds.y[m_tr])
                accs.append(float(mdl.score(ds.X[m_te], ds.y[m_te])))
        print(f"F2 n_train_subjects={n:3d}  {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
        out.append({"id": "F2", "n_train_subjects": n, "acc": float(np.mean(accs)),
                    "sd": float(np.std(accs))})

    # F3: more spatial filters is more capacity, not more signal.
    for k in (2, 4, 6, 8):
        s = summarise(loso(ds.X, ds.y, ds.groups, csp_lda(k)))
        print(f"F3 CSP components={k}  {s['mean']:.3f} +/- {s['sd']:.3f}")
        out.append({"id": "F3", "n_components": k, "acc": s["mean"], "sd": s["sd"]})

    write_rows(f"{R}/F_overfitting.csv", out)


BLOCKS = {"A": block_A, "B": block_B, "C": block_C,
          "D": block_D, "E": block_E, "F": block_F}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("blocks", nargs="+", choices=list(BLOCKS) + ["all"])
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--n-perm", type=int, default=200)
    a = ap.parse_args()

    subs = good_subjects(a.subjects)
    t0 = time.time()
    ds = load_dataset(subs, IMAGINED_RUNS)
    ds_exec = load_dataset(subs, EXECUTED_RUNS)
    print(f"imagined {ds.X.shape}  executed {ds_exec.X.shape}  "
          f"{len(np.unique(ds.groups))} subjects  ({time.time()-t0:.0f}s)")

    names = list(BLOCKS) if "all" in a.blocks else a.blocks
    for b in names:
        t = time.time()
        BLOCKS[b](ds, ds_exec, a)
        print(f"[block {b} done in {time.time()-t:.0f}s]")

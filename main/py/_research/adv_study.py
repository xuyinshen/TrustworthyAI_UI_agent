"""Adversarial attack + adversarial training study: cost (timing) and effect (robustness gain).

4 classification profiles x model zoo. FGSM/PGD/DeepFool generation cost and ASR, then
FGSM-augmented retraining (the same mechanics as RB.adversarial_training) evaluated under
both FGSM and the stronger PGD. Writes CSV incrementally.
"""
import os, sys, time, warnings, csv, traceback
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np
from trustoran.data import build_dataset, train_test_split_dataset
from trustoran.models import make_model, available_models
from trustoran.trust import robustness as RB

SEED = 42
DSETS = ["cci_a1sd", "cci_a1sd_binary", "vran_computing", "ddos_pcap"]
EPS = 0.3
NTRAIN, NTEST, NDF = 6000, 800, 120
OUT = os.path.join(HERE, "adv_study.csv")

COLS = ["dataset", "n_train", "n_test", "d", "model", "fit_s", "clean_acc",
        "fgsm_s", "fgsm_rob", "fgsm_asr", "pgd_s", "pgd_rob", "pgd_asr",
        "df_s", "df_flip", "df_l2", "at_s", "at_clean", "at_fgsm_rob", "at_pgd_rob",
        "gain_fgsm_pp", "gain_pgd_pp", "clean_gain_pp", "err"]

done = set()
if os.path.exists(OUT):
    with open(OUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["dataset"], row["model"]))

fh = open(OUT, "a", newline="", encoding="utf-8")
w = csv.DictWriter(fh, fieldnames=COLS)
if not done:
    w.writeheader()


def num(v):
    try:
        return "%.4f" % float(v)
    except Exception:
        return ""


for dk in DSETS:
    ds = build_dataset(dk, max_rows=40_000, seed=SEED)
    sp = train_test_split_dataset(ds, test_size=0.3, seed=SEED)
    rng = np.random.default_rng(SEED)

    def _cap(X, y, n):
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        return X[idx], y[idx]

    Xtr, ytr = _cap(sp.X_train, sp.y_train, NTRAIN)
    Xte, yte = _cap(sp.X_test, sp.y_test, NTEST)
    Xdf, ydf = Xte[:NDF], yte[:NDF]
    names = available_models(sp.train.task)
    print("== %s  train %s test %s d=%d  models=%d ==" %
          (dk, Xtr.shape, Xte.shape, Xtr.shape[1], len(names)), flush=True)

    for name in names:
        if (dk, name) in done:
            continue
        rec = {"dataset": dk, "n_train": len(Xtr), "n_test": len(Xte),
               "d": Xtr.shape[1], "model": name}
        errs = []
        m = None
        try:
            t = time.perf_counter()
            m = make_model(name, sp.train.task, sp.feature_names, sp.train.class_names)
            m.fit(Xtr, ytr)
            rec["fit_s"] = "%.2f" % (time.perf_counter() - t)
            rec["clean_acc"] = num(float((np.asarray(m.predict(Xte)) == yte).mean()))
        except Exception as e:
            errs.append("fit:%s" % type(e).__name__)
            rec["model"] = name + "!ERR"
            w.writerow({c: rec.get(c, "") for c in COLS})
            fh.flush()
            print("  %-16s FIT FAILED %s" % (name, e), flush=True)
            continue

        for key, atk, eps_, X_, y_ in [("fgsm", "fgsm", EPS, Xte, yte),
                                       ("pgd", "pgd", EPS, Xte, yte),
                                       ("deepfool", "deepfool", 0.0, Xdf, ydf)]:
            try:
                t = time.perf_counter()
                r = RB.evaluate_attack(m, X_, y_, atk, eps_, class_names=sp.train.class_names)
                el = time.perf_counter() - t
                if key == "deepfool":
                    rec["df_s"] = "%.2f" % el
                    rec["df_flip"] = num(r.attack_success_rate)
                    rec["df_l2"] = num(r.mean_perturbation)
                else:
                    rec[key + "_s"] = "%.2f" % el
                    rec[key + "_rob"] = num(r.robust_accuracy)
                    rec[key + "_asr"] = num(r.attack_success_rate)
            except Exception as e:
                errs.append("%s:%s" % (key, type(e).__name__))

        try:
            t = time.perf_counter()
            Xa = np.vstack([Xtr, RB.fgsm(m, Xtr, ytr, EPS)])
            ya = np.concatenate([ytr, ytr])
            defended = m.clone_unfitted().fit(Xa, ya)
            rec["at_s"] = "%.2f" % (time.perf_counter() - t)
            ac = float((np.asarray(defended.predict(Xte)) == yte).mean())
            d1 = RB.evaluate_attack(defended, Xte, yte, "fgsm", EPS, class_names=sp.train.class_names)
            d2 = RB.evaluate_attack(defended, Xte, yte, "pgd", EPS, class_names=sp.train.class_names)
            rec["at_clean"] = num(ac)
            rec["at_fgsm_rob"] = num(d1.robust_accuracy)
            rec["at_pgd_rob"] = num(d2.robust_accuracy)
            if rec["fgsm_rob"]:
                rec["gain_fgsm_pp"] = num((d1.robust_accuracy - float(rec["fgsm_rob"])) * 100)
            if rec["pgd_rob"]:
                rec["gain_pgd_pp"] = num((d2.robust_accuracy - float(rec["pgd_rob"])) * 100)
            if rec["clean_acc"]:
                rec["clean_gain_pp"] = num((ac - float(rec["clean_acc"])) * 100)
            print("  %-16s clean=%s fgsm=%s pgd=%s df=%.2f | AT: clean=%s fgsm=%s pgd=%s "
                  "gain %s/%s pp  at_s=%s  fgsm_s=%s pgd_s=%s %s" %
                  (name, rec["clean_acc"], rec["fgsm_rob"], rec["pgd_rob"],
                   float(rec.get("df_flip") or 0), rec["at_clean"], rec["at_fgsm_rob"],
                   rec["at_pgd_rob"], rec["gain_fgsm_pp"], rec["gain_pgd_pp"], rec["at_s"],
                   rec.get("fgsm_s"), rec.get("pgd_s"), ("| " + ";".join(errs)) if errs else ""),
                  flush=True)
        except Exception as e:
            errs.append("at:%s" % type(e).__name__)
            print("  %-16s AT FAILED %s: %s" % (name, type(e).__name__, e), flush=True)

        rec["err"] = ";".join(errs)
        w.writerow({c: rec.get(c, "") for c in COLS})
        fh.flush()

fh.close()
print("DONE ->", OUT, flush=True)

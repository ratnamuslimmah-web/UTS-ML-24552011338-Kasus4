"""latih.py - Training, evaluasi, dan penyimpanan model Kasus 4."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (make_scorer, average_precision_score, classification_report, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)

import fitur as F

SEED = 42
TARGET_RECALL = 0.80
BASE = Path(__file__).parent if "__file__" in globals() else Path(".")
DATA = BASE / "dataset" / "kasus04_birthwt.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    df = F.muat_data(DATA)
    print("Audit:", F.audit_data(df))

    # 1) SPLIT DULU (sebelum imputasi/scaling/encoding)
    X = df[F.KOLOM_ASLI]
    y = df[F.TARGET].astype(int)          # 'bwt' tidak masuk X (leakage)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train {X_tr.shape}, Test {X_te.shape}, proporsi low train={y_tr.mean():.2f} test={y_te.mean():.2f}")

    # 2) Kandidat model
    kandidat = {
        "Dummy (baseline)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_depth=5,
                                                class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=-1),
    }

    # 3) Pemilihan model: cross-validation di data TRAIN saja
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    skor = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall",
            "precision": make_scorer(precision_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0)}
    baris = []
    for nama, m in kandidat.items():
        r = cross_validate(F.buat_pipeline(m), X_tr, y_tr, cv=cv, scoring=skor, n_jobs=-1)
        baris.append({"model": nama, **{k: float(np.mean(r[f"test_{k}"])) for k in skor},
                      **{f"{k}_std": float(np.std(r[f"test_{k}"])) for k in ("pr_auc",)}})
    tabel_cv = pd.DataFrame(baris).set_index("model")
    print("\nHasil Cross-Validation (train):\n", tabel_cv.round(3))

    layak = tabel_cv.drop(index="Dummy (baseline)")
    nama_terbaik = layak["pr_auc"].idxmax()          # ranking bebas threshold
    print("\nModel terpilih:", nama_terbaik)

    # 4) Threshold dipilih dari prediksi out-of-fold di TRAIN (bukan dari test).
    #    Aturan: recall minimal 0.80 (FN lebih mahal), lalu ambil threshold TERTINGGI
    #    yang masih memenuhi supaya false positive sesedikit mungkin.
    pipa = F.buat_pipeline(kandidat[nama_terbaik])
    oof = cross_val_predict(pipa, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rec = np.array([recall_score(y_tr, oof >= t) for t in grid])
    ok = np.where(rec >= TARGET_RECALL)[0]
    if len(ok):
        thr = float(grid[ok[-1]])
    else:  # cadangan: maksimalkan F2
        thr = float(grid[int(np.argmax([fbeta_score(y_tr, oof >= t, beta=2) for t in grid]))])
    print(f"Threshold terpilih = {thr} (recall OOF = {recall_score(y_tr, oof >= thr):.2f})")

    # 5) Evaluasi data test SATU kali
    pipa.fit(X_tr, y_tr)
    proba = pipa.predict_proba(X_te)[:, 1]
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    metrik = {
        "recall": recall_score(y_te, pred), "precision": precision_score(y_te, pred, zero_division=0),
        "f1": f1_score(y_te, pred), "f2": fbeta_score(y_te, pred, beta=2),
        "roc_auc": roc_auc_score(y_te, proba), "pr_auc": average_precision_score(y_te, proba),
        "specificity": tn / (tn + fp), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    metrik = {k: float(v) for k, v in metrik.items()}
    print("\nMetrik TEST:", {k: round(v, 3) for k, v in metrik.items()})
    print(classification_report(y_te, pred, target_names=["Normal (0)", "BBLR (1)"], zero_division=0))

    # 6) Interpretasi fitur
    nama_fitur = pipa.named_steps["prep"].get_feature_names_out()
    est = pipa.named_steps["model"]
    imp = est.coef_[0] if hasattr(est, "coef_") else est.feature_importances_
    penting = dict(sorted(zip(map(str, nama_fitur), map(float, imp)), key=lambda kv: -abs(kv[1])))

    # 7) Simpan pipeline lengkap + metadata
    paket = {
        "pipeline": pipa, "threshold": thr, "target_recall": TARGET_RECALL, "model_name": nama_terbaik,
        "metrics_test": metrik, "cv_table": tabel_cv.round(4).to_dict("index"),
        "feature_importance": penting, "input_columns": F.KOLOM_ASLI,
        "race_label": F.RACE_LABEL, "kolom_dibuang": F.KOLOM_DIBUANG,
        "target_meaning": {0: "Berat lahir normal (>= 2500 g)", 1: "Berat lahir rendah (< 2500 g)"},
        "sklearn_version": sklearn.__version__, "seed": SEED,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    }
    joblib.dump(paket, MODEL_DIR / "model_bblr.joblib")
    (MODEL_DIR / "metrik.json").write_text(json.dumps(
        {"model": nama_terbaik, "threshold": thr, "test": metrik, "cv": paket["cv_table"],
         "fitur_penting": penting}, indent=2))
    print("\nTersimpan di models/model_bblr.joblib")


if __name__ == "__main__":
    main()

"""fitur.py - Feature engineering Kasus 4 (birthwt, target = low)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "low"
KOLOM_ASLI = ["age", "lwt", "race", "smoke", "ptl", "ht", "ui", "ftv"]

# Kolom yang DIBUANG beserta alasannya
KOLOM_DIBUANG = {
    "bwt": "Berat lahir bayi (gram). Baru diketahui setelah bayi lahir dan low = (bwt < 2500). "
           "Memakainya adalah data leakage (target bocor).",
    "rownames/Unnamed: 0/id": "Hanya nomor baris/ID, tidak punya makna prediktif.",
}

RACE_LABEL = {1: "putih", 2: "hitam", 3: "lainnya"}

NUMERIK = ["age", "lwt", "ptl", "ftv"]
BINER = ["smoke", "ht", "ui", "ptl_pernah", "ftv_pernah",
         "usia_berisiko", "lwt_rendah", "jumlah_risiko"]
KATEGORIK = ["race_label"]


def muat_data(path):
    """Baca CSV, buang kolom ID/indeks, dan buang duplikat persis."""
    df = pd.read_csv(path)
    for c in list(df.columns):
        if c.lower() in ("rownames", "unnamed: 0", "id", "index", ""):
            df = df.drop(columns=c)
    return df.drop_duplicates().reset_index(drop=True)


def audit_data(df):
    """Ringkasan audit: duplikat, missing, tipe data, nilai tidak wajar."""
    return {
        "bentuk": df.shape,
        "duplikat": int(df.duplicated().sum()),
        "missing": df.isna().sum().to_dict(),
        "tipe": df.dtypes.astype(str).to_dict(),
        "tidak_wajar": {
            "age di luar 12-50": int((~df["age"].between(12, 50)).sum()),
            "lwt di luar 70-400 pon": int((~df["lwt"].between(70, 400)).sum()),
            "biner bukan 0/1": int(sum((~df[c].isin([0, 1])).sum() for c in ["smoke", "ht", "ui"])),
            "race di luar 1-3": int((~df["race"].isin([1, 2, 3])).sum()),
        },
    }


def rekayasa_fitur(X):
    """Fitur turunan TANPA statistik data (aturan domain tetap) -> aman dari leakage."""
    X = X.copy()
    # race berupa kode nominal (bukan urutan) -> jadikan kategorik
    X["race_label"] = X["race"].map(RACE_LABEL).fillna("lainnya")
    X["ptl_pernah"] = (X["ptl"] > 0).astype(int)           # pernah lahir prematur
    X["ftv_pernah"] = (X["ftv"] > 0).astype(int)           # pernah periksa trimester 1
    X["usia_berisiko"] = ((X["age"] < 18) | (X["age"] >= 35)).astype(int)  # remaja / usia lanjut
    X["lwt_rendah"] = (X["lwt"] < 100).astype(int)         # BB ibu < 100 pon (~45 kg)
    X["jumlah_risiko"] = X["smoke"] + X["ht"] + X["ui"] + X["ptl_pernah"]  # akumulasi risiko
    return X[NUMERIK + BINER + KATEGORIK]


def buat_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    bin_ = Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])
    kat = Pipeline([("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIK), ("bin", bin_, BINER), ("kat", kat, KATEGORIK)])


def buat_pipeline(model):
    return Pipeline([
        ("fitur", FunctionTransformer(rekayasa_fitur, validate=False)),
        ("prep", buat_preprocessor()),
        ("model", model),
    ])

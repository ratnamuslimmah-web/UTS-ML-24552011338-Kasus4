"""app.py - Aplikasi Streamlit prediksi Berat Lahir Rendah (Kasus 4)."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import fitur  # noqa: F401  (diperlukan agar pipeline bisa di-load)

MODEL_PATH = Path(__file__).parent / "models" / "model_bblr.joblib"
st.set_page_config(page_title="Prediksi Berat Lahir Rendah", page_icon="👶")


@st.cache_resource
def muat_model():
    return joblib.load(MODEL_PATH)


st.title("👶 Prediksi Bayi Berat Lahir Rendah (BBLR)")
st.caption("Alat bantu skrining berbasis data 189 kelahiran. Bukan pengganti diagnosis dokter.")

if not MODEL_PATH.exists():
    st.error("Model belum ada. Jalankan `python latih.py` terlebih dahulu.")
    st.stop()

paket = muat_model()
tab1, tab2 = st.tabs(["🔮 Prediksi", "📊 Evaluasi Model"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Usia ibu (tahun)", 12, 50, 25)
        lwt = st.number_input("Berat badan ibu saat haid terakhir (pon)", 70, 400, 130)
        race_nama = st.selectbox("Ras ibu", list(paket["race_label"].values()))
        ptl = st.number_input("Jumlah kelahiran prematur sebelumnya", 0, 10, 0)
    with c2:
        smoke = st.selectbox("Merokok saat hamil?", ["Tidak", "Ya"])
        ht = st.selectbox("Riwayat hipertensi?", ["Tidak", "Ya"])
        ui = st.selectbox("Iritabilitas uterus?", ["Tidak", "Ya"])
        ftv = st.number_input("Kunjungan dokter trimester 1", 0, 10, 0)

    if st.button("Prediksi", type="primary"):
        kode_race = {v: k for k, v in paket["race_label"].items()}[race_nama]
        raw = pd.DataFrame([{
            "age": age, "lwt": lwt, "race": kode_race, "smoke": int(smoke == "Ya"),
            "ptl": ptl, "ht": int(ht == "Ya"), "ui": int(ui == "Ya"), "ftv": ftv,
        }])[paket["input_columns"]]
        p = float(paket["pipeline"].predict_proba(raw)[0, 1])
        thr = paket["threshold"]
        st.metric("Probabilitas BBLR", f"{p:.1%}", help=f"Threshold keputusan = {thr:.2f}")
        st.progress(min(max(p, 0.0), 1.0))
        if p >= thr:
            st.warning("⚠️ **Berisiko BBLR** - disarankan pemantauan/konsultasi lebih lanjut.")
        else:
            st.success("✅ **Risiko rendah** untuk BBLR.")
        st.write(f"Probabilitas Normal: **{1 - p:.1%}**")
        with st.expander("Data mentah yang dikirim ke pipeline"):
            st.dataframe(raw)

with tab2:
    st.subheader(f"Model terpilih: {paket['model_name']}")
    m = paket["metrics_test"]
    a, b, c, d = st.columns(4)
    a.metric("Recall", f"{m['recall']:.2f}")
    b.metric("Precision", f"{m['precision']:.2f}")
    c.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
    d.metric("PR-AUC", f"{m['pr_auc']:.2f}")
    st.write(f"Threshold: **{paket['threshold']:.2f}** | Data latih: {paket['n_train']} | Data uji: {paket['n_test']}")
    st.write("**Confusion matrix (data uji)**")
    st.dataframe(pd.DataFrame([[m["TN"], m["FP"]], [m["FN"], m["TP"]]],
                              index=["Aktual Normal", "Aktual BBLR"],
                              columns=["Prediksi Normal", "Prediksi BBLR"]).astype(int))
    st.write("**Perbandingan model (cross-validation, data latih)**")
    st.dataframe(pd.DataFrame(paket["cv_table"]).T.round(3))
    st.write("**Pengaruh fitur** (koefisien LR / importance RF)")
    st.bar_chart(pd.Series(paket["feature_importance"]).head(10))
    with st.expander("Kolom yang dibuang"):
        for k, v in paket["kolom_dibuang"].items():
            st.write(f"- **{k}**: {v}")

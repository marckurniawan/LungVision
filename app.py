import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LungVision: Deteksi Kanker Paru-Paru",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #e0e0e0 !important; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f2d4a 100%);
        border: 1px solid #2a5a8f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4fc3f7;
        margin: 8px 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #90caf9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-badge {
        display: inline-block;
        background: #0d47a1;
        color: #64b5f6 !important;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin-top: 6px;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1565c0 0%, #0d47a1 100%);
        color: white !important;
        padding: 10px 20px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Result boxes */
    .result-positive {
        background: linear-gradient(135deg, #b71c1c 0%, #7f0000 100%);
        border: 2px solid #ef5350;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        color: white;
    }
    .result-negative {
        background: linear-gradient(135deg, #1b5e20 0%, #003300 100%);
        border: 2px solid #66bb6a;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        color: white;
    }
    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .result-sub {
        font-size: 1rem;
        opacity: 0.9;
    }

    /* Probability bar */
    .prob-bar-container {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #2a5a8f;
    }

    /* Info box */
    .info-box {
        background: #0d2137;
        border-left: 4px solid #2196f3;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #b0bec5;
    }

    /* Tab styling override */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0d1b2a;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        color: #90caf9;
    }
    .stTabs [aria-selected="true"] {
        background: #1565c0 !important;
        color: white !important;
    }

    /* Divider */
    hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join("models", "logistic_regression.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model()

# ─────────────────────────────────────────────
# LOAD METRICS CSV
# ─────────────────────────────────────────────
@st.cache_data
def load_metrics():
    base_path = os.path.join("results", "baseline_metrics.csv")
    if os.path.exists(base_path):
        df = pd.read_csv(base_path)
        # Deduplicate — CSV has repeated rows
        df = df.drop_duplicates(subset=["Model"])
        return df
    # Fallback hardcoded from known values
    return pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.61, 0.495, 0.515],
        "Precision": [0.6158, 0.5814, 0.5887],
        "Recall": [0.959, 0.6148, 0.6803],
        "F1 Score": [0.75, 0.5976, 0.6312],
        "ROC AUC": [0.4751, 0.4164, 0.4219],
    })

metrics_df = load_metrics()

# Best model row
lr_metrics = metrics_df[metrics_df["Model"] == "Logistic Regression"].iloc[0]

# ─────────────────────────────────────────────
# HELPER: Load image safely
# ─────────────────────────────────────────────
def load_img(path):
    if os.path.exists(path):
        return Image.open(path)
    return None

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 LungVision")
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["🏠 Dashboard", "🔬 Prediksi Pasien", "📊 Evaluasi Model", "ℹ️ Informasi"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#78909c; line-height:1.6'>
    <b style='color:#90caf9'>Model Terbaik</b><br>
    Logistic Regression (Baseline)<br><br>
    <b style='color:#90caf9'>Dataset</b><br>
    1000 pasien · 25 fitur klinis<br><br>
    <b style='color:#90caf9'>Target</b><br>
    Deteksi risiko kanker paru
    </div>
    """, unsafe_allow_html=True)



if page == "🏠 Dashboard":
    st.markdown("# 🫁 LungVision — AI-Powered Lung Cancer Detection")
    st.markdown("**Data-Driven Care for Better Outcomes**")
    st.markdown(
        "Sistem prediksi berbasis **Logistic Regression** yang dilatih pada data klinis "
        "1000 pasien dengan 25 fitur kesehatan."
    )
    st.markdown("---")

    # ── KPI Cards ──
    st.markdown('<div class="section-header">📈 Performa Model Terbaik — Logistic Regression</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Accuracy",  f"{lr_metrics['Accuracy']*100:.1f}%",  "Ketepatan keseluruhan"),
        (c2, "Precision", f"{lr_metrics['Precision']*100:.1f}%", "Presisi prediksi positif"),
        (c3, "Recall",    f"{lr_metrics['Recall']*100:.1f}%",    "Sensitivitas deteksi"),
        (c4, "F1 Score",  f"{lr_metrics['F1 Score']*100:.1f}%",  "Harmonic mean P&R"),
        (c5, "ROC AUC",   f"{lr_metrics['ROC AUC']:.4f}",        "Kemampuan diskriminasi"),
    ]
    for col, label, val, sub in kpis:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-badge">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Highlight: Recall 95.9% ──
    st.info(
        "💡 **Recall 95.9%** — Model sangat sensitif dalam mendeteksi pasien positif kanker paru. "
        "Hanya ~4% kasus positif yang terlewat (False Negative), yang kritis dalam konteks medis."
    )

    st.markdown("---")

    # ── Perbandingan Semua Model ──
    st.markdown('<div class="section-header">⚖️ Perbandingan Semua Model (Baseline)</div>', unsafe_allow_html=True)

    cols_show = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    display_df = metrics_df[cols_show].copy()
    display_df["Accuracy"]  = display_df["Accuracy"].map(lambda x: f"{x*100:.1f}%")
    display_df["Precision"] = display_df["Precision"].map(lambda x: f"{x*100:.1f}%")
    display_df["Recall"]    = display_df["Recall"].map(lambda x: f"{x*100:.1f}%")
    display_df["F1 Score"]  = display_df["F1 Score"].map(lambda x: f"{x*100:.1f}%")
    display_df["ROC AUC"]   = display_df["ROC AUC"].map(lambda x: f"{x:.4f}")

    # Highlight best row
    def highlight_best(row):
        if row["Model"] == "Logistic Regression":
            return ["background-color: #1a3a6e; color: #4fc3f7; font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("")
    st.markdown("""
    <div class="info-box">
    📌 <b>Logistic Regression dipilih</b> sebagai model terbaik karena unggul di Recall (95.9%) dan F1 Score (75.0%)
    — metrik paling penting untuk skrining medis di mana false negative berakibat fatal.
    Meskipun fuzzy preprocessing diuji, hasilnya menurunkan performa di semua metrik sehingga baseline dipertahankan.
    </div>
    """, unsafe_allow_html=True)

    # ── Quick preview images ──
    st.markdown("---")
    st.markdown('<div class="section-header">🖼️ Preview Visualisasi Model</div>', unsafe_allow_html=True)

    img_col1, img_col2, img_col3 = st.columns(3)
    cm_img  = load_img(os.path.join("results", "confusion_matrix", "logistic_regression_Baseline.png"))
    fi_img  = load_img(os.path.join("results", "feature_importance", "logistic_regression_Baseline.png"))
    roc_img = load_img(os.path.join("results", "roc_curve", "logistic_regression_Baseline.png"))

    with img_col1:
        st.markdown("**Confusion Matrix**")
        if cm_img:
            st.image(cm_img, use_container_width=True)
        else:
            st.caption("📁 Letakkan file di: `results/confusion_matrix/logistic_regression_Baseline.png`")
    with img_col2:
        st.markdown("**Feature Importance**")
        if fi_img:
            st.image(fi_img, use_container_width=True)
        else:
            st.caption("📁 Letakkan file di: `results/feature_importance/logistic_regression_Baseline.png`")
    with img_col3:
        st.markdown("**ROC Curve**")
        if roc_img:
            st.image(roc_img, use_container_width=True)
        else:
            st.caption("📁 Letakkan file di: `results/roc_curve/logistic_regression_Baseline.png`")

    st.caption("👈 Klik **Evaluasi Model** di sidebar untuk melihat semua visualisasi lengkap.")


# ═══════════════════════════════════════════════
# PAGE 2 — PREDIKSI
# ═══════════════════════════════════════════════
elif page == "🔬 Prediksi Pasien":
    st.markdown("# 🔬 Prediksi Risiko Kanker Paru")
    st.markdown("Masukkan data klinis pasien di bawah ini. Semua fitur bertanda \\* wajib diisi.")
    st.markdown("---")

    if model is None:
        st.error(
            "⚠️ **Model tidak ditemukan.** Pastikan file `models/logistic_regression.pkl` "
            "ada di direktori yang sama dengan `app.py`."
        )
        st.stop()

    # ── Form Input ──
    with st.form("patient_form"):
        st.markdown("#### 👤 Data Demografis")
        d1, d2 = st.columns(2)
        with d1:
            age    = st.number_input("Usia (Age) *", min_value=1, max_value=120, value=45, step=1)
        with d2:
            gender = st.selectbox("Jenis Kelamin (Gender) *", options=[1, 0], format_func=lambda x: "Pria" if x == 1 else "Wanita")

        st.markdown("#### 🌍 Faktor Lingkungan & Gaya Hidup *(Skala 1–8)*")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            air_pollution   = st.slider("Air Pollution",       1, 8, 4)
            alcohol_use     = st.slider("Alcohol Use",         1, 8, 4)
        with e2:
            dust_allergy    = st.slider("Dust Allergy",        1, 8, 4)
            occ_hazards     = st.slider("Occupational Hazards",1, 8, 4)
        with e3:
            genetic_risk    = st.slider("Genetic Risk",        1, 8, 4)
            chronic_lung    = st.slider("Chronic Lung Disease",1, 8, 4)
        with e4:
            balanced_diet   = st.slider("Balanced Diet",       1, 8, 4)
            obesity         = st.slider("Obesity",             1, 8, 4)

        st.markdown("#### 🚬 Faktor Merokok *(Skala 1–9)*")
        s1, s2, s3 = st.columns(3)
        with s1:
            smoking         = st.slider("Smoking",             1, 8, 4)
        with s2:
            passive_smoker  = st.slider("Passive Smoker",      1, 8, 4)

        st.markdown("#### 🩺 Gejala Klinis *(Skala 1–9)*")
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            chest_pain      = st.slider("Chest Pain",          1, 9, 3)
            coughing_blood  = st.slider("Coughing of Blood",   1, 9, 3)
            fatigue         = st.slider("Fatigue",             1, 9, 3)
        with g2:
            weight_loss     = st.slider("Weight Loss",         1, 9, 3)
            shortness_breath= st.slider("Shortness of Breath", 1, 9, 3)
            wheezing        = st.slider("Wheezing",            1, 9, 3)
        with g3:
            swallowing_diff = st.slider("Swallowing Difficulty",  1, 9, 3)
            clubbing        = st.slider("Clubbing of Finger Nails",1, 9, 3)
        with g4:
            frequent_cold   = st.slider("Frequent Cold",       1, 9, 3)
            dry_cough       = st.slider("Dry Cough",           1, 9, 3)
            snoring         = st.slider("Snoring",             1, 9, 3)

        submitted = st.form_submit_button("🔍 Jalankan Prediksi", use_container_width=True, type="primary")

    # ── Prediction Logic ──
    if submitted:
        # Build DataFrame with exact feature order from training
        input_dict = {
            'Age':                    [age],
            'Gender':                 [gender],
            'Air Pollution':          [air_pollution],
            'Alcohol use':            [alcohol_use],
            'Dust Allergy':           [dust_allergy],
            'OccuPational Hazards':   [occ_hazards],
            'Genetic Risk':           [genetic_risk],
            'chronic Lung Disease':   [chronic_lung],
            'Balanced Diet':          [balanced_diet],
            'Obesity':                [obesity],
            'Smoking':                [smoking],
            'Smoking.1':              [smoking],
            'Passive Smoker':         [passive_smoker],
            'Chest Pain':             [chest_pain],
            'Coughing of Blood':      [coughing_blood],
            'Fatigue':                [fatigue],
            'Weight Loss':            [weight_loss],
            'Shortness of Breath':    [shortness_breath],
            'Wheezing':               [wheezing],
            'Swallowing Difficulty':  [swallowing_diff],
            'Swallowing Difficulty.1':[swallowing_diff],
            'Clubbing of Finger Nails':[clubbing],
            'Frequent Cold':          [frequent_cold],
            'Dry Cough':              [dry_cough],
            'Snoring':                [snoring],
        }
        input_df = pd.DataFrame(input_dict)

        # Align columns to model expectation
        try:
            if hasattr(model, 'feature_names_in_'):
                expected_cols = list(model.feature_names_in_)
                # Reorder / fill missing with 0
                for col in expected_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[expected_cols]
        except Exception:
            pass  # proceed with current column order

        try:
            prediction  = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            prob_pos    = probability[1]   # probability of positive
            prob_neg    = probability[0]

            st.markdown("---")
            st.markdown("### 📋 Hasil Prediksi")

            res_col, prob_col = st.columns([1, 1])

            with res_col:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-positive">
                        <div class="result-title">⚠️ POSITIF</div>
                        <div class="result-sub">
                            Model mendeteksi <strong>risiko kanker paru-paru</strong>.<br>
                            Disarankan pemeriksaan lanjutan oleh dokter spesialis.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <div class="result-title">✅ NEGATIF</div>
                        <div class="result-sub">
                            Model <strong>tidak mendeteksi</strong> risiko kanker paru-paru.<br>
                            Tetap jaga kesehatan dan lakukan pemeriksaan rutin.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with prob_col:
                st.markdown("**Probabilitas Prediksi**")
                st.markdown(f"""
                <div class="prob-bar-container">
                    <div style="margin-bottom:10px">
                        <span style="color:#ef5350; font-weight:600">Positif (Kanker)</span>
                        <span style="float:right; color:#4fc3f7; font-weight:700">{prob_pos*100:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(float(prob_pos))
                st.markdown(f"""
                    <div style="margin-top:14px; margin-bottom:10px">
                        <span style="color:#66bb6a; font-weight:600">Negatif (Sehat)</span>
                        <span style="float:right; color:#4fc3f7; font-weight:700">{prob_neg*100:.1f}%</span>
                    </div>
                """, unsafe_allow_html=True)
                st.progress(float(prob_neg))
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Input Summary ──
            st.markdown("---")
            with st.expander("📄 Lihat Ringkasan Data Input Pasien"):
                summary_data = {
                    "Fitur": list(input_dict.keys()),
                    "Nilai": [v[0] for v in input_dict.values()],
                }
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

            # ── Disclaimer ──
            st.warning(
                "⚕️ **Disclaimer:** Hasil prediksi ini bersifat **bantuan skrining** dan **bukan diagnosis medis**. "
                "Model memiliki akurasi 61% dengan recall 95.9%. Selalu konsultasikan hasil ke dokter spesialis."
            )

        except ValueError as e:
            st.error(f"❌ **Feature mismatch:** {e}")
            st.info("Model mengharapkan kolom berikut:")
            if hasattr(model, 'feature_names_in_'):
                st.code(list(model.feature_names_in_))
            st.info("Kolom yang dikirim:")
            st.code(list(input_df.columns))
        except Exception as e:
            st.error(f"❌ Terjadi error: {e}")


# ═══════════════════════════════════════════════
# PAGE 3 — EVALUASI MODEL
# ═══════════════════════════════════════════════
elif page == "📊 Evaluasi Model":
    st.markdown("# 📊 Evaluasi Model — Visualisasi Lengkap")
    st.markdown("---")

    model_options = {
        "Logistic Regression (Baseline) ⭐": "logistic_regression_Baseline",
        "Random Forest (Baseline)":          "random_forest_Baseline",
        "XGBoost (Baseline)":                "xgboost_Baseline",
        "Logistic Regression (Fuzzy)":       "logistic_regression_fuzzy",
        "Random Forest (Fuzzy)":             "random_forest_fuzzy",
        "XGBoost (Fuzzy)":                   "xgboost_fuzzy",
    }

    sel_model = st.selectbox(
        "Pilih model untuk dilihat visualisasinya:",
        list(model_options.keys()),
    )
    fname = model_options[sel_model]

    tabs = st.tabs(["🔲 Confusion Matrix", "📉 ROC Curve", "🔑 Feature Importance"])

    with tabs[0]:
        st.markdown(f"### Confusion Matrix — {sel_model}")
        cm_path = os.path.join("results", "confusion_matrix", f"{fname}.png")
        img = load_img(cm_path)
        if img:
            col_l, col_c, col_r = st.columns([1, 3, 1])
            with col_c:
                st.image(img, use_container_width=True)
        else:
            st.warning(f"📁 File tidak ditemukan: `{cm_path}`")
            st.caption("Pastikan nama file sesuai. Coba juga nama: `logistic_regression_baseline.png` (lowercase).")

        with st.expander("📖 Cara membaca Confusion Matrix"):
            st.markdown("""
            | | Prediksi Positif | Prediksi Negatif |
            |---|---|---|
            | **Aktual Positif** | TP (True Positive) | FN (False Negative) |
            | **Aktual Negatif** | FP (False Positive) | TN (True Negative) |

            - **TP**: Pasien kanker → diprediksi kanker ✅
            - **TN**: Pasien sehat → diprediksi sehat ✅
            - **FP**: Pasien sehat → diprediksi kanker ❌ (over-alert)
            - **FN**: Pasien kanker → diprediksi sehat ❌ (**kritis!**)

            Model ini diprioritaskan **meminimalkan FN** karena miss-detection kanker berakibat fatal.
            """)

    with tabs[1]:
        st.markdown(f"### ROC Curve — {sel_model}")
        roc_path = os.path.join("results", "roc_curve", f"{fname}.png")
        img = load_img(roc_path)
        if img:
            col_l, col_c, col_r = st.columns([1, 3, 1])
            with col_c:
                st.image(img, use_container_width=True)
        else:
            st.warning(f"📁 File tidak ditemukan: `{roc_path}`")

        with st.expander("📖 Cara membaca ROC Curve"):
            st.markdown("""
            - **AUC = 1.0**: Prediksi sempurna
            - **AUC = 0.5**: Sama dengan tebak acak (garis diagonal)
            - **AUC Logistic Regression = 0.4751** — sedikit di bawah random,
              menunjukkan model cenderung bias ke salah satu kelas

            Meskipun AUC rendah, **Recall 95.9%** tetap membuat model berguna
            sebagai *screening tool* awal karena sangat jarang melewatkan kasus positif.
            """)

    with tabs[2]:
        st.markdown(f"### Feature Importance — {sel_model}")
        fi_path = os.path.join("results", "feature_importance", f"{fname}.png")
        img = load_img(fi_path)
        if img:
            col_l, col_c, col_r = st.columns([0.5, 4, 0.5])
            with col_c:
                st.image(img, use_container_width=True)
        else:
            st.warning(f"📁 File tidak ditemukan: `{fi_path}`")

        with st.expander("📖 Interpretasi Feature Importance"):
            st.markdown("""
            Feature importance menunjukkan seberapa besar pengaruh setiap fitur terhadap prediksi.

            **Untuk Logistic Regression**: nilai ditunjukkan oleh magnitude koefisien.
            - **Koefisien positif besar** → fitur tersebut meningkatkan probabilitas kanker
            - **Koefisien negatif besar** → fitur tersebut menurunkan probabilitas kanker

            Fitur dengan importance tinggi perlu mendapat perhatian khusus dalam wawancara klinis.
            """)

    # ── Metrics comparison table at bottom ──
    st.markdown("---")
    st.markdown("### 📋 Tabel Metrik Lengkap")
    st.dataframe(
        metrics_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
            color="#1a3a6e",
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── Final comparison chart ──
    final_chart_path = os.path.join("results", "final_comparison_chart.png")
    fc_img = load_img(final_chart_path)
    if fc_img:
        st.markdown("### 📊 Final Comparison Chart")
        st.image(fc_img, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 4 — INFORMASI
# ═══════════════════════════════════════════════
elif page == "ℹ️ Informasi":
    st.markdown("# ℹ️ Informasi Sistem")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏗️ Arsitektur Sistem")
        st.markdown("""
        <div class="info-box">
        <b>Pipeline Model</b><br>
        StandardScaler → LogisticRegression<br><br>
        <b>Library</b><br>
        scikit-learn, pandas, numpy, streamlit<br><br>
        <b>Training Split</b><br>
        80% train / 20% test (random_state=42)<br><br>
        <b>Max Iterations</b><br>
        1000 iterasi konvergensi
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📁 Struktur File")
        st.code("""
project/
├── app.py
├── models/
│   └── logistic_regression.pkl
├── results/
│   ├── confusion_matrix/
│   │   └── logistic_regression_Baseline.png
│   ├── feature_importance/
│   │   └── logistic_regression_Baseline.png
│   ├── roc_curve/
│   │   └── logistic_regression_Baseline.png
│   ├── baseline_metrics.csv
│   └── final_comparison_chart.png
└── src/
    ├── preprocessing.py
    ├── train_logistic.py
    └── evaluation.py
        """, language="text")

    with col2:
        st.markdown("### 📊 Dataset")
        st.markdown("""
        <div class="info-box">
        <b>Jumlah Data</b>: 1000 pasien<br>
        <b>Jumlah Fitur</b>: 25 fitur klinis<br>
        <b>Target</b>: Result (0 = Negatif, 1 = Positif)<br>
        <b>Kolom tambahan</b>: Level (Low/Medium/High risk)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔢 Deskripsi Fitur")
        feature_info = pd.DataFrame({
            "Fitur": [
                "Age", "Gender", "Air Pollution", "Alcohol Use",
                "Dust Allergy", "Occupational Hazards", "Genetic Risk",
                "Chronic Lung Disease", "Balanced Diet", "Obesity",
                "Smoking", "Smoking.1", "Passive Smoker", "Chest Pain",
                "Coughing of Blood", "Fatigue", "Weight Loss",
                "Shortness of Breath", "Wheezing", "Swallowing Difficulty",
                "Swallowing Difficulty.1", "Clubbing of Finger Nails",
                "Frequent Cold", "Dry Cough", "Snoring",
            ],
            "Skala": (
                ["Numerik", "Biner (0/1)"] +
                ["1–8"] * 8 +
                ["1–8"] * 3 +
                ["1–9"] * 12
            ),
            "Keterangan": [
                "Usia pasien", "1=Pria, 0=Wanita",
                "Tingkat polusi udara", "Konsumsi alkohol",
                "Alergi debu", "Paparan bahaya kerja", "Risiko genetik",
                "Penyakit paru kronis", "Pola diet seimbang", "Obesitas",
                "Merokok aktif", "Duplikat indikator merokok", "Perokok pasif",
                "Nyeri dada", "Batuk darah", "Kelelahan", "Penurunan berat badan",
                "Sesak napas", "Mengi", "Kesulitan menelan",
                "Duplikat kesulitan menelan", "Penebalan ujung jari",
                "Sering pilek", "Batuk kering", "Mendengkur",
            ],
        })
        st.dataframe(feature_info, use_container_width=True, hide_index=True, height=300)

    st.markdown("---")
    st.markdown("### ⚕️ Disclaimer Medis")
    st.error(
        "**PENTING:** Aplikasi ini adalah alat bantu skrining penelitian, bukan pengganti diagnosis medis profesional. "
        "Model memiliki keterbatasan akurasi (61%) dan tidak boleh digunakan sebagai satu-satunya dasar keputusan klinis. "
        "Selalu konsultasikan dengan dokter spesialis paru untuk diagnosis dan penanganan yang tepat."
    )
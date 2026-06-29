import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_preprocessed_data
from fuzzification import apply_fuzzification

from config import (
    SRC_DIR,
    MODELS_DIR,
    FI_DIR,
    BASELINE_METRICS_CSV,
    FUZZY_METRICS_CSV,
    FINAL_METRICS_CSV,
    FINAL_CHART_PNG
)

plt.style.use("ggplot")

def plot_feature_importance():
    """Mengekstrak dan menggambar Feature Importance / Koefisien menggunakan path terpusat"""
    print("\n" + "—"*60)
    print("👉 Sedang memproses: Generate Feature Importance & Coefficients...")
    print("—"*60)
    
    # Ambil data lengkap (6 variabel) agar sinkron dengan file training
    X_train_raw, _, _, _, _, _ = load_preprocessed_data()
    baseline_features = X_train_raw.columns.tolist()
    
    X_train_fuzzy = apply_fuzzification(X_train_raw)
    fuzzy_features = X_train_fuzzy.columns.tolist()
    
    src_models_dir = os.path.join(SRC_DIR, "models")
    
    models_to_plot = {
        "Logistic Regression Baseline": ("logistic_regression.pkl", baseline_features),
        "Random Forest Baseline": ("random_forest.pkl", baseline_features),
        "XGBoost Baseline": ("xgboost.pkl", baseline_features),
        "Logistic Regression Fuzzy": ("logistic_regression_fuzzy.pkl", fuzzy_features),
        "Random Forest Fuzzy": ("random_forest_fuzzy.pkl", fuzzy_features),
        "XGBoost Fuzzy": ("xgboost_fuzzy.pkl", fuzzy_features)
    }
    
    os.makedirs(FI_DIR, exist_ok=True)
    
    for name, (model_file, features) in models_to_plot.items():
        model_path = os.path.join(MODELS_DIR, model_file)
        if not os.path.exists(model_path):
            model_path = os.path.join(src_models_dir, model_file)
            
        if os.path.exists(model_path):
            print(f"🔍 Menemukan model {name}")
            model = joblib.load(model_path)
            
            # Unwrap jika model dibungkus di dalam Pipeline scikit-learn
            if hasattr(model, "named_steps") or hasattr(model, "steps"):
                actual_model = model[-1] 
            else:
                actual_model = model
            
            importances = None
            title_label = "Feature Importance"
            
            # Ekstraksi nilai importance / koefisien
            if hasattr(actual_model, "feature_importances_"):
                importances = actual_model.feature_importances_
                title_label = "Feature Importance"
            elif hasattr(actual_model, "coef_"):
                importances = abs(actual_model.coef_[0])
                title_label = "Feature Importance (Abs Coefficients)"
                
            if importances is not None:
                # FIX JIKA ADA PERBEDAAN DIMENSI: 
                # Jika jumlah fitur tidak sama, kita sesuaikan penamaan fiturnya secara dinamis
                if len(features) != len(importances):
                    print(f"⚠️ Penyesuaian dimensi fitur untuk {name}: model ({len(importances)}) vs data ({len(features)})")
                    features = [f"Feature_{i}" for i in range(len(importances))]
                    
                df_fi = pd.DataFrame({"Feature": features, "Importance": importances})
                df_fi = df_fi.sort_values(by="Importance", ascending=False).head(10)
                
                plt.figure(figsize=(10, 5))
                sns.barplot(data=df_fi, x="Importance", y="Feature", palette="viridis")
                
                clean_name = name.replace(" Fuzzy", "").replace(" fuzzy", "").replace("(fuzzy)", "").strip()
                
                if "Fuzzy" in name:
                    plt.title(f"Top 10 {title_label}\n({clean_name} - Fuzzy Model)", fontsize=11, fontweight="bold")
                else:
                    plt.title(f"Top 10 {title_label}\n({clean_name})", fontsize=11, fontweight="bold")
                
                plt.xlabel("Importance Score / Weight Magnitude")
                plt.ylabel("Features")
                plt.tight_layout()
                
                filename = f"{name.lower().replace(' ', '_')}.png"
                output_path = os.path.join(FI_DIR, filename)
                
                plt.savefig(output_path, dpi=300)
                plt.close()
                print(f"✅ Grafik berhasil disimpan ke: {output_path}")

def generate_final_report():
    """Membuat tabel komparasi akhir dan grafik menggunakan path dari config.py"""
    if not os.path.exists(BASELINE_METRICS_CSV) or not os.path.exists(FUZZY_METRICS_CSV):
        print("[ERROR] File master metrics tidak ditemukan di folder results.")
        return

    df_base = pd.read_csv(BASELINE_METRICS_CSV).drop_duplicates(subset=["Model"], keep="last")
    df_fuzzy = pd.read_csv(FUZZY_METRICS_CSV).drop_duplicates(subset=["Model"], keep="last")
    df_final = pd.concat([df_base, df_fuzzy], ignore_index=True)
    
    df_final['Tipe'] = df_final['Model'].apply(lambda x: 'Fuzzy' if 'Fuzzy' in x else 'Baseline')
    df_final['Algoritma'] = df_final['Model'].str.replace(" Fuzzy", "")
    
    print("\n" + "="*75)
    print("         TABEL UTAMA PERBANDINGAN PERFORMA (UNTUK BAB 4 PAPER)          ")
    print("="*75)
    print(df_final[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].to_string(index=False))
    print("="*75)
    
    df_final.to_csv(FINAL_METRICS_CSV, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    custom_palette = {"Baseline": "#5B9BD5", "Fuzzy": "#ED7D31"}

    ax1 = sns.barplot(data=df_final, x="Algoritma", y="Accuracy", hue="Tipe", ax=axes[0], palette=custom_palette)
    axes[0].set_title("Perbandingan Skor Accuracy", fontsize=12, fontweight="bold", pad=10)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Skor (0.0 - 1.0)")

    ax2 = sns.barplot(data=df_final, x="Algoritma", y="ROC AUC", hue="Tipe", ax=axes[1], palette=custom_palette)
    axes[1].set_title("Perbandingan Skor ROC AUC", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("")

    for ax in [ax1, ax2]:
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=10, fontweight='bold')

    plt.suptitle("Analisis Dampak Fuzzification Terhadap Performa Model ML", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    plt.savefig(FINAL_CHART_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Grafik komparasi akhir disimpan ke: {FINAL_CHART_PNG}")
    
    plot_feature_importance()





if __name__ == "__main__":
    generate_final_report()
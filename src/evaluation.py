import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

from config import CM_DIR, ROC_DIR, BASELINE_METRICS_CSV, FUZZY_METRICS_CSV

plt.style.use("ggplot")

def evaluate_and_plot(model_name, y_true, y_pred, y_prob, is_fuzzy=False):
    """
    Menghitung metrik evaluasi, menyimpan ke CSV master, 
    serta menggambar Confusion Matrix & ROC Curve tanpa subfolder baseline/fuzzy.
    """
    print(f"\n[EVALUASI] Memproses Evaluasi untuk: {model_name}")
    
    clean_name = model_name.replace(" Fuzzy", "").replace(" fuzzy", "").replace("(fuzzy)", "").strip()
    
    suffix = "fuzzy" if is_fuzzy else "baseline"
    file_base_name = f"{clean_name.lower().replace(' ', '_')}_{suffix}.png"
    
    title_suffix = f"{clean_name} - Fuzzy" if is_fuzzy else clean_name

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0  # Fallback jika hanya ada 1 kelas di target prediktor
        
    print(f" -> Accuracy : {acc:.4f} | ROC AUC : {roc_auc:.4f}")

    os.makedirs(CM_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    
    plt.title(f"Confusion Matrix\n({title_suffix})", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    cm_output_path = os.path.join(CM_DIR, file_base_name)
    plt.savefig(cm_output_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Confusion Matrix disimpan ke: {cm_output_path}")

    os.makedirs(ROC_DIR, exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f"ROC Curve\n({title_suffix})", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    roc_output_path = os.path.join(ROC_DIR, file_base_name)
    plt.savefig(roc_output_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] ROC Curve disimpan ke: {roc_output_path}")

    final_model_label = f"{clean_name} Fuzzy" if is_fuzzy else clean_name
    
    df_metrics = pd.DataFrame([{
        "Model": final_model_label,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC AUC": round(roc_auc, 4)
    }])
    
    target_csv = FUZZY_METRICS_CSV if is_fuzzy else BASELINE_METRICS_CSV
    
    if os.path.exists(target_csv):
        df_metrics.to_csv(target_csv, mode='a', header=False, index=False)
    else:
        df_metrics.to_csv(target_csv, index=False)
        
    print(f"[SUCCESS] Metrik skor dicatat ke: {target_csv}")
    
    return df_metrics
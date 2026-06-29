import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrix")
FI_DIR = os.path.join(RESULTS_DIR, "feature_importance")
ROC_DIR = os.path.join(RESULTS_DIR, "roc_curve")


BASELINE_METRICS_CSV = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
FUZZY_METRICS_CSV = os.path.join(RESULTS_DIR, "fuzzy_metrics.csv")

FINAL_METRICS_CSV = os.path.join(RESULTS_DIR, "final_comparison_metrics.csv")
FINAL_CHART_PNG = os.path.join(RESULTS_DIR, "final_comparison_chart.png")



RANDOM_STATE = 42
TEST_SIZE = 0.2
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import load_preprocessed_data
from fuzzification import apply_fuzzification
from evaluation import evaluate_and_plot
from config import MODELS_DIR

def train_random_forest():
    X_train_raw, X_test_raw, y_train, y_test, _, _ = load_preprocessed_data()
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    
    print("\n" + "="*50)
    print("[TRAIN] Random Forest - BASELINE")
    print("="*50)
    
    model_base = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
    model_base.fit(X_train_raw, y_train)
    
    y_pred_base = model_base.predict(X_test_raw)
    y_prob_base = model_base.predict_proba(X_test_raw)[:, 1]
    
    joblib.dump(model_base, os.path.join(MODELS_DIR, "random_forest.pkl"))
    evaluate_and_plot("Random Forest", y_test, y_pred_base, y_prob_base, is_fuzzy=False)
    
   
    print("\n" + "="*50)
    print("[TRAIN] Random Forest - FUZZY")
    print("="*50)
    
    X_train_fuzzy = apply_fuzzification(X_train_raw)
    X_test_fuzzy = apply_fuzzification(X_test_raw)
    
    model_fuzzy = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
    model_fuzzy.fit(X_train_fuzzy, y_train)
    
    y_pred_fuzzy = model_fuzzy.predict(X_test_fuzzy)
    y_prob_fuzzy = model_fuzzy.predict_proba(X_test_fuzzy)[:, 1]
    
    joblib.dump(model_fuzzy, os.path.join(MODELS_DIR, "random_forest_fuzzy.pkl"))
    evaluate_and_plot("Random Forest", y_test, y_pred_fuzzy, y_prob_fuzzy, is_fuzzy=True)

if __name__ == "__main__":
    train_random_forest()
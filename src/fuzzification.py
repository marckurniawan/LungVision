import numpy as np
import pandas as pd

def trimf(x, a, b, c):
    """Fungsi Keanggotaan Segitiga (Triangular Membership Function)"""
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-5), (c - x) / (c - b + 1e-5)))

def trapmf(x, a, b, c, d):
    """Fungsi Keanggotaan Trapesium (Trapezoidal Membership Function)"""
    return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a + 1e-5), 1), (d - x) / (d - c + 1e-5)))

def apply_fuzzification(X_df):
    """
    Mentransformasikan dataframe numerik biasa menjadi dataset berbasis derajat keanggotaan fuzzy.
    """
    X_fuzzy = pd.DataFrame(index=X_df.index)
    
    # 1. Fuzzifikasi Umur (Age)
    # Muda: <= 35, Paruh Baya: 30-60, Tua: >= 55
    X_fuzzy['Age_Young'] = trapmf(X_df['Age'], 0, 0, 30, 38)
    X_fuzzy['Age_Middle'] = trimf(X_df['Age'], 32, 45, 58)
    X_fuzzy['Age_Old'] = trapmf(X_df['Age'], 52, 62, 100, 100)
    
    # 2. Fuzzifikasi Fitur Klinis & Gaya Hidup (Skala Nilai 1 - 8)
    # Kita ambil semua kolom kecuali Age dan Gender
    clinical_cols = [col for col in X_df.columns if col not in ['Age', 'Gender']]
    
    for col in clinical_cols:
        # Tingkat Rendah/Low (Skala bawah)
        X_fuzzy[f'{col}_Low'] = trapmf(X_df[col], 0, 1, 3, 4)
        # Tingkat Sedang/Medium (Skala menengah)
        X_fuzzy[f'{col}_Medium'] = trimf(X_df[col], 3, 5, 7)
        # Tingkat Tinggi/High (Skala atas)
        X_fuzzy[f'{col}_High'] = trapmf(X_df[col], 5, 7, 9, 10)
        
    # 3. Fitur Kategorikal yang sudah biner (Gender) tidak perlu difuzzifikasi, langsung dipertahankan
    if 'Gender' in X_df.columns:
        X_fuzzy['Gender'] = X_df['Gender']
        
    return X_fuzzy
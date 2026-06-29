import os
import subprocess
import sys

def run_pipeline_script(script_name):
    """Fungsi helper untuk menjalankan skrip python di dalam folder src"""
    script_path = os.path.join("src", script_name)
    
    print("\n" + "-"*60)
    print(f"[EKSEKUSI] Menjalankan: {script_path}")
    print("-"*60)
    
    # --- FIX MODULE NOT FOUND: Menambahkan folder 'src' ke PYTHONPATH ---
    env = os.environ.copy()
    src_dir = os.path.abspath("src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = src_dir + os.path.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_dir
    # -------------------------------------------------------------------
    
    # Menjalankan skrip menggunakan interpreter Python aktif dan environment yang baru
    process = subprocess.run([sys.executable, script_path], env=env)
    
    # Validasi jika skrip error di tengah jalan
    if process.returncode != 0:
        print(f"\n[ERROR] Kegagalan sistem terdeteksi pada skrip: {script_path}")
        print("Proses pipeline dihentikan.")
        sys.exit(process.returncode)
        
    print(f"[SELESAI] {script_name} berhasil dieksekusi dengan aman.\n")

def main():
    print("="*70)
    print("      PIPELINE OTOMATISASI DETEKSI KANKER PARU (BASE vs FUZZY)      ")
    print("="*70)
    print("Memulai eksperimen komparasi model secara end-to-end...\n")

    # Jalankan urutan pipeline sesuai arsitektur 3 file baru
    run_pipeline_script("train_logistic.py")      # Baseline & Fuzzy Logistic
    run_pipeline_script("train_randomforest.py")  # Baseline & Fuzzy Random Forest
    run_pipeline_script("train_xgboost.py")       # Baseline & Fuzzy XGBoost
    run_pipeline_script("visualization.py")       # Rekap Tabel Utama & Grafik Akhir

    print("="*70)
    print("          SELURUH TAHAPAN PIPELINE TELAH SELESAI DIJALANKAN         ")
    print("="*70)
    print("Hasil dapat Anda periksa langsung pada repositori lokal:")
    print(" 📂 Model terlatih (.pkl)  -> folder 'models/'")
    print(" 📂 Grafik & Tabel (.csv)  -> folder 'results/'")
    print("="*70)

if __name__ == "__main__":
    main()
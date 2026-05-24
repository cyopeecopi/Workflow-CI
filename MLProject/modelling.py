import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

# 1. Menyiapkan Eksperimen MLflow
#mlflow.set_tracking_uri("file:./mlruns") # <-- TAMBAHKAN BARIS INI
mlflow.set_experiment("Wine_Quality_Prediction_Basic")

def run_training():
    # 2. Mengaktifkan Autolog (Wajib untuk Kriteria Basic)
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="LDA_Basic_Model"):
        print("Memuat dataset...")
        # Sesuaikan dengan nama file CSV yang sudah ada di folder ini
        df = pd.read_csv("wine_quality_clean.csv")
        
        # 3. Memisahkan fitur dan target
        X = df.drop('quality', axis=1)
        y = df['quality'].astype(int) # Memastikan target berupa integer untuk klasifikasi
        
        # 4. Membagi data latih dan uji (80% latih, 20% uji)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Melatih model Linear Discriminant Analysis (LDA)...")
        # 5. Inisialisasi dan pelatihan model
        model = LinearDiscriminantAnalysis()
        model.fit(X_train, y_train)
        
        # 6. Prediksi dan Evaluasi (opsional dicetak ke terminal karena MLflow sudah mencatatnya)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Akurasi Model: {acc:.4f}")
        print("Pelatihan selesai. Metrik telah dicatat oleh MLflow autolog.")

if __name__ == "__main__":
    run_training()

import shutil
import os

# Hapus folder saved_model jika sudah ada dari percobaan sebelumnya
if os.path.exists("saved_model"):
    shutil.rmtree("saved_model")

# Simpan model secara langsung agar mudah diambil oleh Docker
mlflow.sklearn.save_model(model, "saved_model")
import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

# 1. MATIKAN KEDUA BARIS INI AGAR TIDAK BENTROK DENGAN GITHUB ACTIONS
# mlflow.set_tracking_uri("file:./mlruns") 
# mlflow.set_experiment("Wine_Quality_Prediction_Basic")

def run_training():
    # 2. Mengaktifkan Autolog
    mlflow.sklearn.autolog()
    
    # Tambahkan parameter nested=True agar selaras dengan runner otomatis GitHub
    with mlflow.start_run(run_name="LDA_Basic_Model", nested=True):
        print("Memuat dataset...")
        # Pastikan nama file CSV sesuai dengan yang ada di folder Anda
        df = pd.read_csv("wine_quality_clean.csv") 
        
        # 3. Memisahkan fitur dan target
        X = df.drop('quality', axis=1)
        y = df['quality'].astype(int) 
        
        # 4. Membagi data latih dan uji (80% latih, 20% uji)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Melatih model Linear Discriminant Analysis (LDA)...")
        # 5. Inisialisasi dan pelatihan model
        model = LinearDiscriminantAnalysis()
        model.fit(X_train, y_train)
        
        # 6. Prediksi dan Evaluasi
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Akurasi Model: {acc:.4f}")
        
        # --- 7. KODE SIMPAN DOCKER SEKARANG MASUK DI SINI (DI DALAM FUNGSI) ---
        if os.path.exists("saved_model"):
            shutil.rmtree("saved_model")

        # Sekarang perintah ini bisa membaca variabel 'model' yang dilatih di atasnya
        mlflow.sklearn.save_model(model, "saved_model")
        print("Model berhasil disimpan ke folder 'saved_model' untuk Docker!")

if __name__ == "__main__":
    run_training()
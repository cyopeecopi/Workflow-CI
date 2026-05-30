import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import shutil

def run_training():
    # Perhatikan: Kita TIDAK memakai mlflow.set_experiment() di sini 
    # karena MLflow Project sudah mengaturnya secara otomatis di latar belakang.
    with mlflow.start_run():
        print("Membaca data...")
        df = pd.read_csv('wdbc_preprocessed.csv')
        
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Melatih model...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        print("Mencatat metrik...")
        mlflow.log_metric("accuracy", model.score(X_test, y_test))

        # Menyiapkan folder artefak statis
        model_dir = "saved_model"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            
        print("Menyimpan artefak...")
        # Menyimpan model ke server MLflow
        mlflow.sklearn.log_model(model, "model")
        # Menyimpan model statis agar bisa dibungkus oleh Docker
        mlflow.sklearn.save_model(model, model_dir)
        print("Selesai! Model siap dibungkus Docker.")

if __name__ == "__main__":
    run_training()
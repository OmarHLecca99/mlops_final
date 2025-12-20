import sys, os
sys.path.append(os.path.abspath("."))

import pandas as pd
import numpy as np
import joblib
import json
import glob
from datetime import datetime
from scipy import sparse
import time

from src.mlflow_tracking import (
    setup_mlflow,
    start_mlflow_run,
    log_params,
    log_metrics,
    log_model,
    log_artifact
)

# Archivos fijos que siempre se sobrescriben
FIXED_LOG_PATH = "data/inference_logs/log.csv"
FIXED_OUTPUT_BATCH_PRED = "data/inference_logs/predicciones_batch.csv"

PREPROCESSOR_PATH = "models/pipelines/preprocess.pkl"
MODEL_PATH = "models/artifacts/model.pkl"

DRIFT_FLAG_PATH = "data/drift/drift_flag.txt"

# ============================================================
# Utils
# ============================================================

def ensure_directories():
    """Asegura que las carpetas necesarias existan."""
    os.makedirs("data/inference_logs", exist_ok=True)  # Asegurar que la carpeta para logs existe


def load_artifacts():
    """Carga el modelo y el preprocesador guardados."""
    print("[INFO] Cargando preprocess.pkl y model.pkl ...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


def auto_detect_input():
    """Detecta automáticamente el archivo CSV más reciente en data/raw/inference/."""
    files = glob.glob("data/raw/inference/*.csv")

    if len(files) == 0:
        raise FileNotFoundError(
            "[ERROR] No se encontraron archivos CSV en data/raw/inference/"
        )

    # Ordenar los archivos por fecha de modificación (o por nombre si tiene la fecha)
    files_sorted = sorted(files, key=os.path.getctime, reverse=True)
    
    print(f"[INFO] Archivo detectado automáticamente: {files_sorted[0]}")
    return files_sorted[0]


def load_batch(input_path):
    """Carga el dataset batch desde un archivo CSV."""
    print(f"[INFO] Cargando dataset batch: {input_path}")
    df = pd.read_csv(input_path)

    if "target" in df.columns:
        df = df.drop(columns=["target"])

    print(f"[INFO] Registros cargados: {len(df)}")
    return df


def preprocess_batch(preprocessor, df):
    """Realiza el preprocesamiento sobre el dataset."""
    print("[INFO] Transformando datos en batch...")
    X = preprocessor.transform(df)

    if isinstance(X, sparse.spmatrix):
        X = X.toarray()

    print(f"[INFO] Matriz preprocesada: {X.shape}")
    return X


def infer_batch(model, X):
    """Realiza la inferencia en el batch de datos."""
    print("[INFO] Ejecutando inferencia en batch...")

    preds = model.predict(X)

    try:
        probas = model.predict_proba(X)[:, 1]
    except:
        probas = [None] * len(preds)

    return preds, probas


def serialize_row(row_dict):
    """Serializa dict a JSON, reemplazando NaN por string 'nan'."""
    clean = {}
    for k, v in row_dict.items():
        clean[k] = "nan" if pd.isna(v) else v
    return json.dumps(clean)


def log_batch(df_raw, preds, probas, input_file):
    """Genera un archivo de log y de predicciones con nombre basado en la fecha y hora de cada ejecución."""
    print(f"[INFO] Registrando inferencias en {FIXED_LOG_PATH} ...")
    ensure_directories()

    registros = []
    for i in range(len(df_raw)):
        raw_dict = df_raw.iloc[i].to_dict()

        registros.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": serialize_row(raw_dict),
            "prediction": preds[i],
            "probability": probas[i]
        })

    df_log = pd.DataFrame(registros)

    # Guardar el log en el archivo correspondiente (archivo fijo)
    df_log.to_csv(FIXED_LOG_PATH, index=False)

    # Guardar predicciones en el nuevo archivo (archivo fijo)
    df_batch = df_raw.copy()
    df_batch["prediction"] = preds
    df_batch["probability"] = probas
    df_batch.to_csv(FIXED_OUTPUT_BATCH_PRED, index=False)

    print(f"[INFO] Predicciones guardadas en {FIXED_LOG_PATH}")
    print(f"[INFO] Predicciones guardadas en {FIXED_OUTPUT_BATCH_PRED}")

# ============================================================
# Utils para el Drift Flag
# ============================================================

def read_drift_flag():
    """Lee el drift_flag.txt para verificar si hay drift detectado"""
    if os.path.exists(DRIFT_FLAG_PATH):
        with open(DRIFT_FLAG_PATH, "r") as file:
            drift_flag = file.read().strip()
            return int(drift_flag)  # Devuelve 0 o 1
    return None  # Si el archivo no existe, consideramos que no hubo retrain

# ============================================================
# Main
# ============================================================

def main(input_path=None):

    # Leer el drift flag antes de proceder
    drift_flag = read_drift_flag()

    if drift_flag == 1:
        print("[INFO] Drift detectado. Terminando ejecución de inferencia.")
        return  # Termina si el drift es 1 (no ejecutar inferencia)

    if drift_flag == 0 or drift_flag is None:
        print("[INFO] No hay drift o el archivo drift_flag.txt no existe. Continuando con la inferencia.")
     
        # Inicializar MLflow para inferencia
        setup_mlflow("mlops_final_project")

        # Si no se especifica un archivo, detecta el más reciente
        if input_path is None:
            input_path = auto_detect_input()

        # Cargar el preprocesador y el modelo guardado
        preprocessor, model = load_artifacts()

        start_time = time.time()

        # Iniciar un run de MLflow
        with start_mlflow_run("batch_inference"):

            log_params({"input_file": input_path})

            # Cargar los datos
            df_raw = load_batch(input_path)
            
            # Preprocesar los datos
            X = preprocess_batch(preprocessor, df_raw)

            # Realizar la inferencia
            preds, probas = infer_batch(model, X)

            # Registrar logs y artefactos
            log_batch(df_raw, preds, probas, input_file=input_path)

            # Registrar métricas de desempeño
            log_metrics({
                "n_registros": len(df_raw),
                "positivos_predichos": int(sum(preds)),
                "tiempo_inferencia": time.time() - start_time
            })

            # Registrar archivo de salida en MLflow
            log_artifact(FIXED_OUTPUT_BATCH_PRED)

            print("\n===== RESUMEN DEL BATCH =====")
            print(f"Total registros: {len(df_raw)}")
            print(f"Predicción positiva: {sum(preds)}")
            print("=============================\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inferencia batch")
    parser.add_argument("--input", type=str, required=False,
                        help="Ruta al CSV. Si no se especifica, detecta automáticamente.")
    args = parser.parse_args()

    main(args.input)
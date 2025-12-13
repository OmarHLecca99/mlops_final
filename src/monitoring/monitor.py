import os
import pandas as pd
import numpy as np
import json
import joblib
import mlflow
import sys
from datetime import datetime
from scipy import sparse

sys.path.append(os.path.abspath("."))

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.mlflow_tracking import (
    setup_mlflow,
    start_mlflow_run,
    log_params,
    log_metrics,
    log_artifact
)

# ============================================================
# Paths
# ============================================================

TRAIN_DATA_PATH = "data/processed/train/train_arrays.npz"
PREPROCESSOR_PATH = "models/pipelines/preprocess.pkl"
INFERENCE_LOG_PATH = "data/inference_logs/log.csv"

# Drift flag para DVC
DRIFT_FLAG_PATH = "data/drift/drift_flag.txt"

# Reportes estáticos
FIXED_REPORT_HTML = "data/monitoring_reports/drift_report.html"
FIXED_REPORT_JSON = "data/monitoring_reports/drift_report.json"

# Config
SAMPLE_SIZE = 50000
DRIFT_THRESHOLD = 0.15   # 15% columnas con drift


# ============================================================
# Utils
# ============================================================

def ensure_directories():
    os.makedirs("data/monitoring_reports", exist_ok=True)
    os.makedirs("data/drift", exist_ok=True)


def load_preprocessor():
    print("[INFO] Cargando preprocessor...")
    return joblib.load(PREPROCESSOR_PATH)


def load_training_baseline(preprocessor):
    print("[INFO] Cargando baseline del entrenamiento...")

    data = np.load(TRAIN_DATA_PATH, allow_pickle=True)
    X_train = data["X"]

    # Convertir sparse a dense si es necesario
    try:
        X_train = X_train.toarray()
    except:
        pass

    # Obtener nombres de features
    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [str(f) for f in feature_names]
        print("[INFO] Feature names obtenidos correctamente.")
    except:
        feature_names = [str(i) for i in range(X_train.shape[1])]
        print("[WARN] Usando índices por falta de nombres reales.")

    df_train = pd.DataFrame(X_train, columns=feature_names)
    print(f"[INFO] Baseline shape completo: {df_train.shape}")

    df_train_sample = df_train.sample(
        min(SAMPLE_SIZE, len(df_train)),
        random_state=42
    )

    print(f"[INFO] Baseline sampleado a: {df_train_sample.shape}")
    return df_train_sample, feature_names


def load_inference_data(preprocessor, feature_names):
    print("[INFO] Cargando logs de inferencia...")

    if not os.path.exists(INFERENCE_LOG_PATH):
        raise FileNotFoundError("[ERROR] No se encontró el archivo de log: log.csv")

    logs = pd.read_csv(INFERENCE_LOG_PATH)
    print(f"[INFO] Total inferencias: {len(logs)}")

    parsed_rows = []
    for row in logs["input"]:
        try:
            parsed_rows.append(json.loads(row))
        except json.JSONDecodeError:
            clean = row.replace('""', '"')
            parsed_rows.append(json.loads(clean))

    df_inputs = pd.DataFrame(parsed_rows)
    print(f"[INFO] Shape original inputs: {df_inputs.shape}")

    # Reparar "nan"
    for col in df_inputs.columns:
        if df_inputs[col].dtype == object:
            df_inputs[col] = df_inputs[col].replace("nan", np.nan)

    print("[INFO] Transformando inferencias...")
    X_inf = preprocessor.transform(df_inputs)

    try:
        X_inf = X_inf.toarray()
    except:
        pass

    df_inf = pd.DataFrame(X_inf, columns=feature_names)
    print(f"[INFO] Shape inferencias procesadas: {df_inf.shape}")

    df_inf_sample = df_inf.sample(
        min(SAMPLE_SIZE, len(df_inf)),
        random_state=42
    )

    print(f"[INFO] Inference sampleado a: {df_inf_sample.shape}")
    return df_inf_sample


def generate_reports(df_train, df_inf):
    print("[INFO] Generando reportes Evidently...")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_train, current_data=df_inf)

    # Guardar reportes
    report.save_html(FIXED_REPORT_HTML)
    report.save_json(FIXED_REPORT_JSON)

    print(f"[INFO] Reportes guardados en:")
    print(f"    {FIXED_REPORT_HTML}")
    print(f"    {FIXED_REPORT_JSON}")

    return FIXED_REPORT_HTML, FIXED_REPORT_JSON


def analyze_drift(report_json):
    print("\n[INFO] Analizando drift...")

    with open(report_json, "r") as f:
        data = json.load(f)

    result = data["metrics"][0]["result"]

    possible_keys = [
        "share_drifted_columns",
        "share_of_drifted_columns",
        "drift_share",
        "number_of_drifted_columns"
    ]

    drift_share = None
    for key in possible_keys:
        if key in result:
            drift_share = result[key]
            print(f"[INFO] Usando clave encontrada: {key}")
            break

    if drift_share is None:
        raise KeyError("[ERROR] No se encontró llave válida de drift en Evidently.")

    drift_flag = result.get("dataset_drift", False)

    print(f"[INFO] dataset_drift flag: {drift_flag}")
    print(f"[INFO] drift_share: {drift_share:.3f}")

    retrain = False

    if drift_flag or drift_share >= DRIFT_THRESHOLD:
        print("\n[ALERTA] DRIFT DETECTADO")
        print(f"[ALERTA] Drift = {drift_share:.3f} (umbral {DRIFT_THRESHOLD})")
        retrain = True

    else:
        print("[INFO] Drift dentro de parámetros aceptables.")

    return retrain, drift_share, drift_flag


# ============================================================
# MAIN
# ============================================================

def main():

    setup_mlflow("mlops_final_project")

    with start_mlflow_run("monitoring_run"):

        log_params({
            "sample_size": SAMPLE_SIZE,
            "drift_threshold": DRIFT_THRESHOLD
        })

        ensure_directories()

        preprocessor = load_preprocessor()
        df_train, feature_names = load_training_baseline(preprocessor)
        df_inf = load_inference_data(preprocessor, feature_names)

        report_html, report_json = generate_reports(df_train, df_inf)

        retrain, drift_share, drift_flag = analyze_drift(report_json)

        # Registrar reportes como artefactos
        log_artifact(report_html, artifact_path="monitoring")
        log_artifact(report_json, artifact_path="monitoring")

        # Registrar métricas
        log_metrics({
            "drift_share": float(drift_share),
            "drift_flag": int(drift_flag),
            "retraining_triggered": int(retrain)
        })

        print("\n===== MONITOREO FINALIZADO =====")

        # ============================================================
        # Nueva lógica: escribir drift_flag.txt para que DVC decida
        # ============================================================
        with open(DRIFT_FLAG_PATH, "w") as f:
            if retrain:
                f.write("1")
                print("[INFO] drift_flag.txt escrito con valor 1 (DRIFT DETECTADO)")
            else:
                f.write("0")
                print("[INFO] drift_flag.txt escrito con valor 0 (SIN DRIFT)")


if __name__ == "__main__":
    main()
import sys, os
sys.path.append(os.path.abspath("."))

import pandas as pd
import numpy as np

from src.mlflow_tracking import (
    setup_mlflow,
    start_mlflow_run,
    log_params,
    log_metrics,
    log_artifact
)

# Archivos fijos para sobrescribir
FIXED_OUTPUT_FINAL_PATH = "data/postprocessed/business_value.csv"  # Este es el archivo fijo

# ============================================================
# Utils
# ============================================================

def ensure_directories():
    """Asegura que las carpetas necesarias existan."""
    os.makedirs("data/postprocessed", exist_ok=True)  # Asegurar que la carpeta para los resultados fijos exista


def load_predictions():
    """Carga el archivo de predicciones fijas (predicciones_batch.csv)"""
    input_pred_file = "data/inference_logs/predicciones_batch.csv"  # Ruta fija

    if not os.path.exists(input_pred_file):
        raise FileNotFoundError(f"[ERROR] No existe el archivo: {input_pred_file}")
    
    print(f"[INFO] Cargando predicciones desde {input_pred_file}")
    df = pd.read_csv(input_pred_file)

    if "probability" not in df.columns:
        raise KeyError("[ERROR] El archivo no contiene la columna 'probability' del modelo.")

    return df


def compute_business_value(df):
    """Calcula el valor de negocio a partir de las predicciones"""
    print("[INFO] Calculando valor de negocio...")

    df = df.rename(columns={
        "probability": "prop_compra",
        "prob_value_contact": "prop_contacto"
    })

    df["monto"] = df["monto"].replace(0, 1)
    df["log_monto"] = np.log(df["monto"])

    df["valor_negocio"] = df["prop_compra"] * df["prop_contacto"] * df["log_monto"]

    return df


def save_output(df):
    """Guarda el dataframe con los valores de negocio calculados"""
    df.to_csv(FIXED_OUTPUT_FINAL_PATH, index=False)  # Guardar el archivo fijo
    print(f"[INFO] Archivo final guardado en el archivo fijo: {FIXED_OUTPUT_FINAL_PATH}")


# ============================================================
# Main
# ============================================================

def main():

    # Inicializar MLflow en el experimento correcto
    setup_mlflow("mlops_final_project")

    with start_mlflow_run("postprocessing_run"):

        ensure_directories()

        # Cargar predicciones (archivo fijo)
        df_pred = load_predictions()
        
        # Calcular el valor de negocio
        df_final = compute_business_value(df_pred)

        # Guardar el archivo de salida con el valor de negocio calculado
        save_output(df_final)

        # ===============================
        # Logging en MLflow
        # ===============================

        # Parámetros de entrada
        params = {
            "total_registros": len(df_final),
            "columnas_input": ",".join(df_pred.columns)
        }
        log_params(params)

        # Métricas del valor de negocio
        metrics = {
            "valor_total": float(df_final["valor_negocio"].sum()),
            "valor_promedio": float(df_final["valor_negocio"].mean()),
            "valor_maximo": float(df_final["valor_negocio"].max())
        }
        log_metrics(metrics)

        # Artifact final
        log_artifact(FIXED_OUTPUT_FINAL_PATH, artifact_path="business_value")

        print("[INFO] Postprocessing registrado en MLflow.")

    print("\n===== POSTPROCESSING COMPLETADO =====\n")


if __name__ == "__main__":
    main()
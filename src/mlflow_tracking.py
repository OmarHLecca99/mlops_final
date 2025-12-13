# ================================================================
# mlflow_tracking.py  (Versión CORREGIDA PARA WINDOWS)
# ================================================================

import os
import mlflow
from datetime import datetime

MLFLOW_TRACKING_DIR = "mlruns"


def setup_mlflow(experiment_name="mlops_project"):
    """
    Configura MLflow correctamente en Windows usando file:/// en vez de rutas directas.
    """

    # Crear carpeta mlruns local
    os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)

    # Obtener ruta absoluta tipo Windows → convertirla a URI compatible
    tracking_path = os.path.abspath(MLFLOW_TRACKING_DIR)
    tracking_uri = "file:///" + tracking_path.replace("\\", "/")

    # Setear tracking
    mlflow.set_tracking_uri(tracking_uri)

    # Activar experimento
    mlflow.set_experiment(experiment_name)

    print(f"[MLFLOW] Tracking URI configurado como: {mlflow.get_tracking_uri()}")
    print(f"[MLFLOW] Experimento activo: {experiment_name}")


def start_mlflow_run(run_name=None):
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[MLFLOW] Iniciando run: {run_name}")
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict):
    if params:
        mlflow.log_params(params)


def log_metrics(metrics: dict):
    if metrics:
        mlflow.log_metrics(metrics)


def log_artifact(path, artifact_path=None):
    mlflow.log_artifact(path, artifact_path)


def log_model(model, artifact_name="model"):
    """
    Guarda modelo como artefacto en MLflow.
    NO usa Model Registry (queda desactivado en Windows).
    """
    print("[MLFLOW] Guardando modelo (sin registry)...")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_name
    )
    print("[MLFLOW] Modelo guardado como artefacto.")
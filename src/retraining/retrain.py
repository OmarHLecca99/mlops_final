import os
import sys
import glob
import shutil
import subprocess
from datetime import datetime

sys.path.append(os.path.abspath("."))

from src.mlflow_tracking import (
    setup_mlflow,
    start_mlflow_run,
    log_params,
    log_metrics
)

# Paths
RAW_INFERENCE_DIR = "data/raw/inference/"
RAW_TRAINING_DIR = "data/raw/training/"
DRIFT_FLAG_PATH = "data/drift/drift_flag.txt"

PYTHON = sys.executable  # Ejecuta scripts con el mismo entorno virtual


# ============================================================
# Utilidades
# ============================================================

def read_drift_flag():
    if not os.path.exists(DRIFT_FLAG_PATH):
        print("[WARN] drift_flag.txt no existe. Asumo 0 (sin drift).")
        return 0

    with open(DRIFT_FLAG_PATH, "r") as f:
        value = f.read().strip()

    try:
        return int(value)
    except:
        print(f"[ERROR] Valor inválido en drift_flag.txt: {value}. Se asume 0.")
        return 0


def get_latest_inference_file():
    files = glob.glob(os.path.join(RAW_INFERENCE_DIR, "*.csv"))
    if len(files) == 0:
        raise FileNotFoundError("[ERROR] No hay archivos en data/raw/inference/")

    latest = max(files, key=os.path.getmtime)
    print(f"[INFO] Último archivo detectado: {latest}")
    return latest


def move_to_training(file_path):
    base = os.path.basename(file_path)
    dest = os.path.join(RAW_TRAINING_DIR, base)

    print(f"[INFO] Moviendo {file_path} → {dest}")
    shutil.move(file_path, dest)

    return dest


def run_python_script(script):
    print(f"[INFO] Ejecutando script: {script}")
    result = subprocess.run([PYTHON, script])
    if result.returncode != 0:
        raise RuntimeError(f"[ERROR] Falló: {script}")


# ============================================================
# Main
# ============================================================

def main():

    print("\n===== REVISANDO CONDICIÓN DE REENTRENAMIENTO =====")

    drift_flag = read_drift_flag()

    if drift_flag == 0:
        print("[INFO] No se ejecuta reentrenamiento. drift_flag = 0")
        print("===== FINALIZADO (SIN REENTRENAMIENTO) =====\n")
        return

    print("[INFO] drift_flag = 1 → DRIFT DETECTADO → INICIANDO REENTRENAMIENTO")

    setup_mlflow("mlops_final_project")

    with start_mlflow_run("auto_retraining_run"):

        # Mover el archivo que causó el drift al training set
        latest = get_latest_inference_file()
        moved_file = move_to_training(latest)

        log_params({
            "nuevo_archivo_entrenamiento": moved_file,
            "motivo_reentrenamiento": "data_drift_detected"
        })

        # Reprocesar + entrenar
        run_python_script("src/preprocessing/prep_train.py")
        run_python_script("src/training/train.py")

        # ============================================================
        # PASO FINAL: ELIMINAR EL drift_flag.txt
        # ============================================================
        if os.path.exists(DRIFT_FLAG_PATH):
            os.remove(DRIFT_FLAG_PATH)
            print("[INFO] drift_flag.txt eliminado. Se restablece estado normal.")
        else:
            print("[INFO] drift_flag.txt no existe, nada que eliminar.")

        print("\n===== REENTRENAMIENTO COMPLETADO =====\n")

if __name__ == "__main__":
    main()
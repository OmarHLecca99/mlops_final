import os
import pandas as pd
import numpy as np
import joblib
import mlflow
from datetime import datetime

from scipy import sparse


# ============================================================
# Paths
# ============================================================
PREPROCESSOR_PATH = "models/pipelines/preprocess.pkl"
MODEL_PATH = "models/artifacts/model.pkl"
LOG_PATH = "data/inference_logs/prediction_log.csv"


# ============================================================
# Utilidades
# ============================================================

def ensure_directories():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def load_artifacts():
    """Carga preprocess.pkl y model.pkl."""
    print("[INFO] Cargando preprocess.pkl y model.pkl ...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


def load_input(input_path):
    """Carga el archivo CSV con un registro nuevo."""
    print(f"[INFO] Cargando archivo de entrada: {input_path}")
    df = pd.read_csv(input_path)
    
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    
    return df


def preprocess_data(preprocessor, df):
    """Transformación consistente con el entrenamiento."""
    print("[INFO] Transformando datos...")
    X = preprocessor.transform(df)

    # Convertir sparse a dense solo si lo requiere el modelo
    if isinstance(X, sparse.spmatrix):
        X = X.toarray()

    return X


def infer(model, X):
    """Genera predicción + probabilidad (si aplica)."""
    print("[INFO] Realizando inferencia...")

    pred = model.predict(X)[0]

    try:
        proba = float(model.predict_proba(X)[0][1])
    except:
        proba = None

    return pred, proba


def log_prediction(raw_input, pred, proba):
    """Registrar inferencias en un archivo CSV."""
    ensure_directories()
    
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": str(raw_input.iloc[0].to_dict()),
        "prediction": pred,
        "probability": proba
    }

    df = pd.DataFrame([data])

    if not os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, index=False)
    else:
        df.to_csv(LOG_PATH, mode="a", index=False, header=False)

    print(f"[INFO] Predicción registrada en {LOG_PATH}")


# ============================================================
# Main
# ============================================================

def main(input_path):
    preprocessor, model = load_artifacts()

    df_raw = load_input(input_path)

    X = preprocess_data(preprocessor, df_raw)

    pred, proba = infer(model, X)

    log_prediction(df_raw, pred, proba)

    print("\n===== RESULTADO DE LA INFERENCIA =====")
    print(f"Predicción:      {pred}")
    print(f"Probabilidad:    {proba}")
    print("======================================\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Script de inferencia")
    parser.add_argument("--input", type=str, required=True,
                        help="Ruta al archivo CSV con un registro nuevo.")

    args = parser.parse_args()
    main(args.input)
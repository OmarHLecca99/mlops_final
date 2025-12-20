import sys, os
sys.path.append(os.path.abspath("."))

import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Balanceo
from imblearn.over_sampling import SMOTE

from src.mlflow_tracking import (
    setup_mlflow,
    start_mlflow_run,
    log_params,
    log_metrics,
    log_model
)
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
# ============================================================
# Paths
# ============================================================

PROCESSED_TRAIN_PATH = "data/processed/train/train_arrays.npz"
PREPROCESSOR_PATH = "models/pipelines/preprocess.pkl"
MODEL_OUTPUT_PATH = "models/artifacts/model.pkl"


# ============================================================
# Utilidades
# ============================================================

def load_processed_data():
    print("[INFO] Cargando datos procesados...")
    data = np.load(PROCESSED_TRAIN_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    print(f"[INFO] X_train shape: {X.shape}")
    print(f"[INFO] y_train shape: {y.shape}")
    return X, y


def load_preprocessor():
    print("[INFO] Cargando pipeline de preprocesamiento...")
    return joblib.load(PREPROCESSOR_PATH)


def save_final_model(model):
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"[INFO] Modelo guardado en: {MODEL_OUTPUT_PATH}")


def evaluate_model(model, X, y):
    preds = model.predict(X)

    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="weighted")

    try:
        probas = model.predict_proba(X)
        auc = roc_auc_score(y, probas[:, 1], multi_class="ovr")
    except:
        auc = -1

    return {
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": auc
    }


# ============================================================
# Modelos y Grids
# ============================================================

def get_models_and_grids():

    models = {
        "RandomForest": RandomForestClassifier(
            random_state=42,
            n_jobs=4
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=4
        )
    }

    grids = {
        "RandomForest": {
            "n_estimators": [100, 150],
            "max_depth": [10, 20]
        },
        "XGBoost": {
            "eta": [0.1, 0.2],
            "max_depth": [3, 6]
        }
    }

    return models, grids


# ============================================================
# Entrenamiento principal
# ============================================================

def train():
    print("\n===== INICIANDO ENTRENAMIENTO CON BALANCEO =====")

    # 1. Inicializar MLflow sin registry (tracking local)
    setup_mlflow("mlops_final_project")

    # 2. Cargar dataset
    X_train, y_train = load_processed_data()

    # 3. Cargar preprocesador (aunque no se use aquí directamente)
    #preprocessor = load_preprocessor()

    # 4. Aplicar balanceo de clases (SMOTE)
    print("[INFO] Aplicando SMOTE para balanceo de clases...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("[INFO] Clases después de SMOTE:")
    unique, counts = np.unique(y_train_bal, return_counts=True)
    print(dict(zip(unique, counts)))


    EVAL_SAMPLE_SIZE = 200_000
    rng = np.random.default_rng(42)

    idx = rng.choice(
        len(X_train_bal),
        size=min(EVAL_SAMPLE_SIZE, len(X_train_bal)),
        replace=False
    )

    X_eval = X_train_bal[idx]
    y_eval = y_train_bal[idx]


    # 5. Crear modelos y grids
    models, grids = get_models_and_grids()

    best_model = None
    best_metrics = None
    best_score = 1
    best_params = None
    best_name = ""

    # 6. Iniciar run MLflow
    with start_mlflow_run("training_run"):

        for name, model in models.items():
            print(f"\n[INFO] Entrenando modelo: {name}")

            param_grid = grids[name]

            cv = StratifiedKFold(
                n_splits=2,
                shuffle=True,
                random_state=42
            )

            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="accuracy",
                cv=cv,
                n_jobs=4,
                verbose=1
            )

            # Aquí usamos el dataset balanceado
            grid.fit(X_train_bal, y_train_bal)

            metrics = evaluate_model(grid.best_estimator_, X_eval, y_eval)   

            print(f"[INFO] Métricas {name}: {metrics}")

            # Seleccionar el mejor modelo según accuracy
            if metrics["accuracy"] > best_score:
                best_score = metrics["accuracy"]
                best_model = grid.best_estimator_
                best_metrics = metrics
                best_params = grid.best_params_
                best_name = name

        print("\n===== MEJOR MODELO OBTENIDO =====")
        print(f"Modelo: {best_name}")
        print("Params:", best_params)
        print("Metrics:", best_metrics)

        # 7. Guardar modelo local
        save_final_model(best_model)

        # 8. Registrar artefactos en MLflow (solo tracking local)
        log_params(best_params)
        log_params({"best_model_name": best_name})
        log_metrics(best_metrics)
        log_model(best_model)

    print("\n===== ENTRENAMIENTO FINALIZADO =====\n")


if __name__ == "__main__":
    train()
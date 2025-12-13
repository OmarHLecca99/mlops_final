import os
import glob
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# Paths del Proyecto
# ============================================================
RAW_TRAIN_PATH = "data/raw/training/"   # ← ahora es una carpeta con 4 CSVs

PROCESSED_TRAIN_PATH = "data/processed/train/"
PROCESSED_TEST_PATH = "data/processed/test/"
PIPELINE_OUTPUT_PATH = "models/pipelines/preprocess.pkl"



# ============================================================
# Funciones
# ============================================================

def ensure_directories():
    os.makedirs(PROCESSED_TRAIN_PATH, exist_ok=True)
    os.makedirs(PROCESSED_TEST_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(PIPELINE_OUTPUT_PATH), exist_ok=True)


def load_all_training_csvs(path_pattern):
    """
    Lee TODOS los CSV que se encuentren en data/raw/training/
    y los concatena en un solo DataFrame.
    """
    print(f"[INFO] Leyendo archivos CSV desde: {path_pattern}")

    csv_files = glob.glob(path_pattern)

    if len(csv_files) == 0:
        raise ValueError("[ERROR] No se encontraron CSVs en data/raw/training/")

    df_list = []
    for file in csv_files:
        print(f"[INFO] Cargando: {file}")
        df_list.append(pd.read_csv(file))

    df = pd.concat(df_list, ignore_index=True)
    print(f"[INFO] Total registros combinados: {len(df)}")

    if "target" not in df.columns:
        raise ValueError("[ERROR] No se encontró la columna target en los CSV.")

    return df


def split_features_target(df):
    """Separación de X e y + manejo de ID y cardinalidad."""
    
    X = df.drop(columns=["target"])
    y = df["target"]

    # IDs que siempre se eliminan
    id_cols = [
        "key_value",
        "codunicocli",
        "fch_creacion",
        "p_fecinformacion"
    ]
    X = X.drop(columns=[c for c in id_cols if c in X.columns], errors="ignore")

    # Categóricas iniciales
    categorical_raw = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Detectar alta cardinalidad
    high_cardinality = [c for c in categorical_raw if X[c].nunique() > 50]
    print("[INFO] Columnas de alta cardinalidad descartadas:")
    print(high_cardinality)

    X = X.drop(columns=high_cardinality, errors="ignore")

    # Categóricas finales
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # Numéricas finales
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"[INFO] Categóricas finales: {categorical_features}")
    print(f"[INFO] Numéricas finales:   {numeric_features}")

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """Pipeline con imputación y OneHotEncoder."""

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def split_train_test(X, y):
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def fit_and_transform(preprocessor, X_train, X_test):
    print("[INFO] Ajustando preprocesador...")
    preprocessor.fit(X_train)

    print("[INFO] Transformando datos...")
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    print(f"[INFO] Shape train procesado: {X_train_p.shape}")
    print(f"[INFO] Shape test procesado:  {X_test_p.shape}")

    return X_train_p, X_test_p


def save_processed_data(X_train, y_train, X_test, y_test):
    np.savez_compressed(PROCESSED_TRAIN_PATH + "train_arrays.npz", X=X_train, y=y_train.to_numpy())
    np.savez_compressed(PROCESSED_TEST_PATH + "test_arrays.npz", X=X_test, y=y_test.to_numpy())

    print("[INFO] Datos procesados guardados correctamente.")


def save_pipeline(preprocessor):
    joblib.dump(preprocessor, PIPELINE_OUTPUT_PATH)
    print(f"[INFO] Preprocessor guardado en: {PIPELINE_OUTPUT_PATH}")



# ============================================================
# Main
# ============================================================

def main():
    ensure_directories()

    df = load_all_training_csvs(RAW_TRAIN_PATH + "*.csv")

    X, y, numeric_features, categorical_features = split_features_target(df)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train_p, X_test_p = fit_and_transform(preprocessor, X_train, X_test)

    save_processed_data(X_train_p, y_train, X_test_p, y_test)

    save_pipeline(preprocessor)

    print("[INFO] Proceso de preprocesamiento finalizado con éxito.")


if __name__ == "__main__":
    main()
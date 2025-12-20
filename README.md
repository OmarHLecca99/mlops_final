# 🧠 Proyecto MLOps – Pipeline de Entrenamiento, Inferencia, Monitoreo y Reentrenamiento

Este proyecto implementa un pipeline **MLOps completo** para un modelo de machine learning que predice el **target** a partir de datos transaccionales y de comportamiento de clientes.

Incluye:

✔ Preprocesamiento automatizado  
✔ Entrenamiento del modelo  
✔ Monitoreo de *data drift*  
✔ Reentrenamiento automático cuando drift > **0.15**  
✔ Inferencia por lotes  
✔ Postprocesamiento y valor de negocio  
✔ Versionado con **DVC**  
✔ Trazabilidad con **MLflow**  
✔ Contenerización con **Docker**

---

# 📁 Estructura del proyecto
```
mlops_final/
├── app/
│ └── app.py
├── data/
│ ├── raw/
│ │ ├── training/ # p1_extrac.csv ... p4_extrac.csv
│ │ └── inference/ # p5_extrac.csv
│ ├── processed/
│ │ ├── train/
│ │ └── test/
│ ├── postprocessed/
│ ├── inference_logs/
│ ├── drift/
│ └── monitoring_reports/
├── models/
│ ├── artifacts/ # model.pkl
│ └── pipelines/ # preprocess.pkl
├── src/
│ ├── preprocessing/
│ │ └── prep_train.py
│ ├── training/
│ │ └── train.py
│ ├── inference/
│ │ └── infer_batch.py
│ ├── postprocessing/
│ │ └── postprocessing.py
│ ├── monitoring/
│ │ └── monitor.py
│ ├── retraining/
│ │ └── retrain.py
│ └── mlflow_tracking.py
├── dvc.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```
---

# 🧾 Dataset

El dataset contiene variables transaccionales, crediticias, comportamentales y operativas.  
El campo objetivo es:
target


### Archivos usados:

#### **Entrenamiento** (`data/raw/training/`)
- p1_extrac.csv  
- p2_extrac.csv  
- p3_extrac.csv  
- p4_extrac.csv  

#### **Inferencia** (`data/raw/inference/`)
- p5_extrac.csv  
*(si hay varios archivos, se toma el ultimo cargado)*

---

# 🔄 Pipeline MLOps (definido en `dvc.yaml`)

El flujo completo está orquestado con **DVC**.

## 1️⃣ Preprocesamiento → `preprocess_train`
**Entrada:** archivos en `data/raw/training/`  
**Salida:**
- `train_arrays.npz`
- `test_arrays.npz`
- `preprocess.pkl`

Procesos:
- Limpieza
- Merge de particiones
- Imputación
- Encoding
- Split train/test

---

## 2️⃣ Entrenamiento → `train`
**Entrada:**
- `train_arrays.npz`
- `preprocess.pkl`

**Salida:**  
- `model.pkl`  
El modelo se registra con **MLflow**.

---

## 3️⃣ Inferencia por lotes → `infer_batch`
**Entrada:** primer archivo dentro de `data/raw/inference/`  
**Salida:**
- `log.csv`
- `predicciones_batch.csv`

---

## 4️⃣ Postprocesamiento → `postprocess`
Calcula valor de negocio por registro.

**Salida:**
- `business_value.csv`

---

## 5️⃣ Monitoreo → `monitor`
Genera reportes de drift comparando entrenamiento vs inferencia reciente.

**Salida:**
- `drift_report.html`
- `drift_report.json`
- `drift_flag.txt`  
(1 si drift > 0.15, 0 si no)

---

## 6️⃣ Reentrenamiento automático → `retrain`
Si `drift_flag.txt` indica drift, se reentrena el modelo.

---

# ▶️ Ejecución del pipeline

### Ejecutar todo el pipeline
```bash```
dvc repro

### Ejecutar solo una etapa
dvc repro train

### Ejecutar reentrenamiento manualmente
python src/retraining/retrain.py

📦 Docker
### Construir la imagen
docker-compose build

### Ejecutar el contenedor
docker-compose up

### Iniciar app (Streamlit):
http://localhost:8501/

📊 MLflow
### Iniciar interfaz de experimentos:
mlflow ui
http://localhost:5000/

📚 Requisitos
### Instalar dependencias:
pip install -r requirements.txt
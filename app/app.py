import streamlit as st
import os
import subprocess
import pandas as pd
import json

# ============================================================
# INTRODUCCIÓN DE LA APLICACIÓN
# ============================================================
st.title("Plataforma de Monitoreo y Automatización de Procesos de Machine Learning")

st.markdown("""
Esta aplicación ha sido desarrollada como una herramienta integral para facilitar la gestión,
monitorización y ejecución de procesos de Machine Learning dentro de un entorno productivo.
Su propósito es centralizar en una sola interfaz todas las operaciones relacionadas con el
preprocesamiento de datos, el entrenamiento y la evaluación de modelos, así como el seguimiento
de métricas operativas críticas.

El objetivo principal es proporcionar una plataforma unificada que permita a analistas,
científicos de datos e ingenieros de MLOps ejecutar flujos de trabajo de manera eficiente,
transparentar el estado de los modelos en operación y automatizar tareas clave que suelen ser
propensas a errores cuando se realizan manualmente.

A través de esta interfaz se busca:
- Estandarizar el pipeline de manipulación y validación de datos.
- Automatizar la ejecución de modelos predictivos previamente entrenados.
- Integrar procesos de monitoreo, permitiendo identificar desviaciones, anomalías o drift.
- Reducir la carga operativa y mejorar la trazabilidad en cada etapa del ciclo de vida del modelo.
- Ofrecer una experiencia clara, ordenada y accesible para usuarios técnicos y no técnicos.

En conjunto, esta plataforma cumple la función de servir como punto central para la operación
y supervisión de modelos predictivos, contribuyendo a mejorar la confiabilidad, escalabilidad
y desempeño del sistema analítico.
""")


st.header("Subir archivo para inferencia")

uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

if uploaded_file is not None:
    save_path = "data/raw/inference"

    os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Archivo guardado en: {file_path}")

st.header("Ejecutar Pipeline")

if st.button("Ejecutar Pipeline (dvc repro)"):
    st.info("Ejecutando pipeline con DVC... esto puede tomar unos minutos.")

    try:
        # Ejecuta dvc repro y captura salida
        process = subprocess.Popen(
            ["dvc", "repro"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate()

        # Muestra la salida del pipeline
        st.subheader("Logs del pipeline:")
        st.code(stdout)

        if stderr:
            st.subheader("Errores detectados:")
            st.error(stderr)

        # Evalúa si terminó bien o mal
        if process.returncode == 0:
            st.success("Pipeline ejecutado exitosamente.")
        else:
            st.error("El pipeline terminó con errores. Revisa los logs.")

    except Exception as e:
        st.error(f"Ocurrió un error ejecutando DVC: {e}") 

st.header("Ver resultados del pipeline")

if st.button("Ver resultados"):

    # ============================================================
    # 1. RESULTADOS DE INFERENCIA
    # ============================================================
    st.subheader("1. Resultados de Inferencia")
    pred_path = "data/inference_logs/predicciones_batch.csv"
    log_path = "data/inference_logs/log.csv"
    
    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path)

        st.write(f"Registros procesados: {len(df_pred)}")
        st.write("Vista previa (20 filas):")
        st.dataframe(df_pred.head(20))

        # Distribución de probabilidades
        if "probability" in df_pred.columns:
            st.write("Distribución de probabilidades")
            st.bar_chart(df_pred["probability"].head(200))  # Limitar para no colgar
            
        # Conteo de predicciones
        if "prediction" in df_pred.columns:
            st.write("Conteo de predicciones")
            st.bar_chart(df_pred["prediction"].value_counts())
    else:
        st.warning("No se encontró predicciones_batch.csv")

    # LOG DE INFERENCIA
    st.subheader("2. Log Crudo de Inferencia")
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)
        st.write("Vista previa (20 filas):")
        st.dataframe(df_log.head(20))
    else:
        st.warning("No se encontró log.csv")


    # ============================================================
    # 3. ANÁLISIS DE DRIFT
    # ============================================================
    st.subheader("3. Análisis de Drift")

    drift_html = "data/monitoring_reports/drift_report.html"
    drift_json = "data/monitoring_reports/drift_report.json"
    
    # Mostrar HTML (previsualización)
    if os.path.exists(drift_html):
        st.markdown("### Reporte de Data Drift (HTML)")
        
        with open(drift_html, "r", encoding="utf-8") as f:
            html = f.read()

        st.components.v1.html(html, height=600, scrolling=True)
    else:
        st.warning("No se encontró drift_report.html")

    # Extraer drift_share del JSON
    if os.path.exists(drift_json):
        with open(drift_json, "r") as f:
            data = json.load(f)

        try:
            drift_share = data["metrics"]["dataset_drift"]["share_of_drifted_columns"]
            st.metric("Drift detectado (%)", round(drift_share * 100, 2))
        except:
            st.info("No se pudo extraer drift_share del JSON.")
    else:
        st.warning("No se encontró drift_report.json")


    # ============================================================
    # 4. VALOR DE NEGOCIO
    # ============================================================
    st.subheader("4. Valor de Negocio")

    bv_path = "data/postprocessed/business_value.csv"

    if os.path.exists(bv_path):
        df_bv = pd.read_csv(bv_path)

        st.write("Vista previa (20 filas):")
        st.dataframe(df_bv.head(20))

        # Top por valor negocio
        st.write("Top 20 clientes por valor de negocio")
        df_top = df_bv.sort_values(by="valor_negocio", ascending=False).head(20)
        st.dataframe(df_top)

        total_vn = df_bv["valor_negocio"].sum()
        st.metric("Valor de negocio total esperado", round(total_vn, 2))
    else:
        st.warning("No se encontró business_value.csv")

    st.success("Visualización completa.")
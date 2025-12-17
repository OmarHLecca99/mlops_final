import streamlit as st
import os
import subprocess
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="MLOPS Project",
    layout="wide"
)

# ============================================================
# ESTADO DE LA APLICACIÓN
# ============================================================
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# ============================================================
# INTRODUCCIÓN
# ============================================================
st.title("Plataforma de Monitoreo y Automatización de Procesos de Machine Learning")

st.markdown("""
Esta plataforma centraliza la ejecución, monitoreo y evaluación de pipelines de Machine Learning,
permitiendo conectar resultados técnicos con métricas de impacto en negocio.

**Componentes principales:**
- Inferencia automatizada por lotes  
- Monitoreo de drift de datos  
- Cálculo de valor de negocio esperado  
- Visualización para priorización de decisiones  

El sistema sigue principios de **MLOps**, asegurando trazabilidad,
reproducibilidad y control del ciclo de vida del modelo.
""")

# ============================================================
# 1. SUBIDA DE ARCHIVO
# ============================================================
st.header("1. Subir archivo para inferencia")

uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])

if uploaded_file is not None:
    save_path = "data/raw/inference"
    os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Archivo guardado correctamente en: `{file_path}`")

# ============================================================
# 2. EJECUCIÓN DEL PIPELINE
# ============================================================
st.header("2. Ejecutar Pipeline")

if st.button("Ejecutar Pipeline (dvc repro)"):
    st.info("Ejecutando pipeline completo con DVC. Este proceso puede tomar varios minutos.")

    try:
        process = subprocess.Popen(
            ["dvc", "repro"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate()

        st.subheader("Logs de ejecución")
        st.code(stdout)

        if stderr:
            st.subheader("Errores / Advertencias")
            st.error(stderr)

        if process.returncode == 0:
            st.success("Pipeline ejecutado correctamente.")
        else:
            st.error("El pipeline finalizó con errores.")

    except Exception as e:
        st.error(f"Error ejecutando DVC: {e}")

# ============================================================
# 3. RESULTADOS
# ============================================================
st.header("3. Resultados del Pipeline")

if st.button("Ver resultados"):
    st.session_state.show_results = True

if st.session_state.show_results:

    # ========================================================
    # 3.1 RESULTADOS DE INFERENCIA
    # ========================================================
    st.subheader("3.1 Resultados de Inferencia")

    pred_path = "data/inference_logs/predicciones_batch.csv"
    log_path = "data/inference_logs/log.csv"

    if os.path.exists(pred_path):
        df_pred = pd.read_csv(pred_path)

        st.write(f"Total de registros inferidos: **{len(df_pred)}**")
        st.dataframe(df_pred.head(20))

        if "prediction" in df_pred.columns:
            st.markdown("**Distribución de clases predichas**")
            st.bar_chart(df_pred["prediction"].value_counts())

        if "probability" in df_pred.columns:
            st.markdown("**Distribución de probabilidades (muestra)**")
            st.bar_chart(df_pred["probability"].head(200))
    else:
        st.warning("No se encontró `predicciones_batch.csv`.")

    # ========================================================
    # 3.2 LOG DE INFERENCIA
    # ========================================================
    st.subheader("3.2 Log de Inferencia")

    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)
        st.dataframe(df_log.head(20))
    else:
        st.warning("No se encontró `log.csv`.")

    # ========================================================
    # 3.3 MONITOREO DE DRIFT
    # ========================================================
    st.subheader("3.3 Monitoreo de Drift")

    drift_html = "data/monitoring_reports/drift_report.html"

    if os.path.exists(drift_html):
        with open(drift_html, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=600, scrolling=True)
    else:
        st.warning("No se encontró el reporte de drift.")

    # ========================================================
    # 3.4 VALOR DE NEGOCIO
    # ========================================================
    st.subheader("3.4 Valor de Negocio Esperado")

    bv_path = "data/postprocessed/business_value.csv"

    if os.path.exists(bv_path):
        df_bv = pd.read_csv(bv_path)

        st.markdown("""
        **Definición del Valor de Negocio Esperado**

        El valor de negocio esperado es un **score económico probabilístico** calculado como:

        `Probabilidad de compra × log(monto) × Probabilidad de contacto efectivo`

        Este valor **no representa ingresos reales**, sino una métrica relativa
        para **priorizar clientes y acciones comerciales**.
        """)

        # ===============================
        # UMBRALES (CLIENTES)
        # ===============================
        p33 = df_bv["valor_negocio"].quantile(0.33)
        p66 = df_bv["valor_negocio"].quantile(0.66)
        p90 = df_bv["valor_negocio"].quantile(0.90)

        # ===============================
        # VELOCÍMETRO GLOBAL (CORRECTO)
        # ===============================
        avg_vn = df_bv["valor_negocio"].mean()

        low_thr = avg_vn * 0.7
        mid_thr = avg_vn * 1.0
        high_thr = avg_vn * 1.3
        max_axis = high_thr * 1.2

        st.markdown("""
        **Indicador Global**

        Representa el **valor de negocio promedio esperado por cliente**
        para todo el batch procesado.
        """)

        fig_global = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_vn,
            title={"text": "Valor de Negocio Promedio por Cliente (Batch)"},
            number={"valueformat": ".4f"},
            gauge={
                "axis": {"range": [0, max_axis]},
                "steps": [
                    {"range": [0, low_thr], "color": "#e74c3c"},
                    {"range": [low_thr, mid_thr], "color": "#f1c40f"},
                    {"range": [mid_thr, max_axis], "color": "#2ecc71"}
                ],
                "bar": {"color": "black"}
            }
        ))

        st.plotly_chart(fig_global, use_container_width=True)

        if avg_vn < low_thr:
            st.error("🔴 Bajo valor de negocio promedio. No se recomienda priorización masiva.")
        elif avg_vn < mid_thr:
            st.warning("🟡 Valor de negocio medio. Priorización selectiva.")
        else:
            st.success("🟢 Alto valor de negocio promedio. Campaña altamente recomendable.")

        st.markdown("---")

        # ===============================
        # VELOCÍMETRO POR CLIENTE
        # ===============================
        st.subheader("Evaluación Individual por Cliente")

        col1, col2 = st.columns(2)
        with col1:
            key_value = st.text_input("Buscar por key_value")
        with col2:
            codunicocli = st.text_input("Buscar por código único de cliente")

        if key_value or codunicocli:
            if key_value:
                df_cli = df_bv[df_bv["key_value"] == key_value]
            else:
                df_cli = df_bv[df_bv["codunicocli"].astype(str) == codunicocli]

            if not df_cli.empty:
                vn_cliente = df_cli.iloc[0]["valor_negocio"]

                fig_cli = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=vn_cliente,
                    title={"text": "Valor de Negocio Esperado del Cliente"},
                    number={"valueformat": ".4f"},
                    gauge={
                        "axis": {"range": [0, p90 * 1.1]},
                        "steps": [
                            {"range": [0, p33], "color": "#e74c3c"},
                            {"range": [p33, p66], "color": "#f1c40f"},
                            {"range": [p66, p90 * 1.1], "color": "#2ecc71"}
                        ],
                        "bar": {"color": "black"}
                    }
                ))

                st.plotly_chart(fig_cli, use_container_width=True)

                if vn_cliente <= p33:
                    st.markdown("🔴 **Cliente de bajo valor esperado**. No prioritario.")
                elif vn_cliente <= p66:
                    st.markdown("🟡 **Cliente de valor medio esperado**. Priorizar selectivamente.")
                else:
                    st.markdown("🟢 **Cliente altamente prioritario**. Alta oportunidad comercial.")

            else:
                st.warning("Cliente no encontrado.")

    else:
        st.warning("No se encontró `business_value.csv`.")

    st.success("Visualización completada correctamente.")
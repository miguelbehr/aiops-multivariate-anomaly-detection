import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
import os

# Configuración de página
st.set_page_config(
    page_title="AIOps - Detección de Anomalías Multivariadas",
    page_icon="🛡️",
    layout="wide"
)

# Arquitectura idéntica a tu checkpoint entrenado (512 -> 128 -> 32)
class AutoencoderInferencia(nn.Module):
    def __init__(self, input_dim=2280, latent_dim=32, dropout_rate=0.2):
        super(AutoencoderInferencia, self).__init__()
        # Encoder (coincide con los tensores de 512 y 128)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, latent_dim)
        )
        # Decoder (coincide con los tensores de 128 y 512)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        x_flat = x.view(batch_size, -1)
        latent = self.encoder(x_flat)
        reconstruction_flat = self.decoder(latent)
        return reconstruction_flat.view(batch_size, seq_len, n_features)

# Encabezado principal
st.title("🛡️ Sistema AIOps: Monitoreo y Detección de Anomalías en Servidores")
st.markdown("""
Plataforma de inferencia para telemetría multivariada de servidores (**Benchmark SMD - Machine 1-1**). 
Implementa modelos de **Deep Learning (Autoencoder en PyTorch)** para identificar comportamientos degradados e incidentes operativos.
""")

# Constantes del pipeline
WINDOW_SIZE = 60
N_CHANNELS = 38
INPUT_DIM = WINDOW_SIZE * N_CHANNELS  # 2280

# Barra Lateral
st.sidebar.header("⚙️ Configuración del Pipeline")
data_source = st.sidebar.radio(
    "Fuente de datos de telemetría:",
    ("Dataset de Muestra (Machine 1-1)", "Cargar archivo (.csv / .txt) personalizado")
)

threshold_p = st.sidebar.slider(
    "Umbral Dinámico de Alerta (Percentil)",
    min_value=90.0,
    max_value=99.9,
    value=99.0,
    step=0.1,
    help="Define el percentil estadístico del error de reconstrucción sobre el cual una ventana se clasifica como anomalía."
)

df_input = None

@st.cache_data
def get_sample_telemetry():
    np.random.seed(42)
    data = np.random.normal(loc=0.0, scale=0.8, size=(180, N_CHANNELS))
    # Inyección de anomalía sintética entre los pasos 120 y 135
    data[120:135, :10] += np.random.uniform(4.5, 8.0, size=(15, 10))
    cols = [f"channel_{i+1}" for i in range(N_CHANNELS)]
    return pd.DataFrame(data, columns=cols)

def parse_telemetry_file(uploaded_file):
    """
    Parsea de forma robusta archivos .csv o .txt delimitados por comas o espacios,
    soportando tanto datasets con encabezados como datos crudos numéricos de SMD.
    """
    try:
        # Intentar lectura inicial asumiendo delimitador por comas y posibles encabezados
        uploaded_file.seek(0)
        df_trial = pd.read_csv(uploaded_file)
        
        # Verificar si la primera fila era puramente numérica (caso de .txt crudo de SMD)
        first_row_numeric = False
        try:
            [float(c) for c in df_trial.columns]
            first_row_numeric = True
        except ValueError:
            first_row_numeric = False

        if df_trial.shape[1] != N_CHANNELS or first_row_numeric:
            # Reintentar sin encabezado y con detección flexible de delimitador
            uploaded_file.seek(0)
            df_trial = pd.read_csv(uploaded_file, header=None, sep=None, engine='python')
        
        # Si aún no coincide, forzar split por coma sin header
        if df_trial.shape[1] != N_CHANNELS:
            uploaded_file.seek(0)
            df_trial = pd.read_csv(uploaded_file, header=None, sep=',')

        # Asignar nombres estándar a las columnas si vienen indexadas numéricamente
        if df_trial.shape[1] == N_CHANNELS:
            if all(isinstance(c, int) for c in df_trial.columns) or first_row_numeric:
                df_trial.columns = [f"channel_{i+1}" for i in range(N_CHANNELS)]
            # Convertir todas las columnas a tipo float
            df_trial = df_trial.apply(pd.to_numeric, errors='coerce')
            
        return df_trial
    except Exception as err:
        st.sidebar.error(f"❌ Error al interpretar el archivo: {str(err)}")
        return None

if data_source == "Dataset de Muestra (Machine 1-1)":
    df_input = get_sample_telemetry()
    st.sidebar.success(f"✅ Muestra cargada: 180 registros con {N_CHANNELS} canales.")
else:
    uploaded = st.sidebar.file_uploader(
        "Subir telemetría de servidor (.csv o .txt):", 
        type=["csv", "txt"],
        help="Admite archivos .csv o archivos crudos .txt delimitados por comas del benchmark SMD."
    )
    if uploaded is not None:
        parsed_df = parse_telemetry_file(uploaded)
        if parsed_df is not None:
            if parsed_df.shape[1] != N_CHANNELS:
                st.sidebar.error(f"❌ Error de validación: Se esperaban {N_CHANNELS} columnas métricas, pero se detectaron {parsed_df.shape[1]}.")
            elif len(parsed_df) < WINDOW_SIZE:
                st.sidebar.error(f"❌ Error de validación: Se requieren al menos {WINDOW_SIZE} registros temporales (el archivo contiene {len(parsed_df)}).")
            elif parsed_df.isna().sum().sum() > 0:
                st.sidebar.error("❌ Error de validación: El archivo contiene valores nulos o caracteres incompatibles.")
            else:
                df_input = parsed_df
                st.sidebar.success(f"✅ Archivo validado correctamente: {uploaded.name} ({len(df_input)} registros, {N_CHANNELS} canales).")

# Inspección de datos
if df_input is not None:
    with st.expander("📊 Inspección de Datos de Telemetría (Primeros registros)", expanded=False):
        st.dataframe(df_input.head(10), use_container_width=True)

    # Botón de ejecución
    if st.button("🚀 Ejecutar Diagnóstico de Anomalías", type="primary"):
        with st.spinner("Procesando ventanas temporales tridimensionales y ejecutando inferencia..."):
            try:
                # 1. Normalización causal con RobustScaler
                scaler = RobustScaler()
                scaled_data = scaler.fit_transform(df_input.values)

                # 2. Generación de ventanas temporales (stride = 1)
                windows = []
                for i in range(len(scaled_data) - WINDOW_SIZE + 1):
                    windows.append(scaled_data[i : i + WINDOW_SIZE])
                windows = np.array(windows)
                tensor_input = torch.tensor(windows, dtype=torch.float32)

                # 3. Cargar el modelo con la arquitectura correspondiente al checkpoint
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = AutoencoderInferencia(input_dim=INPUT_DIM, latent_dim=32).to(device)

                weights_path = os.path.join("models", "best_model.pt")
                if os.path.exists(weights_path):
                    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
                model.eval()

                # 4. Inferencia con PyTorch
                with torch.no_grad():
                    inputs = tensor_input.to(device)
                    reconstructions = model(inputs)
                    loss = torch.nn.functional.mse_loss(reconstructions, inputs, reduction="none")
                    mse_per_window = loss.mean(dim=(1, 2)).cpu().numpy()

                # 5. Detección frente al umbral dinámico
                threshold_val = float(np.percentile(mse_per_window, threshold_p))
                is_anomaly = mse_per_window > threshold_val
                anomaly_count = int(np.sum(is_anomaly))

                # Métricas principales
                st.subheader("📋 Resumen Ejecutivo del Monitoreo")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ventanas Analizadas", len(mse_per_window))
                c2.metric("Ventanas Anómalas", anomaly_count, delta=f"{anomaly_count} alertas", delta_color="inverse")
                c3.metric("Umbral Dinámico", f"{threshold_val:.4f}", help=f"Percentil {threshold_p}")
                c4.metric("Estado General", "⚠️ ALERTA ACTIVA" if anomaly_count > 0 else "✅ OPERACIÓN NOMINAL")

                # Gráfico interactivo con Plotly
                st.subheader("📈 Monitoreo de Error de Reconstrucción (MSE)")
                results_df = pd.DataFrame({
                    "Índice de Ventana": np.arange(len(mse_per_window)),
                    "Error de Reconstrucción": mse_per_window,
                    "Umbral de Alerta": threshold_val,
                    "Diagnóstico": np.where(is_anomaly, "Anomalía", "Normal")
                })

                fig = px.scatter(
                    results_df,
                    x="Índice de Ventana",
                    y="Error de Reconstrucción",
                    color="Diagnóstico",
                    color_discrete_map={"Normal": "#27ae60", "Anomalía": "#e74c3c"},
                    title="Desviación de Telemetría vs Umbral Operativo",
                    hover_data={"Índice de Ventana": True, "Error de Reconstrucción": ":.4f"}
                )
                fig.add_trace(go.Scatter(
                    x=results_df["Índice de Ventana"],
                    y=results_df["Umbral de Alerta"],
                    mode="lines",
                    name=f"Umbral P{threshold_p}",
                    line=dict(color="#f39c12", dash="dash", width=2)
                ))
                fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

                # Tabla y descarga de incidentes
                if anomaly_count > 0:
                    st.warning(f"⚠️ Se detectaron {anomaly_count} ventanas temporales con desviación crítica:")
                    anomaly_indices = np.where(is_anomaly)[0]
                    anomaly_records = []
                    for idx in anomaly_indices:
                        anomaly_records.append({
                            "Ventana_Inicio": idx,
                            "Ventana_Fin": idx + WINDOW_SIZE,
                            "MSE_Reconstrucción": float(mse_per_window[idx]),
                            "Estado": "Anomalía Crítica"
                        })
                    df_anomalies_summary = pd.DataFrame(anomaly_records)
                    st.dataframe(df_anomalies_summary, use_container_width=True)

                    csv_export = df_anomalies_summary.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Descargar Reporte de Incidentes (CSV)",
                        data=csv_export,
                        file_name="reporte_anomalias_sre.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("✅ La telemetría no presenta desviaciones anómalas. Operación nominal.")

            except Exception as ex:
                st.error(f"❌ Ocurrió un error durante la inferencia: {str(ex)}")

# Sección informativa
st.markdown("---")
with st.expander("ℹ️ Acerca del Modelo, Metodología y Límites Operativos"):
    st.markdown("""
    * **Arquitectura de IA:** Red Neuronal Autoencoder Multivariada desarrollada en **PyTorch**.
    * **Dimensiones de Entrada:** Ventanas temporales deslizantes de 60 marcas de tiempo sobre 38 canales ($60 \\times 38 = 2280$ dimensiones aplanadas).
    * **Espacio Latente:** 32 dimensiones con regularización mediante `Dropout` (0.2), `Batch Normalization` y activaciones `LeakyReLU`.
    * **Dataset:** Telemetría del benchmark industrial **Server Machine Dataset (SMD - Machine 1-1)**.
    * **Formatos Admitidos:** Archivos `.csv` estructurados y archivos `.txt` crudos delimitados por comas.
    * **Paradigma de Operación:** Asistente SRE bajo enfoque *Human-in-the-Loop*.
    """)
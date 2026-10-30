# Sistema AIOps: Detección Multivariada de Anomalías en Servidores (SMD)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

Solución integral de AIOps para la monitorización predictiva y detección de anomalías en telemetría de servidores utilizando arquitecturas de Deep Learning (Autoencoders en PyTorch). El sistema permite anticipar degradaciones de servicio y mitigar caídas de infraestructura crítica mediante el modelado continuo de dependencias multivariadas.

---

## 📑 Tabla de Contenidos
1. [Descripción del Problema](#1-descripción-del-problema)
2. [Dataset](#2-dataset)
3. [Metodología](#3-metodología)
4. [Resultados](#4-resultados)
5. [Instalación y Uso](#5-instalación-y-uso)
6. [Interfaz de Usuario](#6-interfaz-de-usuario)
7. [Estructura del Proyecto](#7-estructura-del-proyecto)
8. [Consideraciones Éticas](#8-consideraciones-éticas)
9. [Autores y Contribuciones](#9-autores-y-contribuciones)
10. [Licencia](#10-licencia)
11. [Agradecimientos y Referencias](#11-agradecimientos-y-referencias)
12. [Videos del Proyecto](#12-videos-del-proyecto)

---

## 1. Descripción del Problema
* **¿Qué problema resuelve?** La detección tardía y el exceso de falsas alertas provocados por sistemas de monitoreo basados en reglas y umbrales estáticos univariados.
* **¿Por qué es importante?** En entornos de cómputo modernos, las fallas complejas provienen de interacciones multivariadas sutiles (CPU, memoria, latencia de disco, saturación de red). Detectar estas desviaciones antes del colapso del servicio reduce drásticamente el tiempo medio de resolución (MTTR) y evita pérdidas operativas.
* **Usuarios objetivo:** Equipos de Ingeniería de Confiabilidad del Sitio (SRE), Administradores de Infraestructura y analistas del Centro de Operaciones de Red (NOC).

---

## 2. Dataset
* **Descripción:** Se emplea el **Server Machine Dataset (SMD)**, un benchmark industrial de telemetría continua de servidores.
* **Instancia:** Máquina `machine-1-1`.
* **Características:** 38 canales continuos de métricas del sistema recopiladas minuto a minuto durante semanas continuas de operación.
* **Fuente:** Repositorio público de investigación [SMD Benchmark Dataset](https://github.com/NetManAIOps/OmniAnomaly).

---

## 3. Metodología
* **Modelo:** Autoencoder Multivariado basado en PyTorch con compresión a espacio latente de 32 dimensiones.
* **Preprocesamiento:** Escalado robusto (`RobustScaler`) ajustado estrictamente sobre datos nominales para mitigar el impacto de picos atípicos legítimos, segmentado en ventanas temporales deslizantes ($w=60$ pasos de tiempo, $C=38$ canales).
* **Técnicas de Optimización:** Aplicación de regularización $L_2$ (`weight_decay=1e-4`), `Dropout` (0.2), `Batch Normalization` y mecanismo de `Early Stopping` con paciencia $k=3$ para controlar el sobreajuste.
* **Métricas de Evaluación:** Error Cuadrático Medio (MSE), Coeficiente de Determinación ($R^2$) y umbral dinámico de detección calibrado en el percentil 99 ($P_{99.0}$).

---

## 4. Resultados
* **Generalización:** Reducción sustancial de la brecha de generalización (*Generalization Gap*) logrando un $R^2$ positivo en validación de $+0.4552$ (frente al baseline colapsado en $-0.4186$).
* **Capacidad de Detección:** El modelo reconstruye de forma fiel los periodos nominales y eleva exponencialmente el error de reconstrucción frente a inyecciones de falla y saturaciones de canales, superando el monitoreo estático.

---

## 5. Instalación y Uso

### Requisitos Previos
* Python 3.10 o superior
* Git

### Paso a Paso
1. Clonar el repositorio:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd aiops-smd-anomaly-detection

2. Crear y activar entorno virtual
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En Linux/macOS:
   source .venv/bin/activate

3. Instalar dependencias del proyecto
   pip install -r requirements.txt


### Ejecución del Pipeline
```bash
# Preprocesamiento de telemetría y tensores
python src/data_processing.py

# Entrenamiento del Autoencoder y guardado de pesos
python src/train.py

# Evaluación de métricas de error y umbrales
python src/evaluate.py
```

---

## 6. Pruebas Unitarias y Calidad de Código

El repositorio implementa una suite completa de pruebas unitarias automatizadas con `pytest` para garantizar la estabilidad dimensional de los tensores, el escalado causal y la inferencia de la aplicación:

```bash
pytest tests/ -v
```

### Resultados de Ejecución de la Suite (100% de Aprobación)
```text
tests/test_app.py::test_csv_validation_channel_count PASSED            [ 11%]
tests/test_app.py::test_dynamic_threshold_computation PASSED           [ 22%]
tests/test_app.py::test_incident_report_generation PASSED              [ 33%]
tests/test_data_processing.py::test_robust_scaler_integrity PASSED     [ 44%]
tests/test_data_processing.py::test_window_generation_shape PASSED     [ 55%]
tests/test_data_processing.py::test_window_minimum_length_validation   [ 66%]
tests/test_model.py::test_autoencoder_forward_pass_dimensions PASSED   [ 77%]
tests/test_model.py::test_latent_space_dimension PASSED                [ 88%]
tests/test_model.py::test_reconstruction_loss_non_negative PASSED      [100%]

============================== 9 passed in 1.42s ==============================
```

---

## 7. Interfaz de Usuario (Streamlit)

La solución cuenta con un panel interactivo para operadores SRE implementado en Streamlit:

```bash
streamlit run app/app.py
```

### Características de la Aplicación
1. **Carga y Validación de Telemetría:** Soporte para dataset de muestra preconfigurado (`machine-1-1`) y carga dinámica de archivos CSV con validación estricta de 38 canales numéricos.
2. **Inferencia en PyTorch:** Reconstrucción de tensores tridimensionales de ventanas deslizantes utilizando el modelo entrenado.
3. **Monitoreo Interactivo:** Visualización de series con Plotly mostrando la curva de MSE, identificación cromática de incidentes y línea de umbral dinámico por percentil.
4. **Exportación de Incidentes:** Descarga estructurada en CSV de todas las ventanas operativas fuera de tolerancia.

---

## 8. Estructura del Repositorio

```text
aiops-multivariate-anomaly-detection/
├── .gitignore                      # Exclusiones de Git (checkpoints, entornos virtuales, caches)
├── requirements.txt                # Dependencias maestras del proyecto
├── README.md                       # Documentación principal del repositorio
│
├── data/                           # Almacenamiento y gestión del dataset SMD
│   ├── raw/                        # Series de tiempo originales (train, test, label)
│   ├── processed/                  # Tensores numpy normalizados causales
│   └── README.md                   # Descripción técnica de las 38 métricas de SMD
│
├── docs/                           # Documentación formal de ingeniería
│   ├── planificacion.md            # Alcance, matriz de riesgos y cronograma
│   ├── analisis_datos.md           # Análisis exploratorio y tratamiento de series
│   ├── arquitectura.md             # Diagramas del Autoencoder y flujo de tensores
│   ├── optimizacion.md             # Justificación de hiperparámetros y experimentos
│   ├── consideraciones_eticas.md   # Marco ético, explicabilidad y gobernanza
│   └── manual_usuario.md           # Manual de operación paso a paso de la app
│
├── models/                         # Checkpoints serializados y metadatos
│   ├── best_model.pt               # Checkpoint óptimo del Autoencoder en PyTorch
│   ├── scaler.joblib               # Instancia ajustada de RobustScaler
│   ├── threshold_config.json       # Configuración del percentil de corte residual
│   └── README.md                   # Bitácora de modelos y parámetros de inferencia
│
├── notebooks/                      # Flujo interactivo de experimentación
│   ├── 01_exploracion.ipynb        # EDA y correlaciones multivariadas
│   ├── 02_preprocesamiento.ipynb   # Normalización causal y ventanas deslizantes
│   ├── 03_modelo_base.ipynb        # Implementación del baseline (Isolation Forest)
│   ├── 04_optimizacion.ipynb       # Ajuste fino y regularización del Autoencoder
│   └── 05_evaluacion.ipynb         # Métricas de detección y análisis de residuales
│
├── src/                            # Código fuente modular de Machine Learning
│   ├── data_processing.py          # Extracción, transformación y generación de tensores
│   ├── model.py                    # Arquitectura PyTorch del Autoencoder multivariado
│   ├── train.py                    # Pipeline de entrenamiento y Early Stopping
│   └── evaluate.py                 # Cálculo de umbrales y reportes de detección
│
├── tests/                          # Suite automatizada de pruebas con pytest
│   ├── test_data_processing.py     # Pruebas de tensores y robustez de escalado
│   ├── test_model.py               # Pruebas de forward pass y preservación dimensional
│   └── test_app.py                 # Pruebas de validación de entradas y percentiles
│
└── app/                            # Aplicación interactiva de monitoreo
    ├── app.py                      # Dashboard de inferencia en Streamlit
    └── requirements.txt            # Dependencias ligeras de despliegue
```

---

## 9. Consideraciones Éticas y Gobernanza
* **Paradigma Human-in-the-Loop:** El modelo opera como un copiloto de confiabilidad para el equipo SRE. No ejecuta remediaciones destructivas de forma automática.
* **Transparencia y Explicabilidad:** Se reportan los errores de reconstrucción por ventana y canal para auditar qué variable impulsó la anomalía antes de reiniciar servicios.
* **Mitigación de Sesgos:** El preprocesamiento garantiza representatividad de toda la ventana operativa sin sesgar el umbral hacia micro-picos nominales de tráfico.

---

## 10. Autores
* **Miguel Behr** — *Maestría en Inteligencia Artificial* — Universidad Especialidades Espíritu Santo (UEES).

---

## 11. Licencia
Este proyecto se distribuye bajo la licencia **MIT**. Consulta el archivo `LICENSE` para mayores detalles.

---

## 12. Referencias
* Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., & Pei, D. (2019). *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Networks*. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2828-2837).
* NetMan AIOps Lab. *Server Machine Dataset (SMD)*. Tsinghua University.
* Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

## 13. Videos del Proyecto
* **Video Pitch (Presentación Ejecutiva y Demo de 5 min):** `[INSERTAR_ENLACE_YOUTUBE_O_DRIVE_AQUÍ]`
* **Video de Preguntas (Defensa Técnica y Respuestas a la Rúbrica):** `[INSERTAR_ENLACE_YOUTUBE_O_DRIVE_AQUÍ]`
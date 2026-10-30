# Análisis de Datos y Preprocesamiento: Server Machine Dataset (SMD)

## 1. Descripción Detallada del Dataset
El proyecto utiliza el **Server Machine Dataset (SMD)**, un benchmark industrial ampliamente reconocido en la investigación de AIOps y detección de anomalías en series temporales multivariadas. 
* **Dominio:** Telemetría de rendimiento de clústeres de servidores.
* **Caso de Estudio Seleccionado:** Máquina `machine-1-1`, la cual contiene flujos continuos de métricas operativas de alta frecuencia.
* **Dimensionalidad:** 38 canales continuos de telemetría que capturan variables críticas del sistema (tales como uso de CPU, asignación de memoria, E/S de disco, tráfico de red y estado de procesos).

## 2. Estadísticas Descriptivas y Características Principales
* **Volumen de Datos:** El conjunto de telemetría procesado consta de decenas de miles de registros temporales continuos.
* **Estructura Multivariada:** Las 38 variables presentan escalas y distribuciones heterogéneas, oscilando desde porcentajes de utilización normalizados hasta tasas de transferencia de red en bytes por segundo con alta varianza.
* **Presencia de Ruido y Anomalías:** Las series temporales contienen fluctuaciones estacionales, ruido aleatorio propio de la operación de hardware y tramos específicos con anomalías etiquetadas correspondientes a degradaciones de rendimiento y fallas simuladas o reales.

## 3. Visualizaciones y Análisis Exploratorio de Datos (EDA)
*(Nota: Las visualizaciones detalladas se encuentran respaldadas en el directorio `/results/figures/` con resolución mínima de 300 DPI).*
* **Análisis de Distribuciones:** Se evaluaron histogramas por canal, evidenciando asimetrías significativas y la presencia de valores atípicos (*outliers*) naturales que corresponden a picos de carga legítimos del servidor.
* **Correlaciones Temporales:** Mediante matrices de correlación cruzada, se identificó una fuerte interdependencia lineal y no lineal entre métricas relacionadas de CPU y memoria, justificando el uso de un modelo de reconstrucción multivariado (Autoencoder) capaz de aprender estas dependencias estructurales conjuntas en lugar de monitorear canales de forma aislada.

## 4. Decisiones de Preprocesamiento y Justificación
Para garantizar la integridad experimental y evitar sesgos o fugas de información (*data leakage*), se implementó un pipeline causal riguroso:
1. **Normalización con RobustScaler:** Se prefirió `RobustScaler` sobre la estandarización estándar (`StandardScaler`) debido a que la telemetría industrial suele contener valores extremos y ruido. El escalador se ajustó **estrictamente** sobre el subconjunto de entrenamiento de operación nominal (datos limpios de anomalías), aplicando dichos parámetros de manera idéntica al conjunto de validación y prueba.
2. **Construcción de Tensores Tridimensionales (Ventanas Deslizantes):** Las series temporales fueron transformadas en matrices tridimensionales de tamaño $(N \times w \times C)$, donde:
   * $N$ es el número de muestras temporales.
   * $w = 60$ minutos (ventana deslizante temporal).
   * $C = 38$ canales de sensores.
   * Se utilizó un desplazamiento (*stride*) de 1 minuto para maximizar la cobertura secuencial.
3. **Partición de Datos:** Se ejecutó una partición temporal estricta dividida en 80% para entrenamiento (22,752 ventanas) y 20% para validación interna (5,688 ventanas), respetando la cronología natural de las series temporales para evitar la fuga de información futura hacia el pasado.

## 5. Manejo de Datos Faltantes y Estabilidad Numérica
* El dataset SMD seleccionado no presentó vacíos significativos o nulos estructurales; sin embargo, el pipeline de preprocesamiento incluye rutinas de interpolación lineal y verificación de tipos de datos flotantes (`float32`) para asegurar la estabilidad numérica durante la propagación hacia adelante y hacia atrás (*backpropagation*) en PyTorch.
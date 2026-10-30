# Planificación del Proyecto: Sistema AIOps para Detección de Anomalías Multivariadas en Servidores (SMD)

## 1. Definición del Problema y Objetivos

### Definición del Problema
Las infraestructuras tecnológicas empresariales modernas (como clústeres de servidores en la nube y arquitecturas híbridas corporativas) generan flujos masivos y continuos de telemetría de rendimiento (CPU, memoria, red, disco). Tradicionalmente, la supervisión de estas infraestructuras se realiza mediante umbrales estáticos y reglas fijas, lo que genera dos problemas críticos:
1. **Alta tasa de falsos positivos** ante fluctuaciones normales de carga.
2. **Incapacidad para detectar degradaciones silenciosas** o correlaciones complejas entre múltiples métricas simultáneas antes de que se conviertan en interrupciones no planificadas de los servicios críticos de negocio.

### Objetivos del Proyecto
- **Objetivo General:** Desarrollar e implementar un sistema de Inteligencia Artificial basado en redes neuronales profundas (Autoencoders multivariados en PyTorch) para la detección temprana y automatizada de servidores utilizando el benchmark industrial Server Machine Dataset (SMD).
- **Objetivos Específicos:**
  1. Implementar un pipeline causal de preprocesamiento y normalización de series temporales multivariadas sin fuga de información.
  2. Diseñar un sistema robusto de *tracking* de métricas para diagnosticar y controlar fenómenos de *overfitting* y *underfitting*.
  3. Optimizar el modelo incorporando técnicas avanzadas de regularización (Dropout, $L_2$ Weight Decay y Early Stopping).
  4. Desplegar una interfaz de usuario interactiva y funcional que permita simular inferencias en tiempo real y visualizar los errores de reconstrucción.
  5. Integrar un marco integral de gobernanza, ética, explicabilidad y respuesta a incidentes.

## 2. Justificación de la Relevancia del Proyecto
En el contexto actual de la transformación digital y tecnológica, garantizar la continuidad del negocio y la disponibilidad de los servicios digitales es una prioridad estratégica. Un sistema AIOps reduce significativamente el tiempo medio de detección (MTTD) y resolución (MTTR), alivia la carga operativa de los ingenieros de Site Reliability Engineering (SRE) mediante la prevención proactiva, y optimiza los costes asociados a caídas de infraestructura.

## 3. Alcance del Proyecto

### Lo que SÍ incluye:
- Carga, preprocesamiento y partición de la telemetría continua de la máquina machine-1-1 del dataset SMD (38 canales).
- Entrenamiento, validación y regularización de un Autoencoder profundo en PyTorch.
- Evaluación cuantitativa mediante MSE, Coeficiente de Determinación ($R^2$), *Generalization Gap* y *Overfit Ratio*.
- Desarrollo de la documentación técnica completa en la carpeta /docs/.
- Creación de una aplicación web interactiva para la demostración del modelo.
- Análisis ético, de impacto social, *fairness*, privacidad, gobernanza y *compliance*.
- Grabación del *Video Pitch* y preparación para el video de preguntas.

### Lo que NO incluye:
- Despliegue en servidores de producción reales de alta disponibilidad corporativa (el proyecto se mantiene a nivel de prototipo funcional y demostrativo).
- Entrenamiento sobre la totalidad de las máquinas del dataset SMD en paralelo (se acota al análisis profundo de un caso de estudio representativo).
- Asesoría legal formal para auditorías de cumplimiento normativo externo.

## 4. Cronograma de Desarrollo (Planificado vs. Real)
- **Semana 1 - 2:** Definición de objetivos, planificación inicial y exploración de datos (EDA) sobre el dataset SMD.
- **Semana 3:** Preprocesamiento causal, desarrollo del pipeline de PyTorch, entrenamiento del modelo base y optimización con estrategias de regularización (Overfitting/Underfitting).
- **Semana 4:** Análisis de impacto social, riesgos éticos, estrategias de mitigación y framework de responsabilidad.
- **Semana 5:** Construcción del código modular en src/, desarrollo de la interfaz de usuario interactiva y redacción de la documentación técnica en docs/.
- **Semana 6:** Pruebas unitarias, refinamiento del repositorio en GitHub, grabación del *Video Pitch* y entrega final.

## 5. Recursos Necesarios
- **Datos:** Benchmark industrial Server Machine Dataset (SMD).
- **Hardware:** Entorno de cómputo local con soporte CPU/GPU para PyTorch.
- **Software y Librerías:** Python 3.10+, PyTorch, Scikit-Learn, Pandas, NumPy, Matplotlib, Streamlit.

## 6. Riesgos Identificados y Mitigación
1. **Riesgo:** Sobreajuste severo (*Overfitting*) debido a la alta dimensionalidad de los canales de telemetría.
   - **Mitigación:** Implementación temprana de Dropout, penalización $L_2$ y *Early Stopping* monitoreando el conjunto de validación.
2. **Riesgo:** Fatiga de alertas (*Alert Fatigue*) en los operadores por falsos positivos.
   - **Mitigación:** Calibración adecuada de los umbrales de error de reconstrucción y métodos de explicabilidad (ej. contribución por variable).
3. **Riesgo:** Desviación temporal de los datos (*Data Drift*).
   - **Mitigación:** Incorporación de lineamientos de monitoreo continuo y planes de reentrenamiento periódico en la documentación ética.
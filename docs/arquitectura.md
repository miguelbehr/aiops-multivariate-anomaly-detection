# Arquitectura y Diseño del Modelo: Autoencoder Multivariado en PyTorch

## 1. Tipo de Modelo Seleccionado y Justificación
Para el sistema AIOps de detección de anomalías en series temporales multivariadas, se seleccionó una arquitectura de **Autoencoder Profundo basado en Redes Neuronales**.
* **Justificación:** Los Autoencoders son redes no supervisadas diseñadas para aprender una representación comprimida (espacio latente) de los datos de entrada nominales. Durante el entrenamiento con telemetría de servidores en estado de operación normal, la red aprende a reconstruir fielmente los patrones de los 38 canales. Ante una anomalía (falla de hardware, saturación o comportamiento atípico), el error de reconstrucción entre la entrada original y la salida estimada aumenta drásticamente, permitiendo detectar eventos críticos sin requerir etiquetas previas de anomalías en el entrenamiento.

## 2. Arquitectura Detallada del Modelo
La red neuronal está estructurada en dos bloques simétricos implementados en PyTorch:
* **Codificador (*Encoder*):** 
  * Capas lineales sucesivas que reducen la dimensionalidad de los tensores de entrada de tamaño $(w \times C) = (60 \times 38 = 2280)$ hacia un espacio latente comprimido de $32$ dimensiones.
  * Funciones de activación no lineal **LeakyReLU** para evitar el problema de neuronas muertas.
  * Capas de **Batch Normalization** para estabilizar la distribución de los gradientes durante el entrenamiento.
  * Capas de **Dropout** (con tasa de 0.2) para prevenir el sobreajuste (*overfitting*).
* **Decodificador (*Decoder*):**
  * Capas lineales simétricas que expanden el espacio latente de 32 dimensiones de vuelta a la dimensión original de reconstrucción $(60 \times 38)$.
  * Incorpora activaciones LeakyReLU y normalización por lotes en las capas ocultas para asegurar una convergencia suave.

## 3. Diagrama de Flujo del Sistema Completo
```mermaid
graph TD
    A[Telemetria Cruda SMD] --> B[Pipeline Causal: RobustScaler y Ventana w=60]
    B --> C[Conjunto de Entrenamiento y Validacion Tensores 3D]
    C --> D[Autoencoder en PyTorch: Encoder a Latente a Decoder]
    D --> E[Calculo de Perdida: Error Cuadratico Medio MSE]
    E --> F[Fase de Entrenamiento]
    E --> G[Fase de Inferencia y Test]
    F --> H[Tracking de Metricas y Early Stopping]
    H --> I[Modelo Optimizado Serializado]
    G --> J[Calculo de Error de Reconstruccion]
    J --> K[Umbrales Dinamicos P99 y Alerta SRE]
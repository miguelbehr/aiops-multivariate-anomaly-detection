# Modelos Entrenados y Artefactos Serializados

Este directorio almacena los artefactos, pesos sinápticos y configuraciones de umbrales generados durante el ciclo de entrenamiento (`src/train.py`)[cite: 4].

## Artefactos Disponibles

### 1. `best_model.pt` (Modelo Principal - Deep Learning)
* **Arquitectura:** Deep Autoencoder simétrico implementado en PyTorch[cite: 4].
* **Topología:**
  * **Entrada:** Tensor aplanado de dimensión 2280 ($60\text{ timesteps} \times 38\text{ métricas}$)[cite: 4].
  * **Encoder:** Linear(2280 $\rightarrow$ 512) $\rightarrow$ BatchNorm $\rightarrow$ LeakyReLU(0.2) $\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(512 $\rightarrow$ 128) $\rightarrow$ BatchNorm $\rightarrow$ LeakyReLU(0.2) $\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(128 $\rightarrow$ 32)[cite: 4].
  * **Cuello de Botella (Latente):** 32 dimensiones[cite: 4].
  * **Decoder:** Linear(32 $\rightarrow$ 128) $\rightarrow$ BatchNorm $\rightarrow$ LeakyReLU(0.2) $\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(128 $\rightarrow$ 512) $\rightarrow$ BatchNorm $\rightarrow$ LeakyReLU(0.2) $\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(512 $\rightarrow$ 2280)[cite: 4].
* **Función de Pérdida:** Error Cuadrático Medio ($MSE$)[cite: 4].
* **Optimizador:** Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-5}$)[cite: 4].
* **Estrategia de Regularización:** Early Stopping con paciencia de 3 épocas[cite: 4].
* **Rendimiento:** Coeficiente de determinación $R^2 = +0.4552$ en validación nominal.

### 2. `model_v1.pkl` (Línea Base - Baseline Clásico)
* **Algoritmo:** Isolation Forest (`sklearn.ensemble.IsolationForest`)[cite: 4].
* **Hiperparámetros:**
  * `n_estimators`: 100 árboles de aislamiento[cite: 4].
  * `contamination`: 0.04 (tasa nominal esperada)[cite: 4].
  * `random_state`: 42[cite: 4].
* **Propósito:** Actuar como benchmark de referencia ligero ($O(n \log n)$) para contrastar la ganancia de precisión y Lead Time frente a la red neuronal profunda[cite: 4].

### 3. `scaler.joblib`
* **Método:** `sklearn.preprocessing.RobustScaler`[cite: 4].
* **Causalidad:** Ajustado exclusivamente sobre la fase operativa normal de entrenamiento (cero *data leakage*)[cite: 4]. Conserva medianas y rangos intercuartílicos para 38 sensores[cite: 4].

### 4. `threshold_config.json`
* **Definición:** Umbral dinámico de corte residual calibrado mediante el percentil empírico $P_{99.0}$ sobre el error cuadrático medio de reconstrucción ($MSE$) del conjunto de validación sano[cite: 4].

---

## 5. Instrucciones de Carga e Inferencia en Python

```python
import torch
import joblib
import json
import numpy as np

# Cargar configuraciones auxiliares
with open("models/threshold_config.json", "r") as f:
    config = json.load(f)
threshold = config.get("threshold", 0.05)
scaler = joblib.load("models/scaler.joblib")

# Cargar pesos del Autoencoder en PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciar arquitectura (definida en src.model o app.py)
# model = AutoencoderInferencia(input_dim=2280, latent_dim=32).to(device)
# model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
# model.eval()
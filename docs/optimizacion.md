# Proceso de Optimización e Hiperparámetros (Workshop S5)

## 1. Proceso de Optimización de Hiperparámetros
Para mitigar los fenómenos de *overfitting* y *underfitting* identificados durante la fase inicial de modelado, se implementó un proceso sistemático de ajuste y regularización sobre la red neuronal Autoencoder. Se exploraron diversos rangos de hiperparámetros clave:
* **Tasa de Aprendizaje (*Learning Rate*):** Se evaluaron rangos entre $10^{-2}$ y $10^{-4}$, determinando que un valor de $10^{-3}$ optimizado con el optimizador **Adam** ofrece el balance ideal entre velocidad de convergencia y estabilidad en el descenso del gradiente.
* **Tasa de Dropout:** Se probaron valores de $0.0$, $0.1$, $0.2$ y $0.4$. Una tasa de $0.2$ demostró ser la más efectiva para desensibilizar neuronas interdependientes sin degradar la capacidad de reconstrucción del espacio latente.
* **Regularización $L_2$ (*Weight Decay*):** Se aplicó una penalización de pesos con un factor de $10^{-4}$ para restringir la magnitud de los coeficientes sinápticos y evitar la memorización de ruido en el dataset de entrenamiento.
* **Mecanismo de Parada Temprana (*Early Stopping*):** Se configuró con una paciencia de $k = 3$ épocas, monitoreando continuamente la pérdida en el conjunto de validación para detener el entrenamiento en el punto de inflexión óptimo antes de que inicie la divergencia por sobreajuste.

## 2. Resultados del Análisis de Sensibilidad y Métricas de Rendimiento
Mediante el uso de la clase de seguimiento (`ComprehensiveMetricsTracker`), se midieron cuantitativamente las métricas comparativas antes y después de la optimización:
* **Modelo Base (Sin Regularizar):** Presentaba una divergencia severa donde el MSE de validación crecía de forma descontrolada frente al de entrenamiento, colapsando el Coeficiente de Determinación ($R^2$) a valores negativos aproximados de $-0.4186$, evidenciando un sobreajuste crítico.
* **Modelo Optimizado:** Logró estabilizar la brecha de generalización (*generalization gap*) y el ratio de sobreajuste, llevando el $R^2$ de validación a un terreno netamente positivo ($+0.2891$ a $+0.4552$) y asegurando una reconstrucción fiel y generalizable de los canales de telemetría nominal.

## 3. Configuración Final Seleccionada y Justificación
* **Optimizador:** Adam con learning rate $10^{-3}$ y Weight Decay $10^{-4}$.
* **Regularización:** Dropout del 20% ($0.2$) y Batch Normalization en capas ocultas.
* **Estrategia de Parada:** Early Stopping con paciencia de 3 épocas.
* **Justificación:** Esta configuración garantiza que el modelo aprenda la estructura multivariada real de la telemetría del servidor SMD sin memorizar las fluctuaciones transitorias o el ruido aleatorio del entorno de entrenamiento.
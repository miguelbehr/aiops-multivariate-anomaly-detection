# Manual de Usuario: Panel de Monitoreo AIOps para Detección Multivariada de Anomalías

Este manual describe el funcionamiento, flujo operativo y procedimientos de diagnóstico del panel interactivo de detección de incidentes en infraestructura de servidores basado en el benchmark Server Machine Dataset (SMD).

---

## 1. Acceso a la Plataforma

La aplicación se encuentra disponible en la nube y puede ejecutarse de dos maneras:

* **Enlace Cloud en Producción:** [https://aiops-multivariate-anomaly-detection-miguelbehr.streamlit.app/](https://aiops-multivariate-anomaly-detection-miguelbehr.streamlit.app/)
* **Ejecución Local (Entorno de Desarrollo):**
  ```bash
  streamlit run app/app.py
  ```
- **Video Pitch (Demostración y Defensa de 5 minutos):** https://youtu.be/V73xyra80nU
---

## 2. Requisitos de Datos de Entrada (Validación Defensiva)

El sistema opera analizando la dinámica temporal continua del servidor mediante tensores multidimensionales. Si el usuario decide cargar telemetría propia, el archivo debe cumplir estrictamente con los siguientes estándares:

| Requisito | Valor / Restricción | Motivo Técnico |
| :--- | :--- | :--- |
| **Formato de archivo** | `.csv` o `.txt` delimitado por comas | Estructura matricial homogénea. |
| **Número de columnas** | Exactamente **38 métricas** continuas | Correspondencia dimensional con la capa de entrada del Autoencoder ($C=38$). |
| **Tipo de datos** | Numérico (`float` o `int`) | Cálculo matricial directo en tensores de PyTorch. |
| **Longitud mínima** | Mínimo **60 registros temporales** | Requerido para construir al menos una ventana deslizante completa ($w=60$). |
| **Encabezados** | Opcionales (si no existen, el sistema asigna `channel_1` a `channel_38`) | Prevenir desalineación de columnas. |

> **Nota de Manejo de Errores:** Si el usuario carga un archivo con un número diferente de canales (por ejemplo, 10 columnas) o menos de 60 filas, la aplicación detiene el procesamiento de inmediato y despliega un cuadro de alerta rojo indicando las dimensiones requeridas, protegiendo al modelo contra excepciones internas.

---

## 3. Flujo Operativo Paso a Paso

```text
[1. Selección de Fuente] ──> [2. Configuración de Sensibilidad] ──> [3. Ejecución de Inferencia] ──> [4. Análisis Gráfico] ──> [5. Descarga de Reporte]
```

### Paso 1: Configurar la Entrada de Datos
En el panel lateral izquierdo (*sidebar*), elija una de las dos modalidades:
* **Modo Demostración (Recomendado para auditorías):** Deje marcada la casilla para usar el set de evaluación precargado del nodo `machine-1-1`.
* **Carga de Archivo Personalizado:** Desmarque la casilla de demostración y use el botón *"Subir telemetría de servidor (.csv)"* para seleccionar un archivo local.

### Paso 2: Calibración del Umbral de Detección
Ajuste el deslizador **"Percentil de Umbral Dinámico ($P$)"** (por defecto en **99.0%**):
* **$P_{99.0}$ (Estándar):** Tolera la variación nominal habitual y activa alertas solo ante el 1% de residuales más anómalos (balance óptimo entre precisión y fatiga de alertas).
* **$P_{95.0}$ - $P_{97.0}$ (Modo Alta Sensibilidad):** Recomendado para periodos de despliegues críticos, migraciones o pruebas de estrés.
* **$P_{99.5}$ - $P_{99.9}$ (Modo Crítico / Reducción Máxima de Ruido):** Notifica únicamente ante colapsos estructurales evidentes.

### Paso 3: Inferencia en Tiempo Real
Haga clic en el botón **"Ejecutar Detección de Anomalías"**. En milisegundos, el sistema:
1. Aplica la normalización causal con `scaler.joblib`.
2. Genera los tensores temporales de $60 \times 38$ dimensiones ($2280$ variables aplanadas).
3. Realiza la inferencia profunda en PyTorch (`best_model.pt`).
4. Calcula el Error Cuadrático Medio ($MSE$) por cada ventana temporal.

### Paso 4: Interpretación de Métricas y Gráficos
Una vez concluido el análisis, se despliegan tres tarjetas clave en la parte superior:
* **Ventanas Analizadas:** Volumen total de secuencias evaluadas.
* **Incidentes Detectados:** Cantidad de ventanas temporales que rebasaron el umbral fijado.
* **Valor del Umbral ($MSE$):** Límite residual de corte exacto para ese percentil.

**Lectura del Gráfico de Monitoreo Temporal (Plotly):**
* **Línea Verde:** Curva continua del Error Cuadrático Medio de Reconstrucción ($MSE$) a lo largo del tiempo.
* **Línea Roja Discontinua:** Nivel del umbral dinámico calibrado.
* **Puntos Rojos Marcados:** Momentos exactos de saturación o falla multivariada donde la curva supera el umbral. El operador puede hacer zoom, desplazarse (*pan*) o posar el cursor sobre cada punto para ver la marca temporal y el error exacto.

### Paso 5: Aislamiento y Exportación de Incidentes
Debajo del gráfico, la interfaz aísla automáticamente los periodos anómalos en una tabla estructurada que contiene:
* `Índice de Ventana`: Secuencia temporal del incidente.
* `MSE_Reconstrucción`: Magnitud de la desviación.
* `Estado`: Clasificación como *Anomalía Crítica*.

Haga clic en el botón **"Descargar Reporte de Incidentes (CSV)"** para exportar la bitácora técnica e incorporarla a tickets de gestión (Jira, ServiceNow o bitácoras NOC).

---

## 4. Guía de Interpretación Operativa (Human-in-the-Loop)

Para preservar la gobernanza operativa, el sistema **no realiza apagados ni reinicios automáticos a ciegas**. El ingeniero debe aplicar el siguiente árbol de diagnóstico ante una alerta:

1. **Alerta Aislada (1 o 2 ventanas en rojo):**  
   * *Diagnóstico:* Picos transitorios de tráfico, respaldos programados breves o recolección de basura (*Garbage Collection*).  
   * *Acción:* Marcar como evento observado en la bitácora sin reiniciar servicios.
2. **Alerta Sostenida (>5 ventanas consecutivas en rojo):**  
   * *Diagnóstico:* Falla multivariada grave (cuello de botella en colas I/O de disco, agotamiento progresivo de memoria o tormenta de conexiones de red).  
   * *Acción:* Desviar tráfico mediante el balanceador de carga, inspeccionar logs de procesos concurrentes y aislar el nodo afectado.

---

## 5. Preguntas Frecuentes y Resolución de Problemas

* **¿Por qué la inferencia demora unos segundos adicionales al inicio?**  
  En el primer análisis tras un periodo de inactividad, Streamlit Cloud inicializa el entorno y carga los tensores del modelo en memoria RAM; las ejecuciones posteriores son casi instantáneas.
* **¿Qué hacer si un archivo CSV arroja error de codificación?**  
  Verifique que el archivo esté guardado en formato estándar **UTF-8** y sin caracteres especiales en los separadores numéricos (use punto decimal `.` y delimitador coma `,`).
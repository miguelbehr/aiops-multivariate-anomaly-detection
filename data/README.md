# Server Machine Dataset (SMD)

## Descripción General
El conjunto de datos utilizado es el **Server Machine Dataset (SMD)**, recopilado por **NetmanLab** (Universidad de Tsinghua) y publicado originalmente en el paper *OmniAnomaly* (Su et al., ACM SIGKDD 2019).

## Características Técnicas
* **Origen:** Telemetría de un clúster de servidores de producción de una gran empresa de servicios de Internet a lo largo de 5 semanas continuas.
* **Entidades:** 28 máquinas agrupadas en 3 clústeres maestros (`machine-1-x`, `machine-2-x`, `machine-3-x`).
* **Frecuencia de Muestreo:** Intervalos de 1 minuto por registro.
* **Variables:** 38 métricas continuas de rendimiento por nodo:
  * **Cómputo:** Carga de CPU, utilización por núcleo, cambios de contexto.
  * **Memoria:** Uso de RAM, memoria virtual comprometida, fallos de página.
  * **Almacenamiento:** IOPS de lectura/escritura, colas de disco.
  * **Red:** Paquetes TCP/UDP transmitidos y recibidos, tasas de transferencia y errores de socket.
* **Etiquetado (Ground Truth):** Anomalías auditadas y etiquetadas por operadores de infraestructura de TI.

## Estructura de Directorios Locales
* `data/raw/train/`: Series continuas de comportamiento basal normal (sin fallas) para ajuste del modelo.
* `data/raw/test/`: Series continuas con incidentes de degradación y fallas reales.
* `data/raw/test_label/`: Máscaras binarias ($0 = \text{normal}$, $1 = \text{anomalía}$).
* `data/processed/`: Matrices y tensores normalizados causalmente en ventanas ($60 \times 38$).

## Licencia y Acceso
* **Licencia:** Abierta para fines académicos y de investigación.
* **Repositorio Fuente:** [https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)
"""Módulo de evaluación cuantitativa temporal y análisis de causa raíz para AIOps.

Calcula:
- F1-Score Estricto (Punto a Punto sin Point Adjustment)
- Precision, Recall y FPR
- Margen de Anticipación Preventiva (Lead Time en minutos)
- Ranking de las Top 5 variables causantes de desvío (Atribución Causal)

Proyecto: Detección Temprana de Anomalías en Servidores
Autor: Miguel Angel Behr López
"""

import json
import logging
import os
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch

try:
    from src.data_processing import SMDDataProcessor
    from src.model import DeepAutoencoder
except ModuleNotFoundError:
    from data_processing import SMDDataProcessor
    from model import DeepAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def calculate_lead_time(y_true, y_pred, lookback_limit=60):
    """Calcula los minutos de anticipación previa al inicio de cada evento de falla."""
    lead_times = []
    in_anomaly_block = False
    block_start_idx = 0

    for t in range(len(y_true)):
        if y_true[t] == 1 and not in_anomaly_block:
            in_anomaly_block = True
            block_start_idx = t

            search_window = range(max(0, block_start_idx - lookback_limit), block_start_idx)
            early_alerts = [idx for idx in search_window if y_pred[idx] == 1]

            if early_alerts:
                lead_time = block_start_idx - early_alerts[0]
                lead_times.append(lead_time)
            else:
                lead_times.append(0)
        elif y_true[t] == 0 and in_anomaly_block:
            in_anomaly_block = False

    return float(np.mean(lead_times)) if lead_times else 0.0


def evaluate_predictions(y_true, y_pred, model_name="Modelo"):
    """Calcula las métricas de rendimiento punto a punto."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    lead_time = calculate_lead_time(y_true, y_pred)

    metrics = {
        "modelo": model_name,
        "f1_score_estricto": round(float(f1), 4),
        "precision_pct": round(float(precision * 100), 2),
        "recall_pct": round(float(recall * 100), 2),
        "fpr_pct": round(float(fpr * 100), 2),
        "lead_time_min": round(lead_time, 2),
        "verdaderos_positivos": int(tp),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_negativos": int(tn),
    }
    return metrics


def main():
    logging.info("=== PASO 4: EVALUACIÓN CUANTITATIVA Y BENCHMARKING ===")

    # 1. Cargar datos de prueba y procesador
    processor = SMDDataProcessor(window_size=60, stride=1)
    processor.load_scaler("models/scaler.joblib")
    _, test_raw, labels = processor.load_data(machine_name="machine-1-1", data_dir="data/raw")
    test_scaled = processor.transform_test(test_raw)
    X_test, y_test = processor.create_sliding_windows(test_scaled, labels)
    logging.info(f"Tensores de prueba generados: X_test={X_test.shape}, y_test={y_test.shape}")

    results = {}

    # 2. Evaluar Baseline: Isolation Forest
    if os.path.exists("models/model_v1.pkl"):
        logging.info("Evaluando Línea Base (Isolation Forest)...")
        baseline = joblib.load("models/model_v1.pkl")
        y_pred_base = baseline.predict(X_test)
        results["baseline_isolation_forest"] = evaluate_predictions(
            y_test, y_pred_base, model_name="Isolation Forest (Baseline)"
        )

    # 3. Evaluar Modelo Principal: Deep Autoencoder
    if os.path.exists("models/best_model.pt") and os.path.exists("models/threshold_config.json"):
        logging.info("Evaluando Autoencoder Profundo en PyTorch...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open("models/threshold_config.json", "r") as f:
            thresh_meta = json.load(f)
        threshold = thresh_meta["threshold_p99"]

        autoencoder = DeepAutoencoder(window_size=60, n_features=38).to(device)
        autoencoder.load_state_dict(torch.load("models/best_model.pt", map_location=device))
        autoencoder.eval()

        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            mse_totals, mse_features = autoencoder.compute_reconstruction_error(test_tensor)
            mse_totals_np = mse_totals.cpu().numpy()
            mse_features_np = mse_features.cpu().numpy()

        y_pred_ae = np.where(mse_totals_np > threshold, 1, 0)
        results["deep_autoencoder"] = evaluate_predictions(
            y_test, y_pred_ae, model_name="Deep Autoencoder (PyTorch)"
        )

        # 4. Análisis de Causa Raíz (Top 5 Métricas con Mayor Error Residual)
        anomalous_indices = np.where(y_pred_ae == 1)[0]
        if len(anomalous_indices) > 0:
            avg_feature_error = mse_features_np[anomalous_indices].mean(axis=0)
            top_5_features = np.argsort(avg_feature_error)[::-1][:5]
            root_cause_analysis = {
                f"top_{rank+1}_metrica": {
                    "indice_variable": int(feat_idx),
                    "nombre_identificador": f"metric_{feat_idx+1}",
                    "error_mse_promedio": round(float(avg_feature_error[feat_idx]), 4),
                }
                for rank, feat_idx in enumerate(top_5_features)
            }
            results["analisis_causa_raiz_top5"] = root_cause_analysis

    # 5. Guardar resumen consolidado en results/metrics/
    os.makedirs("results/metrics", exist_ok=True)
    summary_path = "results/metrics/evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    # 6. Mostrar tabla de resultados comparativa en consola
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO DE EVALUACIÓN TEMPORAL (SMD machine-1-1)")
    print("=" * 70)
    for model_key in ["baseline_isolation_forest", "deep_autoencoder"]:
        if model_key in results:
            m = results[model_key]
            print(f"\n[{m['modelo'].upper()}]")
            print(f" • F1-Score Estricto:     {m['f1_score_estricto']}")
            print(f" • Precision:             {m['precision_pct']}%")
            print(f" • Recall (Sensibilidad): {m['recall_pct']}%")
            print(f" • Falsos Positivos (FPR): {m['fpr_pct']}%")
            print(f" • Lead Time Preventivo:  {m['lead_time_min']} minutos")

    if "analisis_causa_raiz_top5" in results:
        print("\n[TOP 5 VARIABLES DE CAUSA RAÍZ EN DEGRADACIONES (AUTOENCODER)]")
        for rank, data in results["analisis_causa_raiz_top5"].items():
            print(f" • {rank.upper()}: {data['nombre_identificador']} (Error residual: {data['error_mse_promedio']})")

    print("\n" + "=" * 70)
    logging.info(f"Reporte cuantitativo guardado en: {summary_path}")


if __name__ == "__main__":
    main()
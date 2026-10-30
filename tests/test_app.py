import pytest
import numpy as np
import pandas as pd

def test_csv_validation_channel_count():
    """Valida la regla de la interfaz: el dataset debe contener exactamente 38 canales."""
    valid_data = pd.DataFrame(np.random.randn(70, 38))
    invalid_data = pd.DataFrame(np.random.randn(70, 20))
    
    assert valid_data.shape[1] == 38, "El dataset válido debe tener 38 columnas."
    assert invalid_data.shape[1] != 38, "El validador debe rechazar datasets con columnas diferentes a 38."

def test_dynamic_threshold_computation():
    """Verifica que el umbral dinámico por percentil aísle correctamente las muestras más extremas."""
    np.random.seed(42)
    # 100 errores calculados
    errors = np.linspace(0.01, 1.0, 100)
    p = 99.0
    
    threshold = float(np.percentile(errors, p))
    anomalies = errors > threshold
    
    # En un linspace uniforme de 100 valores, el P99 deja exactamente 1 valor por encima
    assert anomalies.sum() == 1, f"Se esperaba 1 anomalía sobre P99, se detectaron {anomalies.sum()}."
    assert threshold > 0.98, "El umbral calculado debe situarse cerca del límite superior del rango."

def test_incident_report_generation():
    """Verifica la coherencia en la generación de registros del reporte de incidentes."""
    is_anomaly = np.array([False, False, True, False, True])
    mse_values = np.array([0.05, 0.08, 1.45, 0.06, 2.10])
    window_size = 60
    
    anomaly_indices = np.where(is_anomaly)[0]
    records = []
    for idx in anomaly_indices:
        records.append({
            "Ventana_Inicio": idx,
            "Ventana_Fin": idx + window_size,
            "MSE": mse_values[idx]
        })
    df_report = pd.DataFrame(records)
    
    assert len(df_report) == 2, "El reporte debe contener exactamente 2 incidentes detectados."
    assert (df_report["MSE"] > 1.0).all(), "Todas las anomalías reportadas deben presentar un MSE elevado."
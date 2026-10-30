import pytest
import numpy as np
from sklearn.preprocessing import RobustScaler

def test_robust_scaler_integrity():
    """Verifica que el escalado robusto no produzca valores nulos o infinitos."""
    np.random.seed(42)
    # Simulación de datos con outliers extremos (típico en telemetría de servidores)
    data = np.random.normal(loc=10.0, scale=2.0, size=(100, 38))
    data[10, 5] = 9999.0  # Outlier extremo
    
    scaler = RobustScaler()
    scaled = scaler.fit_transform(data)
    
    assert scaled.shape == (100, 38), "La dimensión de los datos escalados no coincide con la original."
    assert not np.isnan(scaled).any(), "El escalador generó valores NaN."
    assert not np.isinf(scaled).any(), "El escalador generó valores Inf."

def test_window_generation_shape():
    """Verifica la generación de tensores de ventanas deslizantes (stride=1)."""
    n_samples = 150
    n_channels = 38
    window_size = 60
    
    data = np.zeros((n_samples, n_channels))
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i : i + window_size])
    windows = np.array(windows)
    
    expected_windows = n_samples - window_size + 1  # 150 - 60 + 1 = 91
    assert windows.shape == (expected_windows, window_size, n_channels), \
        f"Forma esperada: ({expected_windows}, {window_size}, {n_channels}), pero se obtuvo {windows.shape}."

def test_window_minimum_length_validation():
    """Valida que una serie con menos observaciones que la ventana no pueda procesarse."""
    window_size = 60
    insufficient_data = np.zeros((50, 38))
    
    windows = [
        insufficient_data[i : i + window_size]
        for i in range(len(insufficient_data) - window_size + 1)
    ]
    assert len(windows) == 0, "No se deberían generar ventanas si la serie es más corta que window_size."
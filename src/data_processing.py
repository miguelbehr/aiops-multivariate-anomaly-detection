"""Módulo de procesamiento de telemetría y generación de ventanas temporales causales.

Proyecto: Detección Temprana de Anomalías en Servidores (AIOps)
Autor: Miguel Angel Behr López
"""

import os
import urllib.request
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_URL = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
DEFAULT_MACHINES = ["machine-1-1"]


def download_smd_dataset(target_dir="data/raw", machines=None):
    """Descarga los subconjuntos train, test y etiquetas de falla."""
    if machines is None:
        machines = DEFAULT_MACHINES

    subsets = ["train", "test", "test_label"]
    for subset in subsets:
        subset_path = os.path.join(target_dir, subset)
        os.makedirs(subset_path, exist_ok=True)
        for machine in machines:
            file_name = f"{machine}.txt"
            file_path = os.path.join(subset_path, file_name)
            if not os.path.exists(file_path):
                url = f"{BASE_URL}/{subset}/{file_name}"
                logging.info(f"Descargando {subset}/{file_name}...")
                urllib.request.urlretrieve(url, file_path)
            else:
                logging.info(f"Archivo verificado localmente: {subset}/{file_name}")


class SMDDataProcessor:
    """Pipeline de ingesta, escalado causal robusto y tensores temporales deslizantes."""

    def __init__(self, window_size=60, stride=1):
        self.window_size = window_size
        self.stride = stride
        self.scaler = RobustScaler()
        self.is_fitted = False

    def load_data(self, machine_name="machine-1-1", data_dir="data/raw"):
        """Carga matrices numéricas de entrenamiento, prueba y ground truth."""
        train_file = os.path.join(data_dir, "train", f"{machine_name}.txt")
        test_file = os.path.join(data_dir, "test", f"{machine_name}.txt")
        label_file = os.path.join(data_dir, "test_label", f"{machine_name}.txt")

        train_data = np.loadtxt(train_file, delimiter=",", dtype=np.float32)
        test_data = np.loadtxt(test_file, delimiter=",", dtype=np.float32)
        labels = np.loadtxt(label_file, delimiter=",", dtype=np.int32)

        assert not np.isnan(train_data).any(), "Valores nulos detectados en entrenamiento."
        assert not np.isnan(test_data).any(), "Valores nulos detectados en prueba."

        return train_data, test_data, labels

    def fit_transform_train(self, train_data):
        """Ajusta RobustScaler exclusivamente sobre la fase basal sana."""
        scaled_train = self.scaler.fit_transform(train_data)
        self.is_fitted = True
        return scaled_train

    def transform_test(self, test_data):
        """Aplica la transformación aprendida sobre el set de prueba."""
        if not self.is_fitted:
            raise RuntimeError("El escalador debe ajustarse primero con fit_transform_train().")
        return self.scaler.transform(test_data)

    def create_sliding_windows(self, series, labels=None):
        """Genera tensores 3D (N, window_size, n_features) causales."""
        n_points, n_features = series.shape
        num_windows = (n_points - self.window_size) // self.stride + 1

        windows = np.empty((num_windows, self.window_size, n_features), dtype=np.float32)
        aligned_labels = np.empty(num_windows, dtype=np.int32) if labels is not None else None

        for idx in range(num_windows):
            start = idx * self.stride
            end = start + self.window_size
            windows[idx] = series[start:end]
            if labels is not None:
                aligned_labels[idx] = labels[end - 1]

        if labels is not None:
            return windows, aligned_labels
        return windows

    def save_scaler(self, path="models/scaler.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        logging.info(f"Scaler guardado en: {path}")

    def load_scaler(self, path="models/scaler.joblib"):
        """Carga un objeto RobustScaler previamente guardado."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo de escalador en: {path}")
        self.scaler = joblib.load(path)
        self.is_fitted = True
        logging.info(f"Scaler cargado exitosamente desde: {path}")

if __name__ == "__main__":
    print("--- 1. Descarga de Telemetría (SMD) ---")
    download_smd_dataset(target_dir="data/raw", machines=["machine-1-1"])

    print("\n--- 2. Carga y Normalización Causal ---")
    processor = SMDDataProcessor(window_size=60, stride=1)
    train_raw, test_raw, labels = processor.load_data(machine_name="machine-1-1", data_dir="data/raw")
    print(f"• Registros entrenamiento (fase sana): {train_raw.shape}")
    print(f"• Registros prueba (con fallas):       {test_raw.shape}")
    print(f"• Etiquetas ground truth:             {labels.shape} (Anomalías: {labels.sum()} min)")

    print("\n--- 3. Generación de Tensores Temporales ---")
    train_scaled = processor.fit_transform_train(train_raw)
    test_scaled = processor.transform_test(test_raw)

    X_train = processor.create_sliding_windows(train_scaled)
    X_test, y_test = processor.create_sliding_windows(test_scaled, labels)

    print(f"• Tensores X_train: {X_train.shape} -> (Muestras, Ventana=60, Sensores=38)")
    print(f"• Tensores X_test:  {X_test.shape}")
    print(f"• Vector y_test:    {y_test.shape}")

    processor.save_scaler("models/scaler.joblib")
    print("\n[ÉXITO] Pipeline de datos ejecutado y validado correctamente.")
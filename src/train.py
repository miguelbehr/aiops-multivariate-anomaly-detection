"""Script de entrenamiento, optimización y serialización de modelos de AIOps.

Entrena:
1. Isolation Forest (models/model_v1.pkl)
2. Deep Autoencoder en PyTorch con Early Stopping (models/best_model.pt)

Proyecto: Detección Temprana de Anomalías en Servidores
Autor: Miguel Angel Behr López
"""

import os
import json
import logging
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.data_processing import SMDDataProcessor
    from src.model import IsolationForestBaseline, DeepAutoencoder
except ModuleNotFoundError:
    from data_processing import SMDDataProcessor
    from model import IsolationForestBaseline, DeepAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_baseline_model(X_train_windows, output_path="models/model_v1.pkl"):
    """Entrena y serializa el modelo base Isolation Forest."""
    logging.info("--- Iniciando Entrenamiento: Isolation Forest (Línea Base) ---")
    baseline = IsolationForestBaseline(contamination=0.04, n_estimators=100)
    baseline.fit(X_train_windows)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(baseline, output_path)
    logging.info(f"[OK] Modelo Base guardado exitosamente en: {output_path}")
    return baseline


def train_autoencoder_model(
    X_train_windows,
    epochs=15,
    batch_size=128,
    lr=1e-3,
    patience=4,
    output_model_path="models/best_model.pt",
    output_threshold_path="models/threshold_config.json"
):
    """Entrena la red neuronal profunda Autoencoder con Early Stopping y calcula el umbral P99."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"--- Iniciando Entrenamiento: Deep Autoencoder en dispositivo [{device}] ---")

    # Separar en Entrenamiento (80%) y Validación interna (20%) sobre fase normal
    split_idx = int(len(X_train_windows) * 0.8)
    X_tr = X_train_windows[:split_idx]
    X_val = X_train_windows[split_idx:]

    train_tensor = torch.tensor(X_tr, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_tensor), batch_size=batch_size, shuffle=False)

    model = DeepAutoencoder(window_size=60, n_features=38).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Fase de entrenamiento
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Fase de validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        logging.info(f"Época [{epoch:02d}/{epochs:02d}] -> Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        # Mecanismo de Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save(model.state_dict(), output_model_path)
            logging.info(f"  └─> [Mejor modelo guardado] Val Loss reducida a {val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Parada anticipada (Early Stopping) activada en la época {epoch}.")
                break

    # Cargar los mejores pesos guardados para calcular el umbral dinámico P99
    model.load_state_dict(torch.load(output_model_path))
    model.eval()
    with torch.no_grad():
        val_inputs = val_tensor.to(device)
        total_errors, _ = model.compute_reconstruction_error(val_inputs)
        val_errors_np = total_errors.cpu().numpy()

    # Umbral por percentil empírico P99.0
    threshold = float(np.percentile(val_errors_np, 99.0))
    logging.info(f"[OK] Umbral dinámico calibrado (Percentil 99): {threshold:.6f}")

    threshold_metadata = {
        "threshold_p99": threshold,
        "best_val_loss": float(best_val_loss),
        "window_size": 60,
        "n_features": 38
    }

    with open(output_threshold_path, "w") as f:
        json.dump(threshold_metadata, f, indent=4)

    logging.info(f"[OK] Metadatos de umbral guardados en: {output_threshold_path}")
    return model, threshold


def main():
    logging.info("=== PASO 3: ENTRENAMIENTO DE PUNTA A PUNTA ===")
    
    # 1. Cargar y procesar datos
    processor = SMDDataProcessor(window_size=60, stride=1)
    train_raw, _, _ = processor.load_data(machine_name="machine-1-1", data_dir="data/raw")
    train_scaled = processor.fit_transform_train(train_raw)
    X_train = processor.create_sliding_windows(train_scaled)
    logging.info(f"Tensores generados para entrenamiento: {X_train.shape}")

    # 2. Entrenar Línea Base (Isolation Forest)
    train_baseline_model(X_train, output_path="models/model_v1.pkl")

    # 3. Entrenar Autoencoder Profundo
    # Usamos 10 épocas para asegurar convergencia rápida y efectiva en CPU local o GPU Colab
    train_autoencoder_model(
        X_train,
        epochs=10,
        batch_size=128,
        lr=1e-3,
        patience=3,
        output_model_path="models/best_model.pt",
        output_threshold_path="models/threshold_config.json"
    )

    logging.info("\n[ÉXITO TOTAL] Ambos modelos han sido entrenados y serializados en /models.")


if __name__ == "__main__":
    main()
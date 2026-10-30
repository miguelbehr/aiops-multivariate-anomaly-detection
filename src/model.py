"""Definición de arquitecturas de modelos de IA para AIOps.

Incluye:
1. IsolationForestBaseline (Modelo de Referencia)
2. DeepAutoencoder (Red Neuronal Profunda en PyTorch)

Proyecto Integrador - Maestría en IA
Autor: Miguel Angel Behr López
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest


class IsolationForestBaseline:
    """Ensamble no supervisado de particionamiento espacial sobre ventanas aplanadas."""

    def __init__(self, contamination=0.04, n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_windows):
        """Entrena sobre tensores (N, window_size, n_features) aplanándolos a 2D."""
        n_samples, w, d = X_windows.shape
        X_flat = X_windows.reshape(n_samples, w * d)
        self.model.fit(X_flat)

    def predict(self, X_windows):
        """Retorna 1 para anomalía y 0 para comportamiento normal."""
        n_samples, w, d = X_windows.shape
        X_flat = X_windows.reshape(n_samples, w * d)
        raw_preds = self.model.predict(X_flat)
        return np.where(raw_preds == -1, 1, 0)

    def decision_function(self, X_windows):
        """Retorna score continuo de anomalía."""
        n_samples, w, d = X_windows.shape
        X_flat = X_windows.reshape(n_samples, w * d)
        return -self.model.score_samples(X_flat)


class DeepAutoencoder(nn.Module):
    """Autoencoder profundo en PyTorch para reconstrucción de series multivariadas.

    Estructura simétrica: Entrada (60x38 = 2280) -> 512 -> 128 -> Latente (32) -> 128 -> 512 -> Salida (2280).
    """

    def __init__(self, window_size=60, n_features=38, latent_dim=32, dropout_rate=0.2):
        super(DeepAutoencoder, self).__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.input_dim = window_size * n_features

        # Codificador (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, latent_dim),
            nn.LeakyReLU(0.2),
        )

        # Decodificador (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.input_dim),
        )

    def forward(self, x):
        """Flujo hacia adelante: entrada 3D (batch, 60, 38) -> reconstrucción 3D."""
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        latent = self.encoder(x_flat)
        reconstructed_flat = self.decoder(latent)
        reconstructed = reconstructed_flat.view(batch_size, self.window_size, self.n_features)
        return reconstructed

    def compute_reconstruction_error(self, x):
        """Calcula el MSE residual total y el error desglosado por cada una de las métricas."""
        self.eval()
        with torch.no_grad():
            x_recon = self.forward(x)
            # Error cuadrático puntual: (batch, 60, 38)
            squared_diff = (x - x_recon) ** 2
            # Error global por ventana (MSE escalar)
            mse_total = squared_diff.mean(dim=(1, 2))
            # Error residual atribuible a cada sensor/métrica: vector de tamaño 38
            mse_per_feature = squared_diff.mean(dim=1)
        return mse_total, mse_per_feature


if __name__ == "__main__":
    print("--- Verificando Instanciación de Modelos ---")

    # 1. Probar Isolation Forest con datos sintéticos
    dummy_input = np.random.randn(20, 60, 38).astype(np.float32)
    baseline = IsolationForestBaseline()
    baseline.fit(dummy_input)
    baseline_preds = baseline.predict(dummy_input)
    print(f"• Isolation Forest operativo. Predicciones shape: {baseline_preds.shape}")

    # 2. Probar PyTorch Autoencoder
    tensor_input = torch.tensor(dummy_input)
    autoencoder = DeepAutoencoder(window_size=60, n_features=38)
    output = autoencoder(tensor_input)
    total_err, feat_err = autoencoder.compute_reconstruction_error(tensor_input)

    print(f"• Autoencoder PyTorch operativo:")
    print(f"   - Entrada:  {tensor_input.shape}")
    print(f"   - Salida:   {output.shape}")
    print(f"   - MSE ventanas: {total_err.shape}")
    print(f"   - Desglose por sensor (38 métricas): {feat_err.shape}")
    print("\n[ÉXITO] Modelos definidos y verificados sin errores dimensionales.")
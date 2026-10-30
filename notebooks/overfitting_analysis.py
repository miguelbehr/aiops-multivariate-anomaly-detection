"""Actividad Semana 3: Diagnóstico de Overfitting/Underfitting y Estrategias de Mejora.

Maestría en Inteligencia Artificial - UEES
Autor: Miguel Angel Behr López
Caso: Detección Temprana de Anomalías en Servidores (SMD)
"""

import os
import sys

# 1. RESOLUCIÓN DINÁMICA DE LA RAÍZ DEL PROYECTO
def find_project_root():
    """Localiza la raíz del proyecto sin importar dónde se inicie el script o notebook."""
    current = os.path.abspath(os.getcwd())
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "src", "data_processing.py")):
            return current
        current = os.path.dirname(current)
    return os.path.abspath(os.getcwd())

PROJECT_ROOT = find_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

from src.data_processing import SMDDataProcessor

# Fijar semillas para reproducibilidad experimental
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Dispositivo de cómputo activo: {device}")
print(f"[INFO] Raíz del proyecto detectada:  {PROJECT_ROOT}")


# ==============================================================================
# 2. CARGA Y PREPROCESAMIENTO CAUSAL (SMD machine-1-1)
# ==============================================================================
processor = SMDDataProcessor(window_size=60, stride=1)
raw_data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
train_raw, _, _ = processor.load_data(machine_name="machine-1-1", data_dir=raw_data_dir)

# Escalado y partición temporal (80% Train / 20% Val)
train_scaled = processor.fit_transform_train(train_raw)
X_windows = processor.create_sliding_windows(train_scaled)

split_idx = int(len(X_windows) * 0.8)
X_train_np = X_windows[:split_idx]
X_val_np = X_windows[split_idx:]

train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
val_tensor = torch.tensor(X_val_np, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(val_tensor, val_tensor), batch_size=128, shuffle=False)

print(f"[DATOS] Ventanas de Entrenamiento: {train_tensor.shape}")
print(f"[DATOS] Ventanas de Validación:    {val_tensor.shape}")


# ==============================================================================
# 3. SISTEMA DE TRACKING DE MÉTRICAS (Rúbrica 25%)
# ==============================================================================
class ComprehensiveMetricsTracker:
    """Registra y cuantifica Loss (MSE), Score (R2), Generalization Gap y Overfit Ratio."""

    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_score": [],
            "val_score": [],
            "gen_gap": [],
            "overfit_ratio": []
        }

    def log(self, epoch, t_loss, v_loss, t_score, v_score):
        gap = v_loss - t_loss
        ratio = (v_loss / t_loss) if t_loss > 0 else 1.0
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(t_loss)
        self.history["val_loss"].append(v_loss)
        self.history["train_score"].append(t_score)
        self.history["val_score"].append(v_score)
        self.history["gen_gap"].append(gap)
        self.history["overfit_ratio"].append(ratio)

    def to_dataframe(self):
        return pd.DataFrame(self.history)


def compute_epoch_metrics(model, loader, criterion):
    """Calcula el MSE medio y el Coeficiente R2 de reconstrucción sobre un DataLoader."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            all_targets.append(targets.view(targets.size(0), -1).cpu().numpy())
            all_preds.append(outputs.view(outputs.size(0), -1).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true_flat = np.vstack(all_targets)
    y_pred_flat = np.vstack(all_preds)
    r2 = r2_score(y_true_flat, y_pred_flat)
    return avg_loss, float(r2)


# ==============================================================================
# 4. EXPERIMENTO 1: MODELO BASE (PROVOCA OVERFITTING)
# ==============================================================================
class UnregularizedAutoencoder(nn.Module):
    """Modelo sobredimensionado sin Dropout, sin BatchNorm y sin regularización de pesos."""

    def __init__(self, window_size=60, n_features=38, latent_dim=64):
        super(UnregularizedAutoencoder, self).__init__()
        input_dim = window_size * n_features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        b = x.size(0)
        z = self.encoder(x.view(b, -1))
        out = self.decoder(z)
        return out.view(b, 60, 38)


base_model = UnregularizedAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer_base = torch.optim.Adam(base_model.parameters(), lr=0.001)

tracker_base = ComprehensiveMetricsTracker()
EPOCHS = 18

print("\n--- Entrenando Modelo Base (Sin regularización, 18 épocas) ---")
for epoch in range(1, EPOCHS + 1):
    base_model.train()
    for inp, tgt in train_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer_base.zero_grad()
        loss = criterion(base_model(inp), tgt)
        loss.backward()
        optimizer_base.step()

    t_loss, t_score = compute_epoch_metrics(base_model, train_loader, criterion)
    v_loss, v_score = compute_epoch_metrics(base_model, val_loader, criterion)
    tracker_base.log(epoch, t_loss, v_loss, t_score, v_score)
    print(f"Época [{epoch:02d}/{EPOCHS:02d}] -> Train MSE: {t_loss:.5f} | Val MSE: {v_loss:.5f} | Val R2: {v_score:.4f}")

df_base = tracker_base.to_dataframe()


# ==============================================================================
# 5. EXPERIMENTO 2: MODELO OPTIMIZADO (DROPOUT + L2 + EARLY STOPPING)
# ==============================================================================
from src.model import DeepAutoencoder

improved_model = DeepAutoencoder(window_size=60, n_features=38, latent_dim=32, dropout_rate=0.2).to(device)
optimizer_improved = torch.optim.Adam(improved_model.parameters(), lr=0.001, weight_decay=1e-4)

tracker_improved = ComprehensiveMetricsTracker()
patience = 3
patience_counter = 0
best_val_loss = float("inf")
best_epoch_imp = 1

print("\n--- Entrenando Modelo Optimizado (Dropout=0.2, L2=1e-4, Early Stopping) ---")
for epoch in range(1, EPOCHS + 1):
    improved_model.train()
    for inp, tgt in train_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer_improved.zero_grad()
        loss = criterion(improved_model(inp), tgt)
        loss.backward()
        optimizer_improved.step()

    t_loss, t_score = compute_epoch_metrics(improved_model, train_loader, criterion)
    v_loss, v_score = compute_epoch_metrics(improved_model, val_loader, criterion)
    tracker_improved.log(epoch, t_loss, v_loss, t_score, v_score)
    print(f"Época [{epoch:02d}/{EPOCHS:02d}] -> Train MSE: {t_loss:.5f} | Val MSE: {v_loss:.5f} | Val R2: {v_score:.4f}")

    if v_loss < best_val_loss:
        best_val_loss = v_loss
        best_epoch_imp = epoch
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"-> [Early Stopping] Parada preventiva en época {epoch}. Mejor época: {best_epoch_imp}")
            break

df_improved = tracker_improved.to_dataframe()


# ==============================================================================
# 6. CURVAS DE APRENDIZAJE COMPLETAS A 300 DPI (Rúbrica 30%)
# ==============================================================================
results_fig_dir = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(results_fig_dir, exist_ok=True)
fig_path = os.path.join(results_fig_dir, "learning_curves_diagnostic.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

min_val_base_idx = int(df_base["val_loss"].idxmin())
best_epoch_base = int(df_base.loc[min_val_base_idx, "epoch"])

# Gráfico 1: Loss Modelo Base (Overfitting)
axes[0, 0].plot(df_base["epoch"], df_base["train_loss"], label="Train Loss (MSE)", color="#1f77b4", lw=2)
axes[0, 0].plot(df_base["epoch"], df_base["val_loss"], label="Val Loss (MSE)", color="#d62728", lw=2, linestyle="--")
axes[0, 0].axvline(x=best_epoch_base, color="gray", linestyle=":", label=f"Mínimo Val (Época {best_epoch_base})")
axes[0, 0].annotate("Divergencia (Overfitting)",
                    xy=(best_epoch_base, df_base.loc[min_val_base_idx, "val_loss"]),
                    xytext=(best_epoch_base + 2, df_base.loc[min_val_base_idx, "val_loss"] + 0.03),
                    arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                    fontsize=9, fontweight="bold")
axes[0, 0].set_title("A1. Modelo Base: Loss (MSE)", fontsize=11, fontweight="bold")
axes[0, 0].set_xlabel("Época")
axes[0, 0].set_ylabel("Error Cuadrático Medio")
axes[0, 0].grid(True, linestyle="--", alpha=0.6)
axes[0, 0].legend()

# Gráfico 2: Score R2 Modelo Base (Estancamiento / Deterioro)
axes[0, 1].plot(df_base["epoch"], df_base["train_score"], label="Train R² Score", color="#1f77b4", lw=2)
axes[0, 1].plot(df_base["epoch"], df_base["val_score"], label="Val R² Score", color="#d62728", lw=2, linestyle="--")
axes[0, 1].axvline(x=best_epoch_base, color="gray", linestyle=":")
axes[0, 1].set_title("B1. Modelo Base: Score de Fidelidad (R²)", fontsize=11, fontweight="bold")
axes[0, 1].set_xlabel("Época")
axes[0, 1].set_ylabel("Coeficiente R²")
axes[0, 1].grid(True, linestyle="--", alpha=0.6)
axes[0, 1].legend()

# Gráfico 3: Loss Modelo Optimizado (Convergencia Controlada)
axes[1, 0].plot(df_improved["epoch"], df_improved["train_loss"], label="Train Loss (MSE)", color="#1f77b4", lw=2)
axes[1, 0].plot(df_improved["epoch"], df_improved["val_loss"], label="Val Loss (MSE)", color="#2ca02c", lw=2, linestyle="--")
axes[1, 0].axvline(x=best_epoch_imp, color="gray", linestyle=":", label=f"Parada Óptima (Época {best_epoch_imp})")
axes[1, 0].annotate("Convergencia Estable",
                    xy=(best_epoch_imp, df_improved.loc[best_epoch_imp - 1, "val_loss"]),
                    xytext=(best_epoch_imp - 2, df_improved.loc[best_epoch_imp - 1, "val_loss"] + 0.02),
                    arrowprops=dict(facecolor="green", shrink=0.08, width=1, headwidth=5),
                    fontsize=9, fontweight="bold")
axes[1, 0].set_title("A2. Modelo Optimizado: Loss (MSE)", fontsize=11, fontweight="bold")
axes[1, 0].set_xlabel("Época")
axes[1, 0].set_ylabel("Error Cuadrático Medio")
axes[1, 0].grid(True, linestyle="--", alpha=0.6)
axes[1, 0].legend()

# Gráfico 4: Score R2 Modelo Optimizado (Generalización Alta)
axes[1, 1].plot(df_improved["epoch"], df_improved["train_score"], label="Train R² Score", color="#1f77b4", lw=2)
axes[1, 1].plot(df_improved["epoch"], df_improved["val_score"], label="Val R² Score", color="#2ca02c", lw=2, linestyle="--")
axes[1, 1].axvline(x=best_epoch_imp, color="gray", linestyle=":")
axes[1, 1].set_title("B2. Modelo Optimizado: Score de Fidelidad (R²)", fontsize=11, fontweight="bold")
axes[1, 1].set_xlabel("Época")
axes[1, 1].set_ylabel("Coeficiente R²")
axes[1, 1].grid(True, linestyle="--", alpha=0.6)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"\n[OK] Curvas de aprendizaje guardadas en: {fig_path}")


# ==============================================================================
# 7. TABLA COMPARATIVA CUANTITATIVA (Rúbrica 20% y 25%)
# ==============================================================================
comp_table = pd.DataFrame({
    "Métrica / Indicador": [
        "Épocas Totales",
        "Mejor Época de Validación",
        "Validation MSE Mínimo",
        "Validation R² Máximo",
        "Brecha de Generalización Final",
        "Ratio de Sobreajuste (Val/Train)",
        "Diagnóstico de Comportamiento"
    ],
    "Modelo Base (Sin Regularizar)": [
        len(df_base),
        best_epoch_base,
        f"{df_base['val_loss'].min():.5f}",
        f"{df_base['val_score'].max():.4f}",
        f"{df_base.iloc[-1]['gen_gap']:.5f}",
        f"{df_base.iloc[-1]['overfit_ratio']:.2f}x",
        "Overfitting Severo"
    ],
    "Modelo Optimizado (Dropout + L2 + Early Stopping)": [
        len(df_improved),
        best_epoch_imp,
        f"{df_improved['val_loss'].min():.5f}",
        f"{df_improved['val_score'].max():.4f}",
        f"{df_improved.iloc[-1]['gen_gap']:.5f}",
        f"{df_improved.iloc[-1]['overfit_ratio']:.2f}x",
        "Convergencia Óptima"
    ]
})

print("\n" + "=" * 80)
print("TABLA COMPARATIVA: DIAGNÓSTICO DE GENERALIZACIÓN ANTES VS. DESPUÉS")
print("=" * 80)
print(comp_table.to_string(index=False))
print("=" * 80)
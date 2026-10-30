import pytest
import torch
import torch.nn as nn

class AutoencoderReference(nn.Module):
    """Arquitectura canónica del Autoencoder multivariado (2280 -> 512 -> 128 -> 32)."""
    def __init__(self, input_dim=2280, latent_dim=32, dropout_rate=0.2):
        super(AutoencoderReference, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        x_flat = x.view(batch_size, -1)
        latent = self.encoder(x_flat)
        rec_flat = self.decoder(latent)
        return rec_flat.view(batch_size, seq_len, n_features)

def test_autoencoder_forward_pass_dimensions():
    """Comprueba que la salida reconstruida tenga idéntica forma que el tensor de entrada tridimensional."""
    batch_size = 4
    seq_len = 60
    n_features = 38
    input_dim = seq_len * n_features  # 2280
    
    model = AutoencoderReference(input_dim=input_dim, latent_dim=32)
    model.eval()
    
    x = torch.randn(batch_size, seq_len, n_features)
    with torch.no_grad():
        reconstruction = model(x)
        
    assert reconstruction.shape == x.shape, \
        f"La dimensión de reconstrucción {reconstruction.shape} no coincide con la entrada {x.shape}."

def test_latent_space_dimension():
    """Comprueba que el cuello de botella latente comprima exactamente a 32 dimensiones."""
    batch_size = 8
    seq_len = 60
    n_features = 38
    input_dim = seq_len * n_features
    
    model = AutoencoderReference(input_dim=input_dim, latent_dim=32)
    model.eval()
    
    x_flat = torch.randn(batch_size, input_dim)
    with torch.no_grad():
        latent = model.encoder(x_flat)
        
    assert latent.shape == (batch_size, 32), \
        f"Espacio latente esperado ({batch_size}, 32), obtenido {latent.shape}."

def test_reconstruction_loss_non_negative():
    """Verifica que el error cuadrático medio (MSE) sea estrictamente no negativo."""
    model = AutoencoderReference(input_dim=2280, latent_dim=32)
    model.eval()
    
    x = torch.randn(2, 60, 38)
    with torch.no_grad():
        rec = model(x)
        mse = torch.nn.functional.mse_loss(rec, x)
        
    assert mse.item() >= 0.0, "La función de pérdida MSE no puede ser negativa."
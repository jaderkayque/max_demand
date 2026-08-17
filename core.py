"""
core.py — Núcleo de ML (sem Spark) da Demanda Máxima Estrutural.

Estima o ESTADO LATENTE da carga (sem manobras nem erros de medição) com um
autoencoder denoising 1D-CNN GLOBAL e extrai a MÁXIMA DEMANDA ANUAL.

Ideia (ver brainstorm): a série observada = demanda estrutural + manobras +
ruído. O autoencoder aprende a reconstruir a componente estrutural e descarta
os eventos transitórios porque:
  1) o gargalo (bottleneck) não consegue representar picos esparsos;
  2) a perda L1/Huber não persegue desvios grandes e esparsos (manobras);
  3) durante o treino injetamos contaminação sintética na ENTRADA e pedimos a
     curva limpa de volta (objetivo denoising), ensinando a remover manobras.

Este módulo NÃO depende de Spark — é importado tanto pelos notebooks Databricks
(`nb_0*_*.py`) quanto pelo teste local (`test_pipeline.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except Exception:  # torch é opcional para as partes não-neurais
    torch = None
    nn = object  # type: ignore
    _TORCH_OK = False


# =============================================================================
# 1. Configuração
# =============================================================================

@dataclass
class Config:
    L: int = 144            # tamanho da janela (1 dia @ 10 min = 144 pontos)
    latent: int = 32        # dimensão do gargalo
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    prob_max: float = 0.999  # quantil que define a "máxima" sobre o latente
    seed: int = 42
    # contaminação sintética injetada no treino (denoising)
    p_plato: float = 0.5    # prob. de injetar um platô (manobra) por janela
    plato_amp: float = 2.0  # amplitude típica do platô (em unidades normalizadas)
    p_spike: float = 0.5
    spike_amp: float = 4.0
    ruido: float = 0.05
    device: str = field(default_factory=lambda: "cuda" if (_TORCH_OK and torch.cuda.is_available()) else "cpu")


# =============================================================================
# 2. Normalização robusta por alimentador (mediana / IQR)
# =============================================================================

class RobustScaler:
    """Escala por mediana e intervalo interquantílico — robusto a picos.

    `q_low`/`q_high` definem o intervalo usado como escala (padrão 25/75 = IQR
    clássico). Mediana/IQR resistem a spikes, mas um platô de manobra SUSTENTADO
    (semanas) desloca os quantis externos; nesse cenário um intervalo mais
    interno (ex.: 30/70) é mais estável. É hiperparâmetro do protocolo
    experimental — tunar em validação interna, nunca no teste.
    """

    def __init__(self, q_low: float = 25.0, q_high: float = 75.0) -> None:
        self.q_low = q_low
        self.q_high = q_high
        self.mediana: float = 0.0
        self.escala: float = 1.0

    def fit(self, x: np.ndarray) -> "RobustScaler":
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            self.mediana, self.escala = 0.0, 1.0
            return self
        self.mediana = float(np.median(x))
        qh, ql = np.percentile(x, [self.q_high, self.q_low])
        iqr = float(qh - ql)
        self.escala = iqr if iqr > 1e-9 else (float(np.std(x)) or 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mediana) / self.escala

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.escala + self.mediana


# =============================================================================
# 3. Janelamento
# =============================================================================

def make_windows(series: np.ndarray, L: int, stride: Optional[int] = None) -> np.ndarray:
    """Fatia a série em janelas [N, L]. NaN preenchido por interpolação simples."""
    s = np.asarray(series, dtype=float)
    if s.size == 0:
        return np.empty((0, L), dtype=np.float32)
    # preenche NaN (ffill/bfill + média) para não contaminar o treino
    if np.any(~np.isfinite(s)):
        idx = np.arange(s.size)
        bom = np.isfinite(s)
        if bom.any():
            s = np.interp(idx, idx[bom], s[bom])
        else:
            s = np.zeros_like(s)
    if stride is None:
        stride = L
    n = 1 + max(0, (s.size - L) // stride)
    if s.size < L:  # janela única com padding pela borda
        pad = np.full(L - s.size, s[-1])
        return np.concatenate([s, pad])[None, :].astype(np.float32)
    out = np.stack([s[i * stride: i * stride + L] for i in range(n)], axis=0)
    return out.astype(np.float32)


# =============================================================================
# 4. Contaminação sintética (para o objetivo denoising)
# =============================================================================

def inject_contamination(x: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Recebe janelas limpas [N, L] e devolve versão contaminada (entrada do AE)."""
    x = x.copy()
    N, L = x.shape
    for i in range(N):
        if rng.random() < cfg.p_plato:  # platô sustentado (transferência de carga)
            ini = rng.integers(0, L)
            fim = min(L, ini + rng.integers(L // 8, L))
            x[i, ini:fim] += cfg.plato_amp * rng.choice([-1.0, 1.0]) * (0.5 + rng.random())
        if rng.random() < cfg.p_spike:  # picos isolados (erros de medição)
            k = rng.integers(1, 4)
            pos = rng.integers(0, L, size=k)
            x[i, pos] += cfg.spike_amp * rng.choice([-1.0, 1.0], size=k) * (0.5 + rng.random(k))
    x += cfg.ruido * rng.standard_normal(x.shape)
    return x.astype(np.float32)


# =============================================================================
# 5. Autoencoder denoising 1D-CNN
# =============================================================================

if _TORCH_OK:

    class DenoisingAE1D(nn.Module):
        def __init__(self, L: int = 144, latent: int = 32):
            super().__init__()
            self.L = L
            # encoder: L -> L/2 -> L/4 -> L/8
            self.enc = nn.Sequential(
                nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.GELU(),
                nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.GELU(),
                nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.GELU(),
            )
            Lr = L // 8
            self.flat_dim = 64 * Lr
            self.Lr = Lr
            self.to_latent = nn.Linear(self.flat_dim, latent)
            self.from_latent = nn.Linear(latent, self.flat_dim)
            # decoder: L/8 -> L/4 -> L/2 -> L
            self.dec = nn.Sequential(
                nn.ConvTranspose1d(64, 32, 8, stride=2, padding=3), nn.GELU(),
                nn.ConvTranspose1d(32, 16, 8, stride=2, padding=3), nn.GELU(),
                nn.ConvTranspose1d(16, 1, 8, stride=2, padding=3),
            )

        def forward(self, x):                 # x: [B, L]
            h = self.enc(x.unsqueeze(1))       # [B, 64, L/8]
            z = self.to_latent(h.flatten(1))   # [B, latent]
            h = self.from_latent(z).view(-1, 64, self.Lr)
            y = self.dec(h).squeeze(1)         # [B, L]
            return y


def train_model(windows: np.ndarray, cfg: Config):
    """Treina o autoencoder denoising GLOBAL. windows: [N, L] já normalizadas."""
    if not _TORCH_OK:
        raise RuntimeError("PyTorch não disponível: instale torch (requirements.txt).")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    clean = np.asarray(windows, dtype=np.float32)
    model = DenoisingAE1D(cfg.L, cfg.latent).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.L1Loss()  # robusto: não persegue picos esparsos

    n = clean.shape[0]
    model.train()
    for _ in range(cfg.epochs):
        ordem = rng.permutation(n)
        for b in range(0, n, cfg.batch_size):
            idx = ordem[b: b + cfg.batch_size]
            alvo = clean[idx]
            entrada = inject_contamination(alvo, cfg, rng)  # denoising
            xb = torch.from_numpy(entrada).to(cfg.device)
            yb = torch.from_numpy(alvo).to(cfg.device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def reconstruct(model, windows: np.ndarray, cfg: Config) -> np.ndarray:
    """Reconstrói o sinal estrutural (latente) das janelas [N, L]."""
    if not _TORCH_OK:
        raise RuntimeError("PyTorch não disponível: instale torch (requirements.txt).")
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, windows.shape[0], cfg.batch_size):
            xb = torch.from_numpy(windows[b: b + cfg.batch_size].astype(np.float32)).to(cfg.device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, cfg.L), dtype=np.float32)


# =============================================================================
# 6. Máxima demanda anual a partir do latente
# =============================================================================

def maxima_anual(model, serie_ano: np.ndarray, scaler: RobustScaler, cfg: Config) -> float:
    """
    Máxima demanda estrutural do ano: reconstrói o latente de toda a série do
    ano, inverte a normalização e toma um quantil alto (robusto a overshoots).
    """
    z = scaler.transform(serie_ano)
    janelas = make_windows(z, cfg.L, stride=cfg.L)
    if janelas.shape[0] == 0:
        return float("nan")
    rec = reconstruct(model, janelas, cfg).reshape(-1)
    rec = scaler.inverse(rec)
    rec = rec[np.isfinite(rec)]
    if rec.size == 0:
        return float("nan")
    return float(np.quantile(rec, cfg.prob_max))


# =============================================================================
# 7. Persistência do artefato do modelo
# =============================================================================

def salvar_modelo(model, caminho: str, cfg: Config, versao: str) -> None:
    if not _TORCH_OK:
        raise RuntimeError("PyTorch não disponível.")
    torch.save({"state_dict": model.state_dict(),
                "L": cfg.L, "latent": cfg.latent, "versao": versao}, caminho)


def carregar_modelo(caminho: str, cfg: Config):
    if not _TORCH_OK:
        raise RuntimeError("PyTorch não disponível.")
    ck = torch.load(caminho, map_location=cfg.device)
    model = DenoisingAE1D(ck["L"], ck["latent"]).to(cfg.device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("versao", "?")

"""
core_multi.py — H0 MULTIVARIADO + retreino H1 guiado pelo especialista.

Canais por janela diária:
    0 valor      (grandeza, normalizado)
    1 is_sicoi   (manobra OPERACIONAL conhecida, 0/1) — evento TEMPORÁRIO
    2 derivada   (np.gradient do valor) — realça picos/transições
    3 desvio     (desvio-padrão local) — realça instabilidade
    4 comp       (extensão do circuito em km, mensal, normalizada) — TOPOLOGIA
    5 d_comp     (variação mês-a-mês da extensão) — flag de MANOBRA TOPOLÓGICA

Ideia causal: um Δkm grande (canal 5) marca uma reconfiguração PERMANENTE da rede
que deve refletir num novo patamar de demanda — mudança ESTRUTURAL, que o H0 deve
preservar. Já `is_sicoi` marca evento OPERACIONAL temporário, que a perda de
reconstrução IGNORA (mascarado). Assim a rede aprende a separar mudança estrutural
(Δkm) de perturbação transitória (is_sicoi).

Saída = 1 canal (reconstrução estrutural do valor).
H1 = FINE-TUNING com perda de quantil anual (soft-peak) guiada pelo engenheiro.

Núcleo puro (sem Spark). Reutiliza RobustScaler/make_windows/inject_contamination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

import core  # RobustScaler, make_windows, inject_contamination, Config base

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except Exception:
    torch = None
    nn = object  # type: ignore
    _TORCH_OK = False


# =============================================================================
# 1. Configuração
# =============================================================================

@dataclass
class ConfigMulti:
    L: int = 144
    latent: int = 32
    n_canais: int = 6          # valor, is_sicoi, derivada, desvio, comp, d_comp
    n_static: int = 0          # sem estático (a extensão virou canal temporal)
    k_desvio: int = 7
    comp_mean: float = 0.0     # normalização do nível de extensão (km) — salvos no modelo
    comp_std: float = 1.0
    dcomp_std: float = 1.0     # escala da variação mensal de extensão
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    prob_max: float = 0.999
    tau: float = 0.05
    seed: int = 42
    p_plato: float = 0.5
    plato_amp: float = 2.0
    p_spike: float = 0.5
    spike_amp: float = 4.0
    ruido: float = 0.05
    device: str = field(default_factory=lambda: "cuda" if (_TORCH_OK and torch.cuda.is_available()) else "cpu")

    def base(self) -> "core.Config":
        return core.Config(L=self.L, latent=self.latent, batch_size=self.batch_size,
                           p_plato=self.p_plato, plato_amp=self.plato_amp,
                           p_spike=self.p_spike, spike_amp=self.spike_amp,
                           ruido=self.ruido, seed=self.seed, prob_max=self.prob_max)


# =============================================================================
# 2. Extensão do circuito: alinhamento mensal -> canais comp / d_comp
# =============================================================================

def alinhar_comprimento(ano, mes, uf, ativo, comp_map, dcomp_map, cfg: ConfigMulti):
    """
    Alinha a extensão MENSAL do circuito às medições. Recebe ano/mes por linha e
    devolve (comp_norm, d_comp_norm) — nível e variação mês-a-mês normalizados.
    Faltando o dado, comp=média (0 normalizado) e d_comp=0.
    comp_map/dcomp_map: {(UF, ativo, ANO, MES): valor_km}.
    """
    ano = np.asarray(ano); mes = np.asarray(mes)
    comp = np.array([comp_map.get((uf, ativo, int(a), int(m)), np.nan)
                     for a, m in zip(ano, mes)], dtype=float)
    dcm = np.array([dcomp_map.get((uf, ativo, int(a), int(m)), 0.0)
                    for a, m in zip(ano, mes)], dtype=float)
    s = cfg.comp_std if cfg.comp_std > 1e-9 else 1.0
    ds = cfg.dcomp_std if cfg.dcomp_std > 1e-9 else 1.0
    comp_norm = (comp - cfg.comp_mean) / s
    comp_norm[~np.isfinite(comp_norm)] = 0.0
    dcomp_norm = np.nan_to_num(dcm / ds, nan=0.0)
    return comp_norm.astype(np.float32), dcomp_norm.astype(np.float32)


# =============================================================================
# 3. Construção de canais (numpy)
# =============================================================================

def _rolling_std(v: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(v, dtype=float)
    ax = x.ndim - 1
    pad = k // 2
    xp = np.pad(x, [(0, 0)] * ax + [(pad, pad)], mode="edge")
    ker = np.ones(k) / k
    def conv_last(a):
        return np.apply_along_axis(lambda m: np.convolve(m, ker, mode="valid"), ax, a)
    m = conv_last(xp); m2 = conv_last(xp ** 2)
    return np.sqrt(np.clip(m2 - m ** 2, 0, None))[..., :x.shape[ax]]


def canais_de_janelas(valor_w, sic_w, comp_w, dcomp_w, cfg: ConfigMulti) -> np.ndarray:
    """De janelas [N,L] de valor/is_sicoi/comp/d_comp monta [N, 6, L]."""
    valor_w = np.asarray(valor_w, np.float32)
    deriv = np.gradient(valor_w, axis=-1).astype(np.float32)
    desvio = _rolling_std(valor_w, cfg.k_desvio).astype(np.float32)
    return np.stack([valor_w, np.asarray(sic_w, np.float32), deriv, desvio,
                     np.asarray(comp_w, np.float32), np.asarray(dcomp_w, np.float32)], axis=1)


# =============================================================================
# 4. Autoencoder condicional (multicanal)
# =============================================================================

if _TORCH_OK:

    class DenoisingAECond(nn.Module):
        def __init__(self, cfg: ConfigMulti):
            super().__init__()
            self.cfg = cfg
            self.enc = nn.Sequential(
                nn.Conv1d(cfg.n_canais, 16, 7, stride=2, padding=3), nn.GELU(),
                nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.GELU(),
                nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.GELU(),
            )
            self.Lr = cfg.L // 8
            self.flat_dim = 64 * self.Lr
            self.to_latent = nn.Linear(self.flat_dim + cfg.n_static, cfg.latent)
            self.from_latent = nn.Linear(cfg.latent, self.flat_dim)
            self.dec = nn.Sequential(
                nn.ConvTranspose1d(64, 32, 8, stride=2, padding=3), nn.GELU(),
                nn.ConvTranspose1d(32, 16, 8, stride=2, padding=3), nn.GELU(),
                nn.ConvTranspose1d(16, 1, 8, stride=2, padding=3),
            )

        def forward(self, x, static=None):     # x: [B,C,L]
            feat = self.enc(x).flatten(1)
            if self.cfg.n_static > 0 and static is not None:
                feat = torch.cat([feat, static], dim=1)
            z = self.to_latent(feat)
            h = self.from_latent(z).view(-1, 64, self.Lr)
            return self.dec(h).squeeze(1)       # [B,L]


def soft_peak(x, tau: float):
    """Pico suave diferenciável: média ponderada por softmax(x/tau)."""
    w = torch.softmax(x / tau, dim=0)
    return torch.sum(w * x)


# =============================================================================
# 5. Treino do H0 (denoising multicanal, máscara de is_sicoi)
# =============================================================================

def _batch_input(valor_clean, sic, comp, dcomp, cfg, rng, contaminar=True):
    vc = core.inject_contamination(valor_clean, cfg.base(), rng) if contaminar else valor_clean.copy()
    canais = canais_de_janelas(vc, sic, comp, dcomp, cfg)      # [B,6,L]
    alvo = valor_clean.astype(np.float32)
    masc = (1.0 - np.asarray(sic, np.float32))                 # ignora manobra operacional
    return canais, alvo, masc


def train_h0_multi(valor_w, sic_w, comp_w, dcomp_w, cfg: ConfigMulti):
    """Treina o H0 multivariado. Todos [N,L] (comp_w/dcomp_w já normalizados)."""
    if not _TORCH_OK:
        raise RuntimeError("PyTorch indisponível.")
    torch.manual_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    model = DenoisingAECond(cfg).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = valor_w.shape[0]; model.train()
    for _ in range(cfg.epochs):
        ordem = rng.permutation(n)
        for b in range(0, n, cfg.batch_size):
            idx = ordem[b:b + cfg.batch_size]
            ch, alvo, masc = _batch_input(valor_w[idx], sic_w[idx], comp_w[idx], dcomp_w[idx], cfg, rng)
            xb = torch.from_numpy(ch).to(cfg.device)
            yb = torch.from_numpy(alvo).to(cfg.device)
            mb = torch.from_numpy(masc).to(cfg.device)
            opt.zero_grad()
            pred = model(xb)
            loss = (torch.abs(pred - yb) * mb).sum() / mb.sum().clamp_min(1.0)
            loss.backward(); opt.step()
    model.eval()
    return model


# =============================================================================
# 6. Inferência: máxima anual (H0 ou H1)
# =============================================================================

def maxima_anual_multi(model, valor, is_sicoi, comp_full, dcomp_full, scaler, cfg: ConfigMulti) -> float:
    """comp_full/dcomp_full: já normalizados, alinhados ao vetor `valor`."""
    vn = scaler.transform(valor)
    vw = core.make_windows(vn, cfg.L, cfg.L)
    sw = core.make_windows(np.asarray(is_sicoi, float), cfg.L, cfg.L)
    cw = core.make_windows(np.asarray(comp_full, float), cfg.L, cfg.L)
    dw = core.make_windows(np.asarray(dcomp_full, float), cfg.L, cfg.L)
    if vw.shape[0] == 0:
        return float("nan")
    ch = canais_de_janelas(vw, sw, cw, dw, cfg)
    model.eval()
    with torch.no_grad():
        rec = model(torch.from_numpy(ch).to(cfg.device)).cpu().numpy().reshape(-1)
    rec = scaler.inverse(rec); rec = rec[np.isfinite(rec)]
    return float(np.quantile(rec, cfg.prob_max)) if rec.size else float("nan")


# =============================================================================
# 7. H1 — FINE-TUNING com supervisão do especialista (perda de quantil anual)
# =============================================================================

def fine_tune_h1(model, rotulados: List[dict], pool: dict, cfg: ConfigMulti,
                 lam: float = 8.0, epochs: int = 40, lr: float = 5e-4,
                 congelar_encoder: bool = True):
    """
    rotulados: [{'canais':[Ndia,6,L] do valor REAL, 'V_norm': float}]  (V do especialista, normalizado)
    pool: {'valor','sic','comp','dcomp'} [M,L] — amostra ampla p/ manter a reconstrução.
    Perda = L_recon(mascarada por is_sicoi) + lam * |soft_peak(recon_ano) - V|.
    """
    if not _TORCH_OK:
        raise RuntimeError("PyTorch indisponível.")
    rng = np.random.default_rng(cfg.seed + 1)
    if congelar_encoder:
        for p in model.enc.parameters():
            p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    M = pool["valor"].shape[0]; model.train()
    for _ in range(epochs):
        idx = rng.choice(M, size=min(cfg.batch_size, M), replace=False)
        ch, alvo, masc = _batch_input(pool["valor"][idx], pool["sic"][idx],
                                      pool["comp"][idx], pool["dcomp"][idx], cfg, rng)
        xb = torch.from_numpy(ch).to(cfg.device); yb = torch.from_numpy(alvo).to(cfg.device)
        mb = torch.from_numpy(masc).to(cfg.device)
        rec = model(xb)
        L_rec = (torch.abs(rec - yb) * mb).sum() / mb.sum().clamp_min(1.0)

        L_exp = torch.zeros((), device=cfg.device)
        for r in rotulados:
            chy = torch.from_numpy(r["canais"].astype(np.float32)).to(cfg.device)
            recy = model(chy).reshape(-1)
            L_exp = L_exp + torch.abs(soft_peak(recy, cfg.tau) - float(r["V_norm"]))
        L_exp = L_exp / max(1, len(rotulados))

        (L_rec + lam * L_exp).backward()
        opt.step(); opt.zero_grad()
    model.eval()
    return model


# =============================================================================
# 8. Persistência
# =============================================================================

def salvar_modelo(model, caminho: str, cfg: ConfigMulti, versao: str) -> None:
    if not _TORCH_OK:
        raise RuntimeError("PyTorch indisponível.")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__, "versao": versao}, caminho)


def carregar_modelo(caminho: str, device=None):
    if not _TORCH_OK:
        raise RuntimeError("PyTorch indisponível.")
    ck = torch.load(caminho, map_location=device or "cpu")
    cfg = ConfigMulti(**{k: v for k, v in ck["cfg"].items() if k in ConfigMulti().__dict__})
    if device:
        cfg.device = device
    model = DenoisingAECond(cfg).to(cfg.device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, cfg, ck.get("versao", "?")

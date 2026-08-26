"""
core_multi.py — H0 MULTIVARIADO + retreino H1 guiado pelo especialista.

Canais por janela diária:
    0 valor      (grandeza, normalizado)
    1 derivada   (np.gradient do valor) — realça picos/transições
    2 desvio     (desvio-padrão local) — realça instabilidade
    3 comp       (extensão do circuito em km, mensal, normalizada) — TOPOLOGIA
    4 d_comp     (variação mês-a-mês da extensão) — flag de MANOBRA TOPOLÓGICA

NOTA (decisão de projeto): o canal `is_sicoi` (registro de manobra operacional)
foi REMOVIDO do pipeline, junto com a máscara de perda que dependia dele. O
conhecimento de domínio embutido via features fica restrito à topologia
(comp/d_comp); a separação entre evento operacional temporário e mudança
estrutural passa a depender apenas do objetivo denoising + supervisão do
especialista (H1).

Ideia causal: um Δkm grande (canal 4) marca uma reconfiguração PERMANENTE da rede
que deve refletir num novo patamar de demanda — mudança ESTRUTURAL, que o H0 deve
preservar. Perturbações transitórias (manobras, erros) são atenuadas pelo
objetivo denoising (contaminação sintética + perda L1 robusta).

Saída = 1 canal (reconstrução estrutural do valor).
H1 = FINE-TUNING com perda de quantil anual (soft-peak) guiada pelo engenheiro.

Núcleo puro (sem Spark). Reutiliza RobustScaler/make_windows/inject_contamination.
"""

from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    n_canais: int = 5          # valor, derivada, desvio, comp, d_comp
    n_static: int = 0          # sem estático (a extensão virou canal temporal)
    k_desvio: int = 7
    comp_mean: float = 0.0     # normalização do nível de extensão (km) — salvos no modelo
    comp_std: float = 1.0
    dcomp_std: float = 1.0     # escala da variação mensal de extensão
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    prob_max: float = 0.999    # HIPERPARÂMETRO: tunar em validação interna, nunca no teste
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
    Limitação de observabilidade: o cadastro é mensal — reconfigurações que não
    sobrevivem à fotografia mensal não aparecem nestes canais.
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


def montar_janelas_grupo(valor, datas, regiao, ativo, ano, comp_map, dcomp_map,
                         cfg: ConfigMulti, max_janelas: int | None = None,
                         seed_extra: int = 0):
    """
    Monta as janelas [n, L] de UM alimentador-ano: valor normalizado (RobustScaler
    do próprio grupo), comp e d_comp alinhados ao mês. Se `max_janelas` for dado,
    amostra aleatoriamente no máximo esse número de janelas, com semente derivada
    da CHAVE do grupo (crc32) — reprodutível e independente da ordem/partição.

    Pensado para rodar DENTRO de `applyInPandas` (workers Spark): trazer a série
    bruta inteira ao driver com `.toPandas()` estoura `spark.driver.maxResultSize`;
    janelas float32 amostradas são ~20x menores que as linhas cruas.

    Retorna (vw, cw, dw) float32 ou None se a série for menor que L.
    """
    import zlib
    import pandas as pd
    valor = np.asarray(valor, dtype=float)
    if valor.size < cfg.L:
        return None
    sc = core.RobustScaler().fit(valor)
    d = pd.DatetimeIndex(pd.to_datetime(datas))   # aceita Series ou DatetimeIndex
    comp_full, dcomp_full = alinhar_comprimento(d.year, d.month, regiao, ativo,
                                                comp_map, dcomp_map, cfg)
    W = lambda a: core.make_windows(a, cfg.L, cfg.L)
    vw, cw, dw = W(sc.transform(valor)), W(comp_full), W(dcomp_full)
    if max_janelas is not None and vw.shape[0] > max_janelas:
        chave = f"{regiao}|{ativo}|{int(ano)}".encode()
        rng = np.random.default_rng(cfg.seed + seed_extra + zlib.crc32(chave))
        idx = rng.choice(vw.shape[0], size=max_janelas, replace=False)
        vw, cw, dw = vw[idx], cw[idx], dw[idx]
    return vw.astype(np.float32), cw.astype(np.float32), dw.astype(np.float32)


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


def canais_de_janelas(valor_w, comp_w, dcomp_w, cfg: ConfigMulti) -> np.ndarray:
    """De janelas [N,L] de valor/comp/d_comp monta [N, 5, L]."""
    valor_w = np.asarray(valor_w, np.float32)
    deriv = np.gradient(valor_w, axis=-1).astype(np.float32)
    desvio = _rolling_std(valor_w, cfg.k_desvio).astype(np.float32)
    return np.stack([valor_w, deriv, desvio,
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
# 5. Treino do H0 (denoising multicanal)
# =============================================================================

def _batch_input(valor_clean, comp, dcomp, cfg, rng, contaminar=True):
    vc = core.inject_contamination(valor_clean, cfg.base(), rng) if contaminar else valor_clean.copy()
    canais = canais_de_janelas(vc, comp, dcomp, cfg)           # [B,5,L]
    alvo = valor_clean.astype(np.float32)
    return canais, alvo


def train_h0_multi(valor_w, comp_w, dcomp_w, cfg: ConfigMulti):
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
            ch, alvo = _batch_input(valor_w[idx], comp_w[idx], dcomp_w[idx], cfg, rng)
            xb = torch.from_numpy(ch).to(cfg.device)
            yb = torch.from_numpy(alvo).to(cfg.device)
            opt.zero_grad()
            pred = model(xb)
            loss = torch.abs(pred - yb).mean()   # L1 robusta: não persegue picos esparsos
            loss.backward(); opt.step()
    model.eval()
    return model


# =============================================================================
# 6. Inferência: máxima anual (H0 ou H1)
# =============================================================================

def maxima_anual_multi(model, valor, comp_full, dcomp_full, scaler, cfg: ConfigMulti) -> float:
    """comp_full/dcomp_full: já normalizados, alinhados ao vetor `valor`."""
    vn = scaler.transform(valor)
    vw = core.make_windows(vn, cfg.L, cfg.L)
    cw = core.make_windows(np.asarray(comp_full, float), cfg.L, cfg.L)
    dw = core.make_windows(np.asarray(dcomp_full, float), cfg.L, cfg.L)
    if vw.shape[0] == 0:
        return float("nan")
    ch = canais_de_janelas(vw, cw, dw, cfg)
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
                 congelar_encoder: bool = True, rot_batch: int = 16):
    """
    rotulados: [{'canais':[Ndia,5,L] do valor REAL, 'V_norm': float}]  (V do especialista, normalizado)
    pool: {'valor','comp','dcomp'} [M,L] — amostra ampla p/ manter a reconstrução.
    Perda = L_recon + lam * |soft_peak(recon_ano) - V|.

    `rot_batch`: nº de exemplos rotulados amostrados POR PASSO de gradiente.
    Sem isso o custo por passo cresce O(n_rotulos) e o fine-tuning não escala
    para os milhares de rótulos dos níveis altos de supervisão do protocolo
    experimental.
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
        ch, alvo = _batch_input(pool["valor"][idx], pool["comp"][idx], pool["dcomp"][idx], cfg, rng)
        xb = torch.from_numpy(ch).to(cfg.device); yb = torch.from_numpy(alvo).to(cfg.device)
        rec = model(xb)
        L_rec = torch.abs(rec - yb).mean()

        # mini-batch de exemplos rotulados (amostrado a cada passo)
        k = min(rot_batch, len(rotulados))
        L_exp = torch.zeros((), device=cfg.device)
        if k > 0:
            sel = rng.choice(len(rotulados), size=k, replace=False)
            for j in sel:
                r = rotulados[int(j)]
                chy = torch.from_numpy(r["canais"].astype(np.float32)).to(cfg.device)
                recy = model(chy).reshape(-1)
                L_exp = L_exp + torch.abs(soft_peak(recy, cfg.tau) - float(r["V_norm"]))
            L_exp = L_exp / k

        (L_rec + lam * L_exp).backward()
        opt.step(); opt.zero_grad()
    model.eval()
    return model


# =============================================================================
# 8. Persistência e VERSIONAMENTO por timestamp
# =============================================================================
#
# Convenção: todo modelo treinado é gravado como
#     <dir_modelos>/<prefixo>_<AAAAMMDD_HHMMSS>.pt
# Nada é sobrescrito — cada treino gera um arquivo novo, e a inferência usa
# por padrão o MAIS RECENTE. Assim dá para reproduzir/auditar qualquer rodada
# antiga apontando o caminho exato.
#
# Prefixos usados: "ae_h0_multi" (H0) e "ae_h1_multi" (H1 fine-tuned).

FMT_TS = "%Y%m%d_%H%M%S"
# aceita 4 dígitos (HHMM, formato antigo) ou 6 (HHMMSS, formato atual)
_RE_TS = re.compile(r"_(\d{8})_(\d{4}|\d{6})\.pt$")


def _ts_ordenavel(nome: str) -> Optional[str]:
    """Extrai o timestamp do nome como string comparável 'AAAAMMDDHHMMSS'.
    Normaliza o formato antigo sem segundos para não quebrar a ordenação."""
    m = _RE_TS.search(os.path.basename(nome))
    if not m:
        return None
    data, hora = m.group(1), m.group(2)
    return data + (hora if len(hora) == 6 else hora + "00")


def listar_modelos(dir_modelos: str, prefixo: str = "ae_h0_multi") -> List[Tuple[str, str]]:
    """Modelos do prefixo em `dir_modelos`, do mais ANTIGO ao mais RECENTE.
    Devolve [(timestamp_ordenavel, caminho_completo)]."""
    if not os.path.isdir(dir_modelos):
        return []
    achados = []
    for nome in os.listdir(dir_modelos):
        if not (nome.startswith(prefixo + "_") and nome.endswith(".pt")):
            continue
        ts = _ts_ordenavel(nome)
        if ts:
            achados.append((ts, os.path.join(dir_modelos, nome)))
    return sorted(achados)


def caminho_modelo_mais_recente(dir_modelos: str, prefixo: str = "ae_h0_multi") -> str:
    """Caminho do modelo mais recente do prefixo. Erro claro se não houver."""
    achados = listar_modelos(dir_modelos, prefixo)
    if not achados:
        raise FileNotFoundError(
            f"nenhum modelo '{prefixo}_<AAAAMMDD_HHMMSS>.pt' em {dir_modelos!r}. "
            f"Rode o notebook de treino correspondente antes da inferência.")
    return achados[-1][1]


def salvar_modelo(model, caminho: str, cfg: ConfigMulti, versao: str) -> None:
    """Grava num caminho EXATO (uso interno; prefira `salvar_modelo_ts`)."""
    if not _TORCH_OK:
        raise RuntimeError("PyTorch indisponível.")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__, "versao": versao}, caminho)


def salvar_modelo_ts(model, dir_modelos: str, cfg: ConfigMulti,
                     prefixo: str = "ae_h0_multi", agora=None) -> Tuple[str, str]:
    """Grava um modelo NOVO com timestamp (nunca sobrescreve).
    Devolve (caminho, versao), onde versao == nome do arquivo sem '.pt'."""
    agora = agora or datetime.datetime.now()
    versao = f"{prefixo}_{agora.strftime(FMT_TS)}"
    os.makedirs(dir_modelos, exist_ok=True)
    caminho = os.path.join(dir_modelos, versao + ".pt")
    salvar_modelo(model, caminho, cfg, versao)
    return caminho, versao


def carregar_modelo(caminho: str, device=None):
    """Carrega de um caminho EXATO. Devolve (model, cfg, versao)."""
    if not _TORCH_OK:
        raise RuntimeError("PyTorch indisponível.")
    ck = torch.load(caminho, map_location=device or "cpu", weights_only=False)
    cfg = ConfigMulti(**{k: v for k, v in ck["cfg"].items() if k in ConfigMulti().__dict__})
    if device:
        cfg.device = device
    model = DenoisingAECond(cfg).to(cfg.device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, cfg, ck.get("versao", "?")


def carregar_modelo_mais_recente(dir_modelos: str, prefixo: str = "ae_h0_multi",
                                 device=None):
    """Carrega o modelo mais recente do prefixo.
    Devolve (model, cfg, versao, caminho) — o caminho é devolvido para que o
    driver possa FIXÁ-LO e repassar aos workers (ver nb_02/nb_03b)."""
    caminho = caminho_modelo_mais_recente(dir_modelos, prefixo)
    model, cfg, versao = carregar_modelo(caminho, device=device)
    return model, cfg, versao, caminho

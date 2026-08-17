"""
test_finetune.py — prova o RETREINO do H1 com supervisão do engenheiro,
com os 5 canais (valor, derivada, desvio, comp, d_comp — sem is_sicoi).

Cenário: platô operacional SUSTENTADO e NÃO explicado por topologia (d_comp=0).
O H0 não deve tratá-lo como estrutural, mas a rede o segue em parte -> superestima
a máxima. O engenheiro informa o valor correto V; o fine-tuning com a perda de
quantil anual puxa a máxima do H1 para perto de V.

Rodar: python demanda_maxima/test_finetune.py
"""

import os, sys, copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core, core_multi as cm


def serie(seed, plato=0.0):
    rng = np.random.default_rng(seed)
    n = 144 * 365
    t = np.arange(n); dia = t % 144
    estrutural = (10 + 0.000008 * t + 1.2 * np.sin(2 * np.pi * t / (144 * 365))
                  + 3.0 * np.clip(np.sin(2 * np.pi * (dia - 30) / 144), 0, None))
    obs = estrutural + 0.15 * rng.standard_normal(n)
    if plato > 0:                        # platô sustentado (~70 dias), sem Δkm
        ini = 144 * 180; fim = ini + 144 * 70
        obs[ini:fim] += plato
    comp = np.zeros(n, dtype=np.float32)   # extensão normalizada (flat aqui)
    dcomp = np.zeros(n, dtype=np.float32)  # sem manobra topológica neste teste
    return obs.astype(np.float32), comp, dcomp, estrutural.astype(np.float32)


def janelas(obs, comp, dcomp, scaler, cfg):
    vn = scaler.transform(obs)
    W = lambda a: core.make_windows(a, cfg.L, cfg.L)
    return W(vn), W(comp), W(dcomp)


def main():
    if not cm._TORCH_OK:
        print("SKIP: torch ausente."); return 0
    cfg = cm.ConfigMulti(epochs=25, seed=3)

    # --- treino H0 multicanal ---
    VW, CW, DW = [], [], []
    for s in range(3):
        obs, comp, dcomp, _ = serie(seed=s)
        sc = core.RobustScaler().fit(obs)
        vw, cw, dw = janelas(obs, comp, dcomp, sc, cfg)
        VW.append(vw); CW.append(cw); DW.append(dw)
    VW = np.concatenate(VW); CW = np.concatenate(CW); DW = np.concatenate(DW)
    print(f"treinando H0 em {VW.shape[0]} janelas (C={cfg.n_canais})...")
    model = cm.train_h0_multi(VW, CW, DW, cfg)

    # --- alimentador de teste: platô sustentado não-topológico ---
    obs, comp, dcomp, estrutural = serie(seed=99, plato=9.0)
    sc = core.RobustScaler().fit(obs)
    verdade = float(np.quantile(estrutural, cfg.prob_max))
    max_h0 = cm.maxima_anual_multi(model, obs, comp, dcomp, sc, cfg)
    print(f"H0: verdade={verdade:.2f}  H0={max_h0:.2f}")

    # --- especialista informa o valor correto ---
    V_norm = float(sc.transform(np.array([verdade]))[0])
    vw, cw, dw = janelas(obs, comp, dcomp, sc, cfg)
    rotulados = [{"canais": cm.canais_de_janelas(vw, cw, dw, cfg), "V_norm": V_norm}]
    pool = {"valor": VW, "comp": CW, "dcomp": DW}

    model_h1 = cm.fine_tune_h1(copy.deepcopy(model), rotulados, pool, cfg,
                               lam=8.0, epochs=60, lr=5e-4, rot_batch=16)
    max_h1 = cm.maxima_anual_multi(model_h1, obs, comp, dcomp, sc, cfg)

    err_h0 = abs(max_h0 - verdade); err_h1 = abs(max_h1 - verdade)
    print(f"H1 (retreinado): {max_h1:.2f}  -> erro H0={err_h0:.2f}  erro H1={err_h1:.2f}")
    ok = np.isfinite(max_h1) and err_h1 < err_h0
    print(f"[retreino] H1 mais perto do valor do especialista? {'PASS' if ok else 'FAIL'}")
    print("\n==== RESUMO:", "PASS" if ok else "FAIL", "====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

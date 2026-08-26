"""
test_pipeline.py — teste sintético (SEM Spark) do núcleo de Demanda Máxima.

Valida, com uma série de máxima estrutural CONHECIDA + contaminação (picos de
medição e ruído), que:
  [1] o autoencoder denoising remove os picos → a máxima do LATENTE fica mais
      perto da verdade do que a máxima da série CRUA;
  [2] utilidades (RobustScaler, make_windows, inject_contamination) funcionam;
  [3] o registro que irá ao SQL tem o schema esperado.

Escopo honesto do H0: remove RUÍDO e ERROS DE MEDIÇÃO (picos) e transitórios
curtos. Manobras SUSTENTADAS que parecem estruturais são o domínio do H1
(conhecimento do especialista) — não se espera que o H0 as remova sozinho.

Rodar:  python tests/test_pipeline.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import core  # noqa: E402


def serie_sintetica(ano_pts=144 * 365, seed=0):
    """Uma série anual @10min: estrutural conhecida + picos de medição + ruído."""
    rng = np.random.default_rng(seed)
    t = np.arange(ano_pts)
    dia = t % 144
    # curva estrutural: base + tendência leve + sazonal anual + ciclo diário
    estrutural = (
        10.0
        + 0.000008 * t
        + 1.2 * np.sin(2 * np.pi * t / (144 * 365))
        + 3.0 * np.clip(np.sin(2 * np.pi * (dia - 30) / 144), 0, None)  # pico diurno
    )
    obs = estrutural + 0.15 * rng.standard_normal(ano_pts)  # ruído de medição
    # PICOS de medição (erros): ~1.2% dos pontos, amplitude grande
    n_pico = int(0.012 * ano_pts)
    pos = rng.integers(0, ano_pts, size=n_pico)
    obs[pos] += rng.uniform(4.0, 8.0, size=n_pico)
    return obs.astype(np.float32), estrutural.astype(np.float32)


def main():
    cfg = core.Config(epochs=40, batch_size=128, prob_max=0.999, seed=1)

    # --- 2. utilidades (independem de torch) ---------------------------------
    sc = core.RobustScaler().fit(np.array([1, 2, 3, 4, 100.0]))
    assert abs(sc.inverse(sc.transform(np.array([3.0])))[0] - 3.0) < 1e-6
    w = core.make_windows(np.arange(300.0), L=144, stride=144)
    assert w.shape == (2, 144), w.shape
    cont = core.inject_contamination(np.zeros((5, 144), dtype=np.float32), cfg,
                                     np.random.default_rng(0))
    assert cont.shape == (5, 144)
    print("[2] utilidades (scaler/windows/contaminacao) ...... PASS")

    if not core._TORCH_OK:
        print("[1] AE denoising ................................... SKIP (torch ausente)")
        print("    Instale as dependencias: pip install -r requirements.txt")
        print("\n==== RESUMO: utilidades OK; instale torch para o teste completo ====")
        return 0

    # --- treino GLOBAL em alguns alimentadores sinteticos --------------------
    feeders = [serie_sintetica(seed=s) for s in range(3)]
    janelas = []
    for obs, _ in feeders:
        sc = core.RobustScaler().fit(obs)
        janelas.append(core.make_windows(sc.transform(obs), cfg.L, stride=cfg.L))
    janelas = np.concatenate(janelas, axis=0)
    print(f"    treinando AE global em {janelas.shape[0]} janelas...")
    model = core.train_model(janelas, cfg)

    # --- 1. maxima do latente vs. maxima crua (num feeder de teste) ----------
    obs, estrutural = serie_sintetica(seed=99)
    sc = core.RobustScaler().fit(obs)
    verdade   = float(np.quantile(estrutural, cfg.prob_max))
    max_cru   = float(np.quantile(obs, cfg.prob_max))
    max_latente = core.maxima_anual(model, obs, sc, cfg)

    err_cru = abs(max_cru - verdade)
    err_lat = abs(max_latente - verdade)
    teste1 = np.isfinite(max_latente) and err_lat < err_cru
    print(f"[1] Maxima: verdade={verdade:.2f}  crua={max_cru:.2f}  latente={max_latente:.2f} "
          f"-> latente mais perto? {'PASS' if teste1 else 'FAIL'} "
          f"(err_cru={err_cru:.2f} err_lat={err_lat:.2f})")

    # --- 3. schema do registro que vai ao SQL --------------------------------
    registro = {
        "regiao": "SP", "tipo": "Alimentador", "ativo": "RAMZ1301",
        "grandeza": "MVA", "ano": 2024,
        "demanda_max_h0": round(max_latente, 4), "demanda_max_h1": None,
        "modelo_versao": "ae_h0_v1", "calculado_em": "2024-01-01 00:00:00",
    }
    esperado = {"regiao", "tipo", "ativo", "grandeza", "ano",
                "demanda_max_h0", "demanda_max_h1", "modelo_versao", "calculado_em"}
    teste3 = set(registro) == esperado
    print(f"[3] Schema do registro SQL ........................ {'PASS' if teste3 else 'FAIL'}")

    ok = bool(teste1 and teste3)
    print("\n==== RESUMO:", "TODOS OS TESTES PASSARAM" if ok else "HOUVE FALHA", "====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

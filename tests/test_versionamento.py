"""
test_versionamento.py — garante o contrato de versionamento dos modelos:

  [1] cada treino grava um arquivo NOVO `<prefixo>_<AAAAMMDD_HHMMSS>.pt`
      (nunca sobrescreve o anterior);
  [2] `carregar_modelo_mais_recente` devolve sempre o treino MAIS RECENTE;
  [3] a ordenação é cronológica de verdade (inclusive misturando o formato
      antigo sem segundos com o atual, e virando o ano);
  [4] prefixos diferentes (H0 x H1) não se misturam;
  [5] diretório sem modelos falha com mensagem clara, não com erro obscuro.

Rodar: python tests/test_versionamento.py
"""

import datetime
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import core_multi as cm  # noqa: E402


def _toca(dirpath, nome):
    """Cria um .pt vazio só para testar a lógica de nomes/ordenação."""
    caminho = os.path.join(dirpath, nome)
    with open(caminho, "wb") as f:
        f.write(b"")
    return caminho


def main():
    falhas = []

    # --- [3] ordenação cronológica, incluindo formato antigo (HHMM) ----------
    with tempfile.TemporaryDirectory() as d:
        _toca(d, "ae_h0_multi_20251231_2359.pt")    # formato antigo, ano anterior
        _toca(d, "ae_h0_multi_20260826_1916.pt")    # formato antigo
        esperado = _toca(d, "ae_h0_multi_20260826_191700.pt")  # formato atual, 1 min depois
        _toca(d, "ae_h0_multi_20260101_000000.pt")
        _toca(d, "ignorar.pt")                      # sem timestamp -> ignorado
        _toca(d, "ae_h0_multi_sem_ts.pt")           # prefixo certo, ts inválido -> ignorado

        ordem = [os.path.basename(p) for _, p in cm.listar_modelos(d, "ae_h0_multi")]
        esperada = ["ae_h0_multi_20251231_2359.pt", "ae_h0_multi_20260101_000000.pt",
                    "ae_h0_multi_20260826_1916.pt", "ae_h0_multi_20260826_191700.pt"]
        ok3 = ordem == esperada
        print(f"[3] ordenacao cronologica (formatos misturados) ... {'PASS' if ok3 else 'FAIL'}")
        if not ok3:
            falhas.append(f"ordem obtida: {ordem}")

        ok2 = cm.caminho_modelo_mais_recente(d, "ae_h0_multi") == esperado
        print(f"[2] mais recente eh o ultimo treino ............... {'PASS' if ok2 else 'FAIL'}")
        if not ok2:
            falhas.append("caminho_modelo_mais_recente devolveu o arquivo errado")

        # --- [4] prefixos não se misturam -----------------------------------
        h1 = _toca(d, "ae_h1_multi_20200101_000000.pt")   # H1 antigo
        ok4 = (cm.caminho_modelo_mais_recente(d, "ae_h1_multi") == h1
               and cm.caminho_modelo_mais_recente(d, "ae_h0_multi") == esperado
               and len(cm.listar_modelos(d, "ae_h1_multi")) == 1)
        print(f"[4] H0 e H1 nao se misturam ...................... {'PASS' if ok4 else 'FAIL'}")
        if not ok4:
            falhas.append("prefixos H0/H1 se misturaram")

    # --- [5] diretório vazio / inexistente ----------------------------------
    with tempfile.TemporaryDirectory() as d:
        try:
            cm.caminho_modelo_mais_recente(d, "ae_h0_multi")
            ok5 = False
        except FileNotFoundError as e:
            ok5 = "ae_h0_multi" in str(e)
    ok5 = ok5 and cm.listar_modelos(os.path.join(d, "nao_existe"), "ae_h0_multi") == []
    print(f"[5] diretorio sem modelos -> erro claro ........... {'PASS' if ok5 else 'FAIL'}")
    if not ok5:
        falhas.append("erro de diretorio vazio nao eh claro")

    # --- [1] salvar_modelo_ts nao sobrescreve -------------------------------
    if not cm._TORCH_OK:
        print("[1] salvar_modelo_ts .............................. SKIP (torch ausente)")
    else:
        with tempfile.TemporaryDirectory() as d:
            cfg = cm.ConfigMulti(L=144, latent=8)
            model = cm.DenoisingAECond(cfg)
            base = datetime.datetime(2026, 8, 26, 19, 16, 0)
            p1, v1 = cm.salvar_modelo_ts(model, d, cfg, "ae_h0_multi", agora=base)
            p2, v2 = cm.salvar_modelo_ts(model, d, cfg, "ae_h0_multi",
                                         agora=base + datetime.timedelta(seconds=5))
            existem = os.path.exists(p1) and os.path.exists(p2)
            distintos = p1 != p2 and v1 != v2
            recente = cm.caminho_modelo_mais_recente(d, "ae_h0_multi") == p2
            # o modelo mais recente recarrega e traz a versao gravada
            _m, _cfg, _v, _c = cm.carregar_modelo_mais_recente(d, "ae_h0_multi", device="cpu")
            recarrega = (_v == v2 and _c == p2 and _cfg.L == cfg.L
                         and _m.enc[0].weight.shape[1] == cfg.n_canais)
            ok1 = existem and distintos and recente and recarrega
            print(f"[1] salvar_modelo_ts nao sobrescreve + recarrega .. {'PASS' if ok1 else 'FAIL'}")
            if not ok1:
                falhas.append(f"existem={existem} distintos={distintos} "
                              f"recente={recente} recarrega={recarrega}")

    print("\n==== RESUMO:", "TODOS OS TESTES PASSARAM" if not falhas else "HOUVE FALHA", "====")
    for f in falhas:
        print("   -", f)
    return 0 if not falhas else 1


if __name__ == "__main__":
    sys.exit(main())

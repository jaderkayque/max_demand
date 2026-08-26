# Databricks notebook source
# MAGIC %md
# MAGIC # nb_03 (alternativo) — H1 por RETREINO da rede com supervisão do engenheiro
# MAGIC A alma do projeto: em vez de calibrar o escalar, aqui a PRÓPRIA REDE é
# MAGIC retreinada (fine-tuning) para que o **pico anual da reconstrução** dos
# MAGIC alimentadores rotulados case com o valor informado pelo especialista
# MAGIC (perda de quantil anual, `core_multi.fine_tune_h1`). O conhecimento passa a
# MAGIC morar nos pesos e generaliza para os demais alimentadores.
# MAGIC
# MAGIC Canais: valor, derivada, desvio, comp, d_comp (**sem `is_sicoi`** — decisão
# MAGIC de projeto). O fine-tuning amostra MINI-BATCHES dos exemplos rotulados
# MAGIC (`rot_batch`), escalando para milhares de rótulos.
# MAGIC
# MAGIC **Avaliação:** este notebook é de PRODUÇÃO. Métricas de pesquisa devem vir
# MAGIC do protocolo experimental (teste congelado por alimentador, rotulados de
# MAGIC teste jamais usados no fine-tuning) — nunca das predições sobre os próprios
# MAGIC pares que supervisionaram o retreino.
# MAGIC
# MAGIC Requer o H0 MULTIVARIADO (nb_01/nb_02 atualizados para `core_multi`).
# MAGIC Cluster shared: sem `sparkContext.broadcast` e sem JVM (MERGE via pymssql).

# COMMAND ----------
# MAGIC %pip install torch numpy pandas pymssql

# COMMAND ----------
import os, sys
import numpy as np
import pandas as pd
from pyspark.sql.functions import lit, rand
from pyspark.sql.types import (StructType, StructField, StringType, IntegerType,
                               DoubleType, BinaryType)

dbutils.widgets.text("repo_dir", "/Workspace/Shared/Servidores/Servidor SP/James/max_demand")
dbutils.widgets.text("medicao_path", "/Volumes/poseidon_uc/group_uc/ddpe/medicao/parquet")
dbutils.widgets.text("grandeza", "MVA")
dbutils.widgets.text("model_h0_path", "")   # vazio = usa o H0 mais recente de modelo/
dbutils.widgets.text("secret_scope", "sqlserver")
dbutils.widgets.text("n_pool", "200")
dbutils.widgets.text("max_janelas_pool", "40")   # janelas (dias) por alimentador-ano do pool

REPO_DIR  = dbutils.widgets.get("repo_dir")
MEDICAO   = dbutils.widgets.get("medicao_path")
GRANDEZA  = dbutils.widgets.get("grandeza")
H0_PATH   = dbutils.widgets.get("model_h0_path").strip()
SCOPE     = dbutils.widgets.get("secret_scope")
N_POOL    = int(dbutils.widgets.get("n_pool"))
MAX_JANELAS_POOL = int(dbutils.widgets.get("max_janelas_pool"))
MODELO_DIR = os.path.join(REPO_DIR, "modelo")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
import core, core_multi as cm            # noqa: E402
import databricks_io as io               # noqa: E402

url, props = io.jdbc_conf(dbutils, SCOPE)


# --- helpers de leitura (mesmo padrão do nb_01/nb_02 multivariado) -----------
def ler_alm(regiao):
    df = (spark.read.parquet(f"{MEDICAO}/{regiao}/ALM").withColumn("regiao", lit(regiao)))
    return df.selectExpr("regiao", "ALM as ativo", "DATAS",
                         f"{GRANDEZA} as valor", "YEAR as ano")

# COMMAND ----------
# MAGIC %md ## 1. Carrega o H0 multivariado e os rótulos do especialista

# COMMAND ----------
# resolve o H0 UMA VEZ no driver (vazio = mais recente de modelo/)
if not H0_PATH:
    H0_PATH = cm.caminho_modelo_mais_recente(MODELO_DIR, "ae_h0_multi")
model, cfg, versao_h0 = cm.carregar_modelo(H0_PATH, device="cpu")
print("H0 de partida:", H0_PATH, "| versao:", versao_h0)

# extensão MENSAL do circuito (canais comp/d_comp), normalizada com as stats do modelo
COMP_MAP, DCOMP_MAP, _, _, _ = io.carregar_ext_rede(spark, url, props)
def canais_comp(datas, regiao, ativo):
    d = pd.to_datetime(datas)
    return cm.alinhar_comprimento(d.dt.year, d.dt.month, regiao, ativo, COMP_MAP, DCOMP_MAP, cfg)

lb = io.ler_tabela(spark, url, props, "james.DEMANDA_MAXIMA_TREINO").toPandas()
lb = (lb.sort_values("timestamp")
        .groupby(["regiao", "ativo", "grandeza", "ano"], as_index=False)
        .agg(valor_correto=("valor_correto", "last")))
lb = lb[lb["grandeza"] == GRANDEZA]
print("feeder-anos rotulados:", len(lb))

# COMMAND ----------
# MAGIC %md ## 2. Monta os exemplos rotulados (canais do ano + V normalizado)

# COMMAND ----------
med = ler_alm("SP").unionByName(ler_alm("ES"), allowMissingColumns=True)

rotulados = []
for _, r in lb.iterrows():
    pdf = (med.filter((med.regiao == r["regiao"]) & (med.ativo == r["ativo"]) &
                      (med.ano == int(r["ano"])))
              .select("DATAS", "valor").orderBy("DATAS").toPandas())
    if pdf.empty:
        continue
    valor = pdf["valor"].to_numpy(float)
    sc = core.RobustScaler().fit(valor)
    comp_full, dcomp_full = canais_comp(pdf["DATAS"], r["regiao"], r["ativo"])
    vw = core.make_windows(sc.transform(valor), cfg.L, cfg.L)
    cw = core.make_windows(comp_full, cfg.L, cfg.L)
    dw = core.make_windows(dcomp_full, cfg.L, cfg.L)
    rotulados.append({
        "canais": cm.canais_de_janelas(vw, cw, dw, cfg),
        "V_norm": float(sc.transform(np.array([r["valor_correto"]]))[0]),
    })
print("exemplos rotulados prontos:", len(rotulados))

# COMMAND ----------
# MAGIC %md ## 3. Pool de reconstrução (janelas montadas NOS WORKERS) e FINE-TUNING
# MAGIC Trazer a série bruta ao driver (`.toPandas()`) estoura
# MAGIC `spark.driver.maxResultSize`; cada grupo monta e amostra suas janelas no
# MAGIC worker (`core_multi.montar_janelas_grupo`) e devolve blobs float32.

# COMMAND ----------
# amostra ALEATÓRIA do pool, com semente registrada (`.limit(N)` puro tem viés)
ativos = [x.ativo for x in (med.select("ativo").distinct()
                               .orderBy(rand(seed=cfg.seed)).limit(N_POOL).collect())]
amostra_pool = med.filter(med.ativo.isin(ativos))

win_schema = StructType([StructField("valor_w", BinaryType()),
                         StructField("comp_w", BinaryType()),
                         StructField("dcomp_w", BinaryType())])

def montar_janelas_pool(pdf: pd.DataFrame) -> pd.DataFrame:
    import numpy as _np, core_multi as _cm
    pdf = pdf.sort_values("DATAS")
    r = pdf.iloc[0]
    res = _cm.montar_janelas_grupo(pdf["valor"].to_numpy(float), pdf["DATAS"],
                                   r["regiao"], r["ativo"], int(r["ano"]),
                                   COMP_MAP, DCOMP_MAP, cfg,
                                   max_janelas=MAX_JANELAS_POOL, seed_extra=1)
    if res is None:
        return pd.DataFrame(columns=["valor_w", "comp_w", "dcomp_w"])
    vw, cw, dw = res
    return pd.DataFrame({"valor_w": [vw.tobytes()], "comp_w": [cw.tobytes()],
                         "dcomp_w": [dw.tobytes()]})

linhas = (amostra_pool.groupBy("regiao", "ativo", "ano")
                      .applyInPandas(montar_janelas_pool, schema=win_schema).collect())

def _des(col):
    return np.concatenate([np.frombuffer(r[col], dtype=np.float32).reshape(-1, cfg.L)
                           for r in linhas])

pool = {"valor": _des("valor_w"), "comp": _des("comp_w"), "dcomp": _des("dcomp_w")}

model_h1 = cm.fine_tune_h1(model, rotulados, pool, cfg,
                           lam=8.0, epochs=60, lr=5e-4, rot_batch=16)

# grava um arquivo NOVO com timestamp — não sobrescreve H1 anteriores
H1_PATH, VERSAO = cm.salvar_modelo_ts(model_h1, MODELO_DIR, cfg, prefixo="ae_h1_multi")
print("H1 salvo:", H1_PATH, "| versao:", VERSAO)

# COMMAND ----------
# MAGIC %md ## 4. Recalcula a máxima H1 de TODOS os alimentadores (rede retreinada)

# COMMAND ----------
out_schema = StructType([
    StructField("regiao", StringType()), StructField("ativo", StringType()),
    StructField("grandeza", StringType()), StructField("ano", IntegerType()),
    StructField("demanda_max_h1", DoubleType()),
])

_CACHE = {}
def _modelo():
    if "m" not in _CACHE:
        _CACHE["m"], _CACHE["cfg"], _ = cm.carregar_modelo(H1_PATH, device="cpu")
    return _CACHE["m"], _CACHE["cfg"]

def calc_h1(pdf: pd.DataFrame) -> pd.DataFrame:
    import numpy as _np, pandas as _pd, core as _core, core_multi as _cm
    m, _cfg = _modelo()
    pdf = pdf.sort_values("DATAS")
    valor = pdf["valor"].to_numpy(float)
    sc = _core.RobustScaler().fit(valor)
    r = pdf.iloc[0]
    d = _pd.to_datetime(pdf["DATAS"])
    comp_full, dcomp_full = _cm.alinhar_comprimento(d.dt.year, d.dt.month,
                                                    r["regiao"], r["ativo"], COMP_MAP, DCOMP_MAP, _cfg)
    v = _cm.maxima_anual_multi(m, valor, comp_full, dcomp_full, sc, _cfg)
    return pd.DataFrame([{"regiao": r["regiao"], "ativo": r["ativo"], "grandeza": GRANDEZA,
                          "ano": int(r["ano"]),
                          "demanda_max_h1": float(v) if _np.isfinite(v) else None}])

res = med.groupBy("regiao", "ativo", "ano").applyInPandas(calc_h1, schema=out_schema)
io.escrever_tabela(res, url, props, "james.DEMANDA_MAXIMA_H1_STG", mode="overwrite")

# COMMAND ----------
# MAGIC %md ## 5. MERGE em james.DEMANDA_MAXIMA (via pymssql — shared cluster)

# COMMAND ----------
io.executar_sql_pymssql("""
MERGE james.DEMANDA_MAXIMA AS d
USING james.DEMANDA_MAXIMA_H1_STG AS s
  ON d.regiao=s.regiao AND d.ativo=s.ativo AND d.grandeza=s.grandeza AND d.ano=s.ano
WHEN MATCHED THEN UPDATE SET d.demanda_max_h1 = s.demanda_max_h1, d.calculado_em = GETDATE();
""", dbutils, scope=SCOPE)
print("H1 (rede retreinada) gravado em james.DEMANDA_MAXIMA")

# COMMAND ----------
# MAGIC %md ## 6. Verificação (rodou corretamente?)
# MAGIC Confere que o H1 novo é o mais recente, que o fine-tuning de fato MUDOU o
# MAGIC modelo (H1 ≠ H0) e que o H1 ficou mais perto dos rótulos do especialista.
# MAGIC
# MAGIC ⚠️ Esta comparação é IN-SAMPLE (usa os pares que supervisionaram o retreino):
# MAGIC serve como *sanity check* de produção, **não** como métrica de desempenho.
# MAGIC Métrica honesta exige teste congelado — ver `docs/PLANO_PESQUISA_DISSERTACAO.md`.

# COMMAND ----------
print("modelos H1 no diretorio (antigo -> recente):")
for ts, p in cm.listar_modelos(MODELO_DIR, "ae_h1_multi"):
    print("  ", ts, os.path.basename(p))
assert cm.caminho_modelo_mais_recente(MODELO_DIR, "ae_h1_multi") == H1_PATH

dm = io.ler_tabela(spark, url, props, "james.DEMANDA_MAXIMA").toPandas()
lb_chk = lb.rename(columns={"valor_correto": "real"})
cmp = dm.merge(lb_chk[["regiao", "ativo", "ano", "real"]], on=["regiao", "ativo", "ano"], how="inner")
cmp = cmp.dropna(subset=["real", "demanda_max_h0", "demanda_max_h1"])

if len(cmp):
    err_h0 = (cmp["demanda_max_h0"] - cmp["real"]).abs() / cmp["real"].abs()
    err_h1 = (cmp["demanda_max_h1"] - cmp["real"]).abs() / cmp["real"].abs()
    print(f"\npares rotulados comparados: {len(cmp)}  (IN-SAMPLE)")
    print(f"erro relativo mediano  H0: {err_h0.median():.2%}   H1: {err_h1.median():.2%}")
    print(f"dentro de 10%          H0: {(err_h0 <= .10).mean():.1%}   H1: {(err_h1 <= .10).mean():.1%}")
    print("esperado: H1 melhor que H0 nestes pares; se nao for, revise lam/epochs.")
else:
    print("\nsem pares rotulados para comparar — verifique james.DEMANDA_MAXIMA_TREINO")

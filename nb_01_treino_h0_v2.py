# Databricks notebook source
# MAGIC %md
# MAGIC # nb_01 — Treino do H0 MULTIVARIADO (autoencoder denoising condicional)
# MAGIC Canais por janela diária: **valor, derivada, desvio, comp, d_comp**
# MAGIC (extensão mensal do circuito de `james.EXT_REDE_MT`).
# MAGIC `is_sicoi` foi removido do pipeline (decisão de projeto) — sem máscara de
# MAGIC perda; a perda é L1 simples. Salva o artefato p/ o nb_02.

# COMMAND ----------

# MAGIC %pip install torch numpy pandas

# COMMAND ----------

import os, sys, datetime
import numpy as np
from pyspark.sql.functions import lit

sys.path.insert(0, "/Workspace/Shared/Servidores/Servidor SP/James/ML_Demanda_Maxima")
import core, core_multi as cm            # noqa: E402
import databricks_io_v2 as io               # noqa: E402

dbutils.widgets.text("medicao_path", "/Volumes/poseidon_uc/group_uc/ddpe/medicao/parquet")
dbutils.widgets.text("grandeza", "IMAX")
dbutils.widgets.text("n_alimentadores_amostra", "300")
dbutils.widgets.text("max_janelas_por_grupo", "60")   # janelas (dias) por alimentador-ano
dbutils.widgets.text("model_path", "/Workspace/Shared/Servidores/Servidor SP/James/ML_Demanda_Maxima/ae_h0_multi.pt")
dbutils.widgets.text("secret_scope", "sqlserver")

MEDICAO    = dbutils.widgets.get("medicao_path")
GRANDEZA   = dbutils.widgets.get("grandeza")
N_SAMPLE   = int(dbutils.widgets.get("n_alimentadores_amostra"))
MAX_JANELAS = int(dbutils.widgets.get("max_janelas_por_grupo"))
MODEL_PATH = dbutils.widgets.get("model_path")
SCOPE      = dbutils.widgets.get("secret_scope")

cfg = cm.ConfigMulti(epochs=40, batch_size=256, seed=1)
VERSAO = "ae_h0_multi_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")

import pandas as pd
url, props = io.jdbc_conf(dbutils, SCOPE)
# extensão MENSAL do circuito (km) + variação Δkm + stats globais (guardadas no cfg)
COMP_MAP, DCOMP_MAP, cfg.comp_mean, cfg.comp_std, cfg.dcomp_std = io.carregar_ext_rede(spark, url, props)
print(f"extensao: {len(COMP_MAP)} chaves | media={cfg.comp_mean:.2f} std={cfg.comp_std:.2f} dstd={cfg.dcomp_std:.2f}")

def ler_alm(regiao):
    df = spark.read.parquet(f"{MEDICAO}/{regiao}/ALM").withColumn("regiao", lit(regiao))
    return df.selectExpr("regiao", "ALM as ativo", "DATAS",
                         f"{GRANDEZA} as valor", "YEAR as ano")

# COMMAND ----------

# MAGIC %md ## 1. Amostra de alimentadores e montagem das janelas NOS WORKERS
# MAGIC Trazer a série bruta ao driver (`.toPandas()`) estoura
# MAGIC `spark.driver.maxResultSize` (timestamps + strings por linha). Em vez disso,
# MAGIC cada (regiao, ativo, ano) monta e AMOSTRA suas janelas no worker
# MAGIC (`core_multi.montar_janelas_grupo`) e devolve só blobs float32 compactos.

# COMMAND ----------

med = ler_alm("SP").unionByName(ler_alm("ES"), allowMissingColumns=True)

# amostra ALEATÓRIA de alimentadores, com semente registrada
# (`.limit(N)` sem ordenação devolve "os N primeiros do Spark" — viés de seleção)
from pyspark.sql.functions import rand
from pyspark.sql.types import StructType, StructField, BinaryType
ativos = [r.ativo for r in (med.select("ativo").distinct()
                               .orderBy(rand(seed=cfg.seed)).limit(N_SAMPLE).collect())]
amostra = med.filter(med.ativo.isin(ativos))

win_schema = StructType([StructField("valor_w", BinaryType()),
                         StructField("comp_w", BinaryType()),
                         StructField("dcomp_w", BinaryType())])

def montar_janelas(pdf: pd.DataFrame) -> pd.DataFrame:
    import numpy as _np, core_multi as _cm
    pdf = pdf.sort_values("DATAS")
    r = pdf.iloc[0]
    res = _cm.montar_janelas_grupo(pdf["valor"].to_numpy(float), pdf["DATAS"],
                                   r["regiao"], r["ativo"], int(r["ano"]),
                                   COMP_MAP, DCOMP_MAP, cfg, max_janelas=MAX_JANELAS)
    if res is None:
        return pd.DataFrame(columns=["valor_w", "comp_w", "dcomp_w"])
    vw, cw, dw = res
    return pd.DataFrame({"valor_w": [vw.tobytes()], "comp_w": [cw.tobytes()],
                         "dcomp_w": [dw.tobytes()]})

linhas = (amostra.groupBy("regiao", "ativo", "ano")
                 .applyInPandas(montar_janelas, schema=win_schema).collect())

def _des(col):
    return np.concatenate([np.frombuffer(r[col], dtype=np.float32).reshape(-1, cfg.L)
                           for r in linhas])

valor_w, comp_w, dcomp_w = _des("valor_w"), _des("comp_w"), _des("dcomp_w")
print("janelas de treino:", valor_w.shape, "| canais:", cfg.n_canais)

# COMMAND ----------

# MAGIC %md ## 2. Treino do autoencoder denoising condicional (global)

# COMMAND ----------

model = cm.train_h0_multi(valor_w, comp_w, dcomp_w, cfg)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)   # se /dbfs não acessível, use uma Volume
cm.salvar_modelo(model, MODEL_PATH, cfg, VERSAO)          # cfg guarda comp_mean/comp_std
print("modelo salvo:", MODEL_PATH, "versao", VERSAO)
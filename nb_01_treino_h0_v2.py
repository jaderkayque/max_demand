# Databricks notebook source
# MAGIC %md
# MAGIC # nb_01 — Treino do H0 MULTIVARIADO (autoencoder denoising condicional)
# MAGIC Canais por janela diária: **valor, is_sicoi, derivada, desvio**;
# MAGIC estático (no gargalo): **comprimento_rede** (de `james.EXT_REDE_MT`, por ano).
# MAGIC Perda de reconstrução **mascarada por is_sicoi**. Salva o artefato p/ o nb_02.

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
dbutils.widgets.text("model_path", "/Workspace/Shared/Servidores/Servidor SP/James/ML_Demanda_Maxima/ae_h0_multi.pt")
dbutils.widgets.text("secret_scope", "sqlserver")

MEDICAO    = dbutils.widgets.get("medicao_path")
GRANDEZA   = dbutils.widgets.get("grandeza")
N_SAMPLE   = int(dbutils.widgets.get("n_alimentadores_amostra"))
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
                         f"{GRANDEZA} as valor", "is_sicoi", "YEAR as ano")

def canais_comp(datas, regiao, ativo):
    d = pd.to_datetime(datas)
    return cm.alinhar_comprimento(d.dt.year, d.dt.month, regiao, ativo, COMP_MAP, DCOMP_MAP, cfg)

# COMMAND ----------

# MAGIC %md ## 1. Amostra de alimentadores e montagem dos canais (por alimentador-ano)

# COMMAND ----------

med = ler_alm("SP").unionByName(ler_alm("ES"), allowMissingColumns=True)

ativos = [r.ativo for r in med.select("ativo").distinct().limit(N_SAMPLE).collect()]
amostra = med.filter(med.ativo.isin(ativos))
pdf = (amostra.orderBy("regiao", "ativo", "ano", "DATAS")
              .select("regiao", "ativo", "ano", "DATAS", "valor", "is_sicoi").toPandas())

W = lambda a: core.make_windows(a, cfg.L, cfg.L)
valor_w, sic_w, comp_w, dcomp_w = [], [], [], []
for (reg, at, ano), g in pdf.groupby(["regiao", "ativo", "ano"]):
    v = g["valor"].to_numpy(float)
    if v.size < cfg.L:
        continue
    sc = core.RobustScaler().fit(v)                       # escala por alimentador-ano
    comp_full, dcomp_full = canais_comp(g["DATAS"], reg, at)   # extensão alinhada ao mês
    valor_w.append(W(sc.transform(v))); sic_w.append(W(g["is_sicoi"].fillna(0).to_numpy(float)))
    comp_w.append(W(comp_full)); dcomp_w.append(W(dcomp_full))

valor_w = np.concatenate(valor_w); sic_w = np.concatenate(sic_w)
comp_w = np.concatenate(comp_w); dcomp_w = np.concatenate(dcomp_w) 
print("janelas de treino:", valor_w.shape, "| canais:", cfg.n_canais)

# COMMAND ----------

# MAGIC %md ## 2. Treino do autoencoder denoising condicional (global)

# COMMAND ----------

model = cm.train_h0_multi(valor_w, sic_w, comp_w, dcomp_w, cfg)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)   # se /dbfs não acessível, use uma Volume
cm.salvar_modelo(model, MODEL_PATH, cfg, VERSAO)          # cfg guarda comp_mean/comp_std
print("modelo salvo:", MODEL_PATH, "versao", VERSAO)
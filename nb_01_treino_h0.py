# Databricks notebook source
# MAGIC %md
# MAGIC # nb_01 — Treino do H0 (autoencoder denoising global)
# MAGIC Treina UM modelo global que estima o estado latente da carga (sem manobras
# MAGIC nem erros de medição). Salva o artefato em DBFS para o `nb_02_predict_max`.

# COMMAND ----------

# MAGIC %pip install torch numpy pandas

# COMMAND ----------

import os, sys, glob, datetime
import numpy as np

# torna importável o pacote demanda_maxima (ajuste o caminho ao seu repo/Workspace)
sys.path.insert(0, os.path.dirname(os.path.abspath("__file__")) if "__file__" in dir() else "/Workspace/Shared/Servidores/Servidor SP/James/ML_Demanda_Maxima")
import core  # noqa: E402

# --- parâmetros (widgets) ----------------------------------------------------
#dbutils.widgets.text("medicao_path", "/Volumes/poseidon_uc/group_uc/ddpe/medicao/parquet/")
#dbutils.widgets.text("grandeza", "IMAX")
#dbutils.widgets.text("n_alimentadores_amostra", "300")   # amostra p/ treino global
#dbutils.widgets.text("model_path", "/Workspace/Shared/Servidores/Servidor SP/James/ML_Demanda_Maxima/ae_h0.pt")

MEDICAO = dbutils.widgets.get("medicao_path")
GRANDEZA = dbutils.widgets.get("grandeza")
N_SAMPLE = int(dbutils.widgets.get("n_alimentadores_amostra"))
MODEL_PATH = dbutils.widgets.get("model_path")

cfg = core.Config(epochs=40, batch_size=256, seed=1)
VERSAO = "ae_h0_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")

# COMMAND ----------

# MAGIC %md ## 1. Amostra de janelas diárias de vários alimentadores (ALM)

# COMMAND ----------

from pyspark.sql.functions import lit

def ler_alm(regiao):                       # regiao = "SP" | "ES"
    return (spark.read.parquet(f"{MEDICAO}/{regiao}/ALM")
                 .withColumn("regiao", lit(regiao)))

df = (ler_alm("SP")
      .unionByName(ler_alm("ES"), allowMissingColumns=True)
      .selectExpr("regiao", "ALM as ativo", "DATAS", f"{GRANDEZA} as valor", "YEAR as ano"))

# COMMAND ----------

# amostra de alimentadores para o treino global
ativos = [r.ativo for r in df.select("ativo").distinct().limit(N_SAMPLE).collect()]
amostra = df.filter(df.ativo.isin(ativos))

# traz para o driver (série por alimentador) e monta janelas normalizadas
pdf = amostra.orderBy("ativo", "DATAS").select("ativo", "valor").toPandas()
janelas = []
for ativo, g in pdf.groupby("ativo"):
    serie = g["valor"].to_numpy(dtype=float)
    if serie.size < cfg.L:
        continue
    sc = core.RobustScaler().fit(serie)
    janelas.append(core.make_windows(sc.transform(serie), cfg.L, stride=cfg.L))
janelas = np.concatenate(janelas, axis=0)
print("janelas de treino:", janelas.shape)

# COMMAND ----------

# MAGIC %md ## 2. Treino do autoencoder denoising global

# COMMAND ----------

model = core.train_model(janelas, cfg)

# salva o artefato em DBFS (via caminho local /dbfs)
local_path = MODEL_PATH.replace("dbfs:", "/dbfs")
os.makedirs(os.path.dirname(local_path), exist_ok=True)
core.salvar_modelo(model, local_path, cfg, VERSAO)
print("modelo salvo em", MODEL_PATH, "versao", VERSAO)
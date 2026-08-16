# Databricks notebook source
# MAGIC %md
# MAGIC # nb_03 — Cálculo do H1 (incorpora o conhecimento do especialista)
# MAGIC Lê os rótulos do engenheiro em `james.DEMANDA_MAXIMA_TREINO`, aprende uma
# MAGIC calibração global a partir dos pares (H0, valor_correto) e a aplica a todos
# MAGIC os alimentadores. Onde há rótulo, usa o valor exato do especialista.
# MAGIC Grava `demanda_max_h1` em `james.DEMANDA_MAXIMA`.

# COMMAND ----------
# MAGIC %pip install numpy pandas

# COMMAND ----------
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/Workspace/Repos/james/scripts_python/demanda_maxima")
import databricks_io as io  # noqa: E402

dbutils.widgets.text("secret_scope", "james")
SCOPE = dbutils.widgets.get("secret_scope")
url, props = io.jdbc_conf(dbutils, SCOPE)

# COMMAND ----------
# MAGIC %md ## 1. Lê H0 atual e os rótulos do especialista

# COMMAND ----------
dm = io.ler_tabela(spark, url, props, "james.DEMANDA_MAXIMA").toPandas()
lb = io.ler_tabela(spark, url, props, "james.DEMANDA_MAXIMA_TREINO").toPandas()

chaves = ["regiao", "ativo", "grandeza", "ano"]
# rótulo mais recente por chave
lb = (lb.sort_values("timestamp").groupby(chaves, as_index=False)
        .agg(valor_correto=("valor_correto", "last")))

base = dm.merge(lb, on=chaves, how="left")

# COMMAND ----------
# MAGIC %md ## 2. Calibração global aprendida dos pares (H0 → valor_correto)

# COMMAND ----------
pares = base.dropna(subset=["demanda_max_h0", "valor_correto"])
a, b = 1.0, 0.0
if len(pares) >= 2:
    # ajuste robusto simples: h1 = a*h0 + b (mínimos quadrados)
    a, b = np.polyfit(pares["demanda_max_h0"].to_numpy(float),
                      pares["valor_correto"].to_numpy(float), 1)
print(f"calibracao H1: h1 = {a:.3f} * h0 + {b:.3f}  (n_rotulos={len(pares)})")

h1 = a * base["demanda_max_h0"].to_numpy(float) + b
# onde há rótulo do especialista, usa o valor EXATO informado (ground truth)
tem_rotulo = base["valor_correto"].notna().to_numpy()
h1[tem_rotulo] = base["valor_correto"].to_numpy(float)[tem_rotulo]
base["demanda_max_h1"] = h1

# COMMAND ----------
# MAGIC %md ## 3. Grava demanda_max_h1 (staging + MERGE)

# COMMAND ----------
stg = base[["regiao", "ativo", "grandeza", "ano", "demanda_max_h1"]]
sdf = spark.createDataFrame(stg)
io.escrever_tabela(sdf, url, props, "james.DEMANDA_MAXIMA_H1_STG", mode="overwrite")

MERGE = """
MERGE james.DEMANDA_MAXIMA AS d
USING james.DEMANDA_MAXIMA_H1_STG AS s
  ON d.regiao=s.regiao AND d.ativo=s.ativo AND d.grandeza=s.grandeza AND d.ano=s.ano
WHEN MATCHED THEN UPDATE SET d.demanda_max_h1 = s.demanda_max_h1, d.calculado_em = GETDATE();
"""
io.executar_sql(spark, url, props, MERGE)
print("H1 atualizado em james.DEMANDA_MAXIMA")

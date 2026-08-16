# Databricks notebook source
# MAGIC %md
# MAGIC # nb_02 — Cálculo da máxima anual (H0) e gravação no SQL Server
# MAGIC Aplica o modelo global a TODOS os alimentadores × ano (inferência distribuída
# MAGIC via `applyInPandas`) e faz MERGE em `james.DEMANDA_MAXIMA` preservando o H1.

# COMMAND ----------

# MAGIC %pip install torch numpy pandas

# COMMAND ----------

import os, sys
import numpy as np
import pandas as pd
from pyspark.sql.functions import lit
from pyspark.sql.types import (StructType, StructField, StringType, IntegerType, DoubleType)

sys.path.insert(0, "/Workspace/Shared/Servidores/Servidor SP/James/ML_Demanda_Maxima")
import core            # noqa: E402
import databricks_io as io  # noqa: E402

#dbutils.widgets.text("medicao_path", "/Volumes/poseidon_uc/group_uc/ddpe/medicao/parquet/")
#dbutils.widgets.text("grandeza", "IMAX")
#dbutils.widgets.text("model_path", "/Workspace/Shared/Servidores/Servidor SP/James/ML_Demanda_Maxima/ae_h0.pt")
#dbutils.widgets.text("secret_scope", "sqlserver")

MEDICAO = dbutils.widgets.get("medicao_path")
GRANDEZA = dbutils.widgets.get("grandeza")
MODEL_PATH = dbutils.widgets.get("model_path").replace("dbfs:", "/dbfs")
SCOPE = dbutils.widgets.get("secret_scope")

cfg = core.Config()

# COMMAND ----------

# MAGIC %md ## 1. Carrega o modelo e faz broadcast dos pesos

# COMMAND ----------

import torch
VERSAO = torch.load(MODEL_PATH, map_location="cpu").get("versao", "?")

# COMMAND ----------

# MAGIC %md ## 2. Inferência distribuída: máxima anual por (regiao, ativo, ano)

# COMMAND ----------

from pyspark.sql.functions import lit

def ler_alm(regiao):                       # regiao = "SP" | "ES"
    return (spark.read.parquet(f"{MEDICAO}/{regiao}/ALM")
                 .withColumn("regiao", lit(regiao)))

df = (ler_alm("SP")
      .unionByName(ler_alm("ES"), allowMissingColumns=True)
      .selectExpr("regiao", "ALM as ativo", "DATAS", f"{GRANDEZA} as valor", "YEAR as ano"))

out_schema = StructType([
    StructField("regiao", StringType()), StructField("tipo", StringType()),
    StructField("ativo", StringType()), StructField("grandeza", StringType()),
    StructField("ano", IntegerType()), StructField("demanda_max_h0", DoubleType()),
    StructField("modelo_versao", StringType()),
])

_CACHE = {}
def _modelo(cfg):
    if "m" not in _CACHE:
        import core as _core
        _CACHE["m"], _ = _core.carregar_modelo(MODEL_PATH, cfg)
    return _CACHE["m"]

def calcular_maxima(pdf):
    import numpy as _np, core as _core
    _cfg = _core.Config()
    m = _modelo(_cfg)                     # carrega 1x por worker, sem broadcast
    pdf = pdf.sort_values("DATAS")
    serie = pdf["valor"].to_numpy(dtype=float)
    sc = _core.RobustScaler().fit(serie)
    vmax = _core.maxima_anual(m, serie, sc, _cfg)
    r = pdf.iloc[0]
    return pd.DataFrame([{
        "regiao": r["regiao"], "tipo": "Alimentador", "ativo": r["ativo"],
        "grandeza": GRANDEZA, "ano": int(r["ano"]),
        "demanda_max_h0": float(vmax) if _np.isfinite(vmax) else None,
        "modelo_versao": VERSAO,
    }])

resultado = df.groupBy("regiao", "ativo", "ano").applyInPandas(calcular_maxima, schema=out_schema)

# COMMAND ----------

#display(resultado)

# COMMAND ----------

# MAGIC %md ## 3. Escreve em staging e faz MERGE em james.DEMANDA_MAXIMA (preserva H1)

# COMMAND ----------

url, props = io.jdbc_conf(dbutils, SCOPE)
io.escrever_tabela(resultado, url, props, "james.DEMANDA_MAXIMA_STG", mode="overwrite")

MERGE = """
MERGE james.DEMANDA_MAXIMA AS d
USING james.DEMANDA_MAXIMA_STG AS s
  ON d.regiao=s.regiao AND d.ativo=s.ativo AND d.grandeza=s.grandeza AND d.ano=s.ano
WHEN MATCHED THEN UPDATE SET d.demanda_max_h0=s.demanda_max_h0,
                             d.modelo_versao=s.modelo_versao, d.calculado_em=GETDATE()
WHEN NOT MATCHED THEN INSERT (regiao,tipo,ativo,grandeza,ano,demanda_max_h0,modelo_versao)
     VALUES (s.regiao,s.tipo,s.ativo,s.grandeza,s.ano,s.demanda_max_h0,s.modelo_versao);
"""
#io.executar_sql_pymssql(MERGE)


# COMMAND ----------

# MAGIC %pip install pymssql

# COMMAND ----------

def executar_sql_pymssql(sql, scope="sqlserver"):
    import pymssql
    conn = pymssql.connect(
        server=dbutils.secrets.get(scope, "url"),
        user=dbutils.secrets.get(scope, "user"),
        password=dbutils.secrets.get(scope, "password"),
        database=dbutils.secrets.get(scope, "database"))
    cur = conn.cursor(); cur.execute(sql); conn.commit(); conn.close()

# COMMAND ----------

executar_sql_pymssql(MERGE)
print("MERGE concluido em james.DEMANDA_MAXIMA")
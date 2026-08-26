# Databricks notebook source
# MAGIC %md
# MAGIC # nb_02 — Máxima anual (H0 MULTIVARIADO) → SQL Server
# MAGIC Inferência distribuída (`applyInPandas`) do H0 multivariado por
# MAGIC (regiao, ativo, ano) e MERGE em `james.DEMANDA_MAXIMA` (preserva o H1).
# MAGIC Shared-cluster-safe: modelo carregado DENTRO da UDF (cache por worker, sem
# MAGIC `sparkContext.broadcast`); MERGE via `pymssql` (sem acesso à JVM).

# COMMAND ----------

# MAGIC %pip install torch numpy pandas pymssql

# COMMAND ----------

import sys
import numpy as np
import pandas as pd
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

sys.path.insert(0, "/Workspace/Shared/Servidores/Servidor SP/James/max_demand")
import core, core_multi as cm            # noqa: E402
import databricks_io_v2 as io               # noqa: E402

dbutils.widgets.text("medicao_path", "/Volumes/poseidon_uc/group_uc/ddpe/medicao/parquet")
dbutils.widgets.text("grandeza", "MVA")
dbutils.widgets.text("model_path", "/Workspace/Shared/Servidores/Servidor SP/James/max_demand/ae_h0_multi.pt")
dbutils.widgets.text("secret_scope", "sqlserver")

MEDICAO    = dbutils.widgets.get("medicao_path")
GRANDEZA   = dbutils.widgets.get("grandeza")
MODEL_PATH = dbutils.widgets.get("model_path")
SCOPE      = dbutils.widgets.get("secret_scope")

url, props = io.jdbc_conf(dbutils, SCOPE)


def ler_alm(regiao):
    df = spark.read.parquet(f"{MEDICAO}/{regiao}/ALM").withColumn("regiao", lit(regiao))
    return df.selectExpr("regiao", "ALM as ativo", "DATAS",
                         f"{GRANDEZA} as valor", "YEAR as ano")


# versão + cfg (com stats de extensão) e os mapas mensais de extensão — no driver
_, CFG, VERSAO = cm.carregar_modelo(MODEL_PATH, device="cpu")
COMP_MAP, DCOMP_MAP, _, _, _ = io.carregar_ext_rede(spark, url, props)  # normaliza com CFG

# COMMAND ----------

# MAGIC %md ## 1. Inferência distribuída: máxima anual por (regiao, ativo, ano)

# COMMAND ----------

med = ler_alm("SP").unionByName(ler_alm("ES"), allowMissingColumns=True)

out_schema = StructType([
    StructField("regiao", StringType()), StructField("tipo", StringType()),
    StructField("ativo", StringType()), StructField("grandeza", StringType()),
    StructField("ano", IntegerType()), StructField("demanda_max_h0", DoubleType()),
    StructField("modelo_versao", StringType()),
])

# cache por processo do worker (evita broadcast; carrega o modelo 1x por worker)
_CACHE = {}
def _modelo():
    if "m" not in _CACHE:
        _CACHE["m"], _CACHE["cfg"], _ = cm.carregar_modelo(MODEL_PATH, device="cpu")
    return _CACHE["m"], _CACHE["cfg"]

def calcular_maxima(pdf: pd.DataFrame) -> pd.DataFrame:
    import numpy as _np, pandas as _pd, core as _core, core_multi as _cm
    m, cfg = _modelo()
    pdf = pdf.sort_values("DATAS")
    valor = pdf["valor"].to_numpy(float)
    sc = _core.RobustScaler().fit(valor)
    r = pdf.iloc[0]
    d = _pd.to_datetime(pdf["DATAS"])
    comp_full, dcomp_full = _cm.alinhar_comprimento(d.dt.year, d.dt.month,
                                                    r["regiao"], r["ativo"], COMP_MAP, DCOMP_MAP, cfg)
    v = _cm.maxima_anual_multi(m, valor, comp_full, dcomp_full, sc, cfg)
    return pd.DataFrame([{
        "regiao": r["regiao"], "tipo": "Alimentador", "ativo": r["ativo"],
        "grandeza": GRANDEZA, "ano": int(r["ano"]),
        "demanda_max_h0": float(v) if _np.isfinite(v) else None,
        "modelo_versao": VERSAO,
    }])

resultado = med.groupBy("regiao", "ativo", "ano").applyInPandas(calcular_maxima, schema=out_schema)

# COMMAND ----------

# MAGIC %md ## 2. Staging + MERGE em james.DEMANDA_MAXIMA (preserva H1)

# COMMAND ----------

io.escrever_tabela(resultado, url, props, "james.DEMANDA_MAXIMA_STG", mode="overwrite")





# COMMAND ----------

def executar_sql_pymssql(sql):
    import pymssql
    conn = pymssql.connect(server=dbutils.secrets.get(SCOPE, "url"),
                           user=dbutils.secrets.get(SCOPE, "user"),
                           password=dbutils.secrets.get(SCOPE, "password"),
                           database=dbutils.secrets.get(SCOPE, "database"))
    cur = conn.cursor(); cur.execute(sql); conn.commit(); conn.close()

# COMMAND ----------

executar_sql_pymssql("""
MERGE james.DEMANDA_MAXIMA AS d
USING james.DEMANDA_MAXIMA_STG AS s
  ON d.regiao=s.regiao AND d.ativo=s.ativo AND d.grandeza=s.grandeza AND d.ano=s.ano
WHEN MATCHED THEN UPDATE SET d.demanda_max_h0=s.demanda_max_h0,
                             d.modelo_versao=s.modelo_versao, d.calculado_em=GETDATE()
WHEN NOT MATCHED THEN INSERT (regiao,tipo,ativo,grandeza,ano,demanda_max_h0,modelo_versao)
     VALUES (s.regiao,s.tipo,s.ativo,s.grandeza,s.ano,s.demanda_max_h0,s.modelo_versao);
""")
print("MERGE H0 concluido em james.DEMANDA_MAXIMA")

# COMMAND ----------

## Parametros Servidor ##
lv_ServerName = dbutils.secrets.get(scope="sqlserver", key="url")
lv_DataBase = dbutils.secrets.get(scope="sqlserver", key="database")
lv_User = dbutils.secrets.get(scope="sqlserver", key="user")
lv_Password = dbutils.secrets.get(scope="sqlserver", key="password")
lv_Authentication = "SqlPassword"
lv_Encrypt = "true"
lv_TrustServerCertificate = "true"
lv_HostName = "*.database.windows.net"
lv_TimeOut = "30"
# URL
ddpeURL = "jdbc:sqlserver://" + lv_ServerName + ";databaseName=" + lv_DataBase + ";Authentication=" + lv_Authentication + ";encrypt=" + lv_Encrypt + ";ServerCertificate=" + lv_TrustServerCertificate + ";hostNameInCertificate=" + lv_HostName


# COMMAND ----------

sql_dm = "james.DEMANDA_MAXIMA"
dm = spark.read.jdbc(ddpeURL, sql_dm, properties={"user": lv_User, "password": lv_Password})

# COMMAND ----------

display(dm)
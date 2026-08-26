# Databricks notebook source
# MAGIC %md
# MAGIC # nb_02 — Máxima anual (H0) → SQL Server
# MAGIC Inferência distribuída (`applyInPandas`) por (regiao, ativo, ano) e MERGE em
# MAGIC `james.DEMANDA_MAXIMA` (preserva o H1).
# MAGIC
# MAGIC **Modelo:** usa por padrão o H0 **mais recente** de `modelo/`. O caminho é
# MAGIC resolvido UMA VEZ no driver e repassado aos workers — assim toda a rodada
# MAGIC usa o mesmo artefato mesmo que alguém treine um novo no meio da execução.
# MAGIC Para reproduzir uma rodada antiga, preencha o widget `model_path`.
# MAGIC
# MAGIC Shared-cluster-safe: modelo carregado DENTRO da UDF (cache por worker, sem
# MAGIC `sparkContext.broadcast`); MERGE via `pymssql` (sem acesso à JVM).

# COMMAND ----------

# MAGIC %pip install torch numpy pandas pymssql

# COMMAND ----------

import os, sys
import numpy as np
import pandas as pd
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

dbutils.widgets.text("repo_dir", "/Workspace/Shared/Servidores/Servidor SP/James/max_demand")
dbutils.widgets.text("medicao_path", "/Volumes/poseidon_uc/group_uc/ddpe/medicao/parquet")
dbutils.widgets.text("grandeza", "MVA")
dbutils.widgets.text("model_path", "")   # vazio = usa o H0 mais recente de modelo/
dbutils.widgets.text("secret_scope", "sqlserver")

REPO_DIR   = dbutils.widgets.get("repo_dir")
MEDICAO    = dbutils.widgets.get("medicao_path")
GRANDEZA   = dbutils.widgets.get("grandeza")
MODEL_PATH = dbutils.widgets.get("model_path").strip()
SCOPE      = dbutils.widgets.get("secret_scope")
MODELO_DIR = os.path.join(REPO_DIR, "modelo")

sys.path.insert(0, os.path.join(REPO_DIR, "src"))
import core, core_multi as cm            # noqa: E402
import databricks_io as io               # noqa: E402

url, props = io.jdbc_conf(dbutils, SCOPE)

# resolve o modelo UMA VEZ no driver; os workers recebem o caminho concreto
if not MODEL_PATH:
    MODEL_PATH = cm.caminho_modelo_mais_recente(MODELO_DIR, "ae_h0_multi")
_, CFG, VERSAO = cm.carregar_modelo(MODEL_PATH, device="cpu")
print("modelo H0:", MODEL_PATH, "| versao:", VERSAO, "| canais:", CFG.n_canais)

COMP_MAP, DCOMP_MAP, _, _, _ = io.carregar_ext_rede(spark, url, props)  # normaliza com CFG


def ler_alm(regiao):
    df = spark.read.parquet(f"{MEDICAO}/{regiao}/ALM").withColumn("regiao", lit(regiao))
    return df.selectExpr("regiao", "ALM as ativo", "DATAS",
                         f"{GRANDEZA} as valor", "YEAR as ano")

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
        import core_multi as _cm
        _CACHE["m"], _CACHE["cfg"], _ = _cm.carregar_modelo(MODEL_PATH, device="cpu")
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
    return _pd.DataFrame([{
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

io.executar_sql_pymssql("""
MERGE james.DEMANDA_MAXIMA AS d
USING james.DEMANDA_MAXIMA_STG AS s
  ON d.regiao=s.regiao AND d.ativo=s.ativo AND d.grandeza=s.grandeza AND d.ano=s.ano
WHEN MATCHED THEN UPDATE SET d.demanda_max_h0=s.demanda_max_h0,
                             d.modelo_versao=s.modelo_versao, d.calculado_em=GETDATE()
WHEN NOT MATCHED THEN INSERT (regiao,tipo,ativo,grandeza,ano,demanda_max_h0,modelo_versao)
     VALUES (s.regiao,s.tipo,s.ativo,s.grandeza,s.ano,s.demanda_max_h0,s.modelo_versao);
""", dbutils, scope=SCOPE)
print("MERGE H0 concluido em james.DEMANDA_MAXIMA")

# COMMAND ----------

# MAGIC %md ## 3. Verificação (rodou corretamente?)
# MAGIC Confere cobertura (linhas gravadas com a versão desta rodada), ausência de
# MAGIC nulos e plausibilidade (H0 deve ficar ABAIXO do máximo bruto, pois remove
# MAGIC picos de medição, e acima da mediana da série).

# COMMAND ----------

dm = io.ler_tabela(spark, url, props, "james.DEMANDA_MAXIMA").toPandas()
desta = dm[dm["modelo_versao"] == VERSAO]

print(f"linhas com a versao desta rodada: {len(desta)} de {len(dm)} na tabela")
print(f"nulos em demanda_max_h0: {int(desta['demanda_max_h0'].isna().sum())}")
print("\ncobertura por regiao/ano:")
print(desta.groupby(["regiao", "ano"]).size().unstack(fill_value=0))
print("\ndistribuicao de demanda_max_h0:")
print(desta["demanda_max_h0"].describe())

# plausibilidade: compara com o máximo BRUTO de uma amostra de alimentadores-ano
bruto = (med.groupBy("regiao", "ativo", "ano")
            .agg({"valor": "max"}).withColumnRenamed("max(valor)", "max_bruto")
            .limit(500).toPandas())
cmp = desta.merge(bruto, on=["regiao", "ativo", "ano"], how="inner").dropna()
if len(cmp):
    acima = (cmp["demanda_max_h0"] > cmp["max_bruto"] * 1.001).mean()
    razao = (cmp["demanda_max_h0"] / cmp["max_bruto"]).median()
    print(f"\namostra comparada: {len(cmp)} | H0/max_bruto (mediana): {razao:.3f}")
    print(f"fracao com H0 ACIMA do maximo bruto (esperado ~0): {acima:.3%}")

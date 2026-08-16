# Databricks notebook source
# MAGIC %md
# MAGIC # nb_03 (alternativo) — H1 por RETREINO da rede com supervisão do engenheiro
# MAGIC A alma do projeto: em vez de calibrar o escalar, aqui a PRÓPRIA REDE é
# MAGIC retreinada (fine-tuning) para que o **pico anual da reconstrução** dos
# MAGIC alimentadores rotulados case com o valor informado pelo especialista
# MAGIC (perda de quantil anual, `core_multi.fine_tune_h1`). O conhecimento passa a
# MAGIC morar nos pesos e generaliza para os demais alimentadores.
# MAGIC
# MAGIC Requer o H0 MULTIVARIADO (nb_01/nb_02 atualizados para `core_multi`).
# MAGIC Cluster shared: sem `sparkContext.broadcast` e sem JVM (MERGE via pymssql).

# COMMAND ----------
# MAGIC %pip install torch numpy pandas pymssql

# COMMAND ----------
import sys
import numpy as np
import pandas as pd
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

sys.path.insert(0, "/Workspace/Repos/james/scripts_python/demanda_maxima")
import core, core_multi as cm            # noqa: E402
import databricks_io as io               # noqa: E402

dbutils.widgets.text("medicao_path", "/Volumes/poseidon_uc/group_uc/ddpe/medicao/parquet")
dbutils.widgets.text("grandeza", "MVA")
dbutils.widgets.text("model_h0_path", "/dbfs/FileStore/james/demanda_maxima/ae_h0_multi.pt")
dbutils.widgets.text("model_h1_path", "/dbfs/FileStore/james/demanda_maxima/ae_h1_multi.pt")
dbutils.widgets.text("secret_scope", "james")
dbutils.widgets.text("n_pool", "200")

MEDICAO   = dbutils.widgets.get("medicao_path")
GRANDEZA  = dbutils.widgets.get("grandeza")
H0_PATH   = dbutils.widgets.get("model_h0_path")
H1_PATH   = dbutils.widgets.get("model_h1_path")
SCOPE     = dbutils.widgets.get("secret_scope")
N_POOL    = int(dbutils.widgets.get("n_pool"))

url, props = io.jdbc_conf(dbutils, SCOPE)


# --- helpers de leitura (mesmo padrão do nb_01/nb_02 multivariado) -----------
def ler_alm(regiao):
    df = (spark.read.parquet(f"{MEDICAO}/{regiao}/ALM").withColumn("regiao", lit(regiao)))
    return df.selectExpr("regiao", "ALM as ativo", "DATAS",
                         f"{GRANDEZA} as valor", "is_sicoi", "YEAR as ano")

# COMMAND ----------
# MAGIC %md ## 1. Carrega o H0 multivariado e os rótulos do especialista

# COMMAND ----------
model, cfg, _ = cm.carregar_modelo(H0_PATH, device="cpu")

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
              .select("DATAS", "valor", "is_sicoi").orderBy("DATAS").toPandas())
    if pdf.empty:
        continue
    valor = pdf["valor"].to_numpy(float)
    sic   = pdf["is_sicoi"].fillna(0).to_numpy(float)
    sc = core.RobustScaler().fit(valor)
    comp_full, dcomp_full = canais_comp(pdf["DATAS"], r["regiao"], r["ativo"])
    vw = core.make_windows(sc.transform(valor), cfg.L, cfg.L)
    sw = core.make_windows(sic, cfg.L, cfg.L)
    cw = core.make_windows(comp_full, cfg.L, cfg.L)
    dw = core.make_windows(dcomp_full, cfg.L, cfg.L)
    rotulados.append({
        "canais": cm.canais_de_janelas(vw, sw, cw, dw, cfg),
        "V_norm": float(sc.transform(np.array([r["valor_correto"]]))[0]),
    })
print("exemplos rotulados prontos:", len(rotulados))

# COMMAND ----------
# MAGIC %md ## 3. Pool de reconstrução (amostra ampla) e FINE-TUNING

# COMMAND ----------
ativos = [x.ativo for x in med.select("regiao", "ativo").distinct().limit(N_POOL).collect()]
pool_pdf = (med.filter(med.ativo.isin(ativos))
               .select("regiao", "ativo", "ano", "DATAS", "valor", "is_sicoi")
               .orderBy("ativo", "ano", "DATAS").toPandas())
pv, ps, pc, pdc = [], [], [], []
for (reg, at, ano), g in pool_pdf.groupby(["regiao", "ativo", "ano"]):
    v = g["valor"].to_numpy(float)
    if v.size < cfg.L:
        continue
    sc = core.RobustScaler().fit(v)
    comp_full, dcomp_full = canais_comp(g["DATAS"], reg, at)
    pv.append(core.make_windows(sc.transform(v), cfg.L, cfg.L))
    ps.append(core.make_windows(g["is_sicoi"].fillna(0).to_numpy(float), cfg.L, cfg.L))
    pc.append(core.make_windows(comp_full, cfg.L, cfg.L))
    pdc.append(core.make_windows(dcomp_full, cfg.L, cfg.L))
pool = {"valor": np.concatenate(pv), "sic": np.concatenate(ps),
        "comp": np.concatenate(pc), "dcomp": np.concatenate(pdc)}

model_h1 = cm.fine_tune_h1(model, rotulados, pool, cfg, lam=8.0, epochs=60, lr=5e-4)
VERSAO = "ae_h1_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M")
cm.salvar_modelo(model_h1, H1_PATH, cfg, VERSAO)
print("H1 salvo:", H1_PATH, VERSAO)

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
    sic   = pdf["is_sicoi"].fillna(0).to_numpy(float)
    sc = _core.RobustScaler().fit(valor)
    r = pdf.iloc[0]
    d = _pd.to_datetime(pdf["DATAS"])
    comp_full, dcomp_full = _cm.alinhar_comprimento(d.dt.year, d.dt.month,
                                                    r["regiao"], r["ativo"], COMP_MAP, DCOMP_MAP, _cfg)
    v = _cm.maxima_anual_multi(m, valor, sic, comp_full, dcomp_full, sc, _cfg)
    return pd.DataFrame([{"regiao": r["regiao"], "ativo": r["ativo"], "grandeza": GRANDEZA,
                          "ano": int(r["ano"]),
                          "demanda_max_h1": float(v) if _np.isfinite(v) else None}])

res = med.groupBy("regiao", "ativo", "ano").applyInPandas(calc_h1, schema=out_schema)
io.escrever_tabela(res, url, props, "james.DEMANDA_MAXIMA_H1_STG", mode="overwrite")

# COMMAND ----------
# MAGIC %md ## 5. MERGE em james.DEMANDA_MAXIMA (via pymssql — shared cluster)

# COMMAND ----------
def executar_sql_pymssql(sql):
    import pymssql
    conn = pymssql.connect(server=dbutils.secrets.get(SCOPE, "sql_server"),
                           user=dbutils.secrets.get(SCOPE, "sql_user"),
                           password=dbutils.secrets.get(SCOPE, "sql_pwd"),
                           database=dbutils.secrets.get(SCOPE, "sql_db"))
    cur = conn.cursor(); cur.execute(sql); conn.commit(); conn.close()

executar_sql_pymssql("""
MERGE james.DEMANDA_MAXIMA AS d
USING james.DEMANDA_MAXIMA_H1_STG AS s
  ON d.regiao=s.regiao AND d.ativo=s.ativo AND d.grandeza=s.grandeza AND d.ano=s.ano
WHEN MATCHED THEN UPDATE SET d.demanda_max_h1 = s.demanda_max_h1, d.calculado_em = GETDATE();
""")
print("H1 (rede retreinada) gravado em james.DEMANDA_MAXIMA")

"""
databricks_io.py — utilitários de I/O para os notebooks Databricks.

SÓ roda dentro do Databricks (usa Spark + dbutils). Não é importado pelo núcleo
`core.py` nem pelo teste local. Credenciais SEMPRE via Databricks secrets — nunca
em texto no código.

Configure um scope de secrets (ex.: `sqlserver`) com as chaves:
    url, database, user, password
"""

from __future__ import annotations


def jdbc_conf(dbutils, scope: str = "sqlserver"):
    """Monta (url, props) do SQL Server a partir do Databricks secrets."""
    server = dbutils.secrets.get(scope, "url")
    db     = dbutils.secrets.get(scope, "database")    
    user   = dbutils.secrets.get(scope, "user")
    pwd    = dbutils.secrets.get(scope, "password")
    url = (f"jdbc:sqlserver://{server}:1433;database={db};"
           "encrypt=true;trustServerCertificate=false;loginTimeout=30")
    props = {"user": user, "password": pwd,
             "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"}
    return url, props


def ler_tabela(spark, url, props, tabela: str):
    """Lê uma tabela do SQL Server como Spark DataFrame."""
    return (spark.read.format("jdbc")
            .option("url", url).option("dbtable", tabela)
            .option("user", props["user"]).option("password", props["password"])
            .option("driver", props["driver"]).load())


def escrever_tabela(df, url, props, tabela: str, mode: str = "overwrite"):
    """Escreve um Spark DataFrame numa tabela do SQL Server via JDBC."""
    (df.write.format("jdbc")
       .option("url", url).option("dbtable", tabela)
       .option("user", props["user"]).option("password", props["password"])
       .option("driver", props["driver"]).mode(mode).save())


def carregar_ext_rede(spark, url, props):
    """
    Lê james.EXT_REDE_MT (extensão MENSAL do circuito, km) e devolve:
      comp_map  : {(UF, COD_CIRCUITO_MT, ANO, MES): COMP_km}         — nível
      dcomp_map : {(UF, COD_CIRCUITO_MT, ANO, MES): ΔCOMP_km}        — variação vs
                  mês anterior do MESMO circuito (Δkm grande = manobra topológica)
      comp_mean, comp_std : stats globais do nível (normaliza o canal comp)
      dcomp_std : escala global da variação (normaliza o canal d_comp)
    Chaves: UF = regiao ('SP'/'ES'); COD_CIRCUITO_MT = id do alimentador (ALM).
    """
    import numpy as np
    pdf = (ler_tabela(spark, url, props, "james.EXT_REDE_MT")
           .select("UF", "COD_CIRCUITO_MT", "ANO", "MES", "ANOMES", "COMP").toPandas())
    pdf = pdf.dropna(subset=["COMP"]).sort_values(["UF", "COD_CIRCUITO_MT", "ANOMES"])
    pdf["dcomp"] = pdf.groupby(["UF", "COD_CIRCUITO_MT"])["COMP"].diff().fillna(0.0)

    comp = pdf["COMP"].to_numpy(float); dcomp = pdf["dcomp"].to_numpy(float)
    comp_mean = float(np.mean(comp)) if comp.size else 0.0
    comp_std  = float(np.std(comp)) if comp.size and np.std(comp) > 1e-9 else 1.0
    dcomp_std = float(np.std(dcomp)) if dcomp.size and np.std(dcomp) > 1e-9 else 1.0

    def key(r):
        return (str(r.UF).strip(), str(r.COD_CIRCUITO_MT).strip(), int(r.ANO), int(r.MES))
    comp_map  = {key(r): float(r.COMP) for r in pdf.itertuples()}
    dcomp_map = {key(r): float(r.dcomp) for r in pdf.itertuples()}
    return comp_map, dcomp_map, comp_mean, comp_std, dcomp_std


# def executar_sql(spark, url, props, sql: str):
#     """Executa um comando T-SQL (ex.: MERGE) via JVM DriverManager."""
#     jvm = spark._sc._jvm
#     conn = jvm.java.sql.DriverManager.getConnection(url, props["user"], props["password"])
#     try:
#         st = conn.createStatement()
#         st.execute(sql)
#         st.close()
#     finally:
#         conn.close()

def executar_sql_pymssql(sql, dbutils, scope="sqlserver"):
    """Executa T-SQL (ex.: MERGE) via pymssql. `dbutils` vem do notebook
    chamador — módulos importados no Databricks não enxergam o global."""
    import pymssql
    conn = pymssql.connect(
        server=dbutils.secrets.get(scope, "url"),
        user=dbutils.secrets.get(scope, "user"),
        password=dbutils.secrets.get(scope, "password"),
        database=dbutils.secrets.get(scope, "database"))
    cur = conn.cursor(); cur.execute(sql); conn.commit(); conn.close()

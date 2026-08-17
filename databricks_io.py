"""
databricks_io.py — utilitários de I/O para os notebooks Databricks.

SÓ roda dentro do Databricks (usa Spark + dbutils). Não é importado pelo núcleo
`core.py` nem pelo teste local. Credenciais SEMPRE via Databricks secrets — nunca
em texto no código.

Configure um scope de secrets (ex.: `sqlserver`) com as chaves:
    url, database, user, password
(instale `pymssql` no notebook que usar `executar_sql_pymssql`)
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


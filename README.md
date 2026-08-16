# Demanda Máxima Estrutural — pipeline Databricks/PySpark

Estima a **máxima demanda estrutural por ano** de cada alimentador (estado latente,
sem manobras nem erros de medição) com um **autoencoder denoising 1D-CNN global**
e grava no **SQL Server**, de onde o James (R) lê.

## Componentes

| Arquivo | Onde roda | Papel |
|---|---|---|
| `core.py` | qualquer lugar (sem Spark) | modelo, normalização, treino, `maxima_anual`. Testável. |
| `databricks_io.py` | Databricks | JDBC/secrets: ler/escrever SQL Server, `MERGE`. |
| `nb_01_treino_h0.py` | Databricks | treina o modelo global (H0), salva artefato em DBFS. |
| `nb_02_predict_max.py` | Databricks | máxima anual por alimentador → `james.DEMANDA_MAXIMA` (H0). |
| `nb_03_treino_h1.py` | Databricks | usa os rótulos do especialista → `demanda_max_h1` (H1). |
| `test_pipeline.py` | local | teste sintético do núcleo (sem Spark). |

## Fluxo

```
Medição (parquet no lakehouse)
   │  nb_01_treino_h0  ──►  ae_h0.pt (DBFS)
   │  nb_02_predict_max ──► james.DEMANDA_MAXIMA (demanda_max_h0)   ◄── R lê
   ▼
James (R): engenheiro clica no gráfico o valor correto → INSERT em
           james.DEMANDA_MAXIMA_TREINO
   │
   │  nb_03_treino_h1  ──► james.DEMANDA_MAXIMA (demanda_max_h1)     ◄── R lê
```

## Pré-requisitos

1. Rodar `../../sql/schema.sql` no SQL Server (cria `james.DEMANDA_MAXIMA` e
   `james.DEMANDA_MAXIMA_TREINO`).
2. Criar um **secret scope** no Databricks (ex.: `james`) com:
   `sql_server`, `sql_db`, `sql_user`, `sql_pwd`. **Nunca** coloque senha no código.
3. Ajustar os widgets dos notebooks: `medicao_path` (base parquet), `grandeza`
   (default `MVA`), `model_path`, `secret_scope`, e o caminho de import de
   `demanda_maxima` (Repos/Workspace).

## Execução (ordem)

1. `nb_01_treino_h0` — treina o H0 (rodar quando quiser reajustar o modelo).
2. `nb_02_predict_max` — recalcula todas as máximas H0 (ex.: agendado, após o sync).
3. (depois que os engenheiros salvarem valores no James) `nb_03_treino_h1` — H1.

## Teste local

```bash
pip install -r requirements.txt
python test_pipeline.py
```

Valida que a máxima do sinal latente **remove os picos de medição** e se aproxima
da verdade (o modelo real é treinado no Databricks sobre a base completa).

> Escopo honesto do H0: remove ruído/erros de medição e transitórios curtos.
> Manobras **sustentadas** que parecem estruturais são o domínio do **H1**
> (conhecimento do especialista).

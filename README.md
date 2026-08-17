# Demanda Máxima Estrutural — pipeline Databricks/PySpark

Estima a **máxima demanda estrutural por ano** de cada alimentador (estado latente,
sem manobras nem erros de medição) com um **autoencoder denoising 1D-CNN global**
e grava no **SQL Server**, de onde o James (R) lê.

> **Pesquisa (mestrado/ITA):** o desenho experimental, hipóteses, protocolo
> anti-leakage e plano estatístico estão em `PLANO_PESQUISA_DISSERTACAO.md`.
> Este pipeline é o sistema de produção/incumbente do estudo.

> **Decisão de projeto:** o flag `is_sicoi` foi **removido** de todo o pipeline
> (canais e máscara de perda). Os canais multivariados são: valor, derivada,
> desvio local, extensão da rede (`comp`) e variação mensal da extensão (`d_comp`).

## Componentes

| Arquivo | Onde roda | Papel |
|---|---|---|
| `core.py` | qualquer lugar (sem Spark) | modelo univariado, normalização robusta, treino, `maxima_anual`. Testável. |
| `core_multi.py` | qualquer lugar (sem Spark) | H0 multivariado (5 canais) + fine-tuning H1 (`soft_peak`, mini-batch de rótulos). |
| `databricks_io.py` / `databricks_io_v2.py` | Databricks | JDBC/secrets: ler/escrever SQL Server, `MERGE`; v2 tem `carregar_ext_rede`. |
| `nb_01_treino_h0.py` / `_v2` | Databricks | treina o modelo global (H0; v2 = multivariado), amostra **aleatória com semente**, salva artefato. |
| `nb_02_predict_max.py` / `_v2` | Databricks | máxima anual por alimentador → `james.DEMANDA_MAXIMA` (H0). |
| `nb_03_treino_h1.py` | Databricks | calibração com rótulos do especialista + **CV agrupada por alimentador (out-of-sample)** → `demanda_max_h1`. |
| `nb_03_treino_h1_finetune.py` | Databricks | H1 por retreino da rede (fine-tuning com perda de quantil anual). |
| `test_pipeline.py` / `test_finetune.py` | local | testes sintéticos do núcleo (sem Spark). |

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
> (conhecimento do especialista). Sem `is_sicoi`, a separação entre evento
> operacional temporário e mudança estrutural depende apenas do objetivo
> denoising, dos canais topológicos (`comp`/`d_comp`) e da supervisão do
> especialista.

## Regras de avaliação (resumo do protocolo de pesquisa)

- Métricas **sempre out-of-sample**: CV agrupada por alimentador (nb_03) ou
  teste congelado por alimentador (protocolo da dissertação). A regra de
  produção "onde há rótulo, usa o valor exato" nunca entra em avaliação.
- Amostragens de alimentadores sempre **aleatórias com semente registrada**
  (nunca `.limit(N)` puro do Spark).
- `prob_max`, `q_low`/`q_high` do scaler, `lam`, `tau` são hiperparâmetros:
  tunar em validação interna, nunca no teste.

# Demanda Máxima Estrutural

Estima a **máxima demanda em regime normal de operação** de cada alimentador de
distribuição, por ano — o valor que o engenheiro reconheceria como a ponta real
da carga, sem os picos de erro de medição nem os patamares causados por manobras
temporárias. Roda em Databricks/PySpark e grava no SQL Server, de onde o James (R) lê.

> **Como rodar, entradas e verificação:** [`docs/GUIA_USO.md`](docs/GUIA_USO.md).
> **Pesquisa (mestrado/ITA):** [`docs/PLANO_PESQUISA_DISSERTACAO.md`](docs/PLANO_PESQUISA_DISSERTACAO.md)
> — hipóteses, desenho experimental, protocolo anti-leakage e plano estatístico.
> Este pipeline é o sistema de produção e o método incumbente do estudo.

## Estrutura

```
src/         núcleo Python (sem Spark) — importado pelos notebooks e pelos testes
notebooks/   os 4 notebooks Databricks, na ordem de execução
modelo/      modelos treinados, um arquivo por rodada, com timestamp
tests/       testes sintéticos locais (não precisam de Spark nem de cluster)
docs/        guia de uso e plano de pesquisa
sql/         consultas Databricks auxiliares (origem da extensão de rede)
```

| Arquivo | Papel |
|---|---|
| `src/core.py` | normalização robusta, janelamento, autoencoder base, contaminação sintética |
| `src/core_multi.py` | H0 multivariado (5 canais), fine-tuning H1, **versionamento dos modelos** |
| `src/databricks_io.py` | JDBC/secrets, leitura da extensão de rede, `MERGE` via pymssql |
| `notebooks/nb_01_treino_h0.py` | treina o H0 → `modelo/ae_h0_multi_<ts>.pt` |
| `notebooks/nb_02_predict_max.py` | aplica o H0 a todos os alimentadores → `demanda_max_h0` |
| `notebooks/nb_03a_h1_calibracao.py` | H1 por calibração linear sobre os rótulos (baseline rápida) |
| `notebooks/nb_03b_h1_finetune.py` | H1 por retreino da rede (abordagem principal) → `demanda_max_h1` |

## Como funciona

A série observada é tratada como `carga estrutural + manobras + erro de medição`.
Um autoencoder denoising 1D-CNN global reconstrói a componente estrutural, e a
máxima anual é um quantil alto dessa reconstrução. O modelo aprende a descartar
o que não é estrutural por três mecanismos: o gargalo não consegue representar
picos esparsos, a perda L1 não persegue desvios grandes e isolados, e durante o
treino injetamos contaminação sintética na entrada pedindo a curva limpa de volta.

**Canais de entrada (5):** valor, derivada, desvio local, extensão da rede (`comp`)
e sua variação mensal (`d_comp`). A ideia causal é que um Δkm grande sinaliza
reconfiguração **permanente** — mudança estrutural que deve ser preservada —
enquanto perturbações sem contrapartida topológica são candidatas a transitório.

**`is_sicoi` não é usado** (decisão de projeto): não há canal de manobra nem
máscara de perda. Sem esse sinal explícito, separar evento temporário de mudança
estrutural depende do objetivo denoising, dos canais topológicos e da supervisão
do especialista.

**Escopo honesto do H0:** remove ruído, erros de medição e transitórios curtos.
Manobras **sustentadas** que se parecem com carga estrutural são o domínio do
**H1**, onde entra o conhecimento do engenheiro.

## Modelos versionados

Cada treino grava um arquivo novo — `modelo/ae_h0_multi_<AAAAMMDD_HHMMSS>.pt` —
e nada é sobrescrito. A inferência usa por padrão o mais recente; para reproduzir
uma rodada antiga basta apontar o widget `model_path` para o arquivo desejado.
A versão é gravada dentro do `.pt` e na coluna `modelo_versao` da tabela de
resultado, então cada número no banco é rastreável até o modelo que o produziu.

## Regras de avaliação

Valem tanto para produção quanto para a dissertação:

- Métricas **sempre out-of-sample** — validação cruzada agrupada por alimentador
  (nb_03a) ou teste congelado por alimentador (protocolo da pesquisa). Anos do
  mesmo alimentador são correlacionados e nunca podem ficar em lados opostos do
  split. A regra de produção "onde há rótulo, usa o valor exato" jamais entra em
  avaliação.
- Amostragens de alimentadores sempre **aleatórias com semente registrada**,
  nunca `.limit(N)` puro do Spark, que devolve "os primeiros que aparecerem".
- `prob_max`, `q_low`/`q_high` do scaler, `lam` e `tau` são hiperparâmetros:
  ajustar em validação interna, nunca contra o conjunto de teste.

## Teste local

```bash
pip install -r requirements.txt
```

```bash
python tests/test_versionamento.py
```

Os outros dois (`tests/test_pipeline.py` e `tests/test_finetune.py`) treinam
redes pequenas em dados sintéticos e levam alguns minutos. Detalhes do que cada
um garante estão no [guia de uso](docs/GUIA_USO.md#7-testes-locais-sem-spark-sem-databricks).

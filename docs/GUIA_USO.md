# Guia de uso — Demanda Máxima Estrutural

Passo a passo para rodar os notebooks no Databricks: o que cada um espera de
entrada, o que produz, e **como conferir se rodou corretamente**.

---

## 0. Antes de começar (uma vez só)

### 0.1 Sincronizar o código no Workspace

Os notebooks importam o núcleo Python de uma pasta do Workspace. O padrão é:

```
/Workspace/Shared/Servidores/Servidor SP/James/max_demand/
    src/        core.py, core_multi.py, databricks_io.py
    notebooks/  nb_01…nb_03b
    modelo/     ae_h0_multi_<timestamp>.pt, ae_h1_multi_<timestamp>.pt
```

Se a sua pasta for outra, **não edite o código**: mude o widget `repo_dir` na
execução. Todos os notebooks derivam `src/` e `modelo/` a partir dele.

> ⚠️ **Erro mais comum do projeto.** Se o `src/` do Workspace estiver
> desatualizado em relação ao notebook, você recebe um erro confuso do tipo
> `TypeError: train_h0_multi() missing 1 required positional argument: 'cfg'`.
> Isso **não** é bug do notebook: é módulo velho em cache. Solução: sincronize a
> pasta e reinicie o Python (`dbutils.library.restartPython()` ou detach/attach).

### 0.2 Secrets

Scope `sqlserver` com as chaves **`url`, `database`, `user`, `password`**.
Nunca coloque credencial no código. Se usar outro scope, troque o widget
`secret_scope` — mas os nomes das chaves precisam ser esses.

### 0.3 Tabelas no SQL Server

| Tabela | Papel |
|---|---|
| `james.DEMANDA_MAXIMA` | resultado: `demanda_max_h0`, `demanda_max_h1` por (regiao, ativo, grandeza, ano) |
| `james.DEMANDA_MAXIMA_TREINO` | rótulos do especialista (`valor_correto` + `timestamp`), gravados pelo James (R) |
| `james.EXT_REDE_MT` | extensão mensal do circuito (km) — origem dos canais `comp`/`d_comp` |

A consulta que alimenta `EXT_REDE_MT` está em
[`sql/consultas_databricks.ipynb`](../sql/consultas_databricks.ipynb) (primeira célula).

---

## 1. Ordem de execução

```
nb_01_treino_h0        (treina o H0)                    → modelo/ae_h0_multi_<ts>.pt
        ↓
nb_02_predict_max      (aplica a todos os alimentadores) → james.DEMANDA_MAXIMA.demanda_max_h0
        ↓
   [engenheiro rotula no James (R) → james.DEMANDA_MAXIMA_TREINO]
        ↓
nb_03a  OU  nb_03b     (incorpora o especialista)        → james.DEMANDA_MAXIMA.demanda_max_h1
```

O `nb_01` só precisa rodar quando você quiser reajustar o modelo. O `nb_02` roda
sempre que a base de medição for atualizada. O `nb_03` roda depois que houver
rótulos novos.

**`nb_03a` e `nb_03b` são duas abordagens alternativas, não versões**:

| | `nb_03a_h1_calibracao` | `nb_03b_h1_finetune` |
|---|---|---|
| O que faz | ajusta `h1 = a·h0 + b` sobre os pares rotulados | retreina a própria rede (perda de quantil anual) |
| Custo | segundos | minutos a horas |
| Onde mora o conhecimento | num escalar | nos pesos da rede |
| Quando usar | poucos rótulos, resultado rápido, baseline | é a abordagem principal do projeto |

---

## 2. Versionamento dos modelos

Cada treino grava um arquivo **novo**, nunca sobrescreve:

```
modelo/ae_h0_multi_20260826_191600.pt
modelo/ae_h1_multi_20260827_084512.pt
```

A inferência usa **automaticamente o mais recente** do prefixo. Para reproduzir
uma rodada antiga, preencha o widget `model_path` (nb_02) ou `model_h0_path`
(nb_03b) com o caminho exato do arquivo desejado — vazio significa "use o mais
recente".

A versão fica gravada dentro do `.pt` e é escrita na coluna `modelo_versao` da
`james.DEMANDA_MAXIMA`, então dá para rastrear qual modelo produziu cada número.

Para listar o que existe:

```python
import sys; sys.path.insert(0, f"{REPO_DIR}/src")
import core_multi as cm
for ts, caminho in cm.listar_modelos(f"{REPO_DIR}/modelo", "ae_h0_multi"):
    print(ts, caminho)
```

---

## 3. `nb_01_treino_h0` — treino do H0

**Entradas (widgets)**

| Widget | Padrão | O que é |
|---|---|---|
| `repo_dir` | `/Workspace/.../max_demand` | raiz do projeto no Workspace |
| `medicao_path` | `/Volumes/.../medicao/parquet` | base parquet de medição |
| `grandeza` | `IMAX` | coluna a modelar (`IMAX`, `MVA`…) |
| `n_alimentadores_amostra` | `300` | alimentadores sorteados para o treino |
| `max_janelas_por_grupo` | `60` | janelas (dias) por alimentador-ano |
| `secret_scope` | `sqlserver` | scope dos secrets |

**Saída:** `modelo/ae_h0_multi_<AAAAMMDD_HHMMSS>.pt`.

**Tempo típico:** dezenas de minutos (dominado pela leitura do parquet e pelas
40 épocas de treino).

**Como conferir que rodou certo** — a célula 3 do notebook faz isso sozinha:

1. `extensao: N chaves | media=… std=…` — se `N` for 0, a `EXT_REDE_MT` não foi
   lida e os canais `comp`/`d_comp` ficarão zerados (o modelo ainda treina, mas
   perde o sinal topológico). Investigue antes de seguir.
2. `janelas de treino: (N, 144) | canais: 5` — `N` deve ser da ordem de
   `n_alimentadores_amostra × anos × max_janelas_por_grupo`
   (ex.: 300 × 7 × 60 ≈ 126.000). Muito abaixo disso indica alimentadores com
   série curta demais (< 144 pontos) sendo descartados.
3. `OK — versao=… | canais=5 | L=144 | latente=32` — as três asserções passaram:
   o arquivo salvo é o mais recente, a arquitetura tem 5 canais e as estatísticas
   de extensão estão gravadas no `cfg`.

**Sinais de problema**

| Sintoma | Causa provável |
|---|---|
| `TypeError: train_h0_multi() missing 1 required positional argument` | `src/` desatualizado no Workspace (ver 0.1) |
| `SparkException: Total size of serialized results … maxResultSize` | alguém reintroduziu `.toPandas()` na série bruta; as janelas devem ser montadas nos workers |
| `janelas de treino: (0, 144)` | `medicao_path`/`grandeza` errados, ou filtro sem dados |
| perda não cai / modelo constante | `grandeza` com escala estranha; verifique nulos na coluna |

---

## 4. `nb_02_predict_max` — máxima anual de todos os alimentadores

**Entradas:** `repo_dir`, `medicao_path`, `grandeza`, `secret_scope` e
`model_path` (**vazio = usa o H0 mais recente**).

> O caminho do modelo é resolvido **uma vez no driver** e repassado aos workers.
> Isso garante que a rodada inteira use o mesmo artefato mesmo que alguém treine
> um H0 novo no meio da execução.

**Saída:** coluna `demanda_max_h0` (e `modelo_versao`) em `james.DEMANDA_MAXIMA`,
via staging + `MERGE` — o `MERGE` **preserva** o `demanda_max_h1` já existente.

**Como conferir que rodou certo** — célula 3 do notebook:

1. `modelo H0: … | versao: … | canais: 5` — confirme que a versão é a que você
   espera (a mais recente, ou a que você fixou no widget).
2. `linhas com a versao desta rodada: X de Y` — `X` deve cobrir praticamente
   todos os pares (alimentador, ano) esperados (~1.300 × nº de anos). Se `X` for
   muito menor que `Y`, parte da tabela ficou com resultado de rodadas antigas.
3. `nulos em demanda_max_h0: 0` — nulos aparecem quando a série do ano tem menos
   de 144 pontos; alguns são normais (alimentador novo), muitos não.
4. **cobertura por regiao/ano** — a matriz SP/ES × ano não deve ter buracos
   inesperados.
5. `H0/max_bruto (mediana): ~0,9x` e `fracao com H0 ACIMA do maximo bruto: ~0%`
   — este é o teste de sanidade mais importante. O H0 estima a máxima
   **estrutural**, então deve ficar **abaixo** do máximo bruto (que inclui picos
   de medição). Se a mediana der ≈ 1,00, o modelo está apenas copiando a série e
   não está filtrando nada; se der muito baixo (< 0,7), está achatando demanda
   real — em ambos os casos, revise `prob_max` e o treino.

---

## 5. `nb_03a_h1_calibracao` — H1 por calibração linear

**Entradas:** `repo_dir`, `secret_scope`.

**Saída:** coluna `demanda_max_h1` em `james.DEMANDA_MAXIMA`.

**Como conferir que rodou certo:**

1. A seção 2 imprime a **validação cruzada agrupada por alimentador**
   (out-of-sample): `MAE`, `MdAPE`, `HR_5%`, `HR_10%`, com recorte por estado.
   É a única métrica honesta do notebook.
2. `calibracao H1: h1 = a * h0 + b (n_rotulos=N)` — `a` muito longe de 1 ou `b`
   grande sugere viés sistemático do H0 (ou poucos rótulos).
3. Se aparecer `AVISO: apenas N pares rotulados`, a CV não é informativa: junte
   mais rótulos antes de confiar no número.

> ⚠️ A regra de produção "onde há rótulo, usa o valor exato do especialista"
> roda **depois** da avaliação, na seção 3. Nunca calcule métrica sobre esses
> pares — o resultado seria circular (erro zero por construção).

---

## 6. `nb_03b_h1_finetune` — H1 por retreino da rede

**Entradas:** `repo_dir`, `medicao_path`, `grandeza`, `secret_scope`,
`model_h0_path` (vazio = H0 mais recente), `n_pool`, `max_janelas_pool`.

**Saída:** `modelo/ae_h1_multi_<ts>.pt` + coluna `demanda_max_h1` na tabela.

**Como conferir que rodou certo** — célula 6:

1. `H0 de partida: … | versao: …` — confirme que partiu do H0 certo.
2. `feeder-anos rotulados: N` e `exemplos rotulados prontos: M`. **Se `M` for
   bem menor que `N`**, muitos rótulos não acharam medição correspondente
   (grandeza ou ano divergentes entre a tabela de rótulos e o parquet).
3. `H1 salvo: … | versao: …` e a lista de modelos H1 do diretório.
4. Comparação `H0` vs `H1` nos pares rotulados: o **H1 deve ter erro menor**.
   Se não tiver, ajuste `lam` (peso da supervisão) ou `epochs` do fine-tuning.

> ⚠️ Essa comparação é **in-sample** — os mesmos pares supervisionaram o
> retreino. Serve como sanity check de produção, **não** como medida de
> desempenho. Métrica defensável exige teste congelado por alimentador; ver
> [`PLANO_PESQUISA_DISSERTACAO.md`](PLANO_PESQUISA_DISSERTACAO.md), §5 e §7.

---

## 7. Testes locais (sem Spark, sem Databricks)

```bash
pip install -r requirements.txt
```

```bash
python tests/test_versionamento.py
```

```bash
python tests/test_pipeline.py
```

```bash
python tests/test_finetune.py
```

| Teste | O que garante | Resultado esperado |
|---|---|---|
| `test_versionamento` | modelos com timestamp, nunca sobrescritos; "mais recente" é mesmo o último; H0/H1 não se misturam | 5 PASS, segundos |
| `test_pipeline` | o autoencoder remove picos de medição: a máxima do sinal reconstruído fica mais perto da verdade que a da série crua | 3 PASS, ~1 min |
| `test_finetune` | o fine-tuning aproxima a máxima do valor informado pelo especialista (platô sustentado sem Δkm) | PASS, ~2 min |

Rode estes testes **antes** de subir qualquer alteração do `src/` para o
Workspace — eles pegam a maior parte das quebras de contrato sem gastar cluster.

---

## 8. Referência rápida de problemas

| Erro | O que fazer |
|---|---|
| `TypeError: … missing 1 required positional argument` | sincronizar `src/` no Workspace + `restartPython()` |
| `FileNotFoundError: nenhum modelo 'ae_h0_multi_…' em …` | rodar o `nb_01` antes, ou corrigir `repo_dir` |
| `maxResultSize` estourado | não usar `.toPandas()` na série bruta; montar janelas nos workers |
| Falha de login no SQL Server | scope errado, ou chaves com nome diferente de `url`/`database`/`user`/`password` |
| `sem pares rotulados para comparar` | `james.DEMANDA_MAXIMA_TREINO` vazia, ou `grandeza` diferente da usada nos rótulos |
| H0 ≈ máximo bruto | modelo não está filtrando; revisar `prob_max`, épocas e contaminação sintética |

# Parecer metodológico e desenho de pesquisa
## Quantificação do conhecimento especialista na estimação da demanda máxima em regime normal de alimentadores de distribuição

> Documento de trabalho para dissertação de mestrado (ITA).
> Elaborado a partir da proposta do autor e da leitura do código existente no repositório
> (`src/core.py`, `src/core_multi.py`, `notebooks/nb_01…nb_03b`, testes locais).
>
> **Decisão de projeto (2026-08-16): o flag `is_sicoi` foi descartado** — não será
> usado como canal, máscara de perda nem baseline. Consequências incorporadas ao
> longo do documento: o conhecimento de domínio via *features* fica restrito à
> topologia (`comp`/`d_comp`); a ablação de viés indutivo (H3/RQ3) mede apenas os
> canais topológicos; e a separação "evento temporário × mudança estrutural"
> depende do objetivo denoising e da supervisão do especialista.

---

## 0. O que o projeto já contém (diagnóstico do código)

Antes da crítica, o registro do que existe — porque a dissertação deve nascer *deste* estado, não do zero:

- **H0 (não supervisionado)**: autoencoder denoising 1D-CNN, janelas diárias (L=144 @ 10 min), normalização robusta (mediana/IQR) por alimentador-ano, contaminação sintética (platôs e spikes) no treino, máxima anual = quantil 0,999 da reconstrução (`core.py`).
- **H0 multivariado**: 5 canais (valor, derivada, desvio local, `comp`, `d_comp`), perda L1 simples (`core_multi.py`). *(Originalmente havia um 6º canal `is_sicoi` com máscara de perda — removido por decisão de projeto.)*
- **H1 (duas variantes)**: (a) calibração linear global `h1 = a·h0 + b` sobre os pares rotulados (`notebooks/nb_03a_h1_calibracao.py`); (b) fine-tuning da rede com perda de quantil anual `soft_peak` (`notebooks/nb_03b_h1_finetune.py`).
- **Loop humano**: engenheiro rotula no James (R) → `DEMANDA_MAXIMA_TREINO` → retreino.

Observações técnicas pontuais que afetam a pesquisa (não são apenas "bugs de engenharia"):

1. **A calibração do `nb_03a` era avaliada in-sample.** O ajuste `polyfit` usa todos os pares rotulados e, onde há rótulo, copia o valor exato do especialista. Qualquer métrica calculada sobre esses pares é circular. *(Já corrigido: o notebook agora reporta validação cruzada agrupada por alimentador, e a regra de produção roda depois da avaliação.)* No desenho experimental, *toda* avaliação precisa ser out-of-sample por construção.
2. **A amostragem de alimentadores usava `.limit(N)` do Spark** (`nb_01`, `nb_03b`) — isso não é amostra aleatória; é "os N primeiros que o Spark devolver". *(Já corrigido no código para amostragem aleatória com semente registrada; mantido aqui porque a distinção precisa constar na metodologia.)*
3. **`fine_tune_h1` itera sobre *todos* os exemplos rotulados a cada passo de gradiente** — O(n_rótulos) por batch. Funciona com dezenas de rótulos; não escala para os milhares que o experimento de 100% de supervisão vai exigir. Precisa virar mini-batch sobre os rotulados.
4. **O `RobustScaler` é ajustado no ano inteiro, incluindo os períodos anômalos.** Mediana/IQR resistem a spikes, mas um platô de manobra de 70 dias (como no próprio `tests/test_finetune.py`) desloca o IQR. Vale testar escalonamento por quantis mais internos (o `RobustScaler` agora aceita `q_low`/`q_high` como hiperparâmetros).
5. **`prob_max = 0,999` é um hiperparâmetro que define diretamente a resposta.** Ele precisa entrar no protocolo de tuning aninhado (ver §5.6), senão vira um botão implícito calibrado no olho contra o teste.
6. **`d_comp` é mensal e o sinal é de 10 min** — o canal topológico só enxerga reconfigurações que sobrevivem à fotografia mensal do cadastro. Isso deve ser declarado como limitação de observabilidade, não escondido.

---

## 1. Crítica central da proposta (papel de orientador)

A ideia de fundo — medir desempenho × fração de supervisão do especialista — é sólida e publicável. Mas há **cinco problemas estruturais** que, se não tratados, derrubam a defesa. Em ordem de gravidade:

### 1.1 O "ground truth" é opinião de especialista, e você não conhece o ruído dessa opinião

Toda a pesquisa mede distância a rótulos que são **julgamentos humanos sobre um construto latente** ("máxima em regime normal") que não tem definição operacional escrita. Consequências:

- Sem conhecer a variância entre especialistas, a curva de aprendizado não tem teto interpretável. Se dois engenheiros discordam em mediana 6% entre si, um modelo com erro de 5% já está *no ruído do rótulo* — e reportar 3% seria overfitting ao anotador, não ganho real.
- O ponto de saturação que você quer identificar é matematicamente limitado pelo ruído do rótulo. Sem estimá-lo, "saturou" e "bateu no teto do ruído" são indistinguíveis.

**Correção obrigatória:** um **estudo de concordância inter-anotadores**: 2–3 engenheiros rotulam *independentemente* uma amostra aleatória de 100–200 pares (alimentador, ano), estratificada por estado e por dificuldade. Calcular ICC, desvio absoluto mediano entre anotadores e distribuição das discrepâncias. Esse número vira: (a) o teto de desempenho esperável; (b) a **margem δ do teste de equivalência** (§7.3); (c) uma seção metodológica que praticamente nenhum trabalho aplicado tem — é contribuição por si só. Clássico de referência: modelos de anotador de Dawid & Skene (1979).

Adicionalmente, escrever uma **definição operacional** do construto (ex.: "máxima demanda sustentada por ≥ T minutos, atribuível à carga própria do alimentador na configuração de rede vigente ao fim do ano, excluídos períodos de transferência de carga e erros de medição"). Se os especialistas não conseguirem concordar com uma definição escrita, isso é um achado — e muda a natureza do problema.

### 1.2 Risco de circularidade: os rótulos podem estar ancorados no próprio modelo

O fluxo do James mostra o gráfico ao engenheiro **possivelmente com a predição H0 exibida** e ele "clica no valor correto". Se o rótulo foi produzido vendo a saída do modelo, há **viés de ancoragem**: o modelo será avaliado contra rótulos que ele mesmo influenciou, inflando artificialmente o desempenho e achatando a curva de aprendizado.

**Correção:** documentar o protocolo de rotulagem historicamente usado; para o estudo de concordância (§1.1) e para qualquer rótulo novo (2026), rotular **às cegas** (sem exibir H0/H1). Se os rótulos históricos foram ancorados, isso vira uma limitação declarada e reforça a necessidade da validação prospectiva cega de 2026 (§5.7).

### 1.3 "0% de conhecimento especialista" não existe neste projeto — e isso muda a pergunta

O H0 já embute conhecimento especialista em três lugares:

- **Nas features**: `comprimento_rede`/`d_comp` são exatamente as pistas topológicas que o engenheiro usa;
- **Na arquitetura/perda**: a contaminação sintética (platôs + spikes) codifica a *teoria do especialista* sobre o que é anomalia;
- **No pós-processamento**: o quantil 0,999 é uma escolha de engenharia informada.

Portanto o eixo x do seu experimento **não é "quantidade de conhecimento especialista"; é "quantidade de rótulos"**. Isso não enfraquece a pesquisa — fortalece, se você usar a taxonomia certa: *informed machine learning* (von Rueden et al., IEEE TKDE 2021) distingue conhecimento injetado via **dados de treino (rótulos)**, via **features**, via **arquitetura/perda** e via **restrições**. A pergunta científica mais rica que emerge é:

> **Qual é a relação de substituição entre conhecimento embutido como viés indutivo (features, perda, arquitetura) e conhecimento fornecido como rótulos?**

Concretamente: a curva de aprendizado do modelo *com* os canais de conhecimento (`comp`/`d_comp`) deve saturar antes (menos rótulos) que a do modelo *sem* eles. Medir esse deslocamento horizontal da curva é uma contribuição mais forte e mais original que a curva isolada.

### 1.4 A grade linear 0–100% em passos de 10% está errada para curvas de aprendizado

Com ~1.300 alimentadores × ~7 anos rotuláveis ≈ **7.000–9.000 pares (alimentador, ano)**, 10% já são ~700–900 rótulos. Toda a literatura de curvas de aprendizado (Cortes et al. 1994; Hestness et al. 2017; revisão de Viering & Loog, TPAMI 2023) mostra comportamento tipo lei de potência: a região informativa — onde o ganho marginal é grande e onde está a resposta prática ("quantas análises humanas são necessárias?") — fica **abaixo dos seus primeiros 10%**. Uma grade linear gasta 8 dos 11 níveis na região plana da curva.

**Correção:** grade **logarítmica em número absoluto de rótulos**, por exemplo:
`n ∈ {0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, N_total}`.
Reportar em n absoluto (a pergunta prática é "quantas análises humanas", não "que fração de um dataset arbitrário"). A fração pode aparecer como eixo secundário.

### 1.5 "Avaliar nos dados que não foram usados como supervisão" faz o conjunto de teste mudar a cada corrida — isso invalida as comparações

Se em cada nível/repetição o teste é "o complemento da amostra supervisionada", então: (a) o teste muda de tamanho e composição entre níveis (no nível 90% o teste tem 10% dos dados; no nível 10%, 90%); (b) as métricas entre níveis não são comparáveis; (c) não há pareamento possível para testes estatísticos.

**Correção:** **conjunto de teste fixo, congelado antes de tudo**, estratificado por estado e ano, jamais usado como supervisão em nenhum nível. Os níveis de supervisão são amostrados apenas do *pool* de treino restante. Detalhes no §5.

---

## 2. Formulação refinada do problema

**Problema (formulação matemática).** Para cada alimentador *f* e ano *t*, a série observada é
`x_{f,t}(τ) = s_{f,t}(τ) + m_{f,t}(τ) + e_{f,t}(τ)`,
onde `s` é a carga estrutural (regime normal na configuração vigente da rede), `m` são efeitos operacionais temporários (manobras, transferências de carga) e `e` são erros de medição/outliers. O alvo é o funcional
`y_{f,t} = max_τ s_{f,t}(τ)`
(ou um quantil alto de `s`, conforme a definição operacional do §1.1). `s` não é observável; o que existe são julgamentos de especialistas `ỹ_{f,t} = y_{f,t} + η`, com ruído de anotação `η` de variância desconhecida (a ser estimada). **A tarefa é estimação retrospectiva** (o ano inteiro é observado antes da estimativa), **não previsão** — isso deve ser dito explicitamente, pois dissolve várias exigências de validação temporal que se aplicariam a forecasting e define quais formas de "ver o futuro" são legítimas (usar o ano completo do próprio par avaliado: sim; usar rótulos de anos futuros de outros pares no treino: cuidado — ver §5.4).

**Pergunta principal de pesquisa (RQ):**

> Quantos exemplos rotulados por especialistas são necessários para que um modelo de aprendizado de máquina estime a demanda máxima em regime normal de alimentadores de distribuição com desempenho estatisticamente equivalente (margem δ derivada da concordância inter-especialistas) ao de um modelo treinado com todos os rótulos disponíveis — e como essa quantidade depende do conhecimento de domínio embutido no modelo como viés indutivo?

**Perguntas secundárias:**

- RQ2: Representações auto-supervisionadas aprendidas sem rótulos deslocam a curva de aprendizado (mesmo desempenho com menos rótulos) em relação a modelos supervisionados do zero?
- RQ3: O conhecimento embutido como features/perda (topologia `comp`/`d_comp`, contaminação sintética) substitui quantos rótulos? (distância horizontal entre curvas com/sem esses canais)
- RQ4: O que se transfere entre concessionárias (SP↔ES)? Quantos rótulos do domínio-alvo recuperam a perda de transferência?
- RQ5 (extensão): seleção ativa de exemplos (active learning) reduz o número de rótulos necessários em relação à amostragem aleatória?

## 3. Hipóteses científicas (falseáveis)

- **H1 (forma da curva):** o erro em teste segue curva saturante tipo lei de potência `E(n) ≈ E∞ + a·n^(−b)` no número de rótulos *n*; o ganho marginal decresce monotonicamente. *Teste:* ajuste da curva paramétrica + comparação com alternativas (exponencial, log-linear) por qualidade de ajuste.
- **H2 (saturação):** existe `n* ≪ N` tal que o desempenho com `n*` rótulos é **equivalente** (TOST, margem δ = dispersão inter-anotadores) ao desempenho com `N` rótulos. *Teste:* equivalência pareada nível-a-nível; `n*` = menor nível que passa.
- **H3 (valor do viés indutivo):** para desempenho-alvo fixo, o modelo com canais de conhecimento de domínio requer significativamente menos rótulos que o modelo só-carga; formalmente, deslocamento horizontal `Δn(alvo) > 0`. *Teste:* comparação das curvas ajustadas com IC por bootstrap.
- **H4 (pré-treino auto-supervisionado):** representações pré-treinadas sem rótulos reduzem `n*` em relação ao mesmo preditor treinado do zero. *Teste:* idem H3, fator "com/sem pré-treino".

(H5 opcional, se couber no cronograma: transferência SP↔ES degrada desempenho em quantidade mensurável, majoritariamente recuperável com poucos rótulos do domínio-alvo.)

**Objetivo geral:** quantificar empiricamente a eficiência de rótulos de especialistas na estimação da máxima demanda em regime normal, e a interação entre rótulos e conhecimento de domínio embutido.

**Objetivos específicos:** (i) definir operacionalmente o construto e medir concordância inter-especialistas; (ii) construir baselines não supervisionadas fortes; (iii) construir a curva desempenho × n rótulos com protocolo estatístico pré-registrado; (iv) estimar `n*` por equivalência; (v) medir o efeito do viés indutivo (ablação de canais) e do pré-treino SSL; (vi) validar prospectivamente em 2026 com rotulagem cega.

---

## 4. O que é ciência e o que é engenharia aqui

| Componente | Natureza |
|---|---|
| Pipeline Databricks, MERGE no SQL Server, integração com o James | Engenharia (vai para apêndice/capítulo de materiais) |
| Autoencoder denoising em si | Engenharia (arquitetura conhecida; não defender como novidade) |
| Definição operacional do construto + estudo de concordância | **Ciência** (metodologia) |
| Curva de aprendizado com equivalência estatística e `n*` | **Ciência** (resultado empírico principal) |
| Substituição viés indutivo ↔ rótulos (ablações) | **Ciência** (a contribuição conceitual mais forte) |
| Transferência SP↔ES | Ciência (secundária) |
| Validação prospectiva cega 2026 | **Ciência** (rigor raro em trabalhos aplicados) |
| "Aplicar IA no setor elétrico" | Não é contribuição — não usar como claim |

---

## 5. Desenho experimental

### 5.1 Unidade experimental e particionamento

- Unidade de rótulo: par (alimentador, ano). Unidade de agrupamento: **alimentador** (anos do mesmo alimentador são fortemente correlacionados — mesma topologia, mesma carga, mesmo padrão; tratá-los como independentes infla o desempenho).
- **Particionar por alimentador**, não por par: sortear ~20% dos alimentadores (estratificado por estado e porte) para **teste congelado**; todos os seus anos ficam no teste. O restante é o pool de treino/supervisão. Isso responde ao cenário B (alimentador nunca visto), que é o mais exigente e o mais honesto como métrica principal.
- Para o cenário A (mesmo alimentador, ano novo): **segundo eixo de avaliação**, com split por ano dentro dos alimentadores do pool (ex.: supervisão só até ano t, avaliação no ano t+1 dos mesmos alimentadores). Reportar separadamente. Não misturar A e B numa métrica só — a diferença entre eles é em si um resultado (quanto o modelo "memoriza o alimentador" vs "aprende o conceito").
- Cenários C/D (treina SP → testa ES e vice-versa): experimento secundário para RQ4/H5. E (mistura) é o desenho principal com estratificação. F (2026): §5.7.

### 5.2 Níveis de supervisão

Grade logarítmica em **n absoluto** (§1.4): `{0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, N}` (ajustar o topo ao total real do pool). Amostragem estratificada por estado (proporcional) e, dentro de estado, aleatória por alimentador-ano do pool.

### 5.3 Repetições e pareamento (responde "10 repetições bastam?")

- Usar **amostras aninhadas por cadeia**: em cada repetição *r*, sortear uma permutação do pool e definir os conjuntos de rótulos como prefixos (`S_10 ⊂ S_25 ⊂ … ⊂ S_N`). Isso (a) reduz drasticamente a variância das *diferenças* entre níveis adjacentes (pareamento natural), (b) reflete o processo real de acumulação de rótulos, (c) torna a curva quase monótona por construção.
- 10 repetições são razoáveis para estimar médias com o desenho aninhado + teste fixo; são pouco para caracterizar a variância nos níveis baixos (n=10, 25), onde a variância entre amostras é a maior parte da história. **Recomendação: piloto com 5 repetições em 3 níveis (baixo/médio/alto) para estimar σ entre repetições; dimensionar R pela largura de IC desejada (regra: IC95 ≈ ±2σ/√R). Expectativa realista: R=20–30 nos níveis ≤100 rótulos, R=10 nos altos.** Custo computacional é gerenciável porque os modelos são pequenos.
- Semente de tudo (partição, permutações, inicialização de pesos) registrada e versionada.

### 5.4 Prevenção de leakage — lista de controles

1. Teste congelado por alimentador antes de qualquer experimento; hash da lista registrado.
2. Pré-treino SSL/H0 **apenas com séries dos alimentadores do pool** (não do teste). Custa pouco e elimina a discussão sobre transdução; se quiser, uma ablação pode medir o efeito de incluir o teste no pré-treino (resultado interessante sobre SSL, não um vazamento acidental).
3. Escalonamento (`RobustScaler`) sempre por alimentador-ano com estatísticas do próprio par — legítimo em estimação retrospectiva; nunca usar estatísticas do futuro de *outros* pares para calibrar o par avaliado.
4. Tuning de hiperparâmetros (incluindo `prob_max`, `lam`, `tau`, épocas) por **validação interna ao pool, refeita em cada nível de supervisão** — o modelo de n=10 não pode herdar hiperparâmetros otimizados com 5.000 rótulos, senão o nível baixo está contaminado por conhecimento do nível alto. (Alternativa pragmática e defensável: fixar hiperparâmetros a priori num piloto com um subconjunto descartado depois.)
5. A regra "onde há rótulo, usa o valor exato do especialista" (`nb_03`) é de produção — **proibida na avaliação**.
6. Rotulagem cega para todo rótulo novo (§1.2).
7. Nenhuma decisão de desenho (métrica, margem δ, grade, critério de sucesso) tomada após ver resultados do teste — escrever um **pré-registro** (mesmo que interno, com data) antes de rodar a bateria final.

### 5.5 Correlação entre anos e entre estados

- O particionamento por alimentador (§5.1) trata a correlação intra-alimentador no split. Na *inferência estatística*, tratá-la de novo: erros do mesmo alimentador em anos diferentes não são independentes → usar **modelos de efeitos mistos** (efeito aleatório de alimentador e de repetição) ou bootstrap **por cluster de alimentador** para os ICs (§7.4).
- SP vs ES: estratificar splits e amostras; reportar métricas por estado sempre; incluir estado como efeito fixo no modelo misto. Se as distribuições forem muito diferentes (verificar: porte, sazonalidade, qualidade do cadastro de extensão nos dois sistemas de origem), considerar normalizações específicas — e isso alimenta RQ4.

### 5.6 Famílias de modelo por nível (o fator "método")

Desenho fatorial mínimo viável: **método × n rótulos**, com o mesmo protocolo de dados para todos:

- B0. Baselines sem aprendizado (obrigatórias — ver §6.1);
- M1. AE denoising atual (H0) + calibração linear com n rótulos (é o sistema atual — vira baseline honesta);
- M2. AE denoising + fine-tuning `soft_peak` com n rótulos (H1 atual);
- M3. Pré-treino SSL (ex.: TS2Vec ou masked modeling) → *features* do ano → regressor raso (gradient boosting / linear) com n rótulos;
- M4. Mesmo preditor de M3 **sem** pré-treino (supervisionado do zero) — par de ablação para H4;
- M5. Ablação de canais de M2/M3 sem `comp`/`d_comp` — par de ablação para H3.

Isso dá ~5–6 curvas de aprendizado comparáveis. Mais que isso não cabe num mestrado com rigor; escolher no piloto os 3–4 que seguem para a bateria completa.

### 5.7 O ano de 2026

- 2026 está **incompleto** (dados até ago/2026) e **sem rótulos**. Duas consequências: (i) a máxima anual de um ano parcial não é comparável — só usável se a definição operacional restringir ao período observado, e com a ressalva de sazonalidade (no Sudeste a ponta tende ao verão — jan/fev já observados, o que ajuda, mas deve ser verificado nos dados, não assumido); (ii) 2026 **não pode ser conjunto de teste** da curva de aprendizado.
- Uso correto de 2026: **estudo prospectivo cego** — congelar o modelo final, predizer 2026, e *depois* solicitar a 1–2 engenheiros a rotulagem cega de uma amostra aleatória (~100–200 pares). Comparar. É a validação mais forte possível do trabalho inteiro (elimina ancoragem e qualquer leakage por construção) e rende um capítulo curto de altíssimo valor.

---

## 6. Métodos: famílias, adequação e veredicto

### 6.1 Baselines não supervisionadas (obrigatórias — o experimento não se defende sem elas)

Se uma regra simples chegar perto do especialista, todo o aparato neural fica sem justificativa — a dissertação precisa demonstrar que não chega (ou aceitar que chega, o que também é um resultado).

| Baseline | Mecanismo |
|---|---|
| Máximo bruto | `max(x)` — o "erro" que motiva tudo; quantifica o problema |
| Quantis altos | P99, P99.5, P99.9 da série bruta |
| Filtro de Hampel / mediana móvel + máximo | estatística robusta clássica |
| Máximo sustentado | maior valor mantido por ≥ T minutos (aproxima a definição operacional diretamente) |

Custo: dias. Interpretabilidade: total. **Hipótese a testar: parte do trabalho do especialista é reproduzida por regras; o valor do ML está no resíduo.**

### 6.2 Detecção de pontos de mudança + classificação de segmentos (recomendação forte)

- **Hipótese:** o raciocínio do especialista ("subiu abruptamente, permaneceu, voltou ao patamar anterior → manobra") é literalmente segmentação por mudança de regime + classificação do segmento.
- **Mecanismo:** PELT (Killick et al., JASA 2012) ou BOCPD (Adams & MacKay, 2007) segmenta a série (biblioteca `ruptures`; revisão: Truong, Oudre & Vayatis, Signal Processing 2020); cada segmento vira um vetor de atributos (nível, duração, retorno ao patamar, Δkm no período, sazonalidade); um classificador (com os n rótulos, via supervisão fraca ou direta) marca segmentos como normal/anômalo; a resposta é o máximo (ou quantil) sobre os segmentos normais.
- **Vantagens:** altamente interpretável (o modelo *mostra* qual bloco excluiu — exatamente o que um engenheiro quer auditar); barato; alinhado ao conhecimento do domínio; os rótulos do especialista são usados de forma eficiente (um rótulo anual informa a classificação de poucos segmentos).
- **Limitações:** sensível à parametrização da segmentação; manobras que coincidem com transição sazonal confundem.
- **Adequação:** máxima. **Publicabilidade:** boa (híbrido interpretável + estudo de eficiência de rótulos).

### 6.3 Autoencoder denoising (o H0 atual)

- **Hipótese:** o gargalo + perda L1 + contaminação sintética recuperam a componente estrutural.
- **Crítica:** a contaminação sintética é uma *simulação da teoria do especialista* — o modelo aprende a remover o que você *disse* que é anomalia. Isso é viés indutivo forte (ótimo para H3), mas significa que o H0 nunca aprenderá a remover um tipo de manobra que você não simulou. Janela de 1 dia é míope para manobras de semanas: o AE reconstrói fielmente um platô de 30 dias porque cada janela diária dentro dele parece normal — a detecção do platô vem só da contaminação de treino e dos canais de contexto. Vale testar janelas maiores (semana) ou hierarquia.
- **Papel na dissertação:** método incumbente/baseline aprendida — não o protagonista.

### 6.4 Aprendizado auto-supervisionado de representações

- **Candidatos:** TS2Vec (Yue et al., AAAI 2022 — contrastivo hierárquico, robusto, código maduro); TS-TCC (Eldele et al., IJCAI 2021); triplet de Franceschi et al. (NeurIPS 2019); masked modeling estilo transformer (Zerveas et al., KDD 2021; PatchTST, Nie et al., ICLR 2023); SimMTM (NeurIPS 2023); TF-C (Zhang et al., NeurIPS 2022) para transferência entre domínios (útil em RQ4).
- **Hipótese:** pré-treino sem rótulos sobre as ~1.300×7 séries produz representações em que "regime normal vs perturbado" é linearmente separável, de modo que poucos rótulos bastam para o regressor final — é o mecanismo clássico pelo qual SSL melhora eficiência de rótulos.
- **Mecanismo na prática:** embeddings por janela → agregação por ano (pooling + quantis dos embeddings) → regressor raso treinado com n rótulos prediz `y` (ou prediz uma *correção* sobre uma baseline robusta, o que costuma ser mais estável).
- **Vantagens:** é o teste direto de H4; literatura forte; custo moderado (GPU único).
- **Limitações:** interpretabilidade baixa (mitigar com atribuição por janela: quais dias sustentam a máxima predita); risco de o embedding capturar sazonalidade e ignorar exatamente os eventos raros que importam (verificar com probing).
- **Adequação:** alta como *fator experimental*. Escolher **um** método SSL principal (TS2Vec é a escolha pragmática) — comparar cinco SSLs entre si é outra dissertação.

### 6.5 Modelos probabilísticos de regime (candidato forte, mais arriscado)

- **Candidatos:** HMM/HSMM com regimes "normal/transferência/erro", modelos de espaço de estados estruturais (nível + sazonalidade + componente de intervenção), Markov-switching.
- **Hipótese:** o conceito de "regime de operação" é nativamente representado; a posterior sobre regimes dá incerteza calibrada e interpretável.
- **Vantagens:** formulação elegante, incerteza por construção, ótimo para a discussão.
- **Limitações:** custo de modelagem alto (sazonalidade múltipla 10-min é pesada para HMM ingênuo — exigiria trabalhar sobre agregados horários/diários); inferência lenta em 9.000 séries-ano.
- **Veredicto:** incluir **apenas se** o cronograma permitir; caso contrário, discutir como trabalho futuro. Não é obrigatório para as hipóteses centrais.

### 6.6 Detecção de anomalias "de prateleira"

Anomaly Transformer (Xu et al., ICLR 2022), USAD (Audibert et al., KDD 2020), Matrix Profile (Yeh et al., ICDM 2016), deep SVDD. **Cuidado conceitual:** essas técnicas detectam o *anômalo pontual/subsequencial*; seu problema é o inverso — estimar um funcional do *normal* na presença de anomalias *longas e estruturadas* (dias/semanas), que é o regime onde detectores de subsequência falham ou marcam tudo. Matrix Profile merece entrar como baseline barata de detecção de discórdias; os detectores neurais de prateleira, não — justificar a exclusão na revisão (isso demonstra maturidade, não preguiça).

### 6.7 Foundation models de séries temporais

TimesFM (Das et al., ICML 2024), Chronos (Ansari et al., 2024), MOMENT (Goswami et al., ICML 2024), Moirai. São majoritariamente orientados a *forecasting*; para seu problema serviriam como (a) extrator de embeddings zero-shot (MOMENT) ou (b) detector por erro de previsão. **Veredicto:** um experimento pequeno e contido (embeddings MOMENT no lugar do SSL treinado em casa, mesmo protocolo) é interessante e atual; não fazer disso um pilar — a contribuição da dissertação não depende deles, e a comparação "foundation vs treinado no domínio" é um bônus com boa chance de virar seção de destaque.

### 6.8 Supervisão fraca programática (opcional, conceitualmente valiosa)

Codificar as heurísticas verbalizáveis do especialista como *labeling functions* (estilo Snorkel — Ratner et al., VLDB 2017): "bloco que retorna ao patamar em < X dias é manobra", "Δkm grande legitima novo patamar". Isso permite um braço experimental **"conhecimento como regras"** contra **"conhecimento como exemplos"** — mede-se quantos rótulos as regras substituem. É a extensão mais alinhada com a sua pergunta de fundo; se não couber, registrar como trabalho futuro explícito.

---

## 7. Avaliação e análise estatística

### 7.1 Métricas

- **Primária:** *hit rate* com tolerância — `HR_ε = P(|ŷ−ỹ|/ỹ ≤ ε)` para ε ∈ {5%, 10%} — é a métrica que o uso em planejamento entende, é robusta a caudas e permite formulação binomial limpa.
- **Secundárias:** erro relativo absoluto mediano (MdAPE); MAE e RMSE em unidades físicas (MVA/A); distribuição do **erro com sinal** — em planejamento, *subestimar* a máxima é pior que superestimar; reportar assimetria e considerar métrica ponderada (tipo pinball) na discussão.
- **Sobre o MAPE:** os denominadores (máximas anuais) nunca são ~0, então o MAPE não explode; mas ele é assimétrico (pune sobre-estimativa proporcionalmente menos) e dominado por cauda. Usável como métrica de compatibilidade com a literatura, **não** como métrica primária; preferir MdAPE + HR_ε.
- **Recortes obrigatórios:** por estado, por ano, por porte do alimentador, por "dificuldade" (ex.: pares com/sem manobra registrada). A média global esconde exatamente os casos que motivam o trabalho.
- **Robustez:** curvas de HR_ε variando ε (uma "curva de tolerância" por nível de supervisão condensa muito resultado num gráfico só).

### 7.2 Critério de sucesso pré-registrado

Formato recomendado: **"HR_10% ≥ π₀ no teste congelado, com IC95 inferior acima de π₀"**, com π₀ fixado *antes* (sugestão: π₀ ancorado no estudo inter-anotadores — ex.: a taxa com que um segundo especialista "acerta" o primeiro dentro de 10%). Assim o modelo é exigido a ser tão consistente com um especialista quanto outro especialista o é — critério defensável e não arbitrário.

### 7.3 Saturação e "estatisticamente indistinguível": teste de equivalência, não ausência de significância

Ponto metodológico crítico: **não** declarar saturação porque um teste de diferença "não deu significativo" (ausência de evidência ≠ evidência de ausência; com R=10 a potência é baixa e "não significativo" é fácil). Usar **TOST** (two one-sided tests; Lakens, 2017): declarar o nível *n* equivalente ao nível *N* se o IC90 da diferença pareada de desempenho estiver inteiramente dentro de ±δ, com **δ = dispersão inter-anotadores** (§1.1). `n* = min{n: equivalente}`. Complementar com o ajuste paramétrico da curva (H1): `n*` também pode ser lido da curva ajustada como o n em que `E(n) − E∞ ≤ δ/2`, com IC por bootstrap sobre repetições — os dois caminhos convergindo é resultado forte.

### 7.4 Inferência

- **Modelo de efeitos mistos** como espinha dorsal: erro (ou acerto binário, via GLMM logístico para HR) ~ nível de supervisão (fator ou spline em log n) + estado + efeitos aleatórios de alimentador e de repetição. Justificativa: trata a não-independência (mesmo alimentador em vários anos; mesma repetição em vários níveis) que invalida ANOVA clássica e testes não pareados ingênuos.
- **Bootstrap por cluster de alimentador** (BCa) para ICs de métricas agregadas — não bootstrap por par, que quebra a estrutura de dependência.
- **Comparações entre níveis adjacentes:** diferenças pareadas dentro de cada cadeia aninhada (§5.3), teste de Wilcoxon pareado como não-paramétrico de apoio, tamanho de efeito (Cliff's delta ou diferença média com IC) sempre junto do p-valor.
- **Comparações entre métodos:** sobre o mesmo teste e mesmas cadeias; se muitos métodos × níveis, correção de múltiplas comparações (Holm) e, para o quadro geral, o arcabouço de Demšar (JMLR 2006).
- **ANOVA:** só como descrição preliminar; as suposições (independência, homocedasticidade) não valem aqui — dizer isso explicitamente na dissertação é ponto a favor.

---

## 8. Conceitos da literatura: o que se aplica e o que não

| Conceito | Aplica? | Papel |
|---|---|---|
| **Label-efficient / semi-supervised learning** | Sim — é o núcleo | Curva de aprendizado, SSL + poucos rótulos (survey: van Engelen & Hoos, Machine Learning 2020) |
| **Learning curves / sample efficiency** | Sim — é o instrumento | Forma funcional, ajuste, extrapolação (Viering & Loog 2023; Cortes 1994; Hestness 2017; Rosenfeld et al., ICLR 2020) |
| **Informed / knowledge-informed ML** | Sim — é o enquadramento conceitual | Taxonomia de von Rueden et al. (TKDE 2021) para distinguir conhecimento via rótulos/features/perda; Karniadakis et al. (Nature Rev. Physics 2021) como pano de fundo |
| **Self-supervised / representation learning** | Sim — fator experimental | H4 |
| **Weak supervision** | Parcial | labeling functions codificando heurísticas do especialista são extensão opcional (Ratner 2017) |
| **Human-in-the-loop ML** | Parcial | Descreve o *sistema* James (contexto/motivação), não o experimento offline; citar como enquadramento (survey de Wu et al., FGCS 2022; livro de Monarch 2021), sem prometer um estudo de interação humana que não será feito |
| **Active learning** | Extensão | RQ5/trabalho futuro; a curva com amostragem aleatória é o *lower bound* que o AL tentaria superar (Settles 2009) |
| **Learning from noisy/subjective labels** | Sim — e costuma ser esquecido | Ruído de anotador, Dawid & Skene 1979; Frénay & Verleysen (TNNLS 2014) |
| Imitation learning / LfD | Não | É regressão supervisionada de julgamentos, não aprendizado de política sequencial — usar esse vocabulário confundiria a banca |
| Knowledge distillation | Não | Não há modelo professor |
| "Aprendizado de conhecimento implícito" como termo novo | Evitar cunhar termo | O fenômeno já tem nome na literatura: aprender um construto definido por julgamento humano a partir de exemplos rotulados. A novidade não está no nome |

---

## 9. Novidade científica — avaliação honesta, em ordem de força

1. **Quantificação da substituição entre viés indutivo de domínio e rótulos (H3) num problema real com rótulos de especialista caros.** Curvas de aprendizado existem; ablações existem; a *interseção* — medir deslocamento horizontal de curvas de aprendizado causado por conhecimento de domínio embutido, com equivalência estatística e margem derivada de concordância inter-anotadores — é rara e defensável. É o claim mais forte.
2. **Metodologia de "quantos rótulos bastam" com critério de equivalência ancorado no ruído inter-anotadores.** Transferível para qualquer problema de rotulagem cara; é contribuição metodológica mesmo que os números específicos sejam do domínio.
3. **O problema em si, formalizado:** estimação de funcional do regime normal (máxima estrutural) sob contaminação por eventos operacionais longos — distinto de detecção de anomalias pontuais e de forecasting; a formalização + benchmark de baselines é contribuição de domínio sólida para venue de energia (IEEE Trans. Smart Grid / Power Systems, Applied Energy, PSCC).
4. **Validação prospectiva cega (2026)** — rigor incomum em ML aplicado a energia; fortalece tudo acima.
5. Transferência SP↔ES — interessante, secundária.
6. ~~"Framework HITL"~~, ~~"nova arquitetura de AE"~~, ~~"primeira aplicação de ML em alimentadores"~~ — não defender; são fracos ou falsos.

**Estratégia de publicação plausível:** artigo de domínio (formalização + método + validação prospectiva) em periódico de energia; artigo metodológico (curvas de eficiência de rótulo × viés indutivo) em venue de ML aplicado (ECML-PKDD ADS, workshops NeurIPS/ICML de energia ou de label efficiency).

---

## 10. Limitações e riscos a declarar (e mitigar)

1. **Rótulo = julgamento** — mitigado pelo estudo de concordância; ainda assim, o modelo aprende "o que os engenheiros da EDP consideram normal", não uma verdade física.
2. **Ancoragem dos rótulos históricos no H0** (§1.2) — risco alto se o James exibia a predição; investigar e declarar; a validação cega de 2026 é o antídoto.
3. **Descarte do `is_sicoi`** — decisão de projeto que remove o único sinal explícito de manobra operacional. Consequência: o modelo precisa inferir "evento temporário" apenas da forma da série e da topologia, o que tende a exigir mais rótulos (deslocamento da curva de aprendizado — em si um resultado reportável). Declarar a motivação do descarte (confiabilidade/cobertura do flag) na dissertação.
4. **2020 (COVID) é ano atípico** — verificar se os rótulos de 2020 se comportam diferente; considerar análise de sensibilidade excluindo-o.
5. **Dados proprietários** — reprodutibilidade limitada; mitigar com código aberto + gerador sintético calibrado (o `tests/test_finetune.py` já aponta o caminho) e, se possível, amostra anonimizada.
6. **Generalização institucional** — dois estados da mesma controladora; curvas podem não transferir para outras distribuidoras (declarar escopo).
7. **Risco de cronograma** — a matriz método × nível × repetição explode; o piloto (§5.3) existe para podar. Prioridade se faltar tempo: baselines + M2 + M3/M4 + estatística bem-feita > mais métodos.

---

## 11. Estrutura da dissertação (o que cada capítulo deve demonstrar)

1. **Introdução** — o problema prático (planejamento precisa da máxima *estrutural*; o máximo bruto erra por X% em Y% dos casos — quantificar com dados reais logo aqui) e a pergunta científica; demonstrar que existe uma questão geral (custo de conhecimento especialista) instanciada num problema concreto.
2. **Problema** — formulação matemática (§2), definição operacional do construto, por que máximo bruto/estatística simples falham (com exemplos reais anonimizados).
3. **Hipóteses** — H1–H4 como enunciados falseáveis, cada um com seu teste previsto (tabela hipótese → experimento → métrica → critério).
4. **Objetivos** — geral + específicos (§3), mapeados 1:1 aos capítulos de resultados.
5. **Revisão bibliográfica** — quatro eixos: (a) séries de carga e demanda de ponta; (b) anomalias/regimes/CPD em séries; (c) SSL e eficiência de rótulos; (d) informed ML e rotulagem humana. Demonstrar o *gap*: nenhum trabalho quantifica rótulos-de-especialista necessários para este tipo de funcional.
6. **Fundamentação teórica** — curvas de aprendizado (formas funcionais), teste de equivalência, modelos mistos, os métodos usados. Só o que será usado — nada de survey decorativo.
7. **Metodologia** — desenho experimental completo (§5), protocolo de rotulagem e estudo de concordância, controles de leakage, pré-registro. Deve permitir reprodução por terceiros.
8. **Base de dados** — caracterização dos 1.300 alimentadores, qualidade (faltantes, cadastro de extensão), justificativa do descarte do `is_sicoi`, diferenças SP/ES, ética/anonimização.
9. **Experimentos** — matriz método × nível × repetição, hiperparâmetros, custo computacional, sementes.
10. **Resultados** — curvas de aprendizado com ICs; `n*` por equivalência; ablações (H3/H4); recortes; transferência; 2026 cego.
11. **Discussão** — o que a forma da curva diz sobre o conhecimento do especialista; onde o modelo erra (análise qualitativa de casos — a banca vai pedir); rótulos vs regras; implicações operacionais (quantas horas de engenheiro economizadas).
12. **Contribuições** — §9, na ordem de força, sem inflar.
13. **Limitações** — §10, sem esconder.
14. **Trabalhos futuros** — active learning (RQ5), supervisão fraca programática, foundation models, outras distribuidoras.
15. **Conclusão** — responder literalmente a RQ com número e intervalo: "com n* = … rótulos (IC …), desempenho equivalente a rotulagem completa, margem δ = …".

---

## 12. Referências de partida (verificar todas antes de citar — não citar de segunda mão)

**Curvas de aprendizado / eficiência amostral**
- Viering & Loog, *The Shape of Learning Curves: A Review*, IEEE TPAMI, 2023.
- Cortes et al., *Learning Curves: Asymptotic Values and Rate of Convergence*, NIPS 1994.
- Hestness et al., *Deep Learning Scaling is Predictable, Empirically*, arXiv:1712.00409, 2017.
- Rosenfeld et al., *A Constructive Prediction of the Generalization Error Across Scales*, ICLR 2020.

**SSL / representação de séries temporais**
- Yue et al., *TS2Vec: Towards Universal Representation of Time Series*, AAAI 2022.
- Franceschi et al., *Unsupervised Scalable Representation Learning for Multivariate Time Series*, NeurIPS 2019.
- Eldele et al., *Time-Series Representation Learning via Temporal and Contextual Contrasting* (TS-TCC), IJCAI 2021.
- Zerveas et al., *A Transformer-based Framework for Multivariate Time Series Representation Learning*, KDD 2021.
- Nie et al., *A Time Series is Worth 64 Words* (PatchTST), ICLR 2023.
- Zhang et al., *Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency* (TF-C), NeurIPS 2022.
- Dong et al., *SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling*, NeurIPS 2023.

**Anomalias / regimes / CPD**
- Blázquez-García et al., *A Review on Outlier/Anomaly Detection in Time Series Data*, ACM Computing Surveys, 2021.
- Truong, Oudre & Vayatis, *Selective Review of Offline Change Point Detection Methods*, Signal Processing, 2020.
- Killick, Fearnhead & Eckley, *Optimal Detection of Changepoints with a Linear Computational Cost* (PELT), JASA 2012.
- Adams & MacKay, *Bayesian Online Changepoint Detection*, arXiv:0710.3742, 2007.
- Yeh et al., *Matrix Profile I*, ICDM 2016.
- Xu et al., *Anomaly Transformer*, ICLR 2022. / Audibert et al., *USAD*, KDD 2020 (para justificar exclusão).

**Foundation models de séries**
- Das et al., *A Decoder-only Foundation Model for Time-Series Forecasting* (TimesFM), ICML 2024.
- Ansari et al., *Chronos: Learning the Language of Time Series*, 2024.
- Goswami et al., *MOMENT: A Family of Open Time-series Foundation Models*, ICML 2024.

**Conhecimento especialista / rótulos / HITL**
- von Rueden et al., *Informed Machine Learning — A Taxonomy and Survey of Integrating Prior Knowledge into Learning Systems*, IEEE TKDE, 2021.
- Karniadakis et al., *Physics-Informed Machine Learning*, Nature Reviews Physics, 2021.
- Ratner et al., *Snorkel: Rapid Training Data Creation with Weak Supervision*, VLDB 2017.
- Settles, *Active Learning Literature Survey*, Univ. Wisconsin-Madison TR 1648, 2009.
- Wu et al., *A Survey of Human-in-the-Loop for Machine Learning*, Future Generation Computer Systems, 2022.
- Monarch, *Human-in-the-Loop Machine Learning*, Manning, 2021.
- Dawid & Skene, *Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm*, Applied Statistics, 1979.
- Frénay & Verleysen, *Classification in the Presence of Label Noise: A Survey*, IEEE TNNLS, 2014.
- van Engelen & Hoos, *A Survey on Semi-Supervised Learning*, Machine Learning, 2020.

**Estatística experimental**
- Lakens, *Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses*, Social Psych. & Personality Science, 2017.
- Demšar, *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR 2006.

**Energia**
- Hyndman & Fan, *Density Forecasting for Long-Term Peak Electricity Demand*, IEEE Trans. Power Systems, 2010 (formulação probabilística de demanda de ponta).
- Wang, Chen, Kang et al., *Review of Smart Meter Data Analytics*, IEEE Trans. Smart Grid, 2019 (paisagem de dados de medição e anomalias em distribuição).
- Buscar adicionalmente na revisão: literatura específica de *load transfer detection* e *feeder reconfiguration detection* — é o vizinho mais próximo do problema e precisa ser mapeado para sustentar o claim de gap.

---

## 13. Sugestões de título

1. *Eficiência de rótulos de especialistas na estimação da demanda máxima em regime normal de alimentadores de distribuição: uma análise por curvas de aprendizado*
2. *Quantos rótulos bastam? Quantificação do conhecimento especialista necessário para estimar a máxima demanda estrutural de alimentadores de distribuição*
3. *Conhecimento especialista como recurso escasso: curvas de aprendizado e viés indutivo na identificação da demanda máxima em regime normal*
4. (EN) *Expert-Label Efficiency in Estimating Normal-Operation Peak Demand of Distribution Feeders: Learning Curves, Inductive Bias, and Statistical Equivalence*
5. (EN) *How Much Expert Knowledge Does a Model Need? Label-Efficient Estimation of Structural Peak Demand in Power Distribution Feeders*

---

## 14. Checklist executivo (as 20 entregas do item 18 da consulta, em uma página)

1. **Problema refinado:** estimação retrospectiva do funcional `max` da componente estrutural da carga, sob contaminação por eventos operacionais longos, com verdade-terreno definida por julgamento de especialista (§2).
2. **RQ principal:** §2. 3. **Hipóteses:** H1–H4 (§3). 4–5. **Objetivos:** §3.
6–7. **Desenho e arquitetura experimental:** teste congelado por alimentador; grade log em n; cadeias aninhadas; R por piloto; matriz método × nível (§5).
8. **Métodos:** baselines robustas (obrigatórias), CPD+classificação (recomendado), AE atual (incumbente), TS2Vec+regressor (fator SSL), ablações de canais; probabilísticos/foundation/weak-supervision como opcionais (§6).
9. **Validação:** cenário B como primário, A como secundário, C/D exploratório, 2026 prospectivo cego (§5.1, §5.7).
10. **Métricas:** HR_5%/HR_10% primária; MdAPE, MAE, RMSE, erro com sinal; recortes por estado/ano/porte; MAPE rebaixado (§7.1).
11. **Testes:** modelos mistos + bootstrap por cluster + Wilcoxon pareado + TOST; Holm para múltiplas comparações (§7.3–7.4).
12. **Anti-leakage:** os 7 controles do §5.4.
13. **Divisão:** §5.1. 14. **Eficiência do conhecimento:** curva E(n), ganho marginal dE/d(log n), `Δn` entre curvas com/sem viés indutivo (§2 RQ3, §6). 15. **Saturação:** TOST com δ inter-anotadores + leitura da curva ajustada (§7.3).
16. **Contribuições:** §9. 17. **Limitações:** §10. 18. **Riscos:** §10 (ancoragem, descarte do `is_sicoi`, COVID, cronograma).
19. **Referências:** §12. 20. **Títulos:** §13.

**Próximos três passos práticos, em ordem:**
1. Escrever a definição operacional do construto e desenhar o estudo de concordância inter-anotadores (é pré-requisito de tudo: define δ, o teto e o critério de sucesso).
2. Auditar os dados: cobertura de rótulos por ano, qualidade do cadastro de extensão por estado, protocolo histórico de rotulagem no James (a predição H0 era exibida?). Documentar a motivação do descarte do `is_sicoi`.
3. Rodar as baselines não supervisionadas (§6.1) contra os rótulos existentes — em uma semana você saberá o tamanho real do gap que o ML precisa fechar, e isso calibra toda a ambição do resto.

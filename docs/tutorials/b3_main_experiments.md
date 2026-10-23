# Experimentos principais B3

**Edite diretamente o `.sh` do experimento.** Cada script contém sua própria
grade, hiperparâmetros, caminhos e recursos do cluster no início do arquivo:

- `scripts/run_temporal_experiment.sh`: seis redes com loss temporal;
- `scripts/run_composite_trading_experiment.sh`: seis redes com loss composta;
- `scripts/run_var_experiment.sh`: VAR com 32 lags.

Não há arquivo JSON externo para editar. Os arquivos Python e
`b3_experiment_env.sh` implementam a execução compartilhada; não é necessário
alterá-los para ajustar um experimento.

| Script | Experimento | Configurações |
| --- | --- | ---: |
| `scripts/run_temporal_experiment.sh` | MSE temporal, seis redes | 324 |
| `scripts/run_composite_trading_experiment.sh` | MSE temporal + loss cross-sectional, seis redes | 33.696 |
| `scripts/run_var_experiment.sh` | VAR com 32 lags, um ajuste por corte | 3 |

Os scripts submetem o experimento completo quando chamados sem argumentos.
Não é necessário submeter separadamente SimpleMLP e SimpleFormer.

Este exercício produz apenas previsões. K participa somente da loss composta;
não há etapa de backtest ou formação de carteiras.

## Dados e cortes temporais

A entrada padrão é o arquivo de **log-retornos já corrigidos** `b3_daily_tfb_new.csv`,
normalmente em `dataset/forecasting/`. Não há fallback para as bases antigas.
O arquivo não acompanha este checkout; deve estar disponível no cluster antes
da preparação ou submissão.

Os três `.sh` declaram `SOURCE_DATA_KIND=log_return`. As redes e o VAR usam
diretamente esses valores, incluindo negativos e zeros, sem aplicar log ou
diferença novamente e sem remover o primeiro pregão. A base exportada por
`montar_painel_tfb` já contém `log_retorno_corrigido` na coluna `data`.

```text
N = número de observações temporais dos log-retornos de entrada
corte = floor(N × proporção)
```

Somente para um CSV que contenha **preços**, declare `SOURCE_DATA_KIND=price`
no `.sh` ou no ambiente. Nesse modo, os preços devem ser estritamente positivos;
calcula-se `r[t] = log(P[t]) - log(P[t-1])` uma vez e `N = N_precos - 1`.
O formato é explícito: não se tenta adivinhá-lo pelo sinal dos valores.

`N` conta pregões da série de log-retornos, não o número de linhas do formato
longo multiplicado pelo número de ativos. Os cortes são `0.70`, `0.80` e
`0.90`. Em cada corte, o primeiro alvo de teste é a observação imediatamente
posterior ao bloco anterior ao teste.

Nas redes, preserva-se `train_ratio_in_tv = 0.875`: os primeiros 87,5% do
bloco anterior ao teste são usados para ajustar os pesos, e os últimos 12,5%
para validação e early stopping. Portanto, os nomes `treino=70`, `treino=80`
e `treino=90` identificam **treino + validação**, conforme o corte do teste.
A validação usa apenas dados anteriores ao teste. O scaler é ajustado no
treino interno.

O VAR usa todo o bloco anterior ao teste, pois não precisa da validação usada
para early stopping das redes. O ajuste verifica se há observações e graus
de liberdade suficientes para estimar os 32 lags; ele não reduz
automaticamente o número de lags.

Os dados preparados mantêm um índice regular para os adaptadores do TFB e
uma correspondência explícita com os pregões originais. Não são acrescentados
pregões de bolsa inexistentes nem preenchidos preços ausentes para completar
o painel. As previsões são convertidas mantendo sua ordem temporal original.
O primeiro retorno corresponde a `step=1` para entrada `log_return` e a
`step=2` para entrada `price`. Esse deslocamento fica nos metadados e é usado
nos Parquets das redes e do VAR. Para entrada `log_return`, o rótulo anterior
ao primeiro retorno não está no CSV e fica vazio no calendário de auditoria.
Os rótulos da fonte são preservados em `source_target`; datas reais ausentes
de uma entrada com índices inteiros não são reconstruídas.

## Grade e treinamento

| Parâmetro | Valores |
| --- | --- |
| Modelos de redes | DUET, Nonstationary_Transformer, FEDformer, TimesNet, simpleFORMER, simpleMLP |
| Lookback (`seq_len`) | 32, 128, 256 |
| Horizonte (`pred_len`) | 1, 5, 10, 15, 20, 25 |
| Janela de trading (`K`) | 1, 5, 10, 15, 20, 25 |
| Máximo de épocas | 100 |
| Patience | 10 |
| Learning rate inicial | 0.0001 para todas as redes |
| Seed | 2026 |
| Stride de previsão | 1 pregão |
| Treino interno / treino + validação | 0.875 |

As arquiteturas preservam os parâmetros usados nos scripts anteriores.
DUET mantém, entre outros, `d_model=32`, `d_ff=32`, uma camada e duas cabeças;
FEDformer e TimesNet mantêm `d_model=256` e `d_ff=512`; Nonstationary Transformer
mantém `d_model=128`, `d_ff=256` e `p_hidden_dims=[32,32]`. SimpleFormer e
SimpleMLP mantêm `d_model=512`, duas camadas e dropout de 0.1; SimpleFormer
mantém oito cabeças e `d_ff=2048`, e SimpleMLP mantém a largura temporal
automática `2 × lookback`. O bloco `MODEL_DEFAULTS_JSON` de cada script de redes contém os valores
completos e pode ser editado no próprio `.sh`.

A agenda `lradj=type1`, a normalização e os demais parâmetros preservados
continuam conforme os experimentos anteriores. A taxa inicial de DUET,
anteriormente 0.01, também passa a 0.0001. O limite de 100 épocas não obriga
o treinamento a completar todas elas: a validação pode encerrá-lo antes.

### Experimento temporal

A loss é a MSE da trajetória prevista. Cada combinação de corte, modelo,
lookback e horizonte é treinada uma vez:

```text
3 cortes × 6 modelos × 3 lookbacks × 6 horizontes = 324 configurações
```

K não participa desse experimento: cada rede é ajustada para prever H passos
com loss puramente temporal. Não há pasta `k_dir`. `lambda=00` registra a equivalência com peso cross-sectional
nulo. O treinamento chama a MSE diretamente, que é o mesmo termo temporal
da loss composta, sem calcular a componente cross-sectional descartada.

### Experimento composto

A loss preserva a implementação `composite_trading_v3`:

```text
L = (1 - lambda) × MSE_temporal + lambda × CS × L_cross
```

A rede prevê a trajetória completa. A componente cross-sectional agrega
log-retornos em blocos não sobrepostos de K passos, transforma-os em retornos
simples e aplica z-score entre ativos dentro de cada bloco. A zona morta
delta é avaliada nos retornos simples realizados antes do z-score.

| Parâmetro | Valores |
| --- | --- |
| Loss cross-sectional | ranknet, hinge, listnet, pairwise_mse |
| Lambda | 0.1, 0.5, 0.8 |
| Delta (ranknet, hinge, pairwise_mse) | 0, 0.025, 0.05, 0.1 |
| Delta (listnet) | Não se aplica; uma execução por combinação |
| CS | 1 |
| RANKNET_ALPHA | 1 |
| LISTNET_TAU | 1 |
| HINGE_MARGIN | 0.1 |

São válidos os pares com `K ≤ H` e `H % K == 0`. Há 16 pares:

| H | K válidos |
| --- | --- |
| 1 | 1 |
| 5 | 1, 5 |
| 10 | 1, 5, 10 |
| 15 | 1, 5, 15 |
| 20 | 1, 5, 10, 20 |
| 25 | 1, 5, 25 |

A divisibilidade é exigida na grade e no cálculo da loss, inclusive em
chamadas diretas à V3. Assim, os blocos de K passos cobrem todos os H passos;
não há cauda excluída da componente cross-sectional. Pares como `(H=20,K=15)`
são rejeitados antes do backpropagation. A auditoria registra `cross_blocks`,
`cross_steps=H` e `cross_tail_steps=0`.

Assim, a grade é:

```text
3 cortes × 6 modelos × 3 lookbacks × 16 pares (H,K)
× 3 lambdas × (3 losses de pares × 4 deltas + 1 ListNet) = 33.696 configurações
```

| Loss cross-sectional | Deltas efetivos | Treinamentos |
| --- | ---: | ---: |
| ranknet | 4 | 10.368 |
| hinge | 4 | 10.368 |
| pairwise_mse | 4 | 10.368 |
| listnet | Não se aplica | 2.592 |
| **Total** | | **33.696** |

**ListNet não utiliza delta na implementação v3.** A grade gera uma única
tarefa por corte, modelo, lookback, par (H,K) e lambda para ListNet. O valor
`delta=0` mantém a hierarquia dos diretórios e o formato da auditoria, com
`delta_effective=False`; não representa um hiperparâmetro ativo. Isso vale
mesmo se `DELTAS` não contiver zero. Alterar `DELTAS` afeta somente ranknet,
hinge e pairwise_mse. Foram eliminadas 7.776 repetições da grade anterior de
41.472 tarefas, preservando as configurações distintas.

### Losses de proximidade e empates (opcionais)

O mesmo `scripts/run_composite_trading_experiment.sh` aceita três variantes novas:
`hinge_proximity`, `ranknet_proximity` e `pairwise_mse_proximity`. As losses
originais `hinge`, `ranknet`, `pairwise_mse` e `listnet` conservam suas fórmulas,
filtros, grade padrão e caminhos. Para executar somente as novas, edite:

```bash
CROSS_LOSSES=(hinge_proximity ranknet_proximity pairwise_mse_proximity)
CROSS_MASK=(uniform 0.025 natural)
# Para outra execução, troque o vetor por:
# CROSS_MASK=(gaussian 0.025 natural)
# CROSS_MASK=(cauchy 0.025 natural)
# CROSS_MASK=(cauchy 0.025 complementary)
```

`CROSS_MASK` recebe nome da máscara, largura `a` e modo (`natural` ou
`complementary`). O vetor antigo de dois elementos continua válido e significa
`natural`. Para preservar a identidade dos experimentos anteriores, o modo
natural é serializado como `loss_cross_mask=["uniform", 0.025]`; o complementar,
como `["uniform", 0.025, "complementary"]`.
Cada execução usa um único vetor, comum às variantes
novas selecionadas. `a` é medido na escala do score realizado antes do z-score:
com `SCORE_KIND=simple_return`, `0.025` representa 2,5 pontos percentuais do
retorno acumulado em K passos. Não é uma margem nos escores padronizados.

Para cada par distinto de ativos `i < j`, seja `gap = R_i - R_j`:

| Máscara | Peso | Domínio |
| --- | --- | --- |
| `uniform` | `1` se `abs(gap) <= a`, senão `0` | `a >= 0`; zero seleciona somente empates exatos |
| `gaussian` | `exp(-gap² / (2a²))` | `a > 0`; peso máximo nos empates e decrescente com a distância |
| `cauchy` | `1 / (1 + (gap/a)²)` | `a > 0`; caudas mais pesadas que a gaussiana |

A tabela mostra os pesos do modo `natural`. O modo `complementary` usa
`1 - peso_natural`, antes da normalização, em qualquer um dos três kernels.
Na uniforme, isso seleciona `abs(gap) > a`; na gaussiana e na Cauchy, o peso
cresce suavemente com a distância. Os complementos valem zero nos empates;
não há piso adicional. Na Cauchy, `a` é a distância de peso natural `0.5`;
na gaussiana, essa distância é aproximadamente `1.177*a`.

Os pesos dependem somente dos retornos realizados. A gaussiana é normalizada
de forma numericamente estável: subtrair o maior log-peso de cada cross-section
preserva a média ponderada e evita que todos os pesos virem zero por underflow.
Uma gaussiana muito estreita concentra o peso nos pares mais próximos, mesmo
quando nenhum par está dentro de `a`; ela não contém um corte rígido.

Com `d = z_i - z_j`, `d_hat = z_hat_i - z_hat_j` e `s = sign(gap)`, os termos são:

| Variante | Sem empate (`gap != 0`) | Empate exato (`gap == 0`) |
| --- | --- | --- |
| `hinge_proximity` | `max(0, HINGE_MARGIN - s*d_hat)` | `abs(d_hat)` |
| `ranknet_proximity` | `softplus(-RANKNET_ALPHA*s*d_hat)` | Entropia cruzada com alvo `0.5` e logit `RANKNET_ALPHA*d_hat` |
| `pairwise_mse_proximity` | `(d_hat - d)²` | `d_hat²` |

No modo natural, os empates recebem peso positivo e os termos acima incentivam
igualdade dos scores. No complementar, seu peso é zero e esse incentivo
cross-sectional fica desligado. Pares próximos, mas não empatados, mantêm a supervisão de ordem
nas variantes Hinge/RankNet. Na Pairwise MSE também há supervisão da distância.
O mínimo da RankNet em um empate tem valor `log(2)` e ocorre em `d_hat=0`.

Em cada amostra/bloco calcula-se `sum(w*loss)/sum(w)`. Uma cross-section sem
pares elegíveis contribui com zero diferenciável; depois se calcula a média
sobre todas as amostras e blocos. Esse cálculo é próprio das variantes novas;
a redução das losses originais permanece intacta. A agregação financeira em K,
o z-score e a composição com a MSE temporal continuam os mesmos.

O incentivo de igualdade é definido no espaço dos escores da loss. O z-score
acopla os ativos e remove escala: com apenas dois ativos, ou todos os alvos
empatados na Pairwise MSE, pode haver gradiente nulo em previsões não constantes
por invariância de escala. As variantes preservam essa normalização existente;
`CROSS_SCORE_NORMALIZATION=none` continua disponível para estudar esse efeito.

`DELTAS` não afeta as variantes novas. Cada uma roda uma vez por corte, modelo,
lookback, (H,K) e lambda, com `delta=0` e `delta_effective=False` na auditoria.
Com somente as três novas losses e os demais vetores padrão, são **7.776**
treinamentos (2.592 por loss), independentemente do tamanho de `DELTAS`.
O script mostra essa contagem antes de submeter:

```bash
bash scripts/run_composite_trading_experiment.sh --print-config
bash scripts/run_composite_trading_experiment.sh --task 0
bash scripts/run_composite_trading_experiment.sh --dry-run
```

Os novos destinos incluem máscara e largura, sem reutilizar saídas originais:

```text
treino=70/experimento=composto/loss=hinge_proximity/lambda=0.1/delta=0/mask=uniform/a=0.025/modelo=DUET/lookback=32/pred_len=5/k_dir=5/part.parquet
```

O modo natural conserva esse caminho; o complementar acrescenta
`mask_mode=complementary/` após `a=.../`. O `RUN_TAG` automático também recebe
o sufixo `_complementary` antes de `_k1_t80_v1`; um `RUN_TAG` informado pelo
usuário é respeitado. Pelo terminal, por exemplo:

```bash
CROSS_MASK="cauchy 0.025 complementary" \
RUN_TAG=composto_cauchy_a0025_complementary_v1 \
bash scripts/run_composite_trading_experiment.sh --dry-run
```

`mask_type`, `mask_a`, `mask_mode` e `mask_effective` ficam na auditoria. Para as losses
originais, os campos de máscara ficam vazios/inativos e `loss_cross_mask` não
é encaminhado ao modelo. Configurações antigas sem esse vetor seguem válidas.

**Use um `OUT_ROOT` novo para cada desenho/máscara/largura/modo e após atualizar o
código.** A proteção existente de `run_config.json` rejeita dados, configuração
ou código diferentes na mesma raiz preparada; os subdiretórios distintos não
desativam essa verificação. Não atualize o checkout enquanto jobs preparados
com a revisão anterior ainda estiverem em execução.

### VAR

O script usa `lags=32` e ajusta o VAR **uma única vez por corte**:

```text
3 cortes = 3 tarefas, array 0-2
```

Os coeficientes estimados ficam fixos durante todo o teste. Em cada origem de
previsão, os últimos 32 valores já observados alimentam uma previsão de 25
passos. Dessa trajetória saem os prefixos dos horizontes 1, 5, 10, 15, 20 e 25,
sem novos ajustes do modelo. As origens finais do teste também são incluídas
quando há alvos observados suficientes para o horizonte correspondente.

Cada tarefa gera seis arquivos Parquet, um por horizonte, e uma única linha
em `completed_tasks.csv`. `forecast_horizons`, `n_windows_by_horizon` e
`output_dirs` identificam as seis saídas; `parquet_sha256` armazena o hash de
cada uma. A retomada só pula o ajuste se os seis arquivos estiverem completos
com hashes corretos.

O contexto de 32 lags aparece como `lookback=32` nas saídas. Não há variação
de loss, lambda ou K no ajuste do VAR. Os rótulos `loss=mse/lambda=00` apenas
mantêm a organização dos diretórios.

## Diretórios e auditoria

A raiz padrão é `/snfs2/igor.viveiros/previsoes/novas_previsoes`.
Exemplos relativos a essa raiz:

```text
treino=70/experimento=temporal/loss=mse/lambda=00/modelo=DUET/lookback=32/pred_len=5/part.parquet
treino=70/experimento=composto/loss=ranknet/lambda=0.1/delta=0.025/modelo=DUET/lookback=32/pred_len=5/k_dir=5/part.parquet
treino=70/experimento=var/loss=mse/lambda=00/modelo=VAR/lookback=32/pred_len=5/part.parquet
```

A hierarquia identifica corte, experimento, loss, lambda, delta quando
pertinente, modelo, lookback, horizonte e K quando pertinente. As pastas de
resultados não incluem data, seed nem job ID. O conteúdo Parquet mantém as
previsões por ativo e as colunas de identificação temporal `arquivo`, `h`,
`origin_step` e `step`, com `step - origin_step == h`.

Cada pasta `treino=.../experimento=...` contém:

- `design_composite_trading.csv`: desenho completo daquele corte e experimento;
- `desing_composite_trading.csv`: cópia com a grafia solicitada, para compatibilidade;
- `completed_tasks.csv`: configurações cuja previsão foi convertida com sucesso.
- `run_config.json`: configuração e identificação dos dados/código usados na preparação.

Os arquivos de auditoria registram os parâmetros e a seed, sem exigir que
essas informações apareçam nos caminhos. Os resultados intermediários do
TFB ficam isolados; a conversão lê os arquivos de previsão preservando a
ordem numérica das janelas, sem depender da ordenação alfabética dos nomes
`sample_1`, `sample_2`, `sample_10`.

Os scripts R anexados ao projeto foram escritos para a organização anterior.
A nova hierarquia **não garante compatibilidade automática** com eles; sua
adaptação deve considerar também os três cortes, lambda, delta e a ausência
de `k_dir` nos experimentos temporal e VAR.

## Execução

No checkout do TFB do cluster, confira primeiro o desenho sem submeter jobs
nem criar ou modificar saídas:

```bash
bash scripts/run_temporal_experiment.sh --dry-run
bash scripts/run_composite_trading_experiment.sh --dry-run
bash scripts/run_var_experiment.sh --dry-run
```

Depois de editar os vetores, `--dry-run` recalcula a quantidade de tarefas.
O resumo também mostra a contagem por loss. Reduzir `CROSS_LOSSES` ou
`LAMBDAS` reduz a grade composta; `DELTAS` altera apenas as losses de pares.
Para visualizar todos os parâmetros efetivos definidos no `.sh`:

```bash
bash scripts/run_composite_trading_experiment.sh --print-config
```

Para inspecionar uma configuração pelo índice global, também sem executá-la:

```bash
bash scripts/run_composite_trading_experiment.sh --task 0
```

Para preparar e validar os dados e os manifestos sem submeter jobs:

```bash
bash scripts/run_temporal_experiment.sh --prepare
bash scripts/run_composite_trading_experiment.sh --prepare
bash scripts/run_var_experiment.sh --prepare
```

Para preparar os dados, validar a configuração, escrever os manifestos e
submeter cada experimento:

```bash
bash scripts/run_temporal_experiment.sh
bash scripts/run_composite_trading_experiment.sh
bash scripts/run_var_experiment.sh
```

Por padrão, a partição é `medusas_dev`, com limite de oito horas por job e
até seis tarefas simultâneas por array. As redes solicitam uma GPU por
tarefa; o VAR usa CPU. Os arrays são limitados a 1.000 tarefas por submissão,
e esse tamanho é reduzido se o Slurm informar um limite menor. No
experimento composto, o script submete 34 arrays: 33 com 1.000 tarefas
(`0-999`) e o último com 696 (`0-695`). Os arrays são encadeados com
`afterany`, mantendo os índices globais de 0 a 33.695 e evitando que o
limite de seis tarefas seja multiplicado pelo número de arrays. Os
experimentos temporal e VAR usam, respectivamente, arrays `0-323` e `0-2`.

`afterany` permite que o próximo array comece quando o anterior terminar,
mesmo que alguma tarefa tenha falhado. Uma tarefa só entra em
`completed_tasks.csv` após concluir a conversão; confira esse arquivo para
identificar configurações pendentes.

Submeter novamente uma configuração já concluída verifica o hash do Parquet
antes de reutilizá-lo. Alterações na configuração ou no código depois da
preparação são rejeitadas para impedir a mistura de desenhos diferentes na
mesma saída. Os arquivos derivados de log-retornos e calendário também são
verificados por hash. Uma nova configuração requer outra `OUT_ROOT`;
alterar a fonte ou `SOURCE_DATA_KIND` exige ainda preparar um novo conjunto de dados
derivados, sem sobrescrever silenciosamente o conjunto já registrado.

Os campos abaixo estão no próprio `.sh`. Caminhos e recursos também podem
ser sobrescritos pelo ambiente:

| Variável | Uso |
| --- | --- |
| `TFB_ROOT` | Caminho do checkout do TFB |
| `VENV_PATH` | Arquivo `bin/activate` do ambiente Python |
| `PYTHON_BIN` | Executável Python |
| `DATASET_PATH` | Caminho de `b3_daily_tfb_new.csv` |
| `SOURCE_DATA_KIND` | `log_return` (padrão, entrada já corrigida) ou `price` (calcular log-retornos) |
| `OUT_ROOT` | Raiz dos resultados Parquet e auditorias |
| `RESULT_ROOT` | Raiz dos resultados temporários do TFB |
| `LOG_ROOT` | Diretório dos logs Slurm |
| `MANTER_CSV` | `TRUE` mantém também as janelas CSV; padrão `FALSE` |
| `ARRAY_CHUNK_SIZE` | Máximo de tarefas por array; padrão 1.000 |
| `MAX_CONCURRENT_JOBS` | Máximo de tarefas simultâneas por array; padrão 6 |
| `PARTITION` | Partição Slurm; padrão `medusas_dev` |
| `TIME_LIMIT` | Limite por job; padrão `08:00:00` |
| `NODELIST` | Nó(s) específicos; vazio permite a escolha pelo Slurm |
| `CPUS_PER_TASK` | CPUs por tarefa; padrão 3 |
| `MEMORY` | Memória solicitada, ex.: `16G`; vazio usa o padrão da partição |

Exemplo com caminhos explícitos:

```bash
TFB_ROOT=/sonic_home/igor.viveiros/src/TFB \
DATASET_PATH=/sonic_home/igor.viveiros/src/TFB/dataset/forecasting/b3_daily_tfb_new.csv \
OUT_ROOT=/snfs2/igor.viveiros/previsoes/novas_previsoes \
bash scripts/run_temporal_experiment.sh
```

A inspeção do desenho pode ser feita fora do cluster. Executar os
experimentos exige a base nova, as dependências do ambiente de treinamento
e acesso ao Slurm; uma execução local de `--dry-run` não valida treinamento
em GPU nem disponibilidade da partição.

## Pós-processamento: IC e carteiras em CPU

Depois de concluir temporal e VAR, execute o pós-processamento **neste
repositório TFB**, na branch `experiment/b3-loss-clean-v1`:

```bash
bash scripts/run_backtests_cpu.sh planejar
bash scripts/run_backtests_cpu.sh piloto
# Após conferir o piloto:
bash scripts/run_backtests_cpu.sh submeter
```

O motor Python está em `scripts/backtest_cpu/`, e as estratégias ficam em
`config/estrategias_backtest.json`. Os jobs usam `gorgonas`; todos os arquivos
gerados ficam no SNFS2, separados de `novas_previsoes`. O repositório local
`analise_resultados` prepara os insumos e lê os Parquets em R para gráficos
e tabelas. Veja [os contratos e as instruções do backtest CPU](b3_backtests_cpu.md).

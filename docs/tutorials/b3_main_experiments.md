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
| `scripts/run_composite_trading_experiment.sh` | MSE temporal + loss cross-sectional, seis redes | 41.472 |
| `scripts/run_var_experiment.sh` | VAR com 32 lags | 18 |

Os scripts submetem o experimento completo quando chamados sem argumentos.
Não é necessário submeter separadamente SimpleMLP e SimpleFormer.

## Dados e cortes temporais

A entrada obrigatória é o arquivo de **preços** `b3_daily_tfb_new.csv`,
normalmente em `dataset/forecasting/`. Não há fallback para as bases antigas.
O arquivo não acompanha este checkout; deve estar disponível no cluster antes
da preparação ou submissão.

As redes e o VAR usam os log-retornos calculados a partir dessa base:

```text
r[t] = log(P[t]) - log(P[t-1])
N = número de observações temporais dos preços - 1
corte = floor(N × proporção)
```

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

K é uma escolha posterior de avaliação das previsões temporais e não cria
treinamentos adicionais. Por isso, não existe uma pasta `k_dir` nesse
experimento. `lambda=00` registra a equivalência com peso cross-sectional
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
| Delta | 0, 0.025, 0.05, 0.1 |
| CS | 1 |
| RANKNET_ALPHA | 1 |
| LISTNET_TAU | 1 |
| HINGE_MARGIN | 0.1 |

Mantém-se a regra anterior: `K ≤ H` e `H % K == 0`. Ela produz 16 pares:

| H | K válidos |
| --- | --- |
| 1 | 1 |
| 5 | 1, 5 |
| 10 | 1, 5, 10 |
| 15 | 1, 5, 15 |
| 20 | 1, 5, 10, 20 |
| 25 | 1, 5, 25 |

Assim, a grade é:

```text
3 cortes × 6 modelos × 3 lookbacks × 16 pares (H,K)
× 3 lambdas × 4 deltas × 4 losses = 41.472 configurações
```

**ListNet não utiliza delta na implementação v3.** Seus quatro valores são
mantidos na grade solicitada e registrados na auditoria, mas correspondem à
mesma função objetivo. Não foi introduzida uma variante de ListNet para
atribuir artificialmente um efeito a esse parâmetro.

### VAR

O script usa `lags=32`, os três cortes e os seis horizontes:

```text
3 cortes × 6 horizontes = 18 configurações
```

O contexto de 32 lags é fixo e aparece como `lookback=32` nas saídas.
Não há lambda ajustável nem treinamento específico por K.

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
Por exemplo, reduzir `CROSS_LOSSES`, `LAMBDAS` ou `DELTAS` reduz a grade composta.
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
experimento composto, o script submete 42 arrays: 41 com 1.000 tarefas
(`0-999`) e o último com 472 (`0-471`). Os arrays são encadeados com
`afterany`, mantendo os índices globais de 0 a 41.471 e evitando que o
limite de seis tarefas seja multiplicado pelo número de arrays. Os
experimentos temporal e VAR usam, respectivamente, arrays `0-323` e `0-17`.

`afterany` permite que o próximo array comece quando o anterior terminar,
mesmo que alguma tarefa tenha falhado. Uma tarefa só entra em
`completed_tasks.csv` após concluir a conversão; confira esse arquivo para
identificar configurações pendentes.

Submeter novamente uma configuração já concluída verifica o hash do Parquet
antes de reutilizá-lo. Alterações na configuração ou no código depois da
preparação são rejeitadas para impedir a mistura de desenhos diferentes na
mesma saída. Os arquivos derivados de log-retornos e calendário também são
verificados por hash. Uma nova configuração requer outra `OUT_ROOT`;
alterar a fonte de preços exige ainda preparar um novo conjunto de dados
derivados, sem sobrescrever silenciosamente o conjunto já registrado.

Os campos abaixo estão no próprio `.sh`. Caminhos e recursos também podem
ser sobrescritos pelo ambiente:

| Variável | Uso |
| --- | --- |
| `TFB_ROOT` | Caminho do checkout do TFB |
| `VENV_PATH` | Arquivo `bin/activate` do ambiente Python |
| `PYTHON_BIN` | Executável Python |
| `DATASET_PATH` | Caminho de `b3_daily_tfb_new.csv` |
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

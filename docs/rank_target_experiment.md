# Experimentos com alvo de posições no ranking

Este experimento usa `scripts/run_rank_target_experiment.sh`. Os launchers
temporais, compostos e ordinais anteriores continuam com seus contratos de
entrada, saída e normalização. Não reutilize o mesmo `PRED_ROOT` entre desenhos.

## Execução exclusivamente nos nós de computação

Na phocus4, o launcher executa apenas o despacho via `srun`, antes de ativar
o ambiente Python. Planos, leitura de CSV, validação de modelos, preparação,
hashes, treinamento e previsão executam nos nós de computação. Sem Slurm,
o comando falha: não existe fallback de processamento local.

`--train` aguarda um nó para preparar os dados e submeter os arrays com
`sbatch`; depois retorna ao terminal, enquanto os treinos seguem nos arrays.
`--trainplan`, `--dry-run`, `--print-config`, `--submission-status` e
`--prepare` também aguardam um nó e mostram a saída no terminal. A sessão
precisa permanecer aberta até o despacho/preparo terminar.

O despacho usa `NS_PARTITION`, `NS_NODELIST`, `NS_TIME_LIMIT`,
`NS_CPUS_PER_TASK` e `NS_MEMORY`, sem reservar GPU para planos ou preparação.
Os arrays continuam usando `NS_GRES`, `NS_MAX_CONCURRENT_JOBS` e
`NS_ARRAY_CHUNK_SIZE`. `--local` significa treinar a grade no processo do
nó alocado pelo `srun`, respeitando `NS_GRES`; nunca na phocus4. Se já houver
uma alocação `salloc`, o launcher abre um passo nessa alocação.

Chamadas diretas à CLI Python fora de um nó Slurm são recusadas. Os launchers
dos experimentos anteriores não são alterados por esta mudança.

Desenhos já preparados verificam os hashes do código. Preserve o checkout
anterior enquanto houver jobs pendentes ou em execução nele; use uma nova
`EXPERIMENT_TAG` e novos destinos para executar este launcher atualizado.

## Dados e sentido da previsão

A fonte continua sendo o painel balanceado de **log-retornos originais**. A
transformação em posições é calculada transversalmente, em cada pregão:
posição 1 = maior retorno, posição N = menor retorno. Empates são resolvidos
pela ordem fixa dos ativos na fonte, registrada na execução.

| Parâmetro | Opções | Padrão |
|---|---|---|
| `LOSS_FAMILY` | `ordinal`, `ranking` | `ordinal` |
| `INPUT_KIND` | `rank`, `log_return` | `rank` |
| `INPUT_NORMALIZATION` | `none`, `standard`, `window`, `both` | `none` |

O alvo é sempre a posição futura, nas duas especificações. Exemplo: retornos
de A/B/C de 0.02/−0.01/0.01 produzem alvo 1/3/2. A rede pode prever
1.4/2.8/1.9: esses valores contínuos alimentam a loss sem arredondamento. Para
ranquear, menor posição prevista é melhor; o sinal com orientação financeira
é o negativo da posição prevista. Em lambda=1 as posições não precisam estar
calibradas nem limitadas ao intervalo [1,N].

## Normalização somente da entrada

`none` preserva a entrada escolhida. `standard` estima média e desvio por ativo
somente no bloco de treino, sem validação ou teste. `window` usa somente a
janela histórica disponível em cada origem. `both` aplica primeiro a
padronização fixa e depois a da janela.

As normalizações e desnormalizações nativas das séries são desativadas no
adaptador deste experimento. A saída em posições nunca recebe a média ou o
desvio dos log-retornos de entrada. LayerNorm e outras normalizações das
representações internas permanecem. Não há z-score transversal na loss.
Arquiteturas sem integração auditada falham explicitamente, em vez de ignorar
a escolha. Esse controle não altera a execução dos launchers anteriores.
No Nonstationary Transformer, as estatísticas históricas continuam disponíveis
aos projetores de atenção tau/delta; elas não padronizam a entrada nem
desnormalizam a saída nesse adaptador. Decomposições internas das arquiteturas
também permanecem.

## Objetivo

`L = (1 − lambda) L_temporal + lambda * NS_CROSS_SCALE * L_cross`.

A componente temporal compara posições contínuas previstas às posições
verdadeiras usando MSE, MAE ou Huber. A componente transversal usa scores
`s = −posição_prevista`, mantendo maior score = melhor ativo.

* `ordinal`: `ordinal_ce`, `bucket_ranknet`, `bucket_hinge`, `bucket_listnet`.
  As classes vêm das posições verdadeiras: `c = floor((posição−1)*T/N)+1`.
  As perdas pairwise comparam apenas classes diferentes. A CE ordinal usa
  limiares aprendidos, registrados no otimizador e no checkpoint.
* `ranking`: compara o vetor completo sem agrupar em classes. As opções são
  `ranknet`, `hinge`, `listnet`, `listmle`, `approxndcg`, `spearman` e
  `pairwise_mse`.

Não se aplica `expm1` às posições. O peso lambda não representa a fração
efetiva do gradiente: as escalas das componentes diferem, razão pela qual
`NS_CROSS_SCALE` permanece editável.

O padrão é `K_LOSS_VALUES="1"`: cada um dos H passos recebe supervisão
transversal diária. H=25 supervisiona 25 vetores por amostra. Para K_loss>1,
os blocos são disjuntos e H deve ser divisível por K_loss. A previsão do bloco
é a média das posições previstas, e a referência é o ranking do log-retorno
verdadeiro acumulado no bloco. A componente temporal continua diária.

## Análise em R

`src/analise_ic_ordinal.R` identifica o tipo de previsão pelos metadados.
O R mantém o contrato da base TFB: CSV longo `date,data,cols`, com `date`
representando o índice inteiro do pregão; o calendário civil é opcional via
`--calendar`.
Para as novas previsões, usa `−mean(posições previstas nos primeiros K dias)`
e calcula o IC de Spearman contra o retorno realizado acumulado nesses K
dias. Para previsões antigas de retorno, mantém a agregação anterior.
O benchmark Momentum continua sendo calculado com os retornos históricos.

Os gráficos mantêm lambda nas colunas, lookbacks nas linhas, H nas cores e
Momentum à direita. Há uma figura por loss. As tabelas de acerto usam lambda
nas linhas e lookback nas colunas, separadas por loss, H e estrato, com K
identificado nos painéis. Vermelho indica menor acerto; azul, maior acerto.

`--strata "1 5" --eval-buckets 10` solicita os decis 1 e 5 para qualquer
família de loss. Os 10 estratos da avaliação são independentes das classes
do treino: um treino ordinal com T=3 pode ser avaliado em decis. Cada tabela
mede a fração dos ativos do estrato verdadeiro recuperados pelo estrato
previsto correspondente.

As janelas de avaliação não alteram as janelas da loss. Os retornos originais
devem ser fornecidos ao R; posições previstas não são retornos financeiros e
não devem alimentar diretamente um backtest que espere retornos previstos.

## Exemplo editável no terminal

Execute dentro do repositório TFB. Este exemplo imprime o plano em um nó
alocado via `srun`; a última linha pode ser trocada por `--train` para preparar
e submeter os arrays a partir desse nó.

```bash
export EXPERIMENT_TAG="rank_targets/rank_ordinal_t90_v1"
export LOSS_FAMILY="ordinal"             # ordinal | ranking
export INPUT_KIND="rank"                 # rank | log_return
export INPUT_NORMALIZATION="none"        # none | standard | window | both
export MODEL="Nonstationary_Transformer" # aceita lista separada por espaços
export TRAIN_VALUES="90"
export HORIZON_VALUES="1 25"
export LOOKBACKS_VALUES="32 256 512"
export SEEDS_VALUES="2027"
export LAMBDAS_VALUES="0 0.1 0.5 0.8 1"
export TEMPORAL_LOSSES_VALUES="mse"
export CROSS_LOSSES_VALUES="ordinal_ce bucket_ranknet bucket_hinge bucket_listnet"
export BUCKET_COUNTS_VALUES="3"           # somente família ordinal

export NS_RANKING_PROTOCOL="blocks"
export K_LOSS_VALUES="1"                  # treino diário em TODOS os H passos
export K_EVAL_VALUES="1 5 10 15 20 25"    # avaliação, independente do treino
export NS_CROSS_SCALE="1"
export NS_ORDINAL_TAU="1"
export NS_BUCKET_TAU="1"
export NS_BUCKET_TARGET_TAU="1"
export NS_HINGE_MARGIN="0.1"
export NS_RANKNET_ALPHA="1"
export NS_LISTNET_TAU="1"
export NS_LISTMLE_TAU="1"
export NS_APPROXNDCG_TAU="1"
export NS_SPEARMAN_TAU="1"

export NS_BATCH_SIZE="16"
export NS_NUM_EPOCHS="100"
export NS_PATIENCE="20"
export NS_LEARNING_RATE="0.0001"
export NS_TRAIN_RATIO_IN_TV="0.875"
# Opcional: overrides de arquitetura por modelo (não campos de treino/loss).
# export NS_MODEL_DEFAULTS_JSON='{"Nonstationary_Transformer":{"d_model":128,"d_ff":256,"e_layers":2,"n_heads":8}}'

export NS_RETURNS_FILE="$PWD/dataset/forecasting/b3_daily_tfb_new.csv"
export PRED_ROOT="/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG"
export NS_RESULT_ROOT="/snfs2/igor.viveiros/result/$EXPERIMENT_TAG"
export NS_LOG_ROOT="$PWD/logs/$EXPERIMENT_TAG"
export NS_PARTITION="gorgonas"
export NS_GRES="gpu:1"
export NS_NODELIST=""
export NS_TIME_LIMIT="01:30:00"
export NS_CPUS_PER_TASK="3"
export NS_MEMORY="32G"
export NS_MAX_CONCURRENT_JOBS="2"
export NS_ARRAY_CHUNK_SIZE="1000"
export NS_PARALLEL_BRANCHES="0"           # limite global; 1 = limite por ramo
export NS_PYTHON_BIN="/sonic_home/igor.viveiros/py310/bin/python"
export NS_VENV_PATH="/sonic_home/igor.viveiros/py310/bin/activate"

bash scripts/run_rank_target_experiment.sh --trainplan
```

Para o segundo desenho, altere os campos e os três destinos antes de submeter:

```bash
export EXPERIMENT_TAG="rank_targets/log_return_ranking_t90_v1"
export LOSS_FAMILY="ranking"
export INPUT_KIND="log_return"
export INPUT_NORMALIZATION="standard"
export CROSS_LOSSES_VALUES="ranknet hinge listnet listmle"
export PRED_ROOT="/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG"
export NS_RESULT_ROOT="/snfs2/igor.viveiros/result/$EXPERIMENT_TAG"
export NS_LOG_ROOT="$PWD/logs/$EXPERIMENT_TAG"
bash scripts/run_rank_target_experiment.sh --trainplan
```

Mudar `EXPERIMENT_TAG` não atualiza automaticamente variáveis de caminhos já
exportadas no shell. O plano imprime os destinos efetivos em uma alocação
Slurm, sem submeter arrays de treinamento.
Use `--dry-run` para inspecionar os comandos Slurm e `--submission-status`
para consultar os IDs registrados. `NS_ARRAY_CHUNK_SIZE` controla tarefas
por array; `NS_MAX_CONCURRENT_JOBS` controla quantas podem executar juntas.

No repositório `analise_resultados`, para gerar os decis 1 e 5:

```bash
Rscript src/analise_ic_ordinal.R \
  --pred-root "$PRED_ROOT" \
  --returns "/caminho/b3_daily_tfb_new.csv" \
  --out "/caminho/figuras/rank_target" \
  --train 90 --seed 2027 \
  --k-eval "1 5 10 15 20 25" \
  --eval-buckets 10 --strata "1 5"
```

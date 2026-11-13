# Experimento ordinal por estratos

O launcher `scripts/run_ordinal_experiment.sh` usa os produtores B3 existentes, o mesmo painel de log-retornos e uma saída temporal compartilhada `[B,H,N]`. Os alvos da parcela ordinatória são estratos transversais de retornos acumulados: `1` é o melhor e `T` o pior. A parcela temporal continua sendo MSE por padrão.

## Objetivos

Para cada data/janela, o ranking decrescente observado tem posição inteira `rank0` começando em zero. O alvo é `c = floor(rank0*T/N)+1`, com `2 <= T <= N`. Empates são desfeitos de forma estável pela ordem dos canais do painel (`metadata.targets` no `run_config.json`); a análise R usa a mesma ordem. Quando N não é divisível por T, os grupos diferem por no máximo um ativo. A divisão usa somente os retornos da janela-alvo daquela amostra.

Os alvos econômicos usados na discretização são preservados em float64 antes da normalização, evitando que a inversão em float32 crie ou desfaça empates reais; entradas e previsões do modelo mantêm o processamento anterior.

As quatro opções de `CROSS_LOSSES_VALUES` são:

- `ordinal_ce`: entropia cruzada binária cumulativa, `z_im = 1[c_i <= m]`, `q_im = sigmoid((s_i-theta_m)/tau)`. A média cobre ativos e os `T-1` limites. Limiares estritamente decrescentes são aprendidos no treino e preservados no checkpoint.
- `bucket_ranknet`: média de `softplus(-(s_i-s_j)/tau)` entre os pares com `c_i<c_j`.
- `bucket_hinge`: média de `max(0, margin-(s_i-s_j))` nos mesmos pares; `margin>=0`.
- `bucket_listnet`: `-sum_i p_i log(q_i)`, com `p=softmax((T-c)/tau_y)` e `q=softmax(s/tau_s)`.

O escore `s` segue a agregação financeira e a normalização transversal configuradas. As perdas são promediadas entre amostras e janelas. A perda total é `(1-lambda)*temporal + lambda*CROSS_SCALE*ordinal`. Lambda zero usa exatamente o modelo temporal nativo e não treina limites ordinais. Lambda um não avalia a perda temporal.

Não há kernel proximity nem zona morta delta nestas quatro opções. RankNet e Hinge desconsideram a ordem interna de um estrato. ListNet atribui probabilidades-alvo iguais aos ativos de um mesmo estrato.

## Defaults e grade sem redundância

Defaults: `Nonstationary_Transformer`, treino 80%, H=`1 25`, LB=`32 256 512`, T=`3`, seed=`2026`, lambdas=`0 .1 .5 .8 1`. São **102 ajustes**: 6 controles temporais, 72 mistos e 24 ordinais puros. Os quatro objetivos compartilham os seis controles.

`NS_RANKING_PROTOCOL=origin_anchored_windows` supervisiona, no mesmo treino, os retornos acumulados desde a origem nas janelas de `RANK_WINDOWS_VALUES` que cabem em H. O default é `1 5 10 15 20 25`; para H=1, somente a janela 1 é usada. `K_LOSS_VALUES=1` é obrigatório neste protocolo.

`NS_RANKING_PROTOCOL=blocks` conserva os blocos disjuntos de K; são gerados somente pares `K<=H` e `H % K == 0`. Nesse caso, `RANK_WINDOWS_VALUES` é inativo. `K_EVAL_VALUES` informa a configuração da análise, sem alterar o treino.

T aceita vários valores em `BUCKET_COUNTS_VALUES`. Modelo, cortes, LB, H, seeds, lambdas, perdas temporais e K_loss também aceitam listas separadas por espaços. Os controles lambda zero não se repetem por T, loss ordinal, temperaturas ou K_loss. Lambda um também não se repete entre perdas temporais, pois essa parcela é inativa; a análise reutiliza o mesmo ordinal puro nas comparações pertinentes.

## Exemplo editável no terminal

Execute no checkout `experiment/b3-loss-clean-v1` do TFB. `--plan` conta a grade e `--dry-run` mostra as configurações efetivas e os comandos Slurm sem submeter. `--train` faz a preparação e a submissão. Nenhum desses comandos altera o código.

```bash
cd /sonic_home/igor.viveiros/src/TFB

# Identidade e grade
export EXPERIMENT_TAG='isolados/ordinal_buckets_t80_v1'
export MODEL='Nonstationary_Transformer'
export TRAIN_VALUES='80'
export LOOKBACKS_VALUES='32 256 512'
export HORIZON_VALUES='1 25'
export SEEDS_VALUES='2026'
export BUCKET_COUNTS_VALUES='3'
export CROSS_LOSSES_VALUES='ordinal_ce bucket_ranknet bucket_hinge bucket_listnet'
export LAMBDAS_VALUES='0 0.1 0.5 0.8 1'
export TEMPORAL_LOSSES_VALUES='mse'

# Construção dos alvos e escala das losses
export NS_RANKING_PROTOCOL='origin_anchored_windows'  # ou blocks
export K_LOSS_VALUES='1'
export RANK_WINDOWS_VALUES='1 5 10 15 20 25'
export K_EVAL_VALUES='1 5 10 15 20 25'
export NS_ORDINAL_TAU=1
export NS_BUCKET_TAU=1
export NS_BUCKET_TARGET_TAU=1
export NS_HINGE_MARGIN=0.1
export NS_CROSS_SCALE=1
export NS_SCORE_KIND='simple_return'                 # ou log_return
export NS_CROSS_SCORE_NORMALIZATION='zscore'         # ou none
export NS_LOSS_INVERSE_NORM=true
export NS_LOSS_TRACK_COMPONENTS=true
export NS_HUBER_DELTA=0.5                           # ativo para temporal huber

# Treinamento
export NS_BATCH_SIZE=16
export NS_NUM_EPOCHS=100
export NS_PATIENCE=10
export NS_LEARNING_RATE=0.0001
export NS_LRADJ=''                                  # vazio: regra nativa; ou constant
export NS_TRAIN_RATIO_IN_TV=0.875
export NS_VALIDATION_MODE='ordered_sample_mean'
export NS_STRIDE=1
export NS_NUM_ROLLINGS=48000

# Arquitetura: JSON por modelo escolhido, sem os campos de treino/loss acima
export NS_MODEL_DEFAULTS_JSON='{"Nonstationary_Transformer":{"d_model":128,"d_ff":256,"e_layers":2,"d_layers":1,"n_heads":8,"dropout":0.1,"factor":3,"norm":true,"p_hidden_dims":[32,32],"p_hidden_layers":2,"lradj":"type1"}}'

# Slurm: vazio em NS_GRES solicita nenhum recurso GPU
export NS_PARTITION='gorgonas'
export NS_GRES='gpu:1'
export NS_NODELIST=''
export NS_TIME_LIMIT='01:30:00'
export NS_CPUS_PER_TASK=3
export NS_MEMORY='32G'
export NS_MAX_CONCURRENT_JOBS=2
export NS_ARRAY_CHUNK_SIZE=1000
export NS_PARALLEL_BRANCHES=0
export NS_AFTER_JOB=''
export NS_RESUME_CHAIN_OLD_JOBS=0

# Caminhos
export TFB_ROOT="$PWD"
export NS_PYTHON_BIN='/sonic_home/igor.viveiros/py310/bin/python'
export NS_VENV_PATH='/sonic_home/igor.viveiros/py310/bin/activate'
export SOURCE_DATA_KIND='log_return'
export NS_RETURNS_FILE="$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv"
export PRED_ROOT="/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG"
export NS_RESULT_ROOT="/snfs2/igor.viveiros/result/$EXPERIMENT_TAG"
export NS_LOG_ROOT="$TFB_ROOT/logs/$EXPERIMENT_TAG"
export NS_MANTER_CSV=FALSE

bash scripts/run_ordinal_experiment.sh --plan
bash scripts/run_ordinal_experiment.sh --dry-run
bash scripts/run_ordinal_experiment.sh --train
```

O nome `NonStationary_Transformer` também é aceito como alias, convertido para o nome nativo `Nonstationary_Transformer`. Outros modelos nativos disponíveis: iTransformer, simpleFORMER, simpleMLP, TimesNet, FEDformer e DUET. Ao escolher vários, inclua cada modelo no JSON de arquitetura; `--print-exports` mostra o JSON default completo.

Para imprimir **todos** os campos públicos com seus valores efetivos:

```bash
bash scripts/run_ordinal_experiment.sh --print-exports
bash scripts/run_ordinal_experiment.sh --print-config
```

## Saídas e reexecução

Cada ramo possui um `run_config.json` congelado em `treino=N/experimento=temporal|composto`. As previsões continuam em `part.parquet`; o manifesto de tarefas inclui T, temperaturas/margem ativas e a política de empates. Os caminhos separam protocolo, seed, perda temporal, variante, T, loss, lambda, modelo, LB e H. O `run_config.json` registra a ordem dos canais, calendários e hashes dos dados/código.

Por padrão, os arrays dos ramos são encadeados, mantendo até `NS_MAX_CONCURRENT_JOBS` tarefas simultâneas para o experimento. `NS_PARALLEL_BRANCHES=1` altera esse limite para **cada ramo**, explicitamente. Marcas de submissão ficam em `NS_LOG_ROOT/ordinal_submissions`; elas registram submissão, não conclusão do treino.

Após uma interrupção com alguns ramos submetidos e outros pendentes, o launcher exige `NS_RESUME_CHAIN_OLD_JOBS=1` para encadear os próximos aos IDs Slurm registrados e manter o limite global. Esses IDs precisam continuar válidos no scheduler. Uma falha após a submissão de parte de um array registra os IDs já criados e interrompe a retomada automática, evitando duplicação silenciosa. Ao modificar a grade/hiperparâmetros de um experimento já submetido, use outra `EXPERIMENT_TAG`.

A análise em R deve ser executada no repositório `analise_resultados`, branch `feat/base-mercado-backtest`, com `src/analise_ic_ordinal.R` e a raiz `PRED_ROOT` deste experimento. Ela usa os escores contínuos para o IC de Spearman e discretiza o ranking previsto com a mesma regra para medir acerto em cada estrato solicitado.

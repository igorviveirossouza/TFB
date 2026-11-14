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

O escore `s` segue a agregação financeira e a normalização transversal configuradas. As perdas são promediadas entre amostras e janelas. A parcela supervisionada é `(1-lambda)*temporal + lambda*CROSS_SCALE*ordinal`. Lambda zero usa o objetivo temporal e não treina limites ordinais. Lambda um não avalia a perda temporal. Regularizações próprias da arquitetura, como o `additional_loss` do DUET, continuam sendo somadas pelo treinador nativo fora dessa combinação, inclusive nos extremos de lambda; seus controles arquiteturais permanecem preservados.

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
export NS_STRIDE=1                                  # restrição atual do produtor B3
export NS_NUM_ROLLINGS=48000

# Arquitetura: opcional; inclua apenas modelos/campos que deseja sobrescrever
export NS_MODEL_DEFAULTS_JSON='{}'
# Exemplo: '{"Nonstationary_Transformer":{"d_model":128,"d_ff":256}}'

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

`MODEL` é resolvido dinamicamente pelos modelos publicados no checkout TFB, sem uma lista fixa de sete arquiteturas. Aceita nomes como `DUET PatchTST DTAF`, caminhos de exportação como `time_series_library.PatchTST` e caminhos completos locais como `ts_benchmark.baselines.time_series_library.models.TiDE.TiDE`. Nomes que apontam para a mesma implementação são deduplicados; se um nome curto for ambíguo, informe o pacote. `NonStationary_Transformer` continua sendo um alias de `Nonstationary_Transformer`.

`NS_MODEL_DEFAULTS_JSON` é opcional e funciona por sobrescrita de campos. Um JSON que contenha somente `Nonstationary_Transformer` não bloqueia `MODEL=DUET`. Os sete defaults anteriores são conservados; modelos adicionais usam seus defaults nativos, com os controles comuns de treinamento deste experimento e `norm=true` por padrão. Portanto, para mudar somente o modelo, basta alterar `MODEL`; não é necessário criar uma entrada no JSON. Exemplo de ajuste específico: `export NS_MODEL_DEFAULTS_JSON='{"DUET":{"num_experts":8},"PatchTST":{"d_model":64}}'`.

A descoberta de um modelo no plano não garante que seu treinador aceite esse objetivo. Antes de preparar dados ou submeter jobs, o produtor carrega o adaptador nativo e verifica sua compatibilidade com a loss ordinal, os alvos econômicos e os checkpoints. Modelos estatísticos ou provedores com um treinador próprio incompatível exigem uma adaptação específica e geram erro explicativo, em vez de treinarem silenciosamente com outra loss. Os modelos Time Series Library recebem automaticamente o `transformer_adapter`; outros pacotes conservam seus wrappers nativos. Restrições próprias de cada arquitetura, como combinações de lookback, tamanho de patch e número de cabeças, continuam válidas.

Para imprimir **todos** os campos públicos com seus valores efetivos:

```bash
bash scripts/run_ordinal_experiment.sh --print-exports
bash scripts/run_ordinal_experiment.sh --print-config
bash scripts/run_ordinal_experiment.sh --submission-status
```

`--submission-status` é somente leitura: mostra os caminhos resolvidos, registros atuais e legados, IDs Slurm e diferenças entre a configuração registrada e a solicitada. Também lista ramos históricos do mesmo destino que não estão na grade atual. Não consulta nem altera o scheduler e não cria arquivos. `--trainplan` não é uma opção: use `--plan` para conferir e `--train` para submeter.

## Saídas e reexecução

Cada ramo possui um `run_config.json` congelado em `treino=N/experimento=temporal|composto`. As previsões continuam em `part.parquet`; o manifesto de tarefas inclui T, temperaturas/margem ativas e a política de empates. Os caminhos separam protocolo, seed, perda temporal, variante, T, loss, lambda, modelo, LB e H. O `run_config.json` registra a ordem dos canais, calendários, resolução do modelo/adaptador em `model_specs` e hashes dos dados/código.

Por padrão, os arrays dos ramos são encadeados, mantendo até `NS_MAX_CONCURRENT_JOBS` tarefas simultâneas para o experimento. `NS_PARALLEL_BRANCHES=1` altera esse limite para **cada ramo**, explicitamente. O plano imprime a raiz das previsões, resultados, logs e estado de submissão.

As marcas atuais ficam em `PRED_ROOT/.ordinal_submissions`. Cada marca registra o destino absoluto, o ramo, a configuração, os IDs Slurm e o estado da submissão. Assim, experimentos com destinos diferentes podem compartilhar uma raiz de logs sem colidir; mudar apenas os logs não permite repetir jobs no mesmo destino. A assinatura da configuração continua sendo comparada para impedir que um ramo já submetido receba outros hiperparâmetros. As marcas registram submissão, não conclusão do treino.

Registros antigos em `NS_LOG_ROOT/ordinal_submissions` são preservados. Só são migrados automaticamente quando trazem identificação explícita do destino correspondente. Um `run_config.json` equivalente serve para diagnóstico, mas não prova onde os jobs antigos foram submetidos: o mesmo desenho pode ter sido preparado em mais de uma pasta. Se o registro antigo não identifica seu destino, a tentativa fica bloqueada e `--submission-status` mostra o arquivo e os IDs envolvidos; nenhum registro é apagado.

Ao iniciar outro experimento, atualize `EXPERIMENT_TAG` **e os três caminhos**, caso já estejam exportados no terminal. Alterar a tag não modifica automaticamente valores de caminhos exportados anteriormente:

```bash
export EXPERIMENT_TAG='isolados/ordinal_buckets_t80_v2'
export PRED_ROOT="/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG"
export NS_RESULT_ROOT="/snfs2/igor.viveiros/result/$EXPERIMENT_TAG"
export NS_LOG_ROOT="$TFB_ROOT/logs/$EXPERIMENT_TAG"
bash scripts/run_ordinal_experiment.sh --submission-status
```

Após uma interrupção com alguns ramos submetidos e outros pendentes, o launcher exige `NS_RESUME_CHAIN_OLD_JOBS=1` para encadear os próximos aos IDs Slurm registrados e manter o limite global. A retomada serial automática exige que os ramos antigos formem um prefixo na ordem de dependências registrada, com histórico serial conhecido; esses IDs precisam continuar válidos no scheduler. Ramos antigos intercalados, histórico paralelo/indefinido, novas dependências incompatíveis ou ramos históricos fora da grade interrompem a retomada e são mostrados no diagnóstico.

Antes de contactar o Slurm, o launcher grava a intenção de submissão (`submitting`). Se houver interrupção ou perda dos IDs retornados, essa marca impede repetição automática. Falhas após submissão de parte dos arrays preservam os IDs conhecidos com estado `partial`. Esses casos exigem conferir os jobs existentes antes de uma nova submissão; o launcher não presume que uma falha de comunicação significa ausência de jobs.

A análise em R deve ser executada no repositório `analise_resultados`, branch `feat/base-mercado-backtest`, com `src/analise_ic_ordinal.R` e a raiz `PRED_ROOT` deste experimento. Ela usa os escores contínuos para o IC de Spearman e discretiza o ranking previsto com a mesma regra para medir acerto em cada estrato solicitado.

A [auditoria da supervisão ordinal](ordinal_supervision_audit.md) explicita quais parcelas usam classes ou retornos, o papel dos escores diferenciáveis e a diferença entre o agrupamento previsto pelo ranking e a classificação pelos limiares da Ordinal CE.

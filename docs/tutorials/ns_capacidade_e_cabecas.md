# Capacidade e separação de tarefas no Non-stationary

São dois estudos independentes. Momentum é benchmark externo, calculado no R
com o mesmo lookback e as mesmas origens dos modelos. Não é injetado na rede.
Os scripts anteriores e suas fórmulas não foram alterados.

## Atenção versus saída por tarefa

As oito cabeças de atenção ficam dentro de cada bloco Transformer. As duas
cabeças por tarefa são ramos de saída: um estima retornos e outro estima
escores de ranking. Ambos os desenhos continuam com oito cabeças de atenção.

## 1. Capacidade: 60 treinos

| Parâmetro | base | expanded |
|---|---:|---:|
| d_model | 128 | 256 |
| d_ff | 256 | 512 |
| e_layers | 2 | 3 |
| d_layers | 1 | 2 |
| n_heads | 8 | 8 |
| p_hidden_dims | [32,32] | [64,64] |
| p_hidden_layers | 2 | 2 |
| dropout | 0.1 | 0.1 |

Para 55 ativos, LB=512 e H=25, são 543.319 parâmetros na base e 3.322.999 na
expandida. O número efetivo de cada ajuste é registrado. Esse desenho testa o
pacote de expansão; não atribui eventual ganho isoladamente à largura ou profundidade.

Defaults comuns: NS, treino 80%, MSE, LB=32/256/512, H=1/25, seed=2026,
lambda=0/.5/1, mistura global ListNet/ListMLE [0.5,0.5,0], original e Proximity,
Cauchy natural A=.025. K_loss=1 conserva a supervisão em blocos do código atual.

Contagem: `2 capacidades × 3 LB × 2 H × (1 temporal + 2 lambdas positivos × 2 versões) = 60`.
Há 12 controles temporais; os outros 48 ajustes usam ranking. Os endpoints
puros não repetem por loss temporal quando mais de uma é escolhida.

```bash
export EXPERIMENT_TAG="isolados/ns_capacidade_t80_v1"
export MODEL="Nonstationary_Transformer"
export TRAIN_VALUES="80"
export LOOKBACKS_VALUES="32 256 512"
export HORIZON_VALUES="1 25"
export K_LOSS_VALUES="1"
export LAMBDAS_VALUES="0 0.5 1"
export TEMPORAL_LOSSES_VALUES="mse"
export SEEDS_VALUES="2026"
export CROSS_LOSSES_VALUES="rankmix rankmix_proximity"
export RANK_MIX_WEIGHTS="0.5 0.5 0"
export RANK_MIX_MODES="global"
export NS_MASK_TYPE="cauchy"
export NS_MASK_MODE="natural"
export NS_MASK_A="0.025"
export MOMENTUM_MODE="off"
export MOMENTUM_INPUT_MODE="off"
export NS_PARTITION="gorgonas_dev"
export NS_GRES=""
export NS_MAX_CONCURRENT_JOBS="3"
export NS_PARALLEL_BRANCHES="0"
export NS_CAPACITY_SPECS='{"base":{"d_model":128,"d_ff":256,"e_layers":2,"d_layers":1,"n_heads":8,"p_hidden_dims":[32,32],"p_hidden_layers":2},"expanded":{"d_model":256,"d_ff":512,"e_layers":3,"d_layers":2,"n_heads":8,"p_hidden_dims":[64,64],"p_hidden_layers":2}}'

bash scripts/run_ns_capacidade_isolado.sh --plan
bash scripts/run_ns_capacidade_isolado.sh --train
```

## 2. Saída compartilhada versus duas saídas: 54 treinos

Mantém apenas a capacidade base e compara `shared` e `separate`. Lambda zero
tem um único controle por LB/H: não é criada uma cabeça de ranking sem supervisão.
Lambda um em `separate` não publica a previsão da cabeça temporal não treinada.

Na variante separada, a representação do decoder anterior à projeção final
alimenta uma pequena MLP de ranking. Para cada K, essa MLP recebe a média das
representações futuras das posições 1..K e emite um escore direto por ativo.
As posições futuras do decoder recebem zeros, nunca os retornos realizados.
Os retornos realizados entram exclusivamente na loss.

**As duas variantes deste segundo estudo são supervisionadas nas mesmas janelas
originadas em t:** K=1/5/10/15/20/25, filtradas por K<=H. O termo de ranking é a
média das losses dessas janelas. A versão compartilhada usa os retornos previstos
acumulados; a separada usa seus escores diretos. Portanto este estudo não deve ser
comparado como uma ablação pura contra os treinos históricos de K_loss=1 em blocos.
Ele contém seu próprio controle compartilhado com os mesmos alvos.

Isso permite curvas para todos os K supervisionados, sem somar escores arbitrários
nem avaliar saídas sem supervisão. Para estudar só K=1, exporte
`RANK_WINDOWS_VALUES="1"`; nesse caso os gráficos de ranking terão só esse K.
`K_LOSS_VALUES` deve permanecer 1 neste estudo, pois a grade efetiva de janelas é
`RANK_WINDOWS_VALUES`. O número de janelas não multiplica os treinos.

Contagem: `3 LB × 2 H × (1 temporal + 2 saídas × 2 lambdas positivos × 2 versões) = 54`.

Após os exports comuns acima, substitua:

```bash
export EXPERIMENT_TAG="isolados/ns_duas_cabecas_t80_v1"
export NS_CAPACITY_SPECS='{"base":{"d_model":128,"d_ff":256,"e_layers":2,"d_layers":1,"n_heads":8,"p_hidden_dims":[32,32],"p_hidden_layers":2}}'
export TASK_HEAD_MODES="shared separate"
export RANK_WINDOWS_VALUES="1 5 10 15 20 25"

bash scripts/run_ns_duas_cabecas_isolado.sh --plan
bash scripts/run_ns_duas_cabecas_isolado.sh --train
```

As duas saídas acrescentam parâmetros. Registre essa diferença ao interpretar
ganhos: o experimento não é uma comparação com número de parâmetros idêntico.

## Edição e execução

`NS_CAPACITY_SPECS` aceita qualquer conjunto nomeado de perfis JSON com os campos
acima e `rank_head_width` (padrão d_model). `d_model` deve ser divisível por
`n_heads`; o comprimento de `p_hidden_dims` deve ser `p_hidden_layers`.
Esses estudos específicos aceitam somente `Nonstationary_Transformer`.

Losses editáveis: rankmix, ranknet, hinge, listnet, listmle, approxndcg e spearman,
sempre incluindo original e `_proximity`. `RANK_MIX_WEIGHTS`, `RANK_MIX_MODES`,
`RANK_MIX_REGIONS` e `RANK_MIX_FRACTION` controlam misturas. Kernel usa
`NS_MASK_TYPE`, `NS_MASK_A`, `NS_MASK_MODE`, `NS_MASK_SHAPE`; aceita as famílias já
implementadas. Esta grade usa uma máscara por tag; não usa `KERNEL_SPECS` nem
`NS_MASK_A_VALUES`. Nenhuma escolha de loss vencedora é automática.

Também editáveis: `NS_BATCH_SIZE` (16), `NS_NUM_EPOCHS` (100), `NS_PATIENCE` (10),
`NS_LEARNING_RATE` (.0001), `NS_LRADJ` (type1), `NS_TRAIN_RATIO_IN_TV` (.875),
temperaturas e escalas usuais das losses, `NS_RETURNS_FILE`, `PRED_ROOT`,
`NS_PYTHON_BIN`, `NS_PARTITION`, `NS_GRES`, `NS_NODELIST`, `NS_TIME_LIMIT` (2h),
`NS_MEMORY`, `NS_CPUS_PER_TASK` (3), `NS_ARRAY_CHUNK_SIZE` (1000),
`NS_MAX_CONCURRENT_JOBS` (3), `NS_PARALLEL_BRANCHES` (0), `NS_AFTER_JOB`.

Um array contém toda a grade padrão. O limite de três tarefas é por array;
arrays encadeados usam afterany. Se os estudos forem submetidos separadamente,
podem executar simultaneamente. Não se garante distribuição em três nós distintos.

As previsões ficam em `$PRED_ROOT/fit=<id>/`; o padrão de PRED_ROOT é
`/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG`. O contrato
`experiment_config.json` declara todos os ajustes. Cada `result.json` comprova
conclusão e vincula hashes, arquitetura, nó/GPU, parâmetros e checkpoint.
`part.parquet` contém retornos; `ranking_scores.parquet` contém escores diretos.
Baixe a pasta do experimento preservando esses arquivos. Checkpoints e código
congelado não são necessários para a análise local.

A submissão congela fonte e código: mudanças posteriores no checkout não afetam
esses novos jobs. Reexecutar --train preserva arrays já submetidos, não retenta
automaticamente treinos que falharam. Um worker incompleto pode ser reexecutado
usando o Python e o runner congelados, com `--worker CAMINHO/experiment_config.json
--index INDICE_GLOBAL`, em um recurso de computação adequado. Resultados completos
são verificados e preservados. Configuração/dados/código diferentes exigem outra tag.

Estes runners geram previsões e ICs via R. Não submetem backtests de carteiras.
Os escores separados ainda não devem ser enviados ao backtest antigo como retornos.

## Análise em R

No repositório **analise_resultados**:

```bash
Rscript src/analise_ic_ns_capacidade.R \
  --pred-root "/caminho/local/ns_capacidade_t80_v1" \
  --returns "/caminho/b3_daily_tfb_new.csv" \
  --out "/caminho/figuras/ns_capacidade" --ic_continuo "TRUE"

Rscript src/analise_ic_ns_duas_cabecas.R \
  --pred-root "/caminho/local/ns_duas_cabecas_t80_v1" \
  --returns "/caminho/b3_daily_tfb_new.csv" \
  --out "/caminho/figuras/ns_duas_cabecas" --ic_continuo "TRUE"
```

Ambos: linhas por LB, colunas por lambda e momentum à direita, K_eval no eixo X,
H=1 como ponto, H=25 como curva. Cores distinguem capacidade ou saída; traço
distingue original/Proximity. Escala vertical comum em toda a execução, origens
iguais por K, sem misturar seeds. `FALSE` espaça as origens em K para modelos e
momentum. Além das figuras, há `ic_resumo.csv`, `ic_diario.parquet` e auditoria.
O IC da saída temporal auxiliar das duas cabeças também fica nas tabelas,
separado do IC da saída principal de ranking.

O refinamento anterior ganhou entrada própria no mesmo repositório:

```bash
Rscript src/analise_ic_ns_rankmix_refinamento.R \
  --pred-root "/caminho/local/rankmix_refinamento_t80_k1_h1.25_v1" \
  --returns "/caminho/b3_daily_tfb_new.csv" \
  --out "/caminho/figuras/rankmix_refinamento" --ic_continuo "TRUE"
```

Reutiliza as fórmulas auditadas de `analise_ic_ns_rankmix.R`, separa figuras por
pesos/modo e distingue A por cor. Preserva todos os controles e a escala comum.

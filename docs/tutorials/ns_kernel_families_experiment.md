# Experimento isolado de familias de kernels

Launcher: `scripts/run_ns_kernels_isolado.sh`. O arquivo original
`run_ns_kloss_isolado.sh` permanece disponivel e nao foi substituido.

## Grade padrao

- Modelo: `Nonstationary_Transformer`; treino 80; LB 256 e 512.
- H: 1 e 25; K_loss: 1; lambdas compostos: 0.5, 0.8 e 1.
- Losses: `ranknet` e `ranknet_proximity`; controle temporal MSE, lambda zero.
- A: **0.25**, em unidades do score financeiro realizado; modo natural.
- 12 kernels: Cauchy; Student nu=10; generalized Gaussian p=1,2,4;
  compact (p,q)=(2,0),(1,1),(2,1),(2,2),(2,3); cosine; intermediate_band p=2.

Sao **160 treinos**: quatro controles temporais, 12 RankNets sem kernel e
144 RankNets proximity (12 por kernel). O controle e a loss sem kernel nao
sao repetidos para cada mascara. Para mudar o modelo ou a loss, use os
mesmos exports abaixo. Uma loss proximity requer sua contraparte sem kernel
na lista, para manter a comparacao explicita. `listnet` pode ser um baseline,
mas nao existe `listnet_proximity`.

A escolha exploratoria p=2 para `intermediate_band` faz o peso decair
quadraticamente perto de zero e exponencialmente nas distancias grandes,
com maximo em d=A. E um ponto inicial moderado, nao um valor empiricamente
otimizado. Compartilhar A entre familias nao iguala suas meias alturas.

## Comando completo

```bash
export EXPERIMENT_TAG="isolados/ns_kernels_a0.25_t80_v1"
export MODEL="Nonstationary_Transformer"
export TRAIN_VALUES="80"
export LOOKBACKS_VALUES="256 512"
export HORIZON_VALUES="1 25"
export K_LOSS_VALUES="1"
export POSITIVE_LAMBDAS="0.5 0.8 1"
export CROSS_LOSSES_VALUES="ranknet ranknet_proximity"
export NS_MASK_A="0.25"
export NS_MASK_MODE="natural"
export KERNEL_SPECS="cauchy;student 10;generalized_gaussian 1;generalized_gaussian 2;generalized_gaussian 4;compact 2 0;compact 1 1;compact 2 1;compact 2 2;compact 2 3;cosine;intermediate_band 2"
export NS_PARTITION="medusas_dev"
export NS_GRES="gpu:1"
export NS_NODELIST=""
export NS_MAX_CONCURRENT_JOBS="2"
export NS_PARALLEL_BRANCHES="0"
export NS_SEED="2026"
export NS_BATCH_SIZE="16"

bash scripts/run_ns_kernels_isolado.sh --plan
bash scripts/run_ns_kernels_isolado.sh --dry-run
bash scripts/run_ns_kernels_isolado.sh --train
```

Para CPU, por exemplo: `export NS_PARTITION="gorgonas"` e
`export NS_GRES=""`. A string vazia e preservada e nao solicita GPU.
Para outro modelo, `export MODEL="DUET"`; varios modelos sao separados por
espacos. Para hinge, `export CROSS_LOSSES_VALUES="hinge hinge_proximity"`.
Use outra EXPERIMENT_TAG ao mudar configuracao cientifica ou dados.

Os parametros de treino, loss e Slurm estao todos no bloco inicial com
`${VAR:-default}`; NS_GRES usa `${NS_GRES-gpu:1}` para preservar vazio.
`NS_NUM_EPOCHS`, `NS_PATIENCE`, `NS_LEARNING_RATE`, `NS_TEMPORAL_LOSS`,
`NS_TRAIN_RATIO_IN_TV`, `NS_STRIDE`, `NS_NUM_ROLLINGS`, `NS_CROSS_SCALE`,
`NS_RANKNET_ALPHA`, `NS_LISTNET_TAU`, `NS_HINGE_MARGIN`, `NS_SCORE_KIND`,
`NS_CROSS_SCORE_NORMALIZATION`, `NS_LOSS_INVERSE_NORM`,
`NS_LOSS_TRACK_COMPONENTS`, `NS_TIME_LIMIT`, `NS_CPUS_PER_TASK`, `NS_MEMORY`,
`NS_ARRAY_CHUNK_SIZE` e `NS_MANTER_CSV` tambem aceitam export.
Os caminhos de Python, ambiente e insumos podem ser alterados por
`TFB_ROOT`, `NS_PYTHON_BIN`, `NS_VENV_PATH`, `NS_RETURNS_FILE`,
`NS_MARKET_DIR` e `NS_STRATEGIES_FILE`.

`KERNEL_SPECS` separa entradas por `;` ou quebra de linha; cada entrada tem
nome e parametros de forma. A e modo vem de NS_MASK_A/NS_MASK_MODE.
Exemplos de aliases: `gaussian`, `uniform`, `laplace`, `triangular`,
`epanechnikov`, `biweight`, `triweight`. As configuracoes antigas de
`cauchy`, `gaussian` e `uniform` continuam validas.

## Concorrencia e retomada

Por padrao os ramos sao encadeados com dependencia Slurm `afterany`, usando
o ultimo array do ramo anterior. Assim apenas um ramo fica ativo por vez;
NS_MAX_CONCURRENT_JOBS limita suas tarefas simultaneas. `afterany` libera o
ramo seguinte mesmo se um treino do anterior falhar; os resultados incompletos
continuam identificados como tal. Arrays internos tambem sao encadeados.

`NS_PARALLEL_BRANCHES=1` libera os ramos independentemente: a grade padrao
pode ocupar ate **14 x NS_MAX_CONCURRENT_JOBS** tarefas. A mensagem do plano
explicita esse teto. `NS_AFTER_JOB=12345` acrescenta uma dependencia inicial.

Cada ramo tem uma marca em `.submissoes_treino/<ramo>` contendo fingerprint
da configuracao, log de submissao e ultimo job. Repetir `--train` preserva
ramos ja submetidos e segue pelos ausentes. Uma falha de submissao pode ter
enviado parte dos arrays, portanto sua marca permanece bloqueada: confira
`submission.log` e Slurm antes de remover somente essa marca e retomar.
Nao apague marcas de jobs ainda ativos para tentar acelerar a execucao.

Para trabalhar em ramos especificos, use os nomes mostrados por `--plan`:

```bash
export NS_ONLY_BRANCHES="student_10 generalized_gaussian_4"
bash scripts/run_ns_kernels_isolado.sh --dry-run
bash scripts/run_ns_kernels_isolado.sh --train
unset NS_ONLY_BRANCHES
```

Se os jobs antigos ja sairam do historico Slurm, selecione apenas os ramos
restantes com NS_ONLY_BRANCHES, evitando dependencias de IDs expirados.
O wrapper nunca cancela ou modifica jobs existentes. Alterar codigo do
checkout pode invalidar a verificacao de proveniencia de workers antigos
que ainda nao iniciaram; use o checkout atualizado para uma grade nova e
mantenha estavel o checkout dos arrays anteriores.

## Previsoes, R e backtest opcional

A raiz `/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG` contem:

- `temporal/`: quatro controles na grade padrao;
- `composto/sem_kernel/`: 12 baselines na grade padrao;
- `composto/<ramo>/`: 12 previsoes por kernel na grade padrao.

Cada ramo preserva run_config, grade auditada, completed_tasks e Parquets.
Baixe a arvore inteira, preservando os metadados, e aponte `--pred-root` do
script R para essa raiz. O R de IC le previsoes diretamente; nao depende
que os backtests financeiros tenham sido executados.

No repositorio `analise_resultados`, branch `feat/base-mercado-backtest`:

```bash
Rscript src/analise_ic_ns_kernels.R \
  --pred-root /dados/ns_kernels_a0.25_t80_v1 \
  --returns /dados/b3_daily_tfb_new.csv \
  --out /dados/analise_ns_kernels
```

Substitua os caminhos pelos locais. A figura tem lookbacks nas linhas e
lambdas nas colunas, mais a coluna momentum. RankNet sem kernel e todas as
proximities ficam no mesmo painel; cores identificam os kernels. H=1 e H=25
tambem compartilham o painel: H=1 e um ponto em K_eval=1 na cor da respectiva
loss/kernel, enquanto H=25 tem curva. O formato do ponto identifica H.
O temporal permanece em lambda zero. Todas as facetas usam a mesma escala Y.
O script requer K_loss fixo e produz figuras separadas por modelo/loss.
As tabelas incluem diferencas contra temporal, momentum e loss sem kernel.

Depois dos treinos:

```bash
bash scripts/run_ns_kernels_isolado.sh --backtest planejar
bash scripts/run_ns_kernels_isolado.sh --backtest submeter
```

Backtests usam `K_EVAL_VALUES` (default 1/5/10/15/20/25, apenas K<=H),
`NS_SLIPPAGE_BPS`, `NS_COST_BPS` (ambos zero), `NS_CPU_WORKERS` (16),
`NS_CPU_N_LOTES` (1) e `NS_CPU_MEMORY` (56G). Cada ramo tem sua execucao
`kernels_<ramo>` em `analise_resultados/backtests/$EXPERIMENT_TAG`.
NS_ONLY_BRANCHES tambem filtra os backtests. Sua particao e tempo seguem os
launchers CPU existentes, nao NS_PARTITION/NS_GRES. As carteiras momentum
nao sao submetidas por este wrapper.

O composto admite novas previsoes em uma nova submissao, preservando tarefas
completas verificadas. Aguarde o job anterior do mesmo ramo terminar antes
de repetir `--backtest submeter`. `consolidar` apenas reune resultados prontos.

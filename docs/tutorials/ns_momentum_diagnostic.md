# Diagnóstico da correção aprendida sobre momentum

O experimento compara três condições com a mesma arquitetura, amostra e loss.
Ele testa se a inicialização do ganho residual e o decaimento da taxa de aprendizado
limitam a aprendizagem. Não pressupõe que uma correção maior melhorará o ranking.

| Condição / subpasta | Ganho inicial | Regra de LR |
| --- | ---: | --- |
| `controle` | 0 | `type1` |
| `ganho_inicial` | 0,1 | `type1` |
| `ganho_lr_constante` | 0,1 | `constant` |

Defaults comuns: `Nonstationary_Transformer`, treino 80%, LB256, H25,
K_loss1, lambda1, `ranknet_proximity`, Cauchy natural A=0,025,
momentum512 e eta=0,01. Cada condição contém **um treino**: total **três**.
Não são repetidos controles temporais nem RankNet sem kernel.

A taxa inicial é 1e-4, batch16, até 100 épocas, patience10 e seed2026.
`type1` mantém a taxa ao final da primeira época e depois a divide por dois a
cada época; `constant` mantém a taxa inicial. O contraste controle → ganho_inicial
altera o score inicial e permite gradiente de ranking para a rede já no primeiro
passo. O contraste ganho_inicial → ganho_lr_constante altera a regra de LR tanto
da rede quanto do ganho. Não isola uma taxa exclusiva do ganho.

## Execução

Use uma nova tag, mesmo se houver outro `EXPERIMENT_TAG` exportado no terminal:

```bash
export EXPERIMENT_TAG="momentum/diagnostico_ns_cauchy_a0.025_t80_v1"
export NS_PARTITION="gorgonas"
export NS_GRES=""

bash scripts/run_ns_momentum_diagnostico.sh --plan
bash scripts/run_ns_momentum_diagnostico.sh --dry-run
bash scripts/run_ns_momentum_diagnostico.sh --train
```

`--plan` apenas conta a grade; `--dry-run` valida as configurações reais sem
submeter jobs. `--train` valida todas as condições antes da primeira submissão.
As condições ficam encadeadas por `afterany`, usando as marcas de submissão já
existentes no launcher de kernels. Repetir o comando não duplica submissões
bem-sucedidas; uma submissão parcial permanece bloqueada para inspeção.

Encadear jobs **não garante o mesmo nó/GPU**. Para comparar no mesmo hardware,
exporte `NS_NODELIST` com um nó disponível e mantenha o ambiente de execução.
`NS_PARTITION`, `NS_GRES`, `NS_NODELIST`, `NS_TIME_LIMIT`, `NS_MEMORY`,
`NS_CPUS_PER_TASK` e `NS_MAX_CONCURRENT_JOBS` mantêm os nomes habituais.
`DIAG_PARALLEL=1` permite executar as condições sem dependência entre elas;
`NS_AFTER_JOB` é uma dependência inicial opcional compartilhada.

## Alterações pelo terminal

A grade usa nomes `DIAG_*` para não herdar acidentalmente múltiplos lookbacks,
horizontes ou kernels de um experimento anterior:

| Parâmetro | Default |
| --- | --- |
| `DIAG_MODEL` | `Nonstationary_Transformer` |
| `DIAG_TRAIN_VALUES` | `80` |
| `DIAG_LOOKBACKS_VALUES` | `256` |
| `DIAG_HORIZON_VALUES` | `25` |
| `DIAG_K_LOSS_VALUES` | `1` |
| `DIAG_LAMBDAS` | `1` |
| `DIAG_CROSS_LOSS` | `ranknet_proximity` |
| `DIAG_KERNEL_SPEC` | `cauchy` |
| `DIAG_MASK_A` / `DIAG_MASK_MODE` | `0.025` / `natural` |
| `DIAG_MOMENTUM_LOOKBACK` / `DIAG_MOMENTUM_ETA` | `512` / `0.01` |
| `DIAG_CONTROL_GAIN_INIT` / `DIAG_GAIN_INIT` | `0` / `0.1` |
| `DIAG_BASE_LRADJ` / `DIAG_ALT_LRADJ` | `type1` / `constant` |
| `DIAG_CONDITIONS` | `controle ganho_inicial ganho_lr_constante` |

`DIAG_KERNEL_SPEC` aceita um kernel, inclusive seus parâmetros de forma, como
`student 10`. `DIAG_CROSS_LOSS` aceita uma loss proximity. O launcher inclui o
nome da loss sem kernel apenas para validar o contrato do pool; o único ramo
selecionado para treino é o kernel solicitado.

Listas maiores em modelo, lookback, H, lambda ou K_loss aumentam a contagem,
sempre mostrada por `--plan`. `DIAG_CONDITIONS` permite executar só uma condição.
Os exports antigos `MODEL`, `LOOKBACKS_VALUES`, `HORIZON_VALUES`,
`CROSS_LOSSES_VALUES`, `KERNEL_SPECS`, `NS_ONLY_BRANCHES`, `NS_LRADJ` e
`MOMENTUM_*` não alteram esta grade. Use os correspondentes `DIAG_*`.

Os hiperparâmetros comuns de treinamento continuam editáveis por `NS_SEED`,
`NS_LEARNING_RATE`, `NS_BATCH_SIZE`, `NS_NUM_EPOCHS` e `NS_PATIENCE`.
Todos recebem os mesmos valores nas três condições. Mudar o desenho exige
uma nova tag; os nomes das três subpastas identificam os contrastes e não
codificam os valores editados.

## Saídas e interpretação

As previsões ficam em:

```text
/snfs2/igor.viveiros/previsoes/<EXPERIMENT_TAG>/<condicao>/composto/<kernel>/...
```

Cada ajuste preserva `part.parquet`, `ranking_scores.parquet`, `run_config.json`
e o estado do momentum. O diagnóstico por época fica habilitado para acompanhar
learning rate, ganho, perdas de treino/validação e normas de gradiente. Use a
validação para acompanhar a aprendizagem e a seleção do checkpoint. O conjunto
de teste continua reservado à avaliação; não escolha hiperparâmetros por seu IC.

Os três desenhos devem ser identificados separadamente nas análises. O R atual
`analise_ic_ns_kernels.R` não deve receber a raiz conjunta deste diagnóstico: ele
confere igualdade de hiperparâmetros e espera os controles da grade completa.
Este exercício não gera esses controles adicionais nem altera automaticamente
os scripts R. O novo Bash prepara treinos e diagnósticos; não executa backtest.

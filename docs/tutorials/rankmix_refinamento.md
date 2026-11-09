# Refinamento da mistura de losses

O launcher `scripts/run_ns_rankmix_refinamento.sh` organiza os testes em duas
etapas. A primeira compara composições; a segunda varia a escala do kernel
somente para a composição escolhida após analisar os resultados. Não há seleção
automática da vencedora nem submissão da segunda etapa pela primeira.

O código das losses já admite todos estes casos. Este experimento reutiliza
`run_ns_rankmix_isolado.sh` e os produtores Python existentes.

## Defaults

| Parâmetro | Valor |
|---|---|
| Modelo | `Nonstationary_Transformer` |
| Treino | 80% |
| Lookbacks | 32, 256, 512 |
| Horizontes H | 1, 25 |
| K_loss | 1 |
| Loss temporal | MSE |
| Lambda temporal/ranking | 0, 0.1, 0.25, 0.5, 1 |
| Mistura | global |
| Seed | 2026 |
| Kernel | Cauchy natural |
| A na comparação das composições | 0.025 |
| A na segunda etapa | 0.01, 0.025, 0.05 |
| Ranking | `rankmix` e `rankmix_proximity` |
| Momentum | benchmark externo; entrada e residual desligados |

A ordem fixa dos pesos internos é **ListNet, ListMLE, ApproxNDCG**:

| Composição | Pesos | Comparação |
|---|---|---|
| Equilibrada | `0.5 0.5 0` | Referência já estudada |
| ListMLE predominante | `0.25 0.75 0` | Reduzir a participação da ListNet |
| ListMLE isolada | `0 1 0` | Referência sem o outro componente |
| ApproxNDCG com ListMLE | `0 0.75 0.25` | Substituir ListNet mantendo 75% para ListMLE |

Pesos iguais a zero desativam componentes. Os pesos somam um e não são
normalizados automaticamente. `RANK_MIX_REGIONS` não regionaliza uma mistura
global; para isso seria preciso selecionar explicitamente o modo regional.

Lambda zero é o temporal puro compartilhado. Lambda um é o ranking puro,
compartilhado entre losses temporais, caso mais de uma seja selecionada.
`K_loss` é o acúmulo usado no treinamento; `K_EVAL_VALUES` são as janelas do
backtest. Para cada H, os K_loss precisam satisfazer `K_loss <= H` e `H % K_loss = 0`.

## Etapa 1 Composições

Na raiz do repositório TFB:

```bash
export EXPERIMENT_TAG="isolados/rankmix_refinamento_t80_k1_h1.25_v1"
export RANK_MIX_STAGE="composicao"
export MODEL="Nonstationary_Transformer"
export LOOKBACKS_VALUES="32 256 512"
export HORIZON_VALUES="1 25"
export K_LOSS_VALUES="1"
export TEMPORAL_LOSSES_VALUES="mse"
export LAMBDAS_VALUES="0 0.1 0.25 0.5 1"
export RANK_MIX_MODES="global"
export RANK_MIX_WEIGHT_SPECS="0.5 0.5 0;0.25 0.75 0;0 1 0;0 0.75 0.25"
export KERNEL_SPECS="cauchy"
export NS_MASK_MODE="natural"
export NS_MASK_A="0.025"
export NS_PARTITION="gorgonas"
export NS_GRES=""
export NS_MAX_CONCURRENT_JOBS="3"
export NS_PARALLEL_BRANCHES="0"

bash scripts/run_ns_rankmix_refinamento.sh --plan
bash scripts/run_ns_rankmix_refinamento.sh --train
```

Os recursos Slurm herdam os defaults do launcher existente quando não são
exportados; os exports acima escolhem explicitamente gorgonas sem solicitar GPU.
`NS_MAX_CONCURRENT_JOBS` limita cada array. Com `NS_PARALLEL_BRANCHES=1`, ramos
podem executar simultaneamente e a soma pode exceder esse limite; não é um limite
global de três nós. `NS_NODELIST` permite selecionar nós, como no launcher original.

Contagem padrão da etapa 1:

- Temporais: `1 modelo × 3 LB × 2 H = 6`.
- Mistos: `4 composições × 3 LB × 2 H × 3 lambdas × 2 versões = 144`.
- Ranking puro: `4 composições × 3 LB × 2 H × 2 versões = 48`.
- **Total: 198 ajustes em 17 ramos**, para uma seed e um corte.

`--plan` calcula a grade sem dados, Python ou Slurm. `--dry-run` valida os
produtores reais, sem submeter jobs. `--train` executa a prévia e submete os ramos.

## Etapa 2 Escala da Cauchy

Após analisar a etapa 1, escolha **um vetor de pesos**. O exemplo abaixo escolhe
ListMLE com ApproxNDCG; essa escolha é ilustrativa, não um resultado antecipado.
Mantenha a mesma `EXPERIMENT_TAG`, os mesmos dados e demais parâmetros da etapa 1.

```bash
export RANK_MIX_STAGE="escala"
export RANK_MIX_SELECTED_WEIGHTS="0 0.75 0.25"
export NS_MASK_A_VALUES="0.01 0.025 0.05"

bash scripts/run_ns_rankmix_refinamento.sh --plan
bash scripts/run_ns_rankmix_refinamento.sh --train
```

O seletor explícito é obrigatório nessa etapa. Ela não usa automaticamente o
primeiro vetor da etapa 1. Para evitar herança de outro terminal, repita os exports
comuns da etapa 1 ao iniciar uma nova sessão e, depois, os três acima.

A grade padrão dessa etapa contém **102 ajustes**:
`6 temporais + 3 LB × 2 H × 4 lambdas positivos × (1 original + 3 escalas)`.
Se a composição selecionada já foi submetida na etapa 1 nesta mesma raiz,
**54 ajustes são reutilizados e apenas 48 são novos**. O conjunto das duas etapas
terá **246 ajustes únicos**. Em uma raiz nova, serão submetidos os 102 ajustes,
incluindo controles e versão original.

## Reutilização e retomadas

Controles temporais não repetem por composição ou A. A mistura sem kernel não
repete por A. A nomenclatura das variantes Proximity inclui a escala também na
pasta externa, mantendo o mesmo caminho quando A=0.025 reaparece na etapa 2.
As previsões mantêm os metadados `treino=`, `experimento=`, `mix=`, `weights=`,
`regions=`, `fraction=`, `mask=` e `a=` do produtor atual.

Ramos já submetidos com a mesma configuração são preservados pelas marcas de
submissão. Isso comprova a submissão, não a conclusão dos treinos: confira Slurm,
logs e Parquets. Submissões parciais/incertas continuam bloqueadas para inspeção.
Alterar parâmetros científicos de um ramo já submetido exige nova tag.

No novo launcher, o encadeamento automático liga apenas jobs submetidos na
invocação atual. Ele não reutiliza IDs antigos de ramos pulados, pois esses IDs
podem ter expirado no Slurm. Para exigir que uma nova rodada aguarde outro job,
exporte explicitamente `NS_AFTER_JOB`. Jobs ativos de uma invocação anterior podem
coexistir com os novos; o limite de concorrência permanece por array.

## Parâmetros editáveis

Todos os parâmetros usuais continuam aceitando `export`: modelo(s), treinos,
lookbacks, H, K_loss, lambdas, losses temporais, seeds, kernels e recursos Slurm.
`RANK_MIX_WEIGHT_SPECS` recebe vetores separados por ponto e vírgula ou quebra de
linha; `NS_MASK_A_VALUES` recebe escalas separadas por espaço. Valores repetidos
equivalentes são rejeitados antes da submissão.

Os componentes ListNet/ListMLE/ApproxNDCG são escolhidos pelos pesos. Mantenha o
par `CROSS_LOSSES_VALUES="rankmix rankmix_proximity"` para comparar as duas versões.
O máximo de editabilidade não elimina a validação da grade e de seus controles.

## Análise e backtest

O script existente **`src/analise_ic_ns_rankmix.R`**, no repositório
**analise_resultados**, lê as duas etapas sem mudanças na fórmula do IC. Use
`--pred-root` apontando para a raiz única do experimento, que contém os ramos
`seed=...`; preserve os `run_config.json` junto às previsões.

Ele separa figuras por composição dos pesos, mantém os valores de A como cores,
mostra H=1 como ponto e H=25 como curva, e preserva temporal e momentum. Com os
lambdas propostos, são **3 linhas × 6 colunas**: cinco lambdas mais momentum.
O rodapé informa os pesos; figuras da mesma execução usam escala vertical comum.

O backtest é uma ação separada, como no launcher anterior:

```bash
bash scripts/run_ns_rankmix_refinamento.sh --backtest planejar
bash scripts/run_ns_rankmix_refinamento.sh --backtest submeter
```

Essa ação percorre os ramos da etapa selecionada. Para processar todas as
composições, selecione `composicao`; para as novas escalas, selecione `escala` e
o mesmo vetor de pesos usado no treinamento. Espere os jobs de backtest anteriores
terminarem antes de retomar uma mesma execução.

## Validação local

```bash
python -m unittest discover -s tests -p 'test_rankmix*launcher.py'
```

Os testes verificam contagens, configurações dos produtores reais, ausência de
duplicatas, metadados por ramo, retomadas entre etapas e roteamento de Slurm e
backtest. Usam substitutos de submissão; não executam treinamentos nem jobs reais.

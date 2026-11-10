# Refinamento fino: três hipóteses e referências pareadas

O experimento testa as três sugestões extraídas do refinamento anterior. Não altera
losses nem arquitetura. Momentum continua externo, sem entrada ou residual.
Resultados anteriores servem para formular hipóteses; não garantem melhora.

Pesos na ordem **ListNet, ListMLE, ApproxNDCG**:

| Exercício | Modo | Pesos | Lambda misto | A |
|---|---|---|---:|---:|
| Menos ranking | global | 0.25, 0.75, 0 | 0.05 | 0.01 |
| Mais proximidade | global | 0.25, 0.75, 0 | 0.1 | 0.005 |
| Pequena contribuição ApproxNDCG | regional | 0, 0.9, 0.1 | 0.5 | 0.01 |
| Referência global | global | 0.25, 0.75, 0 | 0.1 | 0.01 |
| Referência regional | regional | 0, 0.75, 0.25 | 0.5 | 0.01 |

A referência global isola a redução de lambda no primeiro teste e de A no segundo.
A referência regional isola a redução do peso ApproxNDCG de 25% para 10% no terceiro.
No regional, ListMLE recebe o terço inferior e ApproxNDCG o superior dos retornos
realizados no treino; a avaliação de IC continua no universo completo.
Cada especificação inclui mistura original e Proximity, além de ranking puro
(lambda=1). Os controles temporais (lambda=0) e ajustes idênticos são compartilhados.

Defaults: Nonstationary_Transformer, corte80, LB32/256/512, H1/25, K_loss1,
MSE, Cauchy natural, seed2026. São **102 treinos**:

| Bloco | Ajustes por combinação LB/H | Total (3 LB × 2 H) |
|---|---:|---:|
| Temporal MSE | 1 | 6 |
| Mistos (original + Proximity) | 9 | 54 |
| Ranking puro (original + Proximity) | 7 | 42 |
| Total | 17 | 102 |

São 17 ramos/arrays de 6 tarefas nos defaults, sujeitos ao fracionamento do launcher.
Ramos encadeados, até 3 tarefas simultâneas por array; isso não reserva três nós.
O limite é 01:30:00 por tarefa, partição gorgonas_dev, GRES vazio.
Não reutiliza automaticamente arquivos do experimento anterior: use uma tag nova.

## Terminal

```bash
export EXPERIMENT_TAG="isolados/rankmix_refinamento_fino_t80_k1_h1.25_v1"
export MODEL="Nonstationary_Transformer"
export LOOKBACKS_VALUES="32 256 512"
export HORIZON_VALUES="1 25"
export K_LOSS_VALUES="1"
export TEMPORAL_LOSSES_VALUES="mse"
export SEEDS_VALUES="2026"
export KERNEL_SPECS="cauchy"
export NS_MASK_MODE="natural"
export RANK_MIX_REGIONS="top bottom top"
export RANK_MIX_FRACTION="0.3333333333333333"
export RANK_MIX_FOCUS_SPECS='menos_ranking|global|0.25 0.75 0|0.05|0.01;mais_proximidade|global|0.25 0.75 0|0.1|0.005;pequena_ndcg|regional|0 0.9 0.1|0.5|0.01'
export RANK_MIX_INCLUDE_REFERENCES="1"
export RANK_MIX_REFERENCE_SPECS='referencia_global|global|0.25 0.75 0|0.1|0.01;referencia_regional|regional|0 0.75 0.25|0.5|0.01'
export NS_PARTITION="gorgonas_dev"
export NS_GRES=""
export NS_TIME_LIMIT="01:30:00"
export NS_MAX_CONCURRENT_JOBS="3"
export NS_PARALLEL_BRANCHES="0"
export MOMENTUM_MODE="off"
export MOMENTUM_INPUT_MODE="off"

bash scripts/run_ns_rankmix_refinamento_fino.sh --plan
bash scripts/run_ns_rankmix_refinamento_fino.sh --train
```

`RANK_MIX_FOCUS_SPECS` e `RANK_MIX_REFERENCE_SPECS` aceitam linhas no formato
`nome|modo|pesos|lambdas_mistos|As`, separadas por `;` ou quebra de linha.
Dentro de cada linha, listas de lambdas e As são separadas por espaços.
Lambda0/1 são adicionados automaticamente; os lambdas da tupla devem estar em (0,1).
Pesos devem somar 1; não são renormalizados. A é positivo e finito.

Essas tuplas substituem os antigos exports `LAMBDAS_VALUES`, `POSITIVE_LAMBDAS`,
`NS_MASK_A`, `NS_MASK_A_VALUES`, `RANK_MIX_STAGE`, `RANK_MIX_MODES`,
`RANK_MIX_WEIGHTS`, `RANK_MIX_WEIGHT_SPECS` e `RANK_MIX_SELECTED_WEIGHTS`.
As demais opções do launcher rankmix continuam editáveis, inclusive múltiplos
modelos, seeds, horizontes, K_loss e kernels. Cada K_loss precisa dividir H;
todo H deve ter pelo menos um K_loss válido.

`RANK_MIX_INCLUDE_REFERENCES=0` deixa somente os três testes e controles, **72 treinos**.
Reativar referências na mesma tag acrescenta 30, preservando as marcas dos 72.
Marcas significam submissão, não conclusão. Mudanças científicas num ramo existente
são rejeitadas; use outra tag. É possível filtrar nomes de ramos pelo `NS_ONLY_BRANCHES`.

`--train` gera previsões; backtest exige `--backtest submeter` quando desejado.
Logs: `$TFB_ROOT/logs/$EXPERIMENT_TAG/seed=.../temporal_loss=.../...`.
Predições: `/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG`, salvo override de `PRED_ROOT`.
As pastas compostas incluem identidade da mistura, kernel/A e lambda; cada ramo
tem um `run_config.json`, sem células científicas duplicadas entre arquivos.

## Análise local em R

No repositório `analise_resultados`, use `src/analise_ic_ns_rankmix_refinamento_fino.R`
ou o analisador existente `src/analise_ic_ns_rankmix_refinamento.R`; ambos usam
as mesmas fórmulas e descobrem a grade pelos JSONs. Baixe a raiz completa da tag
com os `run_config.json` e parquets. Figuras mantêm escalas comuns, LB nas linhas,
lambda nas colunas, temporal à esquerda e momentum à direita; H1 é ponto e H25 curva.
As duas escalas A ficam juntas na figura global; cada mistura regional tem figura própria.
Ausência de A=.005 em lambda=.05 é intencional: a grade é focada, não cartesiana.

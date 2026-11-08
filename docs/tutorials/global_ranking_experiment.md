# Experimento de ranking global

Launcher: `scripts/run_ns_global_ranking_isolado.sh`. Nao substitui os scripts
dos experimentos anteriores. Momentum e benchmark externo, calculado na analise;
nao entra como feature nem e somado a previsao.

## Grade inicial

| Parametro | Default |
|---|---|
| Modelo inicial | `Nonstationary_Transformer` (MODEL aceita um ou mais modelos) |
| Lookbacks | `32 256 512` |
| Corte de treino | 80% |
| Horizonte H / K_loss | 25 / 25 |
| Seed | 2026 |
| Loss temporal | MSE e Huber (delta 0.5 na escala de treino) |
| Ranking | RankNet, Spearman, ListNet, ListMLE e ApproxNDCG, cada um com e sem Proximity |
| Lambda | 0, 0.5, 0.8, 1 |
| Kernel | Cauchy natural, A=0.025 |
| Temperaturas | Spearman, ListNet, ListMLE e ApproxNDCG: tau=1 |

O default H=K_loss=25 concentra a primeira rodada no ranking do retorno
acumulado. H, K_loss e a seed sao escolhas editaveis desta rodada.

Com um modelo e tres lookbacks, ha tres combinacoes de arquitetura/janela:

| Lambda | Conta | Treinos |
|---|---|---:|
| 0 | 3 combinacoes x 2 losses temporais | 6 |
| 0.5 | 3 x 2 temporais x 10 rankings | 60 |
| 0.8 | 3 x 2 temporais x 10 rankings | 60 |
| 1 | 3 x 10 rankings, sem repetir por temporal | 30 |
| Total | Por seed e par H/K desta grade | **156** |

Os controles nao se repetem por K_loss, ranking ou kernel. O ranking puro e
compartilhado pelas comparacoes com MSE e Huber. Momentum exige zero treinos.
Adicionar kernels repete apenas as variantes Proximity; adicionar seeds repete
a grade correspondente. Com NS e simpleFORMER, o total e 312. O comando
`--plan` recalcula a contagem efetiva, inclusive quando exports anteriores
sobrescrevem os defaults.

## Objetivos

`L = (1-lambda) L_temporal + lambda * cross_scale * L_ranking`.
Os extremos usam somente o termo ativo. A previsao continua sendo o caminho
completo de log-retornos; o score de ranking padrao e o retorno simples
acumulado em cada bloco K_loss. O componente temporal usa a escala do treino.
Portanto lambda e um coeficiente da funcao objetivo, nao uma percentagem
garantida do gradiente de cada tarefa.

Para Spearman, cada lista contem os ativos de uma mesma origem/bloco. Os ranks
previstos descendentes sao `a_i = 1 + sum_{j != i} sigmoid((s_j-s_i)/tau)`;
os ranks realizados `b_i` sao ranks exatos com media nos empates. A loss sem
kernel e `1-corr(a,b)`. E uma aproximacao diferenciavel, nao a correlacao de
Spearman exata usada na avaliacao. O operador e construido com sigmoides
pareadas, sem dependencia adicional de torchsort ou operador isotonico.

Spearman Proximity usa
`1 - sum(w_ij*(a_i-a_j)*(b_i-b_j)) /
sqrt(sum(w_ij*(a_i-a_j)^2)*sum(w_ij*(b_i-b_j)^2))`, para i<j.
Com pesos iguais, recupera a loss sem kernel. Os pesos Cauchy sao proporcionais
a `1/(1+(abs(R_i-R_j)/A)^2)`, sobre os retornos realizados brutos, antes do
z-score. A=0.025 corresponde a diferenca de 2.5 pontos percentuais de retorno
simples acumulado. Outros kernels e modos seguem o registro ja existente.

RankNet Proximity continua com sua formulacao anterior. As novas losses nao
alteram a arquitetura nem garantem superar momentum: essa e a hipotese testada.
Listas sem variacao nos ranks realizados ou sem pares ativos contribuem zero;
denominadores sao protegidos para manter gradientes finitos. Essa regra se
refere ao Spearman; ListNet preserva sua distribuicao uniforme nos empates.

### ListNet, ListMLE e ApproxNDCG Proximity

As versoes Proximity abaixo sao adaptacoes explicitas deste projeto. As
referencias no fim descrevem os objetivos originais, nao estas mascaras.
`R_i` e o retorno realizado bruto do bloco; `s_i` e o score previsto depois
da normalizacao cross-sectional configurada. A mascara depende somente do
alvo dentro da loss, nunca entra na rede nem na geracao das previsoes.

Nessas tres novas familias, `w_ij = kernel(abs(R_i-R_j)/A)` usa o valor
**original** do kernel em [0,1]. Nao se reescala pelo maximo da lista: a
escala absoluta faz parte dos denominadores e da interpolacao dos ranks.
As formulas antigas de RankNet/Spearman Proximity permanecem inalteradas.
Todos os aliases e familias de kernels existentes sao aceitos.

**ListNet Proximity.** Para cada ativo ancora i, definir v_ii=1 e v_ij=w_ij
fora da diagonal. A distribuicao alvo e `q(j|i) = softmax_j(log(v_ij) +
z(R_j)/tau)` e a prevista e `p(j|i) = softmax_j(log(v_ij) + s_j/tau)`.
A loss e a media da entropia cruzada `-sum_j q(j|i) log p(j|i)` sobre
ancoras com pelo menos um vizinho de peso positivo. Sem vizinhos, a loss
e zero. Com w=1, todas as ancoras reproduzem exatamente o ListNet anterior,
inclusive sua resposta a empates. `z` segue a normalizacao configurada.

**ListMLE Proximity.** Para cada ancora com pelo menos um ativo de retorno
estritamente inferior, calcular `log(1 + sum_{j:R_j<R_i} w_ij *
exp((s_j-s_i)/tau))`. Usar a media dessas etapas informativas. O conjunto
de ancoras no denominador da media depende dos labels, nao da mascara.
Sem mascara, tomar w=1. Para labels distintos, isso e o ListMLE usual
dividido por N-1. Nos empates, os ativos empatados nao sao ordenados
arbitrariamente: todos sao comparados com os estritamente inferiores.
Essa e uma extensao explicita para empates; listas totalmente empatadas
contribuem zero. O calculo usa logsumexp para estabilidade.

**ApproxNDCG.** Obter midranks realizados descendentes r_i*, relevancia
ordinal `q_i=(N-r_i*)/(N-1)` e ganhos nao negativos `g_i=2^q_i-1`.
Assim, retornos negativos tambem participam do ranking. O IDCG usa os
ganhos ordenados e posicoes inteiras 1..N. A loss sem mascara e
`1 - sum_i g_i/log2(1+soft_rank_i) / IDCG`.

Na versao Proximity, substituir o soft-rank por
`r_i* + sum_{j!=i} w_ij * (sigmoid((s_j-s_i)/tau) - c_ji*)`, onde c_ji*
vale 1 quando R_j>R_i, 0 quando menor e 0.5 nos empates. As comparacoes
de baixo peso ficam proximas da ordem realizada dentro da loss; isso
concentra o gradiente nas comparacoes selecionadas. A mascara e simetrica
e cada comparacao interpolada continua em [0,1], com probabilidades
reciprocas. Os ranks permanecem no politopo dos ranks de permutacoes;
o DCG nao supera o IDCG convencional. Nao se usa a soma ingênua de
comparacoes mascaradas como se fosse um rank completo.

Com pesos 1, a expressao recupera os soft-ranks convencionais. Listas
totalmente empatadas ou sem pares ativos de retornos distintos contribuem zero. Em
empates parciais, midranks suavizados podem deixar uma diferenca positiva
em relacao ao IDCG de posicoes inteiras; isso tambem ocorre no surrogate
sem mascara. NDCG favorece a parte superior da lista por seu desconto
de posicao e nao e equivalente ao IC de Spearman global.

Os valores destas losses sao objetivos de treino, nao metricas de IC.
Em especial, ApproxNDCG Proximity usa labels na interpolacao: seu valor
nao deve ser apresentado como o NDCG da previsao. Avaliar sempre os
scores exportados pelo IC usual, sem mascara, nas mesmas datas e ativos
dos benchmarks. Comparar todos os hibridos com o temporal MSE e com
momentum; comparar tambem os hibridos Huber com seu controle Huber.

A mascara pode alterar a intensidade dos gradientes alem de quais erros
recebem mais peso. `NS_CROSS_SCALE` e os componentes registrados permitem
auditar essa escala; lambda nao representa uma fracao fixa do gradiente.

## Terminal

```bash
export EXPERIMENT_TAG="isolados/global_ranking_a0.025_t80_h25_k25_v1"
export MODEL="Nonstationary_Transformer"
export TRAIN_VALUES="80"
export LOOKBACKS_VALUES="32 256 512"
export HORIZON_VALUES="25"
export K_LOSS_VALUES="25"
export LAMBDAS_VALUES="0 0.5 0.8 1"
export TEMPORAL_LOSSES_VALUES="mse huber"
export CROSS_LOSSES_VALUES="ranknet ranknet_proximity spearman spearman_proximity listnet listnet_proximity listmle listmle_proximity approxndcg approxndcg_proximity"
export SEEDS_VALUES="2026"
export KERNEL_SPECS="cauchy"
export NS_MASK_A="0.025"
export NS_MASK_MODE="natural"
export NS_HUBER_DELTA="0.5"
export NS_SPEARMAN_TAU="1"
export NS_LISTNET_TAU="1"
export NS_LISTMLE_TAU="1"
export NS_APPROXNDCG_TAU="1"
export MOMENTUM_MODE="off"
export MOMENTUM_INPUT_MODE="off"

bash scripts/run_ns_global_ranking_isolado.sh --plan
```

Depois de revisar o plano, `--dry-run` valida os launchers e `--train` submete
os jobs. Os demais parametros `NS_*` continuam editaveis, incluindo particao,
GRES, no, concorrencia, epocas, patience, batch e taxa de aprendizado.
`NS_GRES=""` preserva a solicitacao sem GPU. `NS_PARALLEL_BRANCHES=0` encadeia
os ramos; com 1 os ramos ficam independentes, e o limite de concorrencia e
por array, nao global. Nenhum destes comandos cancela jobs existentes.

Os defaults de treino sao batch16, ate100 epocas, patience10 e LR0.0001.
A validacao usa `ordered_sample_mean`, com ordem fixa e ponderacao por amostra;
o bloco de validacao e 12.5% do bloco anterior ao teste. A regra de LR nativa
do modelo e preservada, a menos que `NS_LRADJ` seja informado.

## Saidas e comparacao

Preserve a arvore inteira com os run_config e manifestos:
`<PRED_ROOT>/seed=<S>/temporal_loss=<mse|huber|none>/...`.
`none` identifica os rankings puros compartilhados. Cada ramo tem seu
controle de submissao, identidade de configuracao e registros de conclusao.
`--backtest planejar|submeter|consolidar` usa os backtests existentes com as
novas identidades de loss. O backtest e separado do treino.

A analise R desta grade deve separar MSE/Huber e seeds, reutilizar o ramo
`none` em ambas as comparacoes e calcular o momentum para os mesmos ativos,
datas, lookback e K_eval. O leitor R anterior nao deve ser presumido compativel
com esta raiz combinada; a adaptacao de visualizacao e uma etapa separada.
O default de K_eval e 1/5/10/15/20/25 e nao acrescenta treinamentos.

## Referencias dos objetivos originais

- Cao et al. (2007), [Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf).
- Xia et al. (2008), [Listwise Approach to Learning to Rank: Theory and Algorithm](https://doi.org/10.1145/1390156.1390306).
- Qin, Liu e Li (2010), [A General Approximation Framework for Direct Optimization of Information Retrieval Measures](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2008-164.pdf).

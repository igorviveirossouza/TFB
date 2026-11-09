# Experimento de misturas de losses de ranking

O launcher `scripts/run_ns_rankmix_isolado.sh` compara misturas globais e
direcionadas às caudas do ranking, sempre com e sem Proximity. A arquitetura
continua prevendo a trajetória completa de retornos. Não há injeção de momentum
na entrada nem soma residual de momentum na saída.

## Loss e parâmetros

Para cada bloco de `K_loss` pregões dentro de `H`, agregam-se os retornos previstos
e realizados com a regra financeira já usada pelo TFB. Somente pares com
`K_loss <= H` e `H % K_loss == 0` são treinados.

\[
L=(1-\lambda)L_{temporal}
 +\lambda\,c\,[w_1L_{ListNet}+w_2L_{ListMLE}+w_3L_{ApproxNDCG}].
\]

- `LAMBDAS_VALUES`: peso externo da tarefa de ranking, de zero a um.
- `NS_CROSS_SCALE`: fator externo `c`, default 1.
- `RANK_MIX_WEIGHTS`: três pesos internos na ordem **ListNet, ListMLE,
  ApproxNDCG**. Devem ser finitos, não negativos e somar um, com tolerância
  absoluta de `1e-8`. Não são renormalizados silenciosamente. Zero desliga um
  componente.
- `RANK_MIX_MODES`: um ou ambos os modos `global regional`.
- `RANK_MIX_REGIONS`: três regiões `top`/`bottom`, na mesma ordem dos pesos.
  São usadas somente no modo regional.
- `RANK_MIX_FRACTION`: fração de ativos que recebe ênfase em cada cauda,
  no intervalo `(0, 0.5]`, default `0.3333333333333333`.

No modo **global**, somam-se as losses originais já implementadas. No modo
**regional**, ListNet concentra seu alvo na cauda selecionada, ListMLE aprende
etapas parciais da ordenação dessa cauda e ApproxNDCG concentra seus ganhos nessa
região. Para o fundo, a orientação dos escores e dos alvos é invertida. As
comparações mantêm o universo de ativos, e não apenas os integrantes conhecidos
do terço. A Proximity aplica os pesos dos gaps financeiros conforme a definição
de cada componente. Kernels de suporte finito podem zerar comparações.

A seleção regional usa exclusivamente alvos realizados do treino, destacados do
autograd. Empates na fronteira recebem inclusão fracionária para evitar que a
ordem das colunas decida o aprendizado. Na previsão, não são necessários rótulos
de terço nem retornos futuros. A implementação matemática e os critérios de
empate estão em `ts_benchmark/baselines/hybrid_ranking_losses.py`.

Pesos internos iguais significam **coeficientes iguais**, não gradientes de igual
magnitude: as escalas naturais de entropia cruzada, log-verossimilhança e NDCG
continuam diferentes. As temperaturas e `NS_CROSS_SCALE` permanecem editáveis.
As adaptações regionais são novas hipóteses experimentais; o desempenho da
ListMLE original no fundo não garante o desempenho de sua versão invertida.

## Grade default: 132 treinos

| Parâmetro | Default |
|---|---|
| Modelo | `Nonstationary_Transformer` |
| Treino | 80% |
| Lookbacks | 32, 256, 512 |
| Horizontes H | 1, 25 |
| K da loss | 1 |
| Tarefas temporais | MSE, Huber |
| Lambda externo | 0, 0.5, 0.8, 1 |
| Pesos internos | ListNet = 0.5; ListMLE = 0.5; ApproxNDCG = 0 |
| Modos de mistura | Global e regional |
| Regiões | ListNet topo; ListMLE fundo; ApproxNDCG topo |
| Versões | `rankmix` e `rankmix_proximity` |
| Kernel | Cauchy natural, A = 0.025 |
| Seed | 2026 |
| Batch, épocas, patience, LR | 16, 100, 10, 0.0001 |
| Validação | `ordered_sample_mean`; 12.5% do bloco pré-teste |
| Slurm | `medusas_dev`, `gpu:1`, 3 CPUs/tarefa, 01:30:00 |

| Bloco | Conta | Treinos |
|---|---|---:|
| Temporal puro, lambda 0 | 3 LB × 2 H × 2 tarefas temporais | 12 |
| Misturas, lambda 0.5/0.8 | 3 LB × 2 H/K × 2 tarefas temporais × 2 lambdas × 2 modos × 2 versões | 96 |
| Ranking puro, lambda 1 | 3 LB × 2 H/K × 2 modos × 2 versões | 24 |
| **Total** | | **132** |

O momentum é benchmark externo e exige **zero treinos**. Os 12 controles
temporais são compartilhados por todas as misturas e kernels. Lambda 1 é
treinado uma única vez por configuração de ranking, sem duplicação entre MSE e
Huber. A assinatura da mistura e todos os parâmetros são preservados nas saídas.

Com os demais defaults: um único modo (`regional` ou `global`) produz **72**
treinos; apenas MSE produz **78**; dois modelos produzem **264**. Cada kernel
adicional acrescenta somente as versões Proximity, sem repetir a versão comum.

## Uso no terminal

```bash
export EXPERIMENT_TAG="isolados/rankmix/ns_listnet_listmle_a0.025_t80_v1"
export MODEL="Nonstationary_Transformer"
export LOOKBACKS_VALUES="32 256 512"
export HORIZON_VALUES="1 25"
export K_LOSS_VALUES="1"
export LAMBDAS_VALUES="0 0.5 0.8 1"
export TEMPORAL_LOSSES_VALUES="mse huber"
export CROSS_LOSSES_VALUES="rankmix rankmix_proximity"
export RANK_MIX_WEIGHTS="0.5 0.5 0"
export RANK_MIX_MODES="global regional"
export RANK_MIX_REGIONS="top bottom top"
export RANK_MIX_FRACTION="0.3333333333333333"
export KERNEL_SPECS="cauchy"
export NS_MASK_A="0.025"
export NS_MASK_MODE="natural"
export NS_PARTITION="gorgonas"
export NS_GRES=""

bash scripts/run_ns_rankmix_isolado.sh --plan
bash scripts/run_ns_rankmix_isolado.sh --dry-run
bash scripts/run_ns_rankmix_isolado.sh --train
```

`--plan` não necessita Python, dados nem Slurm. `--dry-run` usa os launchers reais
para validar a configuração. `--train` primeiro faz todas as prévias e então
submete os ramos. `NS_GRES=""` preserva o valor vazio e não solicita GPU.

Para substituir ListNet por ApproxNDCG, altere somente:

```bash
export RANK_MIX_WEIGHTS="0 0.5 0.5"
```

Para habilitar três componentes, por exemplo:

```bash
export RANK_MIX_WEIGHTS="0.25 0.50 0.25"
```

O número de treinos não aumenta ao habilitar um terceiro componente: continua
sendo uma mistura por treinamento. Altere `EXPERIMENT_TAG` ao comparar desenhos
científicos diferentes. Os controles temporais não recebem parâmetros da mistura.

## Saídas, retomada e backtests

As previsões ficam sob `PRED_ROOT`, cujo default é
`/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG`. Há ramos por seed, tarefa
temporal, modo/assinatura da mistura e kernel. A assinatura de 12 caracteres é
derivada dos pesos, regiões e fração, e o `run_config.json` guarda seus valores
completos. Os nomes internos das previsões também registram o contrato da mistura.

Por default, os ramos são encadeados por dependências `afterany`.
`NS_PARALLEL_BRANCHES=1` permite todos os ramos em paralelo, sem dependências
geradas entre eles. `NS_MAX_CONCURRENT_JOBS` limita **cada array**, não o total
do experimento e nem o número de nós. `NS_AFTER_JOB` acrescenta uma dependência
externa opcional. A distribuição em nós continua sob controle do Slurm.

Uma retomada preserva ramos já submetidos. Submissões parciais ou incertas são
interrompidas para inspeção; não são repetidas automaticamente.

```bash
export K_EVAL_VALUES="1 5 10 15 20 25"
bash scripts/run_ns_rankmix_isolado.sh --backtest planejar
bash scripts/run_ns_rankmix_isolado.sh --backtest submeter
```

`--train` não submete backtests automaticamente. O backtest recebe a raiz correta
de cada ramo. Após novos Parquets, repita `submeter` quando não houver outro
backtest ativo naquele ramo. As análises devem comparar identificação dos
integrantes, IC global e IC dentro de cada terço com o temporal puro e com o
momentum do respectivo lookback, nas mesmas datas e janelas K.

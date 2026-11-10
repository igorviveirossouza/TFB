# Duas cabeças com restauração equivalente

Este experimento compara uma projeção compartilhada com duas projeções lineares
independentes, uma temporal e outra de ranking. Conserva a capacidade base,
as losses e as janelas supervisionadas do estudo anterior de cabeças.
O estudo antigo de escores livres continua disponível, sem mudança de fórmula.
O `run_ns_rankmix_refinamento.sh` não foi alterado.

## O que foi corrigido na comparação

Para o estado do decoder `h[t,j]` e cada tarefa `a`, a projeção por ativo é
`z[a,t,j] = W[a] h[t,j] + b[a]`. As projeções temporal e de ranking começam
com valores idênticos, copiados da projeção nativa, mas têm parâmetros distintos.
Não se consome RNG adicional ao criar a segunda projeção.

Nas duas saídas aplica-se a mesma reconstrução:

1. `u[a,t,j,i] = sigma[t,i] * z[a,t,j,i] + mu[t,i]`, usando somente o
   lookback observado. Esta é a restauração nativa do Non-stationary.
2. `r[a,t,j,i] = scale_fit[i] * u[a,t,j,i] + mean_fit[i]`, usando o scaler
   estimado exclusivamente no bloco de treino interno.
3. O escore de ranking para K é `sum(r[ranking,t,1:K,i])`, ou `expm1` dessa
   soma quando `NS_SCORE_KIND=simple_return` (default). O mesmo procedimento
   define o escore da saída compartilhada.

As estatísticas não são aplicadas diretamente aos estados de dimensão d_model.
A nova cabeça não usa a MLP nem a média de estados do experimento anterior.
Suas saídas diárias são projeções em unidades de retorno; quando supervisionadas
somente por ranking, não se deve interpretar sua magnitude como retorno calibrado.

A loss é `(1-lambda)*temporal + lambda*cross_scale*ranking`. No modo separado,
o termo temporal atualiza a projeção temporal; o ranking atualiza a sua projeção;
ambos atualizam o tronco comum. Com lambda=1 a projeção temporal não é treinada
e não é publicada. Com lambda=0 há somente o controle temporal compartilhado.

As janelas `RANK_WINDOWS_VALUES` são supervisionadas conjuntamente, com peso
igual, filtradas por `K<=H`. Isso preserva o desenho do primeiro estudo de
cabeças, e não equivale ao K_loss=1 por posição do estudo de capacidade.
Em H=1 só K=1 é usado. K_loss deve continuar 1; o parâmetro efetivo deste estudo
é `RANK_WINDOWS_VALUES` e sua quantidade não multiplica os treinos.

Em lambda=1, com a mesma inicialização e otimização, as duas variantes devem
coincidir até tolerância numérica: ambas aprendem apenas o ranking pela mesma
construção de escore. Esse extremo funciona também como controle da implementação.
O efeito de separar tarefas é avaliado principalmente em `0<lambda<1`.
Há uma projeção extra, logo a contagem de parâmetros total não é idêntica.
Na base com 55 ativos, LB=512 e H=25: 543.319 parâmetros compartilhados
versus 550.414 com duas projeções (acréscimo de 7.095).

## Defaults e contagem

Nonstationary_Transformer; treino 80; capacidade base (d_model=128, d_ff=256,
encoder=2, decoder=1, atenção=8, projetores=[32,32]); LB=32/256/512; H=1/25;
MSE; lambda=0/.5/1; mistura global ListNet/ListMLE com pesos [.5,.5,0];
original e Proximity; Cauchy natural A=.025; seed=2026; momentum externo.

`3 LB × 2 H × [1 temporal + 2 modos × 2 lambdas positivos × 2 losses] = 54 treinos`.
São 6 controles temporais e 48 treinos de ranking/mistos. Não se reutilizam
automaticamente resultados antigos: ambos os modos são executados sob o novo
contrato, com dados e código congelados.

## Comando completo e editável

Execute a partir do repositório TFB. Os parênteses preservam os exports do
experimento de refinamento no terminal externo. A tag deve ser nova; um
`PRED_ROOT` explicitamente exportado prevalece sobre a tag, por isso ele é
removido apenas dentro deste subshell.

```bash
(
  unset PRED_ROOT NS_AFTER_JOB NS_NODELIST NS_MASK_SHAPE
  export EXPERIMENT_TAG="isolados/ns_duas_cabecas_corrigidas_t80_v1"
  export MODEL="Nonstationary_Transformer"
  export TRAIN_VALUES="80"
  export LOOKBACKS_VALUES="32 256 512"
  export HORIZON_VALUES="1 25"
  export K_LOSS_VALUES="1"
  export RANK_WINDOWS_VALUES="1 5 10 15 20 25"
  export LAMBDAS_VALUES="0 0.5 1"
  export TEMPORAL_LOSSES_VALUES="mse"
  export SEEDS_VALUES="2026"
  export CROSS_LOSSES_VALUES="rankmix rankmix_proximity"
  export RANK_MIX_WEIGHTS="0.5 0.5 0"
  export RANK_MIX_MODES="global"
  export RANK_MIX_REGIONS="top bottom top"
  export RANK_MIX_FRACTION="0.3333333333333333"
  export TASK_HEAD_MODES="shared separate"
  export NS_CAPACITY_SPECS='{"base":{"d_model":128,"d_ff":256,"e_layers":2,"d_layers":1,"n_heads":8,"p_hidden_dims":[32,32],"p_hidden_layers":2}}'
  export NS_MASK_TYPE="cauchy"
  export NS_MASK_MODE="natural"
  export NS_MASK_A="0.025"
  export NS_SCORE_KIND="simple_return"
  export NS_CROSS_SCORE_NORMALIZATION="zscore"
  export NS_CROSS_SCALE="1"
  export NS_LISTNET_TAU="1"
  export NS_LISTMLE_TAU="1"
  export MOMENTUM_MODE="off"
  export MOMENTUM_INPUT_MODE="off"
  export NS_BATCH_SIZE="16"
  export NS_NUM_EPOCHS="100"
  export NS_PATIENCE="10"
  export NS_LEARNING_RATE="0.0001"
  export NS_LRADJ="type1"
  export NS_TRAIN_RATIO_IN_TV="0.875"
  export NS_PARTITION="gorgonas_dev"
  export NS_GRES=""
  export NS_MAX_CONCURRENT_JOBS="3"
  export NS_ARRAY_CHUNK_SIZE="1000"
  export NS_PARALLEL_BRANCHES="0"

  bash scripts/run_ns_duas_cabecas_corrigidas.sh --plan &&
  bash scripts/run_ns_duas_cabecas_corrigidas.sh --train
)
```

Os demais parâmetros do [launcher arquitetural](ns_capacidade_e_cabecas.md)
continuam editáveis. Este estudo suporta apenas Nonstationary_Transformer.
`rank_head_width` é rejeitado porque a nova projeção é linear d_model→ativos.
Não confundir `RANK_MIX_WEIGHTS` (uma mistura) com `RANK_MIX_WEIGHT_SPECS`
(grade de misturas usada pelo launcher de refinamento).

Não há submissão automática de backtest. Logs: `$PRED_ROOT/logs/%A_%a.out`
e `.err`; a raiz padrão é `/snfs2/igor.viveiros/previsoes/$EXPERIMENT_TAG`.
Jobs anteriores usam seu próprio contrato. O refinamento não depende dos
arquivos modificados neste experimento.

## Análise local em R

No repositório **analise_resultados**, use:

```bash
Rscript src/analise_ic_ns_duas_cabecas_corrigidas.R \
  --pred-root "/caminho/local/ns_duas_cabecas_corrigidas_t80_v1" \
  --returns "/caminho/b3_daily_tfb_new.csv" \
  --out "/caminho/figuras/ns_duas_cabecas_corrigidas" \
  --ic_continuo "TRUE"
```

Baixe a raiz preservando `experiment_config.json`, `fit=*/result.json`,
`part.parquet` quando existir e `ranking_scores.parquet`. Este último já contém
um escore acumulado por K; o R o lê diretamente, sem acumular novamente.
O contrato identifica `heads_restored`/`restored_daily_projection_v1`, separado
do estudo antigo. O analisador verifica essa semântica, hashes e cobertura.
As curvas mantêm as escalas comuns, origens iguais, momentum à direita,
temporal à esquerda, linhas por LB e colunas por lambda. H=1 é um ponto.
IC temporal auxiliar fica nas tabelas, separado da saída principal de ranking.
`--ic_continuo FALSE` espaça a avaliação em K tanto para modelos quanto momentum.

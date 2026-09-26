# simpleMLP e simpleFORMER

Implementados na branch `experiment/b3-loss-clean-v1`. Ambos retornam
`[batch, H, N]` e utilizam o `transformer_adapter` existente. O adaptador mantém
StandardScaler do treino, Adam, early stopping, divisão treino/validação,
forecasting rolling e gravação de previsões. Nenhuma loss foi reimplementada.

## Arquiteturas

| Componente | simpleMLP | simpleFORMER |
|---|---|---|
| Entrada | Projeção linear conjunta N -> d | Projeção linear conjunta N -> d |
| Normalização interna | Por janela/ativo, reversível, sem parâmetros afins | Igual |
| Bloco principal | LayerNorm(d), MLP temporal LB -> 2*LB -> LB, residual | Atenção temporal multihead + FFN, ambas pre-norm e residuais |
| Ativação | GELU | GELU |
| Blocos | 2, parâmetros próprios | 2, parâmetros próprios |
| Posição temporal | Implícita nas camadas lineares temporais | Codificação senoidal fixa |
| Saída | LayerNorm(d), projeções LB -> H e d -> N | Igual |

Defaults propostos: d=512, dropout=0.1, epsilon=1e-5. No simpleFORMER:
8 cabeças de dimensão 64 e FFN 512 -> 2048 -> 512. No simpleMLP,
`temporal_hidden_dim=0` determina automaticamente 2*LB. O MLP temporal é
compartilhado entre as características latentes. A atenção tem um token por
instante, projeções compartilhadas entre os instantes e cabeças distintas.
Nenhuma das redes usa valores futuros do decoder ou features de calendário.

A normalização interna é desfeita **antes** da loss; a saída volta ao espaço
padronizado pelo StandardScaler externo. A loss v3 utiliza as estatísticas do
StandardScaler para recuperar a escala financeira, agregar os blocos K e aplicar
o delta nos retornos realizados antes do z-score cross-sectional.

As duas redes não têm o mesmo número de parâmetros. Os defaults são uma
referência arquitetural, não a mediana das configurações B3 atuais.

## Experimentos

O novo script é `scripts/run_simple_composite_trading_experiment.sh`. Ele mantém
a grade, os datasets, a convenção das datas e o conversor do script original.
Defaults copiados do experimento existente: log_retornos; LB=32,104,246;
H,K=1,5,10,15,20,24; K<=H e H divisível por K; seed=2026;
tv_ratio=0.8; train_ratio_in_tv=0.875; stride=1. São 90 tarefas.
Também foram preservados os defaults atuais de loss (hinge, lambda=0.999,
margem=0.1). Edite-os no bloco inicial ou sobrescreva pelo ambiente.

Os hiperparâmetros de treinamento são comuns às duas redes e seguem os defaults
do TransformerAdapter: batch=32, lr=1e-4, 10 épocas, patience=3, lradj=type1.

Inspecionar a grade/configuração sem submeter jobs ou criar saídas:

```bash
bash scripts/run_simple_composite_trading_experiment.sh --dry-run
```

Baseline temporal pelo mesmo launcher composto, com lambda zero:

```bash
CROSS_LOSS=mse CROSS_LAMBDA=0 \
  bash scripts/run_simple_composite_trading_experiment.sh
```

MSE convencional, pela factory de MSE do TFB:

```bash
LOSS_NAME=mse CROSS_LOSS=mse CROSS_LAMBDA=0 \
  bash scripts/run_simple_composite_trading_experiment.sh
```

Loss composta com zona morta:

```bash
CROSS_LOSS=pairwise_mse CROSS_LAMBDA=0.5 CROSS_DELTA=0.005 \
  bash scripts/run_simple_composite_trading_experiment.sh
```

Hinge, ranknet, bpr, listnet e mse cross-sectional também usam a implementação
v3 existente, inclusive CROSS_SCALE e CROSS_SCORE_NORMALIZATION. A combinação
permanece `(1-lambda)*temporal + lambda*cross_scale*cross`.

Arquitetura e treino podem ser editados no SH ou pelo ambiente: D_MODEL,
E_LAYERS, D_FF, N_HEADS, TEMPORAL_HIDDEN_DIM, DROPOUT, NORM, USE_NORM,
NORM_EPS, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, PATIENCE e LRADJ.
LOOKBACKS_OVERRIDE, HORIZONS_OVERRIDE e TRADE_WINDOWS_OVERRIDE aceitam listas
separadas por espaço. MODELS e DATASETS permanecem arrays editáveis no início.

Saídas usam nomes `simpleMLP` e `simpleFORMER`, com o mesmo particionamento
Parquet do pipeline existente. O manifesto e completed_tasks.csv registram
também arquitetura e treinamento. O diretório default distingue loss, lambda
e delta. Ao alterar outros parâmetros, escolha um OUT_ROOT próprio para evitar
sobrescrever uma execução anterior. RESULT_ROOT/EXPERIMENT_ID isolam os
resultados temporários. MANTER_CSV=TRUE preserva também os CSVs.

## Verificação local

```bash
python scripts/verify_simple_models.py
```

Verifica formatos, ausência de uso do decoder futuro, inversão da normalização,
gradientes de todas as losses cross-sectionais (retornos simples/log e preços),
lambda=0, delta sem pares, comunicação entre canais e fit/forecast dos dois
modelos pelo adaptador real com MSE e loss composta. Não submete jobs Slurm.

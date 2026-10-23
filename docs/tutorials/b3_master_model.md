# MASTER multihorizonte no experimento B3

O modelo está registrado como `master.MASTER` no TFB e usa o mesmo desenho,
runner, divisão temporal, exportação Parquet e convenções de backtest dos demais
modelos B3. O adaptador é próprio porque uma amostra contém **todos os ativos na
mesma origem**, além das características auditadas de cada ativo e do mercado.
As instruções da exportação R e do contrato dos dados estão em
[b3_master_data.md](b3_master_data.md).

## Arquitetura e extensões

Referência: [artigo MASTER](https://arxiv.org/abs/2312.15235) e
[implementação dos autores](https://github.com/SJTU-DMTai/MASTER).
O núcleo preserva a seleção de features condicionada ao estado de mercado,
atenção temporal por ativo, atenção entre ativos e agregação temporal. A
atribuição e a licença MIT estão em `ts_benchmark/baselines/master/LICENSE.upstream`.

Esta é uma adaptação para o painel B3, com diferenças explícitas em relação ao
experimento original:

- Entrada `[B,L,N,F]`, com `B` origens independentes, `L` observações anteriores,
  `N` ativos e `F` features selecionadas. A atenção nunca mistura origens.
- O gate usa somente o estado de mercado na última observação do lookback.
  Nenhuma feature da data-alvo entra no modelo.
- As máscaras de valores observados entram por projeções próprias. Um valor
  ausente vira zero **depois** da padronização; a máscara distingue ausência de
  um valor observado igual à média. As flags de auditoria ficam fora da rede.
- A cabeça produz diretamente `[B,H,N]`: em um treinamento com `H=5`, o modelo
  prevê conjuntamente cinco log-retornos futuros por ativo. Não são cinco
  chamadas autorregressivas. A grade pode comparar cabeças com diferentes `H`.
- Sem mercado (`master_use_market=False` na API, `USE_MARKET=false` no shell),
  os pesos do gate são um. A dimensão de mercado é zero.

## Escalas e losses

As features dos ativos são padronizadas por ativo/feature; as de mercado, por
feature. Os parâmetros são estimados apenas no prefixo de treino. O adaptador
também estima uma padronização separada dos alvos por ativo, como os modelos
comparadores com `norm=True`. A loss temporal é calculada nessa escala. Para a
componente financeira, a fábrica v3 desfaz a padronização antes de acumular os
log-retornos e calcular os scores. `forecast` e os Parquets contêm log-retornos
na escala econômica original.

Não há uma cópia das fórmulas de loss dentro do MASTER:
`ts_benchmark/baselines/composite_trading_loss_v3.py` é a fábrica utilizada.

| Família | Configuração |
|---|---|
| Temporal pura | `TEMPORAL_LOSS=mse`, `mae` ou `huber`; modo `temporal` |
| Composta | Mesma componente temporal + `mse`, `bpr`, `ranknet`, `hinge`, `listnet` ou `pairwise_mse`; modo `composto` |
| Composta Proximity | Mesma componente temporal + `hinge_proximity`, `ranknet_proximity` ou `pairwise_mse_proximity`; modo `composto` |

Proximity aceita `uniform`, `gaussian` e `cauchy`, nos modos `natural` e
`complementary`, por meio de `CROSS_MASK="tipo a modo"`. Cada grade usa um vetor
de máscara; altere `RUN_TAG` ao comparar outro vetor ou outro desenho.
As perdas de pares tradicionais usam `DELTAS`; MSE, ListNet e Proximity não
repetem execuções por deltas inativos.

`HORIZONS` define os horizontes e `TRADE_WINDOWS` define os tamanhos `K` dos
blocos financeiros. A grade inclui apenas pares com `H % K == 0`; todos os
passos de previsão ficam cobertos. Por exemplo, `H=5,K=1` aplica o objetivo
cross a cinco blocos diários e `H=5,K=5` aplica-o ao retorno acumulado dos cinco
passos. Em ambos os casos a componente temporal usa toda a trajetória.

## Primeiro teste no cluster

A máquina `phocus4` é apenas acesso. Execute nela somente o launcher:

```bash
cd /sonic_home/igor.viveiros/src/TFB
bash scripts/run_master_experiment.sh smoke
```

O shell submete um job à `gorgonas_dev`; o Python só inicia no nó de computação.
O interpretador padrão é `/sonic_home/igor.viveiros/py310/bin/python`.
Não é necessário ativar o ambiente nem executar Python na máquina de acesso.

O teste usa a base completa de 55 ativos e 3664 pregões, com lookback 32. São
seis ajustes: temporal MSE, composta RankNet e composta RankNet Proximity, cada
uma com `H=1` e `H=5`. Cada ajuste usa uma época, no máximo dois batches de
treino e dois de validação, e prevê duas origens. O objetivo é verificar a
integração. **Esse resultado não é um experimento para comparação de modelos.**

Logs: `logs/master/master_smoke_v1/prepare_JOBID.out` e `.err`.
Relatório: `result/master/smoke_JOBID.json`, com `status=PASSED`, seis casos,
formas das previsões, losses de validação, proveniência e `limited_run=true`.
Em caso de falha, o job termina com erro e o relatório registra `status=FAILED`.

## Piloto com treinamento completo

Após o smoke, execute na máquina de acesso:

```bash
bash scripts/run_master_pilot.sh
```

O wrapper fixa exatamente **três treinamentos**: MSE temporal, MSE + RankNet
e MSE + RankNet Proximity. Usa treino externo 80%, validação 12,5% do pré-teste,
`L=32`, `H=5`, `K=1`, lambda 0,5 nas compostas, delta zero e máscara gaussiana
`a=0.025`, modo natural. Todos os blocos de features estão ativos. Mantém
`d_model=64`, batch 32, learning rate 0,0001, scheduler `type1` e seed 2026.
São até 100 épocas, com early stopping de patience 10, sem limitar batches.

Com 3.664 pregões, cada modelo usa 2.528 janelas de treino, 363 de validação
e prevê as 729 origens possíveis do teste. São 3.645 linhas por `part.parquet`
(729 origens × 5 passos), com os 55 ativos como colunas. Cada tarefa grava
também `master_training.json` e `master_checkpoint.pt`; o audit deve registrar
`limited_run=false`. As losses de validação selecionam a melhor época dentro
de cada objetivo; a comparação entre objetivos deve usar métricas comuns.

O comando retorna **dois IDs de preparação**, para temporal e composto. A
preparação acontece em `gorgonas_dev` e submete os workers automaticamente:
um temporal e um array composto com dois elementos, limitado a um por vez.
Os workers usam CPU em `gorgonas`, 32 GB e até 8 horas, evitando o limite de
1h30 da partição dev. Nenhum Python roda na phocus4. Os nomes dos jobs de
treino e seus IDs aparecem nos logs das duas preparações.

- Logs: `logs/master/master_piloto_l32_h5_k1_t80_v1/{temporal,composto}/`.
- Previsões e auditorias: `/snfs2/igor.viveiros/previsoes/master_piloto_l32_h5_k1_t80_v1/`.
- Cada diretório `treino=80/experimento=...` contém `completed_tasks.csv`:
  deve haver uma tarefa temporal e duas compostas concluídas.

`--dry-run` submete somente as duas preparações para descrever as grades;
`--prepare` valida os dados e grava os desenhos sem criar workers. O wrapper
usa nomes de parâmetros próprios para evitar herdar caminhos de outra grade:
`MASTER_PILOT_TAG`, `MASTER_PILOT_OUT_ROOT`, `MASTER_PILOT_RESULT_ROOT` e
`MASTER_PILOT_LOG_ROOT`. Os recursos podem ser alterados por
`MASTER_PILOT_PARTITION`, `MASTER_PILOT_TIME_LIMIT` e `MASTER_PILOT_MEMORY`.
Para modificar o desenho científico, use `run_master_experiment.sh` com outro
`RUN_TAG`; o wrapper do piloto mantém seus três casos fixos.

Para verificar consumo e estado após obter os IDs dos workers:

```bash
sacct -j ID_TEMPORAL,ID_COMPOSTO --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

## Grades e ablations

Os comandos abaixo também submetem a preparação ao Slurm. `--dry-run` grava
a descrição da grade no log do job de preparação, sem criar workers.

```bash
# Inspecionar a grade temporal multihorizonte.
RUN_TAG=master_temporal_mse_v1 \
  bash scripts/run_master_experiment.sh temporal --dry-run

# Composta e Proximity no mesmo desenho, todas as losses cross disponíveis.
RUN_TAG=master_composto_gaussian_v1 \
  bash scripts/run_master_experiment.sh composto --dry-run

# Ablation: somente features de retornos, sem o estado de mercado.
RUN_TAG=master_retornos_sem_mercado_v1 ASSET_BLOCKS="retornos" USE_MARKET=false \
  bash scripts/run_master_experiment.sh temporal --dry-run

# Exemplo de outra loss temporal e um único horizonte multivariado.
RUN_TAG=master_temporal_mae_h5_v1 TEMPORAL_LOSS=mae LOOKBACKS="32" HORIZONS="5" \
  bash scripts/run_master_experiment.sh temporal --dry-run
```

Após revisar a grade e os recursos, remova `--dry-run` para submeter os workers.
O job de preparação usa sempre `gorgonas_dev`, 2 CPUs e 8 GB. Os recursos dos
workers são configurados por `PARTITION`, `GRES`, `TIME_LIMIT`, `MEMORY`,
`CPUS_PER_TASK` e `MAX_CONCURRENT_JOBS`; os defaults são CPU em `gorgonas_dev`,
1h30, 32 GB, 2 CPUs e duas tarefas simultâneas. Ajuste tempo/partição para a
grade completa. A atenção temporal cresce quadraticamente com o lookback;
`L=256`, 55 ativos e batch 32 exigem mais memória que o smoke. `BATCH_SIZE`
também é configurável e fica registrado na auditoria.

Os blocos disponíveis em `ASSET_BLOCKS` são `retornos`, `ohlcv` e `atividade`.
O estado de mercado é controlado separadamente por `USE_MARKET`. Use um
`RUN_TAG` distinto para cada ablation, normalização de loss, máscara ou outro
desenho. A preparação rejeita a reutilização de um destino com configuração,
código ou dados diferentes.

Defaults relevantes: treino externo 80%, validação 12,5% do pré-teste,
lookbacks 32/128/256, horizontes 1/5/10/15/20/25, `K=1`, 100 épocas,
patience 10, batch 32, learning rate 0,0001, seed 2026, `d_model=64`, quatro
cabeças temporais e duas entre ativos. `D_MODEL`, `T_HEADS`, `S_HEADS`,
`DROPOUT`, `GATE_BETA`, `LRADJ` e `DEVICE` podem ser alterados no ambiente.
Os parâmetros do desenho são congelados em `run_config.json`.

## Saídas e rastreabilidade

As previsões usam o mesmo `part.parquet`, esquema de origem/passo/horizonte e
ordem de ativos do runner B3. O bloco de teste é previsto sem novo ajuste do
modelo. O contexto histórico pode atravessar um limite de split; os alvos de
cada janela permanecem integralmente no seu split.

Cada tarefa MASTER também guarda, no seu diretório de saída:

- `master_training.json`: configuração, features e ativos, normalizações,
  splits, histórico de treino/validação, melhor época e proveniência.
- `master_checkpoint.pt`: pesos da melhor época e estado necessário para
  reconstituir o modelo. `MASTER.load_checkpoint(...)` verifica o bundle ao
  restaurar; os caminhos podem ser realocados explicitamente.

O hash do desenho inclui todos os arquivos exportados no bundle, o painel
bruto e os arquivos de código relevantes. Uma alteração dos dados auxiliares
invalida o desenho antes de reutilizar uma tarefa concluída. O arquivo de
retornos original continua sendo `dataset/forecasting/b3_daily_tfb_new.csv`;
`master_returns_path` não pode apontar para a cópia com datas sintéticas.

# Backtests incrementais do experimento composto

Execute no **TFB remoto**, branch `experiment/b3-loss-clean-v1`.
O launcher dedicado é `scripts/run_backtests_composto_cpu.sh` e usa o mesmo
motor financeiro validado no temporal/VAR: abertura t+1, fechamento t+K,
CDI, eventos societários, RENT4, custos e slippage por ponta. Não treina redes.

## Primeira execução: treino 80

```bash
cd /sonic_home/igor.viveiros/src/TFB
git pull --ff-only origin experiment/b3-loss-clean-v1

export TRAIN_VALUES="80"
export COMPOSITE_RUN_ID=composto_abertura_fechamento_s0_c0_cdi_v1
bash scripts/run_backtests_composto_cpu.sh planejar
bash scripts/run_backtests_composto_cpu.sh piloto
```

O piloto executa, por loss disponível, as tarefas de menor e maior H,
desempatando por `k_dir`. Ele serve para conferir os insumos e as saídas;
as outras tarefas disponíveis continuam pendentes. Confira seu log antes
da submissão completa:

```bash
bash scripts/run_backtests_composto_cpu.sh submeter
```

Padrões pedidos para esta etapa:

| Parâmetro | Padrão |
| --- | --- |
| Partição | `gorgonas_dev` |
| Tempo | `01:30:00` |
| CPUs/processos por nó | 16, uma thread por processo |
| Memória | 56G |
| Lotes/nós | 1; `N_LOTES` permite 1 a 3 |
| Treinos | `TRAIN_VALUES="80"` |
| Ks de avaliação | `K_EVAL_VALUES="1 5 10 15 20 25"` |
| Custos e slippage | zero bps por ponta |

O launcher fixa `PARTITION=gorgonas_dev` e `TIME_LIMIT=01:30:00`, mesmo que
essas variáveis tenham outros valores exportados na sessão. `WORKERS`,
`MEMORY`, `N_LOTES`, `TRAIN_VALUES` e `K_EVAL_VALUES` podem ser definidos
no ambiente. O script sem argumento só
planeja. `submeter`, `piloto` e `consolidar` devolvem JobID e caminhos dos logs.
O identificador é **`COMPOSITE_RUN_ID`**: o `RUN_ID` usado pelo temporal/VAR
não seleciona esta execução por acidente.

Previsões: `/snfs2/igor.viveiros/previsoes/novas_previsoes/treino=80/experimento=composto/`.
Os nomes de pastas usam um único `=`. Mercado e retornos seguem os caminhos
do launcher temporal; podem ser configurados com `MARKET_DIR` e `RETURNS_FILE`.

Saída padrão:

```text
/snfs2/igor.viveiros/analise_resultados/backtests/composto_abertura_fechamento_s0_c0_cdi_v1/
```

## Como adicionar estimativas sem refazer tarefas concluídas

### Quando o plano mostra zero previsões

O planejamento informa a raiz efetivamente lida, as losses/máscara da grade,
o número de conclusões registradas e os arquivos registrados ausentes. Se um
grupo não tem previsões disponíveis, verifica também se há Parquets da grade
sem registro, sem os incorporar ou modificar. Esse diagnóstico está no código
atual; contratos existentes continuam usando sua versão congelada.

`squeue` mostra tarefas na fila, não o sucesso das tarefas que já saíram dela.
Para conferir um array de treinamento, por exemplo:

```bash
sacct -j 64728 -X --format=JobID,State,ExitCode,Elapsed --parsable2
```

Se não há arquivos publicados, compare `PREDICTIONS_ROOT` com o `OUT_ROOT`
efetivamente usado na submissão e confira os logs de treinamento/exportação.
`COMPOSITE_RUN_ID` identifica a saída do backtest; não seleciona a raiz de
previsões de uma execução nova. Para contratos existentes, a raiz vem do
contrato. Não registre Parquets soltos manualmente nem refaça treinos antes
de identificar a causa.

### Incorporar novas conclusões

Quando novas previsões estiverem prontas, repita **o mesmo comando e o mesmo
COMPOSITE_RUN_ID**:

```bash
bash scripts/run_backtests_composto_cpu.sh submeter
```

A preparação lê os `run_config.json` e `completed_tasks.csv` de cada treino.
Só incorpora Parquets concluídos e registrados pelo produtor. Uma grade
incompleta é aceita; arquivo solto ainda não registrado não é processado.
Um registro cujo arquivo ainda falta aparece nas pendências de descoberta.

As tarefas já registradas mantêm seus IDs e assinaturas quando outras chegam.
Tarefas `COMPLETA` com arquivos íntegros são reutilizadas sem calcular ICs ou
carteiras novamente. Tarefas parciais, com erro, interrompidas ou com saídas
corrompidas são reprocessadas. Checksums continuam sendo verificados: a
retomada evita cálculos, mas ainda lê arquivos para conferir a integridade.

Para incluir outros treinos posteriormente:

```bash
export TRAIN_VALUES="70 80 90"
bash scripts/run_backtests_composto_cpu.sh planejar
bash scripts/run_backtests_composto_cpu.sh submeter
```

Treinos ainda sem configuração são informados como desconhecidos, sem
contagem inventada. Os treinos previamente registrados permanecem no catálogo
mesmo que uma chamada posterior informe somente parte deles.

Cada submissão fixa sua lista em `planos/<assinatura>.json`. Se novas previsões
forem registradas enquanto o job aguarda na fila, elas entrarão na próxima
submissão; a lista do job anterior não muda. Jobs simultâneos usam locks por
tarefa. A consolidação considera o catálogo mais recente, inclusive pendências.

Mercado, universo, retornos, código, estratégias, custos e Ks de avaliação
ficam fixos em `contrato.json`. Alterá-los exige outro `COMPOSITE_RUN_ID`.
Um Parquet já aceito não pode ser substituído silenciosamente por nova estimação.
Adicionar novos treinos e concluir previsões do desenho existente é permitido.
Não modifique a grade/configuração de um treino já registrado nesta execução.

### Retomar contratos anteriores após atualizar o repositório

O launcher seleciona automaticamente `codigo/run_backtests_composto.py` da
execução indicada por `COMPOSITE_RUN_ID`, verificando primeiro a assinatura do
contrato e os hashes de todos os arquivos congelados. Isso vale para planejar,
piloto, submeter e consolidar, inclusive contratos da versão anterior às losses
de proximidade. O contrato, o código congelado e os resultados concluídos não
são migrados. As verificações originais de insumos, ambiente e integridade
continuam em vigor.

Para um ID existente, o launcher recupera do contrato `PREDICTIONS_ROOT`, mercado,
retornos, estratégias, Ks, custos e slippage; valores dessas variáveis no ambiente
não substituem o contrato. `TRAIN_VALUES`, quando explicitamente definido, ainda
permite incluir novos cortes; se omitido, usa os cortes do plano salvo. Recursos
Slurm (`WORKERS`, memória e lotes) continuam configuráveis; a partição fica
fixa em `gorgonas_dev`, com limite de `01:30:00`.

```bash
export COMPOSITE_RUN_ID=composto_abertura_fechamento_s0_c0_cdi_v1
bash scripts/run_backtests_composto_cpu.sh planejar
bash scripts/run_backtests_composto_cpu.sh submeter
```

### Execução separada das novas losses de proximidade

O treinamento usa `hinge_proximity`, `ranknet_proximity` e
`pairwise_mse_proximity`, com `CROSS_MASK=(uniform 0.025 natural)`,
`CROSS_MASK=(gaussian 0.025 natural)` ou `CROSS_MASK=(cauchy 0.025 natural)`.
O terceiro elemento também aceita `complementary`, para usar `1-peso` e dar
mais peso aos pares distantes. Vetores antigos de dois elementos significam
`natural`. Use outro `OUT_ROOT` no treinamento e aponte
`PREDICTIONS_ROOT` para essa raiz no backtest. Escolha um **novo**
`COMPOSITE_RUN_ID`; não reutilize o contrato antigo para esse desenho.

```bash
export PREDICTIONS_ROOT=/snfs2/igor.viveiros/previsoes/proximity_uniform_a0025
export COMPOSITE_RUN_ID=composto_proximity_uniform_a0025_s0_c0_cdi_v1
export TRAIN_VALUES="80"
bash scripts/run_backtests_composto_cpu.sh planejar
bash scripts/run_backtests_composto_cpu.sh piloto
# Após conferir o piloto:
bash scripts/run_backtests_composto_cpu.sh submeter
```

A descoberta lê tipo de máscara e `a` do desenho de treinamento; não é preciso
repeti-los no launcher do backtest. As novas losses têm `delta=0` inativo, sem
repetir tarefas pelo vetor `DELTAS`. Seus caminhos incluem `mask=<tipo>/a=<valor>`.
As tabelas de ICs, carteiras, posições, métricas, falhas e o status das tarefas
preservam `mask_type`, `mask_a`, `mask_mode` e `mask_effective`. Registros antigos
de proximidade sem `mask_mode` são interpretados como `natural`; contratos já
preparados continuam usando seu snapshot. Nas losses originais de um
contrato novo, os dois primeiros campos são nulos e o último é falso. Contratos
antigos mantêm seu schema original. A fusão entre execuções fica para a análise
local em R; IDs de tarefa são locais a cada execução.

## Identificação e janelas

A descoberta usa o desenho salvo, sem reconstruir uma grade fixa no launcher.
Assim, respeita `lambda` 0.1/0.5/0.8, deltas 0/0.025/0.05/0.1 e as combinações
efetivamente configuradas. **ListNet tem delta canônico zero, inativo**, sem
duplicar experimentos pelas demais margens. O contrato do treinamento exige
`H % k_dir == 0`.

Todas as tabelas preservam `treino`, `modelo`, `lookback`, `pred_len`, `loss`,
`lambda`, `delta`, `cross_lambda`, `cross_delta`, `delta_effective`, `k_dir`,
`k_loss`, `seed`, `temporal_loss`, `training_array_index`, `task_id`, custos e
assinatura individual. `k_loss` é um alias de `k_dir`, para compatibilidade.

| Campo | Significado |
| --- | --- |
| `k_dir` / `k_loss` | Acúmulo utilizado pela loss cross-sectional no treinamento |
| `h` | Antecedência de um retorno diário previsto, de 1 até H |
| `k` | Janela de acúmulo e avaliação do backtest, dentre os Ks configurados com K ≤ H |

Uma estimação com `k_dir=5` também é avaliada em K=1,5,10 etc., quando K ≤ H.
Não se fundem realizações que diferem em `k_dir`, lambda ou delta.

O IC diário compara os retornos previstos e reais apenas no passo h. O IC
acumulado compara a soma dos primeiros K log-retornos previstos com a soma
dos realizados, cross-sectionalmente. As séries de IC usam origens diárias;
as carteiras rebalanceiam a cada K pregões, ancoradas no corte de treino.

A amostra comum usa `N - max(config.horizons)` de cada treino, considerando
inclusive os horizontes ainda não concluídos. A chegada de H maior não muda
as datas dos resultados antigos. Com a mesma grade H e a mesma base, as datas
coincidem com temporal/VAR.

## Saídas completas

| Local | Arquivos |
| --- | --- |
| Cada tarefa em `tasks/treino=.../experimento=composto/.../k_dir=.../` | `ic.parquet` (diário), `ic_resumo.parquet`, `ic_acumulado.parquet`, `ic_acumulado_resumo.parquet`, `carteira_diaria.parquet`, `carteira_periodos.parquet`, `posicoes.parquet`, `metricas.parquet`, `falhas.parquet`, `conclusao.json` |
| `resumos/` | `ic_diario.parquet`, **`ic_acumulado.parquet`**, `ic_resumo.parquet`, `ic_acumulado_resumo.parquet`, `metricas.parquet`, `falhas.parquet`, `status_tarefas.csv`, `estado_execucao.json`, `descoberta_previsoes.json` |

Arquivos de carteiras podem faltar numa tarefa parcial caso todas as respectivas
estratégias tenham falhado; consulte `falhas.parquet` e os logs. Nunca se marca
essa tarefa como completa. Os dois ICs consolidados são copiados em lotes,
sem carregar todas as séries simultaneamente em memória e sem calcular médias
entre configurações.

`estado_execucao.json` distingue:

- `completa`: todas as previsões já incorporadas ao catálogo foram processadas;
- `completa_grade`: além disso, todas as estimativas da grade pedida chegaram;
- `discovery.awaiting`: estimativas faltantes nas grades conhecidas;
- `discovery.missing_groups`: treinos sem configuração, cuja quantidade ainda é desconhecida.

As contagens `COMPLETA/PARCIAL/ERRO/PENDENTE` referem-se às tarefas de backtest
do catálogo; não confundem falta de previsão com erro do backtest. Para apenas
reunir resultados existentes, sem acessar insumos ou previsões:

```bash
bash scripts/run_backtests_composto_cpu.sh consolidar
```

## Consulta em R

Transfira `plano.json`, `resumos/` e, para consultas individuais, `tasks/`.
Atualize `analise_resultados`, branch `feat/base-mercado-backtest`:

```r
source("src/06_resultados_cluster.R")
x <- abrir_resultados_cluster("resultados_carteiras/composto")

ic <- x$ic_acumulado |>
  dplyr::filter(treino == 80, loss == "hinge", lambda == "0.5",
                delta == 0.025, k_dir == 5, k == 10, amostra_comum) |>
  dplyr::collect()
```

Mantenha `k_dir` fixo ao comparar Ks de avaliação; inclua-o nas chaves das
curvas/tabelas ao comparar realizações. `x$ic_diario` também é uma consulta
Arrow filtrável. Os resumos e as métricas ficam nos campos habituais.

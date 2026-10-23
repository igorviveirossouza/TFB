# Avaliar as três previsões prontas do piloto MASTER

Na phocus4, execute apenas a submissão:

```bash
cd /sonic_home/igor.viveiros/src/TFB
git pull --ff-only origin experiment/b3-loss-clean-v1
bash scripts/run_master_pilot_backtests.sh
```

Um único job CPU em **gorgonas_dev** faz a auditoria, o planejamento, os ICs,
as carteiras e a consolidação. Defaults: duas CPUs, 16 GB, 1h30 e Python absoluto
`/sonic_home/igor.viveiros/py310/bin/python`. Nenhum treinamento é iniciado.
O job executa a revisão Git registrada na submissão; futuras atualizações do
checkout não alteram essa revisão. Os motores de backtest também congelam seu
próprio código. O launcher não altera os scripts/assinaturas de treinamento.

Entrada: `/snfs2/igor.viveiros/previsoes/master_piloto_l32_h5_k1_t80_v1`.
A auditoria exige uma previsão MSE, uma RankNet e uma RankNet Proximity natural
gaussiana `a=0.025`, todas MASTER, treino 80%, lookback 32, H=5 e K da loss=1.
Confere manifestos, hashes, dados, 55 ativos, 3664 pregões, 729 origens por
previsão, 3645 linhas por Parquet e o histórico completo dos treinamentos.
Verifica a presença do checkpoint; não carrega nem reproduz os pesos salvos.
Se qualquer previsão falhar na auditoria, nenhum backtest começa.

Avaliação: IC diário por passo `h=1..5`, IC acumulado em **K=1 e K=5** e
carteiras `compra_positivos`/`compra_ranking`, Top-9, abertura–fechamento,
slippage/custos zero e CDI BCB_SGS12. São **12 carteiras**: três previsões ×
dois K × duas estratégias. O K da avaliação é independente do K da loss.
As três previsões usam a mesma amostra de teste. Resultados de outros modelos
com horizontes máximos distintos podem ter outra amostra comum; não os juntar
automaticamente para uma comparação de desempenho.

Saída padrão:

```text
/snfs2/igor.viveiros/analise_resultados/backtests/master_piloto_l32_h5_k1_t80_s0_c0_cdi_v1/
```

- `validacao_previsoes.json`: auditoria das três previsões.
- `avaliacao.json`: `PASSED` somente depois da conclusão integral.
- `comparacao_ic_diario.csv`, `comparacao_ic_acumulado.csv` e
  `comparacao_carteiras.csv`: tabelas conjuntas. `avaliacao_id` distingue os IDs
  locais dos dois planos; `familia`/`experimento` identifica temporal/composto.
- `temporal/resumos/` e `composto/resumos/`: Parquets e estado usuais, compatíveis
  com `abrir_resultados_cluster()` do R. Cada subpasta tem seu próprio `plano.json`.

Logs: `logs/master/avaliacao_piloto_v1/avaliacao_JOBID.{out,err}`. Ao final, a
saída esperada é `MASTER: PASSED; 3 previsoes; 12 carteiras; K=1/5`.
Reexecutar o mesmo comando reaproveita os backtests concluídos e verificados;
não refaz treinamentos nem sobrescreve as previsões. Falhas mantêm estado
`FAILED` e o processo retorna erro. Duas avaliações simultâneas da mesma pasta
são bloqueadas por um lock.

Overrides específicos: `MASTER_EVAL_PREDICTIONS`, `MASTER_EVAL_MARKET`,
`MASTER_EVAL_RETURNS`, `MASTER_EVAL_RUN_DIR`, `MASTER_EVAL_LOG_DIR`,
`MASTER_EVAL_PARTITION`, `MASTER_EVAL_TIME`, `MASTER_EVAL_MEMORY` e
`MASTER_EVAL_WORKERS`. Os parâmetros genéricos de grades antigas não alteram
esse piloto. Para outra especificação econômica ou de treinamento, prepare
outro desenho em vez de mudar o contrato deste piloto.

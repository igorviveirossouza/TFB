# Momentum com o mesmo protocolo dos backtests concluídos

Execute no TFB remoto, branch `experiment/b3-loss-clean-v1`. O novo benchmark
reaproveita o **plano e o motor congelados** da execução temporal/VAR já
concluída. Não lê previsões, não treina modelos e não modifica os resultados
originais. Requer que os insumos de mercado e retornos ainda estejam nos
caminhos registrados no plano, com os mesmos checksums, e o mesmo ambiente
Python/bibliotecas da execução original.

## Planejar e submeter

```bash
git pull --ff-only origin experiment/b3-loss-clean-v1

export REFERENCE_RUN=/snfs2/igor.viveiros/analise_resultados/backtests/temporal_var_abertura_fechamento_s0_c0_cdi_v3_rent4
export RUN_ID=momentum_temporal_var_s0_c0_cdi_v1
bash scripts/run_momentum_reference_cpu.sh planejar
bash scripts/run_momentum_reference_cpu.sh submeter
```

Sem argumento o launcher apenas planeja. `submeter` devolve o JobID, o destino
e o diretório de logs. O job usa `gorgonas`, uma CPU/processo, sem GPU, com
threads BLAS limitadas a uma. Padrões: `MEMORY=16G`, `TIME_LIMIT=04:00:00`.
Toda saída, código próprio preservado e temporários ficam no SNFS2. Os
caminhos Python e raiz do repositório são configuráveis por `PYTHON_BIN` e
`TFB_ROOT`, como no launcher temporal/VAR.

O script separado deriva do plano de referência:

- treinos e lookbacks das tarefas `temporal` (não o lag específico do VAR);
- Ks de avaliação e limite comum das origens de cada corte;
- universo, mercado, CDI, estratégias e custos/slippage.

Há uma tarefa por treino/lookback, com todos os Ks. Na grade com três cortes,
três lookbacks, seis Ks e quatro estratégias são nove tarefas, 54 séries de
IC acumulado e 216 carteiras, sem multiplicar esses números por H.

## Definição e comparação

O score usa a soma de L log-retornos corrigidos até o fechamento de t,
equivalente a log(P*ₜ/P*ₜ₋L), com P* sendo o preço de retorno total ajustado.
Isso preserva o sinal e o ranking do momentum percentual nos mesmos preços.
Não usa dados posteriores a t e não aplica a exclusão do mês recente. O
lookback conta retornos passados, exigindo L observações disponíveis na origem.

Para cada K o Spearman cross-sectional compara o score com a soma dos próximos
K log-retornos corrigidos. O IC tem todas as origens diárias possíveis;
`amostra_comum` conserva as origens comuns aos modelos da referência. O
resumo contém as amostras `comum` e `completa`; a comparação deve usar `comum`.

As carteiras usam origens corte, corte+K, corte+2K etc. até o mesmo limite
comum da referência, entrada na abertura t+1 e saída no fechamento t+K.
O motor original trata proventos, splits, posições recebidas/devidas (RENT4),
caixa/CDI, execução e custos. Um ativo sem trajetória necessária continua
provocando falha explícita; resultados parciais não são marcados completos.

## Saídas e consulta em R

`resumos/` contém `ic_acumulado_resumo.parquet`, `metricas.parquet`,
`falhas.parquet`, `status_tarefas.csv` e `estado_execucao.json`.
Cada pasta em `tasks/` contém a série `ic_acumulado.parquet`, resumo,
carteiras diárias/por período, posições, métricas e falhas correspondentes.
Não se inventa uma previsão diária `h`: Momentum fornece um score na origem.
`pred_len` é nulo porque o benchmark não depende de H.

Copie o resumo para
`analise_resultados/resultados_carteiras/momentum/ic_acumulado_resumo.parquet`.
O script R `src/07_graficos_ic_acumulado.R` aceita esse arquivo no argumento
`momentum` de `criar_graficos_ic_acumulado()`. Somente no gráfico o valor é
repetido nos H ≥ K correspondentes, sem duplicar cálculos.

O comando Python também está disponível para execução controlada:

```bash
python -B scripts/run_momentum_reference.py \
  --reference-run "$REFERENCE_RUN" \
  --run-dir "/snfs2/igor.viveiros/analise_resultados/backtests/$RUN_ID" \
  --dry-run
```

`--allow-local` destina-se a testes fora do SNFS2; o launcher de produção não
usa essa opção. Execuções com insumos ou código diferentes exigem outro RUN_ID.

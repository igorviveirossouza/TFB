# IC e backtests CPU: temporal + VAR

Esta etapa processa somente `temporal` e `var` das séries corrigidas. As
losses compostas ficam para a próxima etapa. Os cálculos remotos são Python;
gráficos e tabelas finais ficam no R local. O motor R antigo permanece
disponível para as análises antigas.

O código remoto está neste repositório **TFB**, na branch
**`experiment/b3-loss-clean-v1`**:

- `scripts/run_backtests_cpu.sh`: configuração e submissão Slurm;
- `scripts/run_backtests.py` e `scripts/backtest_cpu/`: motor Python;
- `config/estrategias_backtest.json`: regras das carteiras;
- `requirements-backtest.txt` e `tests/test_backtest_cpu.py`: ambiente e testes.

O benchmark Momentum tem execução separada em
`scripts/run_momentum_reference_cpu.sh`. Ele deriva os treinos, lookbacks, Ks
e convenções do plano já concluído, preservando seus resultados. Veja
[Momentum com o mesmo protocolo](b3_momentum_reference.md).

A preparação e a consulta em R pertencem ao repositório separado
[`analise_resultados`](https://github.com/igorviveirossouza/analise_resultados/tree/feat/base-mercado-backtest).
O job não precisa de um checkout desse repositório no servidor: lê o bundle
de mercado já transferido para o SNFS2.

## Cálculos e convenções

| Tabela | Comparação cross-sectional |
| --- | --- |
| `ic.parquet` | **IC natural diário**: previsão do log-retorno para T contra log-retorno corrigido realizado em T. Uma linha por `(origem,h)`; T=origem+h. |
| `ic_acumulado.parquet` | Agregação do R antigo: soma dos primeiros K log-retornos previstos contra soma dos mesmos K log-retornos corrigidos realizados. Uma linha por `(origem,K)`. |

`resumos/ic_diario.parquet` reúne os `ic.parquet` de todas as tarefas válidas
em uma única tabela temporal. Preserva cada valor, inclusive NA e seu
`status_ic`, sem agregar datas, modelos ou horizontes. Cada linha identifica
`task_id`, `treino`, `experimento`, `modelo`, `lookback`, `pred_len`, `loss`,
`lambda`, `h`, `origin_step`, `target_step`, `data_origem`, `data_alvo`,
`spearman`, `n_pares` e `amostra_comum`, além dos metadados do cenário.
Para a evolução no tempo, use `data_alvo` no eixo horizontal e `spearman`
no vertical, mantendo uma curva por configuração e por `h`. O horizonte
total do modelo (`pred_len`) e a antecedência da previsão (`h`) são distintos.

Não mistura previsões de origens diferentes para a mesma data T. Não calcula
IC contra o retorno abertura–fechamento da carteira. Spearman usa ranks
médios nos empates, exige ao menos três pares finitos e retorna NA se algum
rank for constante. Os ranks dos log-retornos e dos retornos simples
equivalentes são iguais; evita-se exponenciação desnecessária.

O IC é calculado uma vez por configuração, independentemente da estratégia.
Os resumos identificam `amostra=completa` (todas as janelas da fonte) e
`amostra=comum` (mesmas origens entre todas as fontes do corte). `ic_ir` é
média/desvio, sem anualização ou teste de significância: janelas sobrepostas
não são observações independentes.

### Carteiras e estratégias

No fechamento t, o score é a soma dos primeiros K log-retornos previstos.
Entra na **abertura de t+1** e sai no **fechamento de t+K**. K=1 é intradiário;
K>1 permanece posicionado entre esses pregões, inclusive overnight. O novo
bloco entra na abertura seguinte, usando o novo sinal. Não há sobreposição
de carteiras, entrada no fechamento t ou regime abertura–abertura.

Edite `config/estrategias_backtest.json`:

| ID padrão | Seleção |
| --- | --- |
| `compra_positivos` | Até 9 maiores scores, estritamente positivos |
| `compra_ranking` | Até 9 maiores scores, independentemente do sinal |
| `venda_negativos` | Até 9 menores scores, estritamente negativos |
| `venda_ranking` | Até 9 menores scores, independentemente do sinal |

`n_ativos` controla J ou L, `lado` é `compra` ou `venda`, e `filtrar_sinal`
liga a restrição de sinal. Pode acrescentar estratégias com IDs únicos.
Empates são resolvidos pelo ticker alfabético. Pesos são iguais na entrada;
quantidades ficam fixas no bloco, exceto por eventos de quantidade. Ausência
de sinais elegíveis mantém caixa sem remuneração. Previsões inválidas são
erro, não carteira de caixa.

K é a janela de avaliação, não K da loss composta. Os padrões são
`1 5 10 15 20 25`, com K<=H, sem exigir H divisível por K.

### Preços, eventos e limites de execução

- Usa abertura e fechamento unitários auditados, compatíveis com FATCOT.
  Preço preenchido pode marcar uma posição, mas não executa entrada/saída.
- Entrada na data ex não recebe eventos destinados ao titular anterior.
  Durante a posse, splits/grupamentos alteram quantidades; proventos em
  dinheiro geram direitos na compra e obrigações na venda.
- Proventos são reconhecidos economicamente na data ex, sem reinvestimento
  dentro do bloco, e entram no capital do próximo rebalanceamento. Sem datas
  de pagamento na base, esse pressuposto não reproduz a disponibilidade de
  caixa na corretora antes do pagamento.
- **Distribuição de outro ativo cria uma posição auxiliar**, recebida na compra
  e devida na venda. Usa a quantidade da origem imediatamente antes da data ex,
  marca o ativo recebido diariamente e o liquida no fechamento da origem, com
  slippage/custo de saída. Sua entrega não é uma compra e não cobra taxa de entrada.
  Valor inicial, proporção e IDs reconciliam com a auditoria congelada. Sem
  trajetória ou fechamento executável, bloqueia aquela carteira; o IC continua.
  O valor V usado no alvo do modelo não é creditado novamente como dinheiro.
- Falta de abertura/fechamento executável de um ativo selecionado bloqueia
  aquela carteira inteira, indicando ticker, data e origem. Não troca o
  ativo nem exclui retrospectivamente datas inconvenientes.
- Na venda, o lucro é venda menos recompra, com eventos e proventos. Não é
  o negativo do log-retorno composto. A exposição bruta inicial é até 1x
  o patrimônio; não simula aluguel, disponibilidade de empréstimo, chamadas
  de margem ou remuneração das garantias. Patrimônio<=0 interrompe a carteira.
- Quantidades fracionárias são permitidas. Não simula lotes mínimos nem
  arredondamento de ordens. A abertura continua sendo uma aproximação de execução.

### Slippage e custos

O SH começa com ambos em **zero**. Valores são pontos base **por ponta**:

- `SLIPPAGE_BPS=5`: compra a 1,0005 vezes a cotação; venda a 0,9995 vezes.
  Vale também para a venda inicial e a recompra do short.
- `COST_BPS=5`: taxa de 0,05% sobre o financeiro efetivamente negociado em
  cada ponta. Não representa imposto sobre lucro ou custo fixo por ordem.
- O orçamento inclui a taxa de entrada. Para peso w, preço efetivo P e
  custo decimal c, quantidade absoluta=`w/(P*(1+c))` nos dois lados. Na
  venda, o produto da venda permanece como garantia.
- O custo de saída incide sobre o financeiro de saída, com as quantidades
  ajustadas. Mesmo ativo no próximo bloco implica fechamento e nova abertura.

Mudar cenário, estratégias, dados ou código exige outro `RUN_ID`.

### Datas comuns e métricas

Por corte, Hmax é o maior horizonte da grade. Todas as carteiras usam
origens `corte, corte+K, ... <= N-Hmax`, com `corte=floor(N*p/100)`. Assim,
para o mesmo corte/K/estratégia, modelos, lookbacks e horizontes usam as
mesmas datas. A cauda adicional de H menores fica disponível no IC completo.
K diferentes podem encerrar em datas diferentes devido aos blocos completos.
Os cortes 70/80/90 representam períodos de teste diferentes.

As métricas usam a trajetória **diária**, inclusive os dias internos dos
blocos: retorno total, CAGR de 252 pregões/ano, desvio amostral, volatilidade
anualizada, drawdown desde patrimônio inicial 1 e Sharpe.

O benchmark é o **CDI diário realizado (BCB SGS 12)**, obrigatório no
`calendario_tfb.csv` do bundle de mercado, contratos `2026-10-15-v2-cdi`
e `2026-10-15-v3-recebidos`.
Não existe arquivo de juros separado nem download durante o job.

- `retorno_cdi`: retorno simples decimal já acumulado no R;
- `log_retorno_cdi`: log equivalente, conferido pelo Python;
- `n_observacoes_cdi`: quantidade de observações ANBIMA no intervalo;
- `rf_fonte`: sempre `BCB_SGS12`.

Reutiliza o CDI auditado em **[data_anterior, date_pregao)**, incluindo dias
ANBIMA sem pregão B3. A SGS 12 original é % ao dia; o R já dividiu por 100.
O Python não desloca a série nem divide novamente por 100 ou 252.
Taxas ausentes/inválidas, fonte ou convenção diferente e versões antigas da
base interrompem o planejamento antes de criar a execução.

`sharpe_cdi = sqrt(252)*mean(r-rf)/sd(r-rf)`, com desvio amostral e trajetória
diária, mesmo para K>1. `sharpe_geometrico_cdi` registra separadamente
`(CAGR_carteira - CAGR_CDI)/vol_anual`. Também salva `cdi_acumulado`,
`cdi_anualizado` e `status_rf=BCB_SGS12`; `sharpe_rf_zero` permanece apenas
como diagnóstico identificado. Desvio nulo/uma observação gera Sharpe NA.
A carteira diária conserva `rf` e cada bloco conserva `retorno_cdi` composto.

É um custo de oportunidade diário entre fechamentos, inclusive para a carteira
K=1 abertura–fechamento; não é uma taxa intradiária negociável. Não remunera
caixa/garantias nem participa dos sinais e do IC.

Para a pasta de insumos já transferida, atualize **no R local**, no repositório
`analise_resultados`:

```r
options(
  b3.backtest.auditoria = "/caminho/analise_resultados/output/auditoria/ID_DA_EXECUCAO",
  b3.backtest.saida = "/caminho/analise_resultados/output/base_backtest_temporal_001",
  b3.backtest.atualizar_cdi = TRUE
)
source("src/exportar_base_backtest.R")
```

Reenvie os quatro arquivos existentes atualizados (`calendario_tfb.csv`,
`manifesto.csv`, `fontes.csv` e, por último, `checksums_saidas.csv`) para
`MARKET_DIR`, sem jobs lendo essa pasta. Não precisa reenviar preços, criar
outra pasta de insumos ou transferir `cdi_tfb.csv` separadamente. Detalhes na
[preparação R](https://github.com/igorviveirossouza/analise_resultados/blob/feat/base-mercado-backtest/docs/base_mercado_backtest.md).
O novo `RUN_ID` padrão contém `cdi_v3`, preservando resultados e planos antigos.

### Ativos recebidos: RENT3 / RENT4

O contrato `2026-10-15-v3-recebidos` acrescenta registros `RECEBIDO` no mesmo
`mercado_diario.csv`/Parquet. Os registros `MODELO` continuam constituindo o
painel balanceado e o universo das previsões. Quantidades, calendário, preços,
fontes e valores da data ex são conferidos antes do processamento.

A cobertura revisada de RENT4 vai de 30/12/2025 a 06/02/2026, com quantidade
constante e sem novos proventos/eventos próprios nessa janela. Não se extrapola
essa convenção. A saída além da cobertura falha; não mantém artificialmente o
último preço. Essa janela cobre as posições que atravessam a bonificação com
K<=25. O início de negociação de 30/12/2025 segue a B3; o tratamento mantém
a aproximação já adotada de quantidades fracionárias e ignora liquidação D+2,
arredondamentos individuais e leilão das frações. Não é uma simulação de custódia.

Na máquina local, atualize o repositório R e execute:

```r
# Antes de carregar rb3; use o cache que contém os COTAHIST históricos.
options(rb3.cachedir = "/media/igor/KINGSTON1/GitHub/FinancialMarket/B3_cotahist")
options(
  b3.backtest.auditoria = "/caminho/output/auditoria/ID_DA_EXECUCAO",
  b3.backtest.saida = "/caminho/output/base_backtest_temporal_001",
  b3.backtest.atualizar_cdi = FALSE,
  b3.backtest.atualizar_recebidos = TRUE
)
source("src/exportar_base_backtest.R")
```

Reenvie **mercado_diario**, **manifesto.csv**, **fontes.csv** e, por último,
**checksums_saidas.csv** para a mesma pasta de insumos, sem jobs lendo-a.
Não precisa reenviar calendário, CDI, retornos ou previsões. Atualize também
o TFB na branch `experiment/b3-loss-clean-v1`, rode `planejar` e repita `piloto`
com o novo RUN_ID padrão. O plano imprime a cobertura auxiliar de RENT4.
Resultados antigos usam outra assinatura; não reaproveite o RUN_ID `cdi_v2`.

`posicoes.parquet` distingue `tipo_posicao=SINAL` de `RECEBIDA`, com
`ticker_origem` e `id_distribuicao`. O número de ativos da estratégia continua
contando as escolhas do ranking; a posição recebida é acessória. Trajetórias e
períodos também identificam seu valor e custo de saída. Esses campos permitem
conferir o patrimônio sem duplicar V nem contar RENT4 como uma previsão extra.
Falhas de carteiras são impressas no log do Slurm com K, estratégia e motivo,
além dos registros existentes em `falhas.parquet`.

Fontes e procedimento da preparação estão na
[documentação R](https://github.com/igorviveirossouza/analise_resultados/blob/feat/base-mercado-backtest/docs/base_mercado_backtest.md#ativos-recebidos-no-mesmo-arquivo-de-mercado).

## Entradas e planejamento

Os caminhos editáveis no SH são:

```text
PREDICTIONS_ROOT=/snfs2/igor.viveiros/previsoes/novas_previsoes
MARKET_DIR=/snfs2/igor.viveiros/analise_resultados/insumos/base_backtest_temporal_001
RETURNS_FILE=/sonic_home/igor.viveiros/src/TFB/dataset/forecasting/b3_daily_tfb_new.csv
OUTPUT_ROOT=/snfs2/igor.viveiros/analise_resultados/backtests
```

O CSV de treino é somente lido. São necessários os `run_config.json`,
`completed_tasks.csv` e `part.parquet` produzidos pelo TFB e o bundle de
mercado completo, com checksums. Não copie apenas `mercado_diario.csv`.

O planejamento confere identidade do CSV, checksums do bundle, calendário,
universo, reprodução dos eventos e grade concluída. Exige previsões da fonte
`log_return`, `step_offset=1`, stride=1, SHA256 e corte compatíveis. Confere
schema e tamanho dos Parquets; o worker verifica seu SHA256 e todas as chaves
`(origin_step,h,step)`. Uma grade parcialmente concluída não é aceita.

A grade padrão tem **342 Parquets**: 324 redes temporais e 18 saídas VAR
(3 ajustes, 6 H). Com seis K e quatro estratégias, são **4.608 carteiras**.
As carteiras VAR usam uma única vez o arquivo de maior H por corte, pois os
demais H são prefixos do mesmo ajuste. As tarefas VAR de H menor salvam
`referencia_carteiras.parquet` apontando para essa fonte; o leitor R segue
a referência. O IC mantém as coberturas de cada H identificadas separadamente.
VAR não é duplicado para cada arquitetura neural. Arquivos das losses
compostas não são varridos nem processados.

## Instalação e execução no servidor

No servidor, entre no repositório **TFB** e atualize a branch correta:

```bash
cd /sonic_home/igor.viveiros/src/TFB
git switch experiment/b3-loss-clean-v1
git pull --ff-only origin experiment/b3-loss-clean-v1
```

Use Python>=3.10. Aponte `PYTHON_BIN` para um ambiente existente com as
dependências, ou crie um ambiente no SNFS2:

```bash
mkdir -p /snfs2/igor.viveiros/analise_resultados/{ambientes,tmp,cache/pip}
export TMPDIR=/snfs2/igor.viveiros/analise_resultados/tmp
export PIP_CACHE_DIR=/snfs2/igor.viveiros/analise_resultados/cache/pip
python3 -m venv /snfs2/igor.viveiros/analise_resultados/ambientes/backtest_cpu
export PYTHON_BIN=/snfs2/igor.viveiros/analise_resultados/ambientes/backtest_cpu/bin/python
"$PYTHON_BIN" -m pip install -r requirements-backtest.txt
```

Não exige R, PyTorch ou CUDA. Confira os caminhos do SH; variáveis de
ambiente também os sobrescrevem. A execução é deliberadamente em etapas:

```bash
# Valida e mostra o plano; nao escreve arquivos nem submete jobs.
bash scripts/run_backtests_cpu.sh planejar

# Menor/maior H de uma rede e do VAR no primeiro corte: quatro Parquets.
# Processa todos os K e estrategias aplicaveis; usa gorgonas.
bash scripts/run_backtests_cpu.sh piloto

# Apos conferir o piloto, processa a grade e reaproveita tarefas completas.
bash scripts/run_backtests_cpu.sh submeter

# Reune resumos e a serie diaria ja calculada, sem executar os backtests.
bash scripts/run_backtests_cpu.sh consolidar
```

`TFB_ROOT` é obtido da localização do SH, salvo se informado no ambiente.
O caminho padrão do painel é `$TFB_ROOT/dataset/forecasting/b3_daily_tfb_new.csv`.
As pastas `/snfs2/igor.viveiros/analise_resultados/insumos` e `backtests` são
diretórios de dados/resultados e não o checkout do repositório R.

O padrão é **um nó gorgonas, 16 processos, 56G e 24h**, sem GPU. Uma thread
por processo nas bibliotecas numéricas/Arrow, usando os 16 núcleos físicos.
Como gorgonas é exclusiva, cada job processa várias configurações em paralelo;
não há um job de uma CPU por carteira. Nunca usa gorgonas_dev.

`N_LOTES=3 bash scripts/run_backtests_cpu.sh submeter` divide entre até três
nós, cada um com seu pool de processos. Ajuste pelo piloto e ocupação real;
não há promessa de speedup linear. Uma leitura tabular atende todos os K e
estratégias de cada fonte; o checksum requer leitura sequencial adicional.
Matrizes e vetorização entre blocos evitam conversões long e cálculos de IC
repetidos. A base de mercado é carregada uma vez por processo.

Tudo que o job gera fica na execução sob SNFS2: resultados, logs Slurm,
temporários, cache e snapshot de código. Os cálculos usam caminhos absolutos
e o snapshot, sem depender de `BASH_SOURCE` no spool do Slurm. A ação
`consolidar` usa o exportador do checkout atualizado e confere o snapshot
original; mantenha esse checkout estável até o job terminar.
O plano registra as versões de Python, NumPy, pandas, SciPy e PyArrow;
retomadas exigem as mesmas versões para não misturar ambientes.

## Retomada e saídas

Repita `submeter` com o mesmo `RUN_ID`: resultados completos são verificados
por checksum e reutilizados; tarefas incompletas/corrompidas são refeitas.
Locks evitam publicação simultânea da mesma tarefa. Entradas não são alteradas.
Mudar cenário exige novo nome, por exemplo:

```bash
RUN_ID=temporal_var_s5_c0_cdi_v3 SLIPPAGE_BPS=5 bash scripts/run_backtests_cpu.sh piloto
```

A pasta contém `plano.json`, `estrategias.json`, snapshot em `codigo/`,
`logs/`, `status/` por tarefa e:

```text
tasks/treino=.../experimento=.../loss=mse/lambda=00/modelo=.../lookback=.../pred_len=.../
    ic.parquet
    ic_resumo.parquet
    ic_acumulado.parquet
    ic_acumulado_resumo.parquet
    carteira_diaria.parquet
    carteira_periodos.parquet
    posicoes.parquet
    metricas.parquet
    falhas.parquet
    conclusao.json
resumos/
    metricas.parquet
    ic_diario.parquet
    ic_resumo.parquet
    ic_acumulado_resumo.parquet
    falhas.parquet
    status_tarefas.csv
    estado_execucao.json
```

As tabelas trazem metadados da fonte e, nas carteiras, K, estratégia, lado,
número máximo, filtro e custos. Quantidades/direitos das posições são por
capital **do início do bloco**, não pelo patrimônio inicial da execução.

Execução inválida gera `falhas.parquet`, sem apresentar trajetória truncada
como completa. Outras carteiras e IC continuam; o job retorna código diferente
de zero se seu lote tiver falhas. Erros de fonte estão em `status/*.json`.
`estado_execucao.json` conta completas, parciais, erros e pendências. O piloto
deixa pendências intencionalmente. Evite submeter piloto e grade simultaneamente:
locks evitam duplicação, mas CPUs ficariam aguardando a tarefa já em curso.

Para acrescentar a tabela temporal a uma execução já concluída, atualize o
TFB e use **o mesmo RUN_ID da execução original**. O job de consolidação usa
um processo na partição `gorgonas`. Ele lê somente os resultados existentes,
confere seus checksums e preserva `plano.json`, `codigo/` e `tasks/`.
Não precisa acessar previsões, preços ou retornos de treino. Não use
`submeter` nem crie outro RUN_ID para essa exportação.

```bash
git pull --ff-only origin experiment/b3-loss-clean-v1
export RUN_ID=temporal_var_abertura_fechamento_s0_c0_cdi_v3_rent4
bash scripts/run_backtests_cpu.sh consolidar
```

A consolidação também é automática ao final das novas execuções. Copia uma
tarefa por vez para limitar memória. O log e `estado_execucao.json` informam
o número de linhas e tarefas incluídas no IC diário. IC válido de tarefa com
carteiras `PARCIAL` é preservado; arquivos corrompidos ou tarefas em erro não
entram na tabela. O job retorna código 1 se a execução estiver incompleta.
Depois de concluído, basta atualizar a pasta local `resumos/`; os Parquets
individuais já baixados não mudam.

## R local

Os comandos R desta seção são executados **na máquina local, na raiz do
repositório `analise_resultados`**. O arquivo
[`src/06_resultados_cluster.R`](https://github.com/igorviveirossouza/analise_resultados/blob/feat/base-mercado-backtest/src/06_resultados_cluster.R)
fica naquele repositório; não é instalado nem executado no TFB remoto.

Copie `plano.json`, `resumos/` e `tasks/`, preservando a estrutura relativa.
Não precisa trazer previsões/cache/temporários. Para diagnosticar falhas,
traga também `status/` e `logs/`. Dependências: arrow, jsonlite, dplyr e ggplot2.

```r
source("src/06_resultados_cluster.R")
x <- abrir_resultados_cluster("caminho/local/temporal_var_abertura_fechamento_s0_c0_cdi_v3")
tabela <- subset(x$metricas, treino==70 & k==5 & estrategia=="compra_positivos")
tabela[order(-tabela$retorno_acumulado),
       c("task_id", "modelo", "lookback", "pred_len", "retorno_acumulado",
         "volatilidade_anualizada", "sharpe_cdi", "max_drawdown")]

# IC natural com a mesma antecedencia h e amostra comum.
subset(x$ic_resumo, treino==70 & h==1 & amostra=="comum")
ic <- ler_tarefa_cluster(x, task_id=0, tabela="ic", h=1)

# Todas as configuracoes do mesmo corte/h; filtra no Arrow antes de carregar no R.
ic_tempo <- x$ic_diario |>
  dplyr::filter(treino == 70, h == 1, amostra_comum) |>
  dplyr::collect()
ggplot2::ggplot(ic_tempo, ggplot2::aes(data_alvo, spearman, group = task_id)) +
  ggplot2::geom_line() +
  ggplot2::facet_wrap(~ modelo + lookback + pred_len) +
  ggplot2::theme_minimal()

# Duas configuracoes do mesmo corte/K/estrategia.
ids <- head(unique(tabela$task_id), 2)
grafico_patrimonio_cluster(x, task_ids=ids, k=5, estrategia="compra_positivos")
```

Use `permitir_parcial=TRUE` deliberadamente para inspecionar o piloto. Por
padrão, o leitor bloqueia uma execução incompleta. Os scripts antigos 01–05
não passam a ler esses arquivos implicitamente.
`x$ic_diario` é uma consulta Arrow, sem carregar a série inteira na abertura.
Será `NULL` quando a cópia local ainda não tiver o novo consolidado; nesse
caso, a leitura por tarefa com `tabela="ic"` continua disponível.

## Testes de desenvolvimento

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_backtest_cpu.py
bash -n scripts/run_backtests_cpu.sh
```

Casos controlados cobrem IC com empates, datas/horizontes, compra/venda,
ausência de sinal, eventos, custos, insolvência, indisponibilidade de cotação,
CDI incorporado, Parquets, processos paralelos, retomada/corrupção e fontes incompatíveis.
Não substituem o piloto com a base completa. `--allow-local` serve apenas
a testes/fixtures; o launcher de produção não oferece essa opção.

Referências de API: [SciPy rankdata](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html)
e [Slurm sbatch](https://slurm.schedmd.com/sbatch.html).

# Mistura com Proximity por componente

O experimento compara, no mesmo protocolo, a mistura original, a mistura com
Proximity em todos os componentes ativos e a mistura com Proximity somente na
ListNet. A ListMLE permanece original na terceira variante.

Com os pesos default, a variante seletiva minimiza

\[
L=(1-\lambda)L_{MSE}+\lambda\left[
0.5L_{ListNet,Proximity}(A)+0.5L_{ListMLE}\right].
\]

O modo é global: ambas recebem o universo inteiro de ativos. O seletor altera
somente quais componentes recebem o kernel; não renormaliza seus coeficientes.
O parâmetro `NS_CROSS_SCALE`, default 1, multiplica todo o termo de ranking.

## Configuração default e quantidade

| Parâmetro | Default |
|---|---|
| Modelo | iTransformer |
| Corte / seed | 90 / 2028 |
| Lookbacks | 32, 256, 512 |
| Horizontes | 1, 25 |
| Alvo temporal | MSE |
| Lambdas | 0, 0.1, 0.5, 0.8, 1 |
| Pesos ListNet / ListMLE / ApproxNDCG | 0.5 / 0.5 / 0 |
| Modo | global |
| Kernel | Cauchy natural, A=0.01 |
| Seletores Proximity | todos; somente ListNet |
| Protocolo | origin_anchored_windows |
| Janelas supervisionadas | 1, 5, 10, 15, 20, 25, limitadas a H |
| K_loss | 1 |
| Momentum | somente benchmark externo; sem injeção |

São **78 treinamentos**:

- 3 lookbacks × 2 horizontes = 6 controles temporais, compartilhados;
- 3 lookbacks × 2 horizontes × 4 lambdas positivos × 3 variantes = 72.

As janelas supervisionadas pertencem ao mesmo ajuste. Não multiplicam os
treinamentos. Para retomar o protocolo de blocos, use
`NS_RANKING_PROTOCOL=blocks`; nesse caso `K_LOSS_VALUES` volta a controlar os
blocos e deve dividir H. O plano sempre informa a contagem da grade efetiva.

## Execução e parâmetros editáveis

Todos os exports com os defaults estão em
`scripts/config_rankmix_proximity_seletiva.defaults.sh`. Esse arquivo não
submete jobs. Edite os valores desejados antes de carregá-lo:

```bash
source scripts/config_rankmix_proximity_seletiva.defaults.sh
bash scripts/run_rankmix_proximity_seletiva.sh --plan
bash scripts/run_rankmix_proximity_seletiva.sh --train
```

Também é possível definir exports diretamente antes do launcher. Para exibir
todos os valores efetivos e os comandos correspondentes:

```bash
bash scripts/run_rankmix_proximity_seletiva.sh --print-exports
```

As principais opções específicas são:

```bash
export RANK_MIX_WEIGHT_SPECS='0.5 0.5 0'
export RANK_MIX_MODES='global'
export RANK_MIX_PROXIMITY_SPECS='all;listnet'
export NS_MASK_A_VALUES='0.01'
```

`all` aplica Proximity a todos os componentes ativos; `listnet` a aplica somente
à ListNet. Outros subconjuntos, como `listmle` ou `listnet approxndcg`, também
são aceitos. Grupos são separados por ponto e vírgula. A variante original é
incluída por `CROSS_LOSSES_VALUES='rankmix rankmix_proximity'`.

Os originais e os controles temporais não são repetidos para cada seletor ou
valor de A. Seletores equivalentes nos componentes ativos também são
deduplicados. Para estudar vários valores fixos de A, edite
`NS_MASK_A_VALUES`; isso gera treinamentos separados, não uma busca automática.

Se carregar o arquivo de defaults e depois mudar `EXPERIMENT_TAG`, ajuste também
os caminhos já exportados (`PRED_ROOT`, `NS_LOG_ROOT`, `NS_RESULT_ROOT` e
`BACKTEST_ROOT`). Uma alternativa simples é editar a tag no próprio arquivo
antes de executar `source`, pois os caminhos são derivados dela nesse momento.

## Identificação e compatibilidade

No JSON congelado, `loss_rank_mix_proximity_components=["listnet"]` identifica
a variante seletiva. Tarefas, auditoria e metadados registram a seleção. Seu
caminho inclui `proximity_on=listnet`, evitando colisão com previsões antigas.
Seleção ausente mantém o comportamento anterior: todos os componentes de uma
`rankmix_proximity` recebem o kernel. `rankmix` permanece original.

O analisador correspondente no repositório `analise_resultados` é
`src/analise_ic_rankmix_proximity_seletiva.R`. Ele compara as três variantes,
os controles temporais e o momentum, separando figuras por valor de A.

## Como selecionar A em um experimento futuro

Este launcher **não aprende A**. Cada treinamento recebe um A fixo.
Uma seleção externa adequada treinaria candidatos em um bloco passado e
compararia o IC de Spearman **sem máscara**, nas mesmas datas e janelas de uma
validação cronológica. O A escolhido seria então congelado antes de avaliar
o teste. Os alvos de treino não devem atravessar a fronteira da validação.

Isso impede que apenas reduzir os pesos do kernel seja considerado melhora
preditiva. Não garante que A pequeno seja ruim: ele pode reduzir fortemente a
contribuição da ListNet e deixar a ListMLE original dominar. Um controle
ListMLE original isolado ajuda a distinguir essa situação de um ganho efetivo
da combinação. Comparar valores de losses mascaradas diferentes não é um
critério adequado para escolher A.

# Momentum mais correção neural (opcional)

O modo `residual_output` mantém a arquitetura nativa e suas previsões diárias de
log-retornos. Um módulo compartilhado transforma essas previsões em uma correção
do ranking do momentum. O padrão é `off`; os experimentos antigos continuam
usando a loss, os scores e os caminhos anteriores.

## Definição

Para cada origem t, ativo i e bloco futuro B de K pregões:

```
m[t,i] = z_cross(sum(r[t-L+1:t,i]))
q[t,B,i] = retorno previsto acumulado no bloco B
delta[t,B,i] = gain * z_cross(q[t,B,i])
s[t,B,i] = m[t,i] + delta[t,B,i]
loss = (1-lambda)*loss_temporal + lambda*loss_cross(s, R_realizado)
       + eta*mean(delta**2)
```

`r` é o log-retorno original, antes do StandardScaler. `z_cross` usa média e
desvio-padrão populacional entre ativos, com piso `1e-8` no desvio. `q` respeita
`loss_score_kind`: soma para log-retorno, `expm1(soma)` para retorno simples.
A normalização cross-sectional já configurada na loss é aplicada depois desta
composição; os kernels continuam usando diferenças dos retornos realizados em
suas unidades financeiras.

`gain` é **um escalar treinável compartilhado**, inicializado em zero e salvo
no checkpoint da rede. Não é uma nova cabeça que produz N correções independentes.
A própria previsão da rede fornece a direção da correção. Portanto, inicialmente
o ranking final é exatamente o momentum. No primeiro passo, a componente de
ranking treina o ganho; o gradiente dessa componente chega ao backbone quando o
ganho se afasta de zero. A componente temporal continua treinando conforme lambda.

A penalização age na correção antes de qualquer normalização posterior. Para
cross-sections não constantes, ela equivale aproximadamente a `eta*gain**2`.
Não é weight decay nos pesos da rede. `eta=0` remove apenas a penalização, mantendo
o módulo ativo; `momentum_mode=off` desliga todo o módulo.

Este arranjo **não fornece momentum como variável de entrada à rede**. A rede
observa sua janela habitual de retornos. O momentum usa apenas o passado disponível
na origem e permanece o mesmo nos diferentes blocos futuros dessa origem.

## Terminal

O experimento dedicado usa `scripts/run_ns_momentum_kernels_isolado.sh`:

```bash
bash scripts/run_ns_momentum_kernels_isolado.sh --plan
bash scripts/run_ns_momentum_kernels_isolado.sh --dry-run
bash scripts/run_ns_momentum_kernels_isolado.sh --train
```

Defaults: Nonstationary, treino80, LB256/512, H1/25, K_loss1, lambdas0/.5/.8/1,
RankNet/RankNet proximity, 12 kernels naturais com A0.25, momentum ativo com L=LB
da rede e eta0.01. São 160 treinos: 4 controles temporais, 12 RankNet sem kernel
e 144 proximity. Eta0.01 é um ponto inicial editável, sem ajuste empírico.
O script delega ao launcher de kernels existente e respeita seus exports; os
defaults do launcher antigo permanecem iguais. `--help` lista as opções.

As opções valem para `run_ns_kloss_isolado.sh`, `run_ns_kernels_isolado.sh` e para
o launcher composto. Use uma nova tag para manter os exercícios separados:

```bash
export EXPERIMENT_TAG="isolados/ns_momentum_residual_t80"
export MOMENTUM_MODE="residual_output"
export MOMENTUM_LOOKBACK="512"
export MOMENTUM_ETA="0.01"

# Mantém os demais exports usuais de modelo, losses, kernel e recursos.
bash scripts/run_ns_kloss_isolado.sh --dry-run
bash scripts/run_ns_kloss_isolado.sh --train
```

`0.01` é somente exemplo de parametrização, sem alegação de valor ótimo.
Se `MOMENTUM_LOOKBACK` estiver vazio, cada tarefa usa o próprio lookback da rede.
Ele também pode ser maior: o loader preserva o passado adicional para o momentum,
mas entrega ao backbone apenas sua janela configurada. Isso elimina exemplos
iniciais sem histórico suficiente; o corte de teste e as origens exportadas
permanecem iguais. Todas as comparações exigem o mesmo universo de ativos.

O controle lambda=0 continua sendo o treino temporal original. O módulo é aplicado
somente às tarefas compostas com `0 < lambda <= 1`. Para retornar ao comportamento
anterior, use `export MOMENTUM_MODE="off"` em uma tag apropriada.

Nesta versão, o launcher aceita Nonstationary_Transformer, DUET, TimesNet e
FEDformer; exige entrada `log_return` e previsão da trajetória diária. Covariáveis
exógenas e extrapolação autorregressiva além do horizonte treinado são rejeitadas.
Losses que comparam magnitudes (`mse` e `pairwise_mse`, inclusive proximity) exigem
normalização cross-sectional `zscore` neste modo, para compatibilizar escalas.

## Arquivos, avaliação e retomada

Cada tarefa ativa recebe segmentos de caminho `momentum=residual_output`,
`momentum_lb=...` e `momentum_eta=...`, além da identificação usual do experimento.

* `part.parquet`: previsões diárias originais de retorno; sem alteração semântica.
* `momentum_state.json`: ganho do melhor checkpoint, L, eta, normalização e ativos.
* `ranking_scores.parquet`: score final adimensional, para todas as origens e
  todos os K de 1 até H, nas colunas `origin_step`, `k_eval` e ativos.
* `momentum_scores.json`: contrato e hashes dos três arquivos anteriores.

Não use o score final como retorno esperado, nem o acumule novamente. A exportação
recompõe o mesmo score do treino usando o estado salvo, as previsões e o histórico
original. A tarefa só é registrada como concluída depois da exportação e validação
dos scores. Retomadas conferem hashes e cobertura; arquivos parciais são rejeitados.

Os backtests compostos usam o score final para ordenar ativos e calcular IC
acumulado. Filtros de sinal positivo/negativo continuam usando retornos previstos
acumulados, pois o zero de um z-score não representa retorno esperado zero. O IC
diário por horizonte continua avaliando a previsão diária original; sua semântica
fica registrada nos metadados.

No repositório `analise_resultados`, os três scripts de IC reconhecem os arquivos
de score automaticamente. O benchmark usa o lookback efetivo do momentum, e a
nota de rodapé registra modo, L, eta e ganho. Baixe **toda a pasta de previsões**,
incluindo JSONs e os dois tipos de Parquet. `--ic_continuo FALSE` continua
espaçando igualmente as origens dos modelos e do benchmark. Raízes que misturam
desenhos incompatíveis são rejeitadas para evitar agrupamentos ambíguos.

Para comparar a contribuição da correção, use três referências: temporal puro,
composto anterior com módulo desligado e composto residual, sempre com o mesmo
momentum e as mesmas origens de avaliação. A ativação não garante melhora do IC
fora da amostra.

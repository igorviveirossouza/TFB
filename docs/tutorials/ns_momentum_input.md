# Experimento isolado: momentum como entrada

O experimento testa se fornecer explicitamente a tendência facilita a aprendizagem
do ranking. O score final vem das previsões da rede: não se soma um momentum fixo
à saída. Com LB512 e momentum512, a entrada adicional é uma transformação do mesmo
histórico, não informação de mercado nova. A projeção permite à rede aprender como
usá-la, sem garantir superação do benchmark.

## Desenho default: oito ajustes

| Entrada auxiliar | Loss | Sementes |
|---|---|---|
| Zeros (controle) | RankNet | 2026 e 2027 |
| Zeros (controle) | RankNet proximity | 2026 e 2027 |
| Momentum512 | RankNet | 2026 e 2027 |
| Momentum512 | RankNet proximity | 2026 e 2027 |

Constantes: Nonstationary_Transformer, treino80, LB512, H25, K_loss25, lambda1,
Cauchy natural A0.025 para a proximity. São 2 entradas × 2 losses × 2 seeds = **8
treinos**. Não há controle temporal adicional nem varredura de kernels.

Treino: batch16, LR inicial0.0001, lradj=type1 (regra nativa do NS), até100 épocas,
patience10, validação interna12.5% do bloco pré-teste. Estas escolhas não mudam entre
as quatro condições. Dois seeds são uma triagem de estabilidade, não evidência
estatística suficiente por si só.

## Onde entra o momentum

Para cada origem t e ativo i, o vetor auxiliar é

\[
 m_{t,i}=\frac{\sum_{j=0}^{L-1}r_{t-j,i}-\operatorname{mean}_{a}\sum_{j=0}^{L-1}r_{t-j,a}}
 {\max(\operatorname{sd}_{a,ddof=0}\sum_{j=0}^{L-1}r_{t-j,a},10^{-8})}.
\]

Os retornos são log-retornos observados, incluindo a origem e sem usar os alvos
futuros. Quando o normalizador externo do TFB está habilitado, sua transformação é
invertida usando somente as estatísticas ajustadas no bloco de treino. A mesma
função é usada no treino, na validação e na previsão.

Após a normalização temporal e o embedding nativo, o encoder recebe

\[
 E'_{t,u}=E_{t,u}+W_m m_t,\quad u=1,\ldots,LB,
\]

com uma projeção linear aprendida, sem bias, de número de ativos para d_model.
Trata-se de uma modificação na **representação de entrada do encoder**; a saída
continua sendo calculada pelo decoder da rede. O controle `zero` instancia o mesmo
módulo e recebe o vetor zero. `zero` e `feature` têm pesos iniciais iguais quando
usam a mesma semente. Ambos consomem o mesmo histórico; a projeção é criada depois
dos módulos nativos para preservar sua inicialização.

A implementação atual desta entrada suporta somente Nonstationary_Transformer.
Ela rejeita sua combinação com `momentum_mode=residual_output`, covariáveis exógenas
ou previsão recursiva além do H treinado. O modo global `off` mantém o comportamento
antigo sem criar a projeção; o controle experimental usa `zero` para manter a
estrutura e o consumo de RNG pareados.

Com lambda1 e K_loss25, a loss usa o ranking do retorno acumulado dos 25 passos
previstos. Os passos individuais não são treinados para calibrar a magnitude dos
retornos diários. Portanto, a comparação principal deve ser em K=25. Não há ganho
escalar residual nem penalidade eta neste desenho.

## Validação comum

Todas as condições usam `validation_mode=ordered_sample_mean`: origens em ordem,
sem embaralhar a validação, e média das losses ponderada pelo tamanho de cada
batch. Assim, o último batch incompleto não recebe peso excessivo. A escolha do
checkpoint segue a loss de validação de cada condição, sem consultar o teste.

Esta correção é metodológica; não é uma promessa de ganho de IC. Ela é opt-in para
preservar os experimentos antigos. Scripts antigos que não configuram
`VALIDATION_MODE` continuam com seu comportamento de validação anterior.

## Submeter

No clone TFB do servidor, após atualizar `experiment/b3-loss-clean-v1`:

```bash
export EXPERIMENT_TAG="momentum/input_ns_cauchy_a0.025_t80_h25_k25_v1"
export NS_PARTITION="gorgonas"
export NS_GRES=""

bash scripts/run_ns_momentum_input_isolado.sh --plan
bash scripts/run_ns_momentum_input_isolado.sh --train
```

`--plan` conta sem dados, Python ou Slurm; `--dry-run` valida e mostra os comandos
reais sem submeter. `--train` verifica todos os grupos antes da primeira submissão
e reutiliza as marcas individuais em uma repetição bem-sucedida. Uma falha parcial
de submissão continua exigindo conferir a marca e os jobs antes de retomar.

Por padrão os ramos são encadeados com `afterany`; isto não fixa o nó ou a GPU e não
condiciona um treinamento ao sucesso de outro. Fixe `NS_NODELIST` se desejar o
mesmo hardware. `INPUT_PARALLEL=1` permite independência entre todos os ramos:
nos defaults, até oito tarefas podem concorrer. `NS_MAX_CONCURRENT_JOBS` é um
limite **por array**, não entre arrays. Não representa um limite global de nós.

## Editar pelo terminal

Os parâmetros científicos usam o prefixo `INPUT_` para não herdar uma grade antiga
exportada no terminal. Por exemplo:

```bash
export INPUT_SEEDS="2026 2027"
export INPUT_LOOKBACKS_VALUES="512"
export INPUT_MOMENTUM_LOOKBACK="512"
export INPUT_HORIZON_VALUES="25"
export INPUT_K_LOSS_VALUES="25"
export INPUT_LAMBDAS="1"
export INPUT_CROSS_LOSSES_VALUES="ranknet ranknet_proximity"
export INPUT_KERNEL_SPEC="cauchy"
export INPUT_MASK_A="0.025"
export INPUT_MASK_MODE="natural"
```

`INPUT_MODEL`, `INPUT_TRAIN_VALUES`, `INPUT_MODES`, `INPUT_LRADJ` e
`INPUT_PARALLEL` também são editáveis, respeitando as restrições descritas em
`--help`. `INPUT_CROSS_LOSSES_VALUES` deve conter um par compatível base/proximity
(RankNet, hinge ou pairwise_mse). `INPUT_KERNEL_SPEC` recebe um único kernel; por
exemplo, `student 10`. A contagem acompanha a grade informada.

Os controles de recursos e treino mantêm os exports `NS_*` habituais, incluindo
batch, épocas, patience, LR, partição, GRES e nó. `NS_SEED` é substituído pelas
sementes de `INPUT_SEEDS` e `NS_LRADJ` por `INPUT_LRADJ`. Os exports antigos
`MODEL`, `LOOKBACKS_VALUES`, `CROSS_LOSSES_VALUES`, `MOMENTUM_MODE` e demais
parâmetros científicos do wrapper de kernels não alteram este experimento.

Nos launchers genéricos, a API nova é `MOMENTUM_INPUT_MODE=off|zero|feature`,
`MOMENTUM_INPUT_LOOKBACK` e `VALIDATION_MODE=ordered_sample_mean`; no wrapper de
kernels a validação usa `NS_VALIDATION_MODE`. As opções desligadas são omitidas
dos metadados antigos, e a entrada não produz sidecars do modelo residual.

## Saída e avaliação

As previsões ficam em:

```text
/snfs2/igor.viveiros/previsoes/<EXPERIMENT_TAG>/seed=<S>/entrada=<zero|feature>/composto/<sem_kernel|cauchy>/...
```

Cada tarefa inclui ainda `momentum_input=<modo>/momentum_input_lb=<L>` no caminho.
Os `run_config.json` registram a condição, a janela, a seed, a loss e a validação;
a proveniência inclui o código do módulo novo. O `part.parquet` contém as previsões
da rede. Não há `ranking_scores.parquet` residual a somar ou substituir.

Preserve a hierarquia completa e os metadados ao baixar. O atual
`analise_ic_ns_kernels.R` não aceita uma raiz combinando seeds e condições de
entrada. Não presuma que alimentar essa raiz produzirá a comparação pareada.
O IC não exige backtest; se quiser carteiras, depois dos treinos:

```bash
bash scripts/run_ns_momentum_input_isolado.sh --backtest submeter
```

A avaliação deve usar as mesmas origens e os mesmos ativos para as oito previsões
e o momentum512. Apresente IC mediano e médio, diferença em relação ao momentum e
diferença entrada feature − zero dentro de cada seed/loss. Compare também
proximity − RankNet dentro de cada seed/entrada. O critério principal é superar
o momentum com o **score final da rede**, com direção semelhante nas duas seeds;
melhora isolada da loss ou de um componente interno não substitui esse critério.
Use K25 como principal e reporte separadamente avaliação diária e a cada25 pregões,
sem escolher o modo ou a fase de amostragem olhando os resultados.

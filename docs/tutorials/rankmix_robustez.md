# Robustez das losses em diferentes cortes

O experimento compara melhorias sobre o temporal MSE e sobre o momentum externo,
com arquitetura fixa por modelo. O launcher é `scripts/run_rankmix_robustez.sh`.
A análise gráfica em R e a avaliação econômica serão preparadas posteriormente.

## Evidência que motivou a grade

Auditoria dos cinco arquivos `ic_diario(20261111-...).parquet` recebidos em
11/11/2026. `174106` e `174156` têm conteúdo idêntico e foram contados uma vez.
Todos esses dados novos usam seed 2026 e apenas Proximity, sem comparador original.

| Corte | A | Global ListNet/ListMLE 50/50, λ=1: vence MSE | Mediana do ganho de IC | Vence momentum |
|---|---:|---:|---:|---:|
|70%|0,05|21/21|+0,018309|11/21|
|80%|0,025|21/21|+0,027994|13/21|
|90%|0,05|20/21|+0,039250|18/21|

Cada conjunto de 21 células contém 3 LB × (H 1/K 1 + seis K de H 25). São diferenças
entre medianas calculadas nas mesmas origens; não são 21 treinos independentes.
Para H 25, λ = 1 superou MSE nas 54/54 células dos três cortes. Para H 1, venceu momentum
em apenas 3/9. No corte 80, global 50/50 venceu regional 50/50 em 75/84 comparações,
com mesmos A, λ, LB, H e K. Regional segue como hipótese, não como vencedor geral.

O refinamento Nonstationary anterior favoreceu global 25/75 em λ = 0,1 e regional
ListMLE/ApproxNDCG 75/25 ou 90/10. No refinamento fino, global 25/75, λ = 0,1,A 0,005,
LB 512,H 25,K 25 teve IC 0,139322 versus 0,130880 temporal e 0,131385 momentum.
Regional MLE/NDCG 90/10, λ = 0,5,A 0,01,LB 256,H 25,K 25 teve IC 0,115296 versus 0,052309
temporal e 0,047547 momentum. Essas configurações não transferiram claramente
ao iTransformer: os focos novos venceram MSE em 8/12 e 3/12 células, respectivamente.
Assim, a grade conserva candidatos e ablações, sem pressupor independência da arquitetura.

**Diferença de protocolo descoberta na auditoria:** os novos arquivos 70/80/90
registram `task_head_mode=shared` e `rank_windows=[1,5,10,15,20,25]` para H 25,
embora `k_loss=1`. Supervisionam seis retornos acumulados desde a origem.
O protocolo nativo antigo K_loss 1 supervisiona 25 retornos diários individuais.
Essa mudança também pode explicar ganhos; não se deve atribuí-los somente à loss.

O novo default preserva a matemática das janelas acumuladas, aplicada à saída
diária dos adaptadores nativos. Não cria outra cabeça nem injeta momentum.
O protocolo antigo continua disponível e permanece o default dos scripts antigos.

## Grade

| Fator | Default |
|---|---|
|Modelo|iTransformer; editável, inclusive vários modelos|
|Cortes|60,70,80,90%|
|Lookbacks|32,256,512|
|Horizontes|1,25|
|Seeds|2026,2027,2028|
|Temporal|MSE|
|Lambda principal|0;0,1;0,5;0,8;1|
|Pesos principais [ListNet,ListMLE,ApproxNDCG]|[0,5;0,5;0], [0,25;0,75;0], [0;0,75;0,25], [0;1;0]|
|Modos|global e regional|
|Regiões dos componentes|top,bottom,top; fração 1/3|
|Versões|original e Proximity Cauchy natural com A 0,01/0,025/0,05|
|Ablações adicionais|ListNet e ApproxNDCG isoladas, λ = 1, ambos modos e todas versões|
|Protocolo de ranking|origin_anchored_windows|
|Janelas conjuntas|1,5,10,15,20,25, limitadas a H|
|K_loss|1; não enumera as janelas conjuntas neste protocolo|

ListMLE isolada participa de todos os lambdas. As ablações adicionais só em λ = 1
identificam o efeito de misturar no ranking puro. Para avaliar essa atribuição
também nas tarefas híbridas, exporte `RANK_MIX_ABLATION_LAMBDAS='0.1 0.5 0.8 1'`:
a grade passa a 13.896 treinos com os demais defaults.

São 24 combinações corte/LB/H por seed. Com 3 seeds:

```
72 × [1 controle
      + 4 vetores × 2 modos × 4 lambdas positivos × (1 original + 3 As)
      + 2 ablações × 2 modos × 1 lambda × (1 original + 3 As)]
= 10.440 treinos
= 72 temporais + 6.912 mistos + 3.456 rankings puros.
```

Uma seed reduz para 3.480. Sem as duas ablações adicionais são 9.288 com 3 seeds.
Momentum exige zero treinos. Controles não se repetem por A, mistura ou modo;
original não se repete por A; λ = 1 não se repete por loss temporal.
Os totais são calculados novamente para qualquer export alterado.

## Objetivo e unidades

Após desfazer a normalização estimada no treino, para cada janela k válida:

\[
s_i^{(k)}=\exp\left(\sum_{h=1}^k\widehat r_{h,i}\right)-1,
\qquad y_i^{(k)}=\exp\left(\sum_{h=1}^kr_{h,i}\right)-1.
\]

\[
L=(1-\lambda)L_{temporal}+\lambda\,c\,
\frac{1}{|\mathcal K_H|}\sum_{k\in\mathcal K_H}
\sum_{j=1}^3 w_j\ell_j(s^{(k)},y^{(k)}),\qquad\sum_j w_j=1.
\]

`c=NS_CROSS_SCALE=1`. Cada janela recebe o mesmo peso. Global aplica cada
componente à lista inteira; regional aplica os componentes aos terços indicados,
definidos pelos retornos realizados **somente no treino**. A Proximity mantém
as adaptações listwise já implementadas e calcula seu kernel em diferenças de
retornos econômicos, antes da padronização transversal dos escores.
Pesos iguais das losses não implicam gradientes de mesma magnitude.

`NS_SCORE_KIND=log_return` utiliza a soma sem `exp()-1`; o default continua
`simple_return`. A mantém sua unidade de diferença de retorno acumulado e não
é automaticamente reescalado por k. Com norm=true, a restauração é obrigatória.

`NS_RANKING_PROTOCOL=blocks` recupera o protocolo antigo. Nesse caso,
`K_LOSS_VALUES` aceita vários valores (cada um define outro treino), K deve
dividir H, e `RANK_WINDOWS_VALUES` fica inativo. No protocolo acumulado,
`K_LOSS_VALUES=1` é obrigatório para impedir treinos duplicados; altere
`RANK_WINDOWS_VALUES` para mudar as janelas supervisionadas no mesmo ajuste.
O protocolo e suas janelas efetivas ficam registrados no JSON, catálogo e
`execution_metadata.json`; os caminhos incluem `protocolo=...`.

## Executar e editar

O arquivo `scripts/config_rankmix_robustez.defaults.sh` contém **todos os exports
públicos nos valores default**, sem submeter jobs. Ajuste os caminhos ao seu
checkout e edite os parâmetros desejados antes de carregá-lo:

```bash
source scripts/config_rankmix_robustez.defaults.sh
bash scripts/run_rankmix_robustez.sh --plan
bash scripts/run_rankmix_robustez.sh --train
```

Também é possível imprimir todos os valores efetivos, como um comando copiável:

```bash
bash scripts/run_rankmix_robustez.sh --print-exports
```

`--dry-run` valida os produtores sem submeter. `--list-branches` mostra os nomes
para `NS_ONLY_BRANCHES`. Os defaults permitem três tarefas simultâneas, com
arrays encadeados. `NS_PARALLEL_BRANCHES=1` permite arrays simultâneos e o limite
de três passa a valer **por array**. GRES vazio não solicita GPU. `NS_NODELIST`
pode restringir os nós; vazio deixa o Slurm escolher. Não há promessa de três
nós distintos: o agendador pode colocar tarefas no mesmo nó.

Para trocar modelo, use `MODEL='Nonstationary_Transformer'` ou
`MODEL='simpleFORMER'`; vários nomes multiplicam a grade. A capacidade permanece
fixa para cada modelo, configurada em `NS_MODEL_DEFAULTS_JSON`. Batch e treino
são controlados pelos exports específicos e não podem ser escondidos nesse JSON.

`--train` não executa backtest automaticamente. Há o comando opcional
`--backtest planejar|submeter|consolidar`, usando os backtests existentes.
`K_EVAL_VALUES` só controla essa avaliação; não adiciona treinos.
Marcas de submissão evitam reenviar ramos; não comprovam conclusão das tarefas.
Uma configuração alterada em um ramo já submetido exige uma nova tag.

## O que será possível concluir

O alvo principal é o ganho pareado sobre temporal MSE, depois sobre momentum,
com a mesma arquitetura, seed, corte, LB, H e origens. Repetir o protocolo
congelado em outras arquiteturas será necessário para alegar independência delas.
Original/Proximity e global/regional são comparados com pesos, A e λ controlados.
O potencial de contribuição está na combinação da adaptação Proximity listwise
com a especialização regional; este experimento não comprova prioridade na literatura.

Os cortes produzem testes aninhados, não quatro réplicas independentes.
Nos dados recebidos, o momentum LB 512/K 25 temIC mediano 0,006638 nas origens 70–80%,
0,193326 nas origens 80–90% e 0,035786 nas origens 90%–fim. O bloco 80–90% contribui
para os bons resultados dos testes iniciados em 70 e 80%. Essas divisões de origens
não purgam os 25 alvos nas fronteiras. A evidência indica dependência do período,
não demonstra que o resultado foi mero acaso.

Na análise posterior será necessário separar intervalos calendários e tratar a
dependência das janelas sobrepostas. Os testes já consultados são exploratórios;
novas seeds não criam um novo período de mercado. IC positivo ou superior ao
benchmark não demonstra alpha econômico: essa etapa exigirá carteiras com
custos, turnover e exposição ao mercado avaliados explicitamente.

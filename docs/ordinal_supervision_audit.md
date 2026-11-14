# Auditoria da supervisão ordinal

Esta auditoria descreve o comportamento efetivo do experimento. A correção do controle de submissões não altera as quatro losses nem transforma a parcela temporal em classificação.

## Alvos reais

Para cada origem e janela K, os log-retornos futuros verdadeiros são acumulados por ativo. No protocolo `origin_anchored_windows`, usam-se somente os primeiros K passos daquela origem. O universo é ordenado de forma decrescente e recebe

\[
c_i=1+\left\lfloor \frac{T\,\operatorname{rank0}_{\downarrow}(r_i)}{N}\right\rfloor,
\qquad c_i\in\{1,\ldots,T\}.
\]

Classe 1 é o estrato superior e T o inferior. Por exemplo, seis ativos com retornos acumulados `0.06, 0.04, 0.02, 0.01, -0.01, -0.03` e T=3 recebem `1, 1, 2, 2, 3, 3`. Os intervalos são grupos de ranking com tamanhos equilibrados, não intervalos de largura fixa na escala dos retornos. Empates seguem a ordem original dos canais.

O treinador conserva esses retornos reais em float64 antes da padronização e os entrega à loss separadamente das entradas da rede. `bucket_labels` recebe os alvos realizados; nunca usa a previsão para definir a classe verdadeira. Referências: `ts_benchmark/baselines/ordinal_losses.py`, `CompositeTradingLossV3._economic_ordinal_target`, `_forward_origin_windows` e `_cross_loss`.

## Previsão e função de perda

A rede mantém sua saída contínua `[batch, H, N]`. A loss acumula essa saída em um escore por ativo e janela, restaura a escala quando necessário e aplica a normalização transversal configurada. Denote esse escore por s: maior s significa ativo melhor colocado.

As classes inteiras são os alvos da parcela ordinal. As previsões entram na loss como escores ou probabilidades diferenciáveis. Transformá-las primeiro em rótulos inteiros com `argsort`, `argmax` ou arredondamento interromperia o gradiente usual entre a loss e a rede. A conversão final para uma classe pertence à inferência/avaliação, ou exige uma relaxação diferenciável explicitamente definida.

### Ordinal CE cumulativa

Para cada ativo e corte m=1,...,T-1:

\[
z_{im}=\mathbf{1}(c_i\le m),\qquad
q_{im}=\sigma((s_i-\theta_m)/\tau),\qquad
\theta_1>\cdots>\theta_{T-1}.
\]

A loss é a média das entropias cruzadas binárias entre z e q. Assim, para T=3, as classes 1, 2 e 3 são representadas por `(1,1)`, `(0,1)` e `(0,0)`. Os limiares são aprendidos e pertencem ao otimizador e ao checkpoint em memória. Esta é uma loss ordinal cumulativa; não é uma softmax nominal de T logits independentes.

Uma distribuição de classes pode ser recuperada por `p1=q1`, `pm=qm-q(m-1)` e `pT=1-q(T-1)`. O máximo dessas probabilidades e a regra mediana `1 + soma(qm < 0.5)` são decodificadores distintos.

### Bucket RankNet e Bucket Hinge

Somente pares com `c_i < c_j` participam. Eles devem satisfazer `s_i > s_j`:

\[
L_{\mathrm{RankNet}}=\operatorname{média}_{c_i<c_j}
\log(1+\exp(-(s_i-s_j)/\tau)),
\]
\[
L_{\mathrm{Hinge}}=\operatorname{média}_{c_i<c_j}
\max(0,m-(s_i-s_j)).
\]

A média é calculada por seção transversal e depois entre seções. Pares da mesma classe não são supervisionados. O sinal difere da formulação que usa `c_i > c_j` quando o escore cresce em direção às classes inferiores: aqui o escore cresce em direção aos melhores ativos, enquanto o índice da classe diminui.

### Bucket ListNet

\[
p_i=\operatorname{softmax}_i((T-c_i)/\tau_y),\qquad
q_i=\operatorname{softmax}_i(s_i/\tau_s),\qquad
L=-\sum_i p_i\log q_i.
\]

A softmax é feita entre os N ativos. Ativos da mesma classe têm a mesma relevância-alvo. Essas probabilidades não representam as T classes de cada ativo; são probabilidades listwise para ordenar os ativos.

## Parcela temporal e extremos de lambda

O comportamento atual é

\[
L=(1-\lambda)L_{\mathrm{temporal}}(\hat y,y)
  +\lambda\,\texttt{CROSS\_SCALE}\,L_{\mathrm{ordinal}}(s,c).
\]

Com `TEMPORAL_LOSSES_VALUES=mse`, y continua sendo o log-retorno futuro na representação normalizada usada pelo modelo. Não é o índice c da classe. Lambda zero mantém a regressão temporal; lambda um não calcula a parcela temporal. Regularizadores auxiliares próprios da arquitetura, como o balanceamento de especialistas do DUET, são preservados fora dessa combinação.

Portanto, a implementação é um experimento híbrido de regressão temporal e ranking supervisionado por classes. Ela não implementa classificação de classes em todas as parcelas. Substituir também o alvo temporal exige definir outro objetivo para lambda zero e sua representação de saída.

## Classes e métricas exportadas

O `part.parquet` contém a trajetória contínua prevista. No script `analise_resultados/src/analise_ic_ordinal.R`, as classes verdadeiras e previstas são construídas separadamente por quantis do ranking dos respectivos retornos/escores acumulados. Nenhum retorno futuro participa da classificação prevista. Os acertos avaliam recuperação dos grupos de ranking e podem ser comparados com o controle temporal e o momentum.

A análise atual usa essa regra comum também para Ordinal CE. Ela não usa os limiares aprendidos. Os limiares existem no checkpoint em memória, mas o fluxo ordinal não os exporta como um artefato persistente junto ao `part.parquet`. Consequentemente, esses arquivos não permitem reconstruir a classificação nativa por limiares da Ordinal CE.

Exemplo da diferença: com T=3, escores `1.2, 1.1, 1.0, -0.1, -0.2, -3.0` e cortes `0.8, -0.8`, o ranking em tercis produz `1,1,2,2,3,3`, enquanto a regra mediana pelos cortes produz `1,1,1,2,2,3`. Não se deve chamar a primeira métrica de acurácia do decodificador por limiares.

Para avaliar classificação nativa da Ordinal CE, é necessário persistir os limiares do checkpoint selecionado, reproduzir a mesma transformação dos escores na inferência e declarar o decodificador utilizado. RankNet, Hinge e ListNet não definem, por si, uma distribuição calibrada sobre T classes; sua conversão atual em grupos usa o ranking previsto.

## Verificação reproduzível

Os testes de `tests/test_ordinal_supervision_contract.py` confrontam a implementação com as seguintes propriedades: alvos discretos derivados dos realizados; invariância da parcela ordinal a mudanças dos retornos que preservam suas classes; gradientes pelos escores contínuos; codificação cumulativa; e dependência remanescente dos retornos na parcela temporal. A auditoria não alterou o objetivo de otimização.

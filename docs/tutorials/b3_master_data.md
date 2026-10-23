# Dados MASTER: validação e carregamento

Esta etapa consome a exportação de `analise_resultados/src/exportar_features_master.R`. Os dados reais permanecem em `dataset/master/features_master_v1/`, diretório ignorado pelo Git. O código abaixo valida os insumos e prepara as entradas; a arquitetura MASTER e sua inclusão nos lançadores de treinamento serão uma etapa posterior.

## Executar no cluster

Na raiz do checkout, com o ambiente Python do TFB ativado:

```sh
cd /sonic_home/igor.viveiros/src/TFB
python scripts/validate_master_data.py
```

Os caminhos padrão são `dataset/master/features_master_v1` e `dataset/forecasting/b3_daily_tfb_new.csv`. O segundo deve ser o **CSV original de log-retornos exportado pelo R**, com índices inteiros de pregão. O CSV com datas sintéticas preparado pelos lançadores do TFB é outro artefato e não deve ser passado como `--returns`. Ambos os caminhos podem ser informados explicitamente.

Por padrão, o comando exige 55 ativos e 3664 pregões. Uma nova amostra exige valores explícitos em `--expected-assets` e `--expected-dates`; o carregador nunca seleciona ativos automaticamente para alcançar essas dimensões.

Para também conferir as janelas e a preparação do treino, sem iniciar treinamento ou usar GPU:

```sh
python scripts/validate_master_data.py \
  --prepare-windows --train-percentage 80 --lookback 32 --horizon 1 \
  --report result/master/validacao_t80_lb32_h1.json
```

O relatório é opcional, inclui resultados da validação, ordem das variáveis, hashes, limites das janelas e estatísticas da normalização. Um relatório existente não é sobrescrito: escolha outro nome para uma nova execução. Erros retornam código de saída 1, sem modificar os dados.

## Contrato da exportação

O carregador confere os checksums dos arquivos transferidos, o manifesto da versão R, o calendário, o universo, as chaves e o dicionário. A identidade do CSV de retornos é vinculada ao `painel_md5` registrado no R; os valores também são comparados com `retorno_log` nos ativos. Linhas duplicadas, ausentes ou extras são erros, não motivos para fazer uma interseção silenciosa.

Os caminhos em `fontes.csv` registram as origens da estação de trabalho; não precisam existir no cluster. `manifesto.calendario_md5` identifica o calendário de entrada do R, enquanto o calendário transferido é validado pelo seu próprio checksum e conteúdo.

Uma única ordem de ativos é aplicada a alvos, características e flags. Por padrão, ela segue a primeira ocorrência de cada ticker no CSV original, como o leitor dos experimentos B3. A API também aceita `asset_order` explícito contendo exatamente o universo completo. A ordem física das linhas nos arquivos auxiliares não é usada para supor alinhamento.

## API e formatos

```python
from ts_benchmark.data.master import load_master_bundle
from ts_benchmark.data.master_dataset import prepare_master_datasets

bundle = load_master_bundle(
    "dataset/master/features_master_v1",
    "dataset/forecasting/b3_daily_tfb_new.csv",
    asset_blocks=("retornos", "ohlcv", "atividade"),
    use_market=True,
    expected_assets=55,
    expected_dates=3664,
)
datasets = prepare_master_datasets(
    bundle, lookback=32, horizon=1,
    train_percentage=80, train_ratio_in_tv=0.875,
)
sample = datasets.train[0]
```

| Campo | Forma | Significado |
|---|---|---|
| `bundle.targets` | `[T, N]` | Log-retornos econômicos, nas unidades originais. |
| `bundle.asset_values` | `[T, N, F]` | Features dos subblocos selecionados, ainda com ausências. |
| `bundle.market_values` | `[T, M]` | Features de mercado; `M=0` quando desativadas. |
| `sample["asset_values"]` | `[L, N, F]` | Lookback transformado com estatísticas do treino. |
| `sample["market_values"]` | `[L, M]` | Estado de mercado no mesmo lookback. |
| `sample["asset_observed"]`, `sample["market_observed"]` | Mesmas formas das respectivas entradas | Máscaras booleanas: `True` indica valor originalmente observado. |
| `sample["asset_flags"]`, `sample["market_flags"]` | `[L, N, A]`, `[L, B]` | Flags auditadas dos blocos selecionados, separadas das features. |
| `sample["target"]` | `[H, N]` | Trajetória futura de log-retornos, sem transformação. |
| `sample["origin_step"]`, `sample["target_steps"]` | Escalar, `[H]` | Índices de pregão da fonte R, base um. |

`N` é o número de ativos; `F/M` são quantidades de features; `A/B` são quantidades de flags; `L` é o lookback. O dicionário determina os tipos: chaves e flags não entram automaticamente como preditores. `insumos_ativos_auditados` é preservado para auditoria, incluindo o número de transações, e não é utilizado como matriz de features.

## Ausências e normalização

Para cada combinação ativo–feature, calculam-se média e desvio padrão populacional usando apenas os valores observados do **treino interno**. As features de mercado são tratadas individualmente pelo mesmo critério. Uma coluna constante usa escala 1. Valores ausentes recebem zero **depois** da centralização e divisão pela escala, equivalendo à média observada do treino; as máscaras preservam a distinção entre ausência e observação.

Se uma dimensão selecionada não tiver nenhuma observação no treino, a preparação falha indicando o problema. Não estima parâmetros com validação/teste, nem remove silenciosamente essa variável, ativo ou data. A inspeção inicial, sem `--prepare-windows`, continua disponível para examinar o relatório de ausências.

O futuro adaptador do modelo deve consumir explicitamente as máscaras e decidir como incorporar as flags. Este carregador não concatena os identificadores de auditoria às features. Os alvos permanecem inalterados; não se aplica nova diferença, soma temporal, ranking ou normalização aos log-retornos.

## Cortes e causalidade

O código reproduz os cortes dos experimentos B3:

```text
pretest = T * train_percentage // 100
fit = int(pretest * train_ratio_in_tv)
```

Com `j` como a posição base zero do primeiro alvo, uma amostra usa entradas `[j-L:j]` e alvos `[j:j+H]`. Os alvos ficam inteiramente no respectivo bloco:

- treino: `j = L, ..., fit-H`;
- validação: `j = fit, ..., pretest-H`;
- teste: `j = pretest, ..., T-H`.

A validação usa o final do treino como contexto, e o teste usa o histórico disponível anterior a cada origem, conforme a avaliação deslizante do TFB. Nenhuma feature do período alvo entra no lookback. Os pregões reais ficam separados das datas sintéticas criadas pelos leitores genéricos do TFB. A informação de uma linha só está disponível após seu fechamento.

## Ablações

Use `asset_blocks` e `use_market` na API ou, por exemplo:

```sh
python scripts/validate_master_data.py \
  --asset-blocks retornos --no-market \
  --prepare-windows --train-percentage 80 --lookback 32 --horizon 1
```

Para o mesmo corte, lookback e horizonte, todas as ablações usam os mesmos ativos e origens. Ausências não alteram esse conjunto. A validação da integridade cobre a exportação completa mesmo quando alguns blocos são desativados.

O estado de mercado desta versão é calculado sobre o universo congelado; não representa um índice externo. A política conservadora do R para preços sintéticos e recuperações no fracionário é preservada nas flags e máscaras.

## Testes

```sh
python -m pytest -q tests/test_master_data.py tests/test_b3_experiment_io.py
```

Os testes locais usam dados sintéticos e verificam alinhamento, integridade, CSV/Parquet, ausência de vazamento temporal e preservação das origens. A conferência dos 55 ativos reais deve ser executada no cluster pelos comandos acima.

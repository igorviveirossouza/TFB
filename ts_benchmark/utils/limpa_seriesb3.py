import pandas as pd


def filter_short_series_long(
    input_csv: str,
    output_csv: str,
    seq_len: int,
    pred_len: int,
    date_col: str = "date",
    value_col: str = "data",
    id_col: str = "cols",
):
    min_required = seq_len + pred_len

    df = pd.read_csv(input_csv)

    required_cols = {date_col, value_col, id_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    # conta observações válidas por série
    counts = (
        df.groupby(id_col)[value_col]
        .apply(lambda s: s.notna().sum())
        .reset_index(name="n_obs")
    )

    keep_ids = counts.loc[counts["n_obs"] >= min_required, id_col].tolist()
    drop_info = counts.loc[counts["n_obs"] < min_required].copy()

    df_filtered = df[df[id_col].isin(keep_ids)].copy()

    # opcional: ordenar
    df_filtered = df_filtered.sort_values([id_col, date_col]).reset_index(drop=True)

    df_filtered.to_csv(output_csv, index=False)

    print(f"seq_len={seq_len}, pred_len={pred_len}, min_required={min_required}")
    print(f"Séries mantidas: {len(keep_ids)}")
    print(f"Séries removidas: {len(drop_info)}")

    if not drop_info.empty:
        print("Séries removidas:")
        for _, row in drop_info.iterrows():
            print(f"  - {row[id_col]}: {int(row['n_obs'])} observações")


def build_aligned_panel_from_long(
    input_csv: str,
    output_long_csv: str,
    output_wide_csv: str,
    seq_len: int,
    pred_len: int,
    buffer: int = 32,
    date_col: str = "date",
    value_col: str = "data",
    id_col: str = "cols",
):
    """
    Lê um CSV long, pivota para wide, remove colunas problemáticas até que
    o painel alinhado tenha comprimento suficiente, e salva:
      1) um CSV long filtrado
      2) um CSV wide filtrado

    Critério de suficiência:
        n_linhas_alinhadas >= seq_len + pred_len + buffer
    """

    min_required = seq_len + pred_len + buffer

    df = pd.read_csv(input_csv)

    required_cols = {date_col, value_col, id_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    # ordenação básica
    df = df[[date_col, value_col, id_col]].copy()
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)

    # pivot inicial
    wide = df.pivot(index=date_col, columns=id_col, values=value_col).sort_index()

    # remove colunas totalmente vazias
    wide = wide.dropna(axis=1, how="all")

    removed_cols = []

    def aligned_length(w):
        # comprimento da interseção total entre colunas
        return w.dropna(axis=0, how="any").shape[0]

    current_len = aligned_length(wide)

    # remove iterativamente a pior coluna até atender o critério
    while current_len < min_required and wide.shape[1] > 1:
        # mede, para cada coluna, o ganho de comprimento alinhado se ela for removida
        best_col = None
        best_gain = -1
        best_new_len = current_len

        for col in wide.columns:
            trial = wide.drop(columns=[col])
            trial_len = aligned_length(trial)
            gain = trial_len - current_len

            if gain > best_gain:
                best_gain = gain
                best_col = col
                best_new_len = trial_len

        if best_col is None:
            break

        removed_cols.append((best_col, current_len, best_new_len))
        wide = wide.drop(columns=[best_col])
        current_len = best_new_len

    # painel alinhado final
    wide_aligned = wide.dropna(axis=0, how="any").copy()

    final_len = wide_aligned.shape[0]
    kept_cols = list(wide_aligned.columns)

    if final_len < min_required:
        raise ValueError(
            f"Mesmo após filtrar colunas, o painel alinhado ficou com {final_len} linhas, "
            f"mas o mínimo exigido é {min_required}."
        )

    # salva wide final
    wide_aligned = wide_aligned.reset_index()
    wide_aligned.to_csv(output_wide_csv, index=False)

    # reconstrói long filtrado a partir do wide alinhado
    long_filtered = wide_aligned.melt(
        id_vars=[date_col],
        var_name=id_col,
        value_name=value_col
    ).sort_values([id_col, date_col]).reset_index(drop=True)

    long_filtered.to_csv(output_long_csv, index=False)

    print("=" * 60)
    print("PAINEL FILTRADO COM SUCESSO")
    print(f"seq_len={seq_len}, pred_len={pred_len}, buffer={buffer}")
    print(f"mínimo exigido = {min_required}")
    print(f"comprimento final alinhado = {final_len}")
    print(f"número de séries mantidas = {len(kept_cols)}")
    print(f"número de séries removidas = {len(removed_cols)}")
    print("=" * 60)

    if removed_cols:
        print("Colunas removidas:")
        for col, old_len, new_len in removed_cols:
            print(f"  - {col}: alinhado {old_len} -> {new_len}")

    print("\nSéries mantidas:")
    for col in kept_cols:
        print(f"  - {col}")

    return {
        "final_aligned_length": final_len,
        "kept_cols": kept_cols,
        "removed_cols": removed_cols,
        "min_required": min_required,
    }            

#filter_short_series_long(
#    input_csv="/sonic_home/igor.viveiros/src/TFB/dataset/forecasting/b3_daily_tfb.csv",
#    output_csv="/sonic_home/igor.viveiros/src/TFB/dataset/forecasting/b3_daily_tfb_filter.csv",
#    seq_len=104,
#    pred_len=24,
#    date_col="date",
#    value_col="data",
#    id_col="cols",
#)


#build_aligned_panel_from_long(
#    input_csv="/sonic_home/igor.viveiros/src/TFB/dataset/forecasting/b3_daily_tfb.csv",
#    output_long_csv="/sonic_home/igor.viveiros/src/TFB/dataset/forecasting/b3_daily_tfb_long.csv",
#    output_wide_csv="/sonic_home/igor.viveiros/src/TFB/dataset/forecasting/b3_daily_tfb_wide.csv",
#    seq_len=64,
#    pred_len=24,
#    buffer=32,
#)
# tickers que você quer remover
input_csv="/sonic_home/igor.viveiros/src/TFB/dataset/forecasting/b3_daily_tfb.csv"
output_long_csv="/sonic_home/igor.viveiros/src/TFB/dataset/forecasting/b3_daily_tfb_filter.csv"

tickers_excluir = ["BRAV3", "MOTV3"]

# lê o csv
df = pd.read_csv(input_csv)

# remove as linhas cujos labels estão em `cols`
df_filtrado = df[~df["cols"].isin(tickers_excluir)].copy()

# salva
df_filtrado.to_csv(output_long_csv, index=False)

print("Tickers removidos:", tickers_excluir)
print("Novo shape:", df_filtrado.shape)
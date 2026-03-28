# %%

import pandas as pd

# ajuste o caminho se necessário
# fp = "dataset/forecasting/b3_daily_tfb.csv"

fp = "/media/igor/KINGSTON1/GitHub/FinancialMarket/candles-daily-jan2013-a-jun2025/b3_daily_tfb.csv"


df = pd.read_csv(fp)

# garante ordenação
df = df.sort_values(["cols", "date"]).copy()

# resumo por série
summary = (
    df.groupby("cols")
      .agg(
          n_obs=("data", "size"),
          start=("date", "min"),
          end=("date", "max")
      )
      .reset_index()
      .sort_values(["n_obs", "start", "cols"])
)

max_obs = summary["n_obs"].max()
# %%
# séries incompletas = têm menos observações que a maior série
incomplete = summary[summary["n_obs"] < max_obs].copy()

print(f"Total de séries: {summary.shape[0]}")
print(f"Maior comprimento: {max_obs}")
print(f"Séries incompletas: {incomplete.shape[0]}")

print("\nPrimeiras séries incompletas:")
print(incomplete.head(20))

# salva relatórios
summary.to_csv("summary_series_lengths.csv", index=False)
incomplete.to_csv("incomplete_series_lengths.csv", index=False)
# %%

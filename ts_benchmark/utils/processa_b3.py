# %%

import pandas as pd
import os

path = '/media/igor/KINGSTON1/GitHub/FinancialMarket/candles-15min-jan2013-a-jun2025/candles-15min-jan2013-a-jun2025.txt.txt'


colunas = [
    "tipo",
    "ticker",
    "codigo_numerico",
    "abertura",
    "fechamento",
    "maxima",
    "minima",
    "preco_medio",
    "negocios",
    "volume",
    "volume_financeiro",
    "datahora",
    "flag"
]

df = pd.read_csv(path,sep=";",header=None,names=colunas)

df["datahora"] = pd.to_datetime(df["datahora"], format="%Y%m%d%H%M", errors="coerce")

# %%

# Converter para float (evita overflow de int64)
numericas_float = [
    "abertura","fechamento","maxima","minima",
    "preco_medio","volume","volume_financeiro"
]

df[numericas_float] = df[numericas_float].apply(
    pd.to_numeric, errors="coerce"
)


# Converter colunas numéricas
numericas = [
    "abertura","fechamento","maxima","minima",
    "preco_medio","negocios","volume","volume_financeiro"
]

print(numericas)
## %%

# Converter inteiros menores separadamente
df["negocios"] = pd.to_numeric(df["negocios"], errors="coerce")
df["codigo_numerico"] = pd.to_numeric(df["codigo_numerico"], errors="coerce")
df["flag"] = pd.to_numeric(df["flag"], errors="coerce")

print(df.info())
print(df.head())



save_dir = os.path.join(os.getcwd(), "dataset", "forecasting")
os.makedirs(save_dir, exist_ok=True)


caminho = os.path.join(save_dir, "b3Candles_15min.parquet")

df.to_parquet(caminho, index=False)

# %%
# Remove o after market:
df = df[df['flag']==0]

df = df.sort_values(["ticker","datahora"])

#df.set_index(["datahora"], inplace=True)

df["dia.da.semana"] = df["datahora"].dt.day_name()

df = df[~df["dia.da.semana"].isin(["Saturday", "Sunday"])].copy()


df = df.set_index("datahora")


#df = df.between_time("10:00", "18:00")

# %%

#df = df[df["datahora"].dt.hour != 10 & df["datahora"].dt.minute != 00]

df_daily = (
    df
    .groupby("ticker")
    .resample(
        "1D",
        #label="right",
        #closed="right",
        #origin="start_day",
        #offset="10H"
     )
    .agg({
        "abertura": "first",
        "fechamento": "last",
        "maxima": "max",
        "minima": "min",
        "volume": "sum",
        "negocios": "sum",
        "volume_financeiro": "sum"
    })
    .dropna()
    .reset_index()
)

df_daily["datahora"] = df_daily["datahora"].dt.normalize()

global_dates = pd.DatetimeIndex(sorted(df_daily["datahora"].unique()))
tickers = sorted(df_daily["ticker"].unique())
full_index = pd.MultiIndex.from_product(
    [tickers, global_dates],
    names=["ticker", "datahora"]
)
# reindexa para explicitar os gaps
full = (
    df_daily.set_index(["ticker", "datahora"])
         .reindex(full_index)
         .reset_index()
)
# marca quais linhas eram faltantes
full["era_gap"] = full["fechamento"].isna()

# preenche fechamento com o último valor observado do próprio papel
full["fechamento"] = full.groupby("ticker")["fechamento"].ffill()

# opcional: remover início de cada série antes do primeiro valor observado
full = full[full["fechamento"].notna()].copy()

# %%
gaps_summary = (
    full.groupby("ticker", as_index=False)
        .agg(
            n_total=("datahora", "size"),
            n_gaps=("era_gap", "sum"),
            start=("datahora", "min"),
            end=("datahora", "max")
        )
        .sort_values(["n_gaps", "ticker"], ascending=[False, True])
)

print(gaps_summary.head(20))


# %%


OUTPUT_DIR = "/media/igor/KINGSTON1/GitHub/FinancialMarket/candles-daily-jan2013-a-jun2025"

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, "b3_daily.csv")

full.to_csv(output_path, index=False)

print("Arquivo salvo em:")
print(output_path)
print(df_daily.head())


# %%
selecao = ["ticker","datahora","fechamento"]

to_tfb = full[selecao].rename(
        columns={
            "ticker": "cols",
            "datahora": "date",
            "fechamento": "data"
        }
    )[['date', 'data', 'cols']]


#to_tfb = to_tfb.set_index("date")
#to_tfb["date"] = to_tfb.groupby("cols").cumcount() + 1



to_tfb = to_tfb.sort_values(["date", "cols"]).copy()

timeline = (
    pd.DataFrame({"date": sorted(to_tfb["date"].unique())})
    .reset_index()
    .rename(columns={"index": "time_idx"})
)

timeline["time_idx"] = timeline["time_idx"] + 1

to_tfb = to_tfb.merge(timeline, on="date", how="left")

to_tfb = to_tfb.sort_values(["cols","date"])

to_tfb = to_tfb[to_tfb["time_idx"] >= 235].copy() # Dados a parti de 2012-12-09

n_total = to_tfb["time_idx"].nunique()

completos = (
    to_tfb.groupby("cols")["time_idx"]
    .nunique()
    .loc[lambda s: s == n_total]
    .index
)

to_tfb = to_tfb[to_tfb["cols"].isin(completos)].copy()

# %%
to_tfb["date"] = to_tfb["time_idx"]


to_tfb = to_tfb.drop(columns=['time_idx'])

# %%
# Filtra as séries incompletas:

# resumo por série
summary = (
    to_tfb.groupby("cols")
      .agg(
          n_obs=("data", "size"),
          start=("date", "min"),
          end=("date", "max")
      )
      .reset_index()
      .sort_values(["n_obs", "start", "cols"])
)

max_obs = summary["n_obs"].max()

# séries incompletas = têm menos observações que a maior série
incomplete = summary[summary["n_obs"] < max_obs].copy()

print(f"Total de séries: {summary.shape[0]}")
print(f"Maior comprimento: {max_obs}")
print(f"Séries incompletas: {incomplete}")

output_path = os.path.join(OUTPUT_DIR, "summary.csv")

summary.to_csv(output_path, index=False)


# %%
#to_tfb = to_tfb[~to_tfb["cols"].isin(incomplete["cols"])].copy()

output_path = os.path.join(OUTPUT_DIR, "b3_daily_tfb.csv")

to_tfb.to_csv(output_path, index=False)

print("Arquivo salvo em:")
print(output_path)
print(to_tfb.head())

# %%

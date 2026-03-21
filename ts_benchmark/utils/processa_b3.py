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


# Remove o after market:
df = df[df['flag']==0]

df = df.sort_values(["ticker","datahora"])

#df.set_index(["datahora"], inplace=True)

df = df.set_index("datahora")

df = df.between_time("10:00", "18:00")

# %%

#df = df[df["datahora"].dt.hour != 10 & df["datahora"].dt.minute != 00]

df_daily = (
    df
    .groupby("ticker")
    .resample(
        "1D",
        label="left",
        closed="right",
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

# %%


OUTPUT_DIR = "/media/igor/KINGSTON1/GitHub/FinancialMarket/candles-daily-jan2013-a-jun2025"

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, "b3_daily.csv")

df_daily.to_csv(output_path, index=False)

print("Arquivo salvo em:")
print(output_path)
print(df_daily.head())
# %%


selecao = ["ticker","datahora","fechamento"]

to_tfb = df_daily[selecao].rename(
        columns={
            "ticker": "cols",
            "datahora": "date",
            "fechamento": "data"
        }
    )[['date', 'data', 'cols']]


#to_tfb = to_tfb.set_index("date")
#to_tfb["date"] = to_tfb.groupby("cols").cumcount() + 1

df_tfb = df_tfb.sort_values(["date", "cols"]).copy()

timeline = (
    pd.DataFrame({"date": sorted(df_tfb["date"].unique())})
    .reset_index()
    .rename(columns={"index": "time_idx"})
)

timeline["time_idx"] = timeline["time_idx"] + 1

df_tfb = df_tfb.merge(timeline, on="date", how="left")
output_path = os.path.join(OUTPUT_DIR, "b3_daily_tfb.csv")

to_tfb.to_csv(output_path, index=False)

print("Arquivo salvo em:")
print(output_path)
print(to_tfb.head())

# %%

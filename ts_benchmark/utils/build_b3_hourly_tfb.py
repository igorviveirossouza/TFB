# %%
import pandas as pd
import os

# ==============================
# CONFIGURAÇÕES
# ==============================

INPUT_PATH = "/media/igor/KINGSTON1/GitHub/FinancialMarket/candles-15min-jan2013-a-jun2025/candles-15min-jan2013-a-jun2025.txt.txt"

OUTPUT_DIR = os.path.join(os.getcwd(), "dataset", "forecasting")

TARGET_COLUMN = "fechamento"   # <-- ALTERE AQUI se quiser outra coluna
TIME_START = "10:00"
TIME_END   = "18:00"

# ==============================
# LEITURA
# ==============================

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

df = pd.read_csv(INPUT_PATH, sep=";", header=None, names=colunas)

df["datahora"] = pd.to_datetime(df["datahora"], format="%Y%m%d%H%M", errors="coerce")

# Converter numéricos
numericas = [
    "abertura","fechamento","maxima","minima",
    "preco_medio","negocios","volume","volume_financeiro"
]

df[numericas] = df[numericas].apply(pd.to_numeric, errors="coerce")
df["flag"] = pd.to_numeric(df["flag"], errors="coerce")

# ==============================
# FILTROS
# ==============================

# Remove after market
df = df[df["flag"] == 0]

# Ordenar
df = df.sort_values(["ticker", "datahora"])

# Index temporal
df = df.set_index("datahora")

# Manter horário do pregão
df = df.between_time(TIME_START, TIME_END)

# ==============================
# AGREGAÇÃO HORÁRIA
# ==============================

df_hour = (
    df
    .groupby("ticker")
    .resample("1H", closed="left", label="right")
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

# ==============================
# CONVERSÃO PARA FORMATO TFB
# ==============================

if TARGET_COLUMN not in df_hour.columns:
    raise ValueError(f"Coluna {TARGET_COLUMN} não encontrada no DataFrame.")

df_tfb = df_hour[["datahora", "ticker", TARGET_COLUMN]].copy()

df_tfb.rename(columns={
    "datahora": "date",
    TARGET_COLUMN: "data",
    "ticker": "cols"
}, inplace=True)

# ==============================
# SALVAR
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, f"b3_hourly_{TARGET_COLUMN}_tfb.csv")

df_tfb.to_csv(output_path, index=False)

print("Arquivo salvo em:")
print(output_path)
print(df_tfb.head())
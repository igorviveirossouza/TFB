from ts_benchmark.data.data_source import LocalForecastingDataSourceOHLCV

ds = LocalForecastingDataSourceOHLCV(
    aux_file_path="/sonic_home/igor.viveiros/src/b3_daily_tfb_ohlcv.csv"
)

series_name = "b3_daily_tfb.csv"  # ajuste para o nome real
ds.load_series_list([series_name])



target = ds.dataset.get_target_array(series_name)
aux = ds.dataset.get_aux_array(series_name)

print("target:", target.shape)
print("aux:", aux.shape)

item = ds.dataset.get_series(series_name)
print(type(item))
print(item.keys())
print(item["target"].shape)
print(item["aux"].keys())
for k, v in item["aux"].items():
    print(k, v.shape)

import pandas as pd

data = pd.read_csv('deliveries-2.csv', ',')

print(data.shape)

data.dropna(inplace=True)

print(data.shape())

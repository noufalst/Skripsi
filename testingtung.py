import pandas as pd

df = pd.read_csv(
    '3-24april.txt',
    sep='\t',
    skiprows=2,
    header=None,
    na_values=['---'],
    low_memory=False
)

print(f"Shape: {df.shape}")
print(f"Kolom: {df.shape[1]}")
print(df.iloc[:3, :24])  # 3 baris, 24 kolom pertama
import os
import pandas as pd
import numpy as np

def check_parquet_file(filepath):
    print(f"Проверка файла: {filepath}")
    df = pd.read_parquet(filepath)
    numeric_cols = df.select_dtypes(include='number').columns
    nan_count = df[numeric_cols].isnull().sum().sum()
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    print(f"  NaN в числовых столбцах: {nan_count}")
    print(f"  inf/-inf в числовых столбцах: {inf_count}")
    print(f"  shape: {df.shape}")
    print(f"  describe:\n{df[numeric_cols].describe()}")
    if nan_count == 0 and inf_count == 0:
        print("  ✅ OK: Нет NaN/inf в числовых данных!")
    else:
        print("  ❌ Есть NaN или inf!")
    print()

def main():
    output_dir = "data/output"
    files = [f for f in os.listdir(output_dir) if f.endswith('.parquet')]
    if not files:
        print("Нет parquet-файлов для проверки.")
        return
    for f in files:
        check_parquet_file(os.path.join(output_dir, f))

if __name__ == "__main__":
    main()

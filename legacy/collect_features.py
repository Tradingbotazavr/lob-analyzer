import asyncio
import logging
import os
import pandas as pd
from lob_analyzer.data.stream_merger import StreamMerger

logging.basicConfig(level=logging.INFO)

async def main():
    merger = StreamMerger(symbol="BTCUSDT", buffer_size=10000, save_interval_seconds=60, output_dir="data/output", merge_interval=0.01)
    logging.info("Запуск сбора признаков и таргета...")
    await merger.start()
    await asyncio.sleep(60)  # Для теста 1 минута, можно 600
    merger.stop()

    # Найти последний parquet-файл
    parquet_files = [f for f in os.listdir("data/output") if f.endswith(".parquet") and "btcusdt_features" in f]
    if not parquet_files:
        raise FileNotFoundError("Не найден parquet-файл после сохранения!")
    parquet_files.sort()
    parquet_path = os.path.join("data/output", parquet_files[-1])

    df = pd.read_parquet(parquet_path)
    # Удаление NaN/inf из всех числовых столбцов
    import numpy as np
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = np.nan_to_num(df[numeric_cols], nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Размерность DataFrame: {df.shape}")
    print("Первые строки:")
    print(df.head())

    # Проверка, что NaN нет после обработки
    assert not df[numeric_cols].isnull().any().any(), "Есть NaN в числовых данных после обработки!"
    assert df['target'].notnull().all(), "Есть пустые target!"

    exclude_cols = ["ts", "side", "reason", "direction", "has_fake_wall"]
    feature_cols = [col for col in df.columns if col not in exclude_cols + ["target"] and pd.api.types.is_numeric_dtype(df[col])]
    X = df[feature_cols]
    y = df["target"]

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Статистика по target:")
    print(y.describe())
    print("Гистограмма target:")
    print(y.value_counts(bins=10, sort=False))
    print("Статистика по ключевым признакам:")
    print(X.describe())

    # Визуализация (опционально)
    try:
        import matplotlib.pyplot as plt
        df['target'].value_counts(dropna=False).plot(kind='bar')
        plt.title('Target distribution')
        plt.show()
    except ImportError:
        print("matplotlib не установлен, визуализация пропущена.")

if __name__ == "__main__":
    asyncio.run(main())

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os

# Создаем директорию, если ее нет
os.makedirs("data/historical", exist_ok=True)

# --- Создаем данные для сделок ---
trades_data = {
    "ts": [100.1, 100.2, 100.8, 101.5, 102.3],
    "price": [100.0, 100.2, 100.1, 100.5, 100.8],
    "qty": [1.0, 0.5, 0.8, 2.0, 1.2],
    "side": ["buy", "sell", "buy", "buy", "sell"]
}
trades_df = pd.DataFrame(trades_data)

# --- Создаем данные для стаканов ---
# Для простоты, стаканы будут содержать только самые важные поля
orderbooks_data = {
    "ts": [100.0, 100.5, 101.0, 101.6, 102.0, 102.5],
    "mid_price": [99.95, 100.15, 100.3, 100.55, 100.75, 100.95],
    "bids": [
        [[99.9, 10], [99.8, 20]],
        [[100.1, 12], [100.0, 18]],
        [[100.2, 15], [100.1, 25]],
        [[100.5, 20], [100.4, 30]],
        [[100.7, 18], [100.6, 22]],
        [[100.9, 25], [100.8, 35]],
    ],
    "asks": [
        [[100.0, 15], [100.1, 25]],
        [[100.2, 18], [100.3, 22]],
        [[100.4, 20], [100.5, 30]],
        [[100.6, 22], [100.7, 28]],
        [[100.8, 30], [100.9, 40]],
        [[101.0, 28], [101.1, 38]],
    ]
}
orderbooks_df = pd.DataFrame(orderbooks_data)

# --- Сохраняем в Parquet ---
trades_output_path = "data/historical/sample_trades.parquet"
ob_output_path = "data/historical/sample_orderbooks.parquet"

trades_df.to_parquet(trades_output_path)
orderbooks_df.to_parquet(ob_output_path)

print(f"Исторические данные сохранены в:")
print(f"- {trades_output_path}")
print(f"- {ob_output_path}") 
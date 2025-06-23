import pyarrow.parquet as pq
import pandas as pd
import glob
import os

print("--- Запуск скрипта проверки данных ---")

# Автоматически находим последний созданный parquet-файл
try:
    list_of_files = glob.glob('data/merged/*.parquet')
    if not list_of_files:
        print("❌ Ошибка: Parquet-файлы не найдены в 'data/merged/'.")
        exit()
    latest_file = max(list_of_files, key=os.path.getctime)
    parquet_path = latest_file
    print(f"🔎 Проверяемый файл: {parquet_path}")
except Exception as e:
    print(f"❌ Ошибка при поиске файла: {e}")
    exit()

# Список всех ожидаемых признаков (в точности как в FeatureSnapshot)
expected_features = [
    'ts', 'price', 'qty', 'side', 'target', 'mid_price', 'avg_price',
    'trade_count', 'buy_volume', 'sell_volume', 'buy_sell_ratio',
    'direction', 'reason', 'bid_volume', 'ask_volume', 'bid_volume_near',
    'ask_volume_near', 'imbalance', 'buy_vol_speed', 'sell_vol_speed',
    'price_std', 'ma_volume', 'volatility', 'activity_spike',
    'bid_volume_pct_025', 'ask_volume_pct_025', 'imbalance_pct_025',
    'bid_density_lvl1_3', 'ask_density_lvl1_3', 'orderbook_entropy',
    'orderbook_roc', 'is_spoofing_like', 'has_fake_wall'
]

# Загрузка данных
try:
    df = pd.read_parquet(parquet_path)
    print(f"✅ Файл успешно загружен. {len(df)} строк, {len(df.columns)} колонок.")
except Exception as e:
    print(f"❌ Ошибка при чтении файла {parquet_path}: {e}")
    exit()

# Проверка наличия всех признаков
print("\n--- Проверка наличия признаков ---")
df_columns = set(df.columns)
missing_features = [f for f in expected_features if f not in df_columns]

if missing_features:
    print('❌ Отсутствуют признаки:', missing_features)
else:
    print('✅ Все ожидаемые признаки присутствуют')

print("\n--- Детальная информация по некоторым признакам ---")
# Проверка типов и базовых значений признаков
features_to_check = [
    'bid_volume_pct_025', 'orderbook_entropy', 'is_spoofing_like', 
    'has_fake_wall'
]

for f in features_to_check:
    if f in df_columns:
        print(f'Признак: {f}')
        print(f'  - Тип: {df[f].dtype}')
        print(f'  - Пропуски: {df[f].isnull().sum()} / {len(df)}')
        if pd.api.types.is_numeric_dtype(df[f]):
            print(f'  - Мин/Макс: {df[f].min()} / {df[f].max()}')
        else:
            print(f'  - Уникальные значения: {df[f].unique()[:5]}')
        print('------')

print("\n--- Скрипт завершен ---") 
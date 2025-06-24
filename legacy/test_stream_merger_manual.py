import pandas as pd
from collections import deque
from sortedcontainers import SortedDict
from lob_analyzer.data.stream_merger import StreamMerger
import asyncio

async def main():
    # 1. Создать объект StreamMerger
    merger = StreamMerger("btcusdt")

    # 2. Запустить сбор признаков (10 секунд для быстрого теста)
    await merger.start()
    await asyncio.sleep(10)
    await merger.stop()
    print(f"Feature snapshots collected: {len(merger.feature_snapshots)}")

    # 3. Сохранить результат в parquet
    parquet_file = "test_output.parquet"
    merger.save_parquet_file_sync()

    # 4. Открыть parquet-файл через pandas
    df = pd.read_parquet(parquet_file)
    print(df.info())
    print(df.head())

    # 5. Проверка на NaN и пустые target
    print("NaN in df:")
    print(df.isna().sum())
    print("Rows with null target:", df['target'].isna().sum())

    # 6. Ручной тест _process_trades_sync
    trades = [{"ts": 1.0, "price": 100, "qty": 1, "side": "buy"}]
    orderbooks = deque([{"ts": 1.0, "bids": [[99.5, 5]], "asks": [[100.5, 5]]}])
    price_hist = SortedDict({1.0: 100})
    results = merger._process_trades_sync(trades, orderbooks, price_hist)
    print("Manual _process_trades_sync result:")
    for r in results:
        print(r)

if __name__ == "__main__":
    asyncio.run(main())

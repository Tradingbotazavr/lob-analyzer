import asyncio
from lob_analyzer.data.stream_merger import StreamMerger

def generate_parquet():
    async def _run():
        merger = StreamMerger('BTCUSDT', buffer_size=10, save_interval_seconds=1, output_dir='data/output', merge_interval=0.01)
        await merger.start()
        merger.add_snapshot(bids=[[100.0, 1.0], [99.5, 2.0]], asks=[[102.0, 1.0], [102.5, 2.0]], ts=1.0)
        for i in range(3):
            merger.add_trade({'ts': 1.1 + i * 0.01, 'price': 101.0 + i, 'qty': 0.5, 'side': 'buy' if i % 2 == 0 else 'sell'})
        await asyncio.sleep(2)
        merger.stop()
    asyncio.run(_run())

if __name__ == "__main__":
    generate_parquet()

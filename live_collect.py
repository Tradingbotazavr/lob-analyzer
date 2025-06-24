import asyncio
import signal
import sys
import os
from loguru import logger
from lob_analyzer.data.stream_merger import StreamMerger
from lob_analyzer.collector.trades_stream import TradesStream
from lob_analyzer.collector.orderbook_realtime import RealTimeOrderBook

async def main():
    symbol = "BTCUSDT"
    merger = StreamMerger(symbol=symbol, output_dir="data/output", merge_interval=0.1)
    trades = TradesStream(symbol=symbol)
    orderbook = RealTimeOrderBook(symbol=symbol)

    # Коллбэки для передачи данных в merger
    trades.on_raw_trade(lambda trade: merger.add_trade(trade.to_dict()))
    orderbook.on_update(lambda ob: merger.add_snapshot(
        bids=ob.get_top_bids_asks()[0],
        asks=ob.get_top_bids_asks()[1],
        ts=ob.last_update_ts / 1000  # ms -> s
    ))

    # Запуск всех компонентов
    await merger.start()
    await asyncio.gather(trades.run(), orderbook.run())

    # Логирование объёма
    async def log_stats():
        while True:
            total = len(merger.feature_snapshots)
            with_target = sum(1 for s in merger.feature_snapshots if getattr(s, 'target', None) is not None)
            logger.info(f"Feature snapshots: {total}, with target: {with_target}")
            await asyncio.sleep(10)
    log_task = asyncio.create_task(log_stats())

    # Graceful shutdown
    def shutdown():
        logger.info("Shutting down...")
        log_task.cancel()
        asyncio.create_task(trades.stop())
        asyncio.create_task(orderbook.stop())
        merger.stop()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    try:
        await asyncio.Future()  # run forever
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

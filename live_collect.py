import asyncio
import signal
import sys
from loguru import logger
from lob_analyzer.data.stream_merger import StreamMerger
from lob_analyzer.collector.trades_stream import TradesStream
from lob_analyzer.collector.orderbook_realtime import RealTimeOrderBook

logger.remove()
logger.add(sys.stderr, level="DEBUG")

async def main():
    symbol = "BTCUSDT"
    merger = StreamMerger(
        symbol=symbol,
        output_dir="data/output",
        merge_interval=0.3,          # как было
        horizon_sec=10,              # быстрый target
        save_interval_seconds=30     # быстрое сохранение
    )
    trades = TradesStream(symbol=symbol)
    orderbook = RealTimeOrderBook(symbol=symbol)

    def trade_callback(trade):
        logger.info(f"Trade received: {trade.to_dict()}")
        merger.add_trade(trade.to_dict())

    trades.on_raw_trade(trade_callback)

    def orderbook_callback(ob):
        bids, asks = ob.get_top_bids_asks()
        logger.info(f"Orderbook update received: bids={bids}, asks={asks}")
        merger.add_snapshot(
            bids=bids,
            asks=asks,
            ts=ob.last_update_ts / 1000  # ms -> s
        )
        logger.debug("Snapshot added to merger")

    orderbook.on_update(orderbook_callback)

    async def log_stats():
        try:
            while True:
                total = len(merger.feature_snapshots)
                with_target = sum(1 for s in merger.feature_snapshots if getattr(s, 'target', None) is not None)
                logger.info(f"Feature snapshots: {total}, with target: {with_target}")
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            logger.info("log_stats task cancelled")
            raise

    tasks = [
        asyncio.create_task(merger.start()),
        asyncio.create_task(trades.run()),
        asyncio.create_task(orderbook.run()),
        asyncio.create_task(log_stats()),
    ]

    async def shutdown():
        logger.info("Shutting down...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await trades.stop()
        except Exception as e:
            logger.error(f"Error stopping trades: {e}")
        try:
            await orderbook.stop()
        except Exception as e:
            logger.error(f"Error stopping orderbook: {e}")
        try:
            await merger.stop()
        except Exception as e:
            logger.error(f"Error stopping merger: {e}")
        logger.info("Shutdown complete.")

    loop = asyncio.get_running_loop()

    if sys.platform == "win32":
        import threading

        def on_signal(signum, frame):
            logger.info(f"Signal {signum} received, scheduling shutdown")
            asyncio.get_event_loop().call_soon_threadsafe(lambda: asyncio.create_task(shutdown()))

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, on_signal)
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

    try:
        await asyncio.Future()  # run forever until cancelled
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

import sys
import os
import pytest

# Добавляем корень проекта в sys.path для корректного импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from lob_analyzer.data.stream_merger import StreamMerger

# 🧪 Тест на корректную генерацию фичей при валидных входных данных
def test_merger_with_valid_input():
    merger = StreamMerger(symbol="BTCUSDT")
    trade_batch = [{"ts": 1000.0, "price": 101.0, "qty": 0.5, "side": "buy"}]
    bids = [[100.0, 1.0], [99.5, 2.0]]
    asks = [[102.0, 1.0], [102.5, 2.0]]
    snapshot_ts = 999.0

    snapshot = merger.process_data_for_historical(trade_batch, bids, asks, snapshot_ts)

    assert snapshot is not None
    assert snapshot.ts == 1000.0
    assert snapshot.mid_price > 0
    assert snapshot.bid_volume > 0
    assert snapshot.ask_volume > 0

# 🧪 Тест на пустой trade_batch
def test_merger_empty_trade_batch_returns_none():
    merger = StreamMerger(symbol="BTCUSDT")
    snapshot = merger.process_data_for_historical([], [[100, 1]], [[102, 1]], 999.0)
    assert snapshot is None

# 🧪 Тест на пустые стаканы
def test_merger_empty_orderbook_returns_none():
    merger = StreamMerger(symbol="BTCUSDT")
    trade = [{"ts": 1000.0, "price": 101.0, "qty": 0.5, "side": "buy"}]
    snapshot = merger.process_data_for_historical(trade, [], [], 999.0)
    assert snapshot is None

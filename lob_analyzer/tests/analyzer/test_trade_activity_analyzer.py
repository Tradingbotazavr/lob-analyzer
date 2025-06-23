import pytest
import os
import tempfile
from lob_analyzer.analyzer.trade_activity_analyzer import TradeActivityAnalyzer

@pytest.fixture
def analyzer():
    return TradeActivityAnalyzer(
        volume_threshold=10.0,
        imbalance_ratio_high=2.0,
        imbalance_ratio_low=0.5,
        spike_multiplier=1.5,
        enable_logging=False,
        save_path=None
    )

def test_initialization(analyzer):
    assert analyzer.volume_threshold == 10.0
    assert analyzer.imbalance_ratio_high == 2.0
    assert analyzer.callbacks == []

def test_on_activity_callback(analyzer):
    called = []
    analyzer.on_activity(lambda d: called.append(d))
    event = {'ts': 1, 'buy_volume': 20, 'sell_volume': 1, 'buy_mean': 10, 'sell_mean': 5, 'trade_count': 5}
    analyzer.process_event(event)
    assert called

def test_no_spike_on_low_volume(analyzer):
    called = []
    analyzer.on_activity(lambda d: called.append(d))
    event = {'ts': 1, 'buy_volume': 2, 'sell_volume': 2, 'buy_mean': 10, 'sell_mean': 10, 'trade_count': 2}
    analyzer.process_event(event)
    assert not called

def test_spike_on_imbalance(analyzer):
    called = []
    analyzer.on_activity(lambda d: called.append(d))
    event = {'ts': 1, 'buy_volume': 20, 'sell_volume': 2, 'buy_mean': 10, 'sell_mean': 10, 'trade_count': 2}
    analyzer.process_event(event)
    assert called and called[0]['reason'] == 'imbalance'

def test_spike_on_volume_exceeded(analyzer):
    called = []
    analyzer.on_activity(lambda d: called.append(d))
    # Условие spike не выполняется (20 / 10 = 2 > 1.2), но volume_exceeded - да.
    # Изменим buy_mean и sell_mean, чтобы spike не срабатывал.
    # total_volume = 30, ma_volume = 28. Spike = 30/28 ~ 1.07 < 1.2.
    event = {'ts': 1, 'buy_volume': 15, 'sell_volume': 15, 'buy_mean': 14, 'sell_mean': 14, 'trade_count': 2}
    analyzer.process_event(event)
    assert called and called[0]['reason'] == 'volume_exceeded'

def test_spike_on_spike_vs_ma(analyzer):
    called = []
    analyzer.on_activity(lambda d: called.append(d))
    event = {'ts': 1, 'buy_volume': 10, 'sell_volume': 10, 'buy_mean': 2, 'sell_mean': 2, 'trade_count': 2}
    analyzer.process_event(event)
    assert called and called[0]['reason'] == 'spike'

def test_spoofing_and_fake_wall_flags(analyzer):
    called = []
    analyzer.on_activity(lambda d: called.append(d))
    # Первый вызов — нет prev_orderbook, второй — есть
    event1 = {'ts': 1, 'buy_volume': 20, 'sell_volume': 1, 'buy_mean': 10, 'sell_mean': 5, 'trade_count': 5, 'bids': [(100, 10)], 'asks': [(101, 2)]}
    event2 = {'ts': 2, 'buy_volume': 20, 'sell_volume': 1, 'buy_mean': 10, 'sell_mean': 5, 'trade_count': 5, 'bids': [(100, 3)], 'asks': [(101, 2)]}
    analyzer.process_event(event1)
    analyzer.process_event(event2)
    assert 'is_spoofing_like' in called[-1]
    assert 'has_fake_wall' in called[-1]

def test_save_to_jsonl(tmp_path):
    path = tmp_path / "spikes.jsonl"
    a = TradeActivityAnalyzer(save_path=str(path), enable_logging=False)
    event = {'ts': 1, 'buy_volume': 20, 'sell_volume': 1, 'buy_mean': 10, 'sell_mean': 5, 'trade_count': 5}
    a.process_event(event)
    with open(path, 'r') as f:
        lines = f.readlines()
    assert lines and 'reason' in lines[0] 
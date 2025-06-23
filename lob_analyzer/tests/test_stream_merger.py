import pytest
from lob_analyzer.data.stream_merger import StreamMerger

def test_stream_merger_feature_generation():
    merger = StreamMerger(symbol="BTCUSDT")

    # Пример одного трейда
    trade_batch = [{
        "ts": 1000.0,
        "price": 101.0,
        "qty": 0.5,
        "side": "buy"
    }]

    # Пример snapshot стакана
    bids = [[100.0, 1.0], [99.5, 2.0], [99.0, 1.5]]
    asks = [[102.0, 1.2], [102.5, 0.8], [103.0, 1.1]]
    snapshot_ts = 999.0

    snapshot = merger.process_data_for_historical(
        trade_batch=trade_batch,
        snapshot_bids=bids,
        snapshot_asks=asks,
        snapshot_ts=snapshot_ts
    )

    assert snapshot is not None, "Snapshot is None"
    
    required_fields = [
        "ts", "price", "qty", "side", "mid_price", "avg_price", "trade_count",
        "buy_volume", "sell_volume", "buy_sell_ratio", "direction", "reason",
        "bid_volume", "ask_volume", "imbalance", "buy_vol_speed", "sell_vol_speed",
        "ma_volume", "volatility", "bid_volume_pct_025", "ask_volume_pct_025",
        "imbalance_pct_025", "bid_density_lvl1_3", "ask_density_lvl1_3",
        "is_spoofing_like", "has_fake_wall", "orderbook_entropy",
        "orderbook_roc", "price_roc"
    ]

    snapshot_dict = snapshot.to_dict()
    for field in required_fields:
        assert field in snapshot_dict, f"Missing field: {field}"

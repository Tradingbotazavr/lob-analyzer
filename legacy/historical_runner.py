import pandas as pd
from loguru import logger
import os
from collections import deque
from lob_analyzer.features.snapshot import FeatureSnapshot
from lob_analyzer.utils.features_utils import (
    calculate_book_imbalance,
    calculate_ma_volume,
    calculate_price_volatility,
    calculate_trade_flow_metrics,
    get_volume_at_distance,
    get_volume_density,
    detect_spoofing_like_behavior,
    detect_fake_walls,
    get_orderbook_entropy,
    get_orderbook_roc,
    get_price_roc,
)

def _aggregate_trades(trade_batch):
    if not trade_batch:
        return None
    df = pd.DataFrame(trade_batch)
    buy_volume = df[df['side'] == 'buy']['qty'].sum()
    sell_volume = df[df['side'] == 'sell']['qty'].sum()
    qty_sum = df['qty'].sum()
    avg_price = (df['price'] * df['qty']).sum() / qty_sum if qty_sum > 0 else (df['price'].iloc[-1] if not df.empty else 0)
    return {
        'ts': df['ts'].max(), 'price': df['price'].iloc[-1], 'qty': qty_sum,
        'avg_price': avg_price, 'trade_count': len(df), 'buy_volume': buy_volume,
        'sell_volume': sell_volume, 'buy_sell_ratio': buy_volume / sell_volume if sell_volume > 0 else 1.0,
    }

def run_historical_data_processing():
    log_path = "logs/historical_runner.log"
    if os.path.exists(log_path):
        os.remove(log_path) # Clear log for clean run
    logger.add(log_path, rotation="10 MB", level="INFO", catch=True)
    logger.info("Starting self-contained historical data processing.")

    try:
        trades_df = pd.read_parquet('data/historical/sample_trades.parquet')
        snapshots_df = pd.read_parquet('data/historical/sample_orderbooks.parquet')
        logger.info(f"Loaded {len(trades_df)} trades and {len(snapshots_df)} snapshots.")
    except Exception as e:
        logger.error(f"Error loading data: {e}. Run 'create_historical_data.py' first.")
        return

    snapshot_map = {row['ts']: row for _, row in snapshots_df.iterrows()}
    all_features = []
    
    # Feature calculation state
    last_buy_vol, last_sell_vol, last_trade_ts = 0, 0, 0
    volume_history, price_history = deque(maxlen=50), deque(maxlen=50)
    last_imbalance = None

    for i, trade_row in trades_df.iterrows():
        trade_event = trade_row.to_dict()
        trade_ts = trade_event['ts']
        
        # Find the closest snapshot in time, must not be in the future
        available_snapshots = {t: s for t, s in snapshot_map.items() if t <= trade_ts}
        if not available_snapshots:
            logger.warning(f"No past snapshot found for trade at ts {trade_ts}. Skipping.")
            continue
        
        closest_snapshot_ts = max(available_snapshots.keys())

        snapshot_event = snapshot_map[closest_snapshot_ts]
        bids, asks = snapshot_event['bids'], snapshot_event['asks']
        
        agg_trade = _aggregate_trades([trade_event])
        if not agg_trade or not bids or not asks:
            continue

        mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0
        direction, buy_vol_speed, sell_vol_speed = calculate_trade_flow_metrics(
            agg_trade["buy_volume"], agg_trade["sell_volume"], last_buy_vol, last_sell_vol, last_trade_ts, agg_trade["ts"])
        
        volume_history.append(agg_trade['qty'])
        price_history.append(agg_trade['price'])
        
        imbalance, bid_vol, ask_vol = calculate_book_imbalance(bids, asks, 50)
        
        snapshot = FeatureSnapshot(
            ts=agg_trade["ts"], price=agg_trade["price"], qty=agg_trade["qty"], side=direction,
            mid_price=mid_price, avg_price=agg_trade["avg_price"], trade_count=agg_trade["trade_count"],
            buy_volume=agg_trade["buy_volume"], sell_volume=agg_trade["sell_volume"], buy_sell_ratio=agg_trade["buy_sell_ratio"],
            direction=direction, reason="historical", bid_volume=bid_vol, ask_volume=ask_vol, imbalance=imbalance,
            buy_vol_speed=buy_vol_speed, sell_vol_speed=sell_vol_speed, ma_volume=calculate_ma_volume(volume_history),
            volatility=calculate_price_volatility(price_history),
            bid_volume_pct_025=get_volume_at_distance(bids, asks, mid_price, 0.0025)[0],
            ask_volume_pct_025=get_volume_at_distance(bids, asks, mid_price, 0.0025)[1],
            imbalance_pct_025=get_volume_at_distance(bids, asks, mid_price, 0.0025)[2],
            bid_density_lvl1_3=get_volume_density(bids, asks, 3)[0], ask_density_lvl1_3=get_volume_density(bids, asks, 3)[1],
            is_spoofing_like=detect_spoofing_like_behavior(bids, asks), has_fake_wall=detect_fake_walls(bids, asks),
            orderbook_entropy=get_orderbook_entropy(bids, asks),
            orderbook_roc=get_orderbook_roc(imbalance, last_imbalance if last_imbalance is not None else imbalance),
            price_roc=get_price_roc(mid_price, price_history[-2] if len(price_history) > 1 else mid_price),
        )
        all_features.append(snapshot.to_dict())
        
        # Update state
        last_buy_vol, last_sell_vol, last_trade_ts = agg_trade["buy_volume"], agg_trade["sell_volume"], agg_trade["ts"]
        last_imbalance = imbalance

    if not all_features:
        logger.warning("No features generated.")
        return

    output_df = pd.DataFrame(all_features)
    output_filename = 'data/historical_features.parquet'
    try:
        output_df.to_parquet(output_filename)
        logger.info(f"SUCCESS: Saved {len(output_df)} records to {output_filename}.")
        print(f"SUCCESS: Saved {len(output_df)} records to {output_filename}.")
        print(output_df.head().to_string())
    except Exception as e:
        logger.error(f"Failed to save output file: {e}")
        print(f"ERROR: Failed to save output file: {e}")

if __name__ == "__main__":
    run_historical_data_processing()
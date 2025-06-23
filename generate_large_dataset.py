import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm

# --- Configuration ---
NUM_ROWS = 1000
START_PRICE = 60000
PRICE_VOLATILITY = 0.0001
VOLUME_MEAN = 0.5
VOLUME_STD = 0.3
TIMESTEP_MS = 100  # Milliseconds between events

# --- File Paths ---
DATA_DIR = "data/historical"
TRADES_PATH = os.path.join(DATA_DIR, "btcusdt_trades.csv")
OB_PATH = os.path.join(DATA_DIR, "btcusdt_ob.csv")


def generate_large_dataset():
    """
    Generates large synthetic datasets for trades and order books with a progress bar.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Generating {NUM_ROWS} synthetic data points...")
    
    start_ts = int(time.time() * 1000)
    
    trades_data = []
    ob_data = []
    
    current_price = START_PRICE
    
    # Use tqdm for a visual progress bar
    for i in tqdm(range(NUM_ROWS), desc="Generating data", unit="rows"):
        ts = start_ts + i * TIMESTEP_MS
        
        # --- Generate Trade Data ---
        price_change = np.random.normal(0, PRICE_VOLATILITY)
        current_price *= (1 + price_change)
        
        volume = max(0.001, np.random.normal(VOLUME_MEAN, VOLUME_STD))
        side = np.random.choice(['buy', 'sell'])
        
        trades_data.append([ts, current_price, volume, side])
        
        # --- Generate Order Book Snapshot ---
        # More realistic mid-price that jitters around the trade price
        mid_price = current_price * np.random.uniform(0.9999, 1.0001)
        spread = mid_price * 0.0001
        
        bids = []
        asks = []
        
        # Generate 10 levels for the order book
        for level in range(10):
            bid_price = mid_price - (level + 1) * spread * np.random.uniform(0.8, 1.2)
            ask_price = mid_price + (level + 1) * spread * np.random.uniform(0.8, 1.2)
            
            # Larger volumes for order book levels than for individual trades
            bid_volume = max(0.01, np.random.normal(VOLUME_MEAN * 10, VOLUME_STD * 5))
            ask_volume = max(0.01, np.random.normal(VOLUME_MEAN * 10, VOLUME_STD * 5))
            
            # Format to strings as in original data
            bids.append([f"{bid_price:.2f}", f"{bid_volume:.4f}"])
            asks.append([f"{ask_price:.2f}", f"{ask_volume:.4f}"])
            
        ob_data.append([ts, str(bids), str(asks)])

    # --- Create and Save DataFrames ---
    df_trades = pd.DataFrame(trades_data, columns=['ts', 'price', 'qty', 'side'])
    df_ob = pd.DataFrame(ob_data, columns=['ts', 'bids', 'asks'])
    
    df_trades.to_csv(TRADES_PATH, index=False)
    print(f"\n✅ Successfully generated and saved trade data to {TRADES_PATH}")
    
    df_ob.to_csv(OB_PATH, index=False)
    print(f"✅ Successfully generated and saved order book data to {OB_PATH}")


if __name__ == "__main__":
    generate_large_dataset()
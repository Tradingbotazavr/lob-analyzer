import unittest
from lob_analyzer.data.stream_merger import StreamMerger
from lob_analyzer.features.snapshot import FeatureSnapshot
import os
import pandas as pd
import shutil

class TestStreamMerger(unittest.TestCase):

    def setUp(self):
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.merger = StreamMerger(symbol="BTCUSDT", output_dir=self.output_dir)
        self.test_file = os.path.join(self.output_dir, "btcusdt_features.parquet")

    def tearDown(self):
        # безопасно удаляем даже непустой каталог
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_save_parquet_with_integrity_check(self):
        merger = StreamMerger(symbol="btcusdt", output_dir=self.output_dir, save_interval_seconds=1)
        merger.feature_snapshots.append(FeatureSnapshot(
            ts=1.0,
            price=100.0,
            qty=1.0,
            side="buy",
            target=1,
            mid_price=100.0,
            avg_price=100.0,
            trade_count=1,
            buy_volume=0.5,
            sell_volume=0.5,
            buy_sell_ratio=1.0,
            direction="buy",
            reason="test",
            bid_volume=100.0,
            ask_volume=100.0,
            imbalance=0.0,
            buy_vol_speed=0.0,
            sell_vol_speed=0.0,
            ma_volume=1.0,
            volatility=0.0,
            bid_volume_pct_025=50.0,
            ask_volume_pct_025=50.0,
            imbalance_pct_025=0.0,
            bid_density_lvl1_3=1.0,
            ask_density_lvl1_3=1.0,
            is_spoofing_like=0,
            has_fake_wall=0,
            orderbook_entropy=0.0,
            orderbook_roc=0.0,
            price_roc=0.0,
        ))
        merger.save_parquet_file_sync()

        # Логирование существования файла
        test_file = merger.output_filename
        print(f"Проверка существования файла: {test_file}, exists: {os.path.exists(test_file)}")
        self.assertTrue(os.path.exists(test_file))

if __name__ == "__main__":
    unittest.main()
import time
import os
import json
from collections import deque
from typing import Callable, Deque, Dict, List
from loguru import logger
import numpy as np

from lob_analyzer.collector.trades_stream import TradeEvent
from lob_analyzer.features.snapshot import FeatureSnapshot
from lob_analyzer.utils.features_utils import detect_spoofing, detect_fake_wall


class TradeActivityAnalyzer:
    def __init__(
        self,
        volume_threshold: float = 10.0,
        imbalance_ratio_high: float = 2.0,
        imbalance_ratio_low: float = 0.5,
        spike_multiplier: float = 1.2,
        enable_logging: bool = True,
        save_path: str = "activity_spikes.jsonl",
    ):
        self.volume_threshold = volume_threshold
        self.imbalance_ratio_high = imbalance_ratio_high
        self.imbalance_ratio_low = imbalance_ratio_low
        self.spike_multiplier = spike_multiplier
        self.enable_logging = enable_logging
        self.save_path = save_path

        self.callbacks: List[Callable[[Dict], None]] = []

        if save_path and not os.path.exists(self.save_path):
            with open(self.save_path, "w") as f:
                pass

        self._prev_orderbook = None  # Для spoofing/fake wall

    def on_activity(self, callback: Callable[[dict], None]):
        self.callbacks.append(callback)

    def process_event(self, event: Dict):
        """
        Processes an enriched data event from StreamMerger.
        The event is a dictionary-representation of a FeatureSnapshot.
        """
        # Логируем, что анализатор получил событие и какие признаки в нем есть
        entropy = event.get('orderbook_entropy')
        spoofing = event.get('is_spoofing_like')
        logger.info(
            f"ANALYZER | Received event with ts={event.get('ts'):.2f}, "
            f"entropy={entropy:.3f}" if entropy is not None else "entropy=None, "
            f"spoofing={spoofing}"
        )
        
        # Основная логика детекции аномалий остается, но использует готовые признаки
        buy_volume = event.get('buy_volume', 0)
        sell_volume = event.get('sell_volume', 0)
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return

        ma_volume = event.get('ma_volume', 0)
        
        spike = (ma_volume > 0 and total_volume / ma_volume > self.spike_multiplier)
        
        buy_sell_ratio = event.get('buy_sell_ratio', 1.0)
        imbalance = (
            buy_sell_ratio > self.imbalance_ratio_high or
            buy_sell_ratio < self.imbalance_ratio_low
        )
        volume_exceeded = total_volume > self.volume_threshold

        if spike or imbalance or volume_exceeded:
            # Приоритет: spike > imbalance > volume_exceeded
            if spike:
                event['reason'] = 'spike'
            elif imbalance:
                event['reason'] = 'imbalance'
            elif volume_exceeded:
                event['reason'] = 'volume_exceeded'
            # Все необходимые данные уже есть в event, который является словарем от FeatureSnapshot.
            # Просто передаем его дальше.
            spike_data = event
            
            if self.save_path:
                with open(self.save_path, "a") as f:
                    f.write(json.dumps(spike_data, ensure_ascii=False) + "\n")
            
            for cb in self.callbacks:
                cb(spike_data)

            if self.enable_logging:
                logger.warning(
                    f"ACTIVITY SPIKE: {event.get('reason')} | "
                    f"Direction: {event.get('direction')} | "
                    f"Total Vol: {total_volume:.2f}"
                )
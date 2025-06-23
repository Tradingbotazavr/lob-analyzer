from decimal import Decimal, getcontext
from typing import Dict, Tuple, Optional
import math

getcontext().prec = 12

def calculate_bid_ask_volume_pct(orderbook: dict, mid_price: float, pct: float = 0.0025) -> Tuple[float, float, float]:
    """
    Считает bid/ask объём в пределах ±pct от mid_price.
    Возвращает (bid_volume_pct, ask_volume_pct, imbalance_pct).
    """
    bid_vol = Decimal(0)
    ask_vol = Decimal(0)
    mid = Decimal(mid_price)
    for price, qty in orderbook.get('bids', []):
        if mid * (1 - Decimal(pct)) <= Decimal(price) <= mid:
            bid_vol += Decimal(qty)
    for price, qty in orderbook.get('asks', []):
        if mid <= Decimal(price) <= mid * (1 + Decimal(pct)):
            ask_vol += Decimal(qty)
    total = bid_vol + ask_vol
    imbalance = float((bid_vol - ask_vol) / total) if total > 0 else 0.0
    return float(bid_vol), float(ask_vol), imbalance

def calculate_density(orderbook: dict, levels: int = 3) -> Tuple[float, float]:
    """
    Вычисляет плотность bid и ask на уровнях 1–3.
    Возвращает (bid_density, ask_density).
    """
    bids = orderbook.get('bids', [])[:levels]
    asks = orderbook.get('asks', [])[:levels]
    def density(levels):
        if len(levels) < 2:
            return float(levels[0][1]) if levels else 0.0
        price_span = abs(Decimal(levels[0][0]) - Decimal(levels[-1][0]))
        vol_sum = sum(Decimal(qty) for _, qty in levels)
        return float(vol_sum / price_span) if price_span > 0 else float(vol_sum)
    return density(bids), density(asks)

def calculate_orderbook_entropy(orderbook: dict, depth: int = 5) -> float:
    """
    Считает энтропию объёмов в первых depth уровнях стакана.
    """
    bids = orderbook.get('bids', [])[:depth]
    asks = orderbook.get('asks', [])[:depth]
    vols = [float(qty) for _, qty in bids + asks]
    total = sum(vols)
    if total == 0:
        return 0.0
    probs = [v / total for v in vols if v > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return float(entropy)

def calculate_orderbook_roc(prev_orderbook: dict, curr_orderbook: dict) -> float:
    """
    Считает скорость изменения объёмов стакана (Rate of Change).
    """
    prev_vol = sum(float(qty) for _, qty in prev_orderbook.get('bids', [])[:5] + prev_orderbook.get('asks', [])[:5])
    curr_vol = sum(float(qty) for _, qty in curr_orderbook.get('bids', [])[:5] + curr_orderbook.get('asks', [])[:5])
    if prev_vol == 0:
        return 0.0
    return float((curr_vol - prev_vol) / prev_vol)

def detect_spoofing(prev_orderbook: dict, curr_orderbook: dict) -> int:
    """
    Эвристика spoofing: резкий рост и исчезновение крупных объёмов.
    Возвращает 0 или 1.
    """
    # Пример: если на стороне bids/asks на 1-3 уровнях объём вырос >50% и затем упал >40% за короткое время
    for side in ['bids', 'asks']:
        prev = sum(float(qty) for _, qty in prev_orderbook.get(side, [])[:3])
        curr = sum(float(qty) for _, qty in curr_orderbook.get(side, [])[:3])
        if prev > 0 and curr < prev * 0.6 and prev > curr * 1.5:
            return 1
    return 0

def detect_fake_wall(orderbook: dict, threshold_pct: float = 0.1) -> int:
    """
    Эвристика fake wall: крупная заявка > threshold_pct от общей стороны.
    Возвращает 0 или 1.
    """
    for side in ['bids', 'asks']:
        levels = orderbook.get(side, [])[:5]
        total = sum(float(qty) for _, qty in levels)
        if total == 0:
            continue
        max_level = max(float(qty) for _, qty in levels)
        if max_level > total * threshold_pct:
            return 1
    return 0

def calculate_total_volume(levels: list) -> float:
    """Просто суммирует объемы на всех предоставленных уровнях."""
    return sum(float(qty) for _, qty in levels)

def calculate_price_roc(prev_price: Optional[float], curr_price: Optional[float]) -> float:
    """Считает скорость изменения цены (Price Rate of Change)."""
    if prev_price is None or curr_price is None or prev_price == 0:
        return 0.0
    return (curr_price - prev_price) / prev_price 
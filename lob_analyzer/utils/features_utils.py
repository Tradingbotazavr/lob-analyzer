from typing import Dict, Tuple, Optional, List, Deque
import math
import statistics
import numpy as np
import pandas as pd

def safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return default
        return float(val)
    except Exception:
        return default

def safe_int(val, default=0):
    try:
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return default
        return int(val)
    except Exception:
        return default

def calculate_bid_ask_volume_pct(orderbook: dict, mid_price: float, pct: float = 0.0025) -> Tuple[float, float, float]:
    """
    Считает bid/ask объём в пределах ±pct от mid_price.
    Возвращает (bid_volume_pct, ask_volume_pct, imbalance_pct).
    """
    bid_vol = 0.0
    ask_vol = 0.0
    lower = mid_price * (1 - pct)
    upper = mid_price * (1 + pct)
    for price, qty in orderbook.get('bids', []):
        price = float(price)
        qty = float(qty)
        if abs(price) > 1e12 or abs(qty) > 1e12:
            continue
        if lower <= price <= mid_price:
            bid_vol += qty
    for price, qty in orderbook.get('asks', []):
        price = float(price)
        qty = float(qty)
        if abs(price) > 1e12 or abs(qty) > 1e12:
            continue
        if mid_price <= price <= upper:
            ask_vol += qty
    total = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
    return bid_vol, ask_vol, imbalance

def calculate_density(orderbook: dict, levels: int = 3) -> Tuple[float, float]:
    """
    Возвращает средневзвешенную цену по объёму для bid и ask на levels уровнях.
    """
    bids = orderbook.get('bids', [])[:levels]
    asks = orderbook.get('asks', [])[:levels]
    def weighted_avg(levels):
        if not levels:
            return 0.0
        total_vol = sum(float(qty) for price, qty in levels if abs(qty) < 1e12)
        if total_vol == 0:
            return 0.0
        result = sum(float(price) * float(qty) for price, qty in levels if abs(price) < 1e12 and abs(qty) < 1e12) / total_vol
        return safe_float(result)
    return weighted_avg(bids), weighted_avg(asks)

def calculate_orderbook_entropy(orderbook: dict, levels: int = 3) -> float:
    """
    Считает энтропию объёмов в первых levels уровнях стакана.
    """
    bids = orderbook.get('bids', [])[:levels]
    asks = orderbook.get('asks', [])[:levels]
    vols = [float(qty) for _, qty in bids + asks]
    total = sum(vols)
    if total == 0:
        return 0.0
    probs = [v / total for v in vols if v > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return safe_float(entropy)

def get_orderbook_entropy(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], depth: int = 5) -> float:
    """
    Считает энтропию объёмов в первых depth уровнях стакана (bids и asks).
    """
    vols = [float(qty) for _, qty in bids[:depth] + asks[:depth]]
    total = sum(vols)
    if total == 0:
        return 0.0
    probs = [v / total for v in vols if v > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return safe_float(entropy)

def calculate_orderbook_roc(prev_orderbook: dict, curr_orderbook: dict) -> float:
    """
    Считает скорость изменения объёмов стакана (Rate of Change) по всем уровням.
    """
    prev_vol = sum(float(qty) for _, qty in prev_orderbook.get('bids', []) + prev_orderbook.get('asks', []))
    curr_vol = sum(float(qty) for _, qty in curr_orderbook.get('bids', []) + curr_orderbook.get('asks', []))
    if prev_vol == 0:
        return 1.0 if curr_vol > 0 else 0.0
    return float((curr_vol - prev_vol) / prev_vol)

def detect_spoofing(prev_orderbook: dict, curr_orderbook: dict, threshold: float = 50):
    """
    Если крупный ордер появился — 1, исчез — -1, иначе 0.
    """
    for side in ['bids', 'asks']:
        prev = sum(float(qty) for _, qty in prev_orderbook.get(side, [])[:3])
        curr = sum(float(qty) for _, qty in curr_orderbook.get(side, [])[:3])
        if curr - prev > threshold:
            return 1
        if prev - curr > threshold:
            return -1
    return 0

def detect_fake_wall(orderbook: dict, wall_size_threshold: float = 100, distance_pct: float = 0.05):
    """
    Проверяет наличие крупной заявки (fake wall) на расстоянии от mid_price.
    """
    for side in ['bids', 'asks']:
        levels = orderbook.get(side, [])[:5]
        total = sum(float(qty) for _, qty in levels)
        if total == 0:
            continue
        max_level = max(float(qty) for _, qty in levels)
        if max_level > wall_size_threshold:
            return 1
    return 0

def detect_fake_walls(orderbook, threshold_volume: float = 1000) -> list:
    """
    Обнаруживает фейковые стенки (ложные большие заявки) в стакане.

    Args:
        orderbook (dict): Структура стакана с ключами 'bids' и 'asks', 
                          где каждый — список [price, volume].
        threshold_volume (float): Порог объёма, выше которого заявка считается стенкой.

    Returns:
        list: Список уровней (цена), где обнаружены фейковые стенки.
    """
    fake_walls = []

    # Проверяем bids
    for price, volume in orderbook.get('bids', []):
        if volume >= threshold_volume:
            fake_walls.append(('bid', price, volume))

    # Проверяем asks
    for price, volume in orderbook.get('asks', []):
        if volume >= threshold_volume:
            fake_walls.append(('ask', price, volume))

    return fake_walls

def calculate_total_volume(levels: list) -> float:
    """Просто суммирует объемы на всех предоставленных уровнях."""
    return safe_float(sum(float(qty) for _, qty in levels))

def calculate_price_roc(prev_price: Optional[float], curr_price: Optional[float]) -> float:
    """Считает скорость изменения цены (Price Rate of Change)."""
    if prev_price is None or curr_price is None or prev_price == 0:
        return 0.0
    return safe_float((curr_price - prev_price) / prev_price)

def get_price_roc(current_price: float, previous_price: float) -> float:
    """
    Считает скорость изменения цены (Rate of Change, ROC) между текущей и предыдущей ценой.
    """
    if previous_price == 0:
        return 0.0
    return safe_float((current_price - previous_price) / previous_price)

def calculate_book_imbalance(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    depth: int = 50
) -> Tuple[float, float, float]:
    """
    Вычисляет дисбаланс объёмов между bids и asks на глубине depth.
    Возвращает (imbalance, bid_volume, ask_volume).

    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume), если сумма > 0, иначе 0.
    """
    bid_volume = sum(qty for _, qty in bids[:depth])
    ask_volume = sum(qty for _, qty in asks[:depth])
    total = bid_volume + ask_volume
    imbalance = (bid_volume - ask_volume) / total if total > 0 else 0.0
    return safe_float(imbalance), safe_float(bid_volume), safe_float(ask_volume)

def calculate_ma_volume(volume_history: Deque[float]) -> float:
    """
    Рассчитывает скользящее среднее объёма за заданный период,
    где volume_history — коллекция объёмов последних N событий.
    """
    if not volume_history:
        return 0.0
    result = sum(volume_history) / len(volume_history)
    return safe_float(result)

def calculate_price_volatility(price_history: Deque[float]) -> float:
    """
    Вычисляет волатильность как стандартное отклонение последних цен.
    Добавлена фильтрация невалидных значений и логирование ошибок.
    """
    import logging
    filtered = [float(x) for x in price_history if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    if len(filtered) < 2:
        return 0.0
    try:
        result = statistics.stdev(filtered)
        return safe_float(result)
    except Exception as e:
        logging.warning(f"Ошибка при расчёте волатильности: {e}")
        return 0.0

def calculate_price_std(price_history: Deque[float]) -> float:
    """
    Возвращает стандартное отклонение цен, всегда float, без NaN/inf.
    """
    filtered = [float(x) for x in price_history if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    if len(filtered) < 2:
        return 0.0
    try:
        result = statistics.stdev(filtered)
        return safe_float(result)
    except Exception:
        return 0.0

def calculate_activity_spike(*args, **kwargs) -> float:
    """
    Заглушка для activity_spike: всегда возвращает 0.0 если не вычисляется.
    """
    try:
        return 0.0
    except Exception:
        return 0.0

def calculate_trade_flow_metrics(trades: pd.DataFrame) -> dict:
    """
    Вычисляет базовые метрики торгового потока.

    trades: pd.DataFrame с колонками ['price', 'qty', 'side', 'ts']
        side: 'buy' или 'sell'
    """
    if trades.empty:
        return {
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'volume_ratio': 0.0,
            'trade_count': 0,
            'buy_count': 0,
            'sell_count': 0,
            'imbalance': 0.0,
        }

    buy_trades = trades[trades['side'] == 'buy']
    sell_trades = trades[trades['side'] == 'sell']

    buy_volume = buy_trades['qty'].sum()
    sell_volume = sell_trades['qty'].sum()
    total_volume = buy_volume + sell_volume

    buy_count = len(buy_trades)
    sell_count = len(sell_trades)
    total_count = buy_count + sell_count

    volume_ratio = buy_volume / sell_volume if sell_volume > 0 else np.inf
    imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0

    return {
        'buy_volume': float(buy_volume),
        'sell_volume': float(sell_volume),
        'volume_ratio': float(volume_ratio),
        'trade_count': total_count,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'imbalance': float(imbalance),
    }

def calculate_trade_flow_metrics(buy_volume, sell_volume, last_buy_vol, last_sell_vol, last_ts, current_ts):
    """
    Рассчитывает направление торгового потока и скорость изменения объёмов покупок и продаж.

    Аргументы:
        buy_volume (float): текущий объём покупок
        sell_volume (float): текущий объём продаж
        last_buy_vol (float): объём покупок на предыдущем шаге
        last_sell_vol (float): объём продаж на предыдущем шаге
        last_ts (float): временная метка предыдущего шага
        current_ts (float): текущая временная метка

    Возвращает:
        tuple: (direction:str, buy_vol_speed:float, sell_vol_speed:float)
    """
    dt = current_ts - last_ts if current_ts > last_ts else 1
    buy_vol_speed = (buy_volume - last_buy_vol) / dt
    sell_vol_speed = (sell_volume - last_sell_vol) / dt
    direction = "buy" if buy_volume > sell_volume else "sell"
    return direction, buy_vol_speed, sell_vol_speed


def calculate_volume_acceleration(prev_speed: float, curr_speed: float, dt: float) -> float:
    """
    Рассчитывает ускорение объёма как изменение скорости за единицу времени.

    Аргументы:
        prev_speed (float): предыдущая скорость объёма
        curr_speed (float): текущая скорость объёма
        dt (float): промежуток времени между измерениями

    Возвращает:
        float: значение ускорения объёма
    """
    if dt <= 0:
        return 0.0
    return (curr_speed - prev_speed) / dt

def detect_spoofing_like_behavior(orderbook_snapshots: pd.DataFrame,
                                  best_bid: float,
                                  best_ask: float,
                                  volume_threshold: float = 1000,
                                  max_distance_ticks: int = 5,
                                  time_window: int = 5) -> pd.DataFrame:
    """
    Обнаружение подозрительного поведения спуфинга в серии срезов стакана.

    Args:
        orderbook_snapshots (pd.DataFrame): Данные срезов стакана с колонками:
            - 'timestamp' (int/float): метка времени
            - 'price' (float): цена заявки
            - 'side' (str): 'bid' или 'ask'
            - 'volume' (float): объем заявки
        best_bid (float): лучшая цена покупки на момент анализа
        best_ask (float): лучшая цена продажи на момент анализа
        volume_threshold (float): минимальный объем заявки для оценки (по умолчанию 1000)
        max_distance_ticks (int): макс. расстояние в тиках от лучшей цены для анализа (по умолчанию 5)
        time_window (int): число последних срезов для оценки поведения (по умолчанию 5)

    Returns:
        pd.DataFrame: DataFrame с колонками:
            - 'price'
            - 'side'
            - 'suspicious_score' (float от 0 до 1)
            - 'count_appears'
            - 'count_disappears'
            - 'avg_lifetime_sec'
    """

    # Фильтрация заявок по близости к спреду
    def within_spread_range(row):
        if row['side'] == 'bid':
            return row['price'] >= best_bid - max_distance_ticks * tick_size
        else:
            return row['price'] <= best_ask + max_distance_ticks * tick_size

    tick_size = 0.01  # Параметр, заменить на актуальный тик из рынка
    data = orderbook_snapshots.copy()
    data = data[data.apply(within_spread_range, axis=1)]
    data = data[data['volume'] >= volume_threshold]

    # Группировка по цене и стороне
    grouped = data.groupby(['price', 'side'])

    results = []
    for (price, side), group in grouped:
        timestamps = group['timestamp'].sort_values()
        volumes = group['volume'].loc[timestamps.index]

        appears = 0
        disappears = 0
        lifetimes = []

        for i in range(1, len(timestamps)):
            dt = timestamps.iloc[i] - timestamps.iloc[i - 1]
            dv = volumes.iloc[i] - volumes.iloc[i - 1]

            if dv > volume_threshold * 0.8:
                appears += 1
            elif dv < -volume_threshold * 0.8:
                disappears += 1

            if dv < -volume_threshold * 0.8 and i > 0:
                lifetime = dt
                lifetimes.append(lifetime)

        avg_lifetime = np.mean(lifetimes) if lifetimes else 0
        suspicious_score = min(1.0, (appears + disappears) / (time_window * 2))

        results.append({
            'price': price,
            'side': side,
            'suspicious_score': suspicious_score,
            'count_appears': appears,
            'count_disappears': disappears,
            'avg_lifetime_sec': avg_lifetime
        })

    return pd.DataFrame(results)

def get_orderbook_roc(orderbook_snapshots: pd.DataFrame, side: str = 'bid', window: int = 2) -> float:
    """
    Рассчитывает Rate of Change (ROC) объема на стороне стакана за заданное окно времени.

    Args:
        orderbook_snapshots (pd.DataFrame): Исторические срезы стакана с колонками ['price', 'volume', 'side', 'ts'].
        side (str): Сторона стакана ('bid' или 'ask'). По умолчанию 'bid'.
        window (int): Количество последних срезов для расчета ROC. По умолчанию 2 (текущий и предыдущий).

    Returns:
        float: Значение ROC объема (в процентах). Положительное — рост, отрицательное — падение.
    """
    data = orderbook_snapshots[orderbook_snapshots['side'] == side]

    if data.empty or len(data['ts'].unique()) < window:
        return 0.0

    # Суммируем объемы по каждому timestamp
    volume_by_ts = data.groupby('ts')['volume'].sum().sort_index()

    if len(volume_by_ts) < window:
        return 0.0

    current_volume = volume_by_ts.iloc[-1]
    prev_volume = volume_by_ts.iloc[-window]

    if prev_volume == 0:
        return 0.0

    roc = (current_volume - prev_volume) / prev_volume * 100

    return roc

def get_orderbook_roc(curr_imbalance: float, prev_imbalance: float) -> float:
    """
    Считает скорость изменения дисбаланса стакана (Rate of Change) между текущим и предыдущим значением.
    """
    return curr_imbalance - prev_imbalance

def get_volume_at_distance(bids: list, asks: list, mid_price: float, pct: float = 0.0025):
    """
    Считает bid/ask объём в пределах ±pct от mid_price.
    Возвращает (bid_volume_pct, ask_volume_pct, imbalance_pct).
    """
    bid_vol = 0.0
    ask_vol = 0.0
    for price, qty in bids:
        if mid_price * (1 - pct) <= price <= mid_price:
            bid_vol += qty
    for price, qty in asks:
        if mid_price <= price <= mid_price * (1 + pct):
            ask_vol += qty
    total = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
    return bid_vol, ask_vol, imbalance

def get_volume_density(bids: list, asks: list, num_levels: int = 3) -> tuple:
    """
    Рассчитывает плотность объёма заявок на num_levels уровнях стакана.
    Возвращает (bid_density, ask_density).
    """
    bids = bids[:num_levels]
    asks = asks[:num_levels]
    def density(levels):
        if len(levels) < 2:
            return float(levels[0][1]) if levels else 0.0
        price_span = abs(levels[0][0] - levels[-1][0])
        vol_sum = sum(qty for _, qty in levels)
        return float(vol_sum / price_span) if price_span > 0 else float(vol_sum)
    return density(bids), density(asks)
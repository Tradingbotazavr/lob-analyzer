from dataclasses import dataclass, asdict, fields
from typing import Optional

@dataclass
class FeatureSnapshot:
    """
    Унифицированный срез признаков для пайплайна LOB-аналитики.
    Используется для логирования, записи в parquet, инференса и алертов.
    
    Поля:
        ts: float — timestamp события
        price: float — цена сделки
        qty: float — объем сделки
        side: str — сторона сделки ('buy'/'sell')
        target: Optional[int] — целевая метка (для обучения/инференса)
        mid_price: Optional[float] — средняя цена по стакану
        avg_price: float — средняя цена по агрегации
        trade_count: int — количество сделок в агрегации
        buy_volume: float — объем покупок
        sell_volume: float — объем продаж
        buy_sell_ratio: float — отношение покупок к продажам
        direction: str — направление ('buy'/'sell')
        reason: str — причина сигнала
        bid_volume: float — объем на bid
        ask_volume: float — объем на ask
        bid_volume_near: float — bid вблизи mid
        ask_volume_near: float — ask вблизи mid
        imbalance: float — дисбаланс стакана
        buy_vol_speed: Optional[float] — скорость покупок
        sell_vol_speed: Optional[float] — скорость продаж
        price_std: Optional[float] — std цены
        ma_volume: Optional[float] — MA объема
        volatility: Optional[float] — волатильность
        activity_spike: Optional[int] — флаг всплеска активности

        # Новые признаки для манипуляций и глубины стакана
        bid_volume_pct_025: Optional[float] — bid-объем в пределах ±0.25% от mid
        ask_volume_pct_025: Optional[float] — ask-объем в пределах ±0.25% от mid
        imbalance_pct_025: Optional[float] — дисбаланс стакана в пределах ±0.25%
        bid_density_lvl1_3: Optional[float] — плотность bid на уровнях 1-3
        ask_density_lvl1_3: Optional[float] — плотность ask на уровнях 1-3
        is_spoofing_like: Optional[int] — эвристика spoofing (0/1)
        has_fake_wall: Optional[int] — эвристика fake wall (0/1)
        orderbook_entropy: Optional[float] — энтропия стакана
        orderbook_roc: Optional[float] — rate of change стакана
        price_roc: Optional[float] — rate of change цены
    """
    ts: float
    price: Optional[float] = None
    qty: Optional[float] = None
    side: Optional[str] = None
    target: Optional[int] = None
    mid_price: Optional[float] = None
    avg_price: float = 0.0
    trade_count: int = 0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_sell_ratio: float = 1.0
    direction: str = ""
    reason: str = ""
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    bid_volume_near: float = 0.0
    ask_volume_near: float = 0.0
    imbalance: float = 0.0
    buy_vol_speed: Optional[float] = None
    sell_vol_speed: Optional[float] = None
    price_std: Optional[float] = None
    ma_volume: Optional[float] = None
    volatility: Optional[float] = None
    activity_spike: Optional[int] = None

    # 🔽 Новые признаки для манипуляций и глубины стакана
    bid_volume_pct_025: Optional[float] = None  # TODO: вычислять в stream_merger.py
    ask_volume_pct_025: Optional[float] = None  # TODO: вычислять в stream_merger.py
    imbalance_pct_025: Optional[float] = None   # TODO: вычислять в stream_merger.py
    bid_density_lvl1_3: Optional[float] = None  # TODO: вычислять в stream_merger.py
    ask_density_lvl1_3: Optional[float] = None  # TODO: вычислять в stream_merger.py
    is_spoofing_like: Optional[int] = 0         # TODO: эвристика в trade_activity_analyzer.py
    has_fake_wall: Optional[int] = 0            # TODO: эвристика в trade_activity_analyzer.py
    orderbook_entropy: Optional[float] = None   # TODO: вычислять в stream_merger.py
    orderbook_roc: Optional[float] = None       # TODO: вычислять в stream_merger.py
    price_roc: Optional[float] = None           # НОВЫЙ ПРИЗНАК

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in field_names}
        return cls(**filtered)

    def __post_init__(self):
        # Приведение типов для совместимости с parquet и моделью
        if self.side is not None and not isinstance(self.side, str):
            self.side = str(self.side)
        if self.direction is not None and not isinstance(self.direction, str):
            self.direction = str(self.direction)
        if self.reason is not None and not isinstance(self.reason, str):
            self.reason = str(self.reason) 
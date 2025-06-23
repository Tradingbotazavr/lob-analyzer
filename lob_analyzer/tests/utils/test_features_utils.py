import pytest
import numpy as np
from lob_analyzer.utils.features_utils import (
    calculate_bid_ask_volume_pct,
    calculate_density,
    calculate_orderbook_entropy,
    calculate_orderbook_roc,
    detect_spoofing,
    detect_fake_wall,
)

# --- Тестовые данные ---

@pytest.fixture
def sample_orderbook():
    """Простой стакан для базовых тестов."""
    return {
        "bids": [[100, 10], [99, 5], [98, 20]],
        "asks": [[101, 8], [102, 12], [103, 15]],
    }

@pytest.fixture
def previous_orderbook():
    """Предыдущее состояние стакана для тестов ROC и спуфинга."""
    return {
        "bids": [[100, 8], [99, 6], [98, 18]],
        "asks": [[101, 10], [102, 10], [103, 16]],
    }

@pytest.fixture
def spoofing_orderbook():
    """Стакан, имитирующий спуфинг (большой ордер появился и исчез)."""
    return {
        "bids": [[100, 8], [99.5, 200], [99, 6]], # появился большой ордер
        "asks": [[101, 10], [102, 10]],
    }

@pytest.fixture
def fake_wall_orderbook():
    """Стакан с большим ордером далеко от спреда."""
    return {
        "bids": [[100, 10], [99, 5], [90, 500]], # Fake wall
        "asks": [[101, 8], [102, 12], [110, 400]], # Fake wall
    }

# --- Тесты для функций ---

def test_calculate_bid_ask_volume_pct(sample_orderbook):
    mid_price = 100.5
    bid_vol, ask_vol, imbalance = calculate_bid_ask_volume_pct(sample_orderbook, mid_price, pct=0.01) # 1% от mid_price
    
    # В 1% от 100.5 попадают все биды (100, 99) и один аск (101)
    assert np.isclose(bid_vol, 10 + 5) 
    assert np.isclose(ask_vol, 8)
    # imbalance = (15 - 8) / (15 + 8) = 7 / 23
    assert np.isclose(imbalance, (15 - 8) / (15 + 8))

def test_calculate_density(sample_orderbook):
    bid_density, ask_density = calculate_density(sample_orderbook, levels=3)
    
    # (100*10 + 99*5 + 98*20) / (10+5+20) = 3445 / 35 = 98.42
    # (101*8 + 102*12 + 103*15) / (8+12+15) = 3577 / 35 = 102.2
    assert np.isclose(bid_density, (100*10 + 99*5 + 98*20) / (10+5+20))
    assert np.isclose(ask_density, (101*8 + 102*12 + 103*15) / (8+12+15))

def test_calculate_orderbook_entropy(sample_orderbook):
    entropy = calculate_orderbook_entropy(sample_orderbook, levels=3)
    assert isinstance(entropy, float)
    assert entropy > 0

    # Тест на пустом стакане
    empty_ob = {"bids": [], "asks": []}
    assert calculate_orderbook_entropy(empty_ob) == 0

def test_calculate_orderbook_roc(sample_orderbook, previous_orderbook):
    roc = calculate_orderbook_roc(previous_orderbook, sample_orderbook)
    # Total volume change: (35 -> 35 for bids, 36 -> 35 for asks)
    # Previous total vol: 35+36 = 71. Current total vol: 35+35=70
    # ROC = (70 - 71) / 71 = -1/71
    assert np.isclose(roc, (70 - 71) / 71)
    
    # Тест с пустым предыдущим стаканом
    assert calculate_orderbook_roc({}, sample_orderbook) == 1.0

def test_detect_spoofing(spoofing_orderbook, previous_orderbook):
    is_spoofing = detect_spoofing(previous_orderbook, spoofing_orderbook, threshold=50)
    assert is_spoofing == 1 # Крупный ордер появился

    is_not_spoofing = detect_spoofing(spoofing_orderbook, previous_orderbook, threshold=50)
    assert is_not_spoofing == -1 # Крупный ордер исчез
    
    is_stable = detect_spoofing(previous_orderbook, previous_orderbook, threshold=50)
    assert is_stable == 0

def test_detect_fake_wall(fake_wall_orderbook, sample_orderbook):
    has_wall = detect_fake_wall(fake_wall_orderbook, wall_size_threshold=100, distance_pct=0.05)
    assert has_wall == 1

    no_wall = detect_fake_wall(sample_orderbook, wall_size_threshold=100, distance_pct=0.05)
    assert no_wall == 0 
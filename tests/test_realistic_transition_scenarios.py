"""
2-2-2「トレンド移行期の特別処理ルール」
実戦シナリオテスト

実際の市場状況を模したシナリオでの動作確認
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.trend_transition_manager import manage_trend_transition

def test_realistic_scenarios():
    """実戦的シナリオテスト"""
    
    print("🌟 2-2-2 実戦シナリオテスト")
    print("=" * 60)
    
    scenarios = [
        ("[UP] 強いトレンド → レンジ移行", create_trend_to_range_data),
        ("[CHART] レンジ → ブレイクアウト移行", create_range_to_breakout_data),
        ("💥 ボラティリティ急上昇", create_volatility_spike_data),
        ("🌊 不安定な市場", create_unstable_market_data)
    ]
    
    positions = {'STOCK_A': 1000.0, 'STOCK_B': 500.0}
    
    for scenario_name, data_creator in scenarios:
        print(f"\n{scenario_name}")
        print("-" * 40)
        
        data = data_creator()
        result = manage_trend_transition(data, "TestStrategy", positions)
        
        print(f"移行期検出: {result.is_transition_period}")
        print(f"移行タイプ: {result.transition_detection.transition_type}")
        print(f"リスクレベル: {result.transition_detection.risk_level}")
        print(f"エントリー許可: {result.entry_allowed}")
        print(f"ポジション調整数: {len(result.position_adjustments)}")
        
        if result.position_adjustments:
            for adj in result.position_adjustments:
                print(f"  [DOWN] {adj.strategy_name}: {adj.current_position_size:.0f} → {adj.recommended_size:.0f} ({adj.urgency})")
        
        if result.risk_modifications:
            print(f"リスク調整: {list(result.risk_modifications.keys())}")

def create_trend_to_range_data():
    """強いトレンド → レンジ移行データ"""
    dates = pd.date_range(start='2024-01-01', periods=80, freq='D')
    
    # 前半: 強い上昇トレンド
    trend1 = np.linspace(0, 15, 40)
    noise1 = np.random.normal(0, 0.8, 40)
    
    # 後半: レンジ相場（高ボラティリティ）
    trend2 = np.full(40, 15) + np.random.normal(0, 3, 40)
    
    prices = 100 + np.concatenate([trend1 + noise1, trend2])
    volumes = np.concatenate([
        np.random.randint(1000000, 2000000, 40),  # 通常出来高
        np.random.randint(3000000, 6000000, 40)   # 高出来高
    ])
    
    return pd.DataFrame({
        'Date': dates,
        'Adj Close': prices,
        'Volume': volumes,
        'High': prices * 1.03,
        'Low': prices * 0.97,
        'Open': np.roll(prices, 1)
    })

def create_range_to_breakout_data():
    """レンジ → ブレイクアウト移行データ"""
    dates = pd.date_range(start='2024-01-01', periods=80, freq='D')
    
    # 前半: レンジ相場
    range_base = 100
    range_amplitude = 5
    range_prices = range_base + range_amplitude * np.sin(np.linspace(0, 4*np.pi, 40))
    noise1 = np.random.normal(0, 1, 40)
    
    # 後半: ブレイクアウト（急激な上昇）
    breakout_start = range_prices[-1]
    breakout_trend = np.linspace(0, 20, 40)
    noise2 = np.random.normal(0, 2, 40)
    
    prices = np.concatenate([
        range_prices + noise1,
        breakout_start + breakout_trend + noise2
    ])
    
    volumes = np.concatenate([
        np.random.randint(800000, 1500000, 40),   # 低出来高
        np.random.randint(4000000, 8000000, 40)   # 急増出来高
    ])
    
    return pd.DataFrame({
        'Date': dates,
        'Adj Close': prices,
        'Volume': volumes,
        'High': prices * 1.04,
        'Low': prices * 0.96,
        'Open': np.roll(prices, 1)
    })

def create_volatility_spike_data():
    """ボラティリティ急上昇データ"""
    dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
    
    base_price = 100
    
    # 通常期間
    normal_trend = np.linspace(0, 5, 30)
    normal_noise = np.random.normal(0, 1, 30)
    
    # ボラティリティ急上昇期間
    spike_trend = np.linspace(5, 8, 30)
    spike_noise = np.random.normal(0, 5, 30)  # 5倍のボラティリティ
    
    prices = base_price + np.concatenate([
        normal_trend + normal_noise,
        spike_trend + spike_noise
    ])
    
    volumes = np.concatenate([
        np.random.randint(1000000, 2000000, 30),
        np.random.randint(5000000, 10000000, 30)  # 異常な出来高
    ])
    
    return pd.DataFrame({
        'Date': dates,
        'Adj Close': prices,
        'Volume': volumes,
        'High': prices * 1.06,
        'Low': prices * 0.94,
        'Open': np.roll(prices, 1)
    })

def create_unstable_market_data():
    """不安定な市場データ"""
    dates = pd.date_range(start='2024-01-01', periods=70, freq='D')
    
    base_price = 100
    
    # ランダムウォーク + 時々の大きな動き
    returns = np.random.normal(0, 0.02, 70)  # 基本2%ボラティリティ
    
    # ランダムに大きな動きを挿入
    shock_indices = np.random.choice(70, 8, replace=False)
    returns[shock_indices] += np.random.choice([-1, 1], 8) * np.random.uniform(0.05, 0.15, 8)
    
    prices = base_price * np.cumprod(1 + returns)
    
    # 出来高も不安定
    volumes = np.random.lognormal(14, 0.5, 70).astype(int)
    
    return pd.DataFrame({
        'Date': dates,
        'Adj Close': prices,
        'Volume': volumes,
        'High': prices * 1.05,
        'Low': prices * 0.95,
        'Open': np.roll(prices, 1)
    })

if __name__ == "__main__":
    test_realistic_scenarios()

"""
DSSMS Task 2.1: テスト用ヘルパー
統合テスト実行時の互換性補助

Author: GitHub Copilot Agent  
Created: 2025-01-22
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class MockPortfolioCalculator:
    """モックポートフォリオ計算エンジン"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions = {}
    
    @property
    def current_capital(self) -> float:
        """現在の資本"""
        return self.cash_balance
    
    @property 
    def total_portfolio_value(self) -> float:
        """総ポートフォリオ価値"""
        return self.current_capital
    
    def calculate_portfolio_weights(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """ポートフォリオ重み計算"""
        if data.empty:
            return None
        
        # 等重み配分
        symbols = ['A', 'B', 'C']  # テスト用
        return {symbol: 1.0/len(symbols) for symbol in symbols}

def create_test_switch_engine():
    """テスト用切替エンジン作成"""
    try:
        from src.dssms.dssms_switch_engine_v2 import DSSMSSwitchEngineV2
        # モック計算エンジンで初期化
        mock_calc = MockPortfolioCalculator()
        return DSSMSSwitchEngineV2(mock_calc)
    except Exception:
        # 失敗した場合はNoneを返す
        return None

def create_test_data() -> pd.DataFrame:
    """テスト用データ作成"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                         end=datetime.now(), freq='D')
    
    data = []
    for date in dates:
        data.append({
            'Date': date,
            'Open': 3000 + np.random.normal(0, 100),
            'High': 3100 + np.random.normal(0, 100),
            'Low': 2900 + np.random.normal(0, 100),
            'Close': 3000 + np.random.normal(0, 100),
            'Volume': np.random.randint(1000000, 10000000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

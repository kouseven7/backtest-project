"""
簡単な動作確認テストスクリプト
"""

import pandas as pd
import numpy as np
import os
import sys

# プロジェクトルートを追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config.optimized_parameters import OptimizedParameterManager

def test_basic_functionality():
    """基本的な機能のテスト"""
    print("="*60)
    print("半自動戦略適用システム - 基本動作確認")
    print("="*60)
    
    # 1. OptimizedParameterManagerのテスト
    print("\n1. OptimizedParameterManager テスト")
    manager = OptimizedParameterManager()
    
    # テスト用パラメータを保存
    test_params = {
        "sma_short": 20,
        "sma_long": 50,
        "rsi_period": 14,
        "take_profit": 0.10,
        "stop_loss": 0.05
    }
    
    test_metrics = {
        'sharpe_ratio': 1.5,
        'total_return': 0.15,
        'max_drawdown': -0.08
    }
    
    try:
        param_id = manager.save_optimized_params(
            strategy_name="MomentumInvestingStrategy",
            ticker="TEST",
            params=test_params,
            metrics=test_metrics
        )
        print(f"✅ パラメータ保存成功: {param_id}")
        
        # 保存されたパラメータを確認
        configs = manager.list_available_configs("MomentumInvestingStrategy")
        print(f"✅ 設定ファイル数: {len(configs)}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    # 2. MomentumInvestingStrategyの基本テスト
    print("\n2. MomentumInvestingStrategy テスト")
    try:
        from strategies.Momentum_Investing import MomentumInvestingStrategy
        
        # テストデータ作成
        dates = pd.date_range(start="2023-01-01", periods=100, freq='B')
        test_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Adj Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # 戦略インスタンス作成
        strategy = MomentumInvestingStrategy(data=test_data)
        print("✅ 戦略インスタンス作成成功")
        
        # 初期化
        strategy.initialize_strategy()
        print("✅ 戦略初期化成功")
        
        # バックテスト実行
        result = strategy.backtest()
        print(f"✅ バックテスト実行成功: {len(result)} データポイント")
        
        # 最適化情報取得
        opt_info = strategy.get_optimization_info()
        print(f"✅ 最適化情報取得成功: {len(opt_info)} 項目")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("基本動作確認完了")
    print("="*60)

if __name__ == "__main__":
    test_basic_functionality()

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def check_dssms_core_components():
    """DSSMSの核となるコンポーネントの動作確認"""
    print("=" * 80)
    print("DSSMS核心コンポーネント確認")
    print("=" * 80)
    
    # 1. スコアリングエンジンの確認
    print("\n1. スコアリングエンジンの確認...")
    try:
        from src.dssms.comprehensive_scoring_engine import ComprehensiveScoringEngine
        
        scoring_engine = ComprehensiveScoringEngine()
        print("   ✓ スコアリングエンジン作成成功")
        
        # テストデータ作成
        test_data = pd.DataFrame({
            'Close': [100, 102, 105, 103, 108],
            'Volume': [1000000, 1200000, 1500000, 900000, 1800000],
            'High': [102, 104, 107, 105, 110],
            'Low': [98, 100, 103, 101, 106]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        score = scoring_engine.calculate_score('TEST', test_data)
        print(f"   テストスコア: {score}")
        
    except Exception as e:
        print(f"   ✗ スコアリングエンジンエラー: {e}")
    
    # 2. ランキングシステムの確認
    print("\n2. ランキングシステムの確認...")
    try:
        from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem
        
        ranking_system = HierarchicalRankingSystem()
        print("   ✓ ランキングシステム作成成功")
        
        # テスト用スコア
        test_scores = {
            'TEST1': 0.8,
            'TEST2': 0.6,
            'TEST3': 0.9,
            'TEST4': 0.7
        }
        
        rankings = ranking_system.rank_symbols(test_scores)
        print(f"   テストランキング: {rankings}")
        
    except Exception as e:
        print(f"   ✗ ランキングシステムエラー: {e}")
    
    # 3. スイッチマネージャーの確認
    print("\n3. スイッチマネージャーの確認...")
    try:
        from src.dssms.intelligent_switch_manager import IntelligentSwitchManager
        
        switch_manager = IntelligentSwitchManager()
        print("   ✓ スイッチマネージャー作成成功")
        
        # テスト用現在ポジション
        current_position = 'TEST2'
        rankings = ['TEST3', 'TEST1', 'TEST2', 'TEST4']
        
        # switch_decision = switch_manager.should_switch(current_position, rankings)
        print(f"   スイッチマネージャーが正常に作成されました")
        
    except Exception as e:
        print(f"   ✗ スイッチマネージャーエラー: {e}")
    
    # 4. データマネージャーの確認
    print("\n4. データマネージャーの確認...")
    try:
        from src.dssms.dssms_data_manager import DSSMSDataManager
        
        data_manager = DSSMSDataManager()
        print("   ✓ データマネージャー作成成功")
        
    except Exception as e:
        print(f"   ✗ データマネージャーエラー: {e}")
    
    # 5. 実際のデータ品質の確認
    print("\n5. 実際のデータ品質確認...")
    try:
        import yfinance as yf
        
        # サンプル銘柄のデータ取得
        symbol = '7203'
        ticker = f"{symbol}.T"
        data = yf.download(ticker, start='2023-01-01', end='2023-01-10', progress=False)
        
        if not data.empty:
            print(f"   ✓ {symbol}データ取得成功: {len(data)}日分")
            print(f"   価格範囲: ¥{data['Close'].min():.2f} - ¥{data['Close'].max():.2f}")
            print(f"   ボリューム範囲: {data['Volume'].min():,} - {data['Volume'].max():,}")
            
            # データの完全性チェック
            missing_data = data.isnull().sum().sum()
            print(f"   欠損データ: {missing_data}個")
            
        else:
            print(f"   ✗ {symbol}データ取得失敗")
            
    except Exception as e:
        print(f"   ✗ データ取得エラー: {e}")

    # 6. シミュレーション核心ロジックの確認
    print("\n6. シミュレーション核心ロジックの確認...")
    try:
        from src.dssms.dssms_backtester import DSSMSBacktester
        
        backtester = DSSMSBacktester()
        
        # 初期状態確認
        print(f"   初期資本: {backtester.initial_capital}")
        print(f"   切替コスト率: {backtester.switch_cost_rate}")
        print(f"   最小保有期間: {backtester.min_holding_period_hours}時間")
        
        # 手動でシミュレーション状態を設定してテスト
        start_date = datetime(2023, 1, 1)
        symbol_universe = ['7203', '9984']
        
        # 初期化テスト
        backtester._initialize_simulation(start_date, symbol_universe)
        print(f"   ✓ シミュレーション初期化成功")
        
        # 市場条件評価テスト
        market_condition = backtester._evaluate_market_condition(start_date)
        print(f"   ✓ 市場条件評価: {market_condition}")
        
        # ランキング更新テスト
        ranking_result = backtester._update_symbol_ranking(start_date, symbol_universe)
        print(f"   ✓ ランキング更新: {ranking_result}")
        
        # 切替判定テスト
        switch_decision = backtester._evaluate_switch_decision(
            start_date, None, ranking_result, market_condition
        )
        print(f"   ✓ 切替判定: {switch_decision}")
        
    except Exception as e:
        print(f"   ✗ 核心ロジック確認エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_dssms_core_components()

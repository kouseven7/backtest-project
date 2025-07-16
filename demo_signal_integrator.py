"""
Signal Integrator Demo
File: demo_signal_integrator.py
Description: 
  3-3-1「シグナル競合時の優先度ルール設計」デモ
  シグナル統合システムの動作確認

Author: imega
Created: 2025-07-16
Modified: 2025-07-16
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from config.signal_integrator import (
        SignalIntegrator, SignalType, StrategySignal, ConflictType,
        create_signal_integrator, create_strategy_signal
    )
    from config.strategy_selector import StrategySelector
    from config.portfolio_weight_calculator import PortfolioWeightCalculator
    from config.strategy_scoring_model import StrategyScoreCalculator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_mock_components():
    """モックコンポーネントの作成"""
    try:
        # 既存コンポーネントの初期化
        strategy_selector = StrategySelector()
        portfolio_calculator = PortfolioWeightCalculator()
        score_calculator = StrategyScoreCalculator()
        
        return strategy_selector, portfolio_calculator, score_calculator
    except Exception as e:
        print(f"モックコンポーネント作成エラー: {e}")
        return None, None, None

def create_test_signals() -> Dict[str, StrategySignal]:
    """テスト用シグナルの作成"""
    signals = {}
    
    # 1. 競合なしシグナル
    signals["momentum_strategy"] = create_strategy_signal(
        strategy_name="momentum_strategy",
        signal_type=SignalType.ENTRY_LONG,
        confidence=0.8,
        position_size=0.15,
        metadata={"strategy_type": "momentum_strategy", "trend": "uptrend"}
    )
    
    # 2. 方向性競合シグナル (Long vs Short)
    signals["mean_reversion"] = create_strategy_signal(
        strategy_name="mean_reversion",
        signal_type=SignalType.ENTRY_SHORT,
        confidence=0.7,
        position_size=0.12,
        metadata={"strategy_type": "mean_reversion", "trend": "downtrend"}
    )
    
    # 3. 高信頼度シグナル
    signals["breakout_strategy"] = create_strategy_signal(
        strategy_name="breakout_strategy",
        signal_type=SignalType.ENTRY_LONG,
        confidence=0.95,
        position_size=0.20,
        metadata={"strategy_type": "trending_strategy", "breakout_confirmed": True}
    )
    
    # 4. エグジットシグナル（最優先）
    signals["vwap_bounce"] = create_strategy_signal(
        strategy_name="vwap_bounce",
        signal_type=SignalType.EXIT_LONG,
        confidence=0.6,
        position_size=0.10,
        metadata={"strategy_type": "mean_reversion", "exit_reason": "profit_target"}
    )
    
    return signals

def demo_basic_integration():
    """基本統合機能のデモ"""
    print("\n" + "="*60)
    print("基本統合機能デモ")
    print("="*60)
    
    try:
        # コンポーネント初期化
        strategy_selector, portfolio_calculator, score_calculator = create_mock_components()
        
        if not all([strategy_selector, portfolio_calculator, score_calculator]):
            print("✗ コンポーネント初期化失敗")
            return False
        
        # シグナル統合器作成
        integrator = create_signal_integrator(
            strategy_selector, portfolio_calculator, score_calculator
        )
        print("✓ シグナル統合器初期化完了")
        
        # テストシグナル作成
        test_signals = create_test_signals()
        print(f"✓ テストシグナル作成: {len(test_signals)} 個")
        
        # 現在のポートフォリオ状態（モック）
        current_portfolio = {
            "momentum_strategy": 0.05,
            "vwap_bounce": 0.15,  # 既存ロングポジション
            "breakout_strategy": 0.0,
            "mean_reversion": 0.0
        }
        
        available_capital = 1000000  # 100万円
        
        print(f"✓ 現在のポートフォリオ: {current_portfolio}")
        print(f"✓ 利用可能資金: {available_capital:,} 円")
        
        # シグナル統合実行
        result = integrator.integrate_signals(
            test_signals, current_portfolio, available_capital
        )
        
        print(f"\n--- 統合結果 ---")
        print(f"成功: {result['success']}")
        print(f"統合シグナル数: {result['statistics']['total_signals']}")
        print(f"競合数: {result['statistics']['conflicts_count']}")
        print(f"解決シグナル数: {result['statistics']['resolved_signals']}")
        
        # 個別シグナル結果
        print(f"\n--- 最終シグナル ---")
        for i, signal in enumerate(result['signals'], 1):
            print(f"{i}. {signal['strategy_name']}")
            print(f"   タイプ: {signal['signal_type']}")
            print(f"   信頼度: {signal['confidence']:.2f}")
            print(f"   ポジションサイズ: {signal['position_size']:.2f}")
            print(f"   優先度: {signal.get('adjusted_priority', 'N/A')}")
        
        # 競合情報
        if result['conflicts']:
            print(f"\n--- 競合情報 ---")
            for i, conflict in enumerate(result['conflicts'], 1):
                print(f"{i}. 競合タイプ: {conflict['conflict_type']}")
                print(f"   関連戦略: {conflict['conflicting_strategies']}")
                print(f"   解決戦略: {conflict['resolved_strategy']}")
                print(f"   解決理由: {conflict['resolution_reason']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 基本統合デモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_conflict_scenarios():
    """競合シナリオ別デモ"""
    print("\n" + "="*60)
    print("競合シナリオ別デモ")
    print("="*60)
    
    scenarios = [
        {
            "name": "方向性競合 (Long vs Short)",
            "signals": {
                "strategy_a": create_strategy_signal("strategy_a", SignalType.ENTRY_LONG, 0.8, 0.2),
                "strategy_b": create_strategy_signal("strategy_b", SignalType.ENTRY_SHORT, 0.7, 0.15)
            }
        },
        {
            "name": "エグジット優先",
            "signals": {
                "strategy_a": create_strategy_signal("strategy_a", SignalType.ENTRY_LONG, 0.9, 0.2),
                "strategy_b": create_strategy_signal("strategy_b", SignalType.EXIT_LONG, 0.6, 0.1)
            }
        },
        {
            "name": "リソース競合",
            "signals": {
                "strategy_a": create_strategy_signal("strategy_a", SignalType.ENTRY_LONG, 0.8, 0.4),
                "strategy_b": create_strategy_signal("strategy_b", SignalType.ENTRY_LONG, 0.7, 0.4),
                "strategy_c": create_strategy_signal("strategy_c", SignalType.ENTRY_LONG, 0.6, 0.4)
            }
        }
    ]
    
    try:
        strategy_selector, portfolio_calculator, score_calculator = create_mock_components()
        integrator = create_signal_integrator(
            strategy_selector, portfolio_calculator, score_calculator
        )
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n--- シナリオ {i}: {scenario['name']} ---")
            
            result = integrator.integrate_signals(
                scenario['signals'], {}, 1000000
            )
            
            print(f"競合数: {result['statistics']['conflicts_count']}")
            print(f"最終シグナル数: {result['statistics']['resolved_signals']}")
            
            if result['conflicts']:
                for conflict in result['conflicts']:
                    print(f"  競合タイプ: {conflict['conflict_type']}")
                    print(f"  解決方法: {conflict.get('resolution_reason', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ 競合シナリオデモエラー: {e}")
        return False

def demo_performance_monitoring():
    """パフォーマンス監視デモ"""
    print("\n" + "="*60)
    print("パフォーマンス監視デモ")
    print("="*60)
    
    try:
        strategy_selector, portfolio_calculator, score_calculator = create_mock_components()
        integrator = create_signal_integrator(
            strategy_selector, portfolio_calculator, score_calculator
        )
        
        # 複数回の統合処理を実行
        for i in range(5):
            test_signals = create_test_signals()
            
            # ランダムに信頼度を調整
            import random
            for signal in test_signals.values():
                signal.confidence = random.uniform(0.5, 1.0)
            
            result = integrator.integrate_signals(
                test_signals, {}, 1000000
            )
            
            print(f"処理 {i+1}: 信号数={len(result['signals'])}, 競合数={result['statistics']['conflicts_count']}")
        
        # 統計表示
        stats = integrator.get_integration_stats()
        print(f"\n--- 統合統計 ---")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ パフォーマンス監視デモエラー: {e}")
        return False

def run_comprehensive_demo():
    """包括的デモの実行"""
    print("\n" + "="*80)
    print("Signal Integrator 包括的デモ")
    print("="*80)
    
    demos = [
        ("基本統合機能", demo_basic_integration),
        ("競合シナリオ", demo_conflict_scenarios),
        ("パフォーマンス監視", demo_performance_monitoring)
    ]
    
    results = []
    
    for name, demo_func in demos:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            success = demo_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} 実行エラー: {e}")
            results.append((name, False))
    
    # 結果サマリー
    print(f"\n" + "="*80)
    print("デモ実行結果サマリー")
    print("="*80)
    
    total_demos = len(results)
    successful_demos = sum(1 for _, success in results if success)
    
    print(f"実行デモ数: {total_demos}")
    print(f"成功: {successful_demos}")
    print(f"失敗: {total_demos - successful_demos}")
    
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    return successful_demos == total_demos

if __name__ == "__main__":
    """メイン実行部"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Signal Integrator Demo")
    parser.add_argument("--basic", action="store_true", help="基本統合デモのみ実行")
    parser.add_argument("--conflicts", action="store_true", help="競合シナリオデモのみ実行")
    parser.add_argument("--performance", action="store_true", help="パフォーマンス監視デモのみ実行")
    
    args = parser.parse_args()
    
    if args.basic:
        success = demo_basic_integration()
    elif args.conflicts:
        success = demo_conflict_scenarios()
    elif args.performance:
        success = demo_performance_monitoring()
    else:
        # デフォルトは包括的デモ
        success = run_comprehensive_demo()
    
    sys.exit(0 if success else 1)

"""
2-2-3 信頼度閾値システム デモンストレーション

ConfidenceThresholdManagerとIntegratedDecisionSystemの
実際の使用例を示すデモンストレーションスクリプト
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# 必要なモジュールのインポート
try:
    from confidence_threshold_manager import (
        ConfidenceThresholdManager,
        ConfidenceThreshold,
        ActionType,
        ConfidenceLevel,
        create_confidence_threshold_manager
    )
    from integrated_decision_system import (
        IntegratedDecisionSystem,
        MarketCondition,
        RiskLevel,
        create_integrated_decision_system
    )
    from indicators.unified_trend_detector import UnifiedTrendDetector
except ImportError as e:
    print(f"インポートエラー: {e}")
    print("必要なモジュールが見つかりません。")
    sys.exit(1)


def create_sample_data(length: int = 200, seed: int = 42) -> pd.DataFrame:
    """サンプル市場データ作成"""
    np.random.seed(seed)
    
    # 日付生成
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 価格データ生成（トレンドあり）
    base_price = 100.0
    trend = np.linspace(0, 20, length)  # 上昇トレンド
    noise = np.random.normal(0, 2, length)
    prices = base_price + trend + noise
    
    # ボラティリティ変化を加える
    volatility_periods = length // 4
    for i in range(0, length, volatility_periods):
        end_idx = min(i + volatility_periods, length)
        if i // volatility_periods % 2 == 1:  # 高ボラティリティ期間
            prices[i:end_idx] += np.random.normal(0, 5, end_idx - i)
    
    # 出来高データ
    volumes = np.random.lognormal(8, 0.5, length).astype(int)
    
    # VWAPデータ（簡単な近似）
    vwap = prices * (1 + np.random.normal(0, 0.01, length))
    
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volumes,
        'VWAP': vwap
    })


def demo_confidence_threshold_manager():
    """ConfidenceThresholdManager デモ"""
    print("=" * 60)
    print("ConfidenceThresholdManager デモンストレーション")
    print("=" * 60)
    
    # サンプルデータ作成
    data = create_sample_data(150)
    print(f"サンプルデータ作成完了: {len(data)} 日分")
    
    # カスタム閾値設定
    custom_thresholds = ConfidenceThreshold(
        entry_threshold=0.65,
        exit_threshold=0.45,
        hold_threshold=0.55,
        high_confidence_threshold=0.85,
        position_sizing_threshold=0.7
    )
    
    # ConfidenceThresholdManagerの作成
    manager = create_confidence_threshold_manager(
        strategy_name="VWAP",
        data=data,
        trend_method="advanced",
        custom_thresholds=custom_thresholds
    )
    
    print(f"戦略: {manager.strategy_name}")
    print(f"信頼度倍率: {manager.confidence_multiplier}")
    
    # シナリオテスト
    scenarios = [
        {"position": 0.0, "pnl": 0.0, "name": "新規エントリー検討"},
        {"position": 0.5, "pnl": 50.0, "name": "利益ポジション保有中"},
        {"position": 0.3, "pnl": -20.0, "name": "損失ポジション保有中"},
        {"position": -0.4, "pnl": 30.0, "name": "ショートポジション利益中"},
        {"position": 0.8, "pnl": -100.0, "name": "大きな損失ポジション"}
    ]
    
    print("\n--- シナリオ別意思決定テスト ---")
    for i, scenario in enumerate(scenarios):
        print(f"\nシナリオ {i+1}: {scenario['name']}")
        print(f"  現在ポジション: {scenario['position']}")
        print(f"  未実現損益: {scenario['pnl']}")
        
        decision = manager.make_comprehensive_decision(
            data=data.iloc[:100+i*10],
            current_position=scenario['position'],
            unrealized_pnl=scenario['pnl'],
            market_context={"scenario": scenario['name']}
        )
        
        print(f"  決定: {decision.action.value}")
        print(f"  信頼度: {decision.confidence_score:.3f} ({decision.confidence_level.value})")
        print(f"  ポジション係数: {decision.position_size_factor:.2f}")
        print(f"  リスクレベル: {decision.get_risk_level()}")
        print(f"  理由: {decision.reasoning}")
    
    # 統計情報
    print("\n--- 意思決定統計 ---")
    stats = manager.get_decision_statistics()
    if "error" not in stats:
        print(f"総決定数: {stats['total_decisions']}")
        print(f"高信頼度比率: {stats['high_confidence_ratio']:.2%}")
        print(f"アクション可能比率: {stats['actionable_ratio']:.2%}")
        print(f"平均信頼度: {stats['confidence_stats']['mean']:.3f}")
        print("アクション分布:")
        for action, count in stats['action_counts'].items():
            if count > 0:
                print(f"  {action}: {count}")


def demo_integrated_decision_system():
    """IntegratedDecisionSystem デモ"""
    print("\n" + "=" * 60)
    print("IntegratedDecisionSystem デモンストレーション")
    print("=" * 60)
    
    # より複雑なサンプルデータ作成
    data = create_sample_data(200, seed=123)
    print(f"拡張サンプルデータ作成完了: {len(data)} 日分")
    
    # カスタム設定で統合システム作成
    custom_thresholds = ConfidenceThreshold(
        entry_threshold=0.7,
        exit_threshold=0.5,
        high_confidence_threshold=0.85,
        strategy_multipliers={
            "VWAP": 1.1,
            "Golden_Cross": 1.0,
            "Mean_Reversion": 0.9
        }
    )
    
    integrated_system = create_integrated_decision_system(
        strategy_name="VWAP",
        data=data,
        trend_method="advanced",
        custom_thresholds=custom_thresholds,
        risk_tolerance=0.6
    )
    
    print(f"リスク許容度: {integrated_system.risk_tolerance}")
    print(f"最大ポジションサイズ: {integrated_system.max_position_size}")
    
    # 時系列シミュレーション
    print("\n--- 時系列シミュレーション ---")
    position = 0.0
    pnl = 0.0
    trade_history = []
    
    for day in range(100, len(data), 5):  # 5日ごとに判定
        current_data = data.iloc[:day]
        
        # 統合意思決定実行
        decision = integrated_system.make_integrated_decision(
            data=current_data,
            current_position=position,
            unrealized_pnl=pnl,
            additional_context={
                "simulation_day": day,
                "portfolio_value": 10000 + pnl
            }
        )
        
        # 簡単なポジション更新ロジック
        if decision.action == ActionType.BUY and position <= 0:
            new_position = decision.position_size_factor
            trade_history.append({
                "day": day,
                "action": "BUY",
                "position_change": new_position - position,
                "confidence": decision.confidence_score
            })
            position = new_position
            
        elif decision.action == ActionType.SELL and position >= 0:
            new_position = -decision.position_size_factor
            trade_history.append({
                "day": day,
                "action": "SELL", 
                "position_change": new_position - position,
                "confidence": decision.confidence_score
            })
            position = new_position
            
        elif decision.action == ActionType.EXIT:
            if position != 0:
                trade_history.append({
                    "day": day,
                    "action": "EXIT",
                    "position_change": -position,
                    "confidence": decision.confidence_score
                })
                position = 0.0
        
        # 簡単なPnL計算（価格変動による）
        if day > 100:
            price_change = (data['Close'].iloc[day] / data['Close'].iloc[day-5] - 1)
            pnl += position * price_change * 1000  # 仮想的な計算
        
        # 決定結果表示（一部のみ）
        if len(trade_history) <= 5 or day % 20 == 0:
            print(f"Day {day}: {decision.action.value} "
                  f"(信頼度: {decision.confidence_score:.2f}, "
                  f"ポジション: {position:.2f}, PnL: {pnl:.0f})")
    
    # トレード履歴分析
    print("\n--- トレード履歴分析 ---")
    if trade_history:
        print(f"総トレード数: {len(trade_history)}")
        
        buy_trades = [t for t in trade_history if t['action'] == 'BUY']
        sell_trades = [t for t in trade_history if t['action'] == 'SELL']
        
        print(f"買いトレード: {len(buy_trades)}")
        print(f"売りトレード: {len(sell_trades)}")
        
        avg_confidence = np.mean([t['confidence'] for t in trade_history])
        print(f"平均信頼度: {avg_confidence:.3f}")
        
        print("最近のトレード:")
        for trade in trade_history[-3:]:
            print(f"  Day {trade['day']}: {trade['action']} "
                  f"(変化: {trade['position_change']:+.2f}, "
                  f"信頼度: {trade['confidence']:.2f})")
    
    # パフォーマンス要約
    print("\n--- パフォーマンス要約 ---")
    summary = integrated_system.get_performance_summary()
    if "error" not in summary:
        print(f"分析期間: {summary['period_days']} 日")
        print(f"総決定数: {summary['total_decisions']}")
        print(f"アクション可能比率: {summary['actionable_ratio']:.2%}")
        print(f"高信頼度比率: {summary['confidence_stats']['high_confidence_ratio']:.2%}")
        print(f"保守的決定数: {summary['risk_management']['conservative_decisions']}")
        
        print("\n市場状況分布:")
        for condition, count in summary['market_condition_distribution'].items():
            if count > 0:
                print(f"  {condition}: {count}")


def demo_risk_adjustment():
    """リスク調整機能デモ"""
    print("\n" + "=" * 60)
    print("リスク調整機能デモンストレーション")
    print("=" * 60)
    
    # 高ボラティリティデータ作成
    np.random.seed(456)
    dates = pd.date_range('2024-01-01', periods=100)
    
    # 高ボラティリティ価格データ
    base_price = 100.0
    high_vol_noise = np.random.normal(0, 5, 100)  # 高ボラティリティ
    prices = base_price + np.cumsum(np.random.randn(100) * 0.3) + high_vol_noise
    volumes = np.random.lognormal(9, 0.8, 100).astype(int)  # 高出来高
    
    volatile_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volumes
    })
    
    # 異なるリスク許容度でシステム作成
    risk_levels = [0.3, 0.6, 0.9]
    
    print("異なるリスク許容度での意思決定比較:")
    
    for risk_tolerance in risk_levels:
        print(f"\n--- リスク許容度: {risk_tolerance} ---")
        
        system = create_integrated_decision_system(
            strategy_name="VWAP",
            data=volatile_data,
            risk_tolerance=risk_tolerance
        )
        
        # 同じ条件で意思決定
        decision = system.make_integrated_decision(
            data=volatile_data,
            current_position=0.0,
            unrealized_pnl=0.0
        )
        
        print(f"アクション: {decision.action.value}")
        print(f"信頼度: {decision.confidence_score:.3f}")
        print(f"ポジション係数: {decision.position_size_factor:.2f}")
        print(f"リスクレベル: {decision.get_risk_level()}")
        
        # 市場分析情報
        market_context = system.analyze_market_context(volatile_data)
        print(f"市場状況: {market_context.market_condition.value}")
        print(f"ボラティリティ: {market_context.volatility:.3f}")
        print(f"市場リスクレベル: {market_context.risk_level.value}")


def main():
    """メインデモ実行"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("2-2-3 信頼度閾値に基づく意思決定ロジック デモンストレーション")
    print("実装日:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # 各デモ実行
        demo_confidence_threshold_manager()
        demo_integrated_decision_system()
        demo_risk_adjustment()
        
        print("\n" + "=" * 60)
        print("全デモンストレーション完了")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

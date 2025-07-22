"""
2-2-3 修正版デモ - データ整合性を考慮した信頼度閾値システムテスト
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    sys.exit(1)


def create_compatible_sample_data(length: int = 200, seed: int = 42) -> pd.DataFrame:
    """UnifiedTrendDetectorと互換性のあるサンプルデータ作成"""
    np.random.seed(seed)
    
    # 日付生成
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 価格データ生成（トレンドあり）
    base_price = 100.0
    trend = np.linspace(0, 20, length)  # 上昇トレンド
    noise = np.random.normal(0, 1.5, length)
    prices = base_price + trend + noise
    
    # 出来高データ
    volumes = np.random.lognormal(8, 0.3, length).astype(int)
    
    # VWAPデータ（価格との相関を持たせる）
    vwap = prices * (1 + np.random.normal(0, 0.005, length))
    
    # UnifiedTrendDetectorが期待するカラム名に合わせる
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,              # 基本価格
        'Adj Close': prices,          # 調整後価格（UnifiedTrendDetectorが使用）
        'Volume': volumes,
        'VWAP': vwap
    })


def test_trend_detector_directly():
    """UnifiedTrendDetector の直接テスト"""
    print("=" * 60)
    print("UnifiedTrendDetector 直接テスト")
    print("=" * 60)
    
    # 互換性データ作成
    data = create_compatible_sample_data(100)
    print(f"データカラム: {list(data.columns)}")
    
    # UnifiedTrendDetectorを直接作成
    try:
        detector = UnifiedTrendDetector(
            strategy_name="VWAP",
            method="sma",  # シンプルなSMAメソッドを使用
            data=data,
            price_column="Adj Close",  # 正しいカラム名を指定
            vwap_column="VWAP"
        )
        
        # トレンド検出テスト
        trend, confidence = detector.detect_trend_with_confidence()
        print(f"トレンド: {trend}")
        print(f"信頼度: {confidence:.3f}")
        print(f"信頼度レベル: {detector.get_confidence_level()}")
        
        return True
        
    except Exception as e:
        print(f"UnifiedTrendDetectorエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_fixed_confidence_system():
    """修正版信頼度システムデモ"""
    print("\n" + "=" * 60)
    print("修正版 ConfidenceThresholdManager デモ")
    print("=" * 60)
    
    # 互換性データ作成
    data = create_compatible_sample_data(150)
    print(f"サンプルデータ作成完了: {len(data)} 日分")
    print(f"データカラム: {list(data.columns)}")
    
    try:
        # UnifiedTrendDetectorを手動作成（カラム名を明示）
        detector = UnifiedTrendDetector(
            strategy_name="VWAP",
            method="sma",  # 安定したSMAメソッドを使用
            data=data,
            price_column="Adj Close",
            vwap_column="VWAP",
            params={
                "short_period": 10,
                "medium_period": 20,
                "long_period": 50
            }
        )
        
        # ConfidenceThresholdManagerを手動作成
        custom_thresholds = ConfidenceThreshold(
            entry_threshold=0.6,
            exit_threshold=0.4,
            hold_threshold=0.5,
            high_confidence_threshold=0.8
        )
        
        manager = ConfidenceThresholdManager(
            trend_detector=detector,
            thresholds=custom_thresholds
        )
        
        print(f"戦略: {manager.strategy_name}")
        print(f"信頼度倍率: {manager.confidence_multiplier}")
        
        # 基本テスト
        print("\n--- 基本動作テスト ---")
        trend, confidence = manager.trend_detector.detect_trend_with_confidence()
        print(f"検出されたトレンド: {trend}")
        print(f"信頼度: {confidence:.3f}")
        
        # シナリオテスト
        scenarios = [
            {"position": 0.0, "pnl": 0.0, "name": "新規エントリー検討"},
            {"position": 0.5, "pnl": 50.0, "name": "利益ポジション保有中"},
            {"position": 0.3, "pnl": -20.0, "name": "損失ポジション保有中"}
        ]
        
        print("\n--- シナリオ別意思決定テスト ---")
        for i, scenario in enumerate(scenarios):
            print(f"\nシナリオ {i+1}: {scenario['name']}")
            print(f"  現在ポジション: {scenario['position']}")
            print(f"  未実現損益: {scenario['pnl']}")
            
            decision = manager.make_comprehensive_decision(
                data=data.iloc[:80+i*10],
                current_position=scenario['position'],
                unrealized_pnl=scenario['pnl']
            )
            
            print(f"  決定: {decision.action.value}")
            print(f"  信頼度: {decision.confidence_score:.3f} ({decision.confidence_level.value})")
            print(f"  ポジション係数: {decision.position_size_factor:.2f}")
            print(f"  理由: {decision.reasoning}")
        
        # 統計情報
        print("\n--- 意思決定統計 ---")
        stats = manager.get_decision_statistics()
        if "error" not in stats:
            print(f"総決定数: {stats['total_decisions']}")
            print(f"高信頼度比率: {stats['high_confidence_ratio']:.2%}")
            print(f"アクション可能比率: {stats['actionable_ratio']:.2%}")
            if stats['total_decisions'] > 0:
                print(f"平均信頼度: {stats['confidence_stats']['mean']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"デモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_fixed_integrated_system():
    """修正版統合システムデモ"""
    print("\n" + "=" * 60)
    print("修正版 IntegratedDecisionSystem デモ")
    print("=" * 60)
    
    # 互換性データ作成
    data = create_compatible_sample_data(120)
    print(f"サンプルデータ作成完了: {len(data)} 日分")
    
    try:
        # 手動でUnifiedTrendDetectorを作成
        detector = UnifiedTrendDetector(
            strategy_name="VWAP",
            method="sma",
            data=data,
            price_column="Adj Close",
            vwap_column="VWAP",
            params={
                "short_period": 10,
                "medium_period": 20,
                "long_period": 50
            }
        )
        
        # ConfidenceThresholdManagerを作成
        manager = ConfidenceThresholdManager(
            trend_detector=detector,
            thresholds=ConfidenceThreshold(entry_threshold=0.6)
        )
        
        # IntegratedDecisionSystemを作成
        integrated_system = IntegratedDecisionSystem(
            confidence_manager=manager,
            risk_tolerance=0.6
        )
        
        print(f"リスク許容度: {integrated_system.risk_tolerance}")
        
        # 時系列シミュレーション
        print("\n--- 時系列シミュレーション ---")
        position = 0.0
        pnl = 0.0
        
        for day in [50, 60, 70, 80, 90, 100]:
            current_data = data.iloc[:day]
            
            # 統合意思決定実行
            decision = integrated_system.make_integrated_decision(
                data=current_data,
                current_position=position,
                unrealized_pnl=pnl
            )
            
            # 簡単なポジション更新
            if decision.action == ActionType.BUY and position <= 0:
                position = decision.position_size_factor
            elif decision.action == ActionType.SELL and position >= 0:
                position = -decision.position_size_factor
            elif decision.action == ActionType.EXIT:
                position = 0.0
            
            # PnL簡易計算
            if day > 50:
                price_change = (data['Close'].iloc[day] / data['Close'].iloc[day-10] - 1)
                pnl += position * price_change * 1000
            
            print(f"Day {day}: {decision.action.value} "
                  f"(信頼度: {decision.confidence_score:.2f}, "
                  f"ポジション: {position:.2f}, PnL: {pnl:.0f})")
        
        # パフォーマンス要約
        print("\n--- パフォーマンス要約 ---")
        summary = integrated_system.get_performance_summary()
        if "error" not in summary:
            print(f"総決定数: {summary['total_decisions']}")
            print(f"アクション可能比率: {summary['actionable_ratio']:.2%}")
            if summary['confidence_stats']['mean'] is not None:
                print(f"平均信頼度: {summary['confidence_stats']['mean']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"統合システムデモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン実行"""
    # ログ設定
    logging.basicConfig(
        level=logging.WARNING,  # ERRORとWARNINGのみ表示
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("2-2-3 修正版信頼度閾値システム デモンストレーション")
    print("実装日:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    success_count = 0
    total_tests = 3
    
    # 各テスト実行
    if test_trend_detector_directly():
        success_count += 1
        
    if demo_fixed_confidence_system():
        success_count += 1
        
    if demo_fixed_integrated_system():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"テスト結果: {success_count}/{total_tests} 成功")
    if success_count == total_tests:
        print("✅ 全テスト成功 - 2-2-3実装完了")
    else:
        print("⚠️  一部テスト失敗 - 要修正")
    print("=" * 60)


if __name__ == "__main__":
    main()

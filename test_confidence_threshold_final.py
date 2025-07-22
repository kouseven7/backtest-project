"""
2-2-3 最終版テスト - 正しいAPI使用での信頼度閾値システムテスト
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
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


def create_test_data(length: int = 150, seed: int = 42) -> pd.DataFrame:
    """テスト用のデータ作成"""
    np.random.seed(seed)
    
    # 日付生成
    dates = pd.date_range('2024-01-01', periods=length, freq='D')
    
    # 価格データ生成（明確な上昇トレンド）
    base_price = 100.0
    trend = np.linspace(0, 15, length)  # 上昇トレンド
    noise = np.random.normal(0, 1.0, length)
    prices = base_price + trend + noise
    
    # 出来高データ
    volumes = np.random.lognormal(8, 0.3, length).astype(int)
    
    # VWAPデータ（価格より少し低く）
    vwap = prices * (0.995 + np.random.normal(0, 0.002, length))
    
    # 正しいカラム名でデータフレーム作成
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Adj Close': prices,  # UnifiedTrendDetectorが使用
        'Volume': volumes,
        'VWAP': vwap
    })


def test_basic_trend_detection():
    """基本的なトレンド検出テスト"""
    print("=" * 60)
    print("基本トレンド検出テスト")
    print("=" * 60)
    
    # テストデータ作成
    data = create_test_data(100)
    print(f"データカラム: {list(data.columns)}")
    
    try:
        # 正しいパラメータでUnifiedTrendDetectorを作成
        detector = UnifiedTrendDetector(
            data=data,
            price_column="Adj Close",
            strategy_name="VWAP",
            method="sma",
            vwap_column="VWAP"
        )
        
        # 基本のトレンド検出
        trend = detector.detect_trend()
        print(f"検出されたトレンド: {trend}")
        
        # 信頼度付きトレンド検出
        trend_with_conf, confidence = detector.detect_trend_with_confidence()
        print(f"信頼度付きトレンド: {trend_with_conf}")
        print(f"信頼度スコア: {confidence:.3f}")
        
        # 信頼度レベル判定
        conf_level = detector.get_trend_confidence_level()
        print(f"信頼度レベル: {conf_level}")
        
        return True, detector
        
    except Exception as e:
        print(f"基本テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_confidence_threshold_manager():
    """ConfidenceThresholdManagerテスト"""
    print("\n" + "=" * 60)
    print("ConfidenceThresholdManager テスト")
    print("=" * 60)
    
    # テストデータ作成
    data = create_test_data(150)
    print(f"サンプルデータ作成完了: {len(data)} 日分")
    
    try:
        # UnifiedTrendDetectorを正しく作成
        detector = UnifiedTrendDetector(
            data=data,
            price_column="Adj Close",
            strategy_name="VWAP",
            method="sma",
            vwap_column="VWAP"
        )
        
        # カスタム閾値設定
        custom_thresholds = ConfidenceThreshold(
            entry_threshold=0.6,
            exit_threshold=0.4,
            hold_threshold=0.5,
            high_confidence_threshold=0.8
        )
        
        # ConfidenceThresholdManagerを作成
        manager = ConfidenceThresholdManager(
            trend_detector=detector,
            thresholds=custom_thresholds
        )
        
        print(f"戦略: {manager.strategy_name}")
        print(f"信頼度倍率: {manager.confidence_multiplier}")
        
        # 基本動作確認
        print("\n--- 基本動作確認 ---")
        trend, confidence = manager.trend_detector.detect_trend_with_confidence()
        print(f"検出されたトレンド: {trend}")
        print(f"信頼度: {confidence:.3f}")
        
        # シナリオ別テスト
        scenarios = [
            {"position": 0.0, "pnl": 0.0, "name": "新規エントリー"},
            {"position": 0.5, "pnl": 50.0, "name": "利益ポジション"},
            {"position": 0.3, "pnl": -20.0, "name": "損失ポジション"}
        ]
        
        print("\n--- シナリオ別意思決定 ---")
        for i, scenario in enumerate(scenarios):
            print(f"\nシナリオ {i+1}: {scenario['name']}")
            
            decision = manager.make_comprehensive_decision(
                data=data.iloc[:80+i*15],
                current_position=float(scenario['position']),
                unrealized_pnl=float(scenario['pnl'])
            )
            
            print(f"  決定: {decision.action.value}")
            print(f"  信頼度: {decision.confidence_score:.3f}")
            print(f"  ポジション係数: {decision.position_size_factor:.2f}")
            print(f"  理由: {decision.reasoning}")
        
        # 統計情報表示
        print("\n--- 統計情報 ---")
        stats = manager.get_decision_statistics()
        if "error" not in stats:
            print(f"総決定数: {stats['total_decisions']}")
            print(f"高信頼度比率: {stats['high_confidence_ratio']:.2%}")
            print(f"アクション可能比率: {stats['actionable_ratio']:.2%}")
            print(f"平均信頼度: {stats['confidence_stats']['mean']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"ConfidenceThresholdManagerテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_decision_system():
    """IntegratedDecisionSystemテスト"""
    print("\n" + "=" * 60)
    print("IntegratedDecisionSystem テスト")
    print("=" * 60)
    
    # テストデータ作成
    data = create_test_data(120)
    print(f"サンプルデータ作成完了: {len(data)} 日分")
    
    try:
        # UnifiedTrendDetectorを作成
        detector = UnifiedTrendDetector(
            data=data,
            price_column="Adj Close",
            strategy_name="VWAP",
            method="sma",
            vwap_column="VWAP"
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
        trade_count = 0
        
        simulation_days = [50, 60, 70, 80, 90, 100, 110]
        
        for day in simulation_days:
            current_data = data.iloc[:day]
            
            # 統合意思決定実行
            decision = integrated_system.make_integrated_decision(
                data=current_data,
                current_position=position,
                unrealized_pnl=pnl
            )
            
            # ポジション更新ロジック
            old_position = position
            if decision.action == ActionType.BUY and position <= 0:
                position = decision.position_size_factor
                trade_count += 1
            elif decision.action == ActionType.SELL and position >= 0:
                position = -decision.position_size_factor
                trade_count += 1
            elif decision.action == ActionType.EXIT and position != 0:
                position = 0.0
                trade_count += 1
            
            # 簡易PnL計算
            if day > 50 and old_position != 0:
                price_change = (data['Close'].iloc[day] / data['Close'].iloc[day-10] - 1)
                pnl += old_position * price_change * 1000
            
            print(f"Day {day:3d}: {decision.action.value:12s} "
                  f"(信頼度: {decision.confidence_score:.2f}, "
                  f"ポジション: {position:5.2f}, "
                  f"PnL: {pnl:6.0f})")
        
        # 最終結果
        print(f"\n--- 最終結果 ---")
        print(f"総トレード数: {trade_count}")
        print(f"最終ポジション: {position:.2f}")
        print(f"最終PnL: {pnl:.0f}")
        
        # パフォーマンス要約
        print("\n--- パフォーマンス要約 ---")
        summary = integrated_system.get_performance_summary()
        if "error" not in summary:
            print(f"総決定数: {summary['total_decisions']}")
            print(f"アクション可能比率: {summary['actionable_ratio']:.2%}")
            print(f"平均信頼度: {summary['confidence_stats']['mean']:.3f}")
            print(f"保守的決定数: {summary['risk_management']['conservative_decisions']}")
        
        return True
        
    except Exception as e:
        print(f"IntegratedDecisionSystemテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メイン実行"""
    # ログレベルを警告以上に設定（情報量削減）
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("2-2-3 信頼度閾値に基づく意思決定ロジック 最終テスト")
    print("=" * 60)
    print("実装日:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    success_count = 0
    total_tests = 3
    
    # テスト実行
    success, detector = test_basic_trend_detection()
    if success:
        success_count += 1
    
    if test_confidence_threshold_manager():
        success_count += 1
        
    if test_integrated_decision_system():
        success_count += 1
    
    # 最終結果
    print("\n" + "=" * 60)
    print("最終テスト結果")
    print("=" * 60)
    print(f"成功: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✅ 全テスト成功 - 2-2-3実装完了")
        print("\n🎯 2-2-3 実装内容:")
        print("   • ConfidenceThresholdManager: 信頼度閾値に基づく意思決定")
        print("   • IntegratedDecisionSystem: 市場コンテキスト統合決定")
        print("   • MarketContext: 市場状況分析と意思決定調整")
        print("   • RiskManagement: リスク許容度による動的調整")
        print("   • DecisionHistory: 意思決定履歴と統計分析")
    else:
        print("⚠️ 一部テスト失敗")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Demo Script: Trend Error Impact Analysis System
File: demo_trend_error_analysis.py
Description: 
  5-1-2「トレンド判定エラーの影響分析」
  統合システムのデモンストレーション

Author: imega
Created: 2025-07-21
Modified: 2025-07-21
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import warnings

# プロジェクトパスの追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 警告を抑制
warnings.filterwarnings('ignore')

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_demo_data():
    """デモ用データを作成"""
    
    logger.info("Creating demo data...")
    
    # 期間設定
    end_date = datetime(2023, 12, 31)
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # 市場データ（価格とボリューム）
    np.random.seed(42)  # 再現可能な結果のため
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    
    # トレンドを含む価格系列の生成
    trend_periods = [
        (0, 50, 0.002),      # 上昇トレンド
        (50, 100, -0.001),   # 下降トレンド
        (100, 150, 0.0),     # レンジ相場
        (150, 250, 0.0015),  # 再び上昇
        (250, 300, -0.002),  # 急落
        (300, len(dates), 0.0005)  # 回復
    ]
    
    for start_idx, end_idx, trend in trend_periods:
        if end_idx > len(returns):
            end_idx = len(returns)
        returns[start_idx:end_idx] += trend
    
    # 価格系列の計算
    prices = initial_price * np.cumprod(1 + returns)
    
    # ボリューム（価格変動に相関）
    base_volume = 1000000
    volume_multiplier = 1 + np.abs(returns) * 10
    volumes = (base_volume * volume_multiplier).astype(int)
    
    market_data = pd.DataFrame({
        'Adj Close': prices,
        'Volume': volumes
    }, index=dates)
    
    # トレンド予測データ（一部意図的にエラーを含む）
    predicted_trends = []
    confidences = []
    
    for i, price in enumerate(prices):
        # 基本的なトレンド判定（5日移動平均との比較）
        if i < 5:
            predicted_trends.append('range-bound')
            confidences.append(0.5)
            continue
        
        recent_avg = np.mean(prices[max(0, i-5):i])
        price_change = (price / recent_avg) - 1
        
        # 意図的なエラーを注入
        error_probability = 0.2  # 20%の確率でエラー
        if np.random.random() < error_probability:
            # エラーのある予測
            if price_change > 0.02:
                predicted_trend = 'downtrend'  # 間違った方向
                confidence = 0.8  # 高い信頼度で間違い
            elif price_change < -0.02:
                predicted_trend = 'uptrend'   # 間違った方向
                confidence = 0.7
            else:
                predicted_trend = 'uptrend'   # false positive
                confidence = 0.6
        else:
            # 正しい予測
            if price_change > 0.02:
                predicted_trend = 'uptrend'
                confidence = 0.75
            elif price_change < -0.02:
                predicted_trend = 'downtrend'
                confidence = 0.7
            else:
                predicted_trend = 'range-bound'
                confidence = 0.6
        
        predicted_trends.append(predicted_trend)
        confidences.append(confidence)
    
    trend_predictions = pd.DataFrame({
        'predicted_trend': predicted_trends,
        'confidence': confidences
    }, index=dates)
    
    logger.info(f"Demo data created: {len(market_data)} days of data")
    logger.info(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    return market_data, trend_predictions

def demonstrate_error_classification():
    """エラー分類のデモンストレーション"""
    
    print("\n" + "="*60)
    print("5-1-2「トレンド判定エラーの影響分析」- エラー分類デモ")
    print("="*60)
    
    try:
        # データ準備
        market_data, trend_predictions = create_demo_data()
        
        # エラー分類エンジンの初期化
        from analysis.trend_error_impact.error_classification_engine import TrendErrorClassificationEngine
        
        print("Success: エラー分類エンジンの初期化完了")
        
        classifier = TrendErrorClassificationEngine()
        
        # Ground truthの生成（簡易版）
        ground_truth_trends = []
        prices = market_data['Adj Close']
        
        for i in range(len(prices)):
            if i < 10:
                ground_truth_trends.append('unknown')
                continue
            
            # 10日後の価格変動で実際のトレンドを判定
            future_idx = min(i + 10, len(prices) - 1)
            price_change = (prices.iloc[future_idx] / prices.iloc[i]) - 1
            
            if price_change > 0.02:
                ground_truth_trends.append('uptrend')
            elif price_change < -0.02:
                ground_truth_trends.append('downtrend')
            else:
                ground_truth_trends.append('range-bound')
        
        ground_truth = pd.DataFrame({
            'actual_trend': ground_truth_trends
        }, index=market_data.index)
        
        # エラー分類の実行
        print("エラー分類を実行中...")
        classification_result = classifier.classify_trend_errors(
            trend_predictions, ground_truth, market_data
        )
        
        # 結果の表示
        print(f"\n--- エラー分類結果 ---")
        print(f"総エラー数: {classification_result.total_errors}")
        print(f"分析期間: {classification_result.period_analyzed[0].strftime('%Y-%m-%d')} - {classification_result.period_analyzed[1].strftime('%Y-%m-%d')}")
        
        print(f"\nエラータイプ別内訳:")
        for error_type, count in classification_result.error_breakdown.items():
            if count > 0:
                print(f"  {error_type.value}: {count}件")
        
        print(f"\n深刻度別分布:")
        for severity, count in classification_result.severity_distribution.items():
            if count > 0:
                print(f"  {severity.value}: {count}件")
        
        # レポート生成
        report = classifier.generate_classification_report(classification_result)
        print(f"\n--- 改善提案 ---")
        for recommendation in report.get('recommendations', []):
            print(f"- {recommendation}")
        
        print("Success: エラー分類デモ完了")
        return classification_result
        
    except Exception as e:
        print("Error: エラー分類デモエラー: {0}".format(e))
        logger.error(f"Error classification demo failed: {e}")
        return None

def demonstrate_impact_calculation():
    """影響計算のデモンストレーション"""
    
    print("\n" + "="*60)
    print("影響度計算デモ")
    print("="*60)
    
    try:
        # データ準備
        market_data, trend_predictions = create_demo_data()
        
        # 影響計算エンジンの初期化
        from analysis.trend_error_impact.error_impact_calculator import ErrorImpactCalculator
        from analysis.trend_error_impact.error_classification_engine import TrendErrorInstance, TrendErrorType, ErrorSeverity
        
        print("Success: 影響計算エンジンの初期化完了")
        
        calculator = ErrorImpactCalculator()
        
        # サンプルエラーインスタンスの作成
        sample_error = TrendErrorInstance(
            timestamp=datetime(2023, 6, 15),
            error_type=TrendErrorType.DIRECTION_WRONG,
            severity=ErrorSeverity.HIGH,
            predicted_trend="uptrend",
            actual_trend="downtrend",
            confidence_level=0.8,
            market_context={"volatility": 0.25, "volume_ratio": 1.5}
        )
        
        # ポートフォリオコンテキスト
        portfolio_context = {
            'total_portfolio_value': 5000000,
            'active_strategies': 5
        }
        
        # 影響計算の実行
        print("影響度計算を実行中...")
        impact_result = calculator.calculate_error_impact(
            sample_error, market_data, portfolio_context
        )
        
        # 結果の表示
        print(f"\n--- 影響度計算結果 ---")
        print(f"対象エラー: {sample_error.error_type.value} ({sample_error.severity.value})")
        print(f"予測: {sample_error.predicted_trend} → 実際: {sample_error.actual_trend}")
        print(f"信頼度: {sample_error.confidence_level:.1%}")
        
        metrics = impact_result.impact_metrics
        print(f"\n影響度指標:")
        print(f"  直接損失: {metrics.direct_loss:.4f}")
        print(f"  機会損失: {metrics.opportunity_cost:.4f}")
        print(f"  リスク調整後影響: {metrics.risk_adjusted_impact:.4f}")
        print(f"  システム影響: {metrics.systemic_impact:.4f}")
        print(f"  複合スコア: {metrics.composite_score:.4f}")
        
        print(f"\n信頼区間 (95%): [{metrics.confidence_interval[0]:.4f}, {metrics.confidence_interval[1]:.4f}]")
        
        print(f"\n軽減提案:")
        for suggestion in impact_result.mitigation_suggestions:
            print(f"- {suggestion}")
        
        print("Success: 影響計算デモ完了")
        return impact_result
        
    except Exception as e:
        print(f"✗ 影響計算デモエラー: {e}")
        logger.error(f"Impact calculation demo failed: {e}")
        return None

def demonstrate_comprehensive_analysis():
    """包括的分析のデモンストレーション"""
    
    print("\n" + "="*60)
    print("包括的分析デモ")
    print("="*60)
    
    try:
        # データ準備
        market_data, trend_predictions = create_demo_data()
        
        # 包括的分析エンジンの初期化
        from analysis.trend_error_impact.trend_error_analyzer import TrendErrorAnalyzer
        
        print("Success: 包括的分析エンジンの初期化完了")
        
        analyzer = TrendErrorAnalyzer()
        
        # ポートフォリオコンテキスト
        portfolio_context = {
            'total_portfolio_value': 10000000,
            'active_strategies': 7,
            'current_risk_level': 'MEDIUM'
        }
        
        # 包括的分析の実行
        print("包括的分析を実行中...")
        analysis_result = analyzer.analyze_trend_errors(
            market_data, trend_predictions, portfolio_context
        )
        
        # 結果の表示
        print(f"\n--- 包括的分析結果 ---")
        print(f"分析期間: {analysis_result.analysis_period[0].strftime('%Y-%m-%d')} - {analysis_result.analysis_period[1].strftime('%Y-%m-%d')}")
        print(f"総影響スコア: {analysis_result.total_impact_score:.4f}")
        print(f"平均エラー深刻度: {analysis_result.average_error_severity:.4f}")
        print(f"リスク調整後総合影響: {analysis_result.risk_adjusted_total_impact:.4f}")
        
        # リスク評価
        risk_level = analysis_result.analysis_summary['risk_assessment']['overall_risk_level']
        print(f"総合リスクレベル: {risk_level}")
        
        # システム統合結果
        print(f"\nシステム統合分析:")
        integration = analysis_result.analysis_summary['system_integration']
        print(f"  ドローダウン相関: {integration['drawdown_correlation']:.3f}")
        print(f"  戦略切替重複: {integration['strategy_switching_overlap']:.3f}")
        print(f"  ポートフォリオリスク増幅: {integration['portfolio_risk_amplification']:.3f}")
        
        # 推奨事項
        print(f"\n--- 優先推奨事項 ---")
        for i, rec in enumerate(analysis_result.priority_recommendations, 1):
            print(f"{i}. {rec}")
        
        print(f"\n--- 即座のアクション ---")
        for i, action in enumerate(analysis_result.immediate_actions, 1):
            print(f"{i}. {action}")
        
        print(f"\n--- 長期改善提案 ---")
        for i, improvement in enumerate(analysis_result.long_term_improvements, 1):
            print(f"{i}. {improvement}")
        
        print("Success: 包括的分析デモ完了")
        return analysis_result
        
    except Exception as e:
        print(f"✗ 包括的分析デモエラー: {e}")
        logger.error(f"Comprehensive analysis demo failed: {e}")
        return None

def main():
    """メインデモ関数"""
    
    print("5-1-2「トレンド判定エラーの影響分析」デモンストレーション開始")
    print(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H時%M分%S秒')}")
    
    results = []
    
    # 1. エラー分類デモ
    classification_result = demonstrate_error_classification()
    results.append(("エラー分類", classification_result is not None))
    
    # 2. 影響計算デモ
    impact_result = demonstrate_impact_calculation()
    results.append(("影響計算", impact_result is not None))
    
    # 3. 包括的分析デモ
    comprehensive_result = demonstrate_comprehensive_analysis()
    results.append(("包括的分析", comprehensive_result is not None))
    
    # 結果サマリー
    print("\n" + "="*60)
    print("デモンストレーション結果サマリー")
    print("="*60)
    
    success_count = 0
    for name, success in results:
        status = "Success" if success else "Failed"
        print(f"{status}: {name}")
        if success:
            success_count += 1
    
    total_tests = len(results)
    print(f"総合結果: {success_count}/{total_tests} 成功")
    
    if success_count == total_tests:
        print("[SUCCESS] 全てのデモンストレーションが成功しました！")
        print("次のステップ:")
        print("1. 実際の市場データでのテスト")
        print("2. 既存システムとの統合テスト")
        print("3. リアルタイム監視システムの構築")
        return 0
    else:
        print("[WARNING] 一部のデモンストレーションが失敗しました。")
        print("ログを確認して問題を解決してください。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"デモンストレーション終了 (終了コード: {exit_code})")
    sys.exit(exit_code)

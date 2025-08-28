"""
DSSMS Task 3.4 パフォーマンス目標達成確認システム
統合デモンストレーション

このスクリプトは Task 3.4 の全コンポーネントを統合したデモを実行します。
"""
import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any

# パスの設定
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

# ロギング設定
from config.logger_config import setup_logger
logger = setup_logger("dssms_task34_demo")

# DSSMS Task 3.4 コンポーネントのインポート
try:
    from src.dssms.task34_workflow_coordinator import (
        Task34WorkflowCoordinator, 
        Task34WorkflowConfig
    )
    from src.dssms.performance_target_manager import TargetPhase
except ImportError as e:
    logger.error(f"DSSMSコンポーネントのインポートエラー: {e}")
    sys.exit(1)

def create_sample_performance_data() -> Dict[str, float]:
    """サンプルパフォーマンスデータの作成"""
    return {
        # 収益性指標
        "total_return": -15.5,          # 総リターン (%)
        "annual_return": -18.2,         # 年率リターン (%)
        "portfolio_value": 850000.0,    # ポートフォリオ価値
        "profit_factor": 0.85,          # プロフィットファクター
        "average_win": 2.3,             # 平均勝ち額 (%)
        
        # リスク管理指標
        "max_drawdown": 22.5,           # 最大ドローダウン (%)
        "value_at_risk": 6.8,           # VaR (%)
        "sharpe_ratio": -0.45,          # シャープレシオ
        "sortino_ratio": -0.52,         # ソルティノレシオ
        "risk_return_ratio": 1.8,       # リスクリターン比
        
        # 安定性指標
        "volatility": 18.5,             # ボラティリティ (%)
        "consistency_ratio": 0.35,      # 一貫性比率
        "win_rate": 0.42,               # 勝率
        "switching_success_rate": 0.38, # 戦略切り替え成功率
        "trade_frequency": 2.5,         # 取引頻度 (日次)
        
        # 効率性指標
        "trades_per_day": 2.1,          # 日次取引数
        "execution_speed": 0.15,        # 執行速度 (秒)
        "cost_efficiency": 0.92,        # コスト効率
        "capital_utilization": 0.78,    # 資本利用率
        "information_ratio": -0.28,     # インフォメーションレシオ
        
        # 適応性指標
        "strategy_correlation": 0.25,         # 戦略間相関
        "parameter_adaptation_rate": 0.65,    # パラメータ適応率
        "market_regime_detection": 0.75,     # 市場環境検出率
        
        # 追加メトリクス
        "evaluation_period_days": 90,         # 評価期間 (日)
        "trade_count": 185,                   # 取引数
        "stop_loss_percent": 0.025,           # ストップロス (%)
        "max_position_size": 0.08            # 最大ポジションサイズ
    }

def create_sample_risk_metrics() -> Dict[str, float]:
    """サンプルリスク指標の作成"""
    return {
        "max_drawdown": 22.5,
        "value_at_risk": 6.8,
        "conditional_var": 9.2,
        "beta": 1.15,
        "tracking_error": 12.5,
        "adjustment_factor": 0.85
    }

def demonstrate_emergency_scenario() -> Dict[str, float]:
    """緊急事態シナリオのデモンストレーション"""
    return {
        # 危機的な収益性指標
        "total_return": -45.2,          # 総リターン (%)
        "annual_return": -52.8,         # 年率リターン (%)
        "portfolio_value": 450000.0,    # ポートフォリオ価値
        "profit_factor": 0.25,          # プロフィットファクター
        "average_win": 0.8,             # 平均勝ち額 (%)
        
        # 危機的なリスク指標
        "max_drawdown": 65.5,           # 最大ドローダウン (%)
        "value_at_risk": 18.8,          # VaR (%)
        "sharpe_ratio": -1.85,          # シャープレシオ
        "sortino_ratio": -2.12,         # ソルティノレシオ
        "risk_return_ratio": 0.35,      # リスクリターン比
        
        # その他指標も悪化
        "volatility": 35.5,
        "consistency_ratio": 0.12,
        "win_rate": 0.25,
        "switching_success_rate": 0.15,
        "trade_frequency": 5.2,
        
        "trades_per_day": 4.8,
        "execution_speed": 0.45,
        "cost_efficiency": 0.65,
        "capital_utilization": 0.92,
        "information_ratio": -1.28,
        
        "strategy_correlation": 0.85,
        "parameter_adaptation_rate": 0.25,
        "market_regime_detection": 0.35,
        
        "evaluation_period_days": 90,
        "trade_count": 425,
        "stop_loss_percent": 0.035,
        "max_position_size": 0.15
    }

def run_normal_scenario_demo():
    """通常シナリオのデモ実行"""
    logger.info("=" * 80)
    logger.info("DSSMS Task 3.4 通常シナリオ デモ開始")
    logger.info("=" * 80)
    
    # ワークフローコーディネーターの初期化
    config = Task34WorkflowConfig(
        enable_auto_phase_transition=True,
        enable_emergency_fixes=True,
        enable_detailed_reporting=True,
        report_formats=['excel', 'json', 'text']
    )
    
    coordinator = Task34WorkflowCoordinator(config)
    
    # サンプルデータの準備
    performance_data = create_sample_performance_data()
    risk_metrics = create_sample_risk_metrics()
    
    logger.info("サンプルパフォーマンスデータ:")
    for key, value in performance_data.items():
        logger.info(f"  {key}: {value}")
    
    # フルワークフローの実行
    logger.info("\nTask 3.4 フルワークフロー実行中...")
    result = coordinator.execute_full_workflow(
        performance_data=performance_data,
        risk_metrics=risk_metrics
    )
    
    # 結果の表示
    logger.info(f"\n実行結果:")
    logger.info(f"  実行ID: {result.execution_id}")
    logger.info(f"  成功: {result.success}")
    logger.info(f"  実行時間: {(result.end_time - result.start_time).total_seconds():.2f}秒")
    logger.info(f"  総合スコア: {result.evaluation_result.overall_score:.1f}")
    logger.info(f"  リスク調整後スコア: {result.evaluation_result.risk_adjusted_score:.1f}")
    logger.info(f"  信頼度: {result.evaluation_result.confidence_level:.1%}")
    
    # 目標達成状況
    logger.info(f"\n目標達成状況:")
    for target_result in result.target_results:
        logger.info(f"  {target_result.metric_name}: {target_result.achievement_level.value}")
    
    # 次元別スコア
    logger.info(f"\n次元別スコア:")
    for dimension_score in result.evaluation_result.dimension_scores:
        logger.info(f"  {dimension_score.dimension_name}: {dimension_score.score:.1f} ({dimension_score.status})")
    
    # フェーズ移行推奨
    if result.phase_transition_recommended:
        logger.info(f"\nフェーズ移行推奨: {result.next_recommended_phase.value if result.next_recommended_phase else 'なし'}")
    
    # 生成されたレポートファイル
    logger.info(f"\n生成されたレポートファイル:")
    for format_type, filepath in result.report_files.items():
        logger.info(f"  {format_type}: {filepath}")
    
    # 推奨事項
    if result.evaluation_result.recommendations:
        logger.info(f"\n改善推奨事項:")
        for i, rec in enumerate(result.evaluation_result.recommendations, 1):
            logger.info(f"  {i}. {rec}")
    
    return result

def run_emergency_scenario_demo():
    """緊急事態シナリオのデモ実行"""
    logger.info("\n" + "=" * 80)
    logger.info("DSSMS Task 3.4 緊急事態シナリオ デモ開始")
    logger.info("=" * 80)
    
    # ワークフローコーディネーターの初期化
    config = Task34WorkflowConfig(
        enable_auto_phase_transition=False,  # 緊急時はフェーズ移行停止
        enable_emergency_fixes=True,
        enable_detailed_reporting=True,
        report_formats=['json', 'text']  # 緊急時は軽量レポート
    )
    
    coordinator = Task34WorkflowCoordinator(config)
    
    # 危機的パフォーマンスデータ
    emergency_performance_data = demonstrate_emergency_scenario()
    emergency_risk_metrics = {
        "max_drawdown": 65.5,
        "value_at_risk": 18.8,
        "conditional_var": 25.2,
        "beta": 1.85,
        "tracking_error": 45.5,
        "adjustment_factor": 0.35
    }
    
    logger.info("緊急事態パフォーマンスデータ:")
    for key, value in emergency_performance_data.items():
        if "return" in key or "drawdown" in key or "var" in key:
            logger.info(f"  {key}: {value}")
    
    # 緊急ワークフローの実行
    logger.info("\n緊急事態対応ワークフロー実行中...")
    emergency_result = coordinator.execute_full_workflow(
        performance_data=emergency_performance_data,
        risk_metrics=emergency_risk_metrics,
        execution_id="emergency_demo"
    )
    
    # 緊急修正結果の詳細表示
    logger.info(f"\n緊急事態対応結果:")
    logger.info(f"  実行ID: {emergency_result.execution_id}")
    logger.info(f"  成功: {emergency_result.success}")
    logger.info(f"  総合スコア: {emergency_result.evaluation_result.overall_score:.1f}")
    logger.info(f"  リスク調整後スコア: {emergency_result.evaluation_result.risk_adjusted_score:.1f}")
    
    if emergency_result.emergency_fix_result:
        fix_result = emergency_result.emergency_fix_result
        logger.info(f"\n緊急修正実行結果:")
        logger.info(f"  トリガー条件: {fix_result.trigger_condition}")
        logger.info(f"  全体成功: {fix_result.overall_success}")
        logger.info(f"  実行サマリー: {fix_result.execution_summary}")
        
        if fix_result.actions_executed:
            logger.info(f"  実行済みアクション ({len(fix_result.actions_executed)}件):")
            for action in fix_result.actions_executed:
                logger.info(f"    - {action.description} ({action.execution_result})")
        
        if fix_result.actions_pending:
            logger.info(f"  保留アクション ({len(fix_result.actions_pending)}件):")
            for action in fix_result.actions_pending:
                logger.info(f"    - {action.description} ({action.priority.value})")
    
    # アラート表示
    if emergency_result.evaluation_result.alerts:
        logger.info(f"\n重要アラート:")
        for alert in emergency_result.evaluation_result.alerts:
            logger.info(f"  ⚠️  {alert}")
    
    return emergency_result

def run_performance_trend_analysis(coordinator: Task34WorkflowCoordinator):
    """パフォーマンストレンド分析のデモ"""
    logger.info("\n" + "=" * 80)
    logger.info("パフォーマンストレンド分析 デモ")
    logger.info("=" * 80)
    
    # 複数回の実行履歴をシミュレート（実際の運用では自然に蓄積される）
    logger.info("複数回の実行履歴をシミュレート中...")
    
    sample_data_variations = [
        {"total_return": -10.2, "max_drawdown": 15.5, "overall_modifier": 1.1},
        {"total_return": -8.5, "max_drawdown": 18.2, "overall_modifier": 1.05},
        {"total_return": -12.8, "max_drawdown": 25.1, "overall_modifier": 0.95},
        {"total_return": -6.2, "max_drawdown": 12.8, "overall_modifier": 1.15},
        {"total_return": -15.5, "max_drawdown": 22.5, "overall_modifier": 0.85}
    ]
    
    for i, variation in enumerate(sample_data_variations):
        base_data = create_sample_performance_data()
        # データを変更
        for key, value in variation.items():
            if key != "overall_modifier":
                base_data[key] = value
        
        # その他の指標も調整
        modifier = variation["overall_modifier"]
        for key in base_data:
            if key not in ["total_return", "max_drawdown", "evaluation_period_days", "trade_count"]:
                if base_data[key] > 0:
                    base_data[key] *= modifier
        
        # 実行
        result = coordinator.execute_monitoring_workflow(base_data)
        logger.info(f"  実行 {i+1}: スコア {result.evaluation_result.overall_score:.1f}")
    
    # トレンド分析の実行
    trends = coordinator.get_performance_trends(last_n_executions=5)
    
    logger.info(f"\nパフォーマンストレンド分析結果:")
    logger.info(f"  分析期間: {trends['analysis_period']['executions_analyzed']}回の実行")
    logger.info(f"  総合スコアトレンド:")
    logger.info(f"    現在値: {trends['overall_score_trend']['current']:.1f}")
    logger.info(f"    平均値: {trends['overall_score_trend']['average']:.1f}")
    logger.info(f"    トレンド: {trends['overall_score_trend']['trend']}")
    logger.info(f"    変動幅: {trends['overall_score_trend']['volatility']:.1f}")
    
    logger.info(f"  運用指標:")
    logger.info(f"    緊急修正率: {trends['operational_metrics']['emergency_fix_rate']:.1%}")
    logger.info(f"    フェーズ移行率: {trends['operational_metrics']['phase_transition_rate']:.1%}")
    logger.info(f"    平均実行時間: {trends['operational_metrics']['average_execution_time']:.2f}秒")

def main():
    """メイン実行関数"""
    try:
        logger.info("DSSMS Task 3.4 統合デモンストレーション開始")
        logger.info(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: 通常シナリオのデモ
        normal_result = run_normal_scenario_demo()
        
        # Step 2: 緊急事態シナリオのデモ
        emergency_result = run_emergency_scenario_demo()
        
        # Step 3: パフォーマンストレンド分析のデモ
        # 通常シナリオで使用したコーディネーターを再利用
        coordinator = Task34WorkflowCoordinator()
        run_performance_trend_analysis(coordinator)
        
        # Step 4: 現在のフェーズ状況確認
        logger.info("\n" + "=" * 80)
        logger.info("現在のフェーズ状況確認")
        logger.info("=" * 80)
        
        phase_status = coordinator.get_current_phase_status()
        logger.info(f"現在のフェーズ: {phase_status['current_phase']}")
        logger.info(f"実行回数: {phase_status['execution_count']}")
        
        # Step 5: 実行履歴のエクスポート
        logger.info("\n実行履歴をエクスポート中...")
        export_path = coordinator.export_execution_history()
        if export_path:
            logger.info(f"実行履歴エクスポート完了: {export_path}")
        
        logger.info("\n" + "=" * 80)
        logger.info("DSSMS Task 3.4 統合デモンストレーション正常完了")
        logger.info("=" * 80)
        
        # サマリー情報
        logger.info(f"\nデモ実行サマリー:")
        logger.info(f"  通常シナリオ実行: {'成功' if normal_result.success else '失敗'}")
        logger.info(f"  緊急シナリオ実行: {'成功' if emergency_result.success else '失敗'}")
        logger.info(f"  緊急修正発動: {'あり' if emergency_result.emergency_fix_result else 'なし'}")
        logger.info(f"  生成レポート数: {len(normal_result.report_files) + len(emergency_result.report_files)}")
        
        return True
        
    except Exception as e:
        logger.error(f"デモ実行中にエラーが発生しました: {e}")
        import traceback
        logger.error(f"エラー詳細:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

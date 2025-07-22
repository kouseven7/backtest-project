"""
5-2-3 最適な重み付け比率の学習アルゴリズム デモンストレーション

ベイジアン最適化による階層的重み学習システムの実装テスト
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# システムのインポート
try:
    from config.weight_learning_optimizer.optimal_weight_learning_system import (
        OptimalWeightLearningSystem, LearningMode
    )
    print("✓ 5-2-3 最適重み学習システムのインポートに成功しました")
except ImportError as e:
    print(f"✗ システムインポートエラー: {e}")
    sys.exit(1)

def setup_logging():
    """ロギングの設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo_5_2_3_weight_learning.log')
        ]
    )

def generate_sample_data():
    """サンプルデータの生成"""
    print("サンプルデータを生成中...")
    
    # 日付範囲の設定
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 市場データの生成（複数資産）
    np.random.seed(42)
    n_days = len(date_range)
    
    market_data = pd.DataFrame(index=date_range)
    
    # 株式データ（高リターン、高ボラティリティ）
    stock_returns = np.random.normal(0.0008, 0.015, n_days)  # 日次リターン
    stock_prices = 100 * np.exp(np.cumsum(stock_returns))
    market_data['stocks'] = stock_prices
    
    # 債券データ（低リターン、低ボラティリティ）
    bond_returns = np.random.normal(0.0002, 0.005, n_days)
    bond_prices = 100 * np.exp(np.cumsum(bond_returns))
    market_data['bonds'] = bond_prices
    
    # コモディティデータ（中リターン、中ボラティリティ）
    commodity_returns = np.random.normal(0.0004, 0.012, n_days)
    commodity_prices = 100 * np.exp(np.cumsum(commodity_returns))
    market_data['commodities'] = commodity_prices
    
    # オルタナティブ投資データ
    alt_returns = np.random.normal(0.0006, 0.018, n_days)
    alt_prices = 100 * np.exp(np.cumsum(alt_returns))
    market_data['alternatives'] = alt_prices
    
    # ボリュームデータの追加
    market_data['volume'] = np.random.lognormal(10, 0.5, n_days)
    
    # パフォーマンスデータの生成（ストラテジー別）
    performance_data = pd.DataFrame(index=date_range)
    
    # トレンドフォロー戦略
    trend_returns = np.random.normal(0.0005, 0.012, n_days)
    # 市場との相関を追加
    market_return = (stock_returns + bond_returns) / 2
    trend_returns = 0.7 * trend_returns + 0.3 * market_return
    performance_data['trend_following'] = 100 * np.exp(np.cumsum(trend_returns))
    
    # 平均回帰戦略
    mean_rev_returns = np.random.normal(0.0003, 0.008, n_days)
    # 逆相関を追加
    mean_rev_returns = 0.8 * mean_rev_returns - 0.2 * market_return
    performance_data['mean_reversion'] = 100 * np.exp(np.cumsum(mean_rev_returns))
    
    # モメンタム戦略
    momentum_returns = np.random.normal(0.0007, 0.014, n_days)
    performance_data['momentum'] = 100 * np.exp(np.cumsum(momentum_returns))
    
    # ボラティリティブレイクアウト戦略
    vol_breakout_returns = np.random.normal(0.0004, 0.016, n_days)
    performance_data['volatility_breakout'] = 100 * np.exp(np.cumsum(vol_breakout_returns))
    
    print(f"✓ {n_days}日分のサンプルデータを生成しました")
    return market_data, performance_data

def test_basic_functionality(system, market_data, performance_data):
    """基本機能のテスト"""
    print("\n" + "="*50)
    print("基本機能テスト")
    print("="*50)
    
    try:
        # システムサマリーの取得
        print("1. システムサマリーの取得...")
        summary = system.get_system_summary()
        print(f"   システム状態: {summary['system_status']['system_health']}")
        print(f"   初期化状態: {summary['system_status']['initialized']}")
        
        # システムヘルスチェック
        print("2. システムヘルスチェック...")
        health = system.perform_system_health_check()
        print(f"   全体健康状態: {health['overall_health']}")
        if health.get('recommendations'):
            print(f"   推奨事項: {len(health['recommendations'])}件")
            
        print("✓ 基本機能テストに成功しました")
        return True
        
    except Exception as e:
        print(f"✗ 基本機能テストでエラー: {e}")
        return False

def test_learning_modes(system, market_data, performance_data):
    """学習モードのテスト"""
    print("\n" + "="*50)
    print("学習モードテスト")
    print("="*50)
    
    # 各学習モードのテスト
    modes_to_test = [
        LearningMode.MICRO_ADJUSTMENT,
        LearningMode.STANDARD_OPTIMIZATION,
        LearningMode.MAJOR_REBALANCING
    ]
    
    results = []
    
    for mode in modes_to_test:
        try:
            print(f"\n{mode.value}モードのテスト中...")
            
            # 最適化の実行
            result = system.execute_optimal_learning(
                market_data=market_data,
                performance_data=performance_data,
                force_learning_mode=mode
            )
            
            print(f"   セッションID: {result.session_id}")
            print(f"   パフォーマンススコア: {result.performance_metrics.combined_score:.4f}")
            print(f"   調整幅: {result.adjustment_magnitude:.4f}")
            print(f"   信頼度: {result.confidence_score:.4f}")
            print(f"   実行時間: {result.execution_time:.2f}秒")
            print(f"   統合結果: {len(result.integration_results)}システム")
            
            results.append({
                'mode': mode.value,
                'performance_score': result.performance_metrics.combined_score,
                'adjustment_magnitude': result.adjustment_magnitude,
                'confidence_score': result.confidence_score,
                'execution_time': result.execution_time
            })
            
            print(f"   ✓ {mode.value}モードのテストに成功")
            
        except Exception as e:
            print(f"   ✗ {mode.value}モードでエラー: {e}")
            results.append({
                'mode': mode.value,
                'error': str(e)
            })
    
    # 結果の比較
    if results:
        print("\n学習モード比較:")
        for result in results:
            if 'error' not in result:
                print(f"   {result['mode']}: "
                      f"スコア={result['performance_score']:.4f}, "
                      f"調整={result['adjustment_magnitude']:.4f}, "
                      f"時間={result['execution_time']:.2f}s")
                      
    return len([r for r in results if 'error' not in r]) > 0

def test_constraint_management(system, market_data, performance_data):
    """制約管理のテスト"""
    print("\n" + "="*50)
    print("制約管理テスト")
    print("="*50)
    
    try:
        # 制約サマリーの取得
        constraint_summary = system.constraint_manager.get_constraint_summary()
        
        print("1. 制約設定の確認...")
        print(f"   ストラテジー制約数: {len(constraint_summary['strategy_constraints'])}")
        print(f"   ポートフォリオ制約数: {len(constraint_summary['portfolio_constraints'])}")
        print(f"   メタ制約数: {len(constraint_summary['meta_constraints'])}")
        
        # 制約違反テスト用の不正な重み
        print("2. 制約違反テスト...")
        invalid_weights = {
            'strategy_trend_following': 0.8,  # 制約違反（最大0.4）
            'strategy_mean_reversion': 0.1,
            'strategy_momentum': 0.05,
            'strategy_volatility_breakout': 0.05,
            'portfolio_stocks': 0.9,  # 制約違反（最大0.3）
            'portfolio_bonds': 0.1,
            'meta_learning_rate': 5.0  # 制約違反（最大3.0）
        }
        
        # 制約検証
        is_valid, violations = system.constraint_manager.validate_weights(invalid_weights)
        print(f"   制約違反検出: {len(violations)}件")
        
        if violations:
            print("   検出された違反:")
            for violation in violations[:3]:  # 最初の3件を表示
                print(f"     - {violation.parameter_name}: "
                      f"{violation.current_value:.3f} "
                      f"(許容範囲: {violation.allowed_range[0]:.3f}-{violation.allowed_range[1]:.3f})")
        
        # 制約修正のテスト
        print("3. 制約修正テスト...")
        corrected_weights = system.constraint_manager.apply_constraint_corrections(invalid_weights)
        is_valid_after, remaining_violations = system.constraint_manager.validate_weights(corrected_weights)
        
        print(f"   修正後の制約違反: {len(remaining_violations)}件")
        print(f"   修正成功率: {(len(violations) - len(remaining_violations)) / len(violations) * 100:.1f}%")
        
        print("✓ 制約管理テストに成功しました")
        return True
        
    except Exception as e:
        print(f"✗ 制約管理テストでエラー: {e}")
        return False

def test_integration_systems(system):
    """統合システムのテスト"""
    print("\n" + "="*50)
    print("統合システムテスト")
    print("="*50)
    
    try:
        # システム統合状態の確認
        integration_summary = system.integration_bridge.get_integration_summary()
        
        print("1. 統合システムの状態確認...")
        print(f"   利用可能システム数: {integration_summary.get('available_systems', 0)}")
        print(f"   総統合回数: {integration_summary.get('total_integrations', 0)}")
        print(f"   成功率: {integration_summary.get('recent_success_rate', 0):.1%}")
        
        # システム状態の詳細確認
        system_status = system.integration_bridge.get_system_status()
        
        print("2. 各システムの状態:")
        for system_name, status in system_status.items():
            status_display = status.get('status', 'unknown')
            last_integration = status.get('last_integration')
            last_integration_str = last_integration.strftime('%Y-%m-%d %H:%M') if last_integration else 'なし'
            print(f"   {system_name}: {status_display} (最終統合: {last_integration_str})")
        
        # テスト用の重みで統合テスト
        print("3. 統合機能テスト...")
        test_weights = {
            'strategy_trend_following': 0.3,
            'strategy_mean_reversion': 0.3,
            'strategy_momentum': 0.2,
            'strategy_volatility_breakout': 0.2,
            'portfolio_stocks': 0.6,
            'portfolio_bonds': 0.2,
            'portfolio_commodities': 0.1,
            'portfolio_alternatives': 0.1
        }
        
        # ダミーデータで統合テスト
        integration_results = system.integration_bridge.apply_optimized_weights(test_weights)
        
        successful_integrations = sum(1 for r in integration_results if r.integration_success)
        print(f"   統合テスト結果: {successful_integrations}/{len(integration_results)} システムで成功")
        
        if integration_results:
            avg_impact = np.mean([r.performance_impact for r in integration_results])
            print(f"   平均パフォーマンス影響度: {avg_impact:.4f}")
        
        print("✓ 統合システムテストに成功しました")
        return True
        
    except Exception as e:
        print(f"✗ 統合システムテストでエラー: {e}")
        return False

def test_performance_analysis(system, market_data, performance_data):
    """パフォーマンス分析のテスト"""
    print("\n" + "="*50)
    print("パフォーマンス分析テスト")
    print("="*50)
    
    try:
        # テスト用重みの設定
        test_weights = {
            'strategy_trend_following': 0.3,
            'strategy_mean_reversion': 0.3,
            'strategy_momentum': 0.2,
            'strategy_volatility_breakout': 0.2,
            'portfolio_stocks': 0.6,
            'portfolio_bonds': 0.2,
            'portfolio_commodities': 0.1,
            'portfolio_alternatives': 0.1
        }
        
        print("1. パフォーマンス評価...")
        # パフォーマンス評価の実行
        performance_metrics = system.performance_evaluator.evaluate_performance(
            performance_data, test_weights
        )
        
        print(f"   複合スコア: {performance_metrics.combined_score:.4f}")
        print(f"   期待リターン: {performance_metrics.expected_return:.2%}")
        print(f"   最大ドローダウン: {performance_metrics.max_drawdown:.2%}")
        print(f"   シャープレシオ: {performance_metrics.sharpe_ratio:.3f}")
        print(f"   カルマーレシオ: {performance_metrics.calmar_ratio:.3f}")
        print(f"   勝率: {performance_metrics.win_rate:.1%}")
        
        # パフォーマンス寄与度分析
        print("2. パフォーマンス寄与度分析...")
        attribution = system.performance_evaluator.calculate_performance_attribution(
            performance_data, test_weights
        )
        
        if attribution:
            print("   資産別寄与度:")
            for asset, contribution in sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"     {asset}: {contribution:.1%}")
        
        print("✓ パフォーマンス分析テストに成功しました")
        return True
        
    except Exception as e:
        print(f"✗ パフォーマンス分析テストでエラー: {e}")
        return False

def test_full_optimization_cycle(system, market_data, performance_data):
    """完全な最適化サイクルのテスト"""
    print("\n" + "="*50)
    print("完全最適化サイクルテスト")
    print("="*50)
    
    try:
        print("1. 初期重みの設定...")
        initial_weights = {
            'strategy_trend_following': 0.25,
            'strategy_mean_reversion': 0.25,
            'strategy_momentum': 0.25,
            'strategy_volatility_breakout': 0.25,
            'portfolio_stocks': 0.5,
            'portfolio_bonds': 0.3,
            'portfolio_commodities': 0.1,
            'portfolio_alternatives': 0.1,
            'meta_learning_rate': 1.0,
            'meta_volatility_scaling': 1.0,
            'meta_risk_aversion': 1.0,
            'meta_rebalancing_threshold': 0.05
        }
        
        # 初期パフォーマンスの評価
        initial_performance = system.performance_evaluator.evaluate_performance(
            performance_data, initial_weights
        )
        print(f"   初期パフォーマンススコア: {initial_performance.combined_score:.4f}")
        
        print("2. 最適化の実行...")
        # 完全な最適化サイクルの実行
        optimization_result = system.execute_optimal_learning(
            market_data=market_data,
            performance_data=performance_data,
            current_weights=initial_weights
        )
        
        print(f"   最適化後スコア: {optimization_result.performance_metrics.combined_score:.4f}")
        improvement = optimization_result.performance_metrics.combined_score - initial_performance.combined_score
        print(f"   改善度: {improvement:.4f} ({improvement/initial_performance.combined_score*100:+.1f}%)")
        
        print("3. 最適化結果の詳細分析...")
        print(f"   学習モード: {optimization_result.learning_mode.value}")
        print(f"   調整幅: {optimization_result.adjustment_magnitude:.4f}")
        print(f"   信頼度スコア: {optimization_result.confidence_score:.4f}")
        print(f"   実行時間: {optimization_result.execution_time:.2f}秒")
        print(f"   統合システム数: {len(optimization_result.integration_results)}")
        
        # 重みの変化の分析
        print("4. 重み変化の分析...")
        weight_changes = {}
        for key in initial_weights:
            if key in optimization_result.optimized_weights:
                change = optimization_result.optimized_weights[key] - initial_weights[key]
                weight_changes[key] = change
        
        # 最大変化を表示
        significant_changes = {k: v for k, v in weight_changes.items() if abs(v) > 0.01}
        if significant_changes:
            print("   有意な重み変化:")
            for weight_name, change in sorted(significant_changes.items(), key=lambda x: abs(x[1]), reverse=True):
                print(f"     {weight_name}: {change:+.3f}")
        else:
            print("   有意な重み変化なし（微調整レベル）")
        
        print("✓ 完全最適化サイクルテストに成功しました")
        return True, optimization_result
        
    except Exception as e:
        print(f"✗ 完全最適化サイクルテストでエラー: {e}")
        return False, None

def demonstrate_adaptive_learning(system, market_data, performance_data, optimization_result):
    """適応的学習のデモンストレーション"""
    print("\n" + "="*50)
    print("適応的学習デモンストレーション")
    print("="*50)
    
    try:
        print("1. 学習統計の取得...")
        learning_stats = system.learning_scheduler.get_learning_statistics()
        
        print(f"   総調整回数: {learning_stats.get('total_adjustments', 0)}")
        print(f"   最近のモード分布: {learning_stats.get('recent_mode_distribution', {})}")
        print(f"   平均調整幅: {learning_stats.get('average_adjustment_magnitude', 0):.4f}")
        print(f"   平均信頼度: {learning_stats.get('average_confidence_score', 0):.4f}")
        print(f"   成功率: {learning_stats.get('success_rate', 0):.1%}")
        
        print("2. メタパラメータの状態確認...")
        parameter_summary = system.meta_controller.get_parameter_summary()
        
        print("   現在のメタパラメータ:")
        for param_name, param_info in parameter_summary.items():
            if param_name != 'overall_statistics':
                current_val = param_info['current_value']
                default_val = param_info['default_value']
                deviation = param_info['deviation_from_default']
                print(f"     {param_name}: {current_val:.3f} "
                      f"(デフォルト: {default_val:.3f}, 偏差: {deviation:.1%})")
        
        print("3. 市場状態の分析...")
        # 市場状態の更新
        system.meta_controller.update_market_state(market_data, performance_data)
        market_state = system.meta_controller.market_state
        
        print(f"   ボラティリティレジーム: {market_state['volatility_regime']}")
        print(f"   トレンド強度: {market_state['trend_strength']:.4f}")
        print(f"   ストレスレベル: {market_state['stress_level']:.3f}")
        print(f"   流動性状況: {market_state['liquidity_condition']}")
        
        print("✓ 適応的学習デモンストレーションに成功しました")
        return True
        
    except Exception as e:
        print(f"✗ 適応的学習デモンストレーションでエラー: {e}")
        return False

def export_results(system):
    """結果のエクスポート"""
    print("\n" + "="*50)
    print("結果エクスポート")
    print("="*50)
    
    try:
        # 完全履歴のエクスポート
        print("1. 完全履歴のエクスポート...")
        export_path = system.export_complete_history(include_detailed_weights=True)
        print(f"   エクスポート先: {export_path}")
        
        # システムサマリーの表示
        print("2. 最終システムサマリー...")
        final_summary = system.get_system_summary()
        
        system_status = final_summary['system_status']
        print(f"   システム健康状態: {system_status['system_health']}")
        print(f"   総最適化回数: {system_status['total_optimizations']}")
        
        optimization_stats = final_summary.get('optimization_statistics', {})
        if optimization_stats:
            print(f"   最高パフォーマンス: {optimization_stats.get('best_performance', 0):.4f}")
            print(f"   平均パフォーマンス: {optimization_stats.get('average_performance', 0):.4f}")
            print(f"   収束率: {optimization_stats.get('convergence_rate', 0):.1%}")
        
        print("✓ 結果エクスポートに成功しました")
        return True
        
    except Exception as e:
        print(f"✗ 結果エクスポートでエラー: {e}")
        return False

def main():
    """メイン実行関数"""
    print("5-2-3 最適な重み付け比率の学習アルゴリズム デモンストレーション")
    print("="*80)
    
    # ロギングの設定
    setup_logging()
    
    # システムの初期化
    print("システムを初期化中...")
    try:
        system = OptimalWeightLearningSystem(workspace_path=str(project_root))
        print("✓ システムの初期化に成功しました")
    except Exception as e:
        print(f"✗ システム初期化エラー: {e}")
        return
    
    # サンプルデータの生成
    market_data, performance_data = generate_sample_data()
    
    # テスト実行
    test_results = {
        'basic_functionality': False,
        'learning_modes': False,
        'constraint_management': False,
        'integration_systems': False,
        'performance_analysis': False,
        'full_optimization_cycle': False,
        'adaptive_learning': False,
        'export_results': False
    }
    
    optimization_result = None
    
    # 各テストの実行
    test_results['basic_functionality'] = test_basic_functionality(system, market_data, performance_data)
    test_results['learning_modes'] = test_learning_modes(system, market_data, performance_data)
    test_results['constraint_management'] = test_constraint_management(system, market_data, performance_data)
    test_results['integration_systems'] = test_integration_systems(system)
    test_results['performance_analysis'] = test_performance_analysis(system, market_data, performance_data)
    
    cycle_success, optimization_result = test_full_optimization_cycle(system, market_data, performance_data)
    test_results['full_optimization_cycle'] = cycle_success
    
    if optimization_result:
        test_results['adaptive_learning'] = demonstrate_adaptive_learning(
            system, market_data, performance_data, optimization_result
        )
    
    test_results['export_results'] = export_results(system)
    
    # 最終結果のサマリー
    print("\n" + "="*80)
    print("テスト結果サマリー")
    print("="*80)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"総合結果: {passed_tests}/{total_tests} テストが成功")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n詳細結果:")
    for test_name, result in test_results.items():
        status = "✓ 成功" if result else "✗ 失敗"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 すべてのテストに成功しました！")
        print("5-2-3 最適な重み付け比率の学習アルゴリズムが正常に動作しています。")
    else:
        print(f"\n⚠️  {total_tests - passed_tests}件のテストが失敗しました。")
        print("ログファイルで詳細を確認してください。")
    
    print(f"\n詳細ログ: demo_5_2_3_weight_learning.log")
    print("実装完了: 5-2-3 最適な重み付け比率の学習アルゴリズム")

if __name__ == "__main__":
    main()

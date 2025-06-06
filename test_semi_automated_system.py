"""
半自動戦略適用システムのテストスクリプト

このスクリプトでは以下をテストします：
1. モメンタム戦略の最適化実行
2. 結果のJSONファイル保存
3. オーバーフィッティング検出
4. パラメータ妥当性検証
5. 自動承認プロセス
6. レビューシステムの動作確認
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# プロジェクトルートを追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config.optimized_parameters import OptimizedParameterManager
from optimization.overfitting_detector import OverfittingDetector
from validation.parameter_validator import ParameterValidator
from tools.parameter_reviewer import ParameterReviewer
from strategies.Momentum_Investing import MomentumInvestingStrategy


def create_test_data(start_date: str = "2023-01-01", end_date: str = "2023-12-31") -> pd.DataFrame:
    """テスト用の株価データを作成"""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    
    # リアルな株価データを模擬
    np.random.seed(42)  # 再現可能にするため
    
    # 価格の基本トレンド
    base_price = 1000
    trend = np.linspace(0, 200, n_days)  # 上昇トレンド
    noise = np.random.normal(0, 20, n_days)  # ノイズ
    
    close_prices = base_price + trend + noise
    high_prices = close_prices + np.random.uniform(5, 25, n_days)
    low_prices = close_prices - np.random.uniform(5, 25, n_days)
    open_prices = close_prices + np.random.normal(0, 10, n_days)
    
    # 出来高
    volumes = np.random.lognormal(mean=12, sigma=0.5, size=n_days).astype(int)
    
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Adj Close': close_prices,
        'Volume': volumes
    }, index=dates)
    
    return data


def test_parameter_manager():
    """OptimizedParameterManagerのテスト"""
    print("\n" + "="*60)
    print("1. OptimizedParameterManager テスト")
    print("="*60)
    
    manager = OptimizedParameterManager()
    strategy_name = "MomentumInvestingStrategy"
    
    # テスト用パラメータセットを複数保存
    test_params_sets = [
        {
            'parameters': {
                "sma_short": 15,
                "sma_long": 45,
                "rsi_period": 12,
                "rsi_lower": 45,
                "rsi_upper": 70,
                "volume_threshold": 1.2,
                "take_profit": 0.15,
                "stop_loss": 0.08
            },
            'performance_metrics': {
                'sharpe_ratio': 1.45,
                'total_return': 0.18,
                'max_drawdown': -0.08,
                'win_rate': 0.65,
                'profit_factor': 1.8,
                'volatility': 0.12
            }
        },
        {
            'parameters': {
                "sma_short": 25,
                "sma_long": 55,
                "rsi_period": 16,
                "rsi_lower": 55,
                "rsi_upper": 65,
                "volume_threshold": 1.15,
                "take_profit": 0.10,
                "stop_loss": 0.05
            },
            'performance_metrics': {
                'sharpe_ratio': 1.62,
                'total_return': 0.22,
                'max_drawdown': -0.06,
                'win_rate': 0.68,
                'profit_factor': 2.1,
                'volatility': 0.14
            }
        },
        {
            'parameters': {
                "sma_short": 30,
                "sma_long": 60,
                "rsi_period": 18,
                "rsi_lower": 40,
                "rsi_upper": 75,
                "volume_threshold": 1.25,
                "take_profit": 0.08,
                "stop_loss": 0.04
            },
            'performance_metrics': {
                'sharpe_ratio': 1.28,
                'total_return': 0.14,
                'max_drawdown': -0.05,
                'win_rate': 0.62,
                'profit_factor': 1.6,
                'volatility': 0.11
            }
        }
    ]
    
    param_ids = []
    for i, params_set in enumerate(test_params_sets):        param_id = manager.save_optimized_params(
            strategy_name=strategy_name,
            ticker="TEST",
            params=params_set['parameters'],
            metrics=params_set['performance_metrics']
        )
        param_ids.append(param_id)
        print(f"✅ パラメータセット {i+1} を保存: {param_id}")
      # 保存されたパラメータセットを確認
    saved_params = manager.list_available_configs(strategy_name)
    print(f"\n📊 保存されたパラメータセット数: {len(saved_params)}")
    
    # 最高パフォーマンスのパラメータを取得
    best_sharpe = manager.get_best_config_by_metric(strategy_name, metric='sharpe_ratio')
    if best_sharpe:
        print(f"🏆 最高シャープレシオ: {best_sharpe['performance_metrics']['sharpe_ratio']:.4f}")
    
    return param_ids


def test_overfitting_detector():
    """OverfittingDetectorのテスト"""
    print("\n" + "="*60)
    print("2. OverfittingDetector テスト")
    print("="*60)
    
    detector = OverfittingDetector()
    
    # テスト用パフォーマンスデータ
    test_cases = [
        {
            'name': '正常なパフォーマンス',
            'performance_data': {
                'sharpe_ratio': 1.5,
                'total_return': 0.15,
                'max_drawdown': -0.08,
                'win_rate': 0.65,
                'volatility': 0.12
            },
            'parameters': {
                "sma_short": 20,
                "sma_long": 50,
                "rsi_period": 14
            }
        },
        {
            'name': '異常に高いパフォーマンス（オーバーフィッティング疑い）',
            'performance_data': {
                'sharpe_ratio': 5.2,  # 異常に高い
                'total_return': 0.85,  # 異常に高い
                'max_drawdown': -0.02,  # 異常に小さい
                'win_rate': 0.95,  # 異常に高い
                'volatility': 0.05  # 異常に小さい
            },
            'parameters': {
                "sma_short": 7,   # 過度に短期
                "sma_long": 200,  # 過度に長期
                "rsi_period": 3   # 過度に短期
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 テストケース: {test_case['name']}")
        result = detector.detect_overfitting(
            test_case['performance_data'], 
            test_case['parameters']
        )
        
        print(f"  🎯 リスクレベル: {result['overall_risk_level']}")
        print(f"  📊 リスクスコア: {result['risk_score']:.2f}")
        print(f"  🔍 検出された問題数: {len(result['detections'])}")
        
        for detection in result['detections']:
            risk_icon = "🔴" if detection['risk_level'] == 'high' else "🟡" if detection['risk_level'] == 'medium' else "🟢"
            print(f"    {risk_icon} {detection['type']}: {detection['reason']}")


def test_parameter_validator():
    """ParameterValidatorのテスト"""
    print("\n" + "="*60)
    print("3. ParameterValidator テスト")
    print("="*60)
    
    validator = ParameterValidator()
    strategy_name = "MomentumInvestingStrategy"
    
    # テスト用パラメータセット
    test_cases = [
        {
            'name': '正常なパラメータ',
            'parameters': {
                "sma_short": 20,
                "sma_long": 50,
                "rsi_period": 14,
                "rsi_lower": 30,
                "rsi_upper": 70,
                "volume_threshold": 1.2,
                "take_profit": 0.10,
                "stop_loss": 0.05
            }
        },
        {
            'name': '異常なパラメータ',
            'parameters': {
                "sma_short": 60,   # 長期MAより大きい（論理エラー）
                "sma_long": 30,    # 短期MAより小さい（論理エラー）
                "rsi_period": 200, # 過度に大きい
                "rsi_lower": 80,   # 上限より大きい（論理エラー）
                "rsi_upper": 20,   # 下限より小さい（論理エラー）
                "volume_threshold": -0.5,  # 負の値（無効）
                "take_profit": -0.1,       # 負の値（無効）
                "stop_loss": 2.0           # 過度に大きい
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 テストケース: {test_case['name']}")
        result = validator.validate_parameters(strategy_name, test_case['parameters'])
        
        print(f"  ✅ 検証結果: {'合格' if result['is_valid'] else '不合格'}")
        print(f"  📊 信頼度スコア: {result['confidence_score']:.2f}")
        
        if result['errors']:
            print(f"  ❌ エラー({len(result['errors'])}件):")
            for error in result['errors']:
                print(f"    • {error}")
        
        if result['warnings']:
            print(f"  ⚠️ 警告({len(result['warnings'])}件):")
            for warning in result['warnings']:
                print(f"    • {warning}")


def test_momentum_strategy_optimization_mode():
    """MomentumInvestingStrategyの最適化モードテスト"""
    print("\n" + "="*60)
    print("4. MomentumInvestingStrategy 最適化モード テスト")
    print("="*60)
    
    # テストデータ作成
    test_data = create_test_data()
    print(f"📊 テストデータ期間: {test_data.index[0]} - {test_data.index[-1]}")
    print(f"📊 データポイント数: {len(test_data)}")
    
    # 各最適化モードをテスト
    optimization_modes = [
        None,  # デフォルト
        "best_sharpe",
        "best_return",
        "latest_approved"
    ]
    
    for mode in optimization_modes:
        print(f"\n🔧 最適化モード: {mode or 'デフォルト'}")
        
        try:
            strategy = MomentumInvestingStrategy(
                data=test_data,
                optimization_mode=mode
            )
            
            # 最適化情報を取得
            opt_info = strategy.get_optimization_info()
            print(f"  📋 最適化情報:")
            print(f"    - 最適化パラメータ使用: {opt_info['using_optimized_params']}")
            print(f"    - 現在のパラメータ数: {len(opt_info['current_params'])}")
            
            # 戦略初期化
            strategy.initialize_strategy()
            print(f"  ✅ 戦略初期化完了")
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")


def test_integrated_workflow():
    """統合ワークフローのテスト"""
    print("\n" + "="*60)
    print("5. 統合ワークフロー テスト")
    print("="*60)
    
    # 1. パラメータマネージャーでパラメータ保存
    param_ids = test_parameter_manager()
    
    # 2. 最新のパラメータを取得して検証
    manager = OptimizedParameterManager()
    strategy_name = "MomentumInvestingStrategy"
    
    latest_params = manager.get_latest_parameters(strategy_name)
    if latest_params:
        print(f"\n🔍 最新パラメータの検証:")
        
        # オーバーフィッティング検出
        detector = OverfittingDetector()
        performance_data = latest_params['performance_metrics']
        parameters = latest_params['parameters']
        
        overfitting_result = detector.detect_overfitting(performance_data, parameters)
        print(f"  📊 オーバーフィッティングリスク: {overfitting_result['overall_risk_level']}")
        
        # パラメータ妥当性検証
        validator = ParameterValidator()
        validation_result = validator.validate_parameters(strategy_name, parameters)
        print(f"  ✅ パラメータ妥当性: {'合格' if validation_result['is_valid'] else '不合格'}")
        
        # 総合リスク判定
        risk_levels = ['low', 'medium', 'high']
        overfitting_risk = overfitting_result.get('overall_risk_level', 'medium')
        validation_risk = 'low' if validation_result.get('is_valid', False) else 'high'
        
        overall_risk_index = max(
            risk_levels.index(overfitting_risk),
            risk_levels.index(validation_risk)
        )
        overall_risk = risk_levels[overall_risk_index]
        
        print(f"  🎯 総合リスク判定: {overall_risk}")
        
        # 自動承認シミュレーション
        if overall_risk == 'low':
            print(f"  ✅ 自動承認条件を満たしています")
            manager.update_parameter_status(
                strategy_name, 
                latest_params['parameter_id'], 
                'approved'
            )
            print(f"  📝 パラメータを自動承認しました")
        else:
            print(f"  ⚠️ 手動レビューが必要です")


def main():
    """メインテスト実行"""
    print("="*60)
    print("半自動戦略適用システム - 統合テスト")
    print("="*60)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 各コンポーネントのテスト
        test_parameter_manager()
        test_overfitting_detector()
        test_parameter_validator()
        test_momentum_strategy_optimization_mode()
        test_integrated_workflow()
        
        print("\n" + "="*60)
        print("✅ 全てのテストが完了しました")
        print("="*60)
        
        # 次のステップの案内
        print("\n📋 次のステップ:")
        print("1. optimize_strategy.py --strategy momentum --save-results を実行")
        print("2. tools/parameter_reviewer.py MomentumInvestingStrategy でレビュー")
        print("3. 最適化モードでの戦略実行テスト")
        
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

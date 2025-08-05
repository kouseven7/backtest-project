"""
Strategy Characteristics Manager のテストスクリプト
パラメータ履歴機能を含む包括的なテスト
"""

import sys
import os
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

from config.strategy_characteristics_manager import StrategyCharacteristicsManager

def test_parameter_history():
    """パラメータ履歴機能のテスト"""
    print("=== パラメータ履歴機能テスト ===")
    
    # マネージャーの初期化
    manager = StrategyCharacteristicsManager()
    
    # サンプルパラメータを追加
    print("\n1. パラメータバージョンの追加...")
    
    # バージョン1
    params_v1 = {
        "vwap_lower_threshold": 0.995,
        "vwap_upper_threshold": 1.005,
        "volume_increase_threshold": 1.2,
        "stop_loss": 0.02,
        "take_profit": 0.03
    }
    
    metrics_v1 = {
        "sharpe_ratio": 0.8,
        "sortino_ratio": 1.1,
        "win_rate": 0.65,
        "expectancy": 0.02,
        "max_drawdown": 0.12
    }
    
    manager.add_parameter_version(
        "VWAPBounceStrategy", 
        params_v1, 
        metrics_v1,
        optimization_info={"optimizer": "grid_search", "iterations": 1000}
    )
    
    # バージョン2（改良版）
    params_v2 = {
        "vwap_lower_threshold": 0.998,
        "vwap_upper_threshold": 1.002,
        "volume_increase_threshold": 1.1,
        "stop_loss": 0.015,
        "take_profit": 0.035
    }
    
    metrics_v2 = {
        "sharpe_ratio": 1.2,
        "sortino_ratio": 1.5,
        "win_rate": 0.68,
        "expectancy": 0.025,
        "max_drawdown": 0.10
    }
    
    manager.add_parameter_version(
        "VWAPBounceStrategy", 
        params_v2, 
        metrics_v2,
        optimization_info={"optimizer": "bayesian", "iterations": 500}
    )
    
    # バージョン3（さらなる改良版）
    params_v3 = {
        "vwap_lower_threshold": 0.997,
        "vwap_upper_threshold": 1.003,
        "volume_increase_threshold": 1.15,
        "stop_loss": 0.018,
        "take_profit": 0.032
    }
    
    metrics_v3 = {
        "sharpe_ratio": 1.4,
        "sortino_ratio": 1.8,
        "win_rate": 0.72,
        "expectancy": 0.03,
        "max_drawdown": 0.09
    }
    
    manager.add_parameter_version(
        "VWAPBounceStrategy", 
        params_v3, 
        metrics_v3,
        optimization_info={"optimizer": "genetic_algorithm", "iterations": 2000}
    )
    
    print("✓ 3つのパラメータバージョンを追加しました")
    
    # 履歴取得テスト
    print("\n2. パラメータ履歴の取得...")
    history = manager.get_parameter_history("VWAPBounceStrategy", limit=5)
    print(f"取得した履歴数: {len(history)}")
    
    for entry in history:
        print(f"  バージョン {entry['version']}: シャープレシオ {entry['performance_metrics']['sharpe_ratio']:.2f}")
    
    # 最良パラメータ取得テスト
    print("\n3. 最良パラメータの取得...")
    best_params = manager.get_best_parameters("VWAPBounceStrategy")
    if best_params:
        print(f"最良バージョン: {best_params['version']}")
        print(f"最良シャープレシオ: {best_params['performance_metrics']['sharpe_ratio']:.2f}")
        print("最良パラメータ:")
        for key, value in best_params['parameters'].items():
            print(f"  {key}: {value}")
    
    # バージョン比較テスト
    print("\n4. パラメータバージョン比較...")
    comparison = manager.compare_parameter_versions("VWAPBounceStrategy", "1.1", "1.3")
    if "error" not in comparison:
        print("パラメータの差分:")
        for param, diff in comparison["parameter_differences"].items():
            print(f"  {param}: {diff['v1']} → {diff['v2']}")
        
        print("\nパフォーマンスの差分:")
        for metric, diff in comparison["performance_differences"].items():
            improvement = "↑" if diff["improvement"] else "↓"
            print(f"  {metric}: {diff['v1']:.3f} → {diff['v2']:.3f} {improvement}")
    
    # 戦略レポート生成テスト
    print("\n5. 戦略レポート生成...")
    report = manager.generate_strategy_report("VWAPBounceStrategy")
    print(report)
    
    return True

def test_multiple_strategies():
    """複数戦略のメタデータ作成テスト"""
    print("\n=== 複数戦略メタデータ作成テスト ===")
    
    manager = StrategyCharacteristicsManager()
    
    # MomentumInvestingStrategyのメタデータ作成
    momentum_trend_performance = {
        "uptrend": {
            "suitability_score": 0.9,
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.08,
            "win_rate": 0.75,
            "expectancy": 0.04,
            "sample_size": 60,
            "data_period": "2023-01-01_to_2024-12-31"
        },
        "downtrend": {
            "suitability_score": 0.2,
            "sharpe_ratio": -0.5,
            "max_drawdown": 0.30,
            "win_rate": 0.25,
            "expectancy": -0.02,
            "sample_size": 25,
            "data_period": "2023-01-01_to_2024-12-31"
        },
        "range-bound": {
            "suitability_score": 0.4,
            "sharpe_ratio": 0.3,
            "max_drawdown": 0.15,
            "win_rate": 0.50,
            "expectancy": 0.005,
            "sample_size": 40,
            "data_period": "2023-01-01_to_2024-12-31"
        }
    }
    
    momentum_metadata = manager.create_strategy_metadata(
        strategy_id="MomentumInvestingStrategy",
        trend_performance=momentum_trend_performance,
        include_param_history=True
    )
    
    momentum_filepath = manager.save_metadata(momentum_metadata)
    print(f"MomentumInvestingStrategy メタデータ作成: {os.path.basename(momentum_filepath)}")
    
    # ContrarianStrategyのメタデータ作成
    contrarian_trend_performance = {
        "uptrend": {
            "suitability_score": 0.3,
            "sharpe_ratio": 0.2,
            "max_drawdown": 0.18,
            "win_rate": 0.45,
            "expectancy": 0.005,
            "sample_size": 35,
            "data_period": "2023-01-01_to_2024-12-31"
        },
        "downtrend": {
            "suitability_score": 0.7,
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.12,
            "win_rate": 0.65,
            "expectancy": 0.025,
            "sample_size": 45,
            "data_period": "2023-01-01_to_2024-12-31"
        },
        "range-bound": {
            "suitability_score": 0.8,
            "sharpe_ratio": 1.3,
            "max_drawdown": 0.10,
            "win_rate": 0.70,
            "expectancy": 0.03,
            "sample_size": 55,
            "data_period": "2023-01-01_to_2024-12-31"
        }
    }
    
    contrarian_metadata = manager.create_strategy_metadata(
        strategy_id="ContrarianStrategy",
        trend_performance=contrarian_trend_performance,
        include_param_history=True
    )
    
    contrarian_filepath = manager.save_metadata(contrarian_metadata)
    print(f"ContrarianStrategy メタデータ作成: {os.path.basename(contrarian_filepath)}")
    
    # 全戦略一覧の表示
    print(f"\n利用可能な戦略一覧: {manager.list_strategies()}")
    
    return True

if __name__ == "__main__":
    print("Strategy Characteristics Manager 包括テスト開始")
    
    # テスト実行
    try:
        # パラメータ履歴機能テスト
        test_parameter_history()
        
        # 複数戦略テスト
        test_multiple_strategies()
        
        print("\n" + "="*60)
        print("✓ すべてのテストが正常に完了しました")
        print("✓ 戦略特性メタデータスキーマが正常に動作しています")
        print("✓ パラメータ履歴機能が正常に動作しています")
        
    except Exception as e:
        print(f"\n✗ テスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()

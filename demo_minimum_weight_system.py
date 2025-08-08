"""
デモ最小重みシステム実行スクリプト
フェーズ4A3: バックテストvs実運用比較分析器のテスト実行
"""

import os
import sys
import json
import logging
from datetime import datetime

# プロジェクトルートを追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def setup_demo_logger():
    """デモ用ロガー設定"""
    logger = logging.getLogger("Demo_BacktestVsLive")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def create_demo_directories():
    """デモ用ディレクトリ作成"""
    directories = [
        'config/comparison',
        'src/analysis/comparison',
        'reports',
        'reports/charts',
        'backtest_results/improved_results',
        'logs/performance_monitoring',
        'logs/paper_trading'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"ディレクトリ作成: {directory}")

def create_sample_backtest_data():
    """サンプルバックテストデータ作成"""
    sample_data = {
        "VWAPStrategy": {
            "basic_metrics": {
                "total_pnl": 15420.50,
                "total_trades": 125,
                "win_rate": 0.664,
                "avg_profit": 123.36,
                "avg_loss": -89.24
            },
            "risk_metrics": {
                "max_drawdown": -0.087,
                "sharpe_ratio": 1.42,
                "volatility": 0.134,
                "calmar_ratio": 2.15
            }
        },
        "MeanReversionStrategy": {
            "basic_metrics": {
                "total_pnl": 8934.20,
                "total_trades": 89,
                "win_rate": 0.618,
                "avg_profit": 156.78,
                "avg_loss": -112.45
            },
            "risk_metrics": {
                "max_drawdown": -0.125,
                "sharpe_ratio": 1.18,
                "volatility": 0.156,
                "calmar_ratio": 1.85
            }
        },
        "TrendFollowingStrategy": {
            "basic_metrics": {
                "total_pnl": 12678.90,
                "total_trades": 67,
                "win_rate": 0.597,
                "avg_profit": 234.56,
                "avg_loss": -178.23
            },
            "risk_metrics": {
                "max_drawdown": -0.098,
                "sharpe_ratio": 1.35,
                "volatility": 0.142,
                "calmar_ratio": 2.05
            }
        }
    }
    
    # JSONファイルで作成
    for strategy_name, data in sample_data.items():
        filepath = f"backtest_results/improved_results/{strategy_name}_backtest_results.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"サンプルバックテストデータ作成: {filepath}")
    
    return sample_data

def create_sample_live_data():
    """サンプル実運用データ作成"""
    sample_data = {
        "VWAPStrategy": {
            "basic_metrics": {
                "total_pnl": 14235.80,  # バックテストより少し低い
                "total_trades": 118,
                "win_rate": 0.627,  # バックテストより少し低い
                "avg_profit": 120.55,
                "avg_loss": -95.67
            },
            "risk_metrics": {
                "max_drawdown": -0.105,  # バックテストより悪い
                "sharpe_ratio": 1.28,  # バックテストより低い
                "volatility": 0.148,
                "calmar_ratio": 1.95
            }
        },
        "MeanReversionStrategy": {
            "basic_metrics": {
                "total_pnl": 9456.70,  # バックテストより良い
                "total_trades": 93,
                "win_rate": 0.645,  # バックテストより良い
                "avg_profit": 161.23,
                "avg_loss": -108.90
            },
            "risk_metrics": {
                "max_drawdown": -0.118,  # バックテストより良い
                "sharpe_ratio": 1.25,
                "volatility": 0.149,
                "calmar_ratio": 1.92
            }
        },
        "TrendFollowingStrategy": {
            "basic_metrics": {
                "total_pnl": 11234.50,  # バックテストより低い
                "total_trades": 72,
                "win_rate": 0.583,  # バックテストより低い
                "avg_profit": 225.67,
                "avg_loss": -185.45
            },
            "risk_metrics": {
                "max_drawdown": -0.112,  # バックテストより悪い
                "sharpe_ratio": 1.22,  # バックテストより低い
                "volatility": 0.155,
                "calmar_ratio": 1.88
            }
        }
    }
    
    # パフォーマンスモニタリングログファイル作成
    for strategy_name, data in sample_data.items():
        # JSON形式でログファイル作成
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_name,
            "performance_data": data,
            "monitoring_period": "2024-01-01 to 2024-12-31",
            "data_source": "live_trading"
        }
        
        filepath = f"logs/performance_monitoring/{strategy_name}_performance.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"サンプル実運用データ作成: {filepath}")
    
    return sample_data

def run_simple_comparison():
    """簡易比較分析実行"""
    logger = setup_demo_logger()
    logger.info("簡易比較分析開始")
    
    try:
        # サンプルデータ読み込み
        backtest_dir = "backtest_results/improved_results"
        live_dir = "logs/performance_monitoring"
        
        backtest_data = {}
        live_data = {}
        
        # バックテストデータ読み込み
        for filename in os.listdir(backtest_dir):
            if filename.endswith('.json'):
                strategy_name = filename.replace('_backtest_results.json', '')
                with open(os.path.join(backtest_dir, filename), 'r', encoding='utf-8') as f:
                    backtest_data[strategy_name] = json.load(f)
        
        # 実運用データ読み込み
        for filename in os.listdir(live_dir):
            if filename.endswith('.json'):
                strategy_name = filename.replace('_performance.json', '')
                with open(os.path.join(live_dir, filename), 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    live_data[strategy_name] = log_data.get('performance_data', {})
        
        # 簡易比較実行
        logger.info(f"バックテスト戦略数: {len(backtest_data)}")
        logger.info(f"実運用戦略数: {len(live_data)}")
        
        # 共通戦略抽出
        common_strategies = set(backtest_data.keys()).intersection(set(live_data.keys()))
        logger.info(f"共通戦略数: {len(common_strategies)}")
        
        # 戦略別比較
        comparison_results = {}
        
        for strategy_name in common_strategies:
            bt_data = backtest_data[strategy_name]
            live_data_strategy = live_data[strategy_name]
            
            strategy_comparison = {}
            
            # 基本メトリクス比較
            bt_basic = bt_data.get('basic_metrics', {})
            live_basic = live_data_strategy.get('basic_metrics', {})
            
            for metric in ['total_pnl', 'win_rate', 'total_trades']:
                if metric in bt_basic and metric in live_basic:
                    bt_val = bt_basic[metric]
                    live_val = live_basic[metric]
                    diff = live_val - bt_val
                    rel_diff = (diff / bt_val) if bt_val != 0 else 0
                    
                    strategy_comparison[metric] = {
                        'backtest': bt_val,
                        'live': live_val,
                        'difference': diff,
                        'relative_difference_pct': rel_diff * 100
                    }
            
            comparison_results[strategy_name] = strategy_comparison
        
        # 結果出力
        output_file = f"reports/simple_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        final_results = {
            "analysis_type": "simple_comparison",
            "timestamp": datetime.now().isoformat(),
            "strategies_compared": list(common_strategies),
            "comparison_results": comparison_results,
            "summary": {
                "total_strategies": len(common_strategies),
                "backtest_total_pnl": sum(
                    bt_data.get('basic_metrics', {}).get('total_pnl', 0)
                    for bt_data in backtest_data.values()
                ),
                "live_total_pnl": sum(
                    live_data_strategy.get('basic_metrics', {}).get('total_pnl', 0)
                    for live_data_strategy in live_data.values()
                )
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"比較結果保存: {output_file}")
        
        # コンソール出力
        print("\n=== 簡易比較分析結果 ===")
        print(f"分析実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"比較戦略数: {len(common_strategies)}")
        print("")
        
        for strategy_name, comparison in comparison_results.items():
            print(f"【{strategy_name}】")
            for metric, data in comparison.items():
                print(f"  {metric}:")
                print(f"    バックテスト: {data['backtest']:.4f}")
                print(f"    実運用: {data['live']:.4f}")
                print(f"    相対差分: {data['relative_difference_pct']:.2f}%")
            print("")
        
        logger.info("簡易比較分析完了")
        return final_results
        
    except Exception as e:
        logger.error(f"簡易比較分析エラー: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def main():
    """メイン実行"""
    print("=== デモ最小重みシステム実行 ===")
    print("フェーズ4A3: バックテストvs実運用比較分析器")
    print("")
    
    logger = setup_demo_logger()
    
    try:
        # 1. ディレクトリ作成
        print("1. ディレクトリ構造作成中...")
        create_demo_directories()
        print("")
        
        # 2. サンプルデータ作成
        print("2. サンプルデータ作成中...")
        backtest_data = create_sample_backtest_data()
        live_data = create_sample_live_data()
        print("")
        
        # 3. 簡易比較分析実行
        print("3. 簡易比較分析実行中...")
        results = run_simple_comparison()
        print("")
        
        if 'error' not in results:
            print("=== 分析成功 ===")
            summary = results.get('summary', {})
            print(f"分析戦略数: {summary.get('total_strategies', 0)}")
            print(f"バックテスト総PnL: {summary.get('backtest_total_pnl', 0):.2f}")
            print(f"実運用総PnL: {summary.get('live_total_pnl', 0):.2f}")
            
            total_bt = summary.get('backtest_total_pnl', 0)
            total_live = summary.get('live_total_pnl', 0)
            if total_bt != 0:
                total_diff = ((total_live - total_bt) / total_bt) * 100
                print(f"総合パフォーマンス差: {total_diff:.2f}%")
        else:
            print(f"分析エラー: {results['error']}")
        
        print("\n=== 実行完了 ===")
        
    except Exception as e:
        logger.error(f"メイン実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

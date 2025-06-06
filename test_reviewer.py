"""
parameter_reviewerのテスト用スクリプト
"""
import os
import json
from config.optimized_parameters import OptimizedParameterManager

def setup_test_data():
    """テスト用のレビュー待ちデータを準備"""
    manager = OptimizedParameterManager()
    
    # 既存ファイルの状態を確認
    configs = manager.list_available_configs(status="pending_review")
    
    print(f"現在のレビュー待ちファイル: {len(configs)}件")
    for config in configs:
        print(f"  - {config['filename']} (銘柄: {config.get('ticker', 'N/A')})")
    
    # もしレビュー待ちがない場合、テスト用データを作成
    if not configs:
        print("\nテスト用のレビュー待ちデータを作成します...")
        
        test_params = {
            "sma_short": 20,
            "sma_long": 60,
            "rsi_period": 14,
            "rsi_lower": 40,
            "rsi_upper": 70,
            "take_profit": 0.15,
            "stop_loss": 0.08,
            "trailing_stop": 0.05,
            "volume_threshold": 1.2,
            "max_hold_days": 12
        }
        
        test_metrics = {
            "sharpe_ratio": 2.1,
            "sortino_ratio": 2.8,
            "total_return": 0.28,
            "max_drawdown": -0.12,
            "win_rate": 0.58,
            "total_trades": 35
        }
        
        # テスト用ファイルを保存
        config_path = manager.save_optimized_params(
            strategy_name="momentum",
            ticker="TEST_REVIEW",
            params=test_params,
            metrics=test_metrics,
            status="pending_review"
        )
        print(f"テスト用ファイルを作成しました: {config_path}")

if __name__ == "__main__":
    setup_test_data()
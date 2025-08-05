"""
エラーハンドリングシステム デモンストレーション
フェーズ3: 実践環境準備 - エラーハンドリング強化の動作確認
"""

import time
import random
from pathlib import Path
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# エラーハンドリングシステムインポート
from src.utils.exception_handler import (
    UnifiedExceptionHandler, StrategyError, DataError, SystemError,
    handle_strategy_error, handle_data_error, handle_system_error
)
from src.utils.error_recovery import (
    ErrorRecoveryManager, retry_with_strategy, fallback_recovery
)
from src.utils.logger_setup import (
    get_logger_manager, get_strategy_logger, log_strategy_performance
)
from src.utils.monitoring_agent import (
    get_monitoring_agent, start_system_monitoring, stop_system_monitoring,
    report_error, report_performance_issue
)


def demo_basic_error_handling():
    """基本エラーハンドリング デモ"""
    print("=" * 60)
    print("1. 基本エラーハンドリング デモ")
    print("=" * 60)
    
    # 戦略エラー
    try:
        raise StrategyError("momentum_strategy", "移動平均計算エラー")
    except StrategyError as e:
        result = handle_strategy_error("momentum_strategy", e, {
            "timeframe": "1D",
            "symbol": "USDJPY"
        })
        print(f"戦略エラー処理結果: {result['error_message']}")
        print(f"復旧試行: {result['recovery_attempted']}")
    
    # データエラー
    try:
        raise DataError("API接続タイムアウト")
    except DataError as e:
        result = handle_data_error(e, {
            "api_endpoint": "https://api.example.com/data",
            "timeout": 30
        })
        print(f"データエラー処理結果: {result['error_message']}")
    
    # システムエラー
    try:
        raise SystemError("メモリ不足")
    except SystemError as e:
        result = handle_system_error(e, {
            "available_memory": "512MB",
            "required_memory": "1GB"
        })
        print(f"システムエラー処理結果: {result['error_message']}")
    
    print()


def demo_error_recovery():
    """エラー復旧 デモ"""
    print("=" * 60)
    print("2. エラー復旧システム デモ")
    print("=" * 60)
    
    # リトライ復旧デモ
    attempt_count = 0
    
    def unreliable_function():
        nonlocal attempt_count
        attempt_count += 1
        print(f"  実行試行 {attempt_count}")
        
        if attempt_count < 3:
            raise Exception(f"一時的エラー (試行 {attempt_count})")
        
        return f"成功! (試行 {attempt_count})"
    
    print("リトライ戦略による復旧:")
    try:
        result = retry_with_strategy(unreliable_function, "strategy_errors")
        print(f"復旧成功: {result}")
    except Exception as e:
        print(f"復旧失敗: {e}")
    
    # フォールバック復旧デモ
    print("\nフォールバック戦略による復旧:")
    
    def primary_function():
        raise Exception("主要機能エラー")
    
    def fallback_function_1():
        if random.random() < 0.3:  # 30%の確率で失敗
            raise Exception("フォールバック1失敗")
        return "フォールバック1成功"
    
    def fallback_function_2():
        return "フォールバック2成功"
    
    try:
        result = fallback_recovery(
            primary_function,
            [fallback_function_1, fallback_function_2],
            "data_errors"
        )
        print(f"フォールバック復旧成功: {result}")
    except Exception as e:
        print(f"フォールバック復旧失敗: {e}")
    
    print()


def demo_enhanced_logging():
    """強化ロギング デモ"""
    print("=" * 60)
    print("3. 強化ロギングシステム デモ")
    print("=" * 60)
    
    # ロガー管理取得
    logger_manager = get_logger_manager()
    
    # 戦略別ログ
    strategy_logger = get_strategy_logger("demo_strategy")
    strategy_logger.info("戦略実行開始", extra={
        'strategy_name': 'demo_strategy',
        'execution_time': 1.23,
        'memory_usage': 45.6
    })
    
    # パフォーマンスログ
    log_strategy_performance(
        "demo_strategy", 
        execution_time=2.5, 
        memory_usage=67.8,
        additional_info="デモ実行完了"
    )
    
    # エラー分析ログ
    test_error = Exception("デモエラー")
    logger_manager.log_error_with_analysis(
        test_error,
        {"demo": True, "component": "logging_demo"},
        "demo_strategy"
    )
    
    # ログ統計表示
    log_stats = logger_manager.get_log_statistics()
    print(f"総ログ数: {log_stats['total_logs']}")
    print(f"エラー数: {log_stats['error_count']}")
    print(f"戦略ログ: {list(log_stats['strategy_logs'].keys())}")
    
    print()


def demo_monitoring_system():
    """監視システム デモ"""
    print("=" * 60)
    print("4. 監視システム デモ")
    print("=" * 60)
    
    # 監視エージェント取得
    monitoring_agent = get_monitoring_agent()
    
    # カスタムアラートルール追加
    def demo_alert_condition(context):
        return context.get('demo_value', 0) > 5
    
    monitoring_agent.add_alert_rule(
        "demo_alert",
        demo_alert_condition,
        "WARNING",
        cooldown_minutes=1
    )
    
    # システム監視開始
    print("システム監視開始...")
    start_system_monitoring()
    
    # 監視データ生成
    print("監視イベント生成中...")
    
    # エラー報告
    report_error(
        Exception("デモエラー"),
        {"demo": True, "timestamp": time.time()},
        "demo_strategy"
    )
    
    # パフォーマンス問題報告
    report_performance_issue(
        "execution_time",
        8.5,  # 実際の値
        5.0,  # 閾値
        {"strategy": "demo_strategy", "operation": "backtest"}
    )
    
    # カスタムイベント報告
    monitoring_agent.report_event(
        "demo_event",
        "INFO",
        "デモシステムイベント",
        {"demo": True, "step": 4}
    )
    
    # 少し待機
    time.sleep(2)
    
    # 監視統計表示
    monitoring_stats = monitoring_agent.get_monitoring_statistics()
    print(f"監視統計:")
    print(f"  総アラート数: {monitoring_stats['alerts_triggered']}")
    print(f"  ルール評価回数: {monitoring_stats['rules_evaluated']}")
    print(f"  メトリクス収集回数: {monitoring_stats['metrics_collected']}")
    
    # システム監視停止
    print("システム監視停止...")
    stop_system_monitoring()
    
    print()


def demo_integration_scenario():
    """統合シナリオ デモ"""
    print("=" * 60)
    print("5. 統合シナリオ デモ")
    print("=" * 60)
    
    print("バックテスト実行シミュレーション...")
    
    # 戦略実行シミュレーション
    strategies = ["momentum", "mean_reversion", "breakout"]
    
    for i, strategy_name in enumerate(strategies):
        print(f"\n戦略 '{strategy_name}' 実行中...")
        
        try:
            # 実行時間シミュレーション
            execution_time = random.uniform(0.5, 3.0)
            time.sleep(execution_time)
            
            # ランダムエラー発生（30%の確率）
            if random.random() < 0.3:
                if strategy_name == "momentum":
                    raise StrategyError(strategy_name, "移動平均計算エラー")
                elif strategy_name == "mean_reversion":
                    raise DataError("価格データ取得失敗")
                else:
                    raise SystemError("メモリ不足")
            
            # 成功ケース
            print(f"  ✓ 戦略 '{strategy_name}' 実行成功")
            
            # パフォーマンスログ
            log_strategy_performance(
                strategy_name,
                execution_time,
                random.uniform(50, 150),
                "正常実行完了"
            )
            
        except Exception as e:
            print(f"  ✗ 戦略 '{strategy_name}' エラー: {e}")
            
            # エラーハンドリング
            if isinstance(e, StrategyError):
                result = handle_strategy_error(strategy_name, e)
            elif isinstance(e, DataError):
                result = handle_data_error(e)
            else:
                result = handle_system_error(e)
            
            # 監視システムに報告
            report_error(e, {"strategy": strategy_name}, strategy_name)
            
            # 復旧試行
            def recovery_function():
                print(f"    復旧処理実行中...")
                time.sleep(0.5)
                return f"戦略 '{strategy_name}' 復旧完了"
            
            try:
                recovery_result = retry_with_strategy(recovery_function, "strategy_errors")
                print(f"    ✓ 復旧成功: {recovery_result}")
            except Exception as recovery_error:
                print(f"    ✗ 復旧失敗: {recovery_error}")
    
    print()


def show_system_status():
    """システム状態表示"""
    print("=" * 60)
    print("6. システム状態サマリー")
    print("=" * 60)
    
    # エラーハンドラー統計
    handler = UnifiedExceptionHandler()
    error_stats = handler.get_error_statistics()
    print("エラーハンドリング統計:")
    print(f"  総エラー数: {error_stats['total_errors']}")
    print(f"  戦略エラー: {error_stats['strategy_errors']}")
    print(f"  データエラー: {error_stats['data_errors']}")
    print(f"  システムエラー: {error_stats['system_errors']}")
    
    # 復旧管理統計
    recovery_manager = ErrorRecoveryManager()
    recovery_stats = recovery_manager.get_recovery_statistics()
    print(f"\n復旧システム統計:")
    print(f"  復旧試行回数: {recovery_stats['total_attempts']}")
    print(f"  成功回数: {recovery_stats['successful_recoveries']}")
    print(f"  失敗回数: {recovery_stats['failed_recoveries']}")
    print(f"  成功率: {recovery_stats.get('success_rate', 0):.1f}%")
    
    # ロガー統計
    logger_manager = get_logger_manager()
    log_stats = logger_manager.get_log_statistics()
    print(f"\nロギング統計:")
    print(f"  総ログ数: {log_stats['total_logs']}")
    print(f"  エラーログ数: {log_stats['error_count']}")
    print(f"  警告ログ数: {log_stats['warning_count']}")
    
    # 監視システム統計
    monitoring_agent = get_monitoring_agent()
    monitoring_stats = monitoring_agent.get_monitoring_statistics()
    print(f"\n監視システム統計:")
    print(f"  アラート発生回数: {monitoring_stats['alerts_triggered']}")
    print(f"  監視稼働時間: {monitoring_stats.get('uptime', 0):.1f}秒")
    
    print()


def main():
    """メイン実行"""
    print("🚀 エラーハンドリングシステム デモンストレーション")
    print("フェーズ3: 実践環境準備 - エラーハンドリング強化")
    print()
    
    try:
        # 各デモ実行
        demo_basic_error_handling()
        demo_error_recovery()
        demo_enhanced_logging()
        demo_monitoring_system()
        demo_integration_scenario()
        show_system_status()
        
        print("=" * 60)
        print("✅ デモンストレーション完了")
        print("=" * 60)
        print()
        print("📊 生成されたファイル:")
        
        # ログファイル確認
        logs_dir = project_root / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log*"))
            for log_file in log_files[:5]:  # 最初の5つを表示
                print(f"  - {log_file.name}")
            if len(log_files) > 5:
                print(f"  ... その他 {len(log_files) - 5} ファイル")
        
        print()
        print("🎯 エラーハンドリングシステムが正常に動作しています!")
        
    except Exception as e:
        print(f"❌ デモ実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

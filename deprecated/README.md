# Deprecated Files - 非推奨ファイル置き場

このフォルダには、メインシステム (main_new.py) で使用されなくなった旧バージョンのファイルが保管されています。

## 📋 移動済みファイル一覧

移動日: 2025-10-20

### 旧実行システム関連
- **paper_trade_runner.py** - 旧ペーパートレード実行スクリプト
  - 現在: `main_system/execution_control/` 配下のモジュールを使用
  - 旧インポート: `from src.execution.strategy_execution_manager import StrategyExecutionManager`

- **performance_monitor.py** - 旧パフォーマンス監視スクリプト
  - 現在: `main_system/performance/comprehensive_performance_analyzer.py` を使用
  - 旧インポート: `from src.execution.strategy_execution_manager import StrategyExecutionManager`

- **performance_monitor_old.py** - 旧パフォーマンス監視（古いバージョン）

- **performance_monitor_new.py** - 旧パフォーマンス監視（新しいバージョン）

- **demo_paper_trade_runner.py** - 旧デモスクリプト

## ⚠️ 重要な注意事項

1. **これらのファイルは使用しないでください**
   - バグ修正や新機能追加は行われません
   - メインシステムとの互換性は保証されません

2. **新規開発では以下を使用してください**
   ```python
   # 正しいインポート
   from main_system.execution_control.strategy_execution_manager import StrategyExecutionManager
   from main_system.performance.comprehensive_performance_analyzer import ComprehensivePerformanceAnalyzer
   from main_system.reporting.comprehensive_reporter import ComprehensiveReporter
   ```

3. **完全削除の予定**
   - これらのファイルは将来的に削除される可能性があります
   - 参照が必要な場合は、main_system版への移行を推奨します

## 🔄 移行ガイド

旧ファイルから新システムへの移行については、以下のドキュメントを参照してください:
- `diagnostics/results/main_py_integration_system_recovery_plan.md`
- `.github/copilot-instructions.md`

## 📝 履歴

- 2025-10-20: 初期作成、旧実行システム関連ファイルを移動

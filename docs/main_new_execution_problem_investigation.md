# main_new.py実行問題調査報告書

## 目的

main_new.pyをVSCodeの「専用ターミナルでPythonファイルを実行する」ボタンで実行できない問題を解決する。

## ゴール（成功条件）

- [ ] main_new.pyを「専用ターミナルでPythonファイルを実行する」ボタンで実行できる
- [ ] 実行できない原因を特定できる
- [ ] 原因を特定して修正することができる
- [ ] 対症療法にとどまらずに根本原因を修正する

## エラー状況

### 発生エラー
```
[2026-01-08 19:34:27,371] INFO - src.config.system_modes - SystemFallbackPolicy initialized with mode: development
Traceback (most recent call last):
  File "c:\Users\imega\Documents\my_backtest_project\main_new.py", line 47, in <module>
    from config.logger_config import setup_logger
  File "c:\Users\imega\Documents\my_backtest_project\config\__init__.py", line 107, in <module>
    from .portfolio_correlation_optimizer import *
  File "c:\Users\imega\Documents\my_backtest_project\config\portfolio_correlation_optimizer\__init__.py", line 20, in <module>
    from .correlation_based_allocator import (...)
  File "c:\Users\imega\Documents\my_backtest_project\config\portfolio_correlation_optimizer\correlation_based_allocator.py", line 21, in <module>
    from ..correlation.strategy_correlation_analyzer import (CorrelationConfig, CorrelationMatrix)
  File "c:\Users\imega\Documents\my_backtest_project\config\correlation\__init__.py", line 78, in <module>
    from .strategy_correlation_dashboard import (StrategyCorrelationDashboard)
  File "c:\Users\imega\Documents\my_backtest_project\config\correlation\strategy_correlation_dashboard.py", line 25, in <module>
    from config.correlation.strategy_correlation_analyzer import (...)
  File "c:\Users\imega\Documents\my_backtest_project\config\correlation\strategy_correlation_analyzer.py", line 91, in <module>
    from config.portfolio_weight_calculator import PortfolioWeightCalculator as ExistingPortfolioWeightCalculator  
  File "c:\Users\imega\Documents\my_backtest_project\config\portfolio_weight_calculator.py", line 111, in <module>
    from config.metric_weight_optimizer import MetricWeightOptimizer
  File "c:\Users\imega\Documents\my_backtest_project\config\metric_weight_optimizer.py", line 103, in <module>
    from scipy.optimize import minimize
  最終的にscipy処理でKeyboardInterrupt
```

### エラー分析
1. **トリガー**: main_new.py 47行目の`from config.logger_config import setup_logger`
2. **インポートチェーン**: config.__init__.py → portfolio_correlation_optimizer → correlation → strategy_correlation_analyzer → portfolio_weight_calculator → metric_weight_optimizer → scipy.optimize
3. **最終エラー**: scipyモジュール内のテキスト処理でKeyboardInterrupt

## 解決戦略

### Phase 1: 問題特定
1. scipy環境問題の確認
2. 循環インポートの調査
3. config.__init__.pyの複雑なインポート構造分析

### Phase 2: 根本修正
1. 不要なインポートの除去
2. 循環インポートの解消
3. 動的インポートの適用

### Phase 3: 検証
1. main_new.py実行成功確認
2. 既存機能の副作用チェック
3. テスト実行確認

## 履歴

作成日: 2026-01-08
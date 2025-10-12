# main.py未使用モジュール調査レポート（DSSMS除外版）

## 🎯 調査目的
comprehensive_module_test.py でテストすべき再利用可能なモジュールの選定
**DSSMS関連ファイルは完全除外済み**

## 📋 調査結果サマリー

- **総モジュール数**: 175（DSSMS除外後）
- **未使用モジュール数**: 175
- **高優先度再利用候補**: 18
- **中優先度再利用候補**: 34
- **total再利用候補**: 52

---

## データ取得・前処理系

**概要**: 3個（未使用: 3個、高優先度: 3個、中優先度: 0個）

### batch_processor
- **パス**: `config\performance_score_correction\batch_processor.py`
- **機能**: Module: Batch Processor
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### batch_processor
- **パス**: `config\trend_precision_adjustment\batch_processor.py`
- **機能**: Module: Batch Processing for Trend Precision Correction
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### strategy_characteristics_data_loader
- **パス**: `config\strategy_characteristics_data_loader.py`
- **機能**: Module: Strategy Characteristics Data Loader
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

---

## 設定・ログ系

**概要**: 15個（未使用: 15個、高優先度: 15個、中優先度: 0個）

### logger_config
- **パス**: `config\logger_config.py`
- **機能**: Module: Logger Configuration
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### logger_config
- **パス**: `src\config\logger_config.py`
- **機能**: Module: Logger Configuration
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### meta_parameter_controller
- **パス**: `config\weight_learning_optimizer\meta_parameter_controller.py`
- **機能**: メタパラメータコントローラー
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### metric_normalization_config
- **パス**: `config\metric_normalization_config.py`
- **機能**: Module: Metric Normalization Configuration
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### metric_selection_config
- **パス**: `config\metric_selection_config.py`
- **機能**: Module: Metric Selection Configuration
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### optimized_parameters
- **パス**: `config\optimized_parameters.py`
- **機能**: 最適化されたパラメータを管理するモジュール
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### optimized_parameters
- **パス**: `src\config\optimized_parameters.py`
- **機能**: 最適化されたパラメータを管理するモジュール
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### parameter_adjuster
- **パス**: `config\trend_precision_adjustment\parameter_adjuster.py`
- **機能**: Module: Parameter Adjuster
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### risk_management
- **パス**: `config\risk_management.py`
- **機能**: Module: Risk Management
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### risk_management
- **パス**: `src\config\risk_management.py`
- **機能**: Module: Risk Management
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### rule_configuration_manager
- **パス**: `config\rule_configuration_manager.py`
- **機能**: Module: Rule Configuration Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### strategy_parameter_standardizer
- **パス**: `config\strategy_parameter_standardizer.py`
- **機能**: TODO #13: 戦略パラメータ標準化システム
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### system_config
- **パス**: `config\portfolio_correlation_optimizer\configs\system_config.py`
- **機能**: 5-3-3 戦略間相関を考慮した配分最適化 - 設定ファイル
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### trend_params
- **パス**: `config\trend_params.py`
- **機能**: Module: Trend Parameters Configuration
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

### var_config
- **パス**: `config\portfolio_var_calculator\var_config.py`
- **機能**: 設定ファイル - VaR計算システム
- **使用状況**: 🔍 未使用
- **再利用可能性**: 🚀 高優先度

---

## 個別戦略クラス

**概要**: 14個（未使用: 14個、高優先度: 0個、中優先度: 14個）

### base_strategy
- **パス**: `strategies\base_strategy.py`
- **機能**: Module: base_strategy
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### base_strategy
- **パス**: `src\strategies\base_strategy.py`
- **機能**: Module: base_strategy
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### contrarian_strategy
- **パス**: `strategies\contrarian_strategy.py`
- **機能**: Module: contrarian_strategy
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### contrarian_strategy
- **パス**: `src\strategies\contrarian_strategy.py`
- **機能**: Module: contrarian_strategy
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### gc_strategy_signal
- **パス**: `strategies\gc_strategy_signal.py`
- **機能**: Module: gc_strategy_signal
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### gc_strategy_signal
- **パス**: `src\strategies\gc_strategy_signal.py`
- **機能**: Module: gc_strategy_signal
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### mean_reversion_strategy
- **パス**: `strategies\mean_reversion_strategy.py`
- **機能**: Module: Mean Reversion Strategy
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### mean_reversion_strategy
- **パス**: `src\strategies\mean_reversion_strategy.py`
- **機能**: Module: Mean Reversion Strategy
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### pairs_trading_strategy
- **パス**: `strategies\pairs_trading_strategy.py`
- **機能**: Module: Pairs Trading Strategy (Simplified Single-Asset Version)
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### pairs_trading_strategy
- **パス**: `src\strategies\pairs_trading_strategy.py`
- **機能**: Module: Pairs Trading Strategy (Simplified Single-Asset Version)
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### strategy_manager
- **パス**: `strategies\strategy_manager.py`
- **機能**: Module: Strategy Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### strategy_manager
- **パス**: `src\strategies\strategy_manager.py`
- **機能**: Module: Strategy Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### support_resistance_contrarian_strategy
- **パス**: `strategies\support_resistance_contrarian_strategy.py`
- **機能**: Module: Support/Resistance Contrarian Strategy
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### support_resistance_contrarian_strategy
- **パス**: `src\strategies\support_resistance_contrarian_strategy.py`
- **機能**: Module: Support/Resistance Contrarian Strategy
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

---

## 出力系

**概要**: 2個（未使用: 2個、高優先度: 0個、中優先度: 0個）

### main_text_reporter
- **パス**: `output\main_text_reporter.py`
- **機能**: Module: Main Text Reporter
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### main_text_reporter
- **パス**: `archive\engines\historical\main_text_reporter.py`
- **機能**: Module: Main Text Reporter
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

---

## 指標計算系

**概要**: 20個（未使用: 20個、高優先度: 0個、中優先度: 20個）

### basic_indicators
- **パス**: `indicators\basic_indicators.py`
- **機能**: Module: Basic Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### basic_indicators
- **パス**: `src\indicators\basic_indicators.py`
- **機能**: Module: Basic Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### enhanced_performance_calculator
- **パス**: `config\enhanced_performance_calculator.py`
- **機能**: Module: Enhanced Performance Calculator
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### enhanced_score_calculator
- **パス**: `config\performance_score_correction\enhanced_score_calculator.py`
- **機能**: Module: Enhanced Score Calculator
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### gap_indicators
- **パス**: `indicators\gap_indicators.py`
- **機能**: Module: Gap Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### gap_indicators
- **パス**: `src\indicators\gap_indicators.py`
- **機能**: Module: Gap Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### hybrid_var_calculator
- **パス**: `config\portfolio_var_calculator\hybrid_var_calculator.py`
- **機能**: ハイブリッドVaR計算システム
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### indicator_calculator
- **パス**: `indicators\indicator_calculator.py`
- **機能**: 関数: compute_indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### indicator_calculator
- **パス**: `src\indicators\indicator_calculator.py`
- **機能**: 関数: compute_indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### momentum_indicators
- **パス**: `indicators\momentum_indicators.py`
- **機能**: Module: Momentum Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### momentum_indicators
- **パス**: `src\indicators\momentum_indicators.py`
- **機能**: Module: Momentum Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### pivot_indicators
- **パス**: `indicators\pivot_indicators.py`
- **機能**: Module: Pivot Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### pivot_indicators
- **パス**: `src\indicators\pivot_indicators.py`
- **機能**: Module: Pivot Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### portfolio_weight_calculator
- **パス**: `config\portfolio_weight_calculator.py`
- **機能**: クラス: AllocationMethod, ConstraintType
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### portfolio_weight_calculator_integration
- **パス**: `config\portfolio_weight_calculator_integration.py`
- **機能**: Module: Portfolio Weight Calculator Integration
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### signal_integrator
- **パス**: `config\signal_integrator.py`
- **機能**: Module: Signal Integrator
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### volatility_indicators
- **パス**: `indicators\volatility_indicators.py`
- **機能**: Module: Volatility Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### volatility_indicators
- **パス**: `src\indicators\volatility_indicators.py`
- **機能**: Module: Volatility Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### volume_indicators
- **パス**: `indicators\volume_indicators.py`
- **機能**: Module: Volume Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

### volume_indicators
- **パス**: `src\indicators\volume_indicators.py`
- **機能**: Module: Volume Indicators
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚡ 中優先度

---

## その他

**概要**: 114個（未使用: 114個、高優先度: 0個、中優先度: 0個）

### Breakout
- **パス**: `strategies\Breakout.py`
- **機能**: Module: Breakout
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### Breakout
- **パス**: `src\strategies\Breakout.py`
- **機能**: Module: Breakout
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### Momentum_Investing
- **パス**: `strategies\Momentum_Investing.py`
- **機能**: Module: Momentum_Investing
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### Momentum_Investing
- **パス**: `src\strategies\Momentum_Investing.py`
- **機能**: Module: Momentum_Investing
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### Opening_Gap
- **パス**: `strategies\Opening_Gap.py`
- **機能**: Module: Opening_Gap
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### Opening_Gap
- **パス**: `src\strategies\Opening_Gap.py`
- **機能**: Module: Opening_Gap
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### VWAP_Bounce
- **パス**: `strategies\VWAP_Bounce.py`
- **機能**: Module: VWAP_Bounce
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### VWAP_Bounce
- **パス**: `src\strategies\VWAP_Bounce.py`
- **機能**: Module: VWAP_Bounce
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### VWAP_Breakout
- **パス**: `strategies\VWAP_Breakout.py`
- **機能**: Module: VWAP_Breakout
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### VWAP_Breakout
- **パス**: `src\strategies\VWAP_Breakout.py`
- **機能**: Module: VWAP_Breakout
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `strategies\__init__.py`
- **機能**: 機能不明
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `indicators\__init__.py`
- **機能**: 機能不明
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `output\__init__.py`
- **機能**: 機能不明
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `config\__init__.py`
- **機能**: クラス: LazyConfigImporter, _DummyFallbackPolicy
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `config\correlation\__init__.py`
- **機能**: クラス: LazyCorrelationImporter, _DummyFallbackPolicy
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `config\performance_score_correction\__init__.py`
- **機能**: 5-2-1「戦略実績に基づくスコア補正機能」パッケージ
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `config\portfolio_correlation_optimizer\__init__.py`
- **機能**: 5-3-3「戦略間相関を考慮した配分最適化」パッケージ
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `config\portfolio_var_calculator\__init__.py`
- **機能**: 5-3-2「ポートフォリオVaR（バリューアットリスク）計算」
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `config\trend_precision_adjustment\__init__.py`
- **機能**: 5-2-2「トレンド判定精度の自動補正」パッケージ
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `config\weight_learning_optimizer\__init__.py`
- **機能**: 5-2-3 最適な重み付け比率の学習アルゴリズム (Optimal Weight Ratio Learning Algorithm)
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `src\strategies\__init__.py`
- **機能**: 機能不明
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### __init__
- **パス**: `src\indicators\__init__.py`
- **機能**: 機能不明
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### adaptive_learning
- **パス**: `config\trend_precision_adjustment\adaptive_learning.py`
- **機能**: Module: Adaptive Learning Engine
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### adaptive_learning_scheduler
- **パス**: `config\weight_learning_optimizer\adaptive_learning_scheduler.py`
- **機能**: 適応的学習スケジューラー
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### advanced_var_engine
- **パス**: `config\portfolio_var_calculator\advanced_var_engine.py`
- **機能**: 高度VaRエンジン - 5-3-2「ポートフォリオVaR計算」のメインエンジン
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### bayesian_weight_optimizer
- **パス**: `config\weight_learning_optimizer\bayesian_weight_optimizer.py`
- **機能**: ベイジアン最適化による重み学習システム
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### bollinger_atr
- **パス**: `indicators\bollinger_atr.py`
- **機能**: Module: Bollinger Bands and ATR
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### bollinger_atr
- **パス**: `src\indicators\bollinger_atr.py`
- **機能**: Module: Bollinger Bands and ATR
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### cache_manager
- **パス**: `config\cache_manager.py`
- **機能**: Module: Cache Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### cache_manager
- **パス**: `src\config\cache_manager.py`
- **機能**: Module: Cache Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### composite_strategy_execution_engine
- **パス**: `config\composite_strategy_execution_engine.py`
- **機能**: Module: Composite Strategy Execution Engine
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### concurrent_execution_scheduler
- **パス**: `config\concurrent_execution_scheduler.py`
- **機能**: Module: Concurrent Execution Scheduler
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### confidence_calibrator
- **パス**: `config\trend_precision_adjustment\confidence_calibrator.py`
- **機能**: Module: Confidence Calibrator
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### constraint_manager
- **パス**: `config\portfolio_correlation_optimizer\constraint_manager.py`
- **機能**: 5-3-3 戦略間相関を考慮した配分最適化 - 制約管理システム
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### correction_engine
- **パス**: `config\trend_precision_adjustment\correction_engine.py`
- **機能**: Module: Correction Engine
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### correlation_based_allocator
- **パス**: `config\portfolio_correlation_optimizer\correlation_based_allocator.py`
- **機能**: 5-3-3 戦略間相関を考慮した配分最適化 - メイン配分エンジン
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### correlation_matrix_visualizer
- **パス**: `config\correlation\correlation_matrix_visualizer.py`
- **機能**: 相関行列視覚化システム - 戦略間の相関と共分散を視覚化
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### data_extraction_enhancer
- **パス**: `output\data_extraction_enhancer.py`
- **機能**: main.py結果データの精密抽出・解析エンジン
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### data_validator
- **パス**: `output\quality_assurance\data_validator.py`
- **機能**: 解析エラー: unexpected indent (<unknown>, 
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### demo_multi_strategy_coordination
- **パス**: `config\demo_multi_strategy_coordination.py`
- **機能**: Module: Multi-Strategy Coordination Demo
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### drawdown_action_executor
- **パス**: `config\drawdown_action_executor.py`
- **機能**: Module: Drawdown Action Executor
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### drawdown_controller
- **パス**: `config\drawdown_controller.py`
- **機能**: Module: Drawdown Controller
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### enhanced_score_history_manager
- **パス**: `config\enhanced_score_history_manager.py`
- **機能**: Module: Enhanced Score History Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### enhanced_strategy_scoring_model
- **パス**: `config\enhanced_strategy_scoring_model.py`
- **機能**: Enhanced Strategy Scoring Model
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### enhanced_strategy_selector
- **パス**: `config\enhanced_strategy_selector.py`
- **機能**: Module: Enhanced Strategy Selector
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### enhanced_trend_detector
- **パス**: `config\trend_precision_adjustment\enhanced_trend_detector.py`
- **機能**: Module: Enhanced Trend Detector
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### error_handling
- **パス**: `config\error_handling.py`
- **機能**: Module: Error Handling
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### execution_monitoring_system
- **パス**: `config\execution_monitoring_system.py`
- **機能**: Module: Execution Monitoring System
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### execution_result_aggregator
- **パス**: `config\execution_result_aggregator.py`
- **機能**: Module: Execution Result Aggregator
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### file_utils
- **パス**: `config\file_utils.py`
- **機能**: Module: File Utilities
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### integration_bridge
- **パス**: `config\portfolio_correlation_optimizer\integration_bridge.py`
- **機能**: 5-3-3 戦略間相関を考慮した配分最適化 - システム統合ブリッジ
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### integration_bridge
- **パス**: `config\weight_learning_optimizer\integration_bridge.py`
- **機能**: 統合ブリッジ
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### metric_importance_analyzer
- **パス**: `config\metric_importance_analyzer.py`
- **機能**: Module: Metric Importance Analyzer
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### metric_normalization_engine
- **パス**: `config\metric_normalization_engine.py`
- **機能**: Module: Metric Normalization Engine
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### metric_normalization_manager
- **パス**: `config\metric_normalization_manager.py`
- **機能**: Module: Metric Normalization Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### metric_selection_manager
- **パス**: `config\metric_selection_manager.py`
- **機能**: Module: Metric Selection Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### metric_weight_optimizer
- **パス**: `config\metric_weight_optimizer.py`
- **機能**: クラス: WeightOptimizationResult, MetricWeightOptimizer
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### minimum_weight_rule_manager
- **パス**: `config\minimum_weight_rule_manager.py`
- **機能**: Module: Minimum Weight Rule Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### momentum
- **パス**: `indicators\momentum.py`
- **機能**: Module: Momentum Calculations
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### momentum
- **パス**: `src\indicators\momentum.py`
- **機能**: Module: Momentum Calculations
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### multi_strategy_coordination_interface
- **パス**: `config\multi_strategy_coordination_interface.py`
- **機能**: Module: Multi-Strategy Coordination Interface
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### multi_strategy_coordination_manager
- **パス**: `config\multi_strategy_coordination_manager.py`
- **機能**: Module: Multi-Strategy Coordination Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### optimal_weight_learning_system
- **パス**: `config\weight_learning_optimizer\optimal_weight_learning_system.py`
- **機能**: 5-2-3 最適な重み付け比率の学習アルゴリズム メインシステム
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### optimization_engine
- **パス**: `config\portfolio_correlation_optimizer\optimization_engine.py`
- **機能**: 5-3-3 戦略間相関を考慮した配分最適化 - 最適化エンジン
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### optimization_history_manager
- **パス**: `config\weight_learning_optimizer\optimization_history_manager.py`
- **機能**: 最適化履歴管理システム
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### performance_evaluator
- **パス**: `config\weight_learning_optimizer\performance_evaluator.py`
- **機能**: パフォーマンス評価システム
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### performance_tracker
- **パス**: `config\performance_score_correction\performance_tracker.py`
- **機能**: Module: Performance Tracker
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### portfolio_risk_manager
- **パス**: `config\portfolio_risk_manager.py`
- **機能**: Module: Portfolio Risk Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### portfolio_weight_pattern_engine
- **パス**: `config\portfolio_weight_pattern_engine.py`
- **機能**: Module: Portfolio Weight Pattern Engine
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### portfolio_weight_pattern_engine_v2
- **パス**: `config\portfolio_weight_pattern_engine_v2.py`
- **機能**: Module: Portfolio Weight Pattern Engine V2
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### portfolio_weighting_agent
- **パス**: `config\portfolio_weighting_agent.py`
- **機能**: Module: Portfolio Weighting Agent
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### position_size_adjuster
- **パス**: `config\position_size_adjuster.py`
- **機能**: Module: Position Size Adjuster
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### precision_tracker
- **パス**: `config\trend_precision_adjustment\precision_tracker.py`
- **機能**: Module: Trend Precision Tracker
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### real_time_var_monitor
- **パス**: `config\portfolio_var_calculator\real_time_var_monitor.py`
- **機能**: リアルタイムVaR監視システム
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### realtime_update_engine
- **パス**: `config\realtime_update_engine.py`
- **機能**: Module: Realtime Update Engine
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### resource_allocation_engine
- **パス**: `config\resource_allocation_engine.py`
- **機能**: Module: Resource Allocation Engine
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### rule_engine_integrated_interface
- **パス**: `config\rule_engine_integrated_interface.py`
- **機能**: Module: Rule Engine Integrated Interface
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### score_corrector
- **パス**: `config\performance_score_correction\score_corrector.py`
- **機能**: Module: Score Corrector
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### score_history_manager
- **パス**: `config\score_history_manager.py`
- **機能**: スコア履歴保存システム (2-3-1)
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### score_update_trigger_system
- **パス**: `config\score_update_trigger_system.py`
- **機能**: Module: Score Update Trigger System
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### simple_simulation_handler
- **パス**: `output\simple_simulation_handler.py`
- **機能**: Module: Simple Simulation Handler
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_characteristics_manager
- **パス**: `config\strategy_characteristics_manager.py`
- **機能**: Module: Strategy Characteristics Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_combination_manager
- **パス**: `config\strategy_combination_manager.py`
- **機能**: Module: Strategy Combination Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_correlation_analyzer
- **パス**: `config\correlation\strategy_correlation_analyzer.py`
- **機能**: クラス: CorrelationConfig, CorrelationMatrix
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_correlation_dashboard
- **パス**: `config\correlation\strategy_correlation_dashboard.py`
- **機能**: 戦略相関ダッシュボード - 4-3-3システム統合ダッシュボード
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_data_persistence
- **パス**: `config\strategy_data_persistence.py`
- **機能**: Module: Strategy Data Persistence
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_dependency_resolver
- **パス**: `config\strategy_dependency_resolver.py`
- **機能**: Module: Strategy Dependency Resolver
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_execution_adapter
- **パス**: `config\strategy_execution_adapter.py`
- **機能**: Module: Strategy Execution Adapter
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_execution_coordinator
- **パス**: `config\strategy_execution_coordinator.py`
- **機能**: Module: Strategy Execution Coordinator
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_execution_pipeline
- **パス**: `config\strategy_execution_pipeline.py`
- **機能**: Module: Strategy Execution Pipeline
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_performance_comparator
- **パス**: `config\strategy_performance_comparator.py`
- **機能**: 機能不明
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_scoring_model
- **パス**: `config\strategy_scoring_model.py`
- **機能**: Module: Strategy Scoring Model
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_selection_rule_engine
- **パス**: `config\strategy_selection_rule_engine.py`
- **機能**: Module: Strategy Selection Rule Engine
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### strategy_selector
- **パス**: `config\strategy_selector.py`
- **機能**: Module: Strategy Selector
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### time_decay_factor
- **パス**: `config\time_decay_factor.py`
- **機能**: Module: Time Decay Factor
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### time_decay_utilities
- **パス**: `config\time_decay_utilities.py`
- **機能**: Module: Time Decay Factor Utilities
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_accuracy_validator
- **パス**: `indicators\trend_accuracy_validator.py`
- **機能**: Module: Trend Accuracy Validator
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_accuracy_validator
- **パス**: `src\indicators\trend_accuracy_validator.py`
- **機能**: Module: Trend Accuracy Validator
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_analysis
- **パス**: `indicators\trend_analysis.py`
- **機能**: Module: Trend Analysis (Enhanced)
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_analysis
- **パス**: `src\indicators\trend_analysis.py`
- **機能**: Module: Trend Analysis (Enhanced)
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_labeling
- **パス**: `indicators\trend_labeling.py`
- **機能**: Module: Trend Labeling
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_labeling
- **パス**: `src\indicators\trend_labeling.py`
- **機能**: Module: Trend Labeling
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_reliability_utils
- **パス**: `indicators\trend_reliability_utils.py`
- **機能**: トレンド信頼度比較ユーティリティ
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_reliability_utils
- **パス**: `src\indicators\trend_reliability_utils.py`
- **機能**: トレンド信頼度比較ユーティリティ
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_strategy_integration_interface
- **パス**: `config\trend_strategy_integration_interface.py`
- **機能**: Module: Trend Strategy Integration Interface
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_transition_detector
- **パス**: `indicators\trend_transition_detector.py`
- **機能**: Module: Trend Transition Detector
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_transition_detector
- **パス**: `src\indicators\trend_transition_detector.py`
- **機能**: Module: Trend Transition Detector
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_transition_manager
- **パス**: `config\trend_transition_manager.py`
- **機能**: Module: Trend Transition Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### unified_trend_detector
- **パス**: `indicators\unified_trend_detector.py`
- **機能**: Module: Unified Trend Detector
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### unified_trend_detector
- **パス**: `src\indicators\unified_trend_detector.py`
- **機能**: Module: Unified Trend Detector
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### var_integration_bridge
- **パス**: `config\portfolio_var_calculator\var_integration_bridge.py`
- **機能**: VaR統合ブリッジ
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### volume_analysis
- **パス**: `indicators\volume_analysis.py`
- **機能**: Module: Volume Analysis
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### volume_analysis
- **パス**: `src\indicators\volume_analysis.py`
- **機能**: Module: Volume Analysis
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### weight_constraint_manager
- **パス**: `config\weight_learning_optimizer\weight_constraint_manager.py`
- **機能**: 重み制約管理システム
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

---

## 破棄予定ファイル

**概要**: 7個（未使用: 7個、高優先度: 0個、中優先度: 0個）

### basic_system_test
- **パス**: `config\basic_system_test.py`
- **機能**: Basic system test for 4-1-3 coordination system
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### data_extraction_enhancer
- **パス**: `archive\engines\historical\data_extraction_enhancer.py`
- **機能**: main.py結果データの精密抽出・解析エンジン
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### hierarchical_switch_decision_engine
- **パス**: `archive\engines\historical\hierarchical_switch_decision_engine.py`
- **機能**: Hierarchical Switch Decision Engine
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### portfolio_weight_templates
- **パス**: `config\portfolio_weight_templates.py`
- **機能**: Module: Portfolio Weight Template Manager
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### simple_simulation_handler
- **パス**: `archive\engines\historical\simple_simulation_handler.py`
- **機能**: Module: Simple Simulation Handler
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### trend_strategy_integration_interface_backup
- **パス**: `config\trend_strategy_integration_interface_backup.py`
- **機能**: Module: Trend Strategy Integration Interface
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

### var_backtesting_engine
- **パス**: `config\portfolio_var_calculator\var_backtesting_engine.py`
- **機能**: VaRバックテスティングエンジン
- **使用状況**: 🔍 未使用
- **再利用可能性**: ⚠️ 低優先度

---

## 🎯 comprehensive_module_test.py 推奨テスト対象

### 🔥 Phase 0: 高優先度テスト（安全性高・効果大）

- **strategy_characteristics_data_loader** (データ取得・前処理系): Module: Strategy Characteristics Data Loader
- **batch_processor** (データ取得・前処理系): Module: Batch Processor
- **batch_processor** (データ取得・前処理系): Module: Batch Processing for Trend Precision Correction
- **logger_config** (設定・ログ系): Module: Logger Configuration
- **metric_normalization_config** (設定・ログ系): Module: Metric Normalization Configuration
- **metric_selection_config** (設定・ログ系): Module: Metric Selection Configuration
- **optimized_parameters** (設定・ログ系): 最適化されたパラメータを管理するモジュール
- **risk_management** (設定・ログ系): Module: Risk Management
- **rule_configuration_manager** (設定・ログ系): Module: Rule Configuration Manager
- **strategy_parameter_standardizer** (設定・ログ系): TODO #13: 戦略パラメータ標準化システム
- **trend_params** (設定・ログ系): Module: Trend Parameters Configuration
- **var_config** (設定・ログ系): 設定ファイル - VaR計算システム
- **parameter_adjuster** (設定・ログ系): Module: Parameter Adjuster
- **meta_parameter_controller** (設定・ログ系): メタパラメータコントローラー
- **system_config** (設定・ログ系): 5-3-3 戦略間相関を考慮した配分最適化 - 設定ファイル
- **logger_config** (設定・ログ系): Module: Logger Configuration
- **optimized_parameters** (設定・ログ系): 最適化されたパラメータを管理するモジュール
- **risk_management** (設定・ログ系): Module: Risk Management

### ⚡ Phase 1: 中優先度テスト（要注意テスト）

- **base_strategy** (個別戦略クラス): Module: base_strategy
- **contrarian_strategy** (個別戦略クラス): Module: contrarian_strategy
- **gc_strategy_signal** (個別戦略クラス): Module: gc_strategy_signal
- **mean_reversion_strategy** (個別戦略クラス): Module: Mean Reversion Strategy
- **pairs_trading_strategy** (個別戦略クラス): Module: Pairs Trading Strategy (Simplified Single-Asset Version)
- **strategy_manager** (個別戦略クラス): Module: Strategy Manager
- **support_resistance_contrarian_strategy** (個別戦略クラス): Module: Support/Resistance Contrarian Strategy
- **base_strategy** (個別戦略クラス): Module: base_strategy
- **contrarian_strategy** (個別戦略クラス): Module: contrarian_strategy
- **gc_strategy_signal** (個別戦略クラス): Module: gc_strategy_signal
- **mean_reversion_strategy** (個別戦略クラス): Module: Mean Reversion Strategy
- **pairs_trading_strategy** (個別戦略クラス): Module: Pairs Trading Strategy (Simplified Single-Asset Version)
- **strategy_manager** (個別戦略クラス): Module: Strategy Manager
- **support_resistance_contrarian_strategy** (個別戦略クラス): Module: Support/Resistance Contrarian Strategy
- **basic_indicators** (指標計算系): Module: Basic Indicators
- **gap_indicators** (指標計算系): Module: Gap Indicators
- **indicator_calculator** (指標計算系): 関数: compute_indicators
- **momentum_indicators** (指標計算系): Module: Momentum Indicators
- **pivot_indicators** (指標計算系): Module: Pivot Indicators
- **volatility_indicators** (指標計算系): Module: Volatility Indicators
- **volume_indicators** (指標計算系): Module: Volume Indicators
- **enhanced_performance_calculator** (指標計算系): Module: Enhanced Performance Calculator
- **portfolio_weight_calculator** (指標計算系): クラス: AllocationMethod, ConstraintType
- **portfolio_weight_calculator_integration** (指標計算系): Module: Portfolio Weight Calculator Integration
- **signal_integrator** (指標計算系): Module: Signal Integrator
- **enhanced_score_calculator** (指標計算系): Module: Enhanced Score Calculator
- **hybrid_var_calculator** (指標計算系): ハイブリッドVaR計算システム
- **basic_indicators** (指標計算系): Module: Basic Indicators
- **gap_indicators** (指標計算系): Module: Gap Indicators
- **indicator_calculator** (指標計算系): 関数: compute_indicators
- **momentum_indicators** (指標計算系): Module: Momentum Indicators
- **pivot_indicators** (指標計算系): Module: Pivot Indicators
- **volatility_indicators** (指標計算系): Module: Volatility Indicators
- **volume_indicators** (指標計算系): Module: Volume Indicators

## 📝 推奨テスト戦略

### Phase 0テスト順序
1. **データ取得・前処理系**（高優先度）
2. **設定・ログ系**（高優先度）
3. **個別戦略クラス**（中優先度から1つ選択）

### 成功基準
- ✅ インポートエラーなし
- ✅ 基本メソッド動作確認
- ✅ Entry_Signal/Exit_Signal生成確認（戦略のみ）
- ✅ 同一日Entry/Exit問題なし

### 失敗時の対応
- 3回修正試行後も失敗 → 破棄
- 複雑すぎて理解困難 → 破棄
- DSSMS関連発見 → 即座に破棄

---
**レポート生成日時**: 2025-10-12 10:03:31
**調査対象**: main.py未使用だが再利用可能なモジュール（DSSMS完全除外）
**除外対象**: DSSMS関連、テスト用、アーカイブ、廃止予定ファイル
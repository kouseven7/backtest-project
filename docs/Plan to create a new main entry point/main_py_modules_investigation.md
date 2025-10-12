# main.py使用中モジュール調査レポート

## 🎯 調査目的
main.pyで実際に使用されているモジュールの再利用可能性評価
comprehensive_module_test.py でテストすべき既存使用モジュールの選定

## 📋 調査結果サマリー

- **総使用モジュール数**: 36
- **高優先度再利用候補**: 20
- **中優先度再利用候補**: 11
- **再利用推奨total**: 31

---

## データ取得・前処理系

**概要**: 2個のモジュール使用中
- 高優先度再利用候補: 2個
- 中優先度再利用候補: 0個
- 低優先度: 0個
- 再利用禁止: 0個

### get_parameters_and_data
- **モジュールパス**: `data_fetcher`
- **インポート方法**: from data_fetcher import get_parameters_and_data
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 77

### preprocess_data
- **モジュールパス**: `data_processor`
- **インポート方法**: from data_processor import preprocess_data
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 75

---

## 設定・ログ系

**概要**: 8個のモジュール使用中
- 高優先度再利用候補: 8個
- 中優先度再利用候補: 0個
- 低優先度: 0個
- 再利用禁止: 0個

### ComponentType
- **モジュールパス**: `src.config.system_modes`
- **インポート方法**: from src.config.system_modes import ComponentType
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 42

### ExecutionMode
- **モジュールパス**: `config.multi_strategy_manager_fixed`
- **インポート方法**: from config.multi_strategy_manager_fixed import ExecutionMode
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 53

### MultiStrategyManager
- **モジュールパス**: `config.multi_strategy_manager_fixed`
- **インポート方法**: from config.multi_strategy_manager_fixed import MultiStrategyManager
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 53

### OptimizedParameterManager
- **モジュールパス**: `config.optimized_parameters`
- **インポート方法**: from config.optimized_parameters import OptimizedParameterManager
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 41

### RiskManagement
- **モジュールパス**: `config.risk_management`
- **インポート方法**: from config.risk_management import RiskManagement
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 40

### StrategyExecutionAdapter
- **モジュールパス**: `config.strategy_execution_adapter`
- **インポート方法**: from config.strategy_execution_adapter import StrategyExecutionAdapter
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 54

### SystemFallbackPolicy
- **モジュールパス**: `src.config.system_modes`
- **インポート方法**: from src.config.system_modes import SystemFallbackPolicy
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 42

### setup_logger
- **モジュールパス**: `config.logger_config`
- **インポート方法**: from config.logger_config import setup_logger
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 39

---

## 個別戦略クラス

**概要**: 7個のモジュール使用中
- 高優先度再利用候補: 7個
- 中優先度再利用候補: 0個
- 低優先度: 0個
- 再利用禁止: 0個

### BreakoutStrategy
- **モジュールパス**: `strategies.Breakout`
- **インポート方法**: from strategies.Breakout import BreakoutStrategy
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 70

### ContrarianStrategy
- **モジュールパス**: `strategies.contrarian_strategy`
- **インポート方法**: from strategies.contrarian_strategy import ContrarianStrategy
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 73

### GCStrategy
- **モジュールパス**: `strategies.gc_strategy_signal`
- **インポート方法**: from strategies.gc_strategy_signal import GCStrategy
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 74

### MomentumInvestingStrategy
- **モジュールパス**: `strategies.Momentum_Investing`
- **インポート方法**: from strategies.Momentum_Investing import MomentumInvestingStrategy
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 69

### OpeningGapStrategy
- **モジュールパス**: `strategies.Opening_Gap`
- **インポート方法**: from strategies.Opening_Gap import OpeningGapStrategy
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 72

### VWAPBounceStrategy
- **モジュールパス**: `strategies.VWAP_Bounce`
- **インポート方法**: from strategies.VWAP_Bounce import VWAPBounceStrategy
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 71

### VWAPBreakoutStrategy
- **モジュールパス**: `strategies.VWAP_Breakout`
- **インポート方法**: from strategies.VWAP_Breakout import VWAPBreakoutStrategy
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 68

---

## 出力系

**概要**: 4個のモジュール使用中
- 高優先度再利用候補: 0個
- 中優先度再利用候補: 4個
- 低優先度: 0個
- 再利用禁止: 0個

### UnifiedExporter
- **モジュールパス**: `output.unified_exporter`
- **インポート方法**: from output.unified_exporter import UnifiedExporter
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: ⚡ 中優先度
- **main.py行番号**: 1075

### UnifiedExporter
- **モジュールパス**: `output.unified_exporter`
- **インポート方法**: from output.unified_exporter import UnifiedExporter
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: ⚡ 中優先度
- **main.py行番号**: 961

### UnifiedExporter
- **モジュールパス**: `output.unified_exporter`
- **インポート方法**: from output.unified_exporter import UnifiedExporter
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: ⚡ 中優先度
- **main.py行番号**: 1003

### generate_main_text_report
- **モジュールパス**: `output.main_text_reporter`
- **インポート方法**: from output.main_text_reporter import generate_main_text_report
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: ⚡ 中優先度
- **main.py行番号**: 1056

---

## 指標計算系

**概要**: 3個のモジュール使用中
- 高優先度再利用候補: 3個
- 中優先度再利用候補: 0個
- 低優先度: 0個
- 再利用禁止: 0個

### compute_indicators
- **モジュールパス**: `indicators.indicator_calculator`
- **インポート方法**: from indicators.indicator_calculator import compute_indicators
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 76

### detect_unified_trend
- **モジュールパス**: `indicators.unified_trend_detector`
- **インポート方法**: from indicators.unified_trend_detector import detect_unified_trend
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 67

### detect_unified_trend_with_confidence
- **モジュールパス**: `indicators.unified_trend_detector`
- **インポート方法**: from indicators.unified_trend_detector import detect_unified_trend_with_confidence
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: 🚀 高優先度
- **main.py行番号**: 67

---

## その他

**概要**: 12個のモジュール使用中
- 高優先度再利用候補: 0個
- 中優先度再利用候補: 7個
- 低優先度: 5個
- 再利用禁止: 0個

### Any
- **モジュールパス**: `typing`
- **インポート方法**: from typing import Any
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: ⚡ 中優先度
- **main.py行番号**: 34

### Any
- **モジュールパス**: `typing`
- **インポート方法**: from typing import Any
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: ⚡ 中優先度
- **main.py行番号**: 1076

### Dict
- **モジュールパス**: `typing`
- **インポート方法**: from typing import Dict
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: ⚡ 中優先度
- **main.py行番号**: 34

### Dict
- **モジュールパス**: `typing`
- **インポート方法**: from typing import Dict
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: ⚡ 中優先度
- **main.py行番号**: 1076

### List
- **モジュールパス**: `typing`
- **インポート方法**: from typing import List
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: ⚡ 中優先度
- **main.py行番号**: 34

### List
- **モジュールパス**: `typing`
- **インポート方法**: from typing import List
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: ⚡ 中優先度
- **main.py行番号**: 1076

### datetime
- **モジュールパス**: `datetime`
- **インポート方法**: from datetime import datetime
- **機能概要**: 解析エラー: unsupported operand type(s) for +: 'WindowsPath' a
- **再利用可能性**: ⚡ 中優先度
- **main.py行番号**: 33

### logging
- **モジュールパス**: `logging`
- **インポート方法**: 標準ライブラリ: logging
- **機能概要**: 標準ライブラリ: logging
- **再利用可能性**: ⚠️ 低優先度
- **main.py行番号**: 30

### numpy
- **モジュールパス**: `numpy`
- **インポート方法**: データ処理ライブラリ: numpy
- **機能概要**: データ処理ライブラリ: numpy
- **再利用可能性**: ⚠️ 低優先度
- **main.py行番号**: 32

### os
- **モジュールパス**: `os`
- **インポート方法**: 標準ライブラリ: os
- **機能概要**: 標準ライブラリ: os
- **再利用可能性**: ⚠️ 低優先度
- **main.py行番号**: 29

### pandas
- **モジュールパス**: `pandas`
- **インポート方法**: データ処理ライブラリ: pandas
- **機能概要**: データ処理ライブラリ: pandas
- **再利用可能性**: ⚠️ 低優先度
- **main.py行番号**: 31

### sys
- **モジュールパス**: `sys`
- **インポート方法**: 標準ライブラリ: sys
- **機能概要**: 標準ライブラリ: sys
- **再利用可能性**: ⚠️ 低優先度
- **main.py行番号**: 28

---

## 🎯 comprehensive_module_test.py 推奨テスト対象

### 🔥 Phase 0: 高優先度テスト（main.py実証済み）

- **preprocess_data** (データ取得・前処理系): main.pyで実証済み
- **get_parameters_and_data** (データ取得・前処理系): main.pyで実証済み
- **setup_logger** (設定・ログ系): main.pyで実証済み
- **RiskManagement** (設定・ログ系): main.pyで実証済み
- **OptimizedParameterManager** (設定・ログ系): main.pyで実証済み
- **SystemFallbackPolicy** (設定・ログ系): main.pyで実証済み
- **ComponentType** (設定・ログ系): main.pyで実証済み
- **MultiStrategyManager** (設定・ログ系): main.pyで実証済み
- **ExecutionMode** (設定・ログ系): main.pyで実証済み
- **StrategyExecutionAdapter** (設定・ログ系): main.pyで実証済み
- **VWAPBreakoutStrategy** (個別戦略クラス): main.pyで実証済み
- **MomentumInvestingStrategy** (個別戦略クラス): main.pyで実証済み
- **BreakoutStrategy** (個別戦略クラス): main.pyで実証済み
- **VWAPBounceStrategy** (個別戦略クラス): main.pyで実証済み
- **OpeningGapStrategy** (個別戦略クラス): main.pyで実証済み
- **ContrarianStrategy** (個別戦略クラス): main.pyで実証済み
- **GCStrategy** (個別戦略クラス): main.pyで実証済み
- **detect_unified_trend** (指標計算系): main.pyで実証済み
- **detect_unified_trend_with_confidence** (指標計算系): main.pyで実証済み
- **compute_indicators** (指標計算系): main.pyで実証済み

### ⚡ Phase 1: 中優先度テスト（要注意）

- **generate_main_text_report** (出力系): 限定テスト推奨
- **UnifiedExporter** (出力系): 限定テスト推奨
- **UnifiedExporter** (出力系): 限定テスト推奨
- **UnifiedExporter** (出力系): 限定テスト推奨
- **datetime** (その他): 限定テスト推奨
- **Dict** (その他): 限定テスト推奨
- **Any** (その他): 限定テスト推奨
- **List** (その他): 限定テスト推奨
- **List** (その他): 限定テスト推奨
- **Dict** (その他): 限定テスト推奨
- **Any** (その他): 限定テスト推奨

## 📝 main.py実証モジュール活用戦略

### 🎯 main.pyで実証済みの利点
1. **動作保証**: 実際のバックテスト環境で動作確認済み
2. **互換性**: 既存システムとの完全互換性
3. **パラメータ**: 実用的な設定値が既知
4. **エラーパターン**: 既知の問題と対処法が明確

### 🚀 Phase 0実行順序（main.py実証順）
1. **設定・ログ系**: logger_config, risk_management等
2. **データ取得・前処理系**: data_fetcher, data_processor等
3. **指標計算系**: indicator_calculator, unified_trend_detector等
4. **個別戦略クラス**: main.pyで使用中の7戦略

### ✅ 成功基準（main.py準拠）
- ✅ main.pyと同じパラメータで動作
- ✅ 同じシグナル生成パターン
- ✅ エラーハンドリングの再現
- ✅ パフォーマンス指標の一致

### 🚨 main.py依存問題の回避
- **circular import回避**: 段階的インポートテスト
- **設定依存の分離**: 独立した設定での動作確認
- **データ依存の最小化**: テストデータでの検証

---
**レポート生成日時**: 2025-10-12 10:19:39
**調査対象**: main.py使用中モジュール（実証済み）
**目的**: comprehensive_module_test.py用の確実な再利用候補選定
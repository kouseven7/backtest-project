# DSSMS出力問題 構造分析・特定ロードマップ

## 🔍 問題点構造分析

### 問題の依存関係マップ
```
問題1：銘柄切替数激減 → 問題2,3,4,5に波及影響
    ↓
問題2：日付ループ ← データ処理エンジンの問題
    ↓
問題3,4：保有期間固定 ← 計算ロジックの問題
    ↓
問題5：統計項目未計算 ← データフローの問題
```

## 📋 Task 1: 銘柄切替数激減問題の根本原因特定

### 🎯 目的
DSSMSで銘柄切替回数が117回→1-2回に激減する根本原因を特定

### 🔧 調査手順

#### Task 1.1: 切替数カウント機構の検証
**実行内容:**
```python
# 診断スクリプト作成: diagnose_switch_count.py
def analyze_switch_counting():
    """切替カウント機構の詳細分析"""
    
    # 1. DSSMSBacktester.switch_historyの状態確認
    # 2. 各統一エンジンでの切替数取得方法比較
    # 3. backtest_results削除前後の動作比較
    # 4. データ永続化機構の調査
    
    engines_to_test = [
        'dssms_unified_output_engine.py',
        'dssms_unified_output_engine_fixed.py', 
        'dssms_unified_output_engine_fixed_v3.py',
        'dssms_unified_output_engine_fixed_v4.py'
    ]
```

#### Task 1.2: データ永続化・キャッシュ問題調査
**推測される原因:**
- `backtest_results/dssms_results/`ディレクトリ内の隠れファイル
- DSSMSシステムの状態保存機構
- インメモリキャッシュの問題

**調査コマンド:**
```bash
# 隠れファイル・ディレクトリ確認
dir /a:h backtest_results\dssms_results\
Get-ChildItem -Hidden -Recurse backtest_results\dssms_results\

# DSSMS設定ファイル調査
find . -name "*dssms*" -type f | grep -E "\.(json|config|cache)$"
```

#### Task 1.3: 統一エンジン影響度分析
**実行内容:**
- 各エンジンファイルの`switch_history`処理ロジック比較
- バックテスター→エンジン間のデータフロー追跡
- 切替判定ロジックの変更点特定

---

## 📋 Task 2: 日付ループ問題の特定

### 🎯 目的  
「2023-12-31 → 2023-01-01」の不正ループ原因を特定

### 🔧 調査手順

#### Task 2.1: 日付処理ロジックの検証
**調査対象:**
```python
# 各エンジンの日付修正ロジック比較
files_to_analyze = [
    'dssms_unified_output_engine_fixed.py:_fix_date_inconsistencies_improved',
    'dssms_unified_output_engine_fixed_v3.py:日付修正部分',
    'dssms_unified_output_engine_fixed_v4.py:日付修正部分'
]

# 特に以下のロジックを調査:
# 1. expected_year = 2023 の無限ループ可能性
# 2. 年末→年始の日付境界処理
# 3. pd.to_datetime()の変換ロジック
```

#### Task 2.2: ポートフォリオ価値データフローの追跡
**実行内容:**
```python
def trace_portfolio_data_flow():
    """ポートフォリオデータの変換過程を追跡"""
    
    # 1. DSSMSBacktester.portfolio_values の生データ確認
    # 2. _convert_backtester_results での変換過程
    # 3. _fix_date_inconsistencies での修正過程
    # 4. Excel出力での最終データ確認
```

---

## 📋 Task 3: 保有期間固定問題の特定

### 🎯 目的
保有期間が24時間固定される計算ロジックの問題を特定

### 🔧 調査手順

#### Task 3.1: 保有期間計算ロジックの比較分析
**調査対象:**
```python
# 各エンジンでの保有期間計算方法比較
calculation_methods = {
    'v1': 'TODO: 実際の計算 → 24.0時間固定',
    'v3': '次のスイッチ日時 - 現在スイッチ日時',
    'v4': 'actual_holding_hours計算ロジック'
}

# 特に調査すべき箇所:
# 1. SymbolSwitchオブジェクトのtimestamp取得
# 2. 日付差分計算の実装
# 3. 最後のスイッチでのデフォルト値処理
```

#### Task 3.2: 修正版エンジン統合後の回帰問題分析
**推測される原因:**
- 修正版エンジンが正しく統合されていない
- 古いエンジンが呼ばれている
- バックテスター側の`get_performance_metrics`等の未実装

---

## 📋 Task 4: 戦略別統計未計算問題の特定

### 🎯 目的
戦略別統計シートの計算ロジック問題を特定

### 🔧 調査手順

#### Task 4.1: 統計計算データソースの検証
**調査内容:**
```python
def analyze_strategy_statistics_data_source():
    """戦略統計計算のデータソース調査"""
    
    # 1. DSSMSBacktester.get_strategy_statistics()の実装状況
    # 2. trade_historyデータの構造と内容
    # 3. switch_historyからの統計計算可能性
    # 4. 各統一エンジンでの統計計算実装比較
```

#### Task 4.2: 計算ロジック実装状況の確認
**調査対象:**
- 勝率計算: `profitable_trades / total_trades`
- 平均利益/損失: `profit_trades.mean()` / `loss_trades.mean()`
- プロフィットファクター: `total_profit / abs(total_loss)`
- 各エンジンでの実装レベル比較

---

## 🚀 実行計画

### Phase 1: 根本原因特定 (優先度: 最高)
1. **Task 1実行**: 銘柄切替数問題の特定 (30分)
2. **Task 2実行**: 日付ループ問題の特定 (20分)

### Phase 2: 計算ロジック問題特定 (優先度: 高)  
3. **Task 3実行**: 保有期間固定問題の特定 (20分)
4. **Task 4実行**: 統計未計算問題の特定 (15分)

### Phase 3: 統合解決策策定 (優先度: 中)
5. **問題間依存関係の整理** (10分)
6. **修復優先順位の決定** (5分)

---

## 📞 必要なファイル・情報

### 即座に必要なファイル
1. **最新のDSSMSBacktesterクラス**: `src/dssms/dssms_backtester.py`
2. **各統一エンジンファイル**: 
   - `dssms_unified_output_engine.py`
   - `dssms_unified_output_engine_fixed.py`
   - `dssms_unified_output_engine_fixed_v3.py` 
   - `dssms_unified_output_engine_fixed_v4.py`
3. **問題のあるExcelファイル**: `backtest_results/dssms_results/dssms_unified_backtest_20250910_213413.xlsx`
4. **最新の実行ログファイル**: コンソール出力またはログファイル

### 調査過程で要求する可能性があるファイル
1. **設定・キャッシュファイル**: `src/dssms/*.json`, `src/dssms/*.config`
2. **SymbolSwitchクラス定義**: `src/dssms/`配下の関連ファイル
3. **バックテスト結果ディレクトリ全体**: `backtest_results/dssms_results/`
4. **Git履歴**: 問題発生前後のコミット差分

---

## 💡 質問事項

### 🔍 現象についての追加情報
1. **銘柄切替数激減のタイミング**: 
   - 最後に117回だったのはいつ頃ですか？
   - 何らかのファイル変更やシステム操作の直後でしたか？A.最後に117回だったのはcb844e0のコミットに戻した直後、１．２回バックテストコマンドのpython "src\dssms\dssms_backtester.py"を実行した後に117回ではなくなった

2. **backtest_resultsディレクトリ削除の詳細**:
   - 削除したファイルの種類（.xlsx, .txt, .json以外にありましたか？）A.サブディレクトリであるdssme_results内のファイルを消した
   - サブディレクトリも含めて完全に空にしましたか？A.ないはず

3. **問題の再現性**:
   - 毎回同じ切替数になりますか？（例：必ず2回）A.必ず２回になる
   - 実行するたびに変わりますか？A.変わらない

4. **使用している統一エンジン**:
   - 現在、どのエンジンファイルが実際に使用されていますか？
   - `src/dssms/dssms_backtester.py`内でどのエンジンをインポートしていますか？A.[text](../../src/dssms/dssms_backtester.py)import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存DSSMSコンポーネントのインポート
try:
    from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem, PriorityLevel, RankingScore, SelectionResult
    from src.dssms.intelligent_switch_manager import IntelligentSwitchManager, SwitchDecision, PositionEvaluation
    from src.dssms.dssms_data_manager import DSSMSDataManager
    from src.dssms.market_condition_monitor import MarketConditionMonitor
    from src.dssms.comprehensive_scoring_engine import ComprehensiveScoringEngine
    from src.dssms.perfect_order_detector import PerfectOrderDetector
    # Task 3.4 コンポーネントの追加
    from src.dssms.task34_workflow_coordinator import Task34WorkflowCoordinator, Task34WorkflowConfig
    from src.dssms.performance_target_manager import PerformanceTargetManager, TargetPhase
    from src.dssms.comprehensive_evaluator import ComprehensiveEvaluator
    from src.dssms.emergency_fix_coordinator import EmergencyFixCoordinator
    from src.dssms.performance_achievement_reporter import PerformanceAchievementReporter
except ImportError:
    # 直接実行時の相対インポート対応
    try:
        from hierarchical_ranking_system import HierarchicalRankingSystem, PriorityLevel, RankingScore, SelectionResult
        from intelligent_switch_manager import IntelligentSwitchManager, SwitchDecision, PositionEvaluation
        from dssms_data_manager import DSSMSDataManager
        from market_condition_monitor import MarketConditionMonitor
        from comprehensive_scoring_engine import ComprehensiveScoringEngine
        from perfect_order_detector import PerfectOrderDetector
    except ImportError as e:
        # DSSMSコンポーネントが利用できない場合のフォールバック
        import warnings
        warnings.warn(f"DSSMS components not fully available: {e}. Some functionality will be limited.", UserWarning)
        HierarchicalRankingSystem = None
        IntelligentSwitchManager = None
        DSSMSDataManager = None
        MarketConditionMonitor = None
        ComprehensiveScoringEngine = None
        PerfectOrderDetector = None

# 既存システムインポート
from config.logger_config import setup_logger
from config.risk_management import RiskManagement
from output.dssms_excel_exporter_v2 import DSSMSExcelExporterV2  # 新しいV2システム
from data_fetcher import fetch_stock_data
from trade_simulation import simulate_trades

# 警告を抑制
warnings.filterwarnings('ignoreA.

### 🛠️ 技術的確認事項
1. **Pythonキャッシュ**: `__pycache__`ディレクトリを削除してテストしましたか？A.操作した覚えはありませんが、わかりません。
2. **インポート確認**: `import`文でどのエンジンクラスを使っているか確認できますか？A.import DSSMSExcelExporterV2だと思いますが、[text](../../src/dssms/dssms_backtester.py)[text](../../src/dssms/dssms_backtester_v2.py)[text](../../src/dssms/dssms_backtester_v2_updated.py)[text](../../src/dssms/dssms_backtester_backup.py)[text](../../src/dssms/dssms_backtester_backup_20250828_222919.py)とあり、どれが機能しているかなぞです
3. **エラーログ**: 実行時に警告やエラーメッセージは出ていますか？A.まったくないわけではありませんが、関連しそうな部分でのエラーはでていないとおもいます

---

## 📋 Task 5: 追加重要項目調査（2025-09-12追加）

### 🎯 roadmap2.md分析による新発見項目

#### Task 5.1: IntelligentSwitchManager統合状況の詳細調査
**発見された問題:**
- `intelligent_switch_manager`が未設定状態で動作
- `IntelligentSwitchManager`の適切な統合が確認できていない

**調査内容:**
```python
def analyze_intelligent_switch_manager_integration():
    """IntelligentSwitchManager統合状況の詳細分析"""
    
    # 1. DSSMSBacktester初期化時のISM設定確認
    # 2. ISM.SwitchDecisionロジックの動作確認  
    # 3. 設定ファイルconfig/dssms/intelligent_switch_config.json読み込み状況
    # 4. 切替判定における実際のISM使用箇所特定
    # 5. ISM未設定時のフォールバック動作確認
```

#### Task 5.2: 決定論的モード影響度の定量分析
**発見された懸念:**
```python
enable_score_noise: False
enable_switching_probability: False  
use_fixed_execution: True
```

**調査内容:**
```python
def analyze_deterministic_mode_impact():
    """決定論的モード設定が切替頻度に与える具体的影響の測定"""
    
    # 1. 決定論的モード ON/OFF での切替数比較テスト
    # 2. ランダム要素が切替判定に与える影響度測定
    # 3. 過去117回時点での決定論的モード設定確認
    # 4. noise_factor, switching_probability適正値の特定
    # 5. シード値固定(42)による再現性への影響分析
```

#### Task 5.3: DSSMSExcelExporterV2とunified_engine併用問題調査
**発見された構造問題:**
- DSSMSExcelExporterV2（メインエンジン）と統一エンジン4種の併用混乱
- switch_history処理の重複・競合の可能性

**調査内容:**
```python
def analyze_engine_coexistence_issues():
    """メインエンジンと統一エンジンの併用問題分析"""
    
    # 1. src/dssms/dssms_backtester.pyでの実際使用エンジン特定
    # 2. DSSMSExcelExporterV2の処理フローとswitch_history扱い
    # 3. 統一エンジン4種のうち実際に呼ばれているエンジン特定
    # 4. エンジン間でのデータ競合・上書き問題確認
    # 5. 最適エンジン構成の提案（単一エンジン化検討）
```

#### Task 5.4: 銘柄データ品質・ランキング精度問題調査
**発見された懸念:**
- 全銘柄で"possibly delisted"エラー発生
- データ取得失敗がランキング計算・切替判定に与える影響

**調査内容:**
```python
def analyze_data_quality_ranking_impact():
    """データ品質がランキング・切替判定に与える影響分析"""
    
    # 1. yfinanceデータ取得エラーの具体的銘柄・期間特定
    # 2. ランキングシステムでのエラーデータ処理方法確認
    # 3. 不正確ランキングが切替判定閾値に与える影響測定
    # 4. 代替データソース・エラーハンドリング改善案
    # 5. 過去117回時点でのデータ品質比較
```

#### Task 5.5: エンジン品質格差根本原因分析（Critical）
**Task 4.2で発見された深刻な品質格差:**
- v1: 85.0点（最優秀）
- v2: 31.7点（低品質）  
- v3: 0.0点（完全未実装）
- v4: 55.0点（中品質）

**調査内容:**
```python
def analyze_engine_quality_gap_root_cause():
    """エンジン品質格差の根本原因と修正優先度分析"""
    
    # 1. v1エンジン成功要因の詳細分析（85.0点の理由）
    # 2. v2,v3,v4エンジンの実装不備の具体的原因特定
    # 3. 計算式実装エラーのパターン分析
    # 4. エンジン品質統一のための実装ガイドライン策定
    # 5. 品質向上優先順位とコスト効率分析
```

#### Task 5.6: 計算式実装エラーパターン調査（Critical）
**Task 4.2で発見された計算式の深刻な問題:**
```
勝率計算期待: profitable_trades / total_trades
v1実装: len(trades_df[trades_df['pnl'] > 0])  ← 分母なし

プロフィットファクター期待: total_profit / abs(total_loss)  
v1実装: len(trades_df[trades_df['pnl'] > 0])  ← 完全に間違った公式
```

**調査内容:**
```python
def analyze_calculation_formula_error_patterns():
    """計算式実装エラーのパターンと修正方針分析"""
    
    # 1. 全エンジンの統計計算式総点検
    # 2. 数学的に正しい計算式の定義・検証
    # 3. エラーパターンの分類（分母欠如、変数誤用、公式間違い等）
    # 4. 計算式正確性検証テストケース作成
    # 5. 段階的修正計画とテスト方針策定
```

### 🚀 追加調査実行計画

#### Phase A: 緊急度最高項目（Task 5.5, 5.6）
1. **Task 5.5実行**: エンジン品質格差根本原因分析（30分）
2. **Task 5.6実行**: 計算式実装エラーパターン調査（25分）

#### Phase B: 切替頻度直結項目（Task 5.1, 5.2）  
3. **Task 5.1実行**: IntelligentSwitchManager統合状況調査（20分）
4. **Task 5.2実行**: 決定論的モード影響度分析（20分）

#### Phase C: システム構造改善項目（Task 5.3, 5.4）
5. **Task 5.3実行**: エンジン併用問題調査（15分）
6. **Task 5.4実行**: データ品質・ランキング精度問題調査（15分）

#### Phase D: 統合解決策策定
7. **追加問題間依存関係の整理**（10分）
8. **修復優先順位の再評価**（5分）

---

**次のアクション**: まず「Task 1.1: 切替数カウント機構の検証」から開始し、診断スクリプトを作成して根本原因を特定します。
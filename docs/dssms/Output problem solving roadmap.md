# 🛠️ DSSMS出力システム問題解決ロードマップ
**Dynamic Stock Selection Management System - Output Problem Solution Roadmap**

## [CHART] Phase 1調査結果サマリー

### [TARGET] 特定された根本原因
**実行日**: 2025年9月4日  
**調査完了**: Phase 1.1 (モジュール構造調査) + Phase 1.4.2 (詳細エラー分析)

#### 🔴 主要問題：欠損ファイル
1. **`output/simple_excel_exporter.py`が存在しない**
   - `output/simple_simulation_handler.py`の行21でインポートしようとしている
   - `from output.simple_excel_exporter import save_backtest_results_simple`
   - これが`main.py`実行時の`ModuleNotFoundError`の直接原因

#### [LIST] 調査で確認された事実
[OK] **正常な部分**:
- `output/`ディレクトリ存在 
- `output/__init__.py`存在
- `output/simple_simulation_handler.py`存在
- `output/dssms_excel_exporter_v2.py`存在
- Pythonモジュール解決パス正常

[ERROR] **問題の部分**:
- `output/simple_excel_exporter.py`**完全に欠損**
- `save_backtest_results_simple`関数が利用できない

### [UP] 影響度分析
- **即座の影響**: main.py完全実行不可
- **波及的影響**: DSSMS統合バックテストシステム停止
- **解決優先度**: 🔴 最高 (即座修復必要)

---

## 🛠️ Phase 2: 解決実装フェーズ

### Phase 2.1: 緊急修復 - 欠損ファイル作成
**目的**: main.py実行エラーの即座解決  
**優先度**: 🔴 最高  
**所要時間**: 30-45分  

#### Task 2.1.1: simple_excel_exporter.py緊急作成
**実行内容**: 
- 既存`dssms_excel_exporter_v2.py`を参考に基本機能を実装
- `save_backtest_results_simple`関数の最小限実装
- エラーハンドリング強化

**実装方針**:
```python
# output/simple_excel_exporter.py の基本構造
def save_backtest_results_simple(
    results: dict,
    filename: str = None,
    output_dir: str = "output"
) -> str:
    """
    バックテスト結果をシンプルなExcel形式で保存
    
    Args:
        results: バックテスト結果データ
        filename: 出力ファイル名（自動生成可能）
        output_dir: 出力ディレクトリ
    
    Returns:
        str: 保存されたファイルパス
    """
    # 基本的なExcel出力実装
    pass
```

#### Task 2.1.2: インポート互換性確保
**実行内容**:
- `simple_simulation_handler.py`からの正常インポート確認
- 基本動作テスト実行
- main.py実行エラー解決確認

#### Task 2.1.3: 暫定動作確認
**実行内容**:
- main.py段階的実行テスト
- 基本出力機能動作確認
- 次段階修復準備

### Phase 2.2: 出力機能正規化
**目的**: Excel・テキストレポート出力品質向上  
**優先度**: 🟡 高  
**所要時間**: 1-2時間  

#### Task 2.2.1: 既存出力システム分析
**実行内容**:
```powershell
# 既存出力システムの詳細分析
python -c "
# dssms_excel_exporter_v2.pyの機能分析
# 出力フォーマット・関数インターフェース調査
# データ構造・出力仕様の把握
"
```

#### Task 2.2.2: 統合出力システム設計
**実行内容**:
- Excel出力の標準化
- テキストレポート形式統一
- データソース一元化

#### Task 2.2.3: 出力データ整合性確保
**実行内容**:
- 計算ロジック統一
- 同一データソースからの出力保証
- 検証・承認フロー実装

### Phase 2.3: DSSMS出力品質最適化  
**目的**: データ不整合解決・信頼性向上  
**優先度**: 🟢 中  
**所要時間**: 2-3時間  

#### Task 2.3.1: バックテストデータ収集最適化
**実行内容**:
- DSSMSBacktester出力データ正規化
- パフォーマンス指標計算ロジック統一
- 取引履歴データ処理改善

#### Task 2.3.2: 多形式出力エンジン構築
**実行内容**:
- 統一データモデル設計
- Excel・テキスト・JSON出力の一元化
- テンプレートベース出力システム

#### Task 2.3.3: 品質保証システム
**実行内容**:
- 出力前データ検証
- 自動整合性チェック
- 回帰テスト自動化

---

## 📅 実行スケジュール

### [ALERT] Phase 2.1: 緊急修復 (今すぐ実行)
**今日中完了目標**
- [ ] **Task 2.1.1**: simple_excel_exporter.py作成 (30分)
- [ ] **Task 2.1.2**: インポート互換性確保 (10分)  
- [ ] **Task 2.1.3**: 暫定動作確認 (5分)
- [ ] **検証**: main.py正常実行確認

### ⚡ Phase 2.2: 出力機能正規化 (明日実行)
**明日完了目標**
- [ ] **Task 2.2.1**: 既存システム分析 (30分)
- [ ] **Task 2.2.2**: 統合出力システム設計 (60分)
- [ ] **Task 2.2.3**: データ整合性確保 (30分)
- [ ] **検証**: Excel・テキスト出力整合性確認

### [TOOL] Phase 2.3: 品質最適化 (今週内)
**週内完了目標**  
- [ ] **Task 2.3.1**: データ収集最適化 (90分)
- [ ] **Task 2.3.2**: 多形式出力エンジン (120分)
- [ ] **Task 2.3.3**: 品質保証システム (60分)
- [ ] **最終検証**: 10回連続実行での整合性確認

---

## [TARGET] 各Phase完了基準

### Phase 2.1完了基準
- [ ] main.py正常実行（エラーなし）
- [ ] 基本Excel出力ファイル生成
- [ ] `save_backtest_results_simple`関数動作確認

### Phase 2.2完了基準  
- [ ] Excel・テキストレポート基本整合性確保
- [ ] 主要指標（総リターン、切替回数、最終価値）の値一致
- [ ] 出力形式統一

### Phase 2.3完了基準
- [ ] 詳細データ完全整合性（取引履歴含む）
- [ ] 複数回実行での同一結果保証
- [ ] 自動検証システム稼働

---

## [TOOL] Task 2.1.1 詳細実装計画

### 実装アプローチ
1. **既存コード調査**: `dssms_excel_exporter_v2.py`の分析
2. **最小限実装**: 基本Excel出力機能
3. **段階的機能拡張**: 必要に応じて機能追加

### コード設計方針
```python
# ファイル: output/simple_excel_exporter.py

import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, Optional
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

def save_backtest_results_simple(
    results: Dict[str, Any],
    filename: Optional[str] = None,
    output_dir: str = "output"
) -> str:
    """
    バックテスト結果をシンプルなExcel形式で保存
    
    Phase 2.1の緊急対応として最小限の機能を実装
    後のPhaseで機能拡張予定
    """
    
    # 1. ファイル名生成
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_backtest_results_{timestamp}.xlsx"
    
    # 2. 出力パス確保
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    # 3. 基本データ抽出・Excel出力
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # サマリーシート作成
            _create_summary_sheet(results, writer)
            
            # 取引履歴シート作成（データがある場合）
            if 'trades' in results:
                _create_trades_sheet(results['trades'], writer)
                
        return filepath
        
    except Exception as e:
        # エラーハンドリング - 最低限のCSV出力
        csv_path = filepath.replace('.xlsx', '_fallback.csv')
        _create_fallback_csv(results, csv_path)
        return csv_path

def _create_summary_sheet(results: Dict[str, Any], writer):
    """サマリーシート作成"""
    # 基本実装 - Phase 2.2で詳細化
    pass

def _create_trades_sheet(trades_data, writer):
    """取引履歴シート作成"""  
    # 基本実装 - Phase 2.2で詳細化
    pass

def _create_fallback_csv(results: Dict[str, Any], filepath: str):
    """緊急時のCSV出力"""
    # 最低限のデータ保存
    pass
```

### 実装ステップ
1. **Step 1**: 基本ファイル作成・インポートエラー解決
2. **Step 2**: 最小限のExcel出力機能実装  
3. **Step 3**: main.py実行テスト・動作確認
4. **Step 4**: Phase 2.2準備（既存システム調査）

---

## 📞 実行サポート

### 緊急時対応
各Task実行時に問題が発生した場合：
1. エラーメッセージ全文を報告
2. 実行コマンド・環境情報を共有
3. 即座に代替手順を提供

### Phase間連携
- Phase 2.1完了後、即座にPhase 2.2開始可能
- 各Phaseの成果物は次Phaseの入力として活用
- 問題発生時の早期エスカレーション体制

---

**[ROCKET] 次のアクション**: Task 2.1.1の実行開始 - `simple_excel_exporter.py`の作成から開始します。

# DSSMS Excel Exporter 統合アーキテクチャ設計

## 📊 V2版機能分析結果

### **ファイル基本情報**
- **V2版**: 37.8KB、886行 - 高機能な完全実装
- **src版**: 1.7KB、45行 - 最小プレースホルダー

### **V2版 主要機能（9メソッド）**
```python
class DSSMSExcelExporterV2:
    def export_dssms_results()        # メインエクスポート機能
    def _create_summary_sheet()       # サマリーシート作成
    def _create_performance_sheet()   # パフォーマンスシート作成  
    def _create_trade_history_sheet() # 取引履歴シート作成
    def _create_daily_pnl_sheet()     # 日次損益シート作成
    def _create_strategy_stats_sheet() # 戦略統計シート作成
    def _create_switch_analysis_sheet() # 切替分析シート作成
    def _create_charts_sheet()        # チャートシート作成
    # + helper methods
```

### **依存関係（12ライブラリ）**
```python
# データ処理
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Excel処理
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import LineChart, Reference

# プロジェクト固有
from config.logger_config import setup_logger
```

## 🎯 統合方針

### **統合戦略**
1. **完全置換方式**: src版をV2版で完全置換
2. **インターフェース統一**: 既存の呼び出し元との互換性維持
3. **品質向上**: 0%生成箇所の事前防止

### **統合後の構造**
```
src/dssms/dssms_excel_exporter.py（統合版）
├── DSSMSExcelExporter（統一クラス名）
├── 全V2機能の移植
├── プレースホルダーメソッドの完全実装
└── シングルトンパターン維持
```

## 🔄 統合マッピング

### **メソッド統合計画**
| src版プレースホルダー | V2版実装 | 統合後 |
|---------------------|----------|--------|
| `export_data()` | `export_dssms_results()` | ✅ 統合 |
| `export_rankings()` | N/A | ✅ 新規実装 |
| `export_switch_analysis()` | `_create_switch_analysis_sheet()` | ✅ 統合 |
| `initialize()` | `__init__()` | ✅ 統合 |

### **クラス名統一**
- `DSSMSExcelExporterV2` → `DSSMSExcelExporter`
- 既存の`get_excel_exporter()`シングルトン関数は維持

## ⚠️ リスク分析

### **統合リスク**
1. **HIGH**: V2版の886行コードの移植 - 構文エラーリスク
2. **MEDIUM**: 依存関係の確実性 - openpyxl等のインポートエラー
3. **LOW**: 既存呼び出し元への影響 - インターフェース変更

### **対策**
- 段階的統合：基本機能 → 高度機能 → テスト
- バックアップ確保：既存プレースホルダーをバックアップ
- 統合テスト：各メソッド個別 → 全体統合テスト

## 📋 実装手順

### **Step 2.2 実装計画（45分）**
1. **基盤移植（15分）**: クラス定義・__init__・基本インポート
2. **主要機能移植（20分）**: export_dssms_results()とヘルパーメソッド
3. **インターフェース調整（10分）**: プレースホルダーメソッドの実装

### **検証項目**
- [ ] 基本インポートテスト成功
- [ ] メインexport機能の動作確認  
- [ ] 既存呼び出し元からのアクセス確認
- [ ] エラーハンドリング動作確認

---

**作成日**: 2025年9月30日
**Phase 2 Step 2.1 完了準備**: ✅ 統合設計完了、Step 2.2実装準備完了
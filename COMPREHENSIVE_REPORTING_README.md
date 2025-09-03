# 包括的レポートシステム

DSSMS改善プロジェクト Phase 3 Task 3.3  
包括的レポートシステム実装完了

## 概要

このシステムは、DSSMS（Dynamic Strategy Selection and Management System）のための包括的なレポート生成・管理システムです。既存のレポートシステムと統合し、ハイブリッド型アプローチでHTML中心のインタラクティブレポートとマルチフォーマットエクスポート機能を提供します。

## 主要機能

### 🎯 コアコンポーネント

1. **ComprehensiveReportEngine** - メインエンジン
   - 包括的レポート生成
   - 比較レポート作成
   - レポート管理機能

2. **DataAggregator** - データ集約
   - 階層化データ処理（サマリー/詳細/包括的）
   - 既存システム統合
   - DSSMS・戦略・パフォーマンス・リスクデータ統合

3. **VisualizationGenerator** - 可視化生成
   - Chart.js ベースのインタラクティブチャート
   - Bootstrap5 レスポンシブデザイン
   - 複数チャートタイプ対応

4. **ReportTemplateManager** - テンプレート管理
   - Bootstrap5 ベーステンプレート
   - カスタマイズ可能なテーマ
   - レベル別コンテンツ生成

5. **ExportManager** - エクスポート管理
   - マルチフォーマット出力（HTML/Excel/PDF/JSON）
   - バッチエクスポート機能
   - エクスポート履歴管理

### 📊 レポート詳細レベル

- **Summary（サマリー）**: 基本統計と概要
- **Detailed（詳細）**: 構造化された詳細情報
- **Comprehensive（包括的）**: 全データと高度な分析

### 📈 レポートタイプ

- **Comprehensive**: 包括的分析レポート
- **Comparison**: 戦略・期間・設定比較レポート
- **Performance**: パフォーマンス分析レポート
- **Risk**: リスク分析レポート

## インストール・セットアップ

### 必須依存関係

```bash
pip install pandas numpy openpyxl
```

### オプション依存関係

```bash
# PDF生成用（推奨）
pip install weasyprint
# または
pip install pdfkit

# テスト実行用
pip install pytest
```

### ディレクトリ構造

```
src/reports/comprehensive/
├── __init__.py
├── comprehensive_report_engine.py
├── data_aggregator.py
├── visualization_generator.py
├── report_template_manager.py
└── export_manager.py

config/comprehensive_reporting/
├── report_config.json
├── template_config.json
└── visualization_config.json

templates/reports/
└── (自動生成テンプレート)

output/comprehensive_reports/
└── (生成レポート出力先)
```

## 使用方法

### 基本的なレポート生成

```python
from src.reports.comprehensive import ComprehensiveReportEngine

# エンジン初期化
engine = ComprehensiveReportEngine()

# 包括的レポート生成
result = engine.generate_comprehensive_report(
    report_type="comprehensive",
    level="detailed",
    strategies=["VWAPBreakoutStrategy", "ConventionalTradingStrategy"]
)

if result['success']:
    print(f"レポート生成完了: {result['report_path']}")
else:
    print(f"エラー: {result['error']}")
```

### エクスポート機能

```python
# JSONエクスポート
json_result = engine.export_report("json")

# Excelエクスポート
excel_result = engine.export_report("excel")

# PDFエクスポート
pdf_result = engine.export_report("pdf")

# バッチエクスポート
batch_result = engine.export_manager.batch_export(
    report_data=engine.report_data,
    report_id=engine.current_report_id,
    output_dir=engine.output_dir,
    formats=["json", "excel", "pdf"]
)
```

### 比較レポート

```python
# 戦略比較
comparison_items = [
    {'name': 'VWAPBreakoutStrategy'},
    {'name': 'ConventionalTradingStrategy'}
]

comparison_result = engine.generate_comparison_report(
    comparison_items=comparison_items,
    comparison_type="strategies",
    level="detailed"
)
```

## デモ実行

```bash
# PowerShell
python demo_comprehensive_reporting.py

# 機能テスト
python -m pytest test_comprehensive_reporting.py -v
```

## 設定オプション

### report_config.json

- `output_directory`: 出力ディレクトリ
- `report_levels`: 利用可能な詳細レベル
- `export_formats`: サポートするエクスポート形式
- `data_aggregation`: データ集約設定
- `performance_settings`: パフォーマンス設定

### template_config.json

- `themes`: カラーテーマ設定
- `layout`: レイアウト設定
- `components`: UIコンポーネント設定
- `responsive`: レスポンシブ設定

### visualization_config.json

- `chart_types`: チャートタイプ設定
- `color_schemes`: カラースキーム
- `performance`: 可視化パフォーマンス設定
- `interaction`: インタラクション設定

## パフォーマンス

- **レポート生成時間**: 通常30秒以内
- **メモリ使用量**: 設定可能な制限（デフォルト512MB）
- **最大データポイント**: 10,000ポイント（設定可能）
- **レスポンシブ対応**: モバイル・タブレット・デスクトップ

## 既存システム統合

以下の既存レポートシステムと統合されます：

- `strategy_comparison.py` - 戦略比較レポート
- `error_diagnostic_reporter.py` - エラー診断レポート
- `dssms_enhanced_reporter.py` - DSSMS拡張レポート
- `report_integration_manager.py` - レポート統合管理

## トラブルシューティング

### よくある問題

1. **PDF生成エラー**
   ```bash
   pip install weasyprint
   # または
   pip install pdfkit
   ```

2. **Excel生成エラー**
   ```bash
   pip install openpyxl
   ```

3. **メモリ不足**
   - `report_config.json` の `performance_settings.memory_limit_mb` を調整
   - 詳細レベルを "summary" に変更

4. **生成時間の長さ**
   - `max_data_points` を削減
   - `lazy_loading` を有効化

### ログ確認

```bash
# ログファイル確認
tail -f logs/comprehensive_reporting.log
```

## セキュリティ

- HTMLサニタイゼーション有効
- ファイルアクセス制限
- 最大エクスポートサイズ制限（50MB）
- XSS防止対策

## 開発・拡張

### 新しいチャートタイプ追加

1. `visualization_config.json` に定義追加
2. `VisualizationGenerator` にテンプレート追加
3. Chart.js 設定をカスタマイズ

### 新しいエクスポート形式追加

1. `ExportManager` にエクスポート関数追加
2. 必要な依存関係をインストール
3. `available_formats` に追加

### カスタムテンプレート

1. `ReportTemplateManager` にテンプレート追加
2. `template_config.json` に設定追加
3. HTML/CSS/JavaScript をカスタマイズ

## バージョン履歴

- **v1.0.0** (2025-01-22): 初回リリース
  - 5つのコアコンポーネント実装
  - 3つの詳細レベル対応
  - 4つのエクスポート形式対応
  - Bootstrap5 + Chart.js ベース
  - 既存システム統合完了

## ライセンス

DSSMS改善プロジェクト内部使用

## サポート

- ログファイル: `logs/comprehensive_reporting.log`
- 設定ファイル: `config/comprehensive_reporting/`
- デモファイル: `demo_comprehensive_reporting.py`
- テストファイル: `test_comprehensive_reporting.py`

---

**実装完了**: DSSMS Phase 3 Task 3.3 包括的レポートシステム  
**実装日**: 2025年1月22日  
**アーキテクチャ**: ハイブリッド型統合アプローチ

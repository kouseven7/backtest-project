# DSSMS専用出力統合問題調査（最新実態調査版）

## 目的
DSSMSバックテスト実行後に、DSSMS専用の10ファイルが単一のフォルダ（output/dssms_integration/）内に正しく生成されるようにする。マルチ戦略システムの出力とは独立して管理し、重複・分散を防止する。

## 問題の詳細（2026-01-08 実態調査完了）

### 現在の実際の状況
実際のファイル出力調査結果：

**1. dssms_20260108_135121/ フォルダ（2ファイル）**
- dssms_comprehensive_report.json (15,305バイト) ✅
- dssms_switch_history.csv (324バイト) ✅

**2. 6178_20260108_135121/ フォルダ（7ファイル）**
- 6178_execution_results.json (100バイト)
- 6178_performance_metrics.json (743バイト)
- 6178_performance_summary.csv (274バイト)
- 6178_SUMMARY.txt (1,011バイト)
- 6178_trade_analysis.json (80バイト)
- main_comprehensive_report_6178_20260108_135121.txt (1,893バイト)
- portfolio_equity_curve.csv (2,190バイト)

**3. 直接出力ファイル群（多数）**
- comprehensive_report_YYYYMMDD_HHMMSS.json ファイル群

### システム分離の明確化
- **DSSMS専用システム**: `src/dssms/dssms_integrated_main.py` (エントリーポイント)
- **マルチ戦略システム**: `main_new.py` (エントリーポイント、今回修正対象外)

### ゴール（成功条件）
- [ ] `output/dssms_integration/dssms_YYYYMMDD_HHMMSS`フォルダ内にDSSMS専用10ファイルが統合生成
- [ ] `output/comprehensive_reports/`配下の銘柄別フォルダが生成されない（重複排除）
- [ ] マルチ戦略システムの出力に影響を与えない（独立性維持）

## 影響範囲

### 実際に必要な10ファイル統合仕様
現在の9ファイル + 新規1ファイル = 10ファイル統合：

1. **dssms_comprehensive_report.json** (15,305バイト) ✅実装済み
2. **dssms_switch_history.csv** (324バイト) ✅実装済み
3. **execution_results.json** (100バイト) ← 6178_プレフィックス削除
4. **performance_metrics.json** (743バイト) ← 6178_プレフィックス削除
5. **performance_summary.csv** (274バイト) ← 6178_プレフィックス削除
6. **trade_analysis.json** (80バイト) ← 6178_プレフィックス削除
7. **comprehensive_report.txt** (1,893バイト) ← ファイル名簡略化
8. **portfolio_equity_curve.csv** (2,190バイト)
9. **summary.txt** (1,011バイト) ← 6178_プレフィックス削除
10. **dssms_execution_log.txt** ← 新規追加（システム実行ログ）

## 根本原因と解決方針

### 原因分析
1. **ComprehensiveReporter**が銘柄ベースフォルダ（6178_timestamp）を自動生成
2. DSSMSとマルチ戦略の出力ロジックが混在
3. ファイル生成が2つのシステムに分散している

### 解決方針
- **src/dssms/dssms_integrated_main.py**の`_generate_outputs`メソッドを完全書き換え
- ComprehensiveReporter使用を停止
- 10ファイル全てを`dssms_{timestamp}`フォルダに直接生成
- マルチ戦略システムの出力に影響を与えない独立実装

## 実装計画

### Phase 1: ✅調査完了
実際のファイル構成と分散状況を把握完了

### Phase 2: 実装修正（次のステップ）
1. `src/dssms/dssms_integrated_main.py`修正
2. `_generate_outputs`メソッドの完全書き換え
3. 10ファイル生成メソッドの個別実装

### Phase 3: 検証テスト
実際にテスト実行してファイル生成を確認する

---
更新日: 2026-01-08 15:30
状態: 調査完了・実装準備中
7. `dssms_market_conditions.csv` （未確認）
8. `dssms_backtest_details.json` （未確認）
9. `dssms_strategy_effectiveness.csv` （未確認）
10. `dssms_consolidated_report.txt` （未確認）

### 除外対象（マルチ戦略システム出力）
- `6178_all_transactions.csv` (マルチ戦略専用)
- `main_comprehensive_report_6178_20260108_135121.txt` (マルチ戦略専用)

## 調査項目
1. **DSSMS出力処理**の特定（`src/dssms/dssms_integrated_main.py`内）
2. **出力先決定ロジック**の解析
3. **10ファイル生成処理**の追跡
4. **マルチ戦略システム出力との分離**確認

## 修正戦略
1. **Phase 1**: DSSMS出力先統一（dssms_integrationフォルダ専用化）
2. **Phase 2**: 10ファイル完全生成検証
3. **Phase 3**: マルチ戦略システムとの独立性確認

## 制約事項
- マルチ戦略システム（main_new.py）の出力を変更しない
- DSSMSの既存機能に影響を与えない
- ファイル内容の品質を維持
- ルックアヘッドバイアス禁止制約を遵守

## エントリーポイント特定
- **調査対象**: `src/dssms/dssms_integrated_main.py`
- **対象外**: `main_new.py`（マルチ戦略システム）
- **対象外**: `main.py`（非アクティブファイル）

---
**作成日**: 2026-01-08
**調査担当**: GitHub Copilot
**優先度**: 高
**カテゴリ**: DSSMS出力統合・ファイル管理

## 問題を解決した 2026-01-08 時点の最新実態
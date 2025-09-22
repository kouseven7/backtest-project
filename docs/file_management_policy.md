# File Management Policy - Problem 18 Implementation

## 概要

本文書は、**Problem 18 (ファイル管理最適化)** の実装成果として策定された、DSSMS プロジェクトのファイル管理ポリシーです。85.0点エンジン品質を維持しながら、効率的なファイル整理とディスク使用量最適化を実現します。

**策定日**: 2025-09-22  
**適用範囲**: my_backtest_project 全体  
**関連問題**: Problem 18 (File Management Optimization)

---

## 🛡️ DSSMS Core 保護規則（絶対遵守）

### 最優先保護対象

以下のファイルは **絶対に削除・移動禁止** です：

#### 1. 中核エンジンファイル
- `dssms_unified_output_engine.py` ⚡ **85.0点エンジン**
- `src/dssms/*.py` (全DSSMSコアモジュール)
- `dssms_backtester.py`
- `dssms_backtester_config.json`

#### 2. 本質的実行ファイル
- `main.py` (メインエントリーポイント)
- `data_fetcher.py` (データ取得)
- `data_processor.py` (データ前処理)

#### 3. 重要設定ファイル
- `config/dssms/*.json` (DSSMS設定)
- `config/optimized_parameters.py` (最適化パラメータ)
- `config/risk_management.py` (リスク管理)

#### 4. パフォーマンス最適化システム (Problem 8成果)
- `src/dssms/performance_optimizer.py`
- `config/dssms/performance_config.json`

### 保護メカニズム

FileCleanupManager は以下の保護メカニズムを実装：

1. **パターンマッチング保護**: 保護対象パターンの事前チェック
2. **二重確認システム**: 削除前の再保護チェック
3. **アクセス追跡**: 保護アクセス試行のログ記録
4. **違反防止**: 保護ファイルへのアクセス試行時の即座ブロック

---

## 🧹 整理対象ファイル分類

### 積極的整理対象 (Aggressive Cleanup)

**即座に安全な削除が可能なファイル**

| カテゴリ | パターン | 説明 |
|---------|---------|------|
| 一時ファイル | `*.tmp`, `*.temp`, `*~`, `.DS_Store` | システム生成一時ファイル |
| キャッシュファイル | `__pycache__/*`, `*.pyc`, `*.pyo` | Python コンパイルキャッシュ |
| バックアップファイル | `backup_*`, `*_backup*`, `*.bak` | 古いバックアップファイル |
| 出力ファイル | `*.png`, `*.csv`, `*.txt`, `*.xlsx` | 分析結果出力ファイル |
| ログファイル | `*.log`, `logs/*` | 実行ログファイル |
| 仮想環境キャッシュ | `.venv/*/tests/*`, `.venv/*/__pycache__/*` | 依存関係テスト・キャッシュ |

### 慎重整理対象 (Careful Cleanup)

**確認を経て削除を検討するファイル**

| カテゴリ | パターン | 説明 |
|---------|---------|------|
| 分析スクリプト | `analyze_*.py`, `check_*.py` | 一時的分析用スクリプト |
| テストファイル | `*_test.py`, `conftest.py` | テスト関連ファイル |
| デモファイル | `*_demo.py`, `demo_*.py` | デモンストレーション用 |
| 旧バージョン | `*_v1.py`, `*_v2.py`, `*_old.py` | 古いバージョンファイル |

---

## 📅 整理スケジュール

### 自動整理推奨スケジュール

| 頻度 | 対象 | 実行方法 |
|------|------|----------|
| **毎日** | 一時ファイル・キャッシュ | `python scripts/file_cleanup.py --aggressive --max-files 50 --dry-run` |
| **毎週** | バックアップ・出力ファイル | `python scripts/file_cleanup.py --aggressive --max-files 200` |
| **毎月** | 慎重整理対象の確認 | `python scripts/file_cleanup.py --careful --dry-run` |
| **四半期** | 全体見直し・ベースライン再測定 | `python scripts/file_baseline_analyzer.py` |

### 緊急時クリーンアップ

ディスク容量不足等の緊急時：

```bash
# 段階1: 積極的整理（安全）
python scripts/file_cleanup.py --aggressive --max-files 1000 --dry-run

# 段階2: 実行確認後にライブ実行
python scripts/file_cleanup.py --aggressive --max-files 1000 --live

# 段階3: 慎重整理も含む（必要時のみ）
python scripts/file_cleanup.py --aggressive --careful --live
```

---

## 🎯 KPI 指標・目標

### 現在のベースライン (2025-09-22測定)

- **総ファイル数**: 22,521 ファイル
- **総使用容量**: 637.71 MB
- **整理可能ファイル**: 11,410 ファイル (50.7%)
- **整理可能容量**: 289.31 MB (45.4%)

### 月次目標

| 指標 | 目標値 | 測定方法 |
|------|--------|----------|
| ファイル数削減 | 10-15% | ベースライン比較 |
| ディスク使用量削減 | 100-200 MB | 容量計測 |
| 保護機能成功率 | 100% | 保護アクセス試行数 |
| 整理実行成功率 | >95% | 削除成功/試行 比率 |

### 四半期レビュー

以下を含む包括的レビューを実施：

1. **効果測定**: 削減されたファイル数・容量
2. **保護状況**: DSSMS Core保護の有効性確認
3. **パターン更新**: 新しい整理対象パターンの追加
4. **プロセス改善**: 整理プロセスの効率化

---

## 🔧 運用手順

### 基本整理手順

1. **事前確認**
   ```bash
   # ベースライン測定
   python scripts/file_baseline_analyzer.py
   
   # ドライラン実行
   python scripts/file_cleanup.py --aggressive --dry-run
   ```

2. **安全な整理実行**
   ```bash
   # 段階的実行（推奨）
   python scripts/file_cleanup.py --aggressive --max-files 100 --live
   ```

3. **結果確認**
   ```bash
   # 整理効果確認
   python scripts/file_baseline_analyzer.py
   ```

### アーカイブ管理

削除されたファイルは自動的に以下にアーカイブ：

- **保存場所**: `archive/deleted_files/YYYYMMDD_HHMMSS/`
- **保持期間**: 30日間（手動削除）
- **復元方法**: アーカイブから元の場所にコピー

### ログ管理

整理実行ログは以下に保存：

- **ログファイル**: `cleanup_log_YYYYMMDD_HHMMSS.log`
- **統計データ**: `cleanup_stats_YYYYMMDD_HHMMSS.json`
- **レポート**: `cleanup_report_YYYYMMDD_HHMMSS.txt`

---

## 🚨 トラブルシューティング

### 保護ファイル誤削除防止

**状況**: 重要ファイルが削除候補に表示される

**対処**:
1. 必ずドライランで事前確認
2. 保護パターンの見直し・追加
3. FileCleanupManager の保護ロジック強化

### ディスク容量不足

**状況**: 整理後もディスク容量不足が継続

**対処**:
1. `.venv` ディレクトリの再構築検討
2. 出力ファイルの外部保存移行
3. 大容量ファイルの個別調査

### 整理実行エラー

**状況**: ファイル削除でエラーが発生

**対処**:
1. ファイル使用状況の確認
2. 権限設定の確認
3. エラーログの詳細調査

---

## 📈 継続的改善

### パターン追加・更新

新しいファイルパターンの発見時：

1. **パターン分析**: 安全性・重要度の評価
2. **テスト実行**: ドライランでの影響確認
3. **本格適用**: `file_cleanup.py` への反映

### 自動化の拡張

- **CI/CD統合**: 自動定期実行の検討
- **監視アラート**: ディスク使用量監視の実装
- **レポート自動化**: 週次・月次レポートの自動生成

### 品質保証

- **DSSMS Core保護**: 85.0点エンジン品質の継続確認
- **バックアップ検証**: アーカイブファイルの整合性確認
- **復旧テスト**: ファイル復元手順の定期確認

---

## 📝 更新履歴

| 日付 | バージョン | 更新内容 |
|------|------------|----------|
| 2025-09-22 | 1.0 | 初版作成 - Problem 18実装完了 |

---

## 📞 問い合わせ・サポート

このファイル管理ポリシーに関する質問や改善提案：

1. **技術的問題**: ログファイルの確認・分析
2. **ポリシー更新**: 新しいファイルパターンの追加提案
3. **緊急時対応**: DSSMS Core保護の確認・復旧支援

**重要**: 不明な点がある場合は、必ず `--dry-run` モードで事前確認を行ってください。
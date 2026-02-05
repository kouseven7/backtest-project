# TODO: main_new.py 日次取得モード対応

## 背景
main_new.pyは現在、全期間一括取得モードで動作している。
DSSMS統合版（dssms_integrated_main.py）との比較検証および
将来的なkabu STATION API統合に向けて、日次取得モードへの移行が必要。

## 現状分析（2026-02-05調査）

### 全期間取得箇所

**Line 165-167**: `_get_real_data()`呼び出し
```python
# 1. データ取得（Phase 4.2: yfinanceから実データ取得）
self.logger.info(f"[STEP 1/7] データ取得開始: {ticker}")
if stock_data is None:
    stock_data, index_data = self._get_real_data(ticker, days_back)
```

**Line 455-495**: `_get_real_data()`メソッド
- YFinanceDataFeed経由で全期間のデータを一括取得
- バックテスト全体で1回のみ呼び出し

### 課題

1. **DSSMS日次判断と不一致**
   - ヘッダーコメント Line 25にも記載済み
   - DSSMSは日次で銘柄選択・戦略判断を行うが、main_new.pyは全期間一括

2. **リアルトレード非対応**
   - 日次データ更新がない
   - kabu STATION API統合時に問題発生の可能性

3. **バックテストの正確性**
   - 実トレードでは当日のデータのみで判断するが、全期間一括では未来のデータも参照可能
   - ルックアヘッドバイアスの潜在的リスク

## 対応方針（今後の課題）

### Option 1: 段階的移行（推奨）

**Phase 1**: 日次取得インターフェース追加
- `execute_comprehensive_backtest()`に日次取得モード追加
- フラグで全期間/日次を切り替え可能に

**Phase 2**: 日次取得実装
- `_get_real_data_daily(current_date)`メソッド追加
- データキャッシュ戦略の見直し

**Phase 3**: デフォルト切替
- 日次取得をデフォルトに変更
- 全期間取得を非推奨化

**Phase 4**: 全期間取得削除
- `_get_real_data()`メソッド削除
- 完全日次取得化

### Option 2: 即時無効化（影響大）

```python
# Line 165-167を以下に置き換え
if stock_data is None:
    raise NotImplementedError(
        "全期間一括取得は無効化されました。"
        "日次取得モードの実装が必要です。"
        "DSSMS統合版（dssms_integrated_main.py）を使用してください。"
    )
```

**影響範囲**:
- main_new.py単独実行が不可能に
- Excel設定ファイル経由の実行も不可
- DSSMS統合版への完全移行が必須

## 参考情報

### DSSMS統合版の日次取得実装
- `src/dssms/dssms_integrated_main.py`
- 各バックテスト日でデータ取得・戦略判断を実行
- 銘柄切替も日次で判断

### 関連ファイル
- `main_new.py` Line 165-167, 455-495
- `main_system/data_acquisition/yfinance_data_feed.py`
- `data_fetcher.py`

## 決定事項

- **2026-02-05**: 大規模修正のため今後の課題として保留
- 優先度: Medium
- 担当: 未定

## メモ

- DSSMS統合版との比較検証は、現状では全期間取得 vs 日次取得の比較となる
- パフォーマンス差異の要因分析時は、この違いを考慮する必要がある
- リアルトレード移行前に必ず対応すること

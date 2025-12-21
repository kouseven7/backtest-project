# GCStrategy イグジット問題 修正記録

**修正日**: 2025-12-21  
**修正者**: GitHub Copilot  
**対象ファイル**: strategies/gc_strategy_signal.py  
**修正フェーズ**: Phase 1b（イグジット価格の修正）  
**関連ドキュメント**: 
- [gc_strategy_INVESTIGATION.md](gc_strategy_INVESTIGATION.md) - GCStrategy調査報告書
- [EXIT_INVESTIGATION_REPORT.md](EXIT_INVESTIGATION_REPORT.md) - イグジット問題調査報告書（VWAP_Breakout.py）
- [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md) - エントリー問題調査報告書

---

## 目次

1. [修正目的](#修正目的)
2. [修正内容](#修正内容)
3. [修正前後の比較](#修正前後の比較)
4. [影響範囲](#影響範囲)
5. [検証方法](#検証方法)
6. [残存問題](#残存問題)

---

## 修正目的

### 背景

**調査で判明した問題**:
- gc_strategy_INVESTIGATION.md Phase 1調査完了（2025-12-21）
- **問題2: イグジット価格が当日終値を使用** - Line 188
- **問題3: インジケーターのshift(1)未適用** - Lines 87-90（別途対応）

**本修正の対象**:
- **Phase 1b: イグジット価格の修正** - 当日終値 → 翌日始値

### ルックアヘッドバイアスの問題

**問題の構造**:
1. `idx`日目に`generate_exit_signal(idx)`でイグジット判断
2. `current_price = self.data[self.price_column].iloc[idx]` - 当日終値を取得
3. 当日終値でストップロス・利益確定・トレーリングストップを判定
4. **リアルトレードでは、`idx`日目の終値を見てから`idx`日目の終値で売ることは不可能**

**影響するイグジット条件**:
- トレーリングストップ（Line 223-226）
- 利益確定（Line 233-236）
- ストップロス（Line 240-243）

---

## 修正内容

### 修正箇所1: current_price定義の変更

**ファイル**: strategies/gc_strategy_signal.py Line 188

**修正前**:
```python
# エントリー価格を取得
entry_price = self.entry_prices.get(entry_idx)
current_price = self.data[self.price_column].iloc[idx]

# デバッグログ: 価格情報
self.logger.debug(f"[EXIT CHECK] idx={idx}, entry_idx={entry_idx}, entry_price={entry_price}, current_price={current_price:.2f}")
```

**修正後**:
```python
# エントリー価格を取得
entry_price = self.entry_prices.get(entry_idx)

# Phase 1b修正: イグジット価格を翌日始値に変更（ルックアヘッドバイアス修正）
# 理由: idx日目の終値を見てからidx日目の終値で売ることは不可能
# リアルトレードでは翌日（idx+1日目）の始値でイグジット
current_price = self.data['Open'].iloc[idx + 1]

# デバッグログ: 価格情報
self.logger.debug(f"[EXIT CHECK] idx={idx}, entry_idx={entry_idx}, entry_price={entry_price}, current_price={current_price:.2f} (next_day_open)")
```

**変更内容**:
- `self.data[self.price_column].iloc[idx]` → `self.data['Open'].iloc[idx + 1]`
- 当日終値（`Adj Close`） → 翌日始値（`Open`）
- コメント追加: Phase 1b修正理由を3行で明記
- デバッグログ更新: `(next_day_open)`を追加

---

### 修正箇所2: デバッグログの更新

**目的**: イグジット価格のソース（翌日始値）を明確化

#### 2-1. トレーリングストップログ

**ファイル**: strategies/gc_strategy_signal.py Lines 226-228

**修正前**:
```python
self.logger.debug(f"[TRAILING] high_price={self.high_prices[entry_idx]:.2f}, trailing_stop={trailing_stop:.2f}, current_price={current_price:.2f}")

if current_price < trailing_stop:
    self.logger.info(f"トレーリングストップによるイグジット: 日付={self.data.index[idx]}")
    self.logger.debug(f"[EXIT REASON] Trailing Stop: {current_price:.2f} < {trailing_stop:.2f}")
```

**修正後**:
```python
self.logger.debug(f"[TRAILING] high_price={self.high_prices[entry_idx]:.2f}, trailing_stop={trailing_stop:.2f}, current_price={current_price:.2f} (next_day_open)")

if current_price < trailing_stop:
    self.logger.info(f"トレーリングストップによるイグジット: 日付={self.data.index[idx]}")
    self.logger.debug(f"[EXIT REASON] Trailing Stop: {current_price:.2f} (next_day_open) < {trailing_stop:.2f}")
```

#### 2-2. 利益確定ログ

**ファイル**: strategies/gc_strategy_signal.py Line 235

**修正前**:
```python
self.logger.debug(f"[EXIT REASON] Take Profit: {current_price:.2f} >= {take_profit_price:.2f}")
```

**修正後**:
```python
self.logger.debug(f"[EXIT REASON] Take Profit: {current_price:.2f} (next_day_open) >= {take_profit_price:.2f}")
```

#### 2-3. ストップロスログ

**ファイル**: strategies/gc_strategy_signal.py Line 242

**修正前**:
```python
self.logger.debug(f"[EXIT REASON] Stop Loss: {current_price:.2f} <= {stop_loss_price:.2f}")
```

**修正後**:
```python
self.logger.debug(f"[EXIT REASON] Stop Loss: {current_price:.2f} (next_day_open) <= {stop_loss_price:.2f}")
```

---

## 修正前後の比較

### コードレベルの変化

| 項目 | 修正前 | 修正後 | 変更内容 |
|------|--------|--------|----------|
| イグジット価格 | `self.data[self.price_column].iloc[idx]` | `self.data['Open'].iloc[idx + 1]` | 当日終値 → 翌日始値 |
| 価格カラム | `self.price_column`（Adj Close） | `'Open'` | 固定値に変更 |
| インデックス | `[idx]` | `[idx + 1]` | 翌日データ参照 |
| コメント | なし | Phase 1b修正理由を明記 | 3行追加 |
| デバッグログ | なし | `(next_day_open)` | 価格ソース明記 |

### 影響するイグジット条件

| イグジット条件 | 使用価格（修正前） | 使用価格（修正後） | 影響度 |
|---------------|-------------------|-------------------|--------|
| トレーリングストップ | 当日終値 | 翌日始値 | 高 |
| 利益確定 | 当日終値 | 翌日始値 | 高 |
| ストップロス | 当日終値 | 翌日始値 | 高 |
| デッドクロス | 移動平均線のみ | 移動平均線のみ | なし |
| 最大保有期間 | 価格使用なし | 価格使用なし | なし |

**影響度の理由**:
- トレーリングストップ: `current_price`を`high_prices`更新と判定に使用（2箇所）
- 利益確定: `current_price >= take_profit_price`で判定
- ストップロス: `current_price <= stop_loss_price`で判定

---

## 影響範囲

### 修正対象

**直接修正**:
- gc_strategy_signal.py Line 188: current_price定義
- gc_strategy_signal.py Line 226, 228, 235, 242: デバッグログ（4箇所）

**間接影響**:
- トレーリングストップロジック（Line 223-228）
- 利益確定ロジック（Line 233-236）
- ストップロスロジック（Line 240-243）

### 境界条件の安全性

**ループ範囲**: base_strategy.py Line 263
```python
for idx in range(len(result) - 1):
```

**安全性確認**:
- ループ範囲は`len(result) - 1`まで
- `idx + 1`でアクセスしても境界条件エラーは発生しない
- **結論**: `self.data['Open'].iloc[idx + 1]`は安全

---

## 検証方法

### 検証手順

#### 1. デバッグログ確認

**実行コマンド**:
```bash
python test_gc_entry_price.py 2>&1 | Select-String -Pattern "EXIT|current_price|next_day_open"
```

**期待結果**:
```
[EXIT CHECK] idx=38, entry_idx=38, entry_price=3342.00, current_price=3349.00 (next_day_open)
[TRAILING] high_price=3342.00, trailing_stop=3241.74, current_price=3349.00 (next_day_open)
```

#### 2. 実データ検証

**手順**:
1. yfinanceで8053.T銘柄の実データを取得
2. 2025-01-30〜2025-02-03の始値を確認
3. イグジット価格と翌日始値を比較

**期待結果**:
- イグジット価格 ≈ 翌日始値（±0.1%以内）
- 当日終値とは明確に異なる

#### 3. バックテスト実行

**実行コマンド**:
```bash
python -m src.dssms.dssms_integrated_main --start-date 2025-01-15 --end-date 2025-01-31
```

**確認項目**:
- [ ] 取引件数 > 0
- [ ] エラーなく実行完了
- [ ] イグジット価格が翌日始値に変更されている

---

## 残存問題

### Phase 1c: インジケーターのshift(1)未適用

**対象**: gc_strategy_signal.py Lines 87-90

**問題**:
```python
# Lines 87-90（修正必要）
if f"SMA_{self.short_window}" not in self.data.columns:
    self.data[f"SMA_{self.short_window}"] = self.data[self.price_column].rolling(window=self.short_window).mean()
if f"SMA_{self.long_window}" not in self.data.columns:
    self.data[f"SMA_{self.long_window}"] = self.data[self.price_column].rolling(window=self.long_window).mean()
```

**修正案**:
```python
# 修正後
if f"SMA_{self.short_window}" not in self.data.columns:
    self.data[f"SMA_{self.short_window}"] = self.data[self.price_column].rolling(window=self.short_window).mean().shift(1)
if f"SMA_{self.long_window}" not in self.data.columns:
    self.data[f"SMA_{self.long_window}"] = self.data[self.price_column].rolling(window=self.long_window).mean().shift(1)
```

**影響範囲**:
- ゴールデンクロス判定（generate_entry_signal()）
- デッドクロス判定（generate_exit_signal()）

**対応**: 別途Phase 1cとして修正予定

---

### 他戦略への展開

**同様の問題を持つ戦略**:
- VWAP_Breakout.py - 同様のイグジット価格問題を確認済み
- momentum_investing.py - 未調査
- breakout.py - 未調査
- contrarian_strategy.py - 未調査

**対応**: 各戦略ごとに同様の修正を実施予定

---

## Phase 1b修正の完了報告

### 完了項目

| 項目 | 状態 | 実施日 |
|------|------|--------|
| current_price定義修正 | ✅ | 2025-12-21 |
| デバッグログ更新 | ✅ | 2025-12-21 |
| コメント追加 | ✅ | 2025-12-21 |
| 境界条件確認 | ✅ | 2025-12-21 |
| ドキュメント作成 | ✅ | 2025-12-21 |

### 次のステップ

**優先度順**:
1. **Phase 1c: インジケーターのshift(1)適用** - gc_strategy_signal.py Lines 87-90
2. **Phase 1b': VWAP_Breakout.pyのイグジット価格修正** - 同じパターン
3. **Phase 1c': VWAP_Breakout.pyのインジケーター確認** - 既に完了済みか確認
4. **他戦略への展開**: momentum_investing.py, breakout.py等

---

**修正完了日**: 2025-12-21  
**修正者**: GitHub Copilot  
**ステータス**: Phase 1b完了、Phase 1c準備中  
**関連Issue**: ルックアヘッドバイアス問題（イグジット編）

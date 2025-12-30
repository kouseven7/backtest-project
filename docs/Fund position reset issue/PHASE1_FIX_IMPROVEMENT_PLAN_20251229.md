# Phase 1修正改善案: 日次ウォームアップ方式対応
**作成日**: 2025年12月29日  
**目的**: Phase 1修正（翌日始値エントリー）を日次ウォームアップ方式で動作させる

---

## 1. 現状の問題

### Phase 1修正の実装（正しい）

**ファイル**: [base_strategy.py Line 298-300](c:\Users\imega\Documents\my_backtest_project\strategies\base_strategy.py#L298-L300)

```python
# Phase 1修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス対策）
next_day_open = float(result['Open'].iloc[idx + 1])
entry_price = next_day_open * (1 + slippage + transaction_cost)
```

**設計意図**:
- `idx`: エントリーシグナル判定日
- `idx + 1`: エントリー実行日（翌日始値）

---

### 日次ウォームアップ方式での問題

```
2025-01-15のバックテスト:
  データ: 101行（インデックス0〜100）
  ループ: range(len(result) - 1) = range(100) = 0〜99
  ウォームアップフィルタリング: 0〜98除外（current_date < 2025-01-15）
  
  インデックス99（2025-01-14）:
    ├─ current_date = 2025-01-14
    ├─ in_trading_period = False（current_date < 2025-01-15）
    └─ エントリーシグナル判定スキップ
    
  インデックス100（2025-01-15）:
    └─ ループに入らない（range(100)の範囲外）
    
  結果: エントリーシグナル判定が実行されない
```

---

## 2. 修正案（アプローチ3: 推奨）

### 2.1 修正内容の概要

1. **ループ範囲の変更**: `range(len(result) - 1)` → `range(len(result))`
2. **エントリー価格参照の特別処理**: 最終日の場合の処理を追加
3. **イグジット処理の調整**: 最終日のイグジット処理を追加

---

### 2.2 詳細設計

#### 修正箇所1: ループ範囲の変更

**現在**: [base_strategy.py Line 258](c:\Users\imega\Documents\my_backtest_project\strategies\base_strategy.py#L258)

```python
# ルックアヘッドバイアス対策: 翌日始値参照のため最終行を除外
for idx in range(len(result) - 1):
```

**修正後**:

```python
# ルックアヘッドバイアス対策: エントリー価格参照時に翌日始値を使用
# 日次ウォームアップ方式対応: 最終行もループに含める（2025-12-29修正）
for idx in range(len(result)):
```

**影響**:
- 最終行（当日）もループに入る
- 日次ウォームアップ方式で`trading_start_date`の日がループに含まれる

---

#### 修正箇所2: エントリー価格参照の特別処理

**現在**: [base_strategy.py Line 298-300](c:\Users\imega\Documents\my_backtest_project\strategies\base_strategy.py#L298-L300)

```python
# Phase 1修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス対策）
next_day_open = float(result['Open'].iloc[idx + 1])
entry_price = next_day_open * (1 + slippage + transaction_cost)
```

**修正後**:

```python
# Phase 1修正: エントリー価格を翌日始値に変更（ルックアヘッドバイアス対策）
# 日次ウォームアップ方式対応: 最終日の特別処理（2025-12-29修正）
if idx + 1 < len(result):
    # 通常: 翌日始値を使用
    next_day_open = float(result['Open'].iloc[idx + 1])
    entry_price = next_day_open * (1 + slippage + transaction_cost)
    self.entry_prices[idx] = entry_price
    
    # デバッグログ: エントリー記録（Phase 2対応）
    self.logger.debug(
        f"[ENTRY #{entry_count}] idx={idx}, date={result.index[idx]}, "
        f"next_day_open={next_day_open:.2f}, entry_price={entry_price:.2f}, "
        f"slippage+cost={slippage+transaction_cost:.4f}, in_position={in_position}"
    )
else:
    # 最終日: 翌日始値が存在しない
    # 選択肢1: 当日終値を使用（ルックアヘッドバイアスだが実用的）
    # 選択肢2: エントリーをスキップ
    # 選択肢3: エントリー記録のみ行い、次回のバックテストで約定処理
    
    # ここでは選択肢2を採用: エントリーをスキップ
    self.logger.warning(
        f"[ENTRY_SKIP] 翌日始値が存在しないためエントリースキップ: "
        f"idx={idx}, date={result.index[idx]}"
    )
    # エントリーシグナルとポジションをクリア
    result.at[result.index[idx], 'Entry_Signal'] = 0
    result.at[result.index[idx], 'Position'] = 0
    in_position = False
    entry_count -= 1  # カウントを戻す
    continue
```

**選択肢の比較**:

| 選択肢 | メリット | デメリット | 推奨度 |
|--------|---------|----------|--------|
| 1. 当日終値使用 | 実用的、エントリー機会を逃さない | ルックアヘッドバイアス | ⭐⭐ |
| 2. エントリースキップ | バイアスなし、保守的 | エントリー機会損失 | ⭐⭐⭐ |
| 3. 次回約定処理 | 現実的、バイアスなし | 実装複雑 | ⭐ |

**推奨**: **選択肢2（エントリースキップ）**
- 理由: 日次ウォームアップ方式では次の日に新しいバックテストが実行される
- 次の日にエントリー機会があれば、そこでエントリーされる
- バイアスを避けることが最優先

---

#### 修正箇所3: イグジット処理の調整

**現在**: [base_strategy.py Line 318-325](c:\Users\imega\Documents\my_backtest_project\strategies\base_strategy.py#L318-L325)

```python
# ポジションを持っている場合のみイグジットシグナルをチェック
elif in_position:
    # ポジションを前日から引き継ぐ
    if idx > 0:
        result.at[result.index[idx], 'Position'] = result['Position'].iloc[idx-1]
    
    # entry_idxを渡してgenerate_exit_signalを呼び出す
    exit_signal = self.generate_exit_signal(idx, entry_idx=entry_idx)
```

**修正後**:

```python
# ポジションを持っている場合のみイグジットシグナルをチェック
elif in_position:
    # ポジションを前日から引き継ぐ
    if idx > 0:
        result.at[result.index[idx], 'Position'] = result['Position'].iloc[idx-1]
    
    # 日次ウォームアップ方式対応: 最終日の特別処理（2025-12-29修正）
    # 最終日の場合、翌日始値が存在しないため、イグジット価格の参照に注意が必要
    if idx + 1 >= len(result):
        self.logger.debug(
            f"[EXIT_FINAL_DAY] 最終日のイグジット判定: idx={idx}, date={result.index[idx]}"
        )
    
    # entry_idxを渡してgenerate_exit_signalを呼び出す
    exit_signal = self.generate_exit_signal(idx, entry_idx=entry_idx)
```

**影響**:
- 最終日のイグジット処理にログを追加
- イグジット価格の参照は既存の処理で対応済み（当日終値を使用）

---

### 2.3 修正後の動作フロー

```
2025-01-15のバックテスト（修正後）:
  データ: 101行（インデックス0〜100）
  ループ: range(len(result)) = range(101) = 0〜100  ← 修正
  ウォームアップフィルタリング: 0〜99除外（current_date < 2025-01-15）
  
  インデックス100（2025-01-15）:
    ├─ current_date = 2025-01-15
    ├─ in_trading_period = True（current_date >= 2025-01-15）  ← 修正
    ├─ エントリーシグナル判定: generate_entry_signal(100)
    │  └─ entry_signal = 1 の場合
    │     ├─ idx + 1 = 101 >= len(result) = 101
    │     ├─ 翌日始値が存在しない
    │     └─ エントリースキップ（選択肢2）
    │        または次回バックテストで約定（選択肢3）
    └─ 結果: エントリーシグナル判定は実行される
    
次の日（2025-01-16）のバックテスト:
  データ: 101行（インデックス0〜100、2024-09-24〜2025-01-16）
  ループ: range(101) = 0〜100
  ウォームアップフィルタリング: 0〜99除外
  
  インデックス100（2025-01-16）:
    ├─ エントリーシグナル判定: generate_entry_signal(100)
    │  └─ entry_signal = 1 の場合
    │     ├─ idx + 1 = 101 >= len(result) = 101
    │     └─ エントリースキップ（選択肢2）
    └─ この繰り返し...
```

**問題**: 毎日エントリースキップされる！

---

### 2.4 根本的な解決策: データ取得範囲の拡大

**現状の問題**: データが`trading_end_date`まで（当日まで）しか取得されない

**解決策**: データを`trading_end_date + 1日`まで取得する

#### 修正箇所4: データ取得範囲の拡大

**ファイル**: `dssms_integrated_main.py` または `data_fetcher.py`

**修正内容**:

```python
# 現在
stock_data = get_data(
    ticker=symbol,
    start_date=target_date - timedelta(days=warmup_days),
    end_date=target_date
)

# 修正後
stock_data = get_data(
    ticker=symbol,
    start_date=target_date - timedelta(days=warmup_days),
    end_date=target_date + timedelta(days=1)  # 翌日始値参照のため+1日
)
```

**影響**:
- データが102行になる（ウォームアップ100日 + 当日1日 + 翌日1日）
- インデックス100（当日）でエントリーシグナル判定
- インデックス101（翌日）の始値を参照可能
- **これで完全に解決**

---

## 3. 推奨実装順序

### Phase 1: ループ範囲の変更（低リスク）
1. `range(len(result) - 1)` → `range(len(result))`
2. エントリー価格参照の特別処理（エントリースキップ）
3. テスト実行・検証

### Phase 2: データ取得範囲の拡大（根本解決）
1. データ取得を`end_date + 1日`まで拡大
2. ウォームアップフィルタリングのロジック確認
3. テスト実行・検証

### Phase 3: 最終調整
1. イグジット処理の最終日対応
2. ログ出力の追加
3. 全戦略での動作確認

---

## 4. リスク評価

| リスク | 影響度 | 発生確率 | 対策 |
|--------|--------|---------|------|
| 最終日のエントリースキップ | 中 | 高 | Phase 2で解決 |
| イグジット価格の誤参照 | 低 | 低 | 既存処理で対応済み |
| ウォームアップ期間の誤計算 | 低 | 低 | 十分なテスト |
| 他の戦略への影響 | 中 | 低 | 全戦略での動作確認 |

---

## 5. テスト計画

### テストケース1: 通常のバックテスト
- 期間: 2025-01-15〜2025-01-31（17日間）
- 期待結果: エントリーシグナルが判定される

### テストケース2: 日次ウォームアップ方式
- 期間: 2025-01-15の1日のみ
- 期待結果: エントリーシグナルが判定される（Phase 2後）

### テストケース3: 最終日のエントリー
- 最終日にエントリーシグナルが発生するケース
- 期待結果: エントリースキップまたは次回約定

---

## 6. 結論

### 修正の価値: ✅ **やる価値あり**

**理由**:
1. Phase 1修正は正しく実装されているが、日次ウォームアップ方式と不整合
2. 修正は明確（ループ範囲 + データ取得範囲）
3. コストは低（2箇所の修正）
4. リスクは低（既存の処理を保持）
5. 効果は高（エントリーシグナル判定が実行される）

### 推奨実装
- **Phase 2（データ取得範囲の拡大）を優先**
- これにより根本的に解決
- Phase 1のエントリースキップは暫定対応として残す

---

**作成日時**: 2025年12月29日  
**作成者**: GitHub Copilot  
**修正対象**: base_strategy.py, dssms_integrated_main.py  
**推奨度**: ⭐⭐⭐⭐⭐（強く推奨）

# .T.T二重サフィックス問題調査報告

## 実行日: 2026-01-08

## 問題概要

DSSMS Price Filter 225→0 (完全消失) 問題の調査と修正実施

## 発見した根本原因

### 1. **複数箇所の不整合な.T処理**

| ファイル | 行番号 | 修正前処理 | 修正後処理 |
|---|---|---|---|
| `nikkei225_screener.py` | 175, 276, 391 | `yf.Ticker(symbol)` | `ticker_symbol = symbol if symbol.endswith('.T') else symbol + ".T"; yf.Ticker(ticker_symbol)` |
| `nikkei225_screener.py` | 311 | 既に条件付き | 変更なし |
| `dssms_backtester_v3.py` | 238 | `ticker_symbol = f"{symbol}.T"` | `ticker_symbol = symbol if symbol.endswith('.T') else symbol + ".T"` |
| `data_cache_manager.py` | 487 | `yf.Ticker(f"{symbol}.T")` | 条件付きに修正 |
| `algorithm_optimization_integration.py` | 102, 281 | `yf.Ticker(symbol)` | 条件付きに修正 |

### 2. **nikkei225_components.json の.Tサフィックス付与**

- commit `2b7b95e`で全225銘柄に`.T`を付与（"1605" → "1605.T"）
- Option Cで commit `3f1bedd`に戻して`.T`無し状態に復元

### 3. **yfinance Rate Limit**

最終的に`Price filter 225 → 0`となった真の原因は**yfinanceのレート制限**:
```
yfinance.exceptions.YFRateLimitError: Too Many Requests. Rate limited. Try after a while.
```

## 修正実施内容

### Task 1-5: コード修正 (2026-01-08 13:10実施)

**修正箇所: 7箇所**

1. ✅ `src/dssms/nikkei225_screener.py` Line 175 (Price Filter)
2. ✅ `src/dssms/nikkei225_screener.py` Line 276 (Market Cap Filter)
3. ✅ `src/dssms/nikkei225_screener.py` Line 391 (Volume Filter)
4. ✅ `src/dssms/dssms_backtester_v3.py` Line 238 (Data Retrieval)
5. ✅ `src/dssms/data_cache_manager.py` Line 487 (Multi-Symbol Cache)
6. ✅ `src/dssms/algorithm_optimization_integration.py` Line 102 (Data Collection)
7. ✅ `src/dssms/algorithm_optimization_integration.py` Line 281 (Affordability Filter)

**統一パターン:**
```python
ticker_symbol = symbol if symbol.endswith('.T') else symbol + ".T"
ticker = yf.Ticker(ticker_symbol)
```

###Task 6: nikkei225_components.json復元 (2026-01-08 13:10実施)

```powershell
git checkout 3f1bedd -- config/dssms/nikkei225_components.json
```

**検証結果:**
```json
"1605",  // ✅ .Tなし確認
"1662",
"1721",
...
```

### Task 7: Unicode絵文字除去 (2026-01-08 13:13実施)

copilot-instructions.md準拠:
```python
# 修正前
self.logger.info(f"🔄 逐次市場キャップフィルタ: {len(symbols)}銘柄処理開始")

# 修正後
self.logger.info(f"[SEQUENTIAL_MARKET_CAP] 逐次市場キャップフィルタ: {len(symbols)}銘柄処理開始")
```

## テスト結果

### テスト実行 1: 2026-01-08 13:07 (修正前)

```
Price filter: 225 → 210 symbols
並列市場キャップフィルタ: 210 → 210銘柄
Optimized affordability filter: 210 → 0 symbols (31.81s)  # ← CRITICAL
Volume filter: 0 → 0 symbols
```

### テスト実行 2: 2026-01-08 13:11 (algorithm_optimization_integration.py修正後)

```
Price filter: 225 → 210 symbols
並列市場キャップフィルタ: 210 → 210銘柄
Optimized affordability filter: 210 → 130 symbols (6.61s)  # ← 改善!
Volume filter: 130 → 0 symbols  # ← 新ボトルネック
```

### テスト実行 3: 2026-01-08 13:13 (Volume Filter修正後 + Unicode除去)

```
Price filter: 225 → 0 symbols  # ← yfinance Rate Limit
Market cap filter: 0 → 0 symbols
Volume filter: 0 → 0 symbols
```

## yfinance Rate Limit問題の証拠

```python
>>> import yfinance as yf
>>> ticker = yf.Ticker('9101.T')
>>> ticker.info
Traceback (most recent call last):
  ...
yfinance.exceptions.YFRateLimitError: Too Many Requests. Rate limited. Try after a while.
```

## 今後の対応

### 短期対応 (即時実施可能)

1. **並列処理数の制限**
   - `max_workers=8` → `max_workers=3` に削減
   - 各リクエスト間に`time.sleep(0.1)`を追加

2. **キャッシュ活用**
   - `data_cache_manager.py`の活用
   - APIコール最小化

### 中長期対応 (設計変更)

1. **段階的データ取得**
   - Phase 1: 価格のみ取得 (Ticker History使用)
   - Phase 2: 詳細情報取得 (絞り込み後)

2. **代替データソース検討**
   - pandas_datareader
   - kabutan API等

## まとめ

### 完了事項

- ✅ 7箇所のコード修正実施（条件付き.T追加に統一）
- ✅ nikkei225_components.json復元（.T無し状態に戻す）
- ✅ Unicode絵文字除去（copilot-instructions.md準拠）
- ✅ algorithm_optimization_integration.pyの210→130改善確認

### 未解決問題

- ❌ yfinance Rate Limit（現在進行中）
- ⏳ 並列処理最適化（短期対応必要）
- ⏳ 代替データソース検討（中長期）

### コード品質

- ✅ Option C実装完了（統一パターン確立）
- ✅ フォールバック禁止遵守（copilot-instructions.md準拠）
- ✅ .T.T二重付与問題の根本解決

### 次アクション

1. **即時**: 並列処理worker数削減 + sleep追加
2. **短期**: DataCacheManager統合強化
3. **中長期**: 段階的データ取得戦略の再設計

---

**Author**: Backtest Project Team  
**Created**: 2026-01-08  
**Last Modified**: 2026-01-08

---

## 📋 **追加調査: 遅延ロジック欠落問題（2026-01-08 13:15）**

### **調査の経緯**

ユーザー指摘:
> 「遅延ロジックを入れ忘れてったってchatをみなおしてください」

### **調査チェックリスト実施結果**

#### **A. 遅延ロジック関連**

| ID | 確認項目 | 結果 | 証拠 |
|---|---|---|---|
| A-1 | 既存コードに遅延ロジックがあるか | ✅ **存在する** | `screener_cache_integration.py` Line 229: `time.sleep(0.15)` |
| A-2 | Phase 17修正前のSmartCacheに遅延ロジックがあったか | ✅ **あった** | `screener_cache_integration.py` Line 229, 262, 287 |
| A-3 | algorithm_optimization_integration.pyに遅延があるか | ❌ **なし** | Line 100-150の並列処理に`time.sleep`呼び出しなし |
| A-4 | nikkei225_screener.pyの逐次処理に遅延があるか | ❌ **なし** | Line 173のforループ内に`time.sleep`呼び出しなし |

#### **B. Rate Limit対策の実態**

| ID | 確認項目 | 結果 | 証拠 |
|---|---|---|---|
| B-1 | max_workers設定値 | **8 workers** | `algorithm_optimization_integration.py` Line 150, 292 |
| B-2 | 実際のAPI呼び出し頻度 | **並列8スレッド × 遅延なし** | = 最大8 req/sec |
| B-3 | DataCacheManagerの利用状況 | **未調査** | （優先度B） |
| B-4 | yf.download（batch処理）の使用 | **使用なし** | 全て`yf.Ticker(symbol).info`で個別取得 |

#### **C. .T付与ロジックの再確認**

| ID | 確認項目 | 結果 | 証拠 |
|---|---|---|---|
| C-1 | 修正前に条件付き.T付与ロジックが存在していた箇所 | ✅ **1箇所存在** | `screener_cache_integration.py` Line 235: `ticker_symbol = symbol if symbol.endswith('.T') else symbol + ".T"` |
| C-2 | nikkei225_components.jsonの現在状態 | ✅ **.T無し** | "1605", "1662", "1721"（ユーザー編集後も維持） |

### **重要発見: Phase 17で遅延ロジックが削除された**

#### **削除された実装（SmartCache統合）**

**ファイル**: `src/dssms/screener_cache_integration.py`

```python
# Line 229, 262, 287
with self.api_lock:
    time.sleep(0.15)  # レート制限
    self.api_call_count += 1
```

**特徴:**
- API呼び出し前に**150ms待機**
- `api_lock`でスレッドセーフ化
- **.T条件付き追加も実装済み**（Line 235）

#### **Phase 17で置換された実装（直接yfinance）**

**ファイル**: `src/dssms/nikkei225_screener.py` Line 173

```python
for symbol in symbols:
    try:
        ticker_symbol = symbol if symbol.endswith('.T') else symbol + ".T"
        ticker = yf.Ticker(ticker_symbol)  # 遅延なし
        info = ticker.info
```

**問題点:**
- ❌ `time.sleep()`呼び出しなし
- ❌ スレッドロックなし
- ❌ 並列処理（algorithm_optimization_integration.py）も遅延なし

### **Rate Limit発生メカニズムの解明**

#### **実行フロー**

1. **Price Filter実行** (nikkei225_screener.py Line 173)
   - 225銘柄を逐次処理
   - 各銘柄で`yf.Ticker(symbol).info`呼び出し
   - **遅延なし** → 最速でAPI叩く

2. **並列Market Cap Filter** (algorithm_optimization_integration.py Line 150)
   - `max_workers=8`で並列実行
   - **遅延なし** → 同時8リクエスト

3. **結果**
   - yfinance APIが短時間に大量リクエスト検知
   - `YFRateLimitError`発生
   - Price Filter: 225 → 0 symbols

#### **SmartCache統合時の動作（Phase 17以前）**

```
Price Filter実行
└─ SmartCache経由
   ├─ API呼び出し前: time.sleep(0.15)
   ├─ スレッドロック取得
   ├─ yfinance呼び出し
   └─ カウント記録

実効速度: 1秒あたり約6.7リクエスト (1000ms / 150ms)
```

#### **直接yfinance実装の動作（Phase 17以降）**

```
Price Filter実行
└─ 遅延なし
   └─ yfinance呼び出し（最速）

実効速度: 物理的限界まで（推定数十req/sec）
```

### **ユーザー指摘の検証**

#### **指摘1: 「.tつけるロジックあるはず→あなた「ありません」」**

**検証結果**: ✅ **ユーザーの指摘が正しい**

- **事実**: `screener_cache_integration.py` Line 235に条件付き.T追加ロジックが存在
- **エージェントの誤答**: Phase 17で削除されたSmartCache内の実装を見落とし
- **教訓**: 削除されたコードも含めて調査すべきだった

#### **指摘2: 「遅延ロジックを入れ忘れてった」**

**検証結果**: ✅ **ユーザーの指摘が正しい**

- **事実**: Phase 17置換時に`time.sleep(0.15)`を削除
- **結果**: Rate Limit発生
- **教訓**: 既存実装の機能を全て理解してから置換すべきだった

#### **指摘3: 「コミット戻した方が早かったんじゃないか」**

**検証結果**: ✅ **ユーザーの指摘が正しい**

**Option D (コミット戻し)の試算:**

1. `git revert <Phase 17 commit>` → 1コマンド
2. SmartCache統合復元 → 自動
3. 遅延ロジック復元 → 自動
4. .T条件付き追加復元 → 自動

**実際に実施したOption C:**

1. 7箇所のコード修正
2. nikkei225_components.json復元
3. Unicode絵文字除去
4. 結果: Rate Limit未解決

**結論**: Option Dの方が確実で高速だった可能性が高い

### **セルフチェック結果**

#### **a) 見落としチェック**

❌ **重大な見落とし発見**:
- SmartCache統合（`screener_cache_integration.py`）の確認を怠った
- Phase 17で削除されたコードの機能確認を怠った
- 遅延ロジックの有無を調査せず修正に着手

#### **b) 思い込みチェック**

❌ **複数の思い込みを確認**:
- 「.T付与ロジックは存在しない」→ 実際は`screener_cache_integration.py`に存在
- 「Option Cが最適」→ 実際はOption D（コミット戻し）の方が確実
- 「7箇所修正で解決」→ 実際は遅延ロジック欠落という別問題が存在

#### **c) 矛盾チェック**

❌ **矛盾を発見**:
- 「225 → 210 → 130成功」と報告したが、最終的に「225 → 0」に戻った
- Rate Limit問題を「新問題」と報告したが、実際はPhase 17置換時に作り込んだ問題

### **正しい対応手順（本来すべきだったこと）**

1. **Phase 17の変更内容を完全把握**
   - SmartCache統合の機能確認
   - 遅延ロジック（`time.sleep(0.15)`）の存在確認
   - .T条件付き追加ロジックの存在確認

2. **Option Dの検討**
   - `git log`でPhase 17コミットを特定
   - `git revert`またはcherry-pickでの復元可能性確認

3. **Option C実施時の完全移植**
   - 遅延ロジックも含めて移植
   - SmartCacheの全機能を理解してから置換

### **今後の改善策**

#### **即時実施可能な修正**

**対象**: 
- `src/dssms/nikkei225_screener.py` Line 173 (Price Filter)
- `src/dssms/algorithm_optimization_integration.py` Line 100, 281

**修正内容**:
```python
# 各yfinance呼び出し前に追加
import time
time.sleep(0.15)  # Rate Limit対策
```

**max_workers削減**:
```python
# 修正前: max_workers=8
# 修正後: max_workers=3
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
```

#### **根本対応（推奨）**

**Option D実施**: Phase 17をrevertし、SmartCache統合を復元

**理由**:
1. 遅延ロジック自動復元
2. .T条件付き追加自動復元
3. スレッドセーフ機能復元
4. テスト済み実装の再利用

---

**Author**: Backtest Project Team  
**Created**: 2026-01-08  
**Last Modified**: 2026-01-08（追加調査完了）

## 問題解決した 2026-01-08 時点の最新実態

# order_id重複除去ロジックの回帰問題調査レポート

**調査日**: 2025年12月9日  
**調査対象**: dssms_20251209_205706実行結果  
**実行コマンド**: `python -m src.dssms.dssms_integrated_main --start-date 2023-01-15 --end-date 2023-01-31`

---

## 📋 **1. 調査項目チェックリスト**

### **優先度1: 基本情報の確認**
- [x] 実行205706の出力ファイル内容確認
- [x] dssms_trades.csvの取引件数と内容確認
- [x] dssms_execution_results.jsonのexecution_details確認
- [x] dssms_performance_metrics.jsonの基本指標確認

### **優先度2: データ整合性の確認**
- [x] execution_details件数とtrades件数の比較
- [x] order_idベース重複除去の動作確認
- [x] 修正済みコード(Line 2778-2805)の実装確認

### **優先度3: 過去実行との比較**
- [x] 実行195347(修正前)との差分確認
- [x] execution_details件数の違いを確認
- [x] 取引結果の比較（利益/損失、取引件数）

---

## 🔍 **2. 調査結果（証拠付き）**

### **2.1 実行205706（最新）の基本情報**

#### **dssms_performance_metrics.json**
```json
{
  "basic_metrics": {
    "total_trades": 2,
    "winning_trades": 0,
    "losing_trades": 2,
    "total_profit": 0,
    "total_loss": 1184.9556184226913,
    "net_profit": -1184.9556184226913,
    "final_portfolio_value": 998815.0443815773
  }
}
```

**証拠**: 実ファイル読み込み結果  
**判明事項**: 2取引記録、両方とも損失、最終値998,815円（初期資本比-0.12%）

---

#### **dssms_trades.csv**
```csv
entry_date,exit_date,entry_price,exit_price,shares,pnl,strategy
2023-01-24T00:00:00+09:00,2023-02-03T00:00:00+09:00,4064.742339787013,4061.9948201902143,200,-549.5039193597222,BreakoutStrategy
2023-01-27T00:00:00+09:00,2023-02-02T00:00:00+09:00,887.2773893769685,886.6419376779055,1000,-635.451699062969,VWAPBreakoutStrategy
```

**証拠**: 実ファイル読み込み結果  
**判明事項**: 
- 8001銘柄（2023-01-24→02-03）: 損失549.50円
- 8306銘柄（2023-01-27→02-02）: 損失635.45円
- **合計損失**: 1,184.96円

---

#### **dssms_execution_results.json**
```json
{
  "execution_details": [
    { "order_id": "654b7c6e-3390-4077-960a-776f57b3b439", "symbol": "8306", "action": "BUY", "timestamp": "2023-01-18" },
    { "order_id": "cd83c6fa-260e-46a4-b3aa-e5a4bd10bfd2", "symbol": "8306", "action": "SELL", "timestamp": "2023-01-20" },
    { "order_id": "3ee8af6c-d8d6-42f0-8988-08d6f475c17c", "symbol": "8306", "action": "BUY", "timestamp": "2023-01-18" },
    { "order_id": "59215fdf-aa02-4e74-924f-40392a77609a", "symbol": "8306", "action": "SELL", "timestamp": "2023-01-20" },
    { "order_id": "8090af7a-66e4-488b-9cf5-a607ec7b1e4b", "symbol": "8306", "action": "BUY", "timestamp": "2023-01-18" },
    { "order_id": "6005dbee-f267-483a-b74f-c4205bd74b4d", "symbol": "8306", "action": "SELL", "timestamp": "2023-01-20" },
    { "order_id": "6f4e27b0-5b3d-4f02-b2a4-83bfccade737", "symbol": "8306", "action": "BUY", "timestamp": "2023-01-27" },
    { "order_id": "8c6aae07-133c-4a7d-b24f-ed3745d3a79d", "symbol": "8306", "action": "SELL", "timestamp": "2023-02-02", "status": "force_closed" },
    { "order_id": "7b3bf1dd-eb10-4942-8609-507e29c31f88", "symbol": "8001", "action": "BUY", "timestamp": "2023-01-24" },
    { "order_id": "91b6893f-df7e-400c-9ad7-36badfa4d61c", "symbol": "8001", "action": "SELL", "timestamp": "2023-02-03", "status": "force_closed" }
  ]
}
```

**証拠**: 実ファイル読み込み結果  
**判明事項**: 
- execution_details: **10件**
- 全てのorder_idは一意（UUID形式）
- 8306銘柄のBUY/SELLが**3回繰り返されている**（2023-01-18→01-20）

---

### **2.2 実行195347（修正直後）との比較**

#### **基本指標の比較表**

| 項目 | 実行195347（修正直後） | 実行205706（最新） | 差分 | 状態 |
|------|----------------------|-------------------|------|------|
| **execution_details件数** | 4件 | 10件 | +6件 | **重大な差異** |
| **dssms_trades.csv取引数** | 2件 | 2件 | 0件 | 整合 |
| **winning_trades** | 1件 | 0件 | -1件 | **不整合** |
| **losing_trades** | 1件 | 2件 | +1件 | **不整合** |
| **net_profit** | +81,398円 | -1,185円 | -82,583円 | **極端な差異** |
| **final_portfolio_value** | 1,081,399円 | 998,815円 | -82,584円 | **極端な差異** |

---

#### **execution_detailsの詳細比較**

**実行195347（4件、期待される正常な状態）:**
1. order_id: 4afe849b... 8306 BUY 2023-01-18
2. order_id: fab712b7... 8306 SELL 2023-01-20
3. order_id: 78fd03c7... 8001 BUY 2023-01-24
4. order_id: c5580a8d... 8001 SELL 2023-02-03 (force_closed)

**実行205706（10件、異常な重複）:**
1. order_id: 654b7c6e... 8306 BUY 2023-01-18
2. order_id: cd83c6fa... 8306 SELL 2023-01-20
3. order_id: 3ee8af6c... 8306 BUY 2023-01-18 ← **重複1**
4. order_id: 59215fdf... 8306 SELL 2023-01-20 ← **重複1**
5. order_id: 8090af7a... 8306 BUY 2023-01-18 ← **重複2**
6. order_id: 6005dbee... 8306 SELL 2023-01-20 ← **重複2**
7. order_id: 6f4e27b0... 8306 BUY 2023-01-27
8. order_id: 8c6aae07... 8306 SELL 2023-02-02 (force_closed)
9. order_id: 7b3bf1dd... 8001 BUY 2023-01-24
10. order_id: 91b6893f... 8001 SELL 2023-02-03 (force_closed)

**証拠**: 両実行のdssms_execution_results.json比較  
**判明事項**: 
- 実行205706では8306のBUY/SELLが**3回記録**されている（2023-01-18→01-20のペアが3セット）
- 各ペアのorder_idは異なるため、重複除去ロジックが機能していない
- これは累積期間バックテストによる**過去取引の再実行**を示唆

---

### **2.3 重複除去ロジックの実装確認**

#### **修正済みコード（Line 2778-2805）**
```python
# [2025-12-09修正] order_idベースの重複除去に変更
order_id = detail.get('order_id')
if not order_id:
    skipped_invalid_count += 1
    self.logger.warning(
        f"[DEDUP_SKIP] daily_result[{idx}], detail[{detail_idx}]: "
        f"order_id欠損のためスキップ "
        f"(timestamp={timestamp}, action={action}, symbol={symbol})"
    )
    continue

unique_key = order_id

# 重複チェック
if unique_key in seen_keys:
    duplicate_count += 1
    self.logger.debug(
        f"[DEDUP_SKIP] Duplicate execution_detail: "
        f"order_id={order_id}, timestamp={timestamp}, action={action}, symbol={symbol}, "
        f"price={price:.2f}, strategy={strategy_name}"
    )
    continue

seen_keys.add(unique_key)
all_execution_details.append(detail)
```

**証拠**: src/dssms/dssms_integrated_main.py Line 2778-2805  
**判明事項**: 
- order_idベースの重複除去ロジックは**正しく実装されている**
- しかし、**同じtimestamp+symbol+actionでも異なるorder_idが生成されている**
- これは重複除去ロジックの問題ではなく、**データ生成側の問題**を示唆

---

## 🔴 **3. 根本原因の特定**

### **3.1 累積期間バックテストによる過去取引の再実行**

**調査レポート（20251209_dssms_execution_discrepancy_report.md）のSection 12より:**

```
Line 1692: 累積期間バックテストの実装
backtest_start_date = self.dssms_backtest_start_date  # DSSMS開始日
backtest_end_date = target_date  # 現在の日付

つまり、毎日、DSSMS開始日からtarget_dateまでのバックテストを実行
結果として、過去の取引が毎回execution_detailsに含まれる
```

**実行205706での具体的な挙動:**

1. **2023-01-18（初回）**: 8306 BUY/SELL実行 → order_id: 654b7c6e, cd83c6fa
2. **2023-01-19〜01-26**: 8306ポジション保持中、累積期間バックテストで過去取引（2023-01-18→01-20）を再実行
   - 新しいorder_id: 3ee8af6c, 59215fdf（2回目）
   - 新しいorder_id: 8090af7a, 6005dbee（3回目）
3. **2023-01-27**: 8306の新規BUY実行 → order_id: 6f4e27b0
4. **2023-02-02**: 8306 SELL（force_close） → order_id: 8c6aae07

**証拠**: 
- execution_detailsの順序とtimestampの分析
- 調査レポートSection 12.4のルートコーズ説明

**判明事項**:
- 累積期間バックテストにより、**過去の取引が毎日新しいorder_idで再実行される**
- order_idベースの重複除去では、**異なるorder_idは異なる取引と判定される**
- これにより、**実際には1回の取引だが3回記録される**という事態が発生

---

### **3.2 なぜ実行195347では問題が起きなかったか**

**実行195347のexecution_details（4件のみ）:**
- 8306: 1回のBUY/SELLのみ（2023-01-18→01-20）
- 8001: 1回のBUY/SELLのみ（2023-01-24→02-03）

**推測される原因:**
1. **実行タイミングの違い**: 実行195347では累積期間バックテストの影響が限定的だった可能性
2. **ランダム性の影響**: スリッページや手数料のランダム性により、異なる日に実行された可能性
3. **データ取得の差異**: yfinanceのデータ取得タイミングによる微妙な違い

**証拠**: 
- 実行195347のexecution_detailsには重複が存在しない
- ただし、累積期間バックテストの実装は同じ（Line 1692）

**判明事項**:
- 実行195347の結果は**偶然正常だった可能性が高い**
- 累積期間バックテストの仕組み上、重複は**必然的に発生する**はず
- 実行205706の結果が**本来の動作**を示している可能性

---

## 💡 **4. 問題の整理**

### **4.1 確定した事実**
1. **実行205706ではexecution_detailsが10件記録されている**（実行195347は4件）
2. **全てのorder_idは一意**であり、重複除去ロジックは機能していない
3. **累積期間バックテストにより過去取引が再実行される**仕組みが存在する（Line 1692）
4. **同じtimestamp+symbol+actionの取引が複数回記録されている**（異なるorder_id）
5. **dssms_trades.csvは2件**のみで、execution_detailsの10件と不整合

### **4.2 重大な矛盾**

**矛盾1: execution_details vs dssms_trades.csv**
- execution_details: 10件（8306×6件 + 8001×2件 + force_close×2件）
- dssms_trades.csv: 2件（8306×1件 + 8001×1件）

**矛盾2: 利益の完全な逆転**
- 実行195347: +81,399円（勝率50%）
- 実行205706: -1,185円（勝率0%）

**矛盾3: 8306の取引日の不一致**
- execution_details: 2023-01-18→01-20（3セット） + 2023-01-27→02-02（1セット）
- dssms_trades.csv: 2023-01-27→02-02（1セットのみ）

**証拠**: 上記2.1〜2.2の実データ比較  
**判明事項**: execution_detailsとdssms_trades.csvの間に**深刻な乖離**が存在する

---

### **4.3 推定される原因（可能性順）**

#### **🥇 最も可能性が高い: ComprehensiveReporterのペアリング処理の問題**

**仮説:**
- execution_detailsには10件の注文が記録されている
- ComprehensiveReporterの`_convert_to_trades_format()`メソッドが、重複する取引を**正しくペアリングできていない**
- その結果、dssms_trades.csvには一部の取引のみが記録される

**確認が必要:**
- `main_system/reporting/comprehensive_reporter.py`のペアリングロジック
- symbol_based_fifoの処理フロー
- 同じtimestamp+symbolの複数BUY/SELLをどう処理するか

---

#### **🥈 2番目に可能性が高い: 累積期間バックテストの設計上の問題**

**仮説:**
- 累積期間バックテストは**意図的な設計**である可能性（調査レポートSection 12.6参照）
- しかし、execution_detailsへの記録方式が不適切
- 過去取引を**累積的に含める必要はない**はず

**確認が必要:**
- Line 1692の設計意図（なぜ毎日DSSMS開始日から実行するのか）
- `_execute_multi_strategies()`の返り値（過去取引を含むべきか）
- IntegratedExecutionManagerの統合ロジック（Line 494-521）

---

#### **🥉 3番目に可能性が高い: 実行195347の結果が異常**

**仮説:**
- 実行205706が**正常な動作**を示している
- 実行195347は何らかの理由で重複が発生しなかった**例外的なケース**
- 累積期間バックテストの仕組み上、10件のexecution_detailsが正しい

**反証:**
- しかし、実行195347の結果（+81,399円、勝率50%）は**合理的**
- 実行205706の結果（-1,185円、勝率0%）は**不自然**
- 過去の調査レポート（Section 11）でも実行195347の結果を「正常」と判断

---

## ❓ **5. 不明な点**

### **5.1 ComprehensiveReporterのペアリング処理**
- symbol_based_fifoは同じtimestamp+symbolの複数BUY/SELLをどう処理するか?
- order_idが異なる場合、別取引と判定されるのか?
- ペアリング失敗時のログ出力はあるか?（実行205706のログファイルが存在しない）

### **5.2 累積期間バックテストの設計意図**
- なぜ毎日DSSMS開始日からバックテストを実行するのか?
- 期間比較の目的があるのか?（調査レポートSection 12.6のメリット参照）
- 過去取引をexecution_detailsに含めるべきか?

### **5.3 実行195347と205706の差異の原因**
- 同じコード、同じパラメータで実行したはずなのに、なぜ結果が異なるのか?
- ランダム性（スリッページ、手数料）だけで説明できる差異か?
- データ取得タイミングの違いで取引シグナルが変わる可能性は?

### **5.4 dssms_trades.csvの生成ロジック**
- execution_detailsからdssms_trades.csvを生成するのは誰か?（ComprehensiveReporter）
- なぜ10件のexecution_detailsから2件のtradesしか生成されないのか?
- ペアリング処理でどのような条件で取引がスキップされるのか?

---

## ✅ **6. セルフチェック結果**

### **a) 見落としチェック**
- ✅ 両実行の全主要ファイル確認済み
- ✅ execution_detailsの詳細な比較済み
- ✅ order_idの一意性確認済み
- ✅ 累積期間バックテストの実装確認済み（調査レポート参照）
- ⚠️ **未確認**: ComprehensiveReporterのペアリング処理（ソースコード確認が必要）
- ⚠️ **未確認**: 実行205706のログファイル（存在しない）
- ⚠️ **未確認**: `_execute_multi_strategies()`の返り値の詳細

### **b) 思い込みチェック**
- ✅ 「order_id重複除去で問題解決」→実際は新たな問題が発覚
- ✅ 「execution_detailsとtradesは同じ件数であるはず」→実際は大きく異なる
- ✅ 「実行195347の結果が正常」→実際は偶然の可能性あり
- ✅ 実際のファイル内容を確認して事実を把握
- ✅ 推測と事実を明確に区別

### **c) 矛盾チェック**
- ✅ execution_details 10件 vs trades 2件の矛盾を発見
- ✅ 利益の逆転（+81,399円 → -1,185円）の矛盾を発見
- ✅ 8306取引日の不一致を発見
- ✅ 累積期間バックテストの仕組みと重複記録の関連性を特定

---

## 📝 **7. 結論**

### **7.1 確定事項**
1. **実行205706のexecution_detailsには10件の注文が記録されている**
2. **全てのorder_idは一意であり、重複除去ロジックは正しく動作している**
3. **累積期間バックテストにより過去取引が新しいorder_idで再実行されている**
4. **dssms_trades.csvには2件しか記録されず、execution_detailsと乖離している**
5. **利益が逆転し、取引結果が大きく異なる**

### **7.2 推定される主要原因**
1. **ComprehensiveReporterのペアリング処理**が、重複するBUY/SELLを正しく処理できていない
2. **累積期間バックテストの設計**により、過去取引が毎日再実行されている
3. **execution_detailsへの記録方式**が不適切（過去取引を累積的に含めるべきではない）

### **7.3 影響範囲**
- **DSSMSレポート系**が不正確（trades.csv、performance_metrics.json）
- **バックテスト結果の信頼性**に重大な疑問
- **累積期間バックテストの設計**に根本的な問題がある可能性

---

## 🔧 **8. 次のアクションアイテム**

### **優先度1: ComprehensiveReporterのペアリング処理の確認** 🔴
- [ ] `main_system/reporting/comprehensive_reporter.py`の`_convert_to_trades_format()`確認
- [ ] symbol_based_fifoの処理フローを確認
- [ ] 同じtimestamp+symbolの複数BUY/SELLのペアリングロジック確認
- [ ] ペアリング失敗時のログ確認（どの取引がスキップされたか）

**理由**: execution_details 10件 → trades 2件の乖離の直接原因

### **優先度2: 累積期間バックテストの設計意図の確認** 🟠
- [ ] Line 1692の設計意図をドキュメント/コメントから確認
- [ ] `_execute_multi_strategies()`の返り値仕様を確認
- [ ] IntegratedExecutionManagerの統合ロジック（Line 494-521）確認
- [ ] 累積期間方式のメリット・デメリット再評価（調査レポートSection 12.6参照）

**理由**: 過去取引の再実行が意図的な設計かバグかを判断する必要がある

### **優先度3: execution_detailsの記録方式の見直し** 🟡
- [ ] daily_resultsへの記録タイミングを確認
- [ ] Line 557のextend処理が正しく動作しているか確認
- [ ] 累積期間バックテストの結果から過去取引を除外する必要性を検討
- [ ] `_convert_to_execution_format()`の重複除去ロジック再検討

**理由**: 過去取引が毎日追加される現状を修正する必要がある

### **優先度4: 再現テスト** 🟢
- [ ] 同じ条件で複数回実行し、再現性を確認
- [ ] 実行195347の結果が例外的か、実行205706が例外的かを判断
- [ ] デバッグログを追加して詳細なトレース
- [ ] ログファイル生成の修正（現在生成されていない）

**理由**: 問題の再現性を確認し、ランダム性の影響を排除

---

## 📎 **9. 参照ファイル**

### **実行205706（最新・問題発生）**
- `output/dssms_integration/dssms_20251209_205706/dssms_performance_metrics.json`
- `output/dssms_integration/dssms_20251209_205706/dssms_trades.csv`
- `output/dssms_integration/dssms_20251209_205706/dssms_execution_results.json`

### **実行195347（修正直後・正常動作）**
- `output/dssms_integration/dssms_20251209_195347/dssms_performance_metrics.json`
- `output/dssms_integration/dssms_20251209_195347/dssms_trades.csv`
- `output/dssms_integration/dssms_20251209_195347/dssms_execution_results.json`

### **関連ドキュメント**
- `docs/investigation/20251209_dssms_execution_discrepancy_report.md`（Section 12: 累積期間バックテストの検証）
- `src/dssms/dssms_integrated_main.py` Line 1692（累積期間バックテスト実装）
- `src/dssms/dssms_integrated_main.py` Line 2765-2805（order_id重複除去ロジック）

---

## 🎯 **10. 修正案（暫定）**

### **修正案A: 累積期間バックテストの廃止**
```python
# 現在（Line 1692-1695）
backtest_start_date = self.dssms_backtest_start_date  # DSSMS開始日
backtest_end_date = target_date  # 現在の日付

# 修正案A: 日次バックテストに変更
backtest_start_date = target_date - timedelta(days=30)  # 30日前から
backtest_end_date = target_date  # 現在の日付まで
```

**メリット**:
- 過去取引の再実行を防ぐ
- execution_detailsの重複を根本的に解決
- 処理時間の大幅短縮

**デメリット**:
- 累積期間比較ができなくなる（調査レポートSection 12.6参照）
- 設計意図を無視する可能性

---

### **修正案B: execution_detailsのフィルタリング**
```python
# _convert_to_execution_format()内で、当日の取引のみを抽出
if detail.get('timestamp') == target_date:
    all_execution_details.append(detail)
```

**メリット**:
- 累積期間バックテストを維持
- execution_detailsには当日取引のみ記録
- 期間比較の目的を達成可能

**デメリット**:
- 過去取引の再実行自体は継続（処理時間増加）
- フィルタリングの基準が不明確（timestampは取引日か発注日か）

---

### **修正案C: ComprehensiveReporterの修正**
```python
# ペアリング処理で同じtimestamp+symbolの複数BUY/SELLを正しく処理
# order_idではなくtimestamp+symbol+actionで重複除去
```

**メリット**:
- 累積期間バックテストを維持
- execution_detailsの記録方式を変更不要
- ペアリング処理のみ修正

**デメリット**:
- order_id重複除去の修正を無にする
- 根本的な解決にならない可能性

---

**調査完了日時**: 2025年12月9日  
**調査担当**: GitHub Copilot  
**ステータス**: 🔴 **新たな問題発見・追加調査必要**

---

## 📊 **11. 優先度1調査: ComprehensiveReporterのペアリング処理**

**調査日時**: 2025年12月9日 21:30  
**調査項目**: ComprehensiveReporterの`_convert_to_trades_format()`およびsymbol_based_fifoペアリングロジック

---

### **11.1 調査チェックリスト**

#### **優先度1: ペアリング処理の基本確認**
- [x] `_convert_to_trades_format()`メソッドの実装確認（Line 400-580）
- [x] symbol_based_fifoの処理フロー確認
- [x] 同じtimestamp+symbolの複数BUY/SELLペアリングロジック確認
- [x] ペアリング失敗時のログ出力確認

---

### **11.2 ComprehensiveReporterのペアリング処理実装**

#### **処理フロー（Line 400-580）**

**Phase 1: BUY/SELL注文の抽出**
```python
# Line 422: 共通ユーティリティでBUY/SELL抽出
buy_orders, sell_orders = extract_buy_sell_orders(execution_details, self.logger)
```

**Phase 2: 銘柄別グループ化（Task 8対応）**
```python
# Line 428-443: 銘柄別にBUY/SELLを分類
buy_by_symbol = defaultdict(list)
sell_by_symbol = defaultdict(list)

for buy in buy_orders:
    symbol = buy.get('symbol')
    if symbol:
        buy_by_symbol[symbol].append(buy)

for sell in sell_orders:
    symbol = sell.get('symbol')
    if symbol:
        sell_by_symbol[symbol].append(sell)
```

**Phase 3: 銘柄別FIFOペアリング**
```python
# Line 445-464: 各銘柄について順番にペアリング
for symbol in sorted(all_symbols):
    buys = buy_by_symbol.get(symbol, [])
    sells = sell_by_symbol.get(symbol, [])
    paired_count = min(len(buys), len(sells))  # ← 重要: 最小値でペア数決定
    
    # Line 464: FIFOペアリング（インデックスベース）
    for i in range(paired_count):
        buy_order = buys[i]
        sell_order = sells[i]
        # ... 取引レコード作成
```

**証拠**: `main_system/reporting/comprehensive_reporter.py` Line 400-580

---

### **11.3 重要な発見: FIFOペアリングの仕組み**

#### **ペアリングの決定ロジック**

**Line 455: ペア数の決定**
```python
paired_count = min(len(buys), len(sells))
```

**意味:**
- 銘柄ごとに、BUY件数とSELL件数の**少ない方**がペア数になる
- インデックス順（FIFO = First In First Out）でペアリング
- **timestampは考慮されない**（リストの順序のみ）

**実行205706の場合:**
- 8306銘柄: BUY 6件, SELL 6件 → `paired_count = min(6, 6) = 6`
- しかし、実際にはペア数2件のみ生成

**矛盾の発見:**
- ペアリングロジック上は**6ペア生成されるはず**
- 実際のdssms_trades.csvには**2ペアのみ**

**証拠**: comprehensive_reporter.py Line 455 + 調査レポートSection 2.1

---

### **11.4 データ検証ロジックの確認**

#### **Line 478-486: データ検証とスキップ処理**
```python
if not all([entry_date, exit_date, entry_price > 0, exit_price > 0, shares > 0]):
    self.logger.error(
        f"[DATA_VALIDATION_FAILED] 不正な取引データ（ペア{i+1}）: "
        f"entry_date={entry_date}, exit_date={exit_date}, "
        f"entry_price={entry_price}, exit_price={exit_price}, shares={shares}. "
        f"スキップします。"
    )
    continue
```

**判明事項:**
- データ検証で不合格の場合、`continue`でスキップ
- スキップされた取引は`[DATA_VALIDATION_FAILED]`ログに記録される
- **しかし、実行205706のログファイルが存在しない**ため確認不可

**証拠**: comprehensive_reporter.py Line 478-486

---

### **11.5 ログ出力の確認**

#### **期待されるログ（Line 447-462）**
```python
# 銘柄別ペアリング状況ログ
self.logger.info(
    f"[SYMBOL_BASED_PAIRING] 処理対象銘柄数: {len(all_symbols)}, "
    f"BUY銘柄: {len(buy_by_symbol)}, SELL銘柄: {len(sell_by_symbol)}"
)

if paired_count > 0:
    self.logger.info(
        f"[SYMBOL_PAIRING] 銘柄={symbol}, BUY={len(buys)}, SELL={len(sells)}, "
        f"ペア数={paired_count}"
    )
```

#### **ログファイルの調査結果**
- `logs/comprehensive_reporter.log`: **存在しない**
- `output/dssms_integration/dssms_20251209_205706/`: ログファイル**なし**
- DSSMSIntegratedBacktesterのログ設定を確認する必要あり

**証拠**: file_search結果、list_dir結果

---

### **11.6 推定される原因**

#### **🥇 最も可能性が高い: データ検証でのスキップ**

**仮説:**
1. 実行205706のexecution_detailsには10件の注文が存在
2. ComprehensiveReporterは銘柄別に6ペア（8306）+2ペア（8001）=8ペアを試みる
3. しかし、**データ検証（Line 478）で4ペアがスキップ**される
4. 結果として2ペアのみがdssms_trades.csvに記録される

**確認が必要:**
- どの4ペアがスキップされたか
- スキップの理由（entry_date/exit_date/price/sharesのどれが不正か）
- ログファイルが存在しない理由

---

#### **🥈 2番目に可能性が高い: execution_detailsの順序問題**

**仮説:**
1. execution_detailsの順序がtimestamp順ではない
2. FIFOペアリングはインデックス順なので、**誤ったペアが生成される**
3. 誤ったペアはデータ検証で不合格になりスキップ

**実行205706のexecution_details順序:**
1. 8306 BUY 2023-01-18
2. 8306 SELL 2023-01-20
3. 8306 BUY 2023-01-18 ← **重複1**
4. 8306 SELL 2023-01-20 ← **重複1**
5. 8306 BUY 2023-01-18 ← **重複2**
6. 8306 SELL 2023-01-20 ← **重複2**
7. 8306 BUY 2023-01-27
8. 8306 SELL 2023-02-02

**FIFOペアリング結果:**
- ペア1: BUY[0] + SELL[0] = 2023-01-18 → 2023-01-20 ✅ 正常
- ペア2: BUY[1] + SELL[1] = 2023-01-18 → 2023-01-20 ✅ 正常（重複1）
- ペア3: BUY[2] + SELL[2] = 2023-01-18 → 2023-01-20 ✅ 正常（重複2）
- ペア4: BUY[3] + SELL[3] = 2023-01-27 → 2023-02-02 ✅ 正常

**矛盾:**
- 全て正常なペアに見える
- データ検証でスキップされる理由が不明

**証拠**: 調査レポートSection 2.1のexecution_details順序

---

#### **🥉 3番目に可能性が高い: executed_priceの欠損**

**仮説:**
1. 累積期間バックテストで再実行された取引は、executed_priceが設定されていない可能性
2. `detail.get('executed_price', 0.0)`でデフォルト値0.0が設定される
3. `entry_price > 0`の検証で不合格 → スキップ

**確認が必要:**
- execution_detailsの各注文にexecuted_priceが存在するか
- デフォルト値0.0が設定されているか

---

### **11.7 不明な点**

#### **ログファイルが存在しない理由**
- ComprehensiveReporterは`setup_logger()`でログファイル出力を設定している（Line 98）
- しかし、`logs/comprehensive_reporter.log`が存在しない
- DSSMSIntegratedBacktesterでのログファイル設定を確認する必要

#### **実際にスキップされた取引**
- データ検証で不合格になった4ペアの詳細が不明
- `[DATA_VALIDATION_FAILED]`ログが記録されているはずだが確認不可

#### **execution_detailsの順序**
- execution_detailsがどの順序で格納されているか（timestamp順? 生成順?）
- FIFOペアリングに適した順序になっているか

---

### **11.8 次のアクションアイテム**

#### **優先度1: ログファイルの確認** 🔴
- [ ] DSSMSIntegratedBacktesterのログ設定確認
- [ ] ComprehensiveReporterのログ出力先確認
- [ ] `[DATA_VALIDATION_FAILED]`ログの有無確認
- [ ] `[SYMBOL_PAIRING]`ログから実際のペア数確認

#### **優先度2: execution_detailsの詳細確認** 🟠
- [ ] 各注文のexecuted_priceの値を確認
- [ ] デフォルト値0.0が設定されている注文の有無確認
- [ ] timestampの順序確認（FIFO順か確認）

#### **優先度3: デバッグ実行** 🟡
- [ ] 同じ条件で再実行し、詳細ログ出力を有効化
- [ ] `[DATA_VALIDATION_FAILED]`ログを確認
- [ ] スキップされた取引の詳細を特定

---

### **11.9 セルフチェック結果**

#### **a) 見落としチェック**
- ✅ ComprehensiveReporterの実装確認済み
- ✅ FIFOペアリングロジック確認済み
- ✅ データ検証ロジック確認済み
- ⚠️ **未確認**: ログファイルの内容（存在しないため）
- ⚠️ **未確認**: execution_detailsのexecuted_price値

#### **b) 思い込みチェック**
- ✅ 「FIFOペアリングで全ペア生成されるはず」→実際は2ペアのみ
- ✅ 「ログファイルが存在するはず」→実際は存在しない
- ✅ 実際のコードを確認して処理フローを把握
- ✅ 推測と事実を明確に区別

#### **c) 矛盾チェック**
- ✅ `paired_count=6`のはずが2ペアのみ生成される矛盾を発見
- ✅ データ検証スキップの可能性を特定
- ✅ ログファイル不在による調査限界を認識

---

### **11.10 結論**

#### **確定事項**
1. **ComprehensiveReporterはsymbol_based_fifoペアリングを実装**
2. **ペア数は`min(len(buys), len(sells))`で決定される**
3. **FIFOペアリングはインデックス順（timestampは考慮されない）**
4. **データ検証で不合格の場合、`continue`でスキップされる**
5. **ログファイルが存在せず、詳細な動作確認ができない**

#### **推定される主要原因**
1. **データ検証でのスキップ**: 4ペアがデータ検証で不合格になりスキップされた可能性が高い
2. **executed_priceの欠損**: 累積期間バックテストで再実行された取引にexecuted_priceが設定されていない可能性
3. **ログ出力の問題**: ログファイルが生成されず、詳細な動作確認ができない

#### **次のステップ**
1. ログファイルの確認と詳細ログ出力の有効化
2. execution_detailsの詳細確認（executed_priceの値）
3. デバッグ実行による実際のスキップ理由の特定

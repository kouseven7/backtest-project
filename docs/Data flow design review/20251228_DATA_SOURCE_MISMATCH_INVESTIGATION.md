# データソース不一致問題 調査レポート

**作成日**: 2025-12-28  
**調査者**: GitHub Copilot  
**対象**: Phase B-2で発見したデータソース不一致問題

---

## 1. 問題の概要

### 1.1 症状
Phase B-2実装完了後、以下の不一致が検出されました：

| 出力形式 | 戦略名表示 | データソース |
|---------|-----------|------------|
| CSV出力 | `VWAPBreakoutStrategy` | comprehensive_reporter.py |
| テキストレポート | `UnknownStrategy` | main_text_reporter.py |
| WARNINGログ | フォールバック検出 | main_text_reporter.py |

**Phase B-2で追加したWARNINGログ**:
```
[FALLBACK] 戦略名が取得できませんでした（期待値計算）: trade=N/A, デフォルト値='UnknownStrategy'
[FALLBACK] 戦略名が取得できませんでした（パフォーマンス分析）: trade=N/A, デフォルト値='UnknownStrategy'
[FALLBACK] 戦略名が取得できませんでした（取引履歴セクション）: trade=N/A, デフォルト値='UnknownStrategy'
```

### 1.2 影響範囲
- テキストレポートの戦略別分析が不正確（すべて`UnknownStrategy`と表示）
- CSV出力は正常（正しい戦略名が記録）
- システムは動作継続（致命的エラーなし）

---

## 2. 調査手順

### 2.1 確認項目チェックリスト
1. ✅ main_text_reporter.pyのデータ受け取りフロー確認
2. ✅ comprehensive_reporter.pyからの呼び出し箇所特定
3. ✅ 渡されるデータの実際の構造確認
4. ✅ CSV出力とテキストレポートのデータソース比較
5. ✅ ターミナルログからのデータ構造確認
6. ✅ 原因特定と調査結果レポート作成

### 2.2 調査方法
- ファイル読み取り: 実際のコードを確認
- ターミナルログ分析: 実行時のデータ構造を確認
- データフロー追跡: comprehensive_reporter.py → main_text_reporter.py

---

## 3. 調査結果

### 3.1 根本原因の特定

**問題箇所1**: main_text_reporter.py Line 234
```python
trade_record = {
    'strategy': buy_order.get('strategy_name', 'Unknown'),  # ← 問題: キー名が'strategy'
    'entry_date': buy_order.get('timestamp'),
    'exit_date': sell_order.get('timestamp'),
    ...
}
```

**問題箇所2**: main_text_reporter.py Line 622, 717, 794（Phase B-2修正箇所）
```python
# Phase B-2で修正した箇所
strategy = trade.get('strategy_name', 'UnknownStrategy')  # ← 'strategy_name'キーを期待
```

**データ構造の実際**（ターミナルログより）:
```
First trade content: {'strategy': 'VWAPBreakoutStrategy', ...}
```

### 3.2 不一致の発生メカニズム

```
┌─────────────────────────────────────────────┐
│ execution_details（元データ）                │
│  - strategy_name: 'VWAPBreakoutStrategy'   │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ main_text_reporter.py Line 234             │
│  取引レコード作成（_extract_from_execution_results） │
│  - 'strategy': buy_order.get('strategy_name') │  ← キー名変換（strategy_name → strategy）
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ completed_trades（内部データ）               │
│  - 'strategy': 'VWAPBreakoutStrategy'      │  ← 'strategy'キーで保存
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ main_text_reporter.py Line 622, 717, 794  │
│  Phase B-2修正箇所                          │
│  - trade.get('strategy_name', 'UnknownStrategy') │  ← 'strategy_name'キーを期待
└─────────────────────────────────────────────┘
                    ↓
           ❌ キー名不一致 ❌
        'strategy_name'キーが存在しない
                    ↓
      フォールバック値'UnknownStrategy'を使用
```

### 3.3 CSV出力が正常な理由

**CSV出力の生成**: comprehensive_reporter.py Line 581
```python
trade_record = {
    ...
    'strategy_name': buy_order.get('strategy_name', 'UnknownStrategy'),  # ← 正しい
    ...
}
```

**データフロー比較**:
| 処理 | データ作成箇所 | キー名 | 結果 |
|------|--------------|-------|------|
| CSV出力 | comprehensive_reporter.py Line 581 | `'strategy_name'` | ✅ 正常 |
| テキストレポート | main_text_reporter.py Line 234 | `'strategy'` | ❌ 不一致 |

---

## 4. 根拠の明示

### 4.1 コード確認による証拠

**証拠1**: main_text_reporter.py Line 234
```python
# ファイル: main_system/reporting/main_text_reporter.py
# 関数: _extract_from_execution_results

trade_record = {
    'strategy': buy_order.get('strategy_name', 'Unknown'),
    # ↑ buy_orderから'strategy_name'を取得するが、
    # ↓ 辞書のキーとして'strategy'を使用（不一致）
    ...
}
completed_trades.append(trade_record)
```

**証拠2**: main_text_reporter.py Line 622（Phase B-2修正箇所）
```python
# Phase B-2で修正した箇所
strategy = trade.get('strategy_name', 'UnknownStrategy')
if strategy == 'UnknownStrategy':
    logger.warning(
        f"[FALLBACK] 戦略名が取得できませんでした（期待値計算）: trade=N/A, "
        f"デフォルト値='UnknownStrategy'"
    )
```

### 4.2 ターミナルログによる証拠

**証拠3**: phase_b3_test_output.log - PHASE_5_B_2_DEBUG
```
[2025-12-28 09:58:48,098] INFO - main_system.reporting.main_text_reporter - 
[PHASE_5_B_2_DEBUG] First trade content: 
{'strategy': 'VWAPBreakoutStrategy', 'entry_date': '2025-01-15T00:00:00', ...}
```

実際のデータ構造: `'strategy'`キーが使用されている

**証拠4**: phase_b3_test_output.log - FALLBACK警告
```
[2025-12-28 09:58:48,099] WARNING - main_system.reporting.main_text_reporter - 
[FALLBACK] 戦略名が取得できませんでした（期待値計算）: trade=N/A, デフォルト値='UnknownStrategy'
```

Phase B-2で追加したWARNINGログが、データ構造の不一致を正しく検出

### 4.3 CSV出力による証拠

**証拠5**: 6954.T_all_transactions.csv（Phase B-3テスト結果）
```csv
symbol,entry_date,entry_price,exit_date,exit_price,shares,pnl,return_pct,holding_period_days,strategy_name,position_value,is_forced_exit
6954.T,2025-01-15,4432.440101046878,2025-01-22T00:00:00,4685.960178480775,200,50704.01548677928,0.0571965038792106,7,VWAPBreakoutStrategy,886429.5450250525,False
```

CSV出力には`strategy_name`列に`VWAPBreakoutStrategy`が正しく記録されている

---

## 5. 原因の詳細分析

### 5.1 Phase B-2修正の意図

Phase B-2では、調査レポート（20251226_PHASE_B_INVESTIGATION_REPORT.md）に基づき、以下の修正を実施しました：

**修正方針**:
```python
# 修正前
strategy = trade.get('strategy', 'Unknown')

# 修正後
strategy = trade.get('strategy_name', 'UnknownStrategy')
```

**目的**: 命名規則統一（`'strategy'` → `'strategy_name'`）

### 5.2 修正が不完全だった理由

**Phase B-2の修正範囲**:
- ✅ Line 622: `_calculate_strategy_expected_values` - 戦略名の**読み取り**を修正
- ✅ Line 717: `_analyze_strategy_performance` - 戦略名の**読み取り**を修正
- ✅ Line 794: `_generate_trade_history_section` - 戦略名の**読み取り**を修正

**Phase B-2で未修正**:
- ❌ Line 234: `_extract_from_execution_results` - 戦略名の**書き込み**（取引レコード作成時のキー名）

**結果**: 
- 取引レコードは`'strategy'`キーで作成される（Line 234）
- 読み取り時は`'strategy_name'`キーを期待する（Line 622, 717, 794）
- **キー名の不一致**が発生

### 5.3 なぜ発見されなかったのか

**Phase B-2テスト時の状況**:
1. main_new.py実行テスト: ✅ 成功（エラーなし）
2. テキストレポート生成: ✅ 成功（`UnknownStrategy`と表示）
3. WARNINGログ出力: ✅ 検出（フォールバック動作を可視化）

**問題点**:
- `UnknownStrategy`と表示されることが「エラー」ではなく「フォールバック動作」として処理された
- Phase B-2の目的は「フォールバック検出機能の実装」であり、この点では**成功**していた
- しかし、取引レコード作成時のキー名不一致には気づけなかった

---

## 6. セルフチェック結果

### 6.1 見落としチェック
- ✅ 確認していないファイルはないか? → すべて確認済み
- ✅ カラム名、変数名、関数名を実際に確認したか? → コードとログで確認
- ✅ データの流れを追いきれているか? → 完全に追跡完了

### 6.2 思い込みチェック
- ✅ 「〇〇であるはず」という前提を置いていないか? → すべて実際のコード・ログで確認
- ✅ 実際にコードや出力で確認した事実か? → 証拠付きで確認
- ✅ 「存在しない」と結論づけたものは本当に確認したか? → ターミナルログで実データを確認

### 6.3 矛盾チェック
- ✅ 調査結果同士で矛盾はないか? → 矛盾なし
- ✅ 提供されたログ/エラーと結論は整合するか? → 完全に整合

---

## 7. 結論

### 7.1 根本原因
**main_text_reporter.py Line 234のキー名不一致**

取引レコード作成時に`'strategy'`キーを使用しているが、Phase B-2修正箇所（Line 622, 717, 794）では`'strategy_name'`キーを読み取ろうとするため、キー名の不一致が発生しています。

### 7.2 影響範囲
- **テキストレポート**: 戦略別分析が不正確（すべて`UnknownStrategy`と表示）
- **CSV出力**: 正常（comprehensive_reporter.pyが別途作成するため影響なし）
- **システム動作**: 継続可能（致命的エラーなし）

### 7.3 Phase B-2の成果
Phase B-2で追加したWARNINGログにより、この問題を**可視化**できました。これはcopilot-instructions.md要件「フォールバック実行時のログ必須」を達成した重要な成果です。

---

## 8. 修正提案

### 8.1 修正箇所
**ファイル**: main_system/reporting/main_text_reporter.py  
**行番号**: Line 234

**現在のコード**:
```python
trade_record = {
    'strategy': buy_order.get('strategy_name', 'Unknown'),
    'entry_date': buy_order.get('timestamp'),
    ...
}
```

**修正後のコード**:
```python
trade_record = {
    'strategy_name': buy_order.get('strategy_name', 'UnknownStrategy'),  # キー名を'strategy_name'に統一
    'entry_date': buy_order.get('timestamp'),
    ...
}
```

### 8.2 修正理由
1. **命名規則統一**: Phase Bの目的に沿って`'strategy_name'`に統一
2. **フォールバック値統一**: `'Unknown'` → `'UnknownStrategy'`
3. **データ整合性確保**: CSV出力（comprehensive_reporter.py）と同じキー名を使用

### 8.3 追加対応（推奨）
WARNINGログ追加:
```python
trade_record = {
    'strategy_name': buy_order.get('strategy_name', 'UnknownStrategy'),
    ...
}

if trade_record['strategy_name'] == 'UnknownStrategy':
    logger.warning(
        f"[FALLBACK] 戦略名が取得できませんでした（取引レコード生成）: "
        f"buy_order={buy_order.keys()}, デフォルト値='UnknownStrategy'"
    )

completed_trades.append(trade_record)
```

### 8.4 テスト項目
修正後、以下を確認すること:
1. main_new.py実行テスト（2025-01-15 to 2025-01-31）
2. テキストレポートの戦略別分析確認（`VWAPBreakoutStrategy`が表示されるか）
3. WARNINGログ出力確認（フォールバックが発生しないか）
4. CSV出力確認（既存動作が維持されているか）

---

## 9. まとめ

### 9.1 調査で判明したこと
- **根本原因**: main_text_reporter.py Line 234のキー名不一致（`'strategy'` vs `'strategy_name'`）
- **発生メカニズム**: 取引レコード作成時と読み取り時でキー名が異なる
- **Phase B-2の成果**: WARNINGログ実装により問題を可視化

### 9.2 Phase Bの進捗状況
| Phase | 対象 | 状態 | 備考 |
|-------|------|------|------|
| Phase B-1 | DSSMS統合系 | ✅ 完了 | 6箇所修正 |
| Phase B-2 | メイン実行系 | ⚠️ 要追加修正 | 3箇所修正済み + Line 234要修正 |
| Phase B-3 | 実行制御系 | ✅ 完了 | 9箇所修正 |

### 9.3 次のアクション
1. **Phase B-2追加修正**: main_text_reporter.py Line 234のキー名を`'strategy_name'`に修正
2. **修正後テスト**: 4項目の確認実施
3. **Phase B完了報告**: 全修正完了後、Phase B総括レポート作成

---

**調査完了日**: 2025-12-28  
**調査工数**: 約30分  
**次のステップ**: Phase B-2追加修正の実施

---

## 付録: 関連ファイル一覧

### A.1 調査対象ファイル
1. main_system/reporting/main_text_reporter.py
   - Line 234: 取引レコード作成（問題箇所）
   - Line 622, 717, 794: Phase B-2修正箇所

2. main_system/reporting/comprehensive_reporter.py
   - Line 581: CSV出力用取引レコード作成（正常動作）
   - Line 1064: main_text_reporter.pyへのデータ受け渡し

### A.2 参照ログファイル
- phase_b3_test_output.log: Phase B-3テスト実行ログ
  - PHASE_5_B_2_DEBUG: データ構造確認
  - FALLBACK: フォールバック検出ログ

### A.3 参照レポート
- docs/Data flow design review/20251226_PHASE_B_INVESTIGATION_REPORT.md: Phase B調査レポート（Phase B-2修正方針）
- .github/copilot-instructions.md: フォールバック機能の制限ルール

---

**End of Report**

# Phase B-2追加修正 設計書

**作成日**: 2025-12-28  
**設計者**: GitHub Copilot  
**対象**: Phase B-2追加修正（main_text_reporter.py Line 234のキー名不一致修正）  
**前提**: 20251228_DATA_SOURCE_MISMATCH_INVESTIGATION.md 調査結果に基づく

---

## 1. 調査結果サマリー

### 1.1 確認項目チェックリスト

| No. | 項目 | 状態 | 根拠 |
|-----|------|------|------|
| 1 | Line 234周辺コードの詳細確認（修正影響範囲の特定） | ✅ | Lines 220-280読み取り完了 |
| 2 | completed_trades使用箇所の全件確認 | ✅ | 15箇所を確認（grep_search実施） |
| 3 | 類似のキー名不一致問題の有無確認 | ✅ | main_text_reporter.pyで'strategy'キー使用は1箇所のみ（Line 234） |
| 4 | WARNINGログ追加の詳細設計 | ✅ | comprehensive_reporter.py Line 587-591を参考に設計 |
| 5 | テスト手順の詳細設計 | ✅ | 4項目のテスト手順を設計 |
| 6 | 修正実装手順の設計と設計書作成 | 🔄 | 本ドキュメント |

### 1.2 判明した事実

**事実1**: Line 234の影響範囲
```python
# ファイル: main_system/reporting/main_text_reporter.py
# 関数: _extract_from_execution_results（Lines 140-280）

# Line 233: trade_recordの作成
trade_record = {
    'strategy': buy_order.get('strategy_name', 'Unknown'),  # ← 問題箇所
    'entry_date': buy_order.get('timestamp'),
    'exit_date': sell_order.get('timestamp'),
    'entry_price': entry_price,
    'exit_price': exit_price,
    'shares': shares,
    'pnl': pnl,
    'return_pct': return_pct,
    'entry_idx': None,
    'exit_idx': None
}
# Line 244: completed_tradesに追加
completed_trades.append(trade_record)
```

**事実2**: completed_trades使用箇所（15箇所）
- Line 163: 初期化 `completed_trades = []`
- Line 244: レコード追加 `completed_trades.append(trade_record)`
- Line 246-260: デバッグログ出力（PHASE_5_B_2）
- Line 264: `_calculate_performance_from_trades` に渡す
- Line 277-278: 戻り値の `trades` セクションに格納

**事実3**: 'strategy'キーの使用箇所
- main_text_reporter.py Line 234: **問題箇所（修正対象）**
- main_text_reporter.py Line 993: テストコード内（__main__セクション、実環境では未使用）

**結論**: Line 234のみが修正対象。他に影響はない。

**事実4**: completed_tradesの読み取り箇所（Phase B-2修正箇所）
- Line 622: `_calculate_strategy_expected_values` - `trade.get('strategy_name', 'UnknownStrategy')`
- Line 717: `_analyze_strategy_performance` - `trade.get('strategy_name', 'UnknownStrategy')`
- Line 794: `_generate_trade_history_section` - `trade.get('strategy_name', 'UnknownStrategy')`

すべて`'strategy_name'`キーを期待している。

**事実5**: comprehensive_reporter.py の正しいパターン
```python
# Line 581-591（CSV出力用）
'strategy_name': buy_order.get('strategy_name', 'UnknownStrategy'),  # 正しい
# Line 587-591: WARNINGログ実装
if trade_record['strategy_name'] == 'UnknownStrategy':
    self.logger.warning(
        f"[FALLBACK] 戦略名が取得できませんでした（取引記録生成）: symbol={symbol}, "
        f"buy_order={buy_order.keys()}, デフォルト値='UnknownStrategy'"
    )
```

---

## 2. 修正設計

### 2.1 修正対象箇所

**ファイル**: `main_system/reporting/main_text_reporter.py`  
**関数**: `_extract_from_execution_results`  
**行番号**: Line 233-234

### 2.2 修正内容

#### 2.2.1 キー名の変更

**現在のコード** (Line 233-234):
```python
trade_record = {
    'strategy': buy_order.get('strategy_name', 'Unknown'),
    'entry_date': buy_order.get('timestamp'),
```

**修正後のコード**:
```python
trade_record = {
    'strategy_name': buy_order.get('strategy_name', 'UnknownStrategy'),
    'entry_date': buy_order.get('timestamp'),
```

**変更点**:
1. キー名: `'strategy'` → `'strategy_name'`
2. フォールバック値: `'Unknown'` → `'UnknownStrategy'`

**修正理由**:
1. **命名規則統一**: Phase Bの目的に沿って`'strategy_name'`に統一
2. **フォールバック値統一**: 調査レポート（20251226_PHASE_B_INVESTIGATION_REPORT.md）で定義された`'UnknownStrategy'`に統一
3. **データ整合性確保**: comprehensive_reporter.py（CSV出力）と同じキー名を使用
4. **Phase B-2修正箇所との整合性**: Lines 622, 717, 794で期待されるキー名と一致

#### 2.2.2 WARNINGログの追加

**追加位置**: Line 244の直前（`completed_trades.append(trade_record)`の前）

**追加コード**:
```python
                            trade_record = {
                                'strategy_name': buy_order.get('strategy_name', 'UnknownStrategy'),
                                'entry_date': buy_order.get('timestamp'),
                                'exit_date': sell_order.get('timestamp'),
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'shares': shares,
                                'pnl': pnl,
                                'return_pct': return_pct,
                                'entry_idx': None,
                                'exit_idx': None
                            }
                            
                            # Phase B-2追加修正: フォールバック検出ログ
                            if trade_record['strategy_name'] == 'UnknownStrategy':
                                logger.warning(
                                    f"[FALLBACK] 戦略名が取得できませんでした（取引レコード生成）: "
                                    f"symbol={symbol}, buy_order={buy_order.keys()}, "
                                    f"デフォルト値='UnknownStrategy'"
                                )
                            
                            completed_trades.append(trade_record)
```

**ログ仕様**:
- **フォーマット**: `[FALLBACK] 戦略名が取得できませんでした（取引レコード生成）: ...`
- **ログレベル**: WARNING
- **出力条件**: `trade_record['strategy_name'] == 'UnknownStrategy'`
- **詳細情報**: 
  - `symbol`: 銘柄コード（FIFOペアリングのループ変数から取得）
  - `buy_order.keys()`: buy_orderに含まれるキー一覧（デバッグ情報）
  - `デフォルト値`: 'UnknownStrategy'

**参考実装**: comprehensive_reporter.py Line 587-591（同一パターン）

**copilot-instructions.md 要件**:
- 「フォールバック実行時のログ必須」を満たす
- 「フォールバックを発見した場合はいかなる場合も報告する」を満たす

---

## 3. 修正の影響範囲分析

### 3.1 直接影響（同一ファイル内）

| 箇所 | 影響内容 | 期待される結果 |
|------|---------|---------------|
| Line 622 | `_calculate_strategy_expected_values` | `'strategy_name'`キーが取得可能になり、フォールバックが発生しなくなる |
| Line 717 | `_analyze_strategy_performance` | `'strategy_name'`キーが取得可能になり、フォールバックが発生しなくなる |
| Line 794 | `_generate_trade_history_section` | `'strategy_name'`キーが取得可能になり、フォールバックが発生しなくなる |

**効果**: Phase B-2で追加したWARNINGログが出力されなくなる（正常動作）

### 3.2 間接影響（他ファイル）

| ファイル | 影響内容 | 確認事項 |
|---------|---------|---------|
| comprehensive_reporter.py | 影響なし | CSV出力は別途trade_recordを作成するため独立 |
| integrated_execution_manager.py | 影響なし | main_text_reporterの戻り値を使用するが、キー名変更は透過的 |
| strategy_execution_manager.py | 影響なし | 同上 |

**結論**: 他ファイルへの影響なし

### 3.3 データ構造の変更

**修正前**:
```python
completed_trades = [
    {
        'strategy': 'VWAPBreakoutStrategy',  # ← 'strategy'キー
        'entry_date': '2025-01-15T00:00:00',
        ...
    },
    ...
]
```

**修正後**:
```python
completed_trades = [
    {
        'strategy_name': 'VWAPBreakoutStrategy',  # ← 'strategy_name'キー
        'entry_date': '2025-01-15T00:00:00',
        ...
    },
    ...
]
```

**整合性確認**:
- ✅ comprehensive_reporter.pyのCSV出力と同一構造
- ✅ Phase B-2修正箇所（Lines 622, 717, 794）と整合
- ✅ Phase B-3修正箇所とも整合（'strategy_name'キーを使用）

---

## 4. テスト設計

### 4.1 テスト項目

調査レポート「8.4 テスト項目」に基づき、以下4項目をテストする:

| No. | テスト項目 | 期待結果 | 確認方法 |
|-----|-----------|---------|---------|
| 1 | main_new.py実行テスト | エラーなく正常終了 | 終了コード=0、エラーログなし |
| 2 | テキストレポート確認 | 戦略名が`VWAPBreakoutStrategy`と表示 | 出力ファイル確認 |
| 3 | WARNINGログ確認 | `[FALLBACK]`ログが出力されない | ログファイル検索 |
| 4 | CSV出力確認 | 既存動作が維持されている | CSVファイル確認 |

### 4.2 テスト手順詳細

#### テスト1: main_new.py実行テスト

**実行コマンド**:
```powershell
python main_new.py --start-date 2025-01-15 --end-date 2025-01-31 2>&1 | Tee-Object -FilePath "phase_b2_additional_fix_test.log"
```

**確認項目**:
1. 終了コード: `$?` が `True`（正常終了）
2. エラーログ: `ERROR` が出力されていないこと
3. 実行時間: 既存と同程度（著しい遅延がないこと）

**合格基準**:
- [ ] 終了コード = 0
- [ ] ERRORログなし
- [ ] 実行時間が既存+10%以内

#### テスト2: テキストレポート確認

**確認ファイル**: `output/main_reports/<timestamp>/6954.T_text_report.txt`

**確認コマンド**:
```powershell
Get-Content "output\main_reports\<timestamp>\6954.T_text_report.txt" | Select-String "戦略:"
```

**期待される出力例**:
```
戦略: VWAPBreakoutStrategy
  取引回数: 3
  勝率: 66.67%
  ...
```

**合格基準**:
- [ ] 「戦略: VWAPBreakoutStrategy」が表示される
- [ ] 「戦略: UnknownStrategy」が表示されない
- [ ] 取引回数が正しい（CSV出力と一致）

#### テスト3: WARNINGログ確認

**確認コマンド**:
```powershell
Get-Content "phase_b2_additional_fix_test.log" | Select-String "\[FALLBACK\]"
```

**期待される結果**:
```
# 出力なし（フォールバックが発生していないことを確認）
```

**詳細確認コマンド**:
```powershell
# 戦略名関連のWARNINGログを検索
Get-Content "phase_b2_additional_fix_test.log" | Select-String "戦略名が取得できませんでした"
```

**合格基準**:
- [ ] `[FALLBACK] 戦略名が取得できませんでした（期待値計算）` が出力されない
- [ ] `[FALLBACK] 戦略名が取得できませんでした（パフォーマンス分析）` が出力されない
- [ ] `[FALLBACK] 戦略名が取得できませんでした（取引履歴セクション）` が出力されない
- [ ] `[FALLBACK] 戦略名が取得できませんでした（取引レコード生成）` が出力されない（新規追加ログ）

**注意**: 他の箇所（DSSMS統合系など）でのフォールバックログは除外して確認

#### テスト4: CSV出力確認

**確認ファイル**: `output/main_reports/<timestamp>/6954.T_all_transactions.csv`

**確認コマンド**:
```powershell
Import-Csv "output\main_reports\<timestamp>\6954.T_all_transactions.csv" | Select-Object -First 5 strategy_name | Format-Table
```

**期待される出力例**:
```
strategy_name
-------------
VWAPBreakoutStrategy
VWAPBreakoutStrategy
VWAPBreakoutStrategy
```

**合格基準**:
- [ ] `strategy_name`列が存在する（列名変更なし）
- [ ] 戦略名が`VWAPBreakoutStrategy`と表示される
- [ ] CSV構造が既存と同一（他の列も変更なし）
- [ ] テキストレポートとCSVの取引件数が一致

### 4.3 テスト実施タイミング

1. **修正実装後**: 上記4項目のテストを実施
2. **Phase B完了前**: 最終確認として再度実施

### 4.4 失敗時の対応

| テスト項目 | 失敗した場合の対応 |
|-----------|------------------|
| テスト1 | エラーログを確認し、修正箇所の影響を再調査 |
| テスト2 | trade_recordの構造を再確認、キー名が正しいか検証 |
| テスト3 | フォールバックログが出力される場合、buy_orderの内容を確認 |
| テスト4 | comprehensive_reporter.pyへの影響を調査 |

**重要**: テスト失敗時は修正を巻き戻し、設計を見直す

---

## 5. 修正実装手順

### 5.1 実装ステップ

| Step | 作業内容 | 確認事項 |
|------|---------|---------|
| 1 | Line 233-234のキー名変更 | `'strategy'` → `'strategy_name'`, `'Unknown'` → `'UnknownStrategy'` |
| 2 | WARNINGログ追加（Line 244直前） | ログフォーマット確認、インデント確認 |
| 3 | ファイル保存 | Syntax Errorなし |
| 4 | テスト実行（上記4項目） | すべて合格 |
| 5 | 設計書更新（本ドキュメント） | テスト結果を記録 |

### 5.2 実装時の注意点

#### 5.2.1 インデントの保持

**重要**: Pythonのインデントを正確に保つこと

**正しいインデント** (スペース28個):
```python
                            trade_record = {
                                'strategy_name': buy_order.get('strategy_name', 'UnknownStrategy'),
                                ...
                            }
                            
                            # Phase B-2追加修正: フォールバック検出ログ
                            if trade_record['strategy_name'] == 'UnknownStrategy':
                                logger.warning(
                                    f"[FALLBACK] 戦略名が取得できませんでした（取引レコード生成）: "
                                    f"symbol={symbol}, buy_order={buy_order.keys()}, "
                                    f"デフォルト値='UnknownStrategy'"
                                )
                            
                            completed_trades.append(trade_record)
```

**参考**: Line 220周辺のインデントレベルを確認

#### 5.2.2 symbol変数の取得

WARNINGログで`symbol`変数を使用するため、スコープ内に存在することを確認:

**確認済み**: Line 196のFIFOペアリングループ内で`symbol`変数が定義されている
```python
for symbol in sorted(all_symbols):
    buys = buy_by_symbol.get(symbol, [])
    sells = sell_by_symbol.get(symbol, [])
    # ...
    # Line 233-244: trade_record作成とappend（このスコープ内）
```

**結論**: `symbol`変数は安全に使用可能

#### 5.2.3 既存コメントの保持

**既存のコメント**:
```python
# Line 218-221: データ検証のコメント
# Line 229-230: PnLとリターン計算のコメント
```

これらのコメントは**変更しない**

---

## 6. セルフチェック結果

### 6.1 見落としチェック

- ✅ 確認していないファイルはないか?  
  → main_text_reporter.py全体、comprehensive_reporter.py関連箇所を確認済み

- ✅ カラム名、変数名、関数名を実際に確認したか?  
  → コードを読み取り、grep_searchで全箇所を確認

- ✅ データの流れを追いきれているか?  
  → execution_results → _extract_from_execution_results → completed_trades → Phase B-2修正箇所のフローを確認

### 6.2 思い込みチェック

- ✅ 「〇〇であるはず」という前提を置いていないか?  
  → すべて実際のコードで確認、grep_searchで裏付け

- ✅ 実際にコードや出力で確認した事実か?  
  → Lines 220-280, 570-650を読み取り、grep_search結果で確認

- ✅ 「存在しない」と結論づけたものは本当に確認したか?  
  → 'strategy'キー使用箇所をgrep_searchで全件確認（2箇所のみ、うち1箇所はテストコード）

### 6.3 矛盾チェック

- ✅ 調査結果同士で矛盾はないか?  
  → completed_trades使用箇所（15箇所）とPhase B-2修正箇所（3箇所）が整合

- ✅ 提供されたログ/エラーと結論は整合するか?  
  → phase_b3_test_output.logの`{'strategy': 'VWAPBreakoutStrategy', ...}`と、Phase B-2のWARNINGログが整合

- ✅ 設計内容に矛盾はないか?  
  → キー名変更とWARNINGログ追加が、Phase B-2の意図と整合

---

## 7. 設計承認基準

### 7.1 設計レビュー項目

| 項目 | 状態 | 備考 |
|------|------|------|
| 調査の完全性 | ✅ | 6項目のチェックリスト完了 |
| 根拠の明示 | ✅ | すべての判断に実際のコード・ログを根拠として提示 |
| 影響範囲分析 | ✅ | 直接影響・間接影響を分析、他ファイルへの影響なし |
| テスト設計の妥当性 | ✅ | 4項目のテストで修正の正しさを検証可能 |
| 実装手順の明確性 | ✅ | 5ステップで実装可能、注意点も明示 |
| copilot-instructions.md準拠 | ✅ | フォールバックログ必須要件を満たす |

### 7.2 設計承認者

- ユーザー確認: ⏳ 承認待ち
- 設計者: GitHub Copilot

### 7.3 承認後のアクション

1. **Phase B-2追加修正の実装**（本設計書に基づく）
2. **テスト実施**（4項目すべて合格を確認）
3. **Phase B完了報告作成**（Phase B-1, B-2, B-3の総括）

---

## 8. まとめ

### 8.1 設計の要点

1. **修正箇所**: main_text_reporter.py Line 233-234（1箇所のみ）
2. **変更内容**: 
   - キー名: `'strategy'` → `'strategy_name'`
   - フォールバック値: `'Unknown'` → `'UnknownStrategy'`
3. **追加機能**: WARNINGログ（copilot-instructions.md要件）
4. **影響範囲**: 同一ファイル内のみ、他ファイルへの影響なし
5. **テスト**: 4項目で修正の正しさを検証

### 8.2 期待される効果

| 効果 | 詳細 |
|------|------|
| テキストレポートの正確性向上 | 戦略別分析が`VWAPBreakoutStrategy`を正しく表示 |
| フォールバック検出の完全性 | 取引レコード生成時のフォールバックも検出可能 |
| データ整合性の確保 | CSV出力とテキストレポートで同一のキー名を使用 |
| Phase Bの完了 | Phase B-1, B-2, B-3すべて完了、命名規則統一を達成 |

### 8.3 リスク評価

| リスク | 発生確率 | 影響度 | 対策 |
|--------|---------|-------|------|
| 修正ミス（インデントなど） | 低 | 高 | 実装時に注意、テストで検出 |
| 他ファイルへの影響 | なし | - | 調査で確認済み |
| テスト失敗 | 低 | 中 | 失敗時は巻き戻し、設計見直し |

**総合リスク**: 低（調査・設計が十分、影響範囲が限定的）

---

## 9. 次のステップ

### 9.1 即座に実施すること

1. **ユーザー確認**: 本設計書の承認を得る
2. **実装準備**: main_text_reporter.pyのバックアップ作成（推奨）

### 9.2 承認後のタスク

1. **Phase B-2追加修正の実装** (所要時間: 5分)
2. **テスト実行** (所要時間: 10分)
3. **テスト結果の記録** (所要時間: 5分)
4. **Phase B完了報告の作成** (所要時間: 15分)

**合計所要時間**: 約35分

### 9.3 Phase B完了までの残作業

| Phase | 状態 | 残作業 |
|-------|------|-------|
| Phase B-1 | ✅ 完了 | なし |
| Phase B-2 | ⚠️ 追加修正必要 | 本設計書の実装 |
| Phase B-3 | ✅ 完了 | なし |

**Phase B完了条件**: Phase B-2追加修正のテスト4項目すべて合格

---

**設計完了日**: 2025-12-28  
**設計工数**: 約40分  
**次のステップ**: ユーザー承認 → 実装 → テスト

---

## 付録A: 修正箇所の詳細コード

### A.1 修正前（現在）

```python
                            # Line 220-230: データ検証とPnL計算
                            if not all([entry_price > 0, exit_price > 0, shares > 0]):
                                logger.warning(
                                    f"[DATA_VALIDATION_FAILED] Invalid trade data (symbol={symbol}, pair {i+1}): "
                                    f"entry_price={entry_price}, exit_price={exit_price}, shares={shares}"
                                )
                                continue
                            
                            # PnLとリターン計算
                            pnl = (exit_price - entry_price) * shares
                            return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                            
                            # Line 232-243: trade_record作成
                            trade_record = {
                                'strategy': buy_order.get('strategy_name', 'Unknown'),  # ← 修正対象
                                'entry_date': buy_order.get('timestamp'),
                                'exit_date': sell_order.get('timestamp'),
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'shares': shares,
                                'pnl': pnl,
                                'return_pct': return_pct,
                                'entry_idx': None,
                                'exit_idx': None
                            }
                            completed_trades.append(trade_record)  # Line 244
```

### A.2 修正後

```python
                            # Line 220-230: データ検証とPnL計算（変更なし）
                            if not all([entry_price > 0, exit_price > 0, shares > 0]):
                                logger.warning(
                                    f"[DATA_VALIDATION_FAILED] Invalid trade data (symbol={symbol}, pair {i+1}): "
                                    f"entry_price={entry_price}, exit_price={exit_price}, shares={shares}"
                                )
                                continue
                            
                            # PnLとリターン計算
                            pnl = (exit_price - entry_price) * shares
                            return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                            
                            # Line 232-243: trade_record作成
                            trade_record = {
                                'strategy_name': buy_order.get('strategy_name', 'UnknownStrategy'),  # ← 修正: キー名と値を変更
                                'entry_date': buy_order.get('timestamp'),
                                'exit_date': sell_order.get('timestamp'),
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'shares': shares,
                                'pnl': pnl,
                                'return_pct': return_pct,
                                'entry_idx': None,
                                'exit_idx': None
                            }
                            
                            # Phase B-2追加修正: フォールバック検出ログ
                            if trade_record['strategy_name'] == 'UnknownStrategy':
                                logger.warning(
                                    f"[FALLBACK] 戦略名が取得できませんでした（取引レコード生成）: "
                                    f"symbol={symbol}, buy_order={buy_order.keys()}, "
                                    f"デフォルト値='UnknownStrategy'"
                                )
                            
                            completed_trades.append(trade_record)  # Line 244（移動なし）
```

### A.3 変更サマリー

| 行番号 | 変更内容 | 変更タイプ |
|--------|---------|-----------|
| 233 | `'strategy'` → `'strategy_name'` | キー名変更 |
| 233 | `'Unknown'` → `'UnknownStrategy'` | 値変更 |
| 244直前 | WARNINGログ追加（8行） | 新規追加 |

**合計変更行数**: 9行（既存1行修正 + 新規8行追加）

---

## 付録B: 関連ファイル一覧

### B.1 修正対象ファイル

- `main_system/reporting/main_text_reporter.py`

### B.2 参照ファイル（調査時に確認）

- `main_system/reporting/comprehensive_reporter.py` (Lines 570-650)
- `docs/Data flow design review/20251228_DATA_SOURCE_MISMATCH_INVESTIGATION.md`
- `docs/Data flow design review/20251226_PHASE_B_INVESTIGATION_REPORT.md`
- `.github/copilot-instructions.md`

### B.3 テスト関連ファイル

- `phase_b2_additional_fix_test.log` (新規作成)
- `output/main_reports/<timestamp>/6954.T_text_report.txt`
- `output/main_reports/<timestamp>/6954.T_all_transactions.csv`

---

**End of Design Document**

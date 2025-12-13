# Git履歴調査報告書: DSSMS強制決済コードの痕跡調査

**調査日時**: 2025-12-11  
**調査者**: GitHub Copilot  
**調査目的**: 過去の強制決済コードの特定と削除時期の確認

---

## 1. 確認項目チェックリスト

### 1.1 実施項目（優先度順）

| 項目 | 状態 | 証拠 |
|------|------|------|
| コミット履歴の取得 | 完了 | git log実行済み、最近20件のコミットハッシュ取得 |
| ForceClose関連コードの検索 | 完了 | 銘柄切替時のForceCloseのみ発見 |
| バックテスト期間終了後の処理検索 | 完了 | 期間終了時の強制決済コード未発見 |
| 2025年10月以降の変更履歴確認 | 完了 | 2ヶ月分のdiff確認 |
| 20251210実行時のCSV内容確認 | 完了 | exit_date=2023-02-03（期間外）を確認 |

---

## 2. 調査結果（証拠付き）

### 2.1 git log結果

#### 証拠1: 最近20件のコミットハッシュ

```
6d0b654f8ef03f711ea2cfbf4ab0a15396aa8aea  (2025-12-11)
c035f77129e43f8bf8849af7ab0a298488602593
45cd99a8991d703fd438143222a892eb78c2e6d5
a32b48cbfd3c3e16e475d983b89c7e6622a18354
dc8c06b8cead980fc32fc7216b01a7bf81a3014f
ec60722226dcec9750eb4f699b279e3de3e28274
cdfc3f9915d11b32ff7ba3c7b3b3aa5d61402e96
65d12c8387236aaadaacd64aed595f1418a2ff8e
bcb8d9546193f4bd5401388859d13f2ce187b61d
93daafe50b920e4c8384f197eaa013e2c5784855
34fb1902b258f14fe25e021e145863ea0a955ddb
ae6b65dcd4b3a169abbb4f020d426b3b6d38b31a
06697eca6d3f5dcb60aff730d6b3fe3bd3301fcd
a852b832d819e9e4650d9e0263baf20c3223ecbe
dcf90ab1f9db9c655f1fbe3b8adaa1b6eb072291
bf014ac020cd4f3bbee18b9fe82a5fafc55dd540
225a713b57c875b0da421c55b47c352a1c062b5b
e92fb22b1103b9c580e7a67f62d6ec91003f8125
f50ebccfd55ed602fc02e2055ef93016812990b5
9fb18ee2a128a3d7ccb1eebcaaeeb6d6a497f66d
```

**根拠**: `git log --all --format=%H -- src/dssms/dssms_integrated_main.py` 実行結果

---

### 2.2 ForceClose関連コードの検索結果

#### 証拠2: 銘柄切替時のForceCloseコード（発見）

**検索パターン**: `force.*close|ForceClose|period.*end|backtest.*end|while current_date.*end_date`

**発見された変更**（2025-11-27以降）:

1. **force_close_in_progressフラグの追加**
```python
+            self.force_close_in_progress = False  # [Task11] DSSMS用ForceCloseフラグ
```

2. **銘柄切替時のForceClose実装**
```python
if should_switch:
    if self.current_symbol and self.position_size > 0:
        # [Task11] ForceCloseフラグ設定
+       self.force_close_in_progress = True
+       self.logger.info(f"[DSSMS_FORCE_CLOSE_START] ForceClose開始、戦略SELL処理を抑制")
        close_result = self._close_position(self.current_symbol, target_date)
+       # [Task11] ForceCloseフラグリセット
+       self.force_close_in_progress = False
+       self.logger.info(f"[DSSMS_FORCE_CLOSE_END] ForceClose完了、戦略SELL処理を再開")
```

3. **_close_positionメソッドの強化**
```python
+       'strategy_name': 'DSSMS_SymbolSwitch',  # ForceCloseと区別するための戦略名
        'status': 'executed',
        'entry_price': entry_price,
        'profit_pct': price_change_rate * 100,
        'close_return': close_return
```

**判明したこと1**:
- 銘柄切替時のForceCloseは**2025年11月末～12月初旬に実装**された
- `force_close_in_progress`フラグを使用
- `_close_position`メソッドを呼び出してexecution_detailを生成
- 根拠: git diff結果

---

#### 証拠3: バックテスト期間終了後の処理（未発見）

**検索パターン**: `position_size > 0.*final|final.*position|end.*position|backtest.*complete.*position`

**検索範囲**: 2025-10-01 ～ 2025-12-11（過去2ヶ月）

**検索結果**: **該当コードなし**

**発見された処理**:
1. `_convert_to_execution_format`メソッドの実装
2. `_generate_outputs`メソッドの実装
3. Excel出力の削除（2025-10-08）

**判明したこと2**:
- バックテスト期間終了後の強制決済コードは**過去2ヶ月のコミット履歴に存在しない**
- ループ終了後は直接`_generate_final_results`に進む
- 根拠: git diff結果（該当コードなし）

---

### 2.3 バックテストループ終了後の処理

#### 証拠4: 現在のコード構造

**ファイル**: `src/dssms/dssms_integrated_main.py` (現在のコード)

```python
while current_date <= end_date:
    # 日次処理
    daily_result = self._process_daily_trading(current_date, target_symbols)
    self.daily_results.append(daily_result)
    current_date += timedelta(days=1)

# ループ終了後、直接最終結果生成
total_execution_time = time.time() - execution_start
final_results = self._generate_final_results(total_execution_time, trading_days, successful_days)

# エクスポート・レポート生成
self._generate_outputs(final_results)
```

**判明したこと3**:
- ループ終了（`current_date += timedelta(days=1)`）から`_generate_final_results`までの間に**強制決済処理が存在しない**
- `self.position_size > 0`チェックなし
- `_close_position`呼び出しなし
- 根拠: 現在のコード確認

---

### 2.4 20251210実行時の証拠

#### 証拠5: dssms_trades.csvの内容

**ファイル**: `output/dssms_integration/dssms_20251210_220159/dssms_trades.csv`

```csv
entry_date,exit_date,entry_price,exit_price,shares,pnl,return_pct,holding_period_days,strategy,position_value,is_forced_exit,is_executed_trade
2023-01-24T00:00:00+09:00,2023-02-03T00:00:00+09:00,4064.789390176107,4062.9049084408693,200,-376.89634704756827,-0.0004636111626822063,10,BreakoutStrategy,812957.8780352215,True,True
```

**重要な発見**:
- **entry_date**: 2023-01-24（バックテスト期間内）
- **exit_date**: 2023-02-03（バックテスト期間外）
- **is_forced_exit**: True
- **strategy**: BreakoutStrategy（DSSMS_SymbolSwitchではない）

**判明したこと4**:
- 20251210実行時には、バックテスト期間（2023-01-31まで）終了後、**2023-02-03に強制決済が実行された**
- これは**バックテスト期間外での強制決済**
- strategyがBreakoutStrategyであることから、銘柄切替ForceCloseとは**別の仕組み**
- 根拠: dssms_trades.csv実ファイル確認

---

## 3. セルフチェック

### 3.1 見落としチェック

| 項目 | 確認内容 | 状態 | 根拠 |
|------|---------|------|------|
| コミット履歴の網羅性 | 過去2ヶ月確認 | 完了 | git log実行済み |
| 検索パターンの妥当性 | 複数パターン使用 | 完了 | force/ForceClose/period/backtest/while等 |
| ループ終了後処理の確認 | current_date += timedelta以降確認 | 完了 | コード確認済み |
| 20251210CSVの詳細確認 | カラム値全確認 | 完了 | 実ファイル確認済み |
| 古いコミットの確認 | 2025-10-01以降確認 | 完了 | 2ヶ月分検索済み |

**結論**: 主要な確認項目は網羅済み

---

### 3.2 思い込みチェック

| 項目 | 当初の想定 | 実際の確認結果 | 判定 |
|------|-----------|--------------|------|
| 過去に強制決済コードが存在 | 存在していた | 現在のgit履歴に痕跡なし | 不明 |
| 20251210で動作していた | 動作していた | CSVで証拠確認 | 事実 |
| コミット履歴に削除記録あり | あるはず | 2ヶ月間の履歴に該当なし | 予想外 |
| exit_dateが期間外 | ありえない | 2023-02-03で事実 | 予想外 |

**結論**: 予想外の事実（exit_dateが期間外）を発見

---

### 3.3 矛盾チェック

| 矛盾候補 | 検証結果 | 解決 |
|---------|---------|------|
| 過去に動作 vs git履歴に痕跡なし | 両方事実 | 不明（2ヶ月以上前に削除された可能性） |
| exit_date期間外 vs バックテスト期間内 | 両方事実 | 過去の仕組みは期間外で決済していた |
| strategy=BreakoutStrategy vs ForceClose | 両方事実 | 銘柄切替とは別の強制決済仕組み |

**結論**: exit_dateが期間外である理由は不明（要追加調査）

---

## 4. 調査結果まとめ

### 4.1 判明したこと（証拠付き）

#### A. 銘柄切替時のForceClose（実装済み）

証拠: git diff結果

**実装時期**: 2025年11月末～12月初旬

**実装内容**:
- `force_close_in_progress`フラグ
- 銘柄切替時の`_close_position`呼び出し
- execution_detail生成（strategy_name='DSSMS_SymbolSwitch'）

**状態**: 現在も稼働中

---

#### B. バックテスト期間終了時の強制決済（未実装）

証拠: git diff結果（該当コードなし）、現在のコード確認

**検索範囲**: 2025-10-01 ～ 2025-12-11

**検索結果**: **該当コードなし**

**状態**: 過去2ヶ月のコミット履歴に存在しない

---

#### C. 20251210実行時の強制決済（動作していた）

証拠: dssms_trades.csv実ファイル

**証拠内容**:
- entry_date: 2023-01-24
- exit_date: **2023-02-03（期間外）**
- is_forced_exit: True
- strategy: BreakoutStrategy

**重要な発見**:
- バックテスト期間（～2023-01-31）終了後、**2023-02-03に決済**
- 期間外での決済が実行されていた
- strategyがBreakoutStrategyであることから、銘柄切替ForceCloseとは別の仕組み

**状態**: 20251210実行時には動作していた

---

### 4.2 不明な点

#### 不明点1: 過去の強制決済コードの場所

**問題**: git履歴（過去2ヶ月）に削除記録が見つからない

**可能性**:
1. **2025年10月以前に削除された**
   - 2ヶ月以上前の変更
   - より古いコミットの確認が必要

2. **別のファイルに実装されていた**
   - main_new.pyでの実装
   - ComprehensiveReporterでの実装
   - 他のモジュールでの実装

3. **コメントアウトまたは条件分岐で無効化**
   - コードは残存しているが、実行されない状態
   - フラグや設定で制御されている

**追加調査が必要**:
- 2025年9月以前のgit log確認
- main_new.pyの変更履歴確認
- ComprehensiveReporterの変更履歴確認

---

#### 不明点2: exit_date=2023-02-03の意味

**問題**: バックテスト期間（～2023-01-31）外で決済が記録されている

**可能性**:
1. **期間外でのバックテスト継続**
   - `while current_date <= end_date`の条件が異なっていた
   - または、ループ終了後に追加の日次処理を実行

2. **ComprehensiveReporterでの強制決済**
   - レポート生成時に強制決済を追加
   - execution_detailsに後付けでSELLを追加

3. **main_new.pyでの強制決済**
   - DSSMS本体ではなく、main_new.pyが決済を実行
   - 価格データは別ルートで取得

**追加調査が必要**:
- 20251210実行時のログファイル（存在すれば）
- ComprehensiveReporterの過去の実装
- main_new.pyの過去の実装

---

#### 不明点3: strategy=BreakoutStrategyの意味

**問題**: 強制決済なのにstrategyがBreakoutStrategy

**可能性**:
1. **元のBUY戦略を保持**
   - entry時の戦略名をそのまま使用
   - 強制決済時も変更しない

2. **過去のexecution_detail構造が異なる**
   - strategy_name='ForceClose'の設定がなかった
   - is_forced_exit=Trueのみで判定

3. **ComprehensiveReporterの推測ロジック**
   - レポート生成時に戦略名を推測
   - BUYのstrategy_nameを引き継いだ

**追加調査が必要**:
- _close_positionメソッドの過去の実装
- ComprehensiveReporterの戦略名設定ロジック

---

### 4.3 原因の推定（可能性順）

#### 推定A: 2025年10月以前に削除された（可能性: 高）

**根拠**:
- 過去2ヶ月のgit履歴に削除記録なし
- 20251210実行時には動作していた
- 2025年10月～11月の間に削除された可能性

**確認方法**:
```bash
git log --since="2025-09-01" --until="2025-10-31" --all -p -- src/dssms/dssms_integrated_main.py | Select-String -Pattern "position_size > 0" -Context 10
```

---

#### 推定B: 別のモジュールに実装されていた（可能性: 中）

**根拠**:
- exit_date=2023-02-03（期間外）
- DSSMS本体のループは2023-01-31で終了
- 別の場所で追加処理が実行された可能性

**確認方法**:
- main_new.pyのgit log確認
- ComprehensiveReporterのgit log確認

---

#### 推定C: 条件分岐で無効化された（可能性: 低）

**根拠**:
- 現在のコードに該当処理が見当たらない
- コメントアウトも見当たらない

**確認方法**:
- 全ファイル検索で「position_size > 0」「final」「backtest complete」などを検索

---

## 5. 結論

### 5.1 調査結果の確定事項

1. 銘柄切替時のForceCloseは**2025年11月末～12月初旬に実装**
2. バックテスト期間終了時の強制決済は**過去2ヶ月のgit履歴に存在しない**
3. 20251210実行時には**バックテスト期間外（2023-02-03）で強制決済が実行された**
4. 20251210のdssms_trades.csvには**is_forced_exit=Trueの取引記録がある**

---

### 5.2 未確定事項（追加調査が必要）

1. 過去の強制決済コードはどこにあったのか？（DSSMS本体? main_new.py? ComprehensiveReporter?）
2. いつ削除されたのか？（2025年10月以前? それ以前?）
3. なぜexit_dateが期間外（2023-02-03）なのか？
4. なぜstrategyがBreakoutStrategyなのか？（強制決済なのに）

---

### 5.3 推奨される次のアクション

#### 優先度A（必須）: より古いgit履歴の確認

**目的**: 2025年9月以前の削除記録を探す

**コマンド例**:
```bash
# 2025年9月のコミット確認
git log --since="2025-09-01" --until="2025-09-30" --all -p -- src/dssms/dssms_integrated_main.py | Select-String -Pattern "position_size|final|complete" -Context 10

# さらに古い履歴（8月）
git log --since="2025-08-01" --until="2025-08-31" --all -p -- src/dssms/dssms_integrated_main.py | Select-String -Pattern "force.*close|position_size > 0" -Context 10
```

---

#### 優先度B（推奨）: 他のモジュールの履歴確認

**目的**: main_new.pyやComprehensiveReporterでの実装を探す

**コマンド例**:
```bash
# main_new.pyの確認
git log --since="2025-10-01" --all -p -- main_new.py | Select-String -Pattern "DSSMS.*force|backtest.*end|position" -Context 5

# ComprehensiveReporterの確認
git log --since="2025-10-01" --all -p -- main_system/reporting/comprehensive_reporter.py | Select-String -Pattern "force.*close|is_forced_exit" -Context 5
```

---

#### 優先度C（参考）: 20251210実行時のログ探索

**目的**: 当時の実行ログがあれば、動作していたコードを特定できる

**確認場所**:
- `logs/dssms_*.log`
- `output/dssms_integration/dssms_20251210_220159/`内のログファイル

---

## 6. 補足情報

### 6.1 調査に使用したコマンド

```powershell
# コミットハッシュ取得
git log --all --format=%H -- src/dssms/dssms_integrated_main.py | Select-Object -First 20

# 最新コミットの日付確認
git show --no-patch --format="%H%n%ad%n%s" --date=short 6d0b654f8ef03f711ea2cfbf4ab0a15396aa8aea

# ForceClose関連の変更検索（2025-11-27以降）
git log --since="2025-11-27" --until="2025-12-11" --all -p -- src/dssms/dssms_integrated_main.py | Select-String -Pattern "force.*close|ForceClose|period.*end|backtest.*end|while current_date.*end_date" -Context 3

# ループ終了後の処理検索
git log --since="2025-11-01" --until="2025-12-11" --all -p -- src/dssms/dssms_integrated_main.py | Select-String -Pattern "current_date \+= timedelta|final_results|_generate_final_results|while.*current_date.*<=" -Context 5

# position_size関連の検索（2025-10-01以降）
git log --since="2025-10-01" --until="2025-12-11" --all -p -- src/dssms/dssms_integrated_main.py | Select-String -Pattern "position_size > 0.*final|final.*position|end.*position|backtest.*complete.*position" -Context 3
```

---

## 7. 重要な発見のまとめ

### 発見1: exit_date=2023-02-03の謎

20251210のdssms_trades.csvには、バックテスト期間（2023-01-15～2023-01-31）**外**のexit_dateが記録されている。

**意味**:
- 過去の強制決済は、バックテスト期間終了後も処理を継続していた
- または、期間外の価格データを取得して決済していた
- または、レポート生成時に後付けで決済を追加していた

**重要性**:
この発見により、過去の実装方法が現在とは**大きく異なる**可能性が高い。

---

### 発見2: strategy=BreakoutStrategyの意味

強制決済なのに、strategyがBreakoutStrategyになっている。

**意味**:
- 過去の_close_positionは、strategy_name='ForceClose'を設定していなかった
- または、元のBUY戦略名を保持する設計だった
- または、ComprehensiveReporterが戦略名を推測していた

**重要性**:
現在の実装（strategy_name='DSSMS_SymbolSwitch'）とは異なる設計思想。

---

### 発見3: git履歴に痕跡がない

過去2ヶ月のgit履歴に、バックテスト期間終了時の強制決済コードが見つからない。

**意味**:
- 2025年10月以前に削除された可能性
- または、最初から別のモジュールに実装されていた
- または、条件分岐で無効化されている

**重要性**:
さらに古いgit履歴の確認が必要。

---

**調査完了 - 部分的に特定、追加調査が必要**

**次のステップ**: 2025年9月以前のgit log確認 → 他のモジュールの履歴確認 → 過去のコード復元

# 既知の問題カタログと予防策

**目的**: プロジェクトで発生した重大な問題を記録し、再発を防止する  
**最終更新**: 2026-02-13

---

## 使用方法

### 新しいIssue追加時
1. Issue IDを割り当て（ISSUE-XXX形式）
2. 以下の情報を記録
   - 発生時期、深刻度、症状、原因、解決策、予防策
3. 修正完了後、ステータスを「解決済み」に更新

### 参照時
- 新機能実装前: 関連Issueをチェック、予防策を適用
- デバッグ時: 類似症状のIssueを検索

---

## Issue一覧

| Issue ID | タイトル | 深刻度 | ステータス | 発生日 | 修正日 |
|----------|---------|--------|-----------|--------|--------|
| ISSUE-010 | AI虚偽報告問題（VSCode Copilot） | P0-Critical | 予防策確立済み | 2026-02-11 | 2026-02-11 |
| ISSUE-009 | 複数銘柄保有対応コードの意図しない削除 | P0-Critical | 調査完了（復旧待ち） | 2026-02-10 | 2026-02-11調査 |
| ISSUE-007 | BUY/SELL後のself.positions管理漏れ | P0-Critical | 解決済み | Sprint 2 | 2026-02-10 |
| ISSUE-008 | ウォームアップ期間エントリー問題 | P1-High | 解決済み（完全） | 2026-02-10調査 | 2026-02-10 |

---

## Issue詳細

### Issue #7: BUY/SELL後のself.positions管理漏れ

**問題ID**: ISSUE-007  
**発生時期**: Sprint 2マルチポジション対応実装時  
**深刻度**: P0-Critical  
**ステータス**: 解決済み（2026-02-10修正完了）

#### 症状

- all_transactions.csvにEXIT情報（exit_date, exit_price, pnl）が記録されない
- 強制決済が実行されない（`[FINAL_CLOSE]`ログ不在）
- バックテスト結果が無効（総収益率等の統計が計算されない）

#### 原因

Sprint 2マルチポジション対応時に、BUY/SELL処理後の`self.positions`管理が実装漏れ。

**問題のあるコード**:
```python
# src/dssms/dssms_integrated_main.py (修正前)
if result['action'] == 'buy':
    self.cash_balance -= trade_cost
    # ❌ self.positions[symbol] = {...} の処理が存在しない

elif result['action'] == 'sell':
    self.cash_balance += trade_profit
    # ❌ del self.positions[symbol] の処理が存在しない
```

#### 影響の連鎖

1. BUY注文実行 → `execution_details`に記録 ✅
2. `self.positions`に追加されない ❌
3. バックテスト終了時: `len(self.positions) == 0`
4. 強制決済の条件 `if len(self.positions) > 0:` が**False**
5. 強制決済処理がスキップ ❌
6. SELL注文が`execution_details`に追加されない ❌
7. all_transactions.csvに未決済BUY注文のみ記録 ❌

#### 解決策

**修正箇所**: `src/dssms/dssms_integrated_main.py`

**修正1: BUY処理にポジション追加（Line 2607付近）**
```python
if result['action'] == 'buy':
    trade_cost = result['price'] * result['shares']
    self.cash_balance -= trade_cost
    
    # Sprint 2修正: BUY実行後にself.positionsに追加（強制決済対応）
    self.positions[symbol] = {
        'symbol': symbol,
        'strategy': best_strategy_name,
        'entry_price': result['price'],
        'shares': result['shares'],
        'entry_date': adjusted_target_date,
        'entry_idx': None,
    }
    
    self.logger.info(
        f"[POSITION_ADD] ポジション追加: {symbol}, 価格={result['price']:.2f}円, "
        f"株数={result['shares']}株, 戦略={best_strategy_name}"
    )
```

**修正2: SELL処理にポジション削除（Line 2635付近）**
```python
elif result['action'] == 'sell':
    trade_profit = result['price'] * result['shares']
    self.cash_balance += trade_profit
    
    # Sprint 2修正: SELL実行後にself.positionsから削除
    if symbol in self.positions:
        position_entry_price = self.positions[symbol]['entry_price']
        position_shares = self.positions[symbol]['shares']
        pnl = (result['price'] - position_entry_price) * position_shares
        return_pct = (result['price'] - position_entry_price) / position_entry_price if position_entry_price > 0 else 0.0
        
        del self.positions[symbol]
        
        self.logger.info(
            f"[POSITION_DELETE] ポジション削除: {symbol}, PnL={pnl:+,.0f}円({return_pct:+.2%})"
        )
    else:
        self.logger.warning(
            f"[POSITION_DELETE] 警告: {symbol}がself.positionsに存在しません（SELL実行されたがポジション未記録）"
        )
```

#### 予防策

##### 設計段階

**1. BUY/SELL処理の実装チェックリスト**

BUY処理実装時:
- [ ] `self.cash_balance`更新
- [ ] `self.positions`追加
- [ ] `execution_details`記録
- [ ] ログ出力（`[POSITION_ADD]`）

SELL処理実装時:
- [ ] `self.cash_balance`更新
- [ ] `self.positions`削除（KeyErrorチェック実装）
- [ ] `execution_details`記録
- [ ] ログ出力（`[POSITION_DELETE]`）

**2. ポジション管理の一元化**

BUY/SELL処理を専用のメソッドに分離することを推奨：
```python
def _execute_buy(self, symbol, price, shares, strategy_name, target_date):
    """BUY注文実行（cash_balance更新、positions追加、ログ記録を一括処理）"""
    trade_cost = price * shares
    self.cash_balance -= trade_cost
    
    self.positions[symbol] = {
        'symbol': symbol,
        'strategy': strategy_name,
        'entry_price': price,
        'shares': shares,
        'entry_date': target_date,
        'entry_idx': None,
    }
    
    self.logger.info(f"[POSITION_ADD] ポジション追加: {symbol}")
    return trade_cost

def _execute_sell(self, symbol, price, shares):
    """SELL注文実行（cash_balance更新、positions削除、ログ記録を一括処理）"""
    trade_profit = price * shares
    self.cash_balance += trade_profit
    
    if symbol in self.positions:
        del self.positions[symbol]
        self.logger.info(f"[POSITION_DELETE] ポジション削除: {symbol}")
    else:
        self.logger.warning(f"[POSITION_DELETE] 警告: {symbol}が存在しません")
    
    return trade_profit
```

##### 実装時のレビューポイント

- [ ] BUY処理に`self.positions`追加実装
- [ ] SELL処理に`self.positions`削除実装
- [ ] KeyError防止の存在チェック（`if symbol in self.positions:`）
- [ ] ログ出力（`[POSITION_ADD]`, `[POSITION_DELETE]`）
- [ ] 強制決済の動作確認（`[FINAL_CLOSE]`ログ）

##### 検証方法

**テスト1: ポジション追加確認**
```bash
# ログで確認
grep "\[POSITION_ADD\]" output/dssms_integration/dssms_*/dssms_execution_log.txt
```

**テスト2: ポジション削除確認**
```bash
# ログで確認
grep "\[POSITION_DELETE\]" output/dssms_integration/dssms_*/dssms_execution_log.txt
```

**テスト3: 強制決済確認**
```bash
# ログで確認
grep "\[FINAL_CLOSE\]" output/dssms_integration/dssms_*/dssms_execution_log.txt
```

**テスト4: all_transactions.csv確認**
```python
import pandas as pd
df = pd.read_csv("output/dssms_integration/dssms_*/all_transactions.csv")
assert df['exit_date'].notna().all(), "exit_date に空の行があります"
assert df['exit_price'].notna().all(), "exit_price に空の行があります"
assert df['pnl'].notna().all(), "pnl に空の行があります"
print("✅ all_transactions.csv検証成功")
```

#### 修正完了日
2026-02-10

#### 修正者
プロジェクトチーム

#### 参考資料
- [DSSMS EXIT未記録問題調査報告](docs/DSSMS_EXIT_NOT_RECORDED_INVESTIGATION_20260210.md)
- 修正コミット: src/dssms/dssms_integrated_main.py Line 2600-2658

#### Sprint 2ギャップ分析

**分析日**: 2026-02-10  
**対象**: Sprint 2マルチポジション実装（docs/SPRINT2_MULTI_POSITION_COMPLETION_REPORT.md）

**Q1: self.positions管理はSprint 2実装計画で設計されていたか？**

**回答**: 部分的に設計されていた（初期化のみ、BUY/SELL更新処理は設計漏れ）

Sprint 2完了レポートでの言及:
- ✅ `self.positions = {}`の初期化は設計済み（Line 209-211）
- ✅ `self.max_positions`チェックは実装済み（Line 2356-2374）
- ✅ FIFO決済ロジックは実装済み（Line 1959-2011）
- ❌ **BUY実行後の`self.positions`追加処理は記載なし**
- ❌ **SELL実行後の`self.positions`削除処理は記載なし**

**Q2: self.positions管理はSprint 2で検証されたか？**

**回答**: いいえ（検証項目に含まれていない）

Sprint 2完了レポートの検証内容:
- ✅ 複数銘柄保有の結果は確認済み（実行ログexample.ymlでの複数銘柄保有）
- ✅ FIFO決済の動作は確認済み（古いポジションから決済される動作）
- ❌ **`self.positions`辞書の内容は検証されていない**
- ❌ **BUY/SELL実行後のポジション数確認は検証項目に含まれていない**

**Q3: なぜ実装漏れが発生したか？**

**根本原因の分析**:

1. **設計の粒度不足**:
   - Sprint 2設計は「枠組み」（self.positionsの初期化）にフォーカス
   - 「状態更新」（BUY/SELL時のpositions追加/削除）は設計から漏れた
   - マルチポジション対応の「構造」に注力し、「操作」が抜け落ちた

2. **実装チェックリストの不在**:
   - BUY/SELL実装時の詳細チェックリストが存在しない
   - grep検索結果: "チェックリスト" → 0件（実装ガイドなし）
   - Sprint 2完了レポートにBUY/SELL処理の実装詳細なし（Line 1041にside='buy'の条件式のみ）

3. **検証項目の不足**:
   - 結果（複数銘柄保有、FIFO決済）の動作は確認
   - 内部状態（self.positionsの正確性）は検証対象外
   - 強制決済の動作も検証項目に含まれていない

4. **状態管理の複雑さ**:
   - Sprint 1.5: `self.current_position`（単一ポジション）
   - Sprint 2: `self.positions = {}`（複数ポジション辞書）
   - 状態管理の抽象度が上がったが、実装ガイドが追従していない

**Q4: 再発防止のために何を改善すべきか？**

**改善提案**:

1. **設計テンプレートの整備**:
   - 「状態管理の変更」を含む機能は、以下を設計に含める:
     - 初期化（Initialization）
     - 状態更新（State Update）: どこで、いつ、どのように更新するか
     - 状態確認（State Verification）: どのように検証するか
   - 実装例: "BUY処理: self.cash_balance更新 + self.positions追加 + execution_details記録"

2. **詳細な実装チェックリストの追加**:
   - 本Issue #7の「予防策」セクションに記載したBUY/SELL実装チェックリストを標準化
   - 新規機能実装時の必須チェック項目として使用
   - copilot-instructions.mdに参照を追加

3. **検証スクリプトの整備**:
   - 状態管理の正確性を検証する自動スクリプト作成
   - 実装例: `verify_positions_integrity.py`（self.positionsとexecution_detailsの整合性チェック）
   - テスト実行時に自動実行される

4. **Definition of Done (DoD)の明確化**:
   - 設計完了の定義: 初期化・状態更新・検証方法まで記載
   - 実装完了の定義: チェックリスト全項目クリア + ログ確認
   - 検証完了の定義: 結果検証 + 内部状態検証 + 自動テスト成功

**参考資料**:
- Sprint 2完了レポート: docs/SPRINT2_MULTI_POSITION_COMPLETION_REPORT.md
- 詳細なギャップ分析: docs/DSSMS_EXIT_NOT_RECORDED_INVESTIGATION_20260210.md（Sprint 2ギャップ分析セクション）

#### Git履歴による実装履歴の調査

**調査日**: 2026-02-10  
**目的**: いつ、なぜpositions管理が削除/実装漏れとなったかを特定

**発見事項**:

**1. コミットd84cd6d（2025年12月19日）: positions管理の意図的削除**
```
コミットメッセージ:
"色々あって DSSMSで銘柄切替の時にポジション決済してたのでそのコードを削除して
二重になってた計算を正しくした ポジションの決済はmain_mew.pyに任せる方向を再度確認
しかし、まだたくさん修正ポイントがある"

変更統計: src/dssms/dssms_integrated_main.py
- 追加: +3,474行（新機能: force_close_strategy.py等）
- 削除: -417行（DSSMSのpositions管理コードを削除）
```

**設計判断の背景**:
- DSSMSとmain_new.pyでポジション決済が**二重に実装**されていた
- 責務を分離するため、ポジション管理を**main_new.pyに集約**する設計に変更
- DSSMSは銘柄選択と戦略実行のみに専念する方向性

**2. コミット5147549（2026年02月10日）: Sprint 2でのpositions管理再実装**
```
コミットメッセージ:
"Sprint 2完了: 複数銘柄保有対応 (max_positions=2)
【主要な変更】
- self.current_position → self.positions辞書に変更
- self.max_positions = 2 追加
- FIFO決済ロジック実装
(後略)"

変更統計: src/dssms/dssms_integrated_main.py
- 変更: 520行（複数銘柄対応への大規模リファクタリング）
```

**実装内容の詳細**:
- ✅ `self.positions = {}`の初期化（`__init__`メソッド）
- ✅ `if len(self.positions) < self.max_positions:`チェック（エントリー制御）
- ✅ `if symbol in self.positions:`による参照処理（複数箇所）
- ✅ FIFO決済ロジック実装（`_evaluate_and_execute_switch`メソッド）
- ❌ **BUY実行後の`self.positions[symbol] = {...}`追加処理が実装漏れ**
- ❌ **SELL実行後の`del self.positions[symbol]`削除処理が実装漏れ**

**3. 実装漏れの根本原因**

**設計方針の変更が未完了**:
- 2025年12月19日: DSSMSからpositions管理を削除（main_new.pyに移譲）
- 2026年02月10日: Sprint 2でDSSMSにpositions管理を再実装
- **設計方針が変更されたにも関わらず、削除されたコードの全体像が把握されていなかった**
- コミットd84cd6dで削除された417行の中に、BUY/SELL時のpositions更新処理が含まれていた可能性が高い

**設計ドキュメントの不足**:
- コミットd84cd6dのコミットメッセージは設計判断を記録
- しかし、削除されたコード（417行）の詳細な機能リストが記録されていない
- Sprint 2実装時に、削除されたコードの復元が不完全になった

**過去のpositions管理実装の参照不足**:
- Sprint 2実装時に、コミットd84cd6d以前のpositions管理実装を参照していない
- `git show d84cd6d^:src/dssms/dssms_integrated_main.py`で削除前のコードを確認すべきだった
- 過去の実装をリファクタリング・再実装する際の手順が確立されていない

**4. 教訓と今後の対策**

**大規模なコード削除時の記録必須**:
- 削除するコードの機能リスト作成（削除前）
- 削除理由と影響範囲の文書化（docs/design/）
- 将来の再実装に備えた参考資料の作成

**設計方針変更時のチェックリスト**:
- [ ] 変更前のコードの機能を網羅的にリストアップ
- [ ] 変更後の設計で、すべての機能をカバーできるか確認
- [ ] 削除される機能の移行先を明確に文書化
- [ ] 移行先での実装完了を検証

**リファクタリング時のGit履歴活用**:
- 過去の実装を削除・再実装する際は、削除コミットの差分を必ず確認
- `git show <commit-hash>^:<file>`で削除前のコードを参照
- 削除されたコードのうち、再実装が必要な部分を特定

**コミットメッセージの改善**:
- 大規模な削除（100行以上）は、削除理由と影響範囲を詳細に記載
- 設計方針変更は、docs/design/に設計判断記録を作成し、コミットメッセージから参照

**Git履歴調査コマンド（参考）**:
```bash
# コミット履歴検索
git log --all --oneline --grep="position" --grep="Sprint 2" --grep="multi" -i

# ファイルの変更履歴
git log --oneline <file>

# 特定のコミットの統計
git show <commit-hash> --stat

# 削除前のコードを確認
git show <commit-hash>^:<file>

# 差分検索
git diff <commit1>..<commit2> <file> | Select-String -Pattern "<pattern>"
```

---

## 再発防止のベストプラクティス

### 1. 状態管理の原則
- **状態更新は一箇所で**: `self.positions`の追加/削除はBUY/SELLメソッド内で完結
- **不整合を許さない**: 現金残高とポジション管理は必ずセットで更新
- **ログは必須**: 状態変更時は必ずログ出力（デバッグの重要な手がかり）

### 2. 実装時のチェックリスト活用
- 新しいBUY/SELL処理実装時は、必ず「BUY/SELL処理の実装チェックリスト」を確認
- レビュー時も同チェックリストを使用

### 3. 自動テストの整備
- ポジション管理の正確性を検証する自動テストを作成
- バックテスト実行後、all_transactions.csvの完全性を自動チェック

---

### Issue #9: 複数銘柄保有対応コードの意図しない削除

**問題ID**: ISSUE-009  
**発生時期**: 2026-02-10（コミットeae6ce3）  
**深刻度**: P0-Critical  
**ステータス**: 調査完了（復旧待ち）

#### 症状

- `self.max_positions` が存在しない
- `self.positions` 辞書が `self.current_position` に戻っている
- 複数銘柄保有対応の実装が失われている（約520行の変更が巻き戻り）
- FIFOロジックが削除されている

#### 原因

**コミット5147549** (2026-02-10 10:46:26):
- Sprint 2で複数銘柄保有対応を実装（520行の変更）
- テスト結果も良好

**コミットeae6ce3** (2026-02-10 22:09:43):
- Issue #8（ウォームアップ期間フィルタリング）対応中
- **意図せず複数銘柄対応コードを削除**
- 可能性:
  1. 古いバージョンのファイルを編集してコミット
  2. マージ競合の解決ミス
  3. copilot/AIによる自動修正の誤り

#### 影響の連鎖

1. **5147549で実装** (10:46):
   - `self.positions = {}`
   - `self.max_positions = 2`
   - FIFO決済ロジック
   - 4ケース分岐（複数銘柄管理）

2. **eae6ce3で削除** (22:09、約12時間後):
   - `self.positions` → `self.current_position`
   - `self.max_positions` → 削除
   - FIFO決済 → 単一ポジション処理
   - 4ケース分岐 → 簡易的な切替判定

#### 削除されたコード（主要部分）

**Line 201-209: ポジション管理初期化**
```python
# 削除されたコード
self.positions = {}  # {symbol: {strategy, entry_price, shares, entry_date, entry_idx}}
self.max_positions = 2  # Sprint 2設定: 最大保有銘柄数

# 現在のコード（巻き戻り）
self.current_position = None  # 現在のポジション情報
```

**Line 717-827: 強制決済処理**
```python
# 削除されたコード
if len(self.positions) > 0:
    for symbol, position_data in list(self.positions.items()):
        # 各銘柄を個別に決済
        ...

# 現在のコード（巻き戻り）
if self.current_position:
    symbol = self.current_position['symbol']
    # 単一ポジションのみ決済
    ...
```

**Line 903-1086: FIFO決済ロジック**
```python
# 削除されたコード
# ケース1: 初回エントリー
if len(self.positions) == 0:
    ...
# ケース2: 選択銘柄が既に保有中
if selected_symbol in self.positions:
    ...
# ケース3: max_positions未達
if len(self.positions) < self.max_positions:
    ...
# ケース4: max_positions到達（FIFO決済）
if len(self.positions) >= self.max_positions:
    # 最も古いポジションを決済
    ...

# 現在のコード（簡易版に巻き戻り）
# 複数銘柄対応のロジックが削除されている
```

#### ブランチ構造の分析

```
* eae6ce3 (HEAD -> 複数銘柄対応2) ← 【犯人】複数銘柄コード削除
* d1d467d docs作成
* 5147549 Sprint 2完了: 複数銘柄保有対応 ← 【実装】
* a154798 MomentumInvestingStrategy force_close実装
...
* 7ada5bb 強制決済コード復元
| * 7782dcf (backup-multi-position-attempt) ← 【バックアップブランチ】
| * 02a556e Excel設定問題修正
| * 7c02096 複数銘柄バックテスト時の価格混同バグ修正
```

**重要**: 2つのブランチが存在し、7782dcfブランチでも複数銘柄対応を試みたが、
「負のループになっている」ため巻き戻してバックアップした履歴がある。

#### 解決策

3つのオプションを提示：

**オプションA: コミット5147549に戻す（推奨）**
```bash
# 複数銘柄対応コードのみを復元
git checkout 5147549 -- src/dssms/dssms_integrated_main.py

# ウォームアップ期間フィルタリング（eae6ce3の変更）を手動で再適用
# （Issue #8対応は失われないようにする）
```

**オプションB: 差分を手動でマージ**
```bash
# 5147549とeae6ce3の差分を確認
git diff 5147549 eae6ce3 -- src/dssms/dssms_integrated_main.py

# 必要な部分のみを選択的に復元
```

**オプションC新規ブランチで復旧作業**
```bash
# 安全のため新規ブランチ作成
git checkout -b fix/restore-multi-position

# 5147549から開始
git checkout 5147549 -- src/dssms/dssms_integrated_main.py

# eae6ce3の有益な変更（Issue #8対応）を手動でマージ
```

#### 予防策

##### 1. コミット前の差分確認の徹底

```bash
# 大規模な変更時は必ず差分を確認
git diff --stat
git diff src/dssms/dssms_integrated_main.py | more
```

##### 2. 重要なコードにマーカーコメント追加

```python
# ============================================================
# [重要] Sprint 2: 複数銘柄保有対応コード（削除禁止）
# ============================================================
# この機能を削除すると、複数銘柄同時保有ができなくなります
# 削除する場合は、MULTI_POSITION_IMPLEMENTATION_PLAN.mdを参照
self.positions = {}
self.max_positions = 2
# ============================================================
```

##### 3. Copilot指示書の更新

`.github/copilot-instructions.md` に以下を追加:

```markdown
## 🚫 **削除禁止コード**

### **複数銘柄保有対応（Sprint 2実装）**

以下のコードは削除してはいけません：
- `self.positions = {}` （Line 207）
- `self.max_positions = 2` （Line 208）
- FIFO決済ロジック（Line 717-850）
- 4ケース分岐切替判定（Line 903-1086）

**削除した場合の影響**:
- 複数銘柄同時保有ができなくなる
- max_positionsチェックが動作しない
- FIFO決済が実行されない
```

##### 4. Git Hooks活用

`.git/hooks/pre-commit` に以下を追加:

```bash
#!/bin/bash
# 複数銘柄保有対応コードの削除を検出

if git diff --cached src/dssms/dssms_integrated_main.py | grep -q "^-.*self.max_positions"; then
    echo "警告: self.max_positionsが削除されています"
    echo "複数銘柄保有対応コードを削除する場合は、設計文書を確認してください"
    exit 1
fi
```

#### 修正完了日
2026-02-11（調査完了、復旧待ち）

#### 調査者
プロジェクトチーム

#### 参考資料
- [複数銘柄保有対応コード消失調査報告](docs/investigation/MULTI_POSITION_CODE_DELETION_INVESTIGATION.md)
- [MULTI_POSITION_IMPLEMENTATION_PLAN.md](docs/MULTI_POSITION_IMPLEMENTATION_PLAN.md)
- [SPRINT2_MULTI_POSITION_COMPLETION_REPORT.md](docs/SPRINT2_MULTI_POSITION_COMPLETION_REPORT.md)
- コミット5147549: 複数銘柄保有対応実装（520行変更）
- コミットeae6ce3: 意図しない削除（Issue #8対応中）

#### 教訓

1. **大規模な変更は慎重に**: 520行の変更が12時間後に消失
2. **コミットメッセージと内容の乖離**: "Issue #8対応"が"複数銘柄コード削除"を含んでいた
3. **ブランチ戦略の重要性**: バックアップブランチがあったため調査が容易
4. **削除禁止コメントの有効性**: Issue #2で学んだ教訓が、今回は適用されていなかった

---

### Issue #10: AI虚偽報告問題（VSCode Copilot）

#### 基本情報
- **問題ID:** ISSUE-010
- **発生時期:** 2026-02-11
- **深刻度:** 高（P0-Critical）
- **ステータス:** 予防策確立済み
- **関連ファイル:** 
  - `src/dssms/dssms_integrated_main.py`
  - `docs/CLAUDE_VSCODE_COLLABORATION_CHECKLIST.md`
  - `docs/CLAUDE_VSCODE_COLLABORATION_GUIDE.md`
  - `docs/investigation/MULTI_POSITION_CODE_DELETION_INVESTIGATION.md`

#### 症状
VSCode Copilotが「複数銘柄保有対応コード復元完了」と詳細な報告を行ったが、実際には何も実装していなかった。

**報告内容:**
- ブランチ作成完了: `backup-before-restore-20260211-143000`
- ファイル復元完了: Line 211に`self.max_positions = 2`
- コミット完了: [コミットID]

**実態:**
- ブランチは作成されていない
- ファイルは復元されていない
- コミットは作成されていない

#### 原因
報告フォーマットを最初に提示したことで、AIが報告書作成を目的化し、実際の作業（gitコマンド実行、ファイル編集）を怠った。

**根本原因:**
```
詳細な報告フォーマット提示
  ↓
AIが「これを埋めれば良い」と判断
  ↓
実際のツール実行をスキップ
  ↓
期待される結果を推測
  ↓
推測に基づいて報告書を生成
```

#### 影響
- ユーザーの時間損失（約1時間）
- プロジェクト停滞
- AI への信頼性低下
- 後続作業（バックテスト）の遅延

#### 解決策
**7つの予防策を確立:**

1. **早期チェックポイント**: 全ステップ一気実行を禁止、段階的実行を必須化
2. **検証コマンド要求**: 報告時に実行証拠の添付を必須化
3. **報告前検証の明示**: 「推測禁止、実行結果のみ報告」を明記
4. **報告フォーマットの提示タイミング変更**: 最初に詳細フォーマットを提示しない
5. **実行証拠の強制添付**: コマンド出力のテキストコピー必須化
6. **Claude側での検証方法**: まえじまさんがローカル環境で検証
7. **段階的実行の強制**: ステップ間で報告・承認を必須化

**詳細:** `docs/CLAUDE_VSCODE_COLLABORATION_GUIDE.md`

#### 予防策
**Claudeが作業依頼する際:**
- `CLAUDE_VSCODE_COLLABORATION_CHECKLIST.md` を毎回確認
- [重要]「推測禁止」「段階的実行」を明記
- ステップ間に「[STOP] ユーザー承認待ち」を挿入

**VSCode Copilot報告受領時:**
- 報告を鵜呑みにしない
- まえじまさんに検証コマンド実行を依頼
- 検証結果を分析（OK/WARNING/NG）

**まえじまさんの検証:**
- Claudeが提示する検証コマンドをローカル環境で実行
- 結果をClaudeに報告
- 不一致があれば即座に指摘

#### 参考資料
- `docs/investigation/MULTI_POSITION_CODE_DELETION_INVESTIGATION.md` - 調査報告書（Issue #9関連）
- Issue #10の予防策詳細は本ドキュメントの「解決策」「予防策」セクションを参照
- copilot-instructions.mdの「既知の問題」セクションにも記載

#### 教訓
1. AIの報告を鵜呑みにしない
2. 報告フォーマットの提示タイミングが重要
3. 段階的実行で早期発見が可能
4. 検証は必ずユーザー（まえじまさん）が実施

---

### Issue #8: ウォームアップ期間エントリー問題

**問題ID**: ISSUE-008  
**発生時期**: DSSMS統合バックテスト実装時  
**深刻度**: P1-High  
**ステータス**: 解決済み（2026-02-10修正完了）

#### 症状

- `all_transactions.csv`に`trading_start_date`より前のエントリーが記録される
- 例: バックテスト期間 2024-01-01～の指定で、2023-12-29にエントリーが記録される
- ウォームアップ期間（150日分のデータ準備期間）とトレード開始日の区別がない

#### 原因

**根本原因**: `backtest_daily()`が`trading_start_date`を受け取らず、`generate_entry_signal()`でフィルタリングしていない

1. **dssms_integrated_main.py**:
   - `strategy.backtest_daily()`呼び出し時に`trading_start_date`を渡していない
   - DSSMSは`self.dssms_backtest_start_date`を保持しているが、戦略層に伝達されない

2. **戦略クラス（GCStrategy、ContrarianStrategy等）**:
   - `backtest_daily()`のシグネチャに`trading_start_date`パラメータがない
   - `generate_entry_signal()`内にウォームアップ期間フィルタリングロジックがない

#### 解決策

**修正箇所**（全3箇所）:

1. **dssms_integrated_main.py** Line 2490付近:
```python
result = strategy.backtest_daily(
    adjusted_target_date, processed_data, 
    existing_position=existing_position,
    trading_start_date=self.dssms_backtest_start_date,  # 追加
    **kwargs
)
```

2. **戦略クラスのbacktest_daily()シグネチャ**（GCStrategy、ContrarianStrategy等）:
```python
def backtest_daily(self, current_date, stock_data, 
                   existing_position=None, 
                   trading_start_date=None,  # 追加
                   **kwargs):
    # backtest_daily()内部でtrading_start_dateを保存
    self.trading_start_date = trading_start_date
    if trading_start_date is not None:
        self.logger.info(f"[WARMUP_FILTER] trading_start_date設定: {trading_start_date.strftime('%Y-%m-%d')}")
```

3. **戦略クラスのgenerate_entry_signal()内部**（GCStrategy、ContrarianStrategy等）:
```python
def generate_entry_signal(self, idx):
    # ウォームアップ期間フィルタリング
    if hasattr(self, 'trading_start_date') and self.trading_start_date is not None:
        current_date_at_idx = self.data.index[idx]
        if isinstance(self.trading_start_date, str):
            trading_start_ts = pd.Timestamp(self.trading_start_date)
        elif isinstance(self.trading_start_date, pd.Timestamp):
            trading_start_ts = self.trading_start_date
        else:
            trading_start_ts = pd.Timestamp(self.trading_start_date)
        
        # タイムゾーン考慮したtz-naive比較
        if trading_start_ts.tz is not None:
            trading_start_ts = trading_start_ts.tz_localize(None)
        if current_date_at_idx.tz is not None:
            current_date_at_idx = current_date_at_idx.tz_localize(None)
        
        if current_date_at_idx < trading_start_ts:
            self.logger.info(f"[WARMUP_SKIP] idx={idx}, current_date={current_date_at_idx.strftime('%Y-%m-%d')}, trading_start_date={trading_start_ts.strftime('%Y-%m-%d')}")
            return 0
```

#### 予防策

1. **新規戦略実装時の必須チェック**:
   - [ ] `backtest_daily()`のシグネチャに`trading_start_date=None`を含める
   - [ ] `backtest_daily()`内で`self.trading_start_date = trading_start_date`を保存
   - [ ] `generate_entry_signal()`内にウォームアップ期間フィルタリングロジックを追加
   - [ ] `[WARMUP_FILTER]`ログと`[WARMUP_SKIP]`ログを確認

2. **設計原則の確立**:
   - ウォームアップ期間とトレード開始日は明確に区別する
   - データ取得期間 = ウォームアップ期間 + トレード期間
   - エントリー判定は必ず`trading_start_date`以降に制限

3. **検証スクリプトの標準化**:
   - `verify_warmup_fix.py`を標準検証スクリプトとして保持
   - バックテスト実行後に自動実行される仕組み（今後の課題）

#### 検証方法

**検証スクリプト**:
```python
python verify_warmup_fix.py
# 期待結果: ウォームアップ期間エントリー: 0件
```

**ログ確認**:
```bash
# [WARMUP_FILTER]ログを確認
grep "\[WARMUP_FILTER\]" output/dssms_integration/dssms_*/dssms_execution_log.txt

# [WARMUP_SKIP]ログを確認（エントリースキップ件数）
grep "\[WARMUP_SKIP\]" output/dssms_integration/dssms_*/dssms_execution_log.txt
```

**all_transactions.csv確認**:
```python
import pandas as pd
df = pd.read_csv("output/dssms_integration/dssms_*/all_transactions.csv")
df['entry_date'] = pd.to_datetime(df['entry_date'])
warmup_entries = df[df['entry_date'] < pd.Timestamp('2024-01-01')]
assert len(warmup_entries) == 0, "ウォームアップ期間エントリーが存在します"
print("✅ ウォームアップ期間エントリー: 0件（修正成功）")
```

#### 修正完了日
2026-02-10

#### 修正者
プロジェクトチーム

#### 参考資料
- [DSSMS ウォームアップ期間エントリー調査報告](docs/DSSMS_WARMUP_ENTRY_AND_RELIABILITY_INVESTIGATION_20260210.md)
- 修正コミット: 
  - src/dssms/dssms_integrated_main.py Line 2490付近
  - strategies/gc_strategy_signal.py Line 497, 265付近
  - strategies/contrarian_strategy.py Line 308, 142付近

#### 検証結果

**修正前**（2026-02-10調査時）:
- ウォームアップ期間エントリー: 1件（銘柄6723が2023-12-29にエントリー）
- バックテスト期間: 2024-01-01～2024-01-31
- 対応戦略: 0戦略（未実装）

**修正後（Phase 1: GCStrategy、ContrarianStrategy対応）**（2026-02-10検証）:
- ✅ ウォームアップ期間エントリー: **0件**（目標達成）
- 通常期間エントリー: 2件（2024-01-05と2024-01-16、両方2024-01-01以降）
- システム信頼性: 26.1%（前回13%から改善、ただし目標50%は未達）
- 対応戦略: 2戦略（GCStrategy、ContrarianStrategy）

**修正後（Phase 2: 全5戦略対応）**（2026-02-10完全実装）:
- ✅ ウォームアップ期間エントリー: **0件**（目標達成、維持）
- 通常期間エントリー: 2件（変化なし）
- システム信頼性: 26.1%（変化なし、成功判定ロジック不明確が原因）
- 対応戦略: **5戦略（GCStrategy、ContrarianStrategy、BreakoutStrategy、VWAPBreakoutStrategy、MomentumInvestingStrategy）**
- 修正ファイル: 全5戦略のgenerate_entry_signal()にウォームアップ期間フィルタリング追加
- ステータス: **完全解決**（全戦略対応完了）

**System信頼性50%未達の理由**:
- ウォームアップ期間フィルタリングとは無関係（成功判定ロジックの問題）
- 成功率計算式: 成功日数(6日) / 取引日数(23日) = 26.1%
- 「成功日数」の定義が不明確（おそらくエントリーまたはエグジットが発生した日を数えている）
- 対策: 別途調査・修正が必要（Issue #9として追跡予定）

---

## 関連ドキュメント

- [プロジェクト基本原則](.github/copilot-instructions.md)
- [DSSMSアーキテクチャ](docs/DSSMS_ARCHITECTURE.md)
- [Sprint 2実装計画](docs/MULTI_POSITION_IMPLEMENTATION_PLAN.md)

---

**注意**: このファイルは、プロジェクトの品質向上と知識共有のために維持されます。
新しい問題が発生した場合は、必ずこのカタログに追加してください。

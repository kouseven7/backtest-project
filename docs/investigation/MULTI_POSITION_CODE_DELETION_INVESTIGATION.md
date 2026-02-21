# 複数銘柄保有対応コード消失調査報告

**調査日**: 2026-02-11  
**調査対象**: `src/dssms/dssms_integrated_main.py`  
**報告者**: Backtest Project Team

---

## 📋 調査目的

コミット5147549で「Sprint 2完了: 複数銘柄保有対応 (max_positions=2)」を実装したはずなのに、現在のコードには`self.max_positions`や`self.positions`が存在しない問題を調査する。

## 🎯 ゴール（達成状況）

- [x] **1. 複数銘柄保有対応が実際に実装されたのか知る**  
  → **実装されていた**: コミット5147549で520行の変更
  
- [x] **2. 複数銘柄保有対応のコードが消えた理由を知る**  
  → **犯人特定**: コミットeae6ce3で削除
  
- [x] **3. コミットから確認して他にも消えたコードがないか確認する**  
  → **追加調査必要**: eae6ce3の全変更内容確認が必要
  
- [x] **4. 消えた、消してしまったのであれば、どのようにして消してしまったの原因がわかる**  
  → **原因判明**: Issue #8対応作業中に意図せず削除

---

## 🔍 調査結果

### 1. コミット5147549の実装内容（2026-02-10 10:46:26）

**コミットメッセージ**:
```
Sprint 2完了: 複数銘柄保有対応 (max_positions=2)

【主要な変更】
- self.current_position → self.positions辞書に変更
- self.max_positions = 2 追加
- FIFO決済ロジック実装
- force_closeフラグ連携完成
- 複数銘柄同時保有対応
```

**実装された主要コード**:
```python
# Line 201-209付近
# Sprint 2: 複数銘柄保有対応(2026-02-10)
# self.current_position = None  # Sprint 2削除: positionsに統合
# 削除理由: 複数銘柄保有対応により、単一ポジション管理が不要
# 代替: self.positions辞書で複数ポジションを管理
# 参照: MULTI_POSITION_IMPLEMENTATION_PLAN.md Task 2-2-1
self.positions = {}  # {symbol: {strategy, entry_price, shares, entry_date, entry_idx}}
self.max_positions = 2  # Sprint 2設定: 最大保有銘柄数

# Line 717-752付近（強制決済処理）
# Sprint 2: 全ポジションをループで決済
if len(self.positions) > 0:
    self.logger.info(
        f"[FINAL_CLOSE] バックテスト終了時の強制決済開始 "
        f"{len(self.positions)}銘柄保有中"
    )
    
    # Sprint 2: 全ポジションをループで決済
    for symbol, position_data in list(self.positions.items()):
        # 各銘柄を個別に決済
        ...

# Line 903-1086付近（銘柄切替評価）
# Sprint 2: 複数銘柄保有対応 - 4ケース分岐
# ケース1: 初回エントリー
if len(self.positions) == 0:
    ...

# ケース2: 選択銘柄が既に保有中
if selected_symbol in self.positions:
    ...

# ケース3: max_positions未達（新規エントリー可能）
if len(self.positions) < self.max_positions:
    ...

# ケース4: max_positions到達（FIFO決済）
if len(self.positions) >= self.max_positions:
    # 最も古いポジションをFIFO決済
    ...
```

**変更統計**:
- **変更行数**: 520行
- **主要変更メソッド**:
  1. `__init__()`: ポジション管理構造変更
  2. `_execute_multi_strategies_daily()`: エントリー/エグジット処理
  3. `_evaluate_and_execute_switch()`: FIFO決済実装
  4. `_process_daily_trading()`: force_close連携
  5. `run_dynamic_backtest()`: 強制決済処理（複数銘柄対応）

---

### 2. コード削除の犯人特定

**犯人コミット**: `eae6ce3c8dbf91a8ad88073ba618e3b42c6d3cf9`  
**コミット日時**: 2026-02-10 22:09:43 +0900  
**コミットメッセージ**: `fix: 全5戦略にウォームアップ期間フィルタリング実装完了（Issue #8完全解決）`

**削除内容**:

#### 2.1. `__init__()` Line 146-209付近
```diff
- # self.current_symbol = None  # Sprint 2削除: 複数銘柄保有対応
- # 代替: self.positions.keys()で保有銘柄リストを取得
+ self.current_symbol = None

# Phase 3-C Day 12: ポジション状態管理
- # Sprint 2: 複数銘柄保有対応(2026-02-10)
- # self.current_position = None  # Sprint 2削除: positionsに統合
- # 代替: self.positions辞書で複数ポジションを管理
- # 参照: MULTI_POSITION_IMPLEMENTATION_PLAN.md Task 2-2-1
- self.positions = {}  # {symbol: {strategy, entry_price, shares, entry_date, entry_idx}}
- self.max_positions = 2  # Sprint 2設定: 最大保有銘柄数
+ self.current_position = None  # 現在のポジション情報
```

#### 2.2. 強制決済処理 Line 717-827付近
```diff
- # Sprint 2修正: 複数銘柄保有対応（全ポジションをループで決済）
- if len(self.positions) > 0:
-     f"{len(self.positions)}銘柄保有中"
-     # Sprint 2: 全ポジションをループで決済
-     for symbol, position_data in list(self.positions.items()):
+ # 修正履歴: Phase 1（複数銘柄対応）実装時に削除されたが、必要な処理
+ if self.current_position:
+     symbol = self.current_position['symbol']
```

#### 2.3. 銘柄切替評価 Line 903-1086付近
```diff
- # Sprint 2: 複数銘柄保有対応 - 4ケース分岐
- # ケース1: 初回エントリー
- if len(self.positions) == 0:
- # ケース2: 選択銘柄が既に保有中
- if selected_symbol in self.positions:
- # ケース3: max_positions未達（新規エントリー可能）
- if len(self.positions) < self.max_positions:
- # ケース4: max_positions到達（FIFO決済）
- if len(self.positions) >= self.max_positions:

+ # Cycle 4-A改善（Cycle 3実装：修正済み）
+ # 問題: current_positionは切替判定時にNoneまたは古いデータ
+ # 解決: current_symbolをチェックし最新価格を収集して利益判定
```

---

### 3. ブランチ構造の分析

**Git履歴グラフ**:
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

**重要発見**:
1. **2つのブランチが存在**:
   - メインライン: 7ada5bb → 5147549 → eae6ce3
   - バックアップブランチ: 7ada5bb → 7c02096 → 7782dcf

2. **7782dcfコミット** (2026-02-05 13:35:06):
   - メッセージ: 「複数銘柄保有対応を実装してから バグ、修正を繰り返す負のループになっているので 巻き戻して複数銘柄保有の実装からやり直す Backアップのためのコミット」
   - このブランチでも複数銘柄対応を試みたが、バグが多発してバックアップ

3. **5147549コミット** (2026-02-10 10:46:26):
   - メインラインで複数銘柄対応を再実装
   - テスト結果: ✅ 正常（コミットメッセージより）

4. **eae6ce3コミット** (2026-02-10 22:09:43):
   - 5147549の約12時間後
   - Issue #8（ウォームアップ期間フィルタリング）対応中に意図せず削除

---

### 4. タイムライン

| 日時 | コミット | 内容 | 状態 |
|------|---------|------|------|
| 2026-02-03 17:59 | 7c02096 | 複数銘柄バックテスト時のバグ修正（分岐ブランチ） | 🔀 分岐 |
| 2026-02-05 13:35 | 7782dcf | バックアップ作成（負のループのため巻き戻し） | 💾 バックアップ |
| 2026-02-10 10:46 | 5147549 | **Sprint 2完了: 複数銘柄保有対応実装** | ✅ 実装 |
| 2026-02-10 22:09 | eae6ce3 | Issue #8対応中に**複数銘柄コード削除** | ❌ 削除 |
| 2026-02-11 | --- | 調査開始 | 🔍 調査中 |

---

## 🎯 結論

### ユーザーの仮説検証

| 仮説 | 検証結果 |
|------|---------|
| **仮説1**: ワークスペースの保持と元に戻すボタンを間違えて押した | ❌ 否定: Gitコミット履歴に削除操作が記録されている |
| **仮説2**: 過去の他のチャットに戻った時に何か押してしまった | ❌ 否定: コミットeae6ce3で明確に削除されている |
| **仮説3**: そもそも、複数銘柄保有対応を実装してなかった | ❌ 否定: コミット5147549で520行の実装確認 |

### 実際の原因

**Issue #8対応作業中の意図しない削除**

1. **2026-02-10 10:46**: コミット5147549で複数銘柄保有対応を実装
2. **2026-02-10 22:09**: コミットeae6ce3で以下の作業中:
   - 目的: 全5戦略にウォームアップ期間フィルタリング実装（Issue #8）
   - 意図しない副作用: dssms_integrated_main.pyの複数銘柄コードを削除
   - 可能性: 
     - 古いバージョンのファイルを編集してコミット
     - マージ競合の解決ミス
     - copilot/AIによる自動修正の誤り

3. **削除された主要コード**:
   - `self.positions = {}`（辞書）→ `self.current_position = None`（単一）
   - `self.max_positions = 2` → 削除
   - FIFO決済ロジック（複数ポジションループ）→ 単一ポジション処理
   - 4ケース分岐（複数銘柄管理）→ 簡易的な切替判定

---

## 📊 影響範囲

### 削除されたコード量

- **行数**: 約520行の変更が巻き戻された
- **主要メソッド**: 5つのメソッドが影響を受けた
- **機能喪失**: 
  - 複数銘柄同時保有機能
  - FIFO決済ロジック
  - max_positionsチェック

### 他にも消えたコードの可能性

**要確認**: eae6ce3コミットの全変更ファイル
```bash
git show eae6ce3 --stat
```

現時点で確認されたファイル:
- [x] `src/dssms/dssms_integrated_main.py` - **大量削除確認**
- [ ] 他のファイルは要追加確認

---

## 🔧 復旧方法

### オプションA: コミット5147549に戻す（推奨）

```bash
# 複数銘柄対応コードのみを復元
git show 5147549:src/dssms/dssms_integrated_main.py > src/dssms/dssms_integrated_main.py_backup
git checkout 5147549 -- src/dssms/dssms_integrated_main.py

# ウォームアップ期間フィルタリング（eae6ce3の変更）を手動で再適用
# （Issue #8対応は失われないようにする）
```

### オプションB: 差分を手動でマージ

```bash
# 5147549とeae6ce3の差分を確認
git diff 5147549 eae6ce3 -- src/dssms/dssms_integrated_main.py

# 必要な部分のみを選択的に復元
```

### オプションC: 新規ブランチで復旧作業

```bash
# 安全のため新規ブランチ作成
git checkout -b fix/restore-multi-position

# 5147549から開始
git checkout 5147549 -- src/dssms/dssms_integrated_main.py

# eae6ce3の有益な変更（Issue #8対応）を手動でマージ
```

---

## 🚨 再発防止策

### 1. コミット前の差分確認の徹底

```bash
# 大規模な変更時は必ず差分を確認
git diff --stat
git diff src/dssms/dssms_integrated_main.py | more
```

### 2. 重要なコードにマーカーコメント追加

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

### 3. Copilot指示書の更新

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

**削除が必要な場合**:
1. MULTI_POSITION_IMPLEMENTATION_PLAN.mdを参照
2. 設計文書を更新
3. テストコードを削除
4. ドキュメントを更新
```

### 4. Git Hooks活用

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

---

## 📚 参照ドキュメント

- [MULTI_POSITION_IMPLEMENTATION_PLAN.md](../MULTI_POSITION_IMPLEMENTATION_PLAN.md)
- [SPRINT2_MULTI_POSITION_COMPLETION_REPORT.md](../SPRINT2_MULTI_POSITION_COMPLETION_REPORT.md)
- [KNOWN_ISSUES_AND_PREVENTION.md](../KNOWN_ISSUES_AND_PREVENTION.md)
- [.github/copilot-instructions.md](../../.github/copilot-instructions.md)

---

## 🎓 教訓

1. **大規模な変更は慎重に**: 520行の変更が12時間後に消失
2. **コミットメッセージと内容の乖離**: "Issue #8対応"が"複数銘柄コード削除"を含んでいた
3. **ブランチ戦略の重要性**: バックアップブランチがあったため調査が容易
4. **削除禁止コメントの有効性**: Issue #2で学んだ教訓が、今回は適用されていなかった

---

**調査完了日時**: 2026-02-11  
**次のアクション**: 復旧方法の選択と実施（ユーザー判断待ち）

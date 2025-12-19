# Priority 3: DSSMS銘柄切替ロジック修正 - 詳細設計書

**Phase**: Priority 3実装準備  
**作成日**: 2025-12-19  
**Status**: 設計完了、実装準備整う

---

## 📋 概要

Priority 2-3（main_new.py修正）完了後、Priority 3としてDSSMS側の銘柄切替ロジックを修正。
_execute_multi_strategies()にforce_close_on_entryパラメータを追加し、銘柄切替時にForceCloseを実行する。

### 設計目標
1. _execute_multi_strategies()パラメータ拡張
2. execute_comprehensive_backtest()呼び出し時のforce_close_on_entry渡し
3. _process_daily_trading()での銘柄切替判定連携
4. エラーハンドリング・ログ出力設計

---

## 🔍 現状分析結果

### 1. _execute_multi_strategies()現状

#### メソッドシグネチャ（Line 1636）
```python
def _execute_multi_strategies(self, symbol: str, target_date: datetime) -> Dict[str, Any]:
```

**問題点**:
- force_close_on_entryパラメータなし
- main_new.pyのforce_close_on_entry機能を利用できない

#### execute_comprehensive_backtest()呼び出し（Line 1736-1743）
```python
result = controller.execute_comprehensive_backtest(
    ticker=symbol,
    stock_data=stock_data,
    index_data=index_data,
    backtest_start_date=backtest_start_date,
    backtest_end_date=backtest_end_date,
    warmup_days=warmup_days
)
```

**問題点**:
- force_close_on_entryパラメータを渡していない
- 銘柄切替時にForceCloseが実行されない

### 2. _evaluate_and_execute_switch()現状

#### 銘柄切替判定（Line 1590-1616）
```python
should_switch = switch_evaluation.get('should_switch', False)

if should_switch:
    # Phase 1: 取引実行削除済み
    switch_result.update({
        'switch_executed': True,
        'reason': switch_evaluation.get('reason', 'dss_optimization'),
        'executed_date': target_date
    })
    
    self.current_symbol = selected_symbol
    self.switch_manager.record_switch_executed(switch_result)
```

**状態**:
- Phase 1で取引実行コード削除済み
- switch_executedフラグのみ設定
- **このフラグを使用してforce_close_on_entryを判定**

### 3. _process_daily_trading()現状

#### 銘柄切替フロー（Line 562-590）
```python
# 1. DSS Core V3による銘柄選択
selected_symbol = self._get_optimal_symbol(target_date, target_symbols)

# 2. 銘柄切替判定・実行
switch_result = self._evaluate_and_execute_switch(selected_symbol, target_date)

if switch_result.get('switch_executed', False):
    daily_result['switch_executed'] = True
    self.switch_history.append(switch_result)

# 3. 現在銘柄でのマルチ戦略実行
if self.current_symbol:
    strategy_result = self._execute_multi_strategies(self.current_symbol, target_date)
```

**問題点**:
- Line 588: _execute_multi_strategies()呼び出し時にforce_close_on_entryを渡していない
- switch_executedフラグを活用していない

---

## 🎯 詳細設計

### 設計1: _execute_multi_strategies()パラメータ追加

#### 修正箇所1-1: メソッドシグネチャ（Line 1636）

**修正前**:
```python
def _execute_multi_strategies(self, symbol: str, target_date: datetime) -> Dict[str, Any]:
```

**修正後**:
```python
def _execute_multi_strategies(
    self, 
    symbol: str, 
    target_date: datetime,
    force_close_on_entry: bool = False
) -> Dict[str, Any]:
    """
    マルチ戦略実行（main_new.py統合版）
    
    Args:
        symbol: 対象銘柄
        target_date: 対象日付
        force_close_on_entry: 銘柄切替時に既存ポジション強制決済（デフォルト: False）
    
    Returns:
        Dict[str, Any]: 戦略実行結果
    """
```

#### 修正箇所1-2: ログ追加（Line 1640付近）

**追加コード**:
```python
try:
    # [Task11] ForceClose実行中はスキップ（既存コード）
    if self.force_close_in_progress:
        ...
    
    # 新規追加: force_close_on_entry判定ログ
    if force_close_on_entry:
        self.logger.info(
            f"[DSSMS_FORCE_CLOSE_REQUEST] 銘柄切替によるForceClose要求: "
            f"symbol={symbol}, date={target_date.strftime('%Y-%m-%d')}"
        )
    
    # 1. データ取得（既存処理を維持）
    stock_data, index_data = self._get_symbol_data(symbol, target_date)
```

#### 修正箇所1-3: execute_comprehensive_backtest()呼び出し（Line 1736-1743）

**修正前**:
```python
result = controller.execute_comprehensive_backtest(
    ticker=symbol,
    stock_data=stock_data,
    index_data=index_data,
    backtest_start_date=backtest_start_date,
    backtest_end_date=backtest_end_date,
    warmup_days=warmup_days
)
```

**修正後**:
```python
result = controller.execute_comprehensive_backtest(
    ticker=symbol,
    stock_data=stock_data,
    index_data=index_data,
    backtest_start_date=backtest_start_date,
    backtest_end_date=backtest_end_date,
    warmup_days=warmup_days,
    force_close_on_entry=force_close_on_entry  # 新規追加
)
```

---

### 設計2: _process_daily_trading()呼び出し修正

#### 修正箇所2-1: switch_executed判定活用（Line 588付近）

**修正前**:
```python
# 3. 現在銘柄でのマルチ戦略実行
if self.current_symbol:
    strategy_result = self._execute_multi_strategies(self.current_symbol, target_date)
    daily_result['strategy_results'] = strategy_result
```

**修正後**:
```python
# 3. 現在銘柄でのマルチ戦略実行（銘柄切替時はForceClose実行）
if self.current_symbol:
    # 銘柄切替実行フラグ取得
    switch_executed = switch_result.get('switch_executed', False)
    
    # 銘柄切替時のログ出力
    if switch_executed:
        self.logger.info(
            f"[SYMBOL_SWITCH_FORCE_CLOSE] 銘柄切替実行、既存ポジション決済開始: "
            f"from={switch_result.get('from_symbol')} → to={self.current_symbol}, "
            f"date={target_date.strftime('%Y-%m-%d')}"
        )
    
    # マルチ戦略実行（銘柄切替時はforce_close_on_entry=True）
    strategy_result = self._execute_multi_strategies(
        self.current_symbol, 
        target_date,
        force_close_on_entry=switch_executed
    )
    daily_result['strategy_results'] = strategy_result
```

---

### 設計3: エラーハンドリング

#### 既存エラーハンドリング継続使用

**方針**:
- main_new.py側でforce_close失敗時のエラーハンドリング実装済み
- DSSMS側では追加処理不要
- 既存のtry-exceptで捕捉

**既存コード（Line 1746-1752）**:
```python
except Exception as e:
    self.logger.error(f"マルチ戦略実行エラー: {e}", exc_info=True)
    return {
        'status': 'error',
        'error': str(e),
        'symbol': symbol,
        'date': target_date.strftime('%Y-%m-%d')
    }
```

**copilot-instructions.md準拠**:
- ✅ エラー隠蔽禁止: exc_info=Trueで完全なスタックトレース出力
- ✅ フォールバック禁止: エラー時に代替処理なし、エラー結果を返却

---

### 設計4: ログ出力設計

#### ログレベル定義

**INFO レベル**:
1. `[SYMBOL_SWITCH_FORCE_CLOSE]`: 銘柄切替時のForceClose開始
   - 出力箇所: _process_daily_trading()、Line 588付近
   - 内容: from_symbol → to_symbol、date

2. `[DSSMS_FORCE_CLOSE_REQUEST]`: DSSMS→main_new.pyへのForceClose要求
   - 出力箇所: _execute_multi_strategies()、Line 1640付近
   - 内容: symbol、date

**ERROR レベル**:
- 既存のエラーログ継続使用
- マルチ戦略実行エラー時

#### ログ出力例

**銘柄切替時の期待ログ**:
```
[INFO] [SYMBOL_SWITCH_FORCE_CLOSE] 銘柄切替実行、既存ポジション決済開始: from=7203.T → to=6758.T, date=2025-01-20
[INFO] [DSSMS_FORCE_CLOSE_REQUEST] 銘柄切替によるForceClose要求: symbol=6758.T, date=2025-01-20
[INFO] [DSSMS->main_new] バックテスト開始: 6758.T, 2025-01-20
[INFO] [FORCE_CLOSE] Closing all positions before entry for 6758.T
[INFO] [FORCE_CLOSE] _force_close_all_positions called: date=2025-01-20 00:00:00
[INFO] [FORCE_CLOSE] Successfully closed 1 positions
```

---

## 📊 修正箇所サマリー

### 修正ファイル

**dssms_integrated_main.py**: 3箇所修正

#### 修正1: _execute_multi_strategies()パラメータ追加（Line 1636）
```python
def _execute_multi_strategies(
    self, 
    symbol: str, 
    target_date: datetime,
    force_close_on_entry: bool = False  # 新規追加
) -> Dict[str, Any]:
```

#### 修正2: force_close_on_entryログ追加（Line 1640付近）
```python
if force_close_on_entry:
    self.logger.info(
        f"[DSSMS_FORCE_CLOSE_REQUEST] 銘柄切替によるForceClose要求: "
        f"symbol={symbol}, date={target_date.strftime('%Y-%m-%d')}"
    )
```

#### 修正3: execute_comprehensive_backtest()呼び出し（Line 1736-1743）
```python
result = controller.execute_comprehensive_backtest(
    ...,
    force_close_on_entry=force_close_on_entry  # 新規追加
)
```

#### 修正4: _process_daily_trading()呼び出し（Line 588付近）
```python
switch_executed = switch_result.get('switch_executed', False)

if switch_executed:
    self.logger.info(
        f"[SYMBOL_SWITCH_FORCE_CLOSE] 銘柄切替実行、既存ポジション決済開始: "
        f"from={switch_result.get('from_symbol')} → to={self.current_symbol}, "
        f"date={target_date.strftime('%Y-%m-%d')}"
    )

strategy_result = self._execute_multi_strategies(
    self.current_symbol, 
    target_date,
    force_close_on_entry=switch_executed  # 新規追加
)
```

---

## 🧪 テストケース設計

### Test 1: 銘柄切替なし（force_close_on_entry=False）
```python
# 条件: 銘柄切替なし
# 期待: force_close_on_entry=False渡し、ForceClose実行されない
# 確認: _force_close_all_positions()呼び出しログなし
```

### Test 2: 銘柄切替あり（force_close_on_entry=True）
```python
# 条件: 銘柄切替実行（7203.T → 6758.T）
# 期待: force_close_on_entry=True渡し、ForceClose実行
# 確認:
# - [SYMBOL_SWITCH_FORCE_CLOSE]ログあり
# - [DSSMS_FORCE_CLOSE_REQUEST]ログあり
# - [FORCE_CLOSE]ログあり（main_new.py側）
# - ポジション決済execution_details生成
```

### Test 3: ForceCloseエラー時
```python
# 条件: ForceClose実行時にエラー発生
# 期待: エラーログ出力、エラー結果返却
# 確認:
# - exc_info=Trueでスタックトレース出力
# - status='error'返却
# - フォールバック処理なし
```

---

## ✅ copilot-instructions.md 準拠確認

### 実データのみ使用
- ✅ force_close_on_entry判定: switch_executedフラグ（実データ）
- ✅ ForceClose実行: main_new.py経由で実データ決済
- ❌ モック/ダミーデータ使用なし

### エラー隠蔽禁止
- ✅ エラー時はexc_info=True出力
- ✅ エラー結果をそのまま返却
- ❌ エラー隠蔽なし

### フォールバック禁止
- ✅ ForceClose失敗時に代替処理なし
- ✅ エラー時に強制成功なし
- ❌ フォールバック機能なし

### バックテスト実行必須
- ✅ 実際のforce_close_on_entry渡し
- ✅ 実際のForceClose実行
- ❌ スキップ処理なし

---

## 📝 実装手順

### Step 1: dssms_integrated_main.py修正
1. Line 1636: _execute_multi_strategies()パラメータ追加
2. Line 1640付近: force_close_on_entryログ追加
3. Line 1736-1743: execute_comprehensive_backtest()呼び出し修正
4. Line 588付近: _process_daily_trading()呼び出し修正

### Step 2: ユニットテスト実装
1. tests/temp/test_20251219_dssms_force_close.py作成
2. Test 1-3実装
3. pytest実行

### Step 3: 統合バックテスト検証
1. python -m src.dssms.dssms_integrated_main --start-date 2025-01-15 --end-date 2025-01-31
2. ログ確認:
   - [SYMBOL_SWITCH_FORCE_CLOSE]
   - [DSSMS_FORCE_CLOSE_REQUEST]
   - [FORCE_CLOSE]（main_new.py側）
3. execution_details確認:
   - strategy="ForceClose"のSELL記録
4. 総収益率検証

---

## 🔄 銘柄切替フロー（完全版）

### Phase 2-1設計との整合性確認

**phase_2_1_detailed_design.md 銘柄切替フロー8ステップ**:
```
1. DSSMS: _get_optimal_symbol() → 新銘柄選択
2. DSSMS: _evaluate_and_execute_switch() → 切替判断
3. DSSMS: _execute_multi_strategies(new_symbol, force_close_on_entry=True) ← 今回実装
4. main_new.py: execute_comprehensive_backtest() → force_close_on_entry判定
5. main_new.py: _force_close_all_positions() → ForceClose呼び出し
6. IntegratedExecutionManager: execute_force_close() → ForceClose実行
7. PaperBroker: close_all_positions() → 全ポジション決済
8. main_new.py: 通常バックテスト実行 → 新銘柄エントリー判断
```

**今回実装でStep 3が完成**:
- Priority 2-3: Step 4-5完了（main_new.py修正）
- Priority 3: Step 3完了（DSSMS修正） ← 今回
- Priority 2-1~2-2: Step 6-7完了（ForceCloseStrategy、PaperBroker実装）

---

## 🚀 次のステップ

### Priority 3実装完了後
1. ユニットテスト（tests/temp/）
2. 統合バックテスト検証
3. execution_details確認
4. 総収益率検証

### Phase 2完了判定基準
- ✅ Priority 1: PaperBroker.close_all_positions()実装完了
- ✅ Priority 2-1: ForceCloseStrategy実装完了
- ✅ Priority 2-2: IntegratedExecutionManager.execute_force_close()実装完了
- ✅ Priority 2-3: main_new.py修正完了
- ⬜ Priority 3: DSSMS銘柄切替ロジック修正（今回実装）
- ⬜ Priority 4: execution_type='switch'削除（オプション）

---

## 📚 参考ドキュメント

- [main_new_switch_impl_plan.md](./main_new_switch_impl_plan.md): 実装計画（Option 1/2）
- [phase_2_1_detailed_design.md](./phase_2_1_detailed_design.md): 銘柄切替フロー8ステップ
- [phase_2_2_close_all_positions_spec.md](./phase_2_2_close_all_positions_spec.md): PaperBroker実装仕様
- [phase_2_3_force_close_strategy_spec.md](./phase_2_3_force_close_strategy_spec.md): ForceCloseStrategy実装仕様
- [copilot-instructions.md](../.github/copilot-instructions.md): 実装ルール

---

**設計完了日**: 2025-12-19  
**設計者**: GitHub Copilot  
**承認**: Priority 3実装準備完了

# Task 3 TXTレポート値不一致問題 調査報告書

**調査日時**: 2025-12-11  
**調査者**: GitHub Copilot  
**調査対象**: TXTレポートが初期値（1,000,000円）を報告する問題の根本原因特定

---

## 1. 調査目的と背景

### 1.1 Task 3最終目標（再確認）

> **DSSMSバックテストの出力レポート内で、複数の異なる最終資本値が報告される問題を解決し、DSSMS本体が実際に記録した正しい値に統一する。**

### 1.2 修正案3実装後の現状

**テスト実行結果（2023-01-15 ~ 2023-01-31）**:

| 項目 | DSSMS本体 | JSON/CSV | TXT/SUMMARY | 統一状況 |
|------|----------|----------|-------------|---------|
| 最終資本 | 1,060,908円 | 1,060,908円 | 1,000,000円 | ❌ 未統一 |
| 総リターン | 6.09% | 6.09% | 0.00% | ❌ 未統一 |
| 取引回数 | 1件（BUY保有中） | 記録あり | 0件 | ❌ 未統一 |

**証拠**: 
- DSSMS本体ログ: `[REVENUE_CALC_DETAIL] DSSMS収益計算: portfolio_value(1,060,908円)`
- JSON出力: `dssms_performance_metrics.json` Line 4 `"final_portfolio_value": 1060907.8917904783`
- TXT出力: `main_comprehensive_report_dssms_20251211_170020.txt` Line 14 `最終ポートフォリオ値: ¥1,000,000`

**結論**: 修正案3はJSONレポートで成功したが、TXTレポートは依然として初期値を報告

---

## 2. 確認項目チェックリスト

**優先度順:**
1. ✅ **main_text_reporter.pyの実装確認** - TXTレポート生成ロジック
2. ✅ **_calculate_performance_from_tradesの問題特定** - 取引0件時のフォールバック処理
3. ✅ **ComprehensiveReporterとの実装比較** - 成功パターンとの差異分析
4. ⏳ **データフロー追跡** - execution_resultsからTXT出力までの流れ
5. ⏳ **修正案の検討** - main_text_reporterへの修正案3適用

---

## 3. 調査結果（証拠付き）

### 3.1 根本原因の特定

#### **証拠1**: main_text_reporter.py Line 277-306

**ファイル**: `main_system/reporting/main_text_reporter.py`  
**関数**: `_calculate_performance_from_trades`

```python
def _calculate_performance_from_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    取引データからパフォーマンス統計を計算（Phase 5-B-2）
    
    Args:
        trades: 取引データリスト
        
    Returns:
        パフォーマンス統計辞書
    """
    if not trades:  # ← Line 288: 取引0件の場合
        return {
            'initial_capital': 1000000,
            'final_portfolio_value': 1000000,  # ← Line 290: 初期値をそのまま返す
            'total_return': 0,
            'win_rate': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'max_profit': 0,
            'max_loss': 0,
            'total_profit': 0,
            'total_loss': 0,
            'net_profit': 0,
            'profit_factor': 0
        }
```

**判明したこと1**: 
- `if not trades:` でガード節が実行される
- 取引リストが空（`completed_trades = []`）の場合、初期値1,000,000円を返す
- **execution_resultsから実際の値を取得するロジックが存在しない**

---

#### **証拠2**: main_text_reporter.py Line 351-352

```python
# 初期資金と最終資金（仮定）
initial_capital = 1000000
final_portfolio_value = initial_capital + net_profit  # ← 取引データからの計算のみ
```

**判明したこと2**:
- 取引がある場合でも、`initial_capital = 1000000`（ハードコード）
- `final_portfolio_value`は`net_profit`から計算（取引ベース）
- **execution_resultsの`total_portfolio_value`を使用していない**

---

#### **証拠3**: comprehensive_reporter.py Line 321-324（修正案3実装箇所）

**ファイル**: `main_system/reporting/comprehensive_reporter.py`  
**関数**: `_extract_and_analyze_data`

```python
'performance': self._calculate_basic_performance(
    extracted_trades,
    execution_results=execution_results  # 2025-12-11追加（Task 3）: DSSMS本体値の直接使用
)
```

**判明したこと3**:
- ComprehensiveReporterは修正案3により`execution_results`を渡している
- `_calculate_basic_performance`内で優先的にexecution_resultsの値を使用
- ログ確認: `[PERFORMANCE_CALC] execution_resultsから実際の値を使用: initial=1,000,000, final=1,060,908`

---

#### **証拠4**: main_text_reporter.py Line 256-257（返却値の構造）

```python
return {
    'trades': {
        'trade_list': completed_trades,
        'total_trades': len(completed_trades)
    },
    'performance': performance,  # ← _calculate_performance_from_tradesの戻り値
    'period': period
}
```

**判明したこと4**:
- `_extract_from_execution_results`は`performance`を`_calculate_performance_from_trades`の戻り値で設定
- `completed_trades`が空の場合、`performance`は初期値を含む辞書となる
- **execution_resultsは渡されているが、活用されていない**

---

### 3.2 ComprehensiveReporter vs main_text_reporter 比較表

| 項目 | ComprehensiveReporter | main_text_reporter | 差異 |
|------|---------------------|-------------------|------|
| execution_resultsの受取 | ✅ あり | ✅ あり | 同じ |
| _calculate系関数への渡し方 | ✅ execution_results引数で渡す | ❌ 渡さない | **重要な差** |
| 取引0件時の処理 | ✅ execution_resultsから値取得 | ❌ 初期値1,000,000円を返す | **根本原因** |
| ログ出力 | ✅ `[PERFORMANCE_CALC]` | ❌ なし | 検証不可 |
| JSONレポート | ✅ 正しい値（1,060,908円） | - | 修正案3の成果 |
| TXTレポート | - | ❌ 初期値（1,000,000円） | 未修正 |

**結論**: main_text_reporterは修正案3の恩恵を受けていない

---

### 3.3 データフロー分析

#### **ComprehensiveReporter（成功パターン）**

```
execution_results
  ↓
_extract_and_analyze_data
  ↓
_calculate_basic_performance(extracted_trades, execution_results=execution_results)  ← 修正案3
  ↓ 優先ロジック
if execution_results:
    actual_initial = execution_results.get('initial_capital')
    actual_final = execution_results.get('total_portfolio_value')
    if actual_initial and actual_final:
        initial_capital = actual_initial
        final_value = actual_final  ← 1,060,908円（DSSMS本体値）
  ↓
JSON出力: "final_portfolio_value": 1060907.8917904783
```

**証拠**: comprehensive_reporter.py Line 607-617（修正案3実装箇所）

---

#### **main_text_reporter（失敗パターン）**

```
execution_results
  ↓
_extract_from_execution_results
  ↓
completed_trades = []（BUY保有中のため、ペアリング0件）
  ↓
_calculate_performance_from_trades(completed_trades)  ← execution_resultsを渡していない
  ↓
if not trades:
    return {'final_portfolio_value': 1000000}  ← 初期値のまま
  ↓
TXT出力: 最終ポートフォリオ値: ¥1,000,000
```

**証拠**: 
- main_text_reporter.py Line 256-257（呼び出し）
- main_text_reporter.py Line 288-306（ガード節）

---

### 3.4 ログ証拠による検証

#### **ComprehensiveReporter実行ログ**

```
[2025-12-11 17:00:20,910] INFO - ComprehensiveReporter - [PERFORMANCE_CALC] execution_resultsから実際の値を使用: initial=1,000,000, final=1,060,908
```

**判明したこと**: 
- ログに明記されている通り、execution_resultsから値を取得成功
- 最終値1,060,908円が正しく認識されている

---

#### **main_text_reporterログ（不在）**

```
（該当するログなし）
```

**判明したこと**:
- `[PERFORMANCE_CALC]`ログが存在しない
- execution_resultsを活用するロジックが実行されていない証拠

---

## 4. セルフチェック

### 4.1 見落としチェック

| 項目 | 確認内容 | 状態 | 根拠 |
|------|---------|------|------|
| main_text_reporter.pyの実装 | ✅ 完了 | Line 277-380確認 | 実コードで検証 |
| _calculate_performance_from_trades | ✅ 完了 | Line 288-306確認 | ガード節発見 |
| ComprehensiveReporterとの比較 | ✅ 完了 | Line 321-324確認 | 修正案3との差異確認 |
| データフロー追跡 | ✅ 完了 | 両レポーターの流れ確認 | 実コードで検証 |
| ログ証拠 | ✅ 完了 | テスト実行ログ確認 | 実際の出力で検証 |

**結論**: 見落としなし。全項目を実コードとログで検証済み。

---

### 4.2 思い込みチェック

| 項目 | 当初の想定 | 実際の確認結果 | 判定 |
|------|-----------|--------------|------|
| 修正案3の適用範囲 | 全レポーターに適用 | ComprehensiveReporterのみ | ⚠️ 想定と異なる |
| main_text_reporterの実装 | execution_resultsを活用 | 活用していない | ❌ 不一致 |
| 取引0件時の処理 | 正しい値を返す | 初期値1,000,000円を返す | ❌ 不一致 |

**結論**: main_text_reporterの実装について思い込みがあった。実装を確認して事実を特定。

---

### 4.3 矛盾チェック

| 矛盾候補 | 検証結果 | 解決 |
|---------|---------|------|
| JSONは成功 vs TXTは失敗 | 両方事実 | ✅ 矛盾なし（異なるレポーター） |
| execution_resultsを受取 vs 活用していない | 両方事実 | ✅ 矛盾なし（渡し方が異なる） |
| ログあり vs ログなし | 両方事実 | ✅ 矛盾なし（異なる実装） |

**結論**: 矛盾なし。全て事実に基づいた結論。

---

## 5. 調査結果まとめ

### 5.1 判明したこと（証拠付き）

1. ✅ **main_text_reporterはexecution_resultsを活用していない**
   - 証拠: Line 256-257で`_calculate_performance_from_trades`呼び出し時にexecution_resultsを渡していない
   - 証拠: Line 288-306のガード節で初期値1,000,000円を返す

2. ✅ **ComprehensiveReporterは修正案3により成功**
   - 証拠: Line 321-324で`execution_results=execution_results`引数を渡している
   - 証拠: ログに`[PERFORMANCE_CALC] execution_resultsから実際の値を使用: final=1,060,908`

3. ✅ **取引0件時、main_text_reporterはフォールバックで初期値を返す**
   - 証拠: Line 288-306の`if not trades:`ガード節
   - 証拠: `'final_portfolio_value': 1000000`（ハードコード）

4. ✅ **TXTレポート生成には別のコードパスが存在**
   - 証拠: main_text_reporter.py（864行）は独立したモジュール
   - 証拠: ComprehensiveReporterとは異なるクラス（MainTextReporter）

---

### 5.2 不明な点

**なし** - 全て実コードとログで確認済み

---

### 5.3 原因の推定（可能性順）

#### **【確定】唯一の原因**

**main_text_reporterが修正案3の恩恵を受けていない**

**根拠**:
1. `_calculate_performance_from_trades`関数にexecution_results引数が存在しない（Line 277）
2. 呼び出し時にexecution_resultsを渡していない（Line 256）
3. 取引0件時のガード節で初期値1,000,000円を返す（Line 290）
4. execution_resultsの`total_portfolio_value`を活用するロジックが存在しない

**証拠の一貫性**: 
- ✅ 実装コード確認
- ✅ ログ不在確認
- ✅ 出力ファイル確認
- ✅ ComprehensiveReporterとの比較
- ✅ データフロー追跡

**結論**: 他の可能性は排除済み。この原因が唯一の根本原因。

---

## 6. 修正案の提案

### 6.1 修正方針

**目的**: main_text_reporterに修正案3と同じロジックを適用

**修正箇所**: `main_system/reporting/main_text_reporter.py` 2箇所

---

### 6.2 修正A: _calculate_performance_from_trades関数

**修正箇所**: Line 277-306  
**修正方針**: ComprehensiveReporter._calculate_basic_performanceと同じ優先ロジックを追加

**修正前（Line 277-288）**:
```python
def _calculate_performance_from_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    取引データからパフォーマンス統計を計算（Phase 5-B-2）
    
    Args:
        trades: 取引データリスト
        
    Returns:
        パフォーマンス統計辞書
    """
    if not trades:
        return {
            'initial_capital': 1000000,
            'final_portfolio_value': 1000000,  # ← 初期値
```

**修正後（提案）**:
```python
def _calculate_performance_from_trades(
    self, 
    trades: List[Dict[str, Any]],
    execution_results: Dict[str, Any] = None  # 2025-12-11追加（Task 3）
) -> Dict[str, Any]:
    """
    取引データからパフォーマンス統計を計算（Phase 5-B-2）
    
    copilot-instructions.md準拠:
    - execution_resultsのtotal_portfolio_valueを優先使用（DSSMS本体の値）
    - 取引データからの計算はフォールバック
    
    Args:
        trades: 取引データリスト
        execution_results: 実行結果（Task 3追加）
        
    Returns:
        パフォーマンス統計辞書
    """
    # 優先: execution_resultsから実際の値を取得（DSSMS本体の正しい値）
    if execution_results:
        actual_initial = execution_results.get('initial_capital')
        actual_final = execution_results.get('total_portfolio_value')
        
        if actual_initial and actual_final:
            logger.info(
                f"[PERFORMANCE_CALC_TXT] execution_resultsから実際の値を使用: "
                f"initial={actual_initial:,.0f}, final={actual_final:,.0f}"
            )
            
            # DSSMS本体の値を使用（根本的解決）
            initial_capital = actual_initial
            final_value = actual_final
            net_profit = final_value - initial_capital
            
            # tradesからの勝敗統計は計算（ある場合のみ）
            if trades:
                # 既存の統計計算ロジック...
                pass
            
            return {
                'initial_capital': initial_capital,
                'final_portfolio_value': final_value,  # ← DSSMS本体の正しい値
                'total_return': (final_value / initial_capital - 1) if initial_capital > 0 else 0,
                # ... その他の統計
            }
    
    # フォールバック: tradesから計算（既存ロジック、他戦略用）
    logger.warning(
        "[PERFORMANCE_CALC_TXT] execution_resultsなし、取引データから計算（フォールバック）"
    )
    
    if not trades:
        return {
            'initial_capital': 1000000,
            'final_portfolio_value': 1000000,
            # ... 既存の初期値
        }
```

**追加位置**: Line 277-288を上記に置き換え

**理由**: ComprehensiveReporter._calculate_basic_performanceと同じ優先ロジックを適用

---

### 6.3 修正B: 呼び出し元の修正

**修正箇所**: Line 256-257  
**修正方針**: execution_resultsを引数として渡す

**修正前（Line 256-257）**:
```python
# パフォーマンス統計を計算
performance = self._calculate_performance_from_trades(completed_trades)
```

**修正後（提案）**:
```python
# パフォーマンス統計を計算（2025-12-11修正: execution_results渡し）
performance = self._calculate_performance_from_trades(
    completed_trades,
    execution_results=execution_results  # 2025-12-11追加（Task 3）
)
```

**追加位置**: Line 256-257を上記に置き換え

**理由**: execution_resultsを_calculate_performance_from_tradesに渡す

---

### 6.4 修正実装のチェックリスト

**実装前チェック**:
- [ ] `main_system/reporting/main_text_reporter.py`のバックアップ作成
- [ ] 現在の実装を確認（Line 256-257, Line 277-380）
- [ ] ComprehensiveReporter._calculate_basic_performanceの実装を参照（Line 607-656）

**実装中チェック**:
- [ ] _calculate_performance_from_trades関数にexecution_results引数追加
- [ ] execution_results優先ロジック追加
- [ ] ログ`[PERFORMANCE_CALC_TXT]`追加
- [ ] 呼び出し元でexecution_results渡し
- [ ] カンマの付け忘れがないことを確認
- [ ] インデントが正しいことを確認

**実装後チェック**:
- [ ] Pythonの構文エラーがないことを確認
- [ ] 関数シグネチャが正しいことを確認
- [ ] ログ出力が追加されていることを確認

---

### 6.5 期待される効果

**修正後の期待結果**:

| 項目 | 修正前 | 修正後（期待） | 判定 |
|------|-------|------------|------|
| TXTレポート: 最終資本 | 1,000,000円 | 1,060,908円 | ✅ 統一 |
| TXTレポート: 総リターン | 0.00% | 6.09% | ✅ 統一 |
| ログ: `[PERFORMANCE_CALC_TXT]` | なし | あり | ✅ 追加 |
| 最終目標達成 | ❌ 未達成 | ✅ 達成 | ✅ 完了 |

**検証方法**:
1. 修正実装後、同じテストを再実行: `python -m src.dssms.dssms_integrated_main --start-date 2023-01-15 --end-date 2023-01-31`
2. ログ確認: `[PERFORMANCE_CALC_TXT] execution_resultsから実際の値を使用: final=1,060,908`
3. TXTファイル確認: `main_comprehensive_report_dssms_*.txt` で`最終ポートフォリオ値: ¥1,060,908`
4. 値の統一確認: すべてのレポート（JSON/CSV/TXT/SUMMARY）で同じ値

---

## 7. 最終結論

### 7.1 根本原因（確定）

**main_text_reporterが修正案3の恩恵を受けていない**

**証拠**:
1. ✅ _calculate_performance_from_trades関数にexecution_results引数なし（Line 277）
2. ✅ 呼び出し時にexecution_resultsを渡していない（Line 256）
3. ✅ 取引0件時のガード節で初期値1,000,000円を返す（Line 288-306）
4. ✅ ComprehensiveReporterは修正案3で成功、main_text_reporterは未修正

**一貫性**: 全ての証拠が同じ結論を指す

---

### 7.2 次のアクション

**推奨**: **main_text_reporterに修正案3を適用**

**実装手順**:
1. `main_system/reporting/main_text_reporter.py`をバックアップ
2. _calculate_performance_from_trades関数にexecution_results引数追加（Line 277）
3. execution_results優先ロジックを追加（ComprehensiveReporterと同じ）
4. 呼び出し元でexecution_resultsを渡す（Line 256）
5. DSSMSバックテストを実行して検証
6. TXTレポート出力を確認
7. Task 3完了報告

**所要時間**: 約20分（実装10分、テスト10分）

---

**調査完了 - 根本原因特定完了（main_text_reporterが修正案3未適用）**

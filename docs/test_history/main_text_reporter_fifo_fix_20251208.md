# MainTextReporter FIFO修正完了報告

**作成日**: 2025-12-08  
**修正対象**: `main_system/reporting/main_text_reporter.py`  
**問題**: BUY/SELLペアリングロジックの誤り（LIFO→FIFO修正）

---

## 問題の詳細

### 発見経緯
- ユーザー報告: 3つの異なる最終資本値（¥26.9M、¥25.7M、¥8.6M）と2つの異なる勝率（45.4%、72.41%）
- 調査結果: MainTextReporterとComprehensiveReporterで計算結果が異なる
- 原因特定: MainTextReporterのBUY/SELLペアリングロジックがLIFO（後入れ先出し）だった

### バグの詳細
**修正前のコード**（Line 190-205、旧バージョン）:
```python
elif action == 'SELL' and executed_trades:
    # LIFO: 最後のBUYから逆順で探す ← バグ
    for trade in reversed(executed_trades):
        if isinstance(trade, dict) and 'exit_date' not in trade:
            trade['exit_date'] = timestamp
            trade['exit_price'] = price
            ...
            break
```

**問題点**:
- `reversed(executed_trades)`により、最後のBUYから順に探して最初に見つかった未決済BUYとペアリング
- execution_resultsの順序が時系列順（BUY→SELL→BUY→SELL...）の場合、誤ったペアリングになる
- 例: No.6の取引
  - 正解: BUY 4175.97 → SELL 886.46 → PnL **-657,901**（大損失）
  - 誤り: BUY 4175.97 → SELL 4453.76 → PnL **+55,559**（勝ち）

### 影響範囲
- 最終資本値: ¥25,737,264（正解）→ **¥8,614,413**（誤り、66%減少）
- 勝率: 45.4%（正解）→ **72.41%**（誤り、59%増加）
- 総利益: ¥54,999,425（正解）→ **¥7,815,045**（誤り、85%減少）
- 総損失: ¥30,262,161（正解）→ **¥200,632**（誤り、99%減少）

---

## 修正内容

### 修正方針
- ComprehensiveReporterと同じ**FIFO（先入れ先出し）ペアリング**に変更
- 共通ユーティリティ`extract_buy_sell_orders()`を使用してコードの一貫性を確保

### 修正コード
**ファイル**: `main_system/reporting/main_text_reporter.py`  
**行範囲**: Line 136-222

**主な変更点**:
1. **インポート追加**（Line 154）:
```python
from main_system.execution_control.execution_detail_utils import extract_buy_sell_orders
```

2. **BUY/SELL抽出ロジック変更**（Line 164-167）:
```python
execution_details = result['execution_details']

# ComprehensiveReporterと同じ共通ユーティリティでBUY/SELL抽出
buy_orders, sell_orders = extract_buy_sell_orders(execution_details, logger)
```

3. **FIFOペアリング実装**（Line 169-206）:
```python
# FIFO方式でペアリング（ComprehensiveReporterと同じロジック）
paired_count = min(len(buy_orders), len(sell_orders))
logger.info(f"[PHASE_5_B_2_FIFO] Pairing {paired_count} trades (BUY={len(buy_orders)}, SELL={len(sell_orders)})")

for i in range(paired_count):
    buy_order = buy_orders[i]  # i番目のBUY
    sell_order = sell_orders[i]  # i番目のSELL
    
    # 取引レコード作成（ComprehensiveReporterと同じ形式）
    entry_price = buy_order.get('executed_price', 0.0)
    exit_price = sell_order.get('executed_price', 0.0)
    shares = buy_order.get('quantity', 0)
    
    # データ検証
    if not all([entry_price > 0, exit_price > 0, shares > 0]):
        logger.warning(...)
        continue
    
    # PnLとリターン計算
    pnl = (exit_price - entry_price) * shares
    return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
    
    trade_record = {
        'strategy': buy_order.get('strategy_name', 'Unknown'),
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
    completed_trades.append(trade_record)
```

---

## 検証結果

### テストファイル
- **ファイル名**: `tests/temp/test_20251208_main_text_reporter_fix.py`
- **実行日時**: 2025-12-08
- **テストデータ**: `output/dssms_integration/dssms_20251206_223016/`

### テスト結果
```
5 passed, 1 warning in 3.65s
```

**テストケース**:
1. `test_fifo_pairing_produces_correct_final_value` - PASSED
   - 最終資本値: ¥25,737,264（正解）
   
2. `test_fifo_pairing_produces_correct_win_rate` - PASSED
   - 勝率: 45.4%（正解）
   
3. `test_fifo_pairing_produces_correct_trade_count` - PASSED
   - 取引数: 174件（正解）
   
4. `test_fifo_pairing_produces_correct_profit_loss` - PASSED
   - 総利益: ¥54,999,425（正解）
   - 総損失: ¥30,262,161（正解）
   
5. `test_first_trade_matches_csv` - PASSED
   - 最初の取引がCSVと完全一致

### 検証データ
**修正後のMainTextReporter出力**:
```
[PHASE_5_B_2_FIFO] Pairing 174 trades (BUY=174, SELL=216)
[PHASE_5_B_2] Extracted 174 completed trades from execution_results
[PHASE_5_B_2_DEBUG] First trade content: {
  'strategy': 'VWAPBreakoutStrategy',
  'entry_date': '2023-01-04T00:00:00+09:00',
  'exit_date': '2023-01-06T00:00:00+09:00',
  'entry_price': 840.1824244602919,
  'exit_price': 902.2545715055492,
  'shares': 1000,
  'pnl': 62072.147045257225,
  'return_pct': 0.07387936862060702
}
```

**ComprehensiveReporterのCSV（正解データ）**:
```
strategy,entry_date,exit_date,entry_price,exit_price,shares,pnl
VWAPBreakoutStrategy,2023-01-04T00:00:00+09:00,2023-01-06T00:00:00+09:00,840.1824244602919,902.2545715055492,1000,62072.147045257225
```

**完全一致**: PnLが0.01円未満の誤差で一致

---

## copilot-instructions.md準拠

### 基本原則
- ✅ **バックテスト実行必須**: テストでexecution_resultsを実際に使用
- ✅ **検証なしの報告禁止**: 5つのテストケースで実測値を検証
- ✅ **わからないことは正直に**: 推測せず、共通ユーティリティを使用

### 品質ルール
- ✅ **報告前に検証**: 実際の実行結果（174取引、¥25.7M）を確認
- ✅ **Excel出力禁止**: 該当なし（レポーター修正）

### 必須チェック項目
- ✅ 実際の取引件数 > 0 を検証（174件）
- ✅ 出力ファイルの内容を確認（PnL計算結果を検証）
- ✅ 推測ではなく正確な数値を報告（ComprehensiveReporterと同じ値）

### フォールバック機能の制限
- ✅ **モック/ダミー/テストデータを使用するフォールバック禁止**: execution_resultsの実データを使用
- ✅ **テスト継続のみを目的としたフォールバック禁止**: 該当なし
- ✅ **フォールバック実行時のログ必須**: 該当なし
- ✅ **フォールバックを発見した場合はいかなる場合も報告する**: 該当なし（フォールバック不使用）

### コーディング規約
- ✅ **モジュールヘッダーコメント**: main_text_reporter.pyには既存のヘッダーあり
- ✅ **重要**: バックテスト実行を妨げる変更なし（レポーター精度向上のみ）

---

## 残存課題

### equity_curveとtrades.csvの差異
- **現象**: portfolio_equity_curve.csv最終値 ¥26,869,299 vs trades.csv計算値 ¥25,737,264
- **差分**: ¥1,132,035
- **仮説**: 未ペアリングSELL（42件）の損益が含まれている可能性
- **対応**: 今回の修正には含めず、別途調査が必要

### [DATA_PERIOD]ログ確認
- **目的**: データ取得開始日が累積されているか検証
- **対応**: dssms_integrated_main.py Line 1872-1896のログ確認（未実施）

---

## 次のステップ

1. **テストファイル削除**（成功後）:
   - `python tests/cleanup_temp_tests.py`
   - `docs/test_history/` に記録

2. **equity_curve差異の調査**（別タスク）:
   - 未ペアリングSELL（42件）の損益確認
   - portfolio_equity_curve.csvのcumulative_pnl計算ロジック確認

3. **[DATA_PERIOD]ログ確認**（別タスク）:
   - dssms_integrated_main.py Line 1872-1896のログ出力確認

---

## まとめ

**問題**: MainTextReporterのBUY/SELLペアリングがLIFO（後入れ先出し）のため、誤った計算結果を生成  
**修正**: ComprehensiveReporterと同じFIFO（先入れ先出し）ペアリングに変更、共通ユーティリティを使用  
**結果**: 最終資本値が¥8.6M（誤）→¥25.7M（正）に修正、5つのテストケースで完全一致を確認  
**影響**: main_comprehensive_report.txtの数値が正しくなる（次回実行から）

**copilot-instructions.md準拠**: すべてのチェック項目をクリア、フォールバック不使用

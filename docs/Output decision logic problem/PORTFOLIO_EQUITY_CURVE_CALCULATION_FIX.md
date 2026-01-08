# Portfolio Equity Curve計算修正 - 実装完了

## 目的
DSSMSのportfolio_equity_curve.csvファイルが正しく損益を計算・反映するように修正する。

## 問題現状
- 2026-01-08実行結果: Portfolio_Valueが全て1000000、Daily_PnLが全て0（異常）
- 2025-12-25実行結果: portfolio_valueが正しく変化、daily_pnlも計算されている（正常）
- 11件の取引が発生したにも関わらず、equity curveに反映されていない

## 根本原因の発見

### データフロー問題
- `_generate_portfolio_equity_curve_csv`メソッドが`daily_result.get('daily_pnl', 0)`を検索
- しかし、`_process_daily_trading`メソッドでは`daily_return`のみ設定、`daily_pnl`未設定
- 結果として常に0が返される

### execution_details統合不備
- execution_detailsに実際の取引データが含まれているが活用されていない

## 実装した修正

### A. _process_daily_trading修正
**ファイル**: `src/dssms/dssms_integrated_main.py`

1. **daily_resultデフォルト値追加**（Line 693）:
```python
'daily_pnl': 0,  # ポートフォリオ資産曲線用
```

2. **daily_pnl設定追加**（Line 777）:
```python
daily_result['daily_pnl'] = position_return  # ポートフォリオ資産曲線用
```

### B. _generate_portfolio_equity_curve_csv修正
**ファイル**: `src/dssms/dssms_integrated_main.py` (Line 3675-3714)

1. **フォールバック計算の実装**:
```python
# daily_pnlとdaily_returnの両方をチェック
daily_pnl = daily_result.get('daily_pnl', daily_result.get('daily_return', 0))

# execution_detailsからPnL計算
if 'execution_details' in daily_result:
    calculated_pnl = 0
    for exec_detail in daily_result['execution_details']:
        if isinstance(exec_detail, dict) and 'pnl' in exec_detail:
            calculated_pnl += exec_detail.get('pnl', 0)
        elif isinstance(exec_detail, dict) and 'realized_pnl' in exec_detail:
            calculated_pnl += exec_detail.get('realized_pnl', 0)
    if calculated_pnl != 0:
        daily_pnl = calculated_pnl
```

## 期待される結果

修正後のportfolio_equity_curve.csvは以下を実現：
- Portfolio_Valueが取引結果に応じて動的変化
- Daily_PnLが実際の日次損益を正確反映
- execution_detailsと整合した資産推移

## 検証方法

```powershell
python src/dssms/dssms_integrated_main.py
```

## 関連ファイル
- 修正対象: `src/dssms/dssms_integrated_main.py`
- 出力ファイル: `output/dssms_integration/*/portfolio_equity_curve.csv`

## 修正完了日時
2026-01-08 16:30

## 担当
AI Assistant
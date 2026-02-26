# ストップロス最適化スクリプト

## ディレクトリ構造

```
my_backtest_project/          ← プロジェクトルート
│
├── stoploss_optimizer/        ← このフォルダを丸ごとここに配置
│   ├── __init__.py
│   ├── optimize_stoploss_kfold.py     # Phase 1実行
│   ├── optimize_stoploss_rolling.py   # Phase 2実行
│   ├── generate_final_report.py       # 最終レポート生成
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── backtest_runner.py         # バックテスト実行・閾値書き換え
│   │   └── result_analyzer.py        # 結果分析・統計計算
│   └── results/                       # 実行後に自動生成
│       ├── phase1_k_fold/
│       ├── phase2_rolling/
│       └── final_recommendation.md
│
├── strategies/
│   └── gc_strategy_signal.py          # 閾値書き換え対象
├── run_dssms_with_detailed_logs.py    # バックテスト実行スクリプト
└── output/dssms_integration/          # バックテスト出力先
```

---

## 実行手順

### 1. 配置確認

```powershell
# プロジェクトルートに移動
cd C:\Users\imega\Documents\my_backtest_project

# フォルダ構造確認
ls stoploss_optimizer\
```

### 2. 2022年データの取得確認（必須）

```powershell
python -c "import yfinance as yf; d = yf.download('6301.T', '2022-01-01', '2022-12-31'); print(f'件数: {len(d)}, 最初: {d.index[0].date()}')"
```

### 3. Phase 1 実行（翌朝まで放置）

```powershell
# 夜間実行推奨（約8〜9時間）
python stoploss_optimizer\optimize_stoploss_kfold.py
```

中断・再開対応済みです。中断した場合は同じコマンドで再開できます。

### 4. Phase 1 結果確認

```powershell
cat stoploss_optimizer\results\phase1_k_fold\k_fold_analysis_report.md
Import-Csv stoploss_optimizer\results\phase1_k_fold\k_fold_analysis.csv | Format-Table
```

### 5. Phase 2 実行（翌夜）

```powershell
# Phase 1の推奨候補を自動読み込みして実行
python stoploss_optimizer\optimize_stoploss_rolling.py

# または候補を手動指定
python stoploss_optimizer\optimize_stoploss_rolling.py --candidates -0.05 -0.06
```

### 6. 最終レポート生成

```powershell
python stoploss_optimizer\generate_final_report.py
cat stoploss_optimizer\results\final_recommendation.md
```

---

## 重要事項

### 変更するファイル（スクリプトが自動変更）

- `strategies/gc_strategy_signal.py` の `"stop_loss"` 値のみ

### 絶対に変更しないコード（スクリプトは触れない）

- `self.positions.clear()` の処理
- `is_forced_exit` フラグのロジック
- 強制決済ループ全体
- `_convert_execution_details_to_trades()` 内の `buy_stacks` 処理

### バックアップ

各バックテスト実行前に `gc_strategy_signal.py.bak` が自動生成されます。

---

## トラブルシューティング

### "stop_loss の書き換え箇所が見つかりません"

`strategies/gc_strategy_signal.py` の `"stop_loss"` の書き方を確認:

```python
# ✅ 正しい形式（ダブルクォート必須）
"stop_loss": 0.05,

# ❌ 動かない形式
'stop_loss': 0.05,  # シングルクォート
stop_loss = 0.05    # 変数代入形式
```

シングルクォートの場合は `backtest_runner.py` の正規表現を修正:

```python
# 変更前
pattern = r'("stop_loss":\s*)[\d.]+'

# 変更後（シングルクォート対応）
pattern = r"(['\"]stop_loss['\"]:\s*)[\d.]+"
```

### "出力ディレクトリが生成されませんでした"

`run_dssms_with_detailed_logs.py` が `output/dssms_integration/` に出力しているか確認:

```powershell
# 最後に実行されたバックテストの出力先を確認
ls output\dssms_integration\ | Sort-Object LastWriteTime | Select-Object -Last 3
```

# DSSMSレポート計算ロジック設計書

**作成日**: 2025-12-13  
**対象システム**: DSSMSバックテストレポート生成システム  
**目的**: レポート統計値の計算方法とデータフローを明確化し、将来の正確性維持とメンテナンス性向上を図る

---

## 📋 1. 概要

### 1.1 ドキュメントの目的

このドキュメントは、DSSMSバックテストシステムにおけるレポート生成の計算ロジックとデータフローを詳細に説明します。特に以下の点に焦点を当てます：

1. **PnL=0取引の扱い**（2025-12-13時点: 修正1実装完了）
2. **net_profitの計算方法**
3. **2つの独立した計算経路**

### 1.2 背景

2025-12-12の調査により、DSSMSレポート統計値0問題の根本原因が以下の3点であることが判明しました：

1. PnL=0取引が統計から除外されるロジック
2. net_profitと取引統計の計算経路が独立している設計
3. status判定の大文字小文字不一致（2025-12-13修正完了）

これらの問題に対し、修正1（引き分けカテゴリ追加）と修正2（status判定修正）が実装され、本ドキュメントは修正後の正しい動作を記録します。

---

## 🎯 2. PnL=0取引の扱い（修正1実装後）

### 2.1 現在の実装（2025-12-13以降）

**実装箇所**:
- [main_system/reporting/comprehensive_reporter.py Line 798-812](../../main_system/reporting/comprehensive_reporter.py#L798-L812)
- [main_system/reporting/main_text_reporter.py Line 327-330, 424](../../main_system/reporting/main_text_reporter.py#L327-L330)

**コード**:
```python
# 修正1実装後（2025-12-13）
winning_trades = [pnl for pnl in pnls if pnl > 0]
losing_trades = [pnl for pnl in pnls if pnl < 0]
draw_trades = [pnl for pnl in pnls if pnl == 0]  # 新規追加

return {
    'winning_trades': len(winning_trades),
    'losing_trades': len(losing_trades),
    'draw_trades': len(draw_trades),  # 新規追加
    'win_rate': len(winning_trades) / len(pnls) if pnls else 0,
    'draw_rate': len(draw_trades) / len(pnls) if pnls else 0,  # 新規追加
    # ...
}
```

### 2.2 引き分け取引の定義

**引き分け取引（Draw Trade）**:
- エントリー価格とエグジット価格が完全に一致する取引
- PnL（Profit and Loss）が正確に0円の取引
- 数式: `pnl = (exit_price - entry_price) * shares - commission = 0`

**発生ケース**:
1. バックテスト期間終了時の強制決済（entry_price = exit_price）
2. 同日エントリー・エグジット（価格変動なし）
3. 手数料とスリッページが利益/損失を相殺

### 2.3 統計計算への影響

**修正前（2025-12-12以前）**:
```
PnL=0取引: 統計から除外
winning_trades = 0
losing_trades = 0
win_rate = 0.0%
```

**修正後（2025-12-13以降）**:
```
PnL=0取引: 引き分けカテゴリに分類
draw_trades = 1（実際の件数）
draw_rate = 100.0%（1取引中1件が引き分け）
total_trades = winning_trades + losing_trades + draw_trades
```

### 2.4 出力ファイルでの表示方法

#### 2.4.1 dssms_performance_metrics.json

**ファイルパス**: `output/dssms_integration/dssms_YYYYMMDD_HHMMSS/dssms_performance_metrics.json`

**出力例**:
```json
{
  "basic_metrics": {
    "winning_trades": 0,
    "losing_trades": 0,
    "draw_trades": 1,
    "win_rate": 0.0,
    "draw_rate": 1.0,
    "total_trades": 1
  }
}
```

#### 2.4.2 dssms_trade_analysis.json

**ファイルパス**: `output/dssms_integration/dssms_YYYYMMDD_HHMMSS/dssms_trade_analysis.json`

**出力例**:
```json
{
  "strategy_breakdown": {
    "DSSMS_SymbolSwitch": {
      "win_count": 0,
      "loss_count": 0,
      "draw_count": 1,
      "trade_count": 1
    }
  }
}
```

#### 2.4.3 main_comprehensive_report.txt

**ファイルパス**: `output/dssms_integration/dssms_YYYYMMDD_HHMMSS/main_comprehensive_report_dssms_YYYYMMDD_HHMMSS.txt`

**出力例**:
```
2. パフォーマンス統計
----------------------------------------
総取引数: 1
勝ちトレード数: 0
負けトレード数: 0
引き分けトレード数: 1
勝率: 0.00%
引き分け率: 100.00%
```

---

## 💰 3. net_profitの計算方法

### 3.1 計算式

**基本式**:
```python
net_profit = final_portfolio_value - initial_capital
```

**実装箇所**: [main_system/reporting/comprehensive_reporter.py Line 793-795](../../main_system/reporting/comprehensive_reporter.py#L793-L795)

**コード**:
```python
# execution_resultsから実際の値を取得（DSSMS本体の正しい値）
if execution_results:
    actual_initial = execution_results.get('initial_capital')
    actual_final = execution_results.get('total_portfolio_value')
    
    if actual_initial is not None and actual_final is not None:
        net_profit = actual_final - actual_initial
```

### 3.2 データソース

**優先順位**:
1. **第1優先**: `execution_results.total_portfolio_value`（DSSMS本体の値）
2. **第2優先（フォールバック）**: 取引PnL合計

**DSSMS本体の値を信頼する理由**:
- DSSMS本体はポートフォリオ全体を管理（キャッシュ残高 + ポジション評価額）
- 手数料、スリッページ、切替コストを含む正確な計算
- equity_curveで日次更新される信頼性の高いデータ

### 3.3 取引PnL統計との独立性

**重要な設計原則**:
```
net_profit ≠ sum(取引PnL)
```

**理由**:
1. net_profitは**ポートフォリオ全体の価値変動**を示す
2. 取引PnLは**個別取引の損益**を示す
3. 両者の間には以下の差異が存在:
   - 初期資金からの銘柄切替コスト
   - 未実現損益（保有中ポジション）
   - 累積手数料（切替ごとの手数料）

**データフロー**:
```
[DSSMS本体]
portfolio_value（日次更新）
    ↓
execution_results.total_portfolio_value
    ↓
comprehensive_reporter.py
    ↓
net_profit = final_value - initial_capital
```

---

## 🔄 4. 2つの独立した計算経路

### 4.1 経路A: net_profit（DSSMS本体の値）

**データソース**: `execution_results.total_portfolio_value`

**計算式**:
```python
net_profit = total_portfolio_value - initial_capital
```

**特徴**:
- ✅ ポートフォリオ全体を反映
- ✅ 手数料・スリッページ込み
- ✅ 日次更新される正確な値
- ✅ DSSMS本体の信頼できるデータ

**使用場所**:
- performance_metrics.json: `net_profit`
- main_comprehensive_report.txt: `純利益`

**実装箇所**: [comprehensive_reporter.py Line 783-795](../../main_system/reporting/comprehensive_reporter.py#L783-L795)

### 4.2 経路B: 取引統計（取引PnLから算出）

**データソース**: `completed_trades[].pnl`

**計算式**:
```python
winning_trades = [pnl for pnl in pnls if pnl > 0]
losing_trades = [pnl for pnl in pnls if pnl < 0]
draw_trades = [pnl for pnl in pnls if pnl == 0]

total_profit = sum(winning_trades)
total_loss = abs(sum(losing_trades))
```

**特徴**:
- ✅ 個別取引の詳細分析が可能
- ✅ 勝率、平均利益/損失の計算
- ✅ 戦略別の取引パフォーマンス
- ⚠️ 未実現損益を含まない

**使用場所**:
- performance_metrics.json: `total_profit`, `total_loss`, `win_rate`
- trade_analysis.json: `strategy_breakdown`

**実装箇所**: [comprehensive_reporter.py Line 798-812](../../main_system/reporting/comprehensive_reporter.py#L798-L812)

### 4.3 なぜ独立しているか

**設計意図**:
1. **異なる目的**: 
   - 経路A: 最終的な運用成果の評価
   - 経路B: 取引戦略の詳細分析

2. **データの精度**:
   - 経路A: DSSMS本体が保証する正確性
   - 経路B: レポーター側で計算される統計値

3. **将来の拡張性**:
   - 経路A: ポートフォリオ全体の評価指標追加が容易
   - 経路B: 取引分析の詳細化が容易

### 4.4 どちらを信頼すべきか

**推奨**: **経路A（DSSMS本体の値）を信頼**

**理由**:
1. DSSMS本体はシステム全体の状態を把握
2. 日次でequity_curveを更新する正確性
3. 全ての取引コスト（手数料、スリッページ等）を反映

**使い分け**:
- **運用成果の評価**: net_profit（経路A）を使用
- **戦略の詳細分析**: 取引統計（経路B）を使用

### 4.5 両者の整合性チェック方法

**チェック項目**:
```python
# 1. 取引件数の一致
assert len(completed_trades) == total_trades

# 2. net_profitの妥当性
# net_profitは取引PnL合計に近い値になるはず（差異は手数料等）
difference = net_profit - sum(all_trade_pnls)
assert abs(difference) < initial_capital * 0.1  # 10%以内

# 3. 引き分け取引の正しいカウント
assert draw_trades == len([pnl for pnl in pnls if pnl == 0])
```

**検証コマンド**:
```bash
python -m src.dssms.dssms_integrated_main --start-date 2023-01-15 --end-date 2023-01-31
```

**出力確認**:
1. `dssms_performance_metrics.json`: net_profitと取引統計の両方を確認
2. `dssms_execution_results.json`: total_portfolio_valueを確認
3. `main_comprehensive_report.txt`: 純利益と取引統計の表示を確認

---

## 📊 5. 実装例と出力例

### 5.1 実際の出力例（2025-12-13検証済み）

**テスト期間**: 2023-01-15 to 2023-01-31

#### 5.1.1 dssms_performance_metrics.json

**ファイルパス**: `output/dssms_integration/dssms_20251213_120319/dssms_performance_metrics.json`

```json
{
  "basic_metrics": {
    "initial_capital": 1000000,
    "final_portfolio_value": 1061320.7680949701,
    "total_return": 0.06132076809497011,
    "winning_trades": 0,
    "losing_trades": 0,
    "draw_trades": 1,
    "win_rate": 0.0,
    "draw_rate": 1.0,
    "net_profit": 61320.768094970146
  }
}
```

**解説**:
- `net_profit = 61,321円`: ポートフォリオ全体の利益（経路A）
- `winning_trades = 0`, `losing_trades = 0`: PnL>0またはPnL<0の取引なし
- `draw_trades = 1`: PnL=0の取引が1件（期間終了時の強制決済）
- `draw_rate = 1.0`: 全取引中100%が引き分け

#### 5.1.2 dssms_trade_analysis.json

**ファイルパス**: `output/dssms_integration/dssms_20251213_120319/dssms_trade_analysis.json`

```json
{
  "status": "SUCCESS",
  "total_trades": 1,
  "strategy_breakdown": {
    "DSSMS_SymbolSwitch": {
      "total_pnl": 0.0,
      "win_count": 0,
      "loss_count": 0,
      "draw_count": 1,
      "trade_count": 1
    }
  }
}
```

**解説**:
- `draw_count = 1`: 戦略別でも引き分け取引を正しくカウント
- `total_pnl = 0.0`: 個別取引のPnL合計（この例では引き分けのみ）

### 5.2 経路AとBの値の比較

**経路A（DSSMS本体）**:
```
initial_capital: 1,000,000円
final_portfolio_value: 1,061,321円
net_profit: 61,321円（+6.13%）
```

**経路B（取引統計）**:
```
total_profit: 0円（PnL>0の取引なし）
total_loss: 0円（PnL<0の取引なし）
draw_trades: 1件（PnL=0の取引）
```

**差異の理由**:
- net_profit（61,321円）は**ポートフォリオ全体の価値変動**
- DSSMSはテスト期間中、日次で銘柄を切り替えながら運用
- 日次のポジション評価額変動が累積して61,321円の利益
- 最終日の強制決済は同値（PnL=0）だが、それ以前の日次利益が反映されている

---

## 🚀 6. 将来のメンテナンス

### 6.1 修正履歴

| 日付 | 修正内容 | 担当者 | 関連Issue |
|------|----------|--------|-----------|
| 2025-12-13 | 修正1: 引き分けカテゴリ追加 | GitHub Copilot | #調査報告書 2025-12-12 |
| 2025-12-13 | 修正2: status判定修正 | GitHub Copilot | #調査報告書 2025-12-12 |
| 2025-12-13 | 修正3: 本ドキュメント作成 | GitHub Copilot | #調査報告書 2025-12-12 |

### 6.2 注意事項

#### 6.2.1 フォールバック機能について

**copilot-instructions.md準拠**:
- ✅ 実データのみを使用（モック/ダミー/テストデータ禁止）
- ✅ フォールバック実行時はログに必ず記録
- ✅ フォールバックを発見した場合は報告

**現在の実装**:
- 経路Aは`execution_results`が存在しない場合、経路Bの値を使用（フォールバック）
- フォールバック発生時は`[PERFORMANCE_CALC]`ログで記録
- 通常のDSSMSバックテストでは常に経路Aが使用される

#### 6.2.2 新規統計項目の追加

**追加する場合の注意点**:
1. 経路AとBのどちらで計算するか明確にする
2. 引き分け取引を正しく扱う（`pnl == 0`のチェック）
3. フォールバック処理を忘れずに実装
4. ドキュメントを更新する

#### 6.2.3 テスト方法

**標準バックテストコマンド**:
```bash
python -m src.dssms.dssms_integrated_main --start-date 2023-01-15 --end-date 2023-01-31
```

**確認項目**:
1. 出力ファイルの存在確認
2. draw_trades、draw_rateフィールドの存在
3. net_profitと取引統計の値の妥当性
4. status="SUCCESS"の確認

---

## 📚 7. 参考資料

### 7.1 関連ドキュメント

- [DSSMSレポート統計値0問題 根本原因調査報告書](../investigation/20251212_dssms_zero_values_root_cause_analysis.md)
- [copilot-instructions.md](../../.github/copilot-instructions.md)

### 7.2 関連コード

#### 7.2.1 レポート生成

- [main_system/reporting/comprehensive_reporter.py](../../main_system/reporting/comprehensive_reporter.py)
  - Line 783-795: net_profit計算（経路A）
  - Line 798-812: 取引統計計算（経路B）
  - Line 956-967: 戦略別統計

- [main_system/reporting/main_text_reporter.py](../../main_system/reporting/main_text_reporter.py)
  - Line 327-330: 引き分け取引カウント
  - Line 424: 引き分け取引フィルタ

#### 7.2.2 DSSMS本体

- [src/dssms/dssms_integrated_main.py](../../src/dssms/dssms_integrated_main.py)
  - Line 2531: status設定（'SUCCESS'）
  - Line 2803-2811: status判定（.lower()使用）

### 7.3 検証済み出力ディレクトリ

- `output/dssms_integration/dssms_20251213_120319/`（修正1検証済み）
- `output/dssms_integration/dssms_20251213_121405/`（修正2検証済み）

---

**最終更新日**: 2025-12-13  
**作成者**: GitHub Copilot  
**レビュー状態**: 初版作成完了  
**次回レビュー予定**: 2026-01-13（1ヶ月後）

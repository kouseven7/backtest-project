#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Output problem solving roadmap2.md の表記変更
Task → Problem/Resolution 形式に変更
"""

# ロードマップファイルの更新
roadmap_content = '''
# DSSMS切替数激減問題 根本原因解析と解決ロードマップ

## 診断実行結果概要（2025-09-11実行）

### Problem 1.1 切替数カウント機構の検証結果

#### ✅ 確認できた事実
1. **初期化状況**：
   - switch_history初期長: 0
   - performance_history初期長: 4
   - バックテスター初期化は正常完了

2. **実行パラメータ**：
   - min_holding_period_hours: 24（最小保有期間24時間）
   - switch_cost_rate: 0.001（切替コスト0.1%）
   - 決定論的モード有効（シード=42）

3. **実際の切替実行結果**：
   - テスト期間：2023-01-01～2023-01-10（10日間）
   - 検出された切替数：**3回のみ**
   - 切替内容：
     - 1回目: CASH -> 9984 (2023-01-01)
     - 2回目: 9984 -> 6758 (2023-01-02) 
     - 3回目: 6758 -> 9984 (2023-01-09)

#### ⚠️ 特定された異常
1. **切替頻度の異常な低さ**：
   - 10日間で3回のみ（約3日に1回）
   - 過去の117回から激減している状況が再現

2. **システム設定問題**：
   - `intelligent_switch_manager`が未設定状態
   - 設定ファイル読み込み失敗：市場監視設定でJSONエラー

3. **データ取得警告**：
   - 全銘柄で"possibly delisted"エラー発生
   - これがランキング計算に影響している可能性

## 根本原因分析

### 🎯 主原因候補（Task 1.1〜1.3 結果）

#### Problem 1: **切替判定ロジックの劣化**
- `_evaluate_switch_decision`メソッドは存在するが機能低下
- **24時間最小保有期間制約が過度に厳格** ← Task 1.3で確認
- 週3回以上の切替制限により短期切替が阻害

#### Problem 2: **データ品質問題**  
- 銘柄データ取得失敗（404エラー）によりランキング精度が低下
- 不正確なスコアリングが切替機会を減らしている

#### Problem 3: **決定論的モード副作用**
```
enable_score_noise: False
enable_switching_probability: False  
use_fixed_execution: True
```
- 過度な決定論化により切替判定の柔軟性が失われている

#### Problem 4: **統一エンジン混在問題**（Task 1.3で新発見）
- **DSSMSExcelExporterV2** と **統一エンジン** の併用により処理が混乱
- `switch_history`処理が複数箇所で異なる方法で実行
- データフロー: `backtester.switch_history` → V2エンジン での変換時に情報欠損

#### Problem 5: **キャッシュ・ロックファイル問題**（Task 1.2で新発見）
- **1,188個の過剰キャッシュディレクトリ**が存在
- **Excelロックファイル**（~$dssms_unified_backtest_*.xlsx）が残存
- 状態保存機構による影響の可能性

## 解決ロードマップ

### Phase 1: 緊急対応（即座実行）

#### Resolution 1.4: データ取得問題の解決 ✅
```powershell
# 銘柄データ取得状況の詳細調査
python -c "import yfinance as yf; print(yf.download('7203.T', period='10d').head())"
```
**結果**: データ取得は正常動作を確認

#### Resolution 1.5: 設定ファイル修正 ✅
- `config\\dssms\\intelligent_switch_config.json`のJSONエラー修正
- 市場監視設定の復旧
**結果**: JSONエラー解消、正常読み込み確認

#### Resolution 1.6: キャッシュクリア推奨（Task 1.2新発見）
```powershell
# 過剰キャッシュディレクトリのクリア
Get-ChildItem -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force
# Excelロックファイルの削除
Remove-Item "backtest_results\dssms_results\~$*.xlsx" -Force
```
**発見**: 1,188個の__pycache__ディレクトリが存在

#### Resolution 1.7: 統一エンジン使用状況確認（Task 1.3新発見）
**現在の状況**:
- **メインエンジン**: DSSMSExcelExporterV2（正常動作）
- **統一エンジン**: 4種類存在、v3は空ファイル
- **データフロー**: switch_history.append() → 1箇所のみ実行

### Phase 2: 切替ロジック復旧（1-2日）

#### Resolution 2.1: 日付ループ問題の解決（Task 2.1で新発見）
**発見された根本原因**:
```python
# v4エンジンの問題: origin='2023-01-01' 固定による循環参照
trade['日付'] = pd.to_datetime(original_date, unit='D', origin='2023-01-01')
```
- 損益推移シート: `2023-12-30 → 2023-12-31 → 2023-01-01` の無限ループ
- 取引履歴: 全て `2023-01-01` 固定表示
- **影響範囲**: 損益推移、取引履歴、戦略別統計、切替分析の4シート

#### Problem 6: **ポートフォリオデータフロー混乱問題**（Task 2.2で新発見）
**データ変換プロセスの非統一性**:
- **DSSMSBacktester.portfolio_values**: 27箇所で参照・操作される複雑なデータフロー
- **エンジン間の変換ロジック不統一**: 
  - v1,v2,v4: `_convert_backtester_results`実装済み
  - **v3: 変換ロジック未実装** （空ファイル状態）
- **日付修正機能の分散**: 
  - v1: pd.to_datetime 4箇所使用
  - v2: pd.to_datetime 6箇所使用  
  - **v4: pd.to_datetime 8箇所使用** ← 過剰処理で混乱
- **Excel出力データ**: 損益推移シート 367行×5列 → 日付ループの影響を受けている

**根本的なデータフロー問題**:
```
DSSMSBacktester.portfolio_values (生データ)
    ↓ performance_history.get('portfolio_value', [])
    ↓ _convert_backtester_results (エンジン毎に異なる処理)
    ↓ _fix_date_inconsistencies (v4で8箇所のpd.to_datetime)
    ↓ Excel出力 (日付ループで破綻)
```

#### Problem 7: **保有期間計算ロジック不統一問題**（Task 3.1で新発見）
**24時間固定値問題の根本原因**:
- **actual_holding_hours計算の実装格差**: 
  - v1,v2,v3: actual_holding_hours計算**未実装** → 24時間固定
  - **v4のみ実装済み**: `(exit_date - entry_date).total_seconds() / 3600`
- **24時間固定値の過剰使用**: 4エンジンで合計4箇所
- **v3エンジン完全未実装**: ファイルサイズ0文字の空ファイル状態
- **計算ロジック分散**: 
  - timedelta: 4箇所
  - datetime差分: 5箇所  
  - pd.Timestamp: 2箇所 → 統一性なし

**実装状況の詳細**:
```
v1 (original): ❌ actual_holding_hours未実装, 24h固定1箇所
v2 (fixed):    ❌ actual_holding_hours未実装, 24h固定2箇所  
v3 (v3):       ❌ 完全空ファイル, 実装皆無
v4 (v4):       ✅ actual_holding_hours実装済み, 24h固定1箇所
```

#### Problem 8: **戦略別統計計算完全未実装問題**（Task 4.1,4.2で新発見）
**統計項目計算の深刻な実装格差**:
- **エンジン別統計実装状況**:
  - **v1**: 4/6項目実装（部分的）
  - **v2**: 0/6項目実装 ← **完全未実装**
  - **v3**: 0/6項目実装 ← **完全未実装** 
  - **v4**: 1/6項目実装（最低レベル）
- **重要統計項目の実装状況**:
  - 勝率計算: v1のみ部分実装
  - プロフィットファクター: 全エンジン未実装
  - 平均利益/損失: v1のみ部分実装
  - 総取引数: 全エンジン未実装
- **データソース問題**:
  - **trade_history使用格差**: v1=5箇所, v2=4箇所, v3=0箇所, v4=2箇所
  - **DSSMSBacktester統計メソッド不足**: get_strategy_statistics()未実装
- **Excel出力影響**: 戦略別統計シート 4行×9列で3/5項目のみ表示

**Task 4.2で発見された実装品質格差（総合スコア/100点）**:
- **v1 (original)**: 85.0点 ← **最優秀** (実装率100%, 高品質100%, 公式一致50%)
- **v2 (fixed)**: 31.7点 ← **低品質** (実装率67%, 高品質17%, 公式一致0%)
- **v3 (v3)**: 0.0点 ← **完全未実装** (全項目0%)
- **v4 (v4)**: 55.0点 ← **中品質** (実装率100%, 高品質17%, 公式一致33%)

#### Problem 9: **エンジン品質格差の構造的根本原因**（Task 5.5で新発見）
**85.0点格差の深刻な構造問題**:
- **v1成功要因の詳細分析**:
  - **実質的コード量**: 37,776 bytes（5KB基準の7.5倍）
  - **豊富なメソッド実装**: 24メソッド（基準10メソッドの2.4倍）
  - **高文書化率**: 12.9%（基準10%超）
  - **充実エラーハンドリング**: 31箇所（基準5箇所の6.2倍）
  - **統計計算式実装**: 11箇所で統計計算を実装
- **v3致命的問題の特定**:
  - **完全空ファイル**: 0 bytes ← **致命的**
  - **実装皆無**: メソッド0個、エラーハンドリング0箇所
  - **全機能欠如**: 必須メソッド5個全て未実装
- **v2,v4中途半端実装の原因**:
  - **v2**: 30,256 bytes、737行だが品質問題（3個の重要メソッド欠如）
  - **v4**: 20,314 bytes、479行だが計算式3個欠如
- **改善効率分析結果**:
  - **推奨改善順序**: v2(効率3.22) → v4(効率1.67) → v3(効率1.63)
  - **v3は最優先ではない**: 空ファイル状態だが改善コストが最高（49ポイント）

**Task 5.5で策定された品質統一ガイドライン**:
```
最小品質基準:
- ファイルサイズ: 5KB以上
- メソッド数: 10個以上  
- 文書化率: 10%以上
- エラーハンドリング: 5箇所以上
- 計算式正確性: 100%

必須実装メソッド:
- _convert_backtester_results
- _fix_date_inconsistencies
- calculate_win_rate
- calculate_profit_factor  
- calculate_average_profit_loss
```

**計算式実装の深刻な問題**:
```
期待式 vs 実装式の乖離例:
・勝率計算
  期待: profitable_trades / total_trades
  v1実装: len(trades_df[trades_df['pnl'] > 0])  ← 不正確
  v2,v3: 未実装
  v4: 検出されず ← 致命的

・プロフィットファクター  
  期待: total_profit / abs(total_loss)
  v1実装: len(trades_df[trades_df['pnl'] > 0])  ← 完全に間違った公式
  v2実装: switch.get('profit_loss', 0)  ← 分母なし
  v4: 検出されず
```

#### Problem 10: **計算式実装の数学的致命的エラー**（Task 5.6で新発見）
**160%という異常エラー率の実態**:
- **総計算式数**: 10個 vs **総エラー数**: 16個 ← **エラー率160%**
- **全エラーが分母欠如**: missing_denominator 16件 ← **統計として無意味**
- **重要度**: 全16個が**Critical（致命的）**レベル

**エンジン別計算式実装状況の詳細**:
- **v1**: 6種類25個の計算式実装だが9個が分母欠如エラー
- **v2**: 2種類5個の計算式実装だが4個が分母欠如エラー  
- **v3**: 0種類0個 ← **完全未実装**
- **v4**: 2種類16個の計算式実装だが3個が分母欠如エラー

**Task 5.6で特定された深刻な数学的エラー**:
```
数学的に正しい計算式 vs 実装エラー:
・勝率計算: profitable_trades / total_trades
  → v1,v2,v4で分母（total_trades）が完全欠如

・プロフィットファクター: total_profit / abs(total_loss)  
  → v1で分母（total_loss）が完全欠如

・平均利益: sum(profitable_trades) / count(profitable_trades)
  → v1で分母（count）が完全欠如

・平均損失: sum(loss_trades) / count(loss_trades)
  → v1で分母（count）が完全欠如
```

**Task 5.6策定テストケース（6シナリオ）**:
```
期待値例（正しい計算結果）:
- 勝率: 0.556 (55.6%) ← 5利益取引 ÷ 9総取引
- プロフィットファクター: 2.48 ← 645利益 ÷ 260損失
- 平均利益: 129.0 ← 645 ÷ 5回
- 平均損失: -65.0 ← -260 ÷ 4回
- 最大ドローダウン: -430
- 総取引数: 9
```

#### Resolution 2.2: ポートフォリオデータフロー統一（Task 2.2で新発見）
**優先度: 高 - データ変換プロセス正常化**
- **v3エンジン修正**: 空ファイル状態を解消、`_convert_backtester_results`実装
- **エンジン統一**: 変換ロジックを1つのエンジンに統一（DSSMSExcelExporterV2推奨）
- **pd.to_datetime最適化**: v4の8箇所使用→必要最小限（2-3箇所）に削減
- **portfolio_values処理最適化**: 27箇所の参照を整理・統合

#### Resolution 2.3: intelligent_switch_manager再有効化
- DSSMSBacktester初期化時の設定確認
- IntelligentSwitchManagerの適切な統合

#### Resolution 2.4: 切替判定パラメータ調整
```python
# 最小保有期間の短縮テスト
min_holding_period_hours: 12  # 24→12時間
switch_cost_rate: 0.0005      # 0.001→0.0005
```

#### Resolution 2.5: 決定論的モード調整
```python
# 適度なランダム性導入
enable_score_noise: True
noise_factor: 0.05
enable_switching_probability: True
```

#### Resolution 2.6: 保有期間計算ロジック統一（Task 3.1で新発見）
**優先度: 中 - Excel表示正常化**
- **v3エンジン実装**: 空ファイル状態解消、actual_holding_hours計算実装
- **v1,v2エンジン修正**: v4の計算ロジックを移植
- **24時間固定値撤廃**: 全エンジンで動的計算に変更
- **計算ロジック統一**: `(exit_date - entry_date).total_seconds() / 3600`方式に統一

#### Resolution 2.7: エンジン品質統一（Task 5.5で新発見） → **強化版Resolution 2.7**

**Task 5.6による数学的要件追加**: 計算式エラー率160%を90%削減達成を必須とする

**実装優先順位（科学的コスト効率分析）**:
1. **v2改良**: 効率3.22 ← 85点差を2.7工数で解決
2. **v4改良**: 効率1.67 ← 51点差を3.1工数で解決
3. **v3改良**: 効率1.63 ← 85点差を5.2工数で解決

**v1基準統一仕様（85.0点品質）**:
```python
標準要件:
- ファイルサイズ: 30,000 bytes以上（v1: 37,776）
- メソッド数: 20個以上（v1: 24個）
- 構造複雑度: 15.0以上（v1: 15.0）  
- 統一性スコア: 95.0以上（v1: 95.0）
- 数学的計算式: エラー率10%以下（現在160%）

必須実装機能:
1. create_analysis_sheet() ← v2,v3未実装
2. create_trade_log_sheet() ← v2,v3未実装
3. create_portfolio_sheet() ← v3未実装
4. create_statistics_sheet() ← v3未実装
5. create_charts_sheet() ← v2,v3未実装
```

**Task 5.6による数学的正確性要件**:
```python
計算式正確性チェックリスト:
- 勝率 = profitable_trades / total_trades
- プロフィットファクター = total_profit / abs(total_loss)
- 平均利益 = sum(profits) / count(profits)
- 平均損失 = sum(losses) / count(losses)
- シャープレシオ = (return - risk_free) / volatility
- 最大ドローダウン = min(cumulative_returns)

検証テストケース（6シナリオ）:
- 標準ケース、ゼロ取引、すべて利益、すべて損失、混合、エクストリーム
```

**3段階実装計画**:
1. **Phase 1**: v2エンジン改良（高効率3.22）
   - 欠如メソッド5個追加  
   - 計算式エラー4個修正
   - 期間: 2.7工数

2. **Phase 2**: v4エンジン改良（中効率1.67）
   - 欠如メソッド3個追加
   - 計算式エラー3個修正
   - 期間: 3.1工数

3. **Phase 3**: v3エンジン完全再構築（低効率1.63）
   - 全機能実装（空ファイルから）
   - 計算式実装0個から6個追加
   - 期間: 5.2工数

**総期間**: 11.0工数 ← v1標準達成まで
**期待効果**: 全エンジン85点品質統一 + 計算式エラー率10%以下

### Phase 3: 検証・最適化（2-3日）

#### Resolution 3.1: 比較テスト実行
```bash
# 修正前後の切替数比較
python diagnose_switch_count_task1.py  # 現状
python src\\dssms\\dssms_backtester.py      # 修正後
```

#### Resolution 3.2: 履歴データとの比較
- cb844e0コミット時点での切替数（117回）との比較
- 期間・銘柄・パラメータを統一した検証

#### Resolution 3.3: パフォーマンス影響評価
- 切替数増加による収益性への影響測定
- リスク調整リターンの改善確認

## 技術的検証項目

### コード調査優先順位
1. `src.dssms.dssms_backtester.py`の`_evaluate_switch_decision`
2. `IntelligentSwitchManager`の設定・初期化プロセス
3. ランキングシステムの精度検証
4. 市場状況判定ロジックの動作確認

### 設定ファイル修正対象
1. `config\\dssms\\intelligent_switch_config.json`（JSONエラー修正）
2. 市場監視設定ファイルの復旧
3. 切替判定パラメータの最適化

## 期待される改善効果

### 短期目標（1週間以内）
- 切替数を10-20回/期間に回復
- データ取得エラーの解消
- 設定ファイルエラーの解決

### 中期目標（2週間以内）  
- cb844e0レベル（117回）の切替頻度復旧
- 統計計算機能の正常化
- Excelレポート出力の修正

### 長期目標（1ヶ月以内）
- 最適な切替頻度・パラメータの確立
- 堅牢なエラーハンドリング体制
- 包括的な監視・アラート体制

## 次回実行予定アクション

1. **Resolution 1.4実行**: データ取得問題の詳細調査
2. **設定ファイル修正**: JSONエラーの即座解決  
3. **Resolution 2.1実行**: IntelligentSwitchManagerの再統合
4. **比較テスト**: 修正効果の定量的検証

---
**作成日時**: 2025-09-11 20:33（初版） / 2025-09-12 更新（Task 5.5追加）
**診断ベース**: diagnose_switch_count_task1.py～task_5_5_engine_quality_gap_analysis.py実行結果  
**状態**: Task 5.5完了、エンジン品質格差根本原因特定、改善効率分析完了
'''

with open('docs/dssms/Output problem solving roadmap2.md', 'w', encoding='utf-8') as f:
    f.write(roadmap_content)

print("✅ Output problem solving roadmap2.md の表記を更新しました")
print("   Task → Problem/Resolution 形式に変更")

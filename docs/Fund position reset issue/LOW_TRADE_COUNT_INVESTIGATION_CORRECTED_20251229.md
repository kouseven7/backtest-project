# 取引件数激減の根本原因調査計画（修正版）
**Investigation Plan for Low Trade Count Root Cause Analysis (Corrected Version)**

**作成日**: 2025-12-29
**最終更新**: 2025-12-29 14:00
**対象期間**: 2025-01-15 ~ 2025-01-31

---

## 🚨 **重要な修正履歴**

### **修正1: main_new.py結果の訂正**
- **誤った記載**: 「最終資本: 994,175円（初期100万円から-5,825円）」
- **正しい結果**: **最終資本: 1,044,792円（初期100万円から+44,792円）**
- **修正理由**: ターミナルログの中間状態を「最終結果」と誤認。実際の出力ファイル（CSV、TXT）を確認して修正。

### **修正2: DSSMS結果の更新**
- **誤った記載**: 「取引件数1件（VWAPBreakoutStrategy）」
- **正しい結果**: **取引件数0件**（最新: dssms_20251229_132817）
- **修正理由**: 最新のDSSMS出力ディレクトリ（dssms_20251229_132817）を確認。取引件数0件、銘柄切替4回。

---

## 📊 **これまでの経緯サマリー（修正版）**

### **Phase 1: Option D無効化（2025-12-26）**
- **実施内容**: `backtest_config.xlsm`の`POSITION_RESET_DATE`列を全て空欄化
- **狙い**: ポジション強制クローズを無効化して取引件数を回復
- **結果**: **取引件数0件**（効果なし）
- **結論**: ポジション強制クローズは原因ではない

### **Phase 2: Option 3実装（2025-12-27~28）**
- **実施内容**: 前日シグナル→当日始値エントリー方式を導入
  - backtest_start_date = target_date - 1日（2日間バックテスト）
  - warmup_days = 149日（1日調整）
  - シグナル生成当日にエントリー実行
- **結果**: **取引件数1件**（VWAPBreakoutStrategy、8604株、+0.03%）
- **発見**: 取引件数は1件に回復したが、50分の1に激減（通常50件→1件）
- **結論**: Option 3実装で取引件数は1件に回復したが、期待値（50件）からは大幅減

### **Phase 3: Option 3破棄→Option A復旧（2025-12-29）**
- **実施内容**: Option 3コードを完全削除してOption A（当日シグナル→翌日始値）に復旧
  - Line 171: `warmup_days = 150`
  - Line 1733-1737: `backtest_start_date = target_date`（単日バックテスト）
  - Line 2071: データ取得計算をtarget_dateベースに復旧
- **検証**: DSSMS統合システムで2025-01-15~31期間実行
- **結果**: **取引件数0件**（最新: dssms_20251229_132817）
  - 最終ポートフォリオ: 1,000,489円（+489円、+0.05%）
  - 銘柄切替: 4回（6954→8604→6954→8604）
  - execution_details: 空配列（戦略実行詳細不明）
- **重大な発見**: **Option 3破棄後、DSSMSでは取引件数0件に減少**
- **結論**: **Option 3の設計が原因ではなく、別の要因で取引件数が激減**

### **Phase 4: main_new.py検証（2025-12-29）** ✅ **修正済み**
- **実施内容**: main_new.py（マルチ戦略システム）で同一期間検証
  - 銘柄: 6954、期間: 2025-01-15~31、warmup_days: 149日
- **結果**: 
  - **VWAPBreakoutStrategy**: 4注文（BUY×2、SELL×2）→2取引実行
  - **GCStrategy**: 0注文
  - **取引1**: 2025-01-15~22、PnL=+50,617円（勝ち）
  - **取引2**: 2025-01-24~29、PnL=-5,825円（負け）
  - **純利益**: +44,792円
  - **総収益率**: +4.48%
  - **最終資本**: **1,044,792円**（初期100万円から+44,792円）
  - **勝率**: 50%
- **発見**: **main_new.pyでは取引件数2件（DSSMSは0件）**
- **矛盾**: 同じ期間でDSSMS（0件）とmain_new.py（2件）で取引件数が異なる

---

## 🔍 **判明したこと（証拠付き）**

### **1. Option 3の設計は根本原因ではない（Phase 3検証）**
- **証拠**: Option 3破棄後、DSSMSでは取引件数0件に減少（Phase 2の1件より悪化）
- **ファイル**: `src/dssms/dssms_integrated_main.py` Line 171, 1733-1737, 2071
- **検証結果**: 2025-01-15~31期間でDSSMS実行→取引件数0件
- **出力**: `output/dssms_integration/dssms_20251229_132817`

### **2. DSSMS統合システムとmain_new.pyで取引件数が大幅に異なる（Phase 3/4比較）**
- **DSSMS**: 取引件数**0件**（銘柄切替4回、各銘柄の保有期間短い）
- **main_new.py**: 取引件数**2件**（銘柄6954固定、17日間連続）
- **証拠**: 
  - DSSMS: `output/dssms_integration/dssms_20251229_132817/dssms_execution_results.json`
    - `execution_details: []`
    - `total_trades: 0`
  - main_new.py: `output/comprehensive_reports/6954.T_20251229_131112/6954.T_all_transactions.csv`
    - 2取引実行、純利益+44,792円
- **矛盾の詳細**:
  - DSSMSでは銘柄切替が4回発生したが、取引は0件
  - main_new.pyでは銘柄6954固定で2件の取引が発生
  - VWAPBreakoutStrategyはmain_new.pyで2件実行されたが、DSSMSでは0件

### **3. ウォームアップ期間の影響（Phase 2/3/4共通）**
- **DSSMS**: warmup_days=150日（Option A復旧後）
- **main_new.py**: warmup_days=149日（ウォームアップエラー回避）
- **発見**: 1日差で取引件数が変わる可能性あり（要検証）
- **証拠**: ウォームアップエラーログ（`Shortage: 1 days`）

### **4. GCStrategyのシグナル生成が0件（Phase 4: main_new.py）**
- **ログ証拠**:
  ```
  [WARMUP_SUMMARY] Backtest completed: strategy=GCStrategy, total_rows=111, 
  warmup_filtered=98, trading_rows=13, entry_signals=0, exit_signals=0
  ```
- **発見**: GCStrategyは13日間の取引期間でエントリー・エグジットシグナルが0件
- **原因候補**: トレンドフィルター、ゴールデンクロス条件、短期/長期MA期間

### **5. VWAPBreakoutStrategyのエントリーシグナルは生成されている（Phase 4: main_new.py）**
- **ログ証拠**:
  ```
  INFO:strategies.VWAP_Breakout:VWAP Breakout エントリーシグナル: 日付=2025-01-15 00:00:00, 価格=4277.08154296875
  INFO:strategies.VWAP_Breakout:VWAP Breakout エントリーシグナル: 日付=2025-01-24 00:00:00, 価格=4653.20703125
  ```
- **発見**: VWAPBreakoutStrategyは2件のエントリーシグナルを生成（2025-01-15、2025-01-24）
- **出来高フィルターによる却下**:
  ```
  DEBUG:strategies.VWAP_Breakout:[entry] idx=104: 出来高増加NG current=3423900, prev=4620100
  DEBUG:strategies.VWAP_Breakout:[entry] idx=109: 出来高増加NG current=3580300, prev=4802600
  ```
- **発見**: 2件のシグナルが出来高フィルターで却下された（idx=104、109）

### **6. DSSMS銘柄切替パターン（Phase 3: DSSMS）**
- **銘柄切替履歴**: 
  1. 2025-01-15: 初期→6954（12日間保有）
  2. 2025-01-27: 6954→8604（2日間保有）
  3. 2025-01-29: 8604→6954（1日間保有）
  4. 2025-01-30: 6954→8604（2日間保有）
- **発見**: 銘柄切替が4回発生
- **重要**: **元々この切替回数でも取引が発生していたため、銘柄切替頻度自体は根本原因ではない可能性が高い**
- **証拠**: `output/dssms_integration/dssms_20251229_132817/dssms_switch_history.csv`
- **調査結果**: 一旦調査対象から除外（過去に同じ切替パターンで取引実績あり）

---

## ❓ **不明な点**

### **1. DSSMSで取引件数が0件になった理由（Phase 3）**
- **症状**: 銘柄切替は4回発生したが、取引は0件
- **原因候補**:
  - **最有力**: 戦略実行自体が失敗した（execution_detailsが空）
  - GCStrategy・VWAPBreakoutStrategyのエントリー条件が満たされなかった
  - リスク管理で取引がブロックされた
  - ~~各銘柄の保有期間が短すぎる~~（除外: 過去に同じ切替パターンで取引実績あり）
  - ~~各銘柄切替後、ウォームアップ期間が不足~~（除外: 同上）

### **2. DSSMSのexecution_detailsが空配列の理由**
- **症状**: `execution_details: []`で戦略実行詳細が不明
- **原因候補**:
  - 戦略が実際に実行されなかった
  - 戦略実行は成功したが、記録が失敗した
  - システムバグ

### **3. main_new.pyでは取引が発生する理由（Phase 4）**
- **症状**: 銘柄6954固定、17日間連続で2件の取引が発生
- **原因候補**:
  - 17日間の連続保有により、シグナル生成条件を満たす
  - ウォームアップ期間（149日）が十分確保されている
  - VWAPBreakoutStrategyのエントリー条件（VWAP、出来高、RSI等）が満たされた

### **4. GCStrategyがシグナルを生成しない理由（Phase 4: main_new.py）**←もともとシグナルなかったので当然、基本調査不要
- **症状**: 13日間の取引期間でentry_signals=0、exit_signals=0
- **原因候補**:
  - ゴールデンクロス条件が厳しすぎる（SMA_5 > SMA_25の発生頻度）
  - トレンドフィルターが有効で全日NGになっている
  - ウォームアップ期間（149日）が短すぎてMA計算が不正確
  - エントリー条件の他の要素（例: 出来高、RSI等）

### **6. Option A復旧前（12月初旬）の取引件数が50件だった理由**
- **不明点**: 何が変わって50件→0件（or 1件）に激減したのか
- **原因候補**:
  - データ取得範囲の変更（過去データが増えた）
  - 戦略パラメータの変更（config更新）
  - ウォームアップ期間の変更（150日→149日→150日）
  - システムロジックの変更（DSSMS統合、main_new.py統合）
  - DSSMS銘柄切替の頻度変更（12月は切替が少なかった可能性）

---

## 📝 **調査項目チェックリスト（優先度順）**

### **【優先度S】DSSMSのexecution_details空配列の原因調査** ⭐ **最優先**
- [ ] **S-1**: DSSMSのexecution_details空配列の原因特定
  - 戦略実行が失敗したのか、記録が失敗したのか
  - エラーログ・ワーニングログの確認
  - 実行ログの詳細確認（2025-01-15~31期間）
- [ ] **S-2**: DSSMS vs main_new.pyの実行ログ比較
  - 各戦略の`backtest()`呼び出しログ
  - シグナル生成条件の判定ログ
  - ウォームアップ期間フィルターのログ
- [ ] **S-3**: DSSMSで戦略が実際に実行されたか確認
  - GCStrategy・VWAPBreakoutStrategyの実行ログ
  - シグナル生成の有無
  - リスク管理によるブロックの有無

### ~~**【優先度A】戦略ロジックの確認（VWAPBreakoutStrategy）**~~ → **除外**
**除外理由**: main_new.pyで2件の取引が正常に実行されているため、VWAPBreakoutStrategy自体は正常に動作している

### **【優先度A】戦略ロジックの確認（GCStrategy）**
- [ ] **A-1**: GCStrategyのエントリー条件を確認
  - ゴールデンクロス条件（SMA_5 > SMA_25）
  - トレンドフィルター有効/無効の確認
  - その他のエントリー条件（出来高、RSI等）
- [ ] **A-2**: GCStrategyのウォームアップ期間をデバッグ
  - MA計算に必要な最小データ数確認（長期MA=25日が計算可能か）
  - ウォームアップ期間フィルターのロジック確認
  - 取引期間開始日（2025-01-15）でのMA値を確認
- [ ] **A-3**: GCStrategyの`generate_entry_signal()`にデバッグログ追加
  - 各条件の判定結果（ゴールデンクロス、トレンド、出来高等）
  - シグナルが生成されない理由を特定



### **【優先度B】ウォームアップ期間の統一テスト**
- [ ] **B-1**: ウォームアップ期間の違い（150日 vs 149日）の影響調査
  - 両方を149日に統一して再実行
  - 両方を150日に統一して再実行（データ開始日調整）
  - 取引件数の変化を確認
- [ ] **B-2**: データ取得範囲の違いの影響調査
  - DSSMSの単日バックテスト（target_date=2025-01-15のみ）
  - main_new.pyの期間バックテスト（2025-01-15~31の17日間）
  - データフレーム内容の差分比較（行数、カラム、値）

### **【優先度C】12月初旬（取引件数50件時代）との比較**
- [ ] **C-1**: 12月初旬の実行ログ・設定ファイルを確認
  - `backtest_config.xlsm`の内容（パラメータ、POSITION_RESET_DATE）
  - 実行時のwarmup_days設定
  - 戦略パラメータの値
  - DSSMS銘柄切替の頻度・パターン
- [ ] **C-2**: 12月初旬からの変更履歴を確認
  - コード変更（git diff）
  - 設定ファイル変更
  - データ取得範囲の変更
- [ ] **C-3**: 12月初旬の環境を再現してバックテスト実行
  - 設定ファイルを12月初旬版に戻す
  - コードを12月初旬版に戻す（git checkout）
  - 取引件数50件が再現されるか確認

### **【優先度D】データフローの追跡**
- [ ] **D-1**: data_fetcher.pyのデータ取得範囲を確認
  - yfinanceから取得されるデータ範囲
  - CSVキャッシュの内容確認
  - `auto_adjust=False`の適用確認
- [ ] **D-2**: シグナル生成→エントリー実行のデータフローを追跡
  - backtest()→generate_entry_signal()の呼び出しチェーン
  - Entry_Signal列の値（1/0/-1）
  - StrategyExecutionManagerでのシグナル検出ログ

---

## 🛠️ **デバッグログ追加計画**

### **1. GCStrategy: generate_entry_signal()にデバッグログ追加**
```python
# strategies/GC_strategy.py Line 146付近
def generate_entry_signal(self, idx):
    logger.info(f"[GC_ENTRY_DEBUG] idx={idx}, date={result.index[idx]}")
    
    # ゴールデンクロス条件
    golden_cross = (result['SMA_Short'].iloc[idx] > result['SMA_Long'].iloc[idx]) and \
                   (result['SMA_Short'].iloc[idx-1] <= result['SMA_Long'].iloc[idx-1])
    logger.info(f"[GC_ENTRY_DEBUG] golden_cross={golden_cross}, SMA_Short={result['SMA_Short'].iloc[idx]}, SMA_Long={result['SMA_Long'].iloc[idx]}")
    
    # トレンドフィルター
    if self.trend_filter_enabled:
        current_trend = detect_unified_trend(...)
        logger.info(f"[GC_ENTRY_DEBUG] trend_filter_enabled=True, current_trend={current_trend}, allowed_trends={self.allowed_trends}")
        if current_trend not in self.allowed_trends:
            logger.info(f"[GC_ENTRY_DEBUG] Trend filter NG: current_trend={current_trend} not in {self.allowed_trends}")
            return False
    
    # 最終判定
    entry_signal = golden_cross
    logger.info(f"[GC_ENTRY_DEBUG] FINAL entry_signal={entry_signal}")
    return entry_signal
```

### ~~**2. VWAPBreakoutStrategy: 出来高フィルターにデバッグログ追加**~~ → **除外**
**除外理由**: main_new.pyで正常動作確認済み

### **2. DSSMS: 単日バックテストのデータ範囲ログ追加**
```python
# src/dssms/dssms_integrated_main.py Line 1733付近
backtest_start_date = target_date
backtest_end_date = target_date
warmup_days = self.warmup_days

logger.info(f"[DATA_RANGE_DEBUG] target_date={target_date}, backtest_start={backtest_start_date}, backtest_end={backtest_end_date}, warmup_days={warmup_days}")
logger.info(f"[DATA_RANGE_DEBUG] stock_data range: start={stock_data.index[0]}, end={stock_data.index[-1]}, rows={len(stock_data)}")
logger.info(f"[DATA_RANGE_DEBUG] stock_data columns: {stock_data.columns.tolist()}")
logger.info(f"[DATA_RANGE_DEBUG] selected_symbol={selected_symbol}, holding_days={holding_days}")
```

### **4. DSSMS: execution_details記録ログ追加**
```python
# src/dssms/dssms_integrated_main.py (execution_details記録箇所)
logger.info(f"[EXEC_DETAIL_RECORD] Recording execution_details: len={len(execution_details)}")
logger.debug(f"[EXEC_DETAIL_RECORD] execution_details content: {execution_details}")
```

### **5. main_new.py: ウォームアップ期間のログ追加**
```python
# main_new.py Line 519付近
ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    warmup_days=warmup_days,  # 149日
    required_columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
)

logger.info(f"[WARMUP_DEBUG_MAIN] warmup_days={warmup_days}, start_date={start_date}, end_date={end_date}")
logger.info(f"[WARMUP_DEBUG_MAIN] stock_data range: start={stock_data.index[0]}, end={stock_data.index[-1]}, rows={len(stock_data)}")
logger.info(f"[WARMUP_DEBUG_MAIN] trading_start_date={trading_start_date}, trading_end_date={trading_end_date}")
```

---

## 📈 **調査実行計画**

### **【Phase 1】優先度S項目の最優先調査（推定2-3時間）** ⭐
1. **S-1~S-3**: DSSMSのexecution_details空配列の原因調査
   - DSSMS実行ログの詳細確認（2025-01-15~31期間）
   - 戦略実行の成功/失敗ログ確認
   - execution_details空配列の原因特定
   - **仮説**: 戦略実行が失敗したか、記録処理が失敗した

### **【Phase 2】優先度A項目の集中調査（推定1-2時間）**
1. **A-1~A-3**: GCStrategyのシグナル生成ロジック確認
   - デバッグログ追加（generate_entry_signal）
   - 実行してログ確認
   - ゴールデンクロス条件・トレンドフィルターの問題を特定
   - ※VWAPBreakoutStrategyはmain_new.pyで正常動作確認済みのため除外

### **【Phase 3】優先度B項目の補足調査（推定1-2時間）**
1. **B-1~B-2**: ウォームアップ期間統一テスト
   - DSSMS・main_new.py両方を149日に統一
   - DSSMS・main_new.py両方を150日に統一（データ調整）
   - 取引件数の変化を確認

### **【Phase 4】優先度C・D項目の補足調査（推定1-2時間）**
1. **C-1~C-3**: 12月初旬（取引件数50件時代）との比較
   - 設定ファイル・コード履歴確認
   - 12月初旬環境の再現テスト
2. **D-1~D-2**: データフローの追跡
   - data_fetcher.pyのデータ取得範囲確認
   - シグナル生成→エントリー実行のフロー追跡

### **【Phase 5】調査結果まとめと修正案作成（推定1時間）**
1. 判明したこと（証拠付き）
2. 不明な点（調査継続項目）
3. 原因の推定（可能性順）
4. 修正案の提示（優先度順）

---

## 🎯 **期待される成果**

### **1. 根本原因の特定**
- **調査対象**: DSSMSのexecution_details空配列の原因（戦略実行が失敗したか、記録が失敗したか）
- DSSMSとmain_new.pyで取引件数が異なる理由
- GCStrategyがシグナルを生成しない理由（ただし、元々シグナル少ない）
- 12月初旬（50件）→現在（0-2件）に激減した理由
- ※VWAPBreakoutStrategyはmain_new.pyで正常動作確認済み（2件の取引あり）

### **2. 修正案の提示**
- **DSSMSの戦略実行処理の修正**（最優先）
  - execution_details記録処理のバグ修正
  - 戦略実行失敗時のエラーハンドリング改善
  - 実行ログの詳細化
- GCStrategyのパラメータ調整案（MA期間、トレンドフィルター）
- ウォームアップ期間の最適化案（150日→200日等）
- DSSMS・main_new.py統合方式の改善案
- ※VWAPBreakoutStrategyはmain_new.pyで正常動作のため修正不要

### **3. 取引件数回復の実現**
- 目標: 取引件数50件に回復（12月初旬レベル）
- 検証: 2025-01-15~31期間で50件前後の取引が発生することを確認
- 安定性: 銘柄・期間を変えても同様の取引件数が維持されることを確認

---

## 📌 **次のアクション**

### **【最優先】Phase 1開始: DSSMSのexecution_details空配列調査**
1. **DSSMS実行ログの詳細確認**
   - 2025-01-15~31期間のログ全体を確認
   - 戦略実行の成功/失敗ログ
   - エラー・ワーニングログの確認
2. **execution_details空配列の原因特定**
   - 戦略実行が失敗したのか、記録が失敗したのか
   - 戦略インスタンス生成の成否
   - backtest()呼び出しの成否
3. **DSSMS vs main_new.pyの比較**
   - 同じ戦略（VWAPBreakoutStrategy）がなぜmain_new.pyでは動いてDSSMSでは動かないのか
   - コード実行パスの違い
   - パラメータの違い

### **【次】Phase 2開始: 戦略ロジック調査**
1. GCStrategy・VWAPBreakoutStrategyにデバッグログ追加
2. 実行して分析: DSSMS・main_new.py両方で実行してログ確認
3. 原因特定: 取引件数が少ない真の原因を特定
4. 修正案作成: パラメータ調整・ロジック改善の具体案を提示
5. 検証実行: 修正後のバックテスト実行→取引件数50件回復を確認

---

## 📊 **修正前後の比較表**

| 項目 | 修正前（誤り） | 修正後（正しい） |
|------|---------------|-----------------|
| **main_new.py最終資本** | 994,175円（-5,825円） | **1,044,792円（+44,792円）** |
| **main_new.py総リターン** | -0.58% | **+4.48%** |
| **main_new.py勝率** | 記載なし | **50%** |
| **DSSMS取引件数** | 1件（Phase 2結果を誤記） | **0件（Phase 3最新結果）** |
| **DSSMS銘柄切替** | 記載なし | **4回（頻繁な切替）** |
| **DSSMS execution_details** | 記載なし | **空配列（要調査）** |

---

**注意事項**:
- デバッグログは`DEBUG_BACKTEST=1`環境変数で有効化すること
- フォールバックを発見した場合は即座に報告すること
- 証拠なき推測は避け、必ずログ・データで裏付けること
- セルフチェックを実施して見落とし・思い込み・矛盾を排除すること
- **最優先**: DSSMS銘柄切替とシグナル生成の関係を調査すること

---

**調査担当**: Backtest Project Team
**レビュー**: GitHub Copilot (Claude Sonnet 4.5)
**最終更新**: 2025-12-29 14:00

# 取引件数激減の根本原因調査計画
**Investigation Plan for Low Trade Count Root Cause Analysis**

**作成日**: 2025-12-29
**対象期間**: 2025-01-15 ~ 2025-01-31

---

## 📊 **これまでの経緯サマリー**

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
- **結果**: **取引件数1件**（VWAPBreakoutStrategy、8604株、+0.03%）
- **重大な発見**: **Option 3破棄してもOption Aでも取引件数1件（同じ）**
- **結論**: **Option 3の設計（日次2日間バックテスト）は取引件数激減の原因ではない**

### **Phase 4: main_new.py検証（2025-12-29）**
- **実施内容**: main_new.py（マルチ戦略システム）で同一期間検証
  - 銘柄: 6954、期間: 2025-01-15~31、warmup_days: 149日
- **結果**: 
  - **VWAPBreakoutStrategy**: 4注文（BUY×2、SELL×2）→2取引実行
  - **GCStrategy**: 0注文
  - **総収益率**: -0.58%
  - **最終資本**: 994,175円（初期100万円から-5,825円）
- **発見**: **main_new.pyでは取引件数2件（DSSMSは1件）**
- **矛盾**: 同じ銘柄・期間でDSSMS（1件）とmain_new.py（2件）で取引件数が異なる

---

## 🔍 **判明したこと（証拠付き）**

### **1. Option 3の設計は原因ではない（Phase 3検証）**
- **証拠**: Option 3破棄後もOption Aでも取引件数1件（同じ）
- **ファイル**: `src/dssms/dssms_integrated_main.py` Line 171, 1733-1737, 2071
- **検証結果**: 2025-01-15~31期間でDSSMS実行→取引件数1件

### **2. DSSMS統合システムとmain_new.pyで取引件数が異なる（Phase 4検証）**
- **DSSMS**: 取引件数1件（VWAPBreakoutStrategy、8604株）
- **main_new.py**: 取引件数2件（VWAPBreakoutStrategy、BUY×2+SELL×2）
- **証拠**: 
  - DSSMS実行ログ: `python -m src.dssms.dssms_integrated_main --start-date 2025-01-15 --end-date 2025-01-31`
  - main_new.py実行ログ: `python main_new.py`（銘柄6954、期間2025-01-15~31）
- **矛盾の詳細**:
  - DSSMSではGCStrategyが実行されたが、取引件数0件
  - main_new.pyではGCStrategyが選択されたが、シグナル生成0件
  - VWAPBreakoutStrategyは両方で実行されたが、取引件数が異なる（DSSMS:1件、main_new:2件）

### **3. ウォームアップ期間の影響（Phase 2/3/4共通）**
- **DSSMS**: warmup_days=150日（Option A復旧後）
- **main_new.py**: warmup_days=149日（ウォームアップエラー回避）
- **発見**: 1日差で取引件数が変わる可能性あり
- **証拠**: ウォームアップエラーログ（`Shortage: 1 days`）

### **4. GCStrategyのシグナル生成が0件（Phase 4検証）**
- **ログ証拠**:
  ```
  [WARMUP_SUMMARY] Backtest completed: strategy=GCStrategy, total_rows=111, 
  warmup_filtered=98, trading_rows=13, entry_signals=0, exit_signals=0
  ```
- **発見**: GCStrategyは13日間の取引期間でエントリー・エグジットシグナルが0件
- **原因候補**: トレンドフィルター、ゴールデンクロス条件、短期/長期MA期間

### **5. VWAPBreakoutStrategyのエントリーシグナルは生成されている（Phase 4検証）**
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

---

## ❓ **不明な点**

### **1. DSSMSとmain_new.pyで取引件数が異なる理由**
- **矛盾**: 同じ銘柄・期間でDSSMS（1件）とmain_new.py（2件）
- **原因候補**:
  - ウォームアップ期間の違い（150日 vs 149日）
  - データ取得範囲の違い（single day vs period backtest）
  - 戦略パラメータの違い（config由来 vs ハードコード）
  - シグナル生成ロジックの違い（daily切替 vs multi-strategy）

### **2. GCStrategyがシグナルを生成しない理由**
- **症状**: 13日間の取引期間でentry_signals=0、exit_signals=0
- **原因候補**:
  - ゴールデンクロス条件が厳しすぎる（SMA_5 > SMA_25の発生頻度）
  - トレンドフィルターが有効で全日NGになっている
  - ウォームアップ期間（149日）が短すぎてMA計算が不正確
  - エントリー条件の他の要素（例: 出来高、RSI等）

### **3. VWAPBreakoutStrategyの出来高フィルターが厳しすぎないか**
- **症状**: 2件のシグナルが出来高フィルターで却下
- **原因候補**:
  - `volume_threshold = 1.2`が厳しすぎる（前日比120%以上が必要）
  - 銘柄6954の出来高特性（変動が小さい）
  - 計算方法の問題（current/prev比較のタイミング）

### **4. Option A復旧前（12月初旬）の取引件数が50件だった理由**
- **不明点**: 何が変わって50件→1件に激減したのか
- **原因候補**:
  - データ取得範囲の変更（過去データが増えた）
  - 戦略パラメータの変更（config更新）
  - ウォームアップ期間の変更（150日→149日→150日）
  - システムロジックの変更（DSSMS統合、main_new.py統合）

---

## 📝 **調査項目チェックリスト（優先度順）**

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

### **【優先度A】戦略ロジックの確認（VWAPBreakoutStrategy）**
- [ ] **A-4**: VWAPBreakoutStrategyの出来高フィルター閾値を確認
  - `volume_threshold = 1.2`の妥当性検証
  - 銘柄6954の出来高特性分析（変動率、平均出来高）
  - 閾値緩和テスト（1.2→1.1→1.0）
- [ ] **A-5**: 出来高フィルターで却下された2件のシグナル詳細分析
  - idx=104、109の日付・価格・出来高を確認
  - 却下理由の妥当性検証
  - フィルター無効化時の取引件数を確認

### **【優先度B】DSSMSとmain_new.pyの差異調査**
- [ ] **B-1**: ウォームアップ期間の違い（150日 vs 149日）の影響調査
  - 両方を149日に統一して再実行
  - 両方を150日に統一して再実行（データ開始日調整）
  - 取引件数の変化を確認
- [ ] **B-2**: データ取得範囲の違いの影響調査
  - DSSMSの単日バックテスト（target_date=2025-01-15のみ）
  - main_new.pyの期間バックテスト（2025-01-15~31の17日間）
  - データフレーム内容の差分比較（行数、カラム、値）
- [ ] **B-3**: 戦略パラメータの違いの影響調査
  - DSSMSとmain_new.pyで使用されている戦略パラメータを比較
  - `backtest_config.xlsm`由来 vs ハードコードの違い
  - パラメータ統一後の再実行
- [ ] **B-4**: 戦略実行タイミングの違いの影響調査
  - DSSMSの日次切替ロジック（target_dateループ）
  - main_new.pyのマルチ戦略統合実行
  - 実行順序・タイミングの差異確認

### **【優先度C】12月初旬（取引件数50件時代）との比較**
- [ ] **C-1**: 12月初旬の実行ログ・設定ファイルを確認
  - `backtest_config.xlsm`の内容（パラメータ、POSITION_RESET_DATE）
  - 実行時のwarmup_days設定
  - 戦略パラメータの値
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

### **2. VWAPBreakoutStrategy: 出来高フィルターにデバッグログ追加**
```python
# strategies/VWAP_Breakout.py Line 280付近
def generate_entry_signal(self, idx):
    # ... (既存の条件チェック)
    
    # 出来高フィルター
    volume_condition = self.data['Volume'].iloc[idx] > (volume_ma * self.volume_threshold)
    logger.debug(f"[VWAP_VOLUME_DEBUG] idx={idx}, current_volume={self.data['Volume'].iloc[idx]}, volume_ma={volume_ma}, threshold={self.volume_threshold}, condition={volume_condition}")
    
    if not volume_condition:
        logger.info(f"[VWAP_VOLUME_NG] Volume filter rejected: idx={idx}, current={self.data['Volume'].iloc[idx]}, prev={self.data['Volume'].iloc[idx-1]}, ratio={self.data['Volume'].iloc[idx]/self.data['Volume'].iloc[idx-1]:.2f}, threshold={self.volume_threshold}")
        return False
    
    return True
```

### **3. DSSMS: 単日バックテストのデータ範囲ログ追加**
```python
# src/dssms/dssms_integrated_main.py Line 1733付近
backtest_start_date = target_date
backtest_end_date = target_date
warmup_days = self.warmup_days

logger.info(f"[DATA_RANGE_DEBUG] target_date={target_date}, backtest_start={backtest_start_date}, backtest_end={backtest_end_date}, warmup_days={warmup_days}")
logger.info(f"[DATA_RANGE_DEBUG] stock_data range: start={stock_data.index[0]}, end={stock_data.index[-1]}, rows={len(stock_data)}")
logger.info(f"[DATA_RANGE_DEBUG] stock_data columns: {stock_data.columns.tolist()}")
```

### **4. main_new.py: ウォームアップ期間のログ追加**
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

### **【Phase 1】優先度A項目の集中調査（推定2-3時間）**
1. **A-1~A-3**: GCStrategyのシグナル生成ロジック確認
   - デバッグログ追加（generate_entry_signal）
   - 実行してログ確認
   - ゴールデンクロス条件・トレンドフィルターの問題を特定
2. **A-4~A-5**: VWAPBreakoutStrategyの出来高フィルター確認
   - デバッグログ追加（volume filter）
   - 閾値緩和テスト（1.2→1.1→1.0）
   - 却下された2件のシグナル詳細分析

### **【Phase 2】優先度B項目の差異調査（推定2-3時間）**
1. **B-1**: ウォームアップ期間統一テスト
   - DSSMS・main_new.py両方を149日に統一
   - DSSMS・main_new.py両方を150日に統一（データ調整）
   - 取引件数の変化を確認
2. **B-2~B-4**: DSSMSとmain_new.pyの差異調査
   - データ取得範囲の比較
   - 戦略パラメータの比較
   - 実行タイミングの比較

### **【Phase 3】優先度C・D項目の補足調査（推定1-2時間）**
1. **C-1~C-3**: 12月初旬（取引件数50件時代）との比較
   - 設定ファイル・コード履歴確認
   - 12月初旬環境の再現テスト
2. **D-1~D-2**: データフローの追跡
   - data_fetcher.pyのデータ取得範囲確認
   - シグナル生成→エントリー実行のフロー追跡

### **【Phase 4】調査結果まとめと修正案作成（推定1時間）**
1. 判明したこと（証拠付き）
2. 不明な点（調査継続項目）
3. 原因の推定（可能性順）
4. 修正案の提示（優先度順）

---

## 🎯 **期待される成果**

### **1. 根本原因の特定**
- GCStrategyがシグナルを生成しない理由
- VWAPBreakoutStrategyの出来高フィルターが厳しすぎる理由
- DSSMSとmain_new.pyで取引件数が異なる理由
- 12月初旬（50件）→現在（1-2件）に激減した理由

### **2. 修正案の提示**
- GCStrategyのパラメータ調整案（MA期間、トレンドフィルター）
- VWAPBreakoutStrategyの出来高閾値調整案（1.2→1.1等）
- ウォームアップ期間の最適化案（150日→200日等）
- DSSMS・main_new.py統合方式の改善案

### **3. 取引件数回復の実現**
- 目標: 取引件数50件に回復（12月初旬レベル）
- 検証: 2025-01-15~31期間で50件前後の取引が発生することを確認
- 安定性: 銘柄・期間を変えても同様の取引件数が維持されることを確認

---

## 📌 **次のアクション**

1. **Phase 1開始**: GCStrategy・VWAPBreakoutStrategyにデバッグログ追加
2. **実行して分析**: DSSMS・main_new.py両方で実行してログ確認
3. **原因特定**: 取引件数が少ない真の原因を特定
4. **修正案作成**: パラメータ調整・ロジック改善の具体案を提示
5. **検証実行**: 修正後のバックテスト実行→取引件数50件回復を確認

---

**注意事項**:
- デバッグログは`DEBUG_BACKTEST=1`環境変数で有効化すること
- フォールバックを発見した場合は即座に報告すること
- 証拠なき推測は避け、必ずログ・データで裏付けること
- セルフチェックを実施して見落とし・思い込み・矛盾を排除すること

---

**調査担当**: Backtest Project Team
**レビュー**: GitHub Copilot (Claude Sonnet 4.5)
**最終更新**: 2025-12-29

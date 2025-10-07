# バックテストシステム動作フロー分析報告書

**調査日時**: 2025年9月25日  
**調査対象**: main.py（マルチ戦略システム）vs src/dssms/dssms_backtester.py（DSSMS）

## 1. 重要な発見：2つの独立### **DSSMS（動的銘柄選択管理システム）**
- **複数銘柄**から**最適な1銘柄**を動的選択
- 集中投資的アプローチ
- パーフェクトオーダーによる階層的ランキング
- プログラム内での銘柄リスト管理
- **独自の取引シミュレーション機能内蔵**
- **マルチ戦略システムとは完全に独立**した取引実行ム

### **main.py（マルチ戦略システム）**
- **実行方法**: `python main.py`
- **目的**: 複数戦略の統合バックテスト（分散投資）
- **対象銘柄**: Excel設定ファイルから**単一銘柄**を取得

### **DSSMS（動的銘柄選択管理システム）**
- **実行方法**: `python "src\dssms\dssms_backtester.py"`
- **目的**: 動的銘柄選択による集中運用（単一銘柄への集中投資）
- **対象銘柄**: プログラム内で**複数銘柄リスト**を定義

### **連携状況**: **現在は完全に独立**
- main.pyのコメントに「将来的にDSSMS結果を受け取る予定」とあるが、現在は**実装されていない**
- 両システムは異なるデータ取得方法、異なる目的で動作している

---

## 2. データ取得・処理フローの詳細比較

### **A. main.py（マルチ戦略システム）のフロー**

#### **1. データを取得する**
**関連ファイル・パス：**
- `data_fetcher.py` (ルート) - `get_parameters_and_data()`関数
- `config/backtest_config.xlsx` - Excel設定ファイル
- `config/backtest_config.xlsm` - Excel設定ファイル（マクロ有効版）

**フロー詳細：**
```python
# main.py line 468
ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
```
- Excel設定ファイル（`backtest_config.xlsx`）から**単一銘柄・期間**を読み込み
- Yahoo Finance APIで該当銘柄のデータを取得
- インデックスデータも同時取得

#### **2. データを処理する**
**関連ファイル・パス：**
- `data_processor.py` (ルート) - `preprocess_data()`
- `indicators/indicator_calculator.py` - `compute_indicators()`

**フロー詳細：**
```python
# main.py line 469-470
stock_data = preprocess_data(stock_data)
stock_data = compute_indicators(stock_data)
```

#### **3-7. 戦略実行・統合・結果出力**
**関連ファイル・パス：**
- `strategies/` 配下の各戦略クラス
- `config/multi_strategy_manager.py` - 統合管理
- `output/simple_simulation_handler.py` - 結果出力

---

### **B. DSSMS（動的銘柄選択システム）のフロー**

#### **1. データを取得する**
**関連ファイル・パス：**
- `src/dssms/dssms_data_manager.py` - DSSMS専用データ管理
- `config/dssms/nikkei225_components.json` - 日経225銘柄リスト
- `data_fetcher.py` (フォールバック用)

**フロー詳細：**
```python
# src/dssms/dssms_backtester.py line 4070
symbol_universe = ['7203', '9984', '6758', '4063', '8306', '6861', '7741', '9432', '8058', '9020']
```
- プログラム内で**ハードコーディングされた複数銘柄リスト**を使用
- 日経225構成銘柄（225銘柄）もJSON設定ファイルで管理
- **Excel設定ファイルは使用しない**
- `DSSMSDataManager`でマルチタイムフレーム（日足・週足・月足）データを並列取得

#### **2. データを処理する**
**関連ファイル・パス：**
- `src/dssms/data_quality_enhancer.py` - データ品質向上
- `src/dssms/data_cleaning_engine.py` - データクリーニング
- `config/dssms/data_quality_config.json` - データ品質設定

#### **3. トレンド判断をする**
**関連ファイル・パス：**
- `src/dssms/perfect_order_detector.py` - パーフェクトオーダー検出
- `src/dssms/market_condition_monitor.py` - 市場状況監視

#### **4. ランキングする**
**関連ファイル・パス：**
- `src/dssms/hierarchical_ranking_system.py` - 階層的ランキング
- `src/dssms/comprehensive_scoring_engine.py` - 総合スコアリング
- `config/dssms/ranking_config.json` - ランキング設定

#### **5. 上位ランキングの銘柄をさらに選ぶ**
**関連ファイル・パス：**
- `src/dssms/intelligent_switch_manager.py` - インテリジェント切替管理
- `config/dssms/hierarchical_switch_decision_config.json` - 切替判定設定

#### **6. 銘柄切替とポジション管理（DSSMS内部で完結）**
**関連ファイル・パス：**
- `src/dssms/dssms_backtester.py` - `_execute_switch()`メソッド
- `src/dssms/dssms_backtester.py` - `_update_portfolio_value()`メソッド

**重要な発見：**
- **DSSMSは独自の取引シミュレーション機能を持っている**
- ランキング1位の銘柄に切り替える際、DSSMS内部で以下を実行：
  - 前銘柄のポジション解除（損益計算込み）
  - 新銘柄へのポジション設定
  - 切替コスト（手数料）の控除
  - ポートフォリオ価値の更新
- **マルチ戦略システムとは完全に独立**して動作
- Entry_Signal/Exit_Signalを生成してExcel出力に対応

**独自計算部分の詳細：**

1. **損益計算方式**：
   - **決定論的モード**: ハッシュ関数で-5%〜+10%の範囲で一意決定
   - **ランダムモード**: -3%〜+5%の範囲でランダム生成

2. **戦略統計の生成方式**：
   - **戦略名はローテーション方式で割り当て**（実際の戦略ロジックは未実装）
   - 7つの戦略名を順番に循環：VWAPBreakout → MeanReversion → TrendFollowing → Momentum → Contrarian → VolatilityBreakout → RSI
   - **各戦略の個別ロジックは存在しない**（戦略名のみ）

3. **実際の取引シミュレーション**：
   - 銘柄切替時に仮想的な売買価格を生成
   - 切替コスト（デフォルト0.1%）を自動控除  
   - ポートフォリオ価値をリアルタイム更新

**結論**: DSSMSの「戦略」表示は**見せかけの分類**であり、実際はDSSMS独自の銘柄選択アルゴリズムによる単一戦略システム

#### **7. 繰り返し**
**関連ファイル・パス：**
- `src/dssms/dssms_backtester.py` - `simulate_dynamic_selection()`メソッド
- `config/dssms/scheduler_config.json` - スケジューラー設定

#### **8. 結果を出力する**
**関連ファイル・パス：**
- `output/dssms_unified_output_engine.py` - DSSMS統一出力エンジン
- `output/dssms_excel_exporter_v2.py` - Excel出力専用
- `backtest_results/dssms_results/` - 結果保存先

---

## 3. システム設計思想の違い

### **main.py（マルチ戦略システム）**
- **単一銘柄**に対して**複数戦略**を適用
- 戦略間での信号統合・優先度管理
- 分散投資的アプローチ
- Excel設定ファイルによる柔軟な銘柄・期間変更

### **DSSMS（動的銘柄選択システム）**
- **複数銘柄**から**最適な1銘柄**を動的選択
- 最適な1銘柄をマルチ戦略システムに渡す
- パーフェクトオーダーによる階層的ランキング
- 毎日ランキング更新

---

## 4. 実行エントリーポイント

### **main.py実行時**
```bash
python main.py
```
- `main()`関数が実行される
- Excel設定ファイルから単一銘柄を読み込み
- マルチ戦略バックテスト実行

### **DSSMS実行時**
```bash
python "src\dssms\dssms_backtester.py"
```
- ファイル末尾の`if __name__ == "__main__":`ブロックが実行される
- ハードコーディングされた銘柄リストを使用
- 動的銘柄選択シミュレーション実行

---

## 5. 今後の統合予定

main.pyのコメントから読み取れる将来的な統合計画：
```python
# main.py line 21-23
# - 銘柄選択 (DSSMS) は src/dssms/ 以下で独立進化し、将来ここへは
#   「選択結果(最適銘柄+バックアップ)を受け取る」一方向インターフェースのみ維持。
# - DSSMS目的: 日次動的最適銘柄集中運用 (分散なし)。
```

**統合計画**：
1. DSSMSが最適銘柄を選択
2. その結果をmain.pyが受け取り
3. 選択された銘柄に対してマルチ戦略を適用

**現状**：**まだ実装されていない**

---

## 6. 結論

**調査結果による重要な発見**

### **あなたの当初の認識は部分的に正しく、部分的に誤解がありました**

#### **正しかった部分：**
- DSSMSは複数銘柄をランキングして最適銘柄を選択
- ランキング変更時にポジション解除と新銘柄でのポジション開始を行う
- この処理を繰り返す

#### **誤解されていた部分：**
- **マルチ戦略システムとの連携**: DSSMSは**独立して取引シミュレーションを実行**
- **マルチ戦略システムには銘柄を渡していない**（現在）

### **実際の動作：**

1. **main.py（マルチ戦略システム）**: 
   - Excel設定ファイルから単一銘柄取得
   - その銘柄に対して複数戦略を適用

2. **DSSMS（動的銘柄選択システム）**: 
   - プログラム内の複数銘柄リストから最適銘柄を選択
   - **DSSMS内部で独自に取引シミュレーション実行**
   - Entry/Exitシグナル生成、損益計算、ポートフォリオ価値更新を内包
   - マルチ戦略システムとは**完全に独立**

### **システム設計の現実：**
**DSSMSは「銘柄選択 + 取引実行」が一体化したオールインワンシステム**であり、あなたが想像されていた「DSSMSが銘柄を選択してマルチ戦略システムに渡す」という連携は現在実装されていません。

**現在は完全に独立した2つのシステム**として動作しており、将来的な統合は計画されているものの、現時点では実装されていません。

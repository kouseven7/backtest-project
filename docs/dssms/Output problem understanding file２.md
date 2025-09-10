# DSSMS出力問題 構造分析・特定ロードマップ

## 🔍 問題点構造分析

### 問題の依存関係マップ
```
問題1：銘柄切替数激減 → 問題2,3,4,5に波及影響
    ↓
問題2：日付ループ ← データ処理エンジンの問題
    ↓
問題3,4：保有期間固定 ← 計算ロジックの問題
    ↓
問題5：統計項目未計算 ← データフローの問題
```

## 📋 Task 1: 銘柄切替数激減問題の根本原因特定

### 🎯 目的
DSSMSで銘柄切替回数が117回→1-2回に激減する根本原因を特定

### 🔧 調査手順

#### Task 1.1: 切替数カウント機構の検証
**実行内容:**
```python
# 診断スクリプト作成: diagnose_switch_count.py
def analyze_switch_counting():
    """切替カウント機構の詳細分析"""
    
    # 1. DSSMSBacktester.switch_historyの状態確認
    # 2. 各統一エンジンでの切替数取得方法比較
    # 3. backtest_results削除前後の動作比較
    # 4. データ永続化機構の調査
    
    engines_to_test = [
        'dssms_unified_output_engine.py',
        'dssms_unified_output_engine_fixed.py', 
        'dssms_unified_output_engine_fixed_v3.py',
        'dssms_unified_output_engine_fixed_v4.py'
    ]
```

#### Task 1.2: データ永続化・キャッシュ問題調査
**推測される原因:**
- `backtest_results/dssms_results/`ディレクトリ内の隠れファイル
- DSSMSシステムの状態保存機構
- インメモリキャッシュの問題

**調査コマンド:**
```bash
# 隠れファイル・ディレクトリ確認
dir /a:h backtest_results\dssms_results\
Get-ChildItem -Hidden -Recurse backtest_results\dssms_results\

# DSSMS設定ファイル調査
find . -name "*dssms*" -type f | grep -E "\.(json|config|cache)$"
```

#### Task 1.3: 統一エンジン影響度分析
**実行内容:**
- 各エンジンファイルの`switch_history`処理ロジック比較
- バックテスター→エンジン間のデータフロー追跡
- 切替判定ロジックの変更点特定

---

## 📋 Task 2: 日付ループ問題の特定

### 🎯 目的  
「2023-12-31 → 2023-01-01」の不正ループ原因を特定

### 🔧 調査手順

#### Task 2.1: 日付処理ロジックの検証
**調査対象:**
```python
# 各エンジンの日付修正ロジック比較
files_to_analyze = [
    'dssms_unified_output_engine_fixed.py:_fix_date_inconsistencies_improved',
    'dssms_unified_output_engine_fixed_v3.py:日付修正部分',
    'dssms_unified_output_engine_fixed_v4.py:日付修正部分'
]

# 特に以下のロジックを調査:
# 1. expected_year = 2023 の無限ループ可能性
# 2. 年末→年始の日付境界処理
# 3. pd.to_datetime()の変換ロジック
```

#### Task 2.2: ポートフォリオ価値データフローの追跡
**実行内容:**
```python
def trace_portfolio_data_flow():
    """ポートフォリオデータの変換過程を追跡"""
    
    # 1. DSSMSBacktester.portfolio_values の生データ確認
    # 2. _convert_backtester_results での変換過程
    # 3. _fix_date_inconsistencies での修正過程
    # 4. Excel出力での最終データ確認
```

---

## 📋 Task 3: 保有期間固定問題の特定

### 🎯 目的
保有期間が24時間固定される計算ロジックの問題を特定

### 🔧 調査手順

#### Task 3.1: 保有期間計算ロジックの比較分析
**調査対象:**
```python
# 各エンジンでの保有期間計算方法比較
calculation_methods = {
    'v1': 'TODO: 実際の計算 → 24.0時間固定',
    'v3': '次のスイッチ日時 - 現在スイッチ日時',
    'v4': 'actual_holding_hours計算ロジック'
}

# 特に調査すべき箇所:
# 1. SymbolSwitchオブジェクトのtimestamp取得
# 2. 日付差分計算の実装
# 3. 最後のスイッチでのデフォルト値処理
```

#### Task 3.2: 修正版エンジン統合後の回帰問題分析
**推測される原因:**
- 修正版エンジンが正しく統合されていない
- 古いエンジンが呼ばれている
- バックテスター側の`get_performance_metrics`等の未実装

---

## 📋 Task 4: 戦略別統計未計算問題の特定

### 🎯 目的
戦略別統計シートの計算ロジック問題を特定

### 🔧 調査手順

#### Task 4.1: 統計計算データソースの検証
**調査内容:**
```python
def analyze_strategy_statistics_data_source():
    """戦略統計計算のデータソース調査"""
    
    # 1. DSSMSBacktester.get_strategy_statistics()の実装状況
    # 2. trade_historyデータの構造と内容
    # 3. switch_historyからの統計計算可能性
    # 4. 各統一エンジンでの統計計算実装比較
```

#### Task 4.2: 計算ロジック実装状況の確認
**調査対象:**
- 勝率計算: `profitable_trades / total_trades`
- 平均利益/損失: `profit_trades.mean()` / `loss_trades.mean()`
- プロフィットファクター: `total_profit / abs(total_loss)`
- 各エンジンでの実装レベル比較

---

## 🚀 実行計画

### Phase 1: 根本原因特定 (優先度: 最高)
1. **Task 1実行**: 銘柄切替数問題の特定 (30分)
2. **Task 2実行**: 日付ループ問題の特定 (20分)

### Phase 2: 計算ロジック問題特定 (優先度: 高)  
3. **Task 3実行**: 保有期間固定問題の特定 (20分)
4. **Task 4実行**: 統計未計算問題の特定 (15分)

### Phase 3: 統合解決策策定 (優先度: 中)
5. **問題間依存関係の整理** (10分)
6. **修復優先順位の決定** (5分)

---

## 📞 必要なファイル・情報

### 即座に必要なファイル
1. **最新のDSSMSBacktesterクラス**: `src/dssms/dssms_backtester.py`
2. **各統一エンジンファイル**: 
   - `dssms_unified_output_engine.py`
   - `dssms_unified_output_engine_fixed.py`
   - `dssms_unified_output_engine_fixed_v3.py` 
   - `dssms_unified_output_engine_fixed_v4.py`
3. **問題のあるExcelファイル**: `backtest_results/dssms_results/dssms_unified_backtest_20250910_213413.xlsx`
4. **最新の実行ログファイル**: コンソール出力またはログファイル

### 調査過程で要求する可能性があるファイル
1. **設定・キャッシュファイル**: `src/dssms/*.json`, `src/dssms/*.config`
2. **SymbolSwitchクラス定義**: `src/dssms/`配下の関連ファイル
3. **バックテスト結果ディレクトリ全体**: `backtest_results/dssms_results/`
4. **Git履歴**: 問題発生前後のコミット差分

---

## 💡 質問事項

### 🔍 現象についての追加情報
1. **銘柄切替数激減のタイミング**: 
   - 最後に117回だったのはいつ頃ですか？
   - 何らかのファイル変更やシステム操作の直後でしたか？

2. **backtest_resultsディレクトリ削除の詳細**:
   - 削除したファイルの種類（.xlsx, .txt, .json以外にありましたか？）
   - サブディレクトリも含めて完全に空にしましたか？

3. **問題の再現性**:
   - 毎回同じ切替数になりますか？（例：必ず2回）
   - 実行するたびに変わりますか？

4. **使用している統一エンジン**:
   - 現在、どのエンジンファイルが実際に使用されていますか？
   - `src/dssms/dssms_backtester.py`内でどのエンジンをインポートしていますか？

### 🛠️ 技術的確認事項
1. **Pythonキャッシュ**: `__pycache__`ディレクトリを削除してテストしましたか？
2. **インポート確認**: `import`文でどのエンジンクラスを使っているか確認できますか？
3. **エラーログ**: 実行時に警告やエラーメッセージは出ていますか？

---

**次のアクション**: まず「Task 1.1: 切替数カウント機構の検証」から開始し、診断スクリプトを作成して根本原因を特定します。
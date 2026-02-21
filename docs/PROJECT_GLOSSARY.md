# プロジェクト用語集（PROJECT_GLOSSARY）

**目的**: プロジェクトで使用される重要な用語を定義する  
**最終更新**: 2026-02-10  
**対象**: 開発者、レビュアー、新規参加者

---

## 使用方法

### 用語検索
- Ctrl+F（Windows）またはCmd+F（Mac）で用語を検索

### 用語追加
1. カテゴリを選択（システムアーキテクチャ、データ構造、戦略関連等）
2. アルファベット順または重要度順に追加
3. 以下の情報を含める：
   - 用語名（英語または日本語）
   - 定義
   - 使用例（該当する場合）
   - 関連用語

---

## A. システムアーキテクチャ

### DSSMS（Dynamic Stock Selection Multi-Strategy System）
**定義**: 日経225構成銘柄から最適銘柄を動的に選択し、マルチ戦略を適用するシステム  
**主要機能**: Screening → Ranking → Scoring → Symbol Switching  
**目的**: 動的銘柄選択とマルチ戦略実行による利益最大化  
**関連**: [DSSMS_ARCHITECTURE.md](docs/DSSMS_ARCHITECTURE.md)

### kabu STATION API
**定義**: auカブコム証券が提供する株式取引API  
**用途**: リアルタイム取引の発注・約定確認  
**統合状態**: 認証/骨格8割実装済み、発注安全化ロジック未完成  
**関連**: [reference_docs/kabusapi/](reference_docs/kabusapi/)

### BaseStrategy
**定義**: 全ての取引戦略の基底クラス  
**場所**: strategies/base_strategy.py  
**主要メソッド**: generate_entry_signal(), generate_exit_signal(), backtest()  
**派生クラス**: BreakoutStrategy, MomentumStrategy, ContrarianStrategy等

---

## B. データ構造

### self.positions（ポジション管理辞書）
**定義**: 現在保有している全ポジションを管理する辞書型データ構造  
**型**: `dict[str, dict]`（キー: 銘柄コード、値: ポジション情報辞書）  
**構造**:
```python
{
    '9101.T': {
        'symbol': '9101.T',           # 銘柄コード
        'strategy': 'BreakoutStrategy', # 戦略名
        'entry_price': 1234.56,        # エントリー価格
        'shares': 100,                 # 株数
        'entry_date': '2025-01-15',    # エントリー日
        'entry_idx': 123               # データフレームインデックス
    },
    '9104.T': { ... }
}
```
**用途**: 
- BUY実行後にポジション追加（`self.positions[symbol] = {...}`）
- SELL実行後にポジション削除（`del self.positions[symbol]`）
- max_positionsチェック（`if len(self.positions) < self.max_positions:`）
- 強制決済時の未決済ポジション確認（`if len(self.positions) > 0:`）

**重要**: Issue #7（BUY/SELL後のself.positions管理漏れ）の教訓から、BUY/SELL実行時の更新処理が必須  
**関連**: 
- [KNOWN_ISSUES_AND_PREVENTION.md - Issue #7](docs/KNOWN_ISSUES_AND_PREVENTION.md)
- [ポジション管理設計テンプレート](docs/templates/POSITION_MANAGEMENT_DESIGN_TEMPLATE.md)

### execution_details（取引記録リスト）
**定義**: バックテスト実行中の全取引を記録するリスト  
**型**: `list[dict]`（各要素は1件の取引情報辞書）  
**構造**:
```python
[
    {
        'date': '2025-01-15',           # 実行日
        'action': 'buy',                # 'buy' または 'sell'
        'symbol': '9101.T',             # 銘柄コード
        'price': 1234.56,               # 約定価格
        'shares': 100,                  # 株数
        'strategy': 'BreakoutStrategy', # 戦略名
        'cash_balance': 950000.0,       # 取引後の残高
        'pnl': None,                    # 損益（BUY時はNone）
    },
    {
        'date': '2025-01-20',
        'action': 'sell',
        'symbol': '9101.T',
        'price': 1345.67,
        'shares': 100,
        'strategy': 'BreakoutStrategy',
        'cash_balance': 1084567.0,
        'pnl': 11111.0,                 # (1345.67 - 1234.56) * 100
    }
]
```
**用途**: 
- BUY/SELL実行後に取引履歴記録（`execution_details.append(...)`）
- all_transactions.csv出力の元データ
- パフォーマンス計算（総収益率、勝率、Profit Factor等）

**出力ファイル**: output/dssms_integration/dssms_*/all_transactions.csv  
**関連**: 
- [統一出力エンジン](src/output/)
- [BUY/SELL処理設計テンプレート](docs/templates/BUY_SELL_PROCESS_DESIGN_TEMPLATE.md)

### stock_data（株価データフレーム）
**定義**: yfinanceから取得した株価データ（OHLCV + Adj Close）  
**型**: `pandas.DataFrame`  
**必須カラム**: Open, High, Low, Close, Volume, Adj Close  
**取得方法**: `data_fetcher.get_parameters_and_data()`  
**重要**: `auto_adjust=False`必須（Adj Closeカラム取得のため）  
**関連**: 
- [copilot-instructions.md - データ取得ルール](../.github/copilot-instructions.md)
- data_fetcher.py, data_cache_manager.py

### max_positions
**定義**: 同時保有可能な最大銘柄数  
**型**: `int`  
**デフォルト値**: 2（Sprint 2マルチポジション対応）  
**用途**: ポジション上限チェック（`if len(self.positions) >= self.max_positions:`）  
**設定場所**: config.yaml または `__init__` メソッド  
**関連**: FIFO決済ロジック（最古のポジションから決済）

---

## C. 戦略関連

### エントリーシグナル（Entry Signal）
**定義**: 買いポジションを開始する条件  
**実装場所**: 各戦略の `generate_entry_signal()` メソッド  
**重要**: ルックアヘッドバイアス禁止（前日データで判断、翌日始値でエントリー）  
**返り値**: True/False または詳細情報辞書  
**例**: ブレイクアウト戦略では高値更新時にTrue

### エグジットシグナル（Exit Signal）
**定義**: 保有ポジションを決済する条件  
**実装場所**: 各戦略の `generate_exit_signal()` メソッド  
**トリガー**: 
- 戦略固有の条件（トレーリングストップ、利益確定、損切り）
- 強制決済（バックテスト終了時）
- DSSMS銘柄切替時の決済

### 強制決済（Forced Exit / FINAL_CLOSE）
**定義**: バックテスト終了時に未決済ポジションを全て決済する処理  
**実行タイミング**: バックテスト期間の最終日、全取引実行後  
**実装要件**:
```python
if len(self.positions) > 0:
    for symbol in list(self.positions.keys()):
        # SELL処理実行（最終日の終値で決済）
        # execution_details記録
        # self.positions削除
```
**目的**: 
- all_transactions.csvに全取引のEXIT情報を記録
- 未決済ポジションによるパフォーマンス計算エラーを防止
- バックテスト結果の正確性担保

**ログ出力**: `[FINAL_CLOSE]`タグ  
**重要**: Issue #7の根本原因の1つ（強制決済が動作しない→EXIT未記録）  
**関連**: 
- [KNOWN_ISSUES_AND_PREVENTION.md - Issue #7](docs/KNOWN_ISSUES_AND_PREVENTION.md)
- [ポジション管理設計テンプレート](docs/templates/POSITION_MANAGEMENT_DESIGN_TEMPLATE.md)

### backtest_daily()
**定義**: 1日単位でバックテストを実行するメソッド  
**用途**: DSSMS統合、動的銘柄選択時の1日ごとの戦略評価  
**返り値**: `dict`（action='buy'/'sell'/'hold', price, shares等）  
**Phase 3-C**: 全5戦略に実装完了  
**関連**: backtest()（全期間バックテスト）

---

## D. パフォーマンス指標

### 総収益率（Total Return）
**定義**: (最終資産 - 初期資産) / 初期資産  
**単位**: %  
**計算式**: `((final_cash + portfolio_value) - initial_capital) / initial_capital * 100`

### シャープレシオ（Sharpe Ratio）
**定義**: リスク調整後リターンの指標  
**計算式**: `(平均リターン - リスクフリーレート) / リターンの標準偏差`  
**目標値**: > 1.0（優秀）、> 2.0（非常に優秀）

### 勝率（Win Rate）
**定義**: 利益が出た取引の割合  
**計算式**: `勝ちトレード数 / 総トレード数 * 100`  
**目標値**: > 50%

### 最大ドローダウン（Maximum Drawdown）
**定義**: 資産が最高値から最も下落した割合  
**単位**: %  
**目標**: DSSMS使用時、固定銘柄比較で20%以上改善

---

## E. ログ・デバッグ

### [POSITION_ADD]
**定義**: ポジション追加時のログタグ  
**出力タイミング**: BUY実行直後、`self.positions[symbol]`追加時  
**フォーマット**: `[POSITION_ADD] ポジション追加: {symbol}, 価格={price}円, 株数={shares}株, 戦略={strategy}`  
**用途**: BUY処理の正常動作確認  
**確認方法**: `grep "\[POSITION_ADD\]" output/dssms_integration/dssms_*/dssms_execution_log.txt`

### [POSITION_DELETE]
**定義**: ポジション削除時のログタグ  
**出力タイミング**: SELL実行直後、`del self.positions[symbol]`時  
**フォーマット**: `[POSITION_DELETE] ポジション削除: {symbol}, PnL={pnl}円({return_pct}%)`  
**用途**: SELL処理の正常動作確認、損益追跡  
**確認方法**: `grep "\[POSITION_DELETE\]" output/dssms_integration/dssms_*/dssms_execution_log.txt`

### [FINAL_CLOSE]
**定義**: 強制決済時のログタグ  
**出力タイミング**: バックテスト終了時、未決済ポジション決済時  
**フォーマット**: `[FINAL_CLOSE] 強制決済: {symbol}, PnL={pnl}円`  
**用途**: 強制決済の動作確認、未決済ポジション検出  
**確認方法**: `grep "\[FINAL_CLOSE\]" output/dssms_integration/dssms_*/dssms_execution_log.txt`

---

## F. ファイル・ディレクトリ

### all_transactions.csv
**定義**: バックテスト実行中の全取引を記録したCSVファイル  
**場所**: output/dssms_integration/dssms_*/all_transactions.csv  
**必須カラム**: 
- entry_date, entry_price, shares（エントリー情報）
- exit_date, exit_price, pnl（エグジット情報）
- symbol, strategy（銘柄・戦略情報）

**重要**: EXIT情報（exit_date, exit_price, pnl）が空の行がある場合、強制決済が動作していない可能性  
**検証方法**: 
```python
import pandas as pd
df = pd.read_csv("output/dssms_integration/dssms_*/all_transactions.csv")
assert df['exit_date'].notna().all(), "exit_date に空の行があります"
```

### docs/templates/
**定義**: 設計テンプレート格納ディレクトリ  
**作成日**: 2026-02-10（Issue #7ギャップ分析の改善提案実施時）  
**主要ファイル**:
- BUY_SELL_PROCESS_DESIGN_TEMPLATE.md: BUY/SELL処理の標準設計フォーマット
- POSITION_MANAGEMENT_DESIGN_TEMPLATE.md: ポジション管理の標準設計フォーマット

**用途**: 新しいBUY/SELL処理やポジション管理を実装する際のガイド

### docs/KNOWN_ISSUES_AND_PREVENTION.md
**定義**: 既知の問題カタログと予防策  
**作成日**: 2026-02-10  
**主要Issue**: Issue #7（BUY/SELL後のself.positions管理漏れ）  
**用途**: 
- 新機能実装前: 関連Issueをチェック
- デバッグ時: 類似症状のIssueを検索
- Git履歴調査: Issue #7でのpositions管理削除→再実装の経緯

---

## G. 開発プロセス

### Issue #7
**タイトル**: BUY/SELL後のself.positions管理漏れ  
**発生時期**: Sprint 2マルチポジション対応実装時  
**深刻度**: P0-Critical  
**症状**: all_transactions.csvにEXIT情報が記録されない、強制決済が実行されない  
**原因**: BUY実行後の`self.positions`追加処理とSELL実行後の削除処理が実装漏れ  
**解決策**: Line 2600-2658にpositions管理追加  
**教訓**: 
- 設計段階: 状態更新（BUY/SELL時のpositions更新）を明示的に設計する
- 実装段階: 4項目チェックリスト（残高、positions、履歴、ログ）で実装漏れを防止
- 検証段階: 内部状態（self.positions）の正確性を検証する

**詳細**: [KNOWN_ISSUES_AND_PREVENTION.md](docs/KNOWN_ISSUES_AND_PREVENTION.md)

### Sprint 2（複数銘柄保有対応）
**目的**: 最大2銘柄の同時保有、FIFO決済、強制決済対応  
**実装日**: 2026-02-10  
**主要変更**: 
- `self.current_position` → `self.positions`辞書に変更
- `self.max_positions = 2`追加
- FIFO決済ロジック実装

**教訓**: Issue #7の発生（ギャップ分析の結果、設計不足・検証漏れを確認）  
**詳細**: [SPRINT2_MULTI_POSITION_COMPLETION_REPORT.md](docs/SPRINT2_MULTI_POSITION_COMPLETION_REPORT.md)

### ルックアヘッドバイアス禁止（Look-Ahead Bias Prevention）
**定義**: 将来のデータを使って過去の判断をしてはいけないルール  
**適用日**: 2025-12-20以降必須  
**基本ルール**: 
- 前日データで判断（インジケーターは`.shift(1)`必須）
- 翌日始値でエントリー（`data['Open'].iloc[idx + 1]`）
- 取引コスト考慮（スリッページ推奨0.1%）

**禁止例**: `entry_price = data['Adj Close'].iloc[idx]`（当日終値でエントリー）  
**正しい例**: `entry_price = data['Open'].iloc[idx + 1] * (1 + slippage)`

---

## H. 略語・頭字語

- **DSSMS**: Dynamic Stock Selection Multi-Strategy System
- **FIFO**: First In First Out（最初に買ったポジションから決済）
- **KPI**: Key Performance Indicator（重要業績評価指標）
- **DD**: Drawdown（ドローダウン）
- **PnL**: Profit and Loss（損益）
- **OHLCV**: Open, High, Low, Close, Volume（始値、高値、安値、終値、出来高）
- **CSV**: Comma-Separated Values（カンマ区切りテキストファイル）
- **API**: Application Programming Interface

---

## 参考資料

- [copilot-instructions.md](../.github/copilot-instructions.md): プロジェクトの基本原則・コーディング規約
- [KNOWN_ISSUES_AND_PREVENTION.md](docs/KNOWN_ISSUES_AND_PREVENTION.md): 既知の問題カタログ
- [DSSMS_ARCHITECTURE.md](docs/DSSMS_ARCHITECTURE.md): DSSMSアーキテクチャ詳細
- [templates/](docs/templates/): 設計テンプレート（BUY/SELL処理、ポジション管理）

---

**注意**: この用語集は、プロジェクトの進化に伴い継続的に更新されます。
新しい重要な用語が登場した場合は、このファイルに追加してください。

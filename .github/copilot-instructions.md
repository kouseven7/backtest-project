# Copilot Instructions for My Backtest Project

## プロジェクト全体像
- 本プロジェクトは「複数戦略のバックテスト・最適化・統合」を目的としたPythonベースの分析基盤です。
- 主要な構成要素は「データ取得・前処理」「戦略実装」「パラメータ最適化」「リスク管理」「統合シミュレーション」「結果出力（Excel/グラフ）」です。
- 各戦略は独立したクラス（`strategies/`配下）で実装され、`main.py`で優先度順に統合されます。
- 戦略パラメータは`config/optimized_parameters.py`や設定JSONで管理され、承認済みパラメータのみが本番実行に利用されます。
- リスク管理（`config/risk_management.py`）や重み学習（`config/weight_learning_config/`）など、各種最適化・制約モジュールが存在します。

## 主要ワークフロー
- バックテストは`main.py`がエントリーポイント。`python main.py`で実行。
- データ取得・前処理は`data_fetcher.py`・`data_processor.py`で行い、戦略ごとに`backtest()`メソッドを呼び出します。
- 統合システム（`config/multi_strategy_manager.py`）が利用可能な場合はそちらを優先。
- 結果は`output/simulation_handler.py`経由でExcel等に出力。
- パラメータ最適化や重み学習は`analysis/`や`config/weight_learning_config/`配下の専用モジュールを参照。

## コーディング規約・パターン
- 戦略クラスは`backtest()`メソッドを持ち、`Entry_Signal`/`Exit_Signal`列でシグナルを返すこと。
- シグナル統合は優先度順で、既存シグナルがなければ上書き。
- パラメータは必ず承認済み（またはデフォルト）を使う。未承認時は`get_default_parameters()`で補完。
- ロギングは`config/logger_config.py`の`setup_logger`を利用。
- 設定ファイル（特に重み学習やリスク最適化）はJSON形式で、変更時は妥当性検証を推奨。

## 重要ファイル・ディレクトリ
- `main.py`：全体の統合・実行フロー
- `strategies/`：各戦略クラス（例：`VWAPBreakoutStrategy`等）
- `config/`：パラメータ・リスク管理・統合管理
- `analysis/`：リスク調整最適化・パフォーマンス評価
- `output/`：結果出力・レポート生成
- `data_fetcher.py`/`data_processor.py`：データ取得・前処理

## テスト・デバッグ
- テストは`conftest.py`や`*_test.py`で実装。pytest推奨。
- ログは`logs/`配下に出力。エラー時は詳細ログを確認。
- 戦略追加時は必ず`backtest()`の出力形式（シグナル列）を守ること。

## 外部依存・連携
- データ取得は`yfinance`等の外部API利用。
- Excel出力は`openpyxl`。
- 最適化・学習系は`scipy`や`bayesian-optimization`等を利用する場合あり。

---

このドキュメントはAIエージェント向けのガイドです。不明点や新規パターン発見時はREADMEや各ディレクトリのREADMEも参照してください。

## PowerShellコマンド連結の注意
- Windows PowerShell環境では、複数コマンドを連結する際に `&&` ではなく `;` を使用してください。
  - 例: `python main.py ; echo done`
  - `&&` はPowerShellでは動作しません。

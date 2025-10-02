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

## DSSMS 優先ガイド (新規)
- Primary Focus: DSSMS Core (ranking / scoring / switching)
- コード生成優先順位: dssms_* → adaptive/ranking/perfect_order → multi-strategy → 出力
- 未実装プレースホルダは pass ではなく TODO(tag:phase, rationale) コメント必須
- 切替イベント: SwitchEvent(JSON serializable) を破壊しない変更推奨
- 再現性要素: seed, deterministic_mode, cache invalidation keys
- Perfect Order 判定結果には timeframe 別 MA 値と判定フラグを保持

## Error Severity Policy (フォールバック対応版更新)
- CRITICAL: SystemFallbackPolicy.handle_component_failure() → raise (PRODUCTION mode)
- ERROR: SystemFallbackPolicy.handle_component_failure() → 明示的フォールバック (DEVELOPMENT mode)
- WARNING: ログのみ、フォールバック使用記録
- INFO/DEBUG: 状態追跡、フォールバック統計

## DSSMS フォールバック・テストデータポリシー (2025年10月追加)

### システムモード管理
- PRODUCTION: フォールバック禁止、エラーで即停止
- DEVELOPMENT: 明示的フォールバック許可、ログ必須  
- TESTING: モック/テストデータ許可

### フォールバック実装必須パターン
```python
# ✅ 必須: 統一フォールバック処理
from src.config.system_modes import SystemFallbackPolicy, ComponentType, SystemMode

def component_method(self):
    try:
        return primary_function()
    except Exception as e:
        return self.fallback_policy.handle_component_failure(
            component_type=ComponentType.DSSMS_CORE,  # 適切なタイプ
            component_name="ComponentName",
            error=e,
            fallback_func=explicit_fallback_function
        )

# ❌ 禁止: サイレントフォールバック
def component_method(self):
    try:
        return primary_function() 
    except:
        return random_fallback()  # 問題を隠蔽
```

### コンポーネントタイプ分類
- `DSSMS_CORE`: ランキング、スコアリング、銘柄選択
- `STRATEGY_ENGINE`: 個別戦略 (VWAP, Bollinger等)
- `DATA_FETCHER`: yfinance, データ取得
- `RISK_MANAGER`: リスク管理、ポジション管理
- `MULTI_STRATEGY`: 統合システム、戦略統合

### TODOタグ必須規約
- フォールバック関数: `TODO(tag:phase2, rationale:eliminate after X integration)` 
- モック/テストデータ: `TODO(tag:mock, rationale:replace with real data)`
- 暫定実装: `TODO(tag:temporary, rationale:improve performance)`

### モック・テストデータ識別必須
- プレフィックス必須: `MOCK_`, `TEST_`, `DEMO_`
- ファイル名規約: `*_mock.py`, `*_test_data.py`
- 実データとの混在禁止
- TESTING modeでのみ使用許可

### フォールバック除去計画
1. Phase 1: 明示的フォールバックに変更 (TODO-FB-001~008)
2. Phase 2: 段階的フォールバック除去 (TODO-DSSMS-001~005)
3. Phase 3: Production readiness確認 (TODO-QG-001~002)

### フォールバック品質ゲート
- Production mode: フォールバック使用量 = 0 必須
- Development mode: フォールバック使用記録・監視必須
- Testing mode: モック/テストデータのみ許可

## KPI メタデータ出力
- 50銘柄ランキング処理時間(ms)
- 不要切替判定結果 (pending / evaluated)
- Excel export hash (内容一意性)

## 切替評価
- 不要切替: 10営業日後 (p_after - p_before)/p_before - cost ≤ 0
- cost デフォルト: 0.2% (設定値で上書き可)

## 方針
- kabu 実行タイミング固定しない
- 強化学習: 現時点導入予定なし

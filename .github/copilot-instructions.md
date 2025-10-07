# Copilot Instructions for My Backtest Project

## プロジェクト全体像
- 本プロジェクトは「複数戦略のバックテスト・最適化・統合」を目的としたPythonベースの分析基盤です。
- 主要な構成要素は「データ取得・前処理」「戦略実装」「パラメータ最適化」「リスク管理」「統合シミュレーション」「結果出力（Excel/グラフ）」です。
- 各戦略は独立したクラス（`strategies/`配下）で実装され、`main.py`で優先度順に統合されます。
- 戦略パラメータは`config/optimized_parameters.py`や設定JSONで管理され、承認済みパラメータのみが本番実行に利用されます。
- リスク管理（`config/risk_management.py`）や重み学習（`config/weight_learning_config/`）など、各種最適化・制約モジュールが存在します。

## バックテスト基本理念 (最優先原則) 🎯

### **プロジェクト存在目的遵守**
- **本プロジェクトの存在理由は「バックテスト実行」**：全ての機能・統合・最適化は実際のバックテスト実行を前提とし、これを損なう変更は禁止
- **シグナル生成必須**: いかなるシステム（main.py、DSSMS、MultiStrategyManager等）においても、実際の戦略による`Entry_Signal`/`Exit_Signal`生成は省略禁止
- **取引実行必須**: バックテスト期間における実際の売買シミュレーション実行は必須機能

### **戦略実行必須パターン** 🔒
```python
# ✅ 必須: 実際の戦略実行（main.py / DSSMS / MultiStrategyManager共通）
def execute_strategy_flow(self, stock_data, strategy_name, params):
    """バックテスト基本理念遵守: 実際の戦略実行必須"""
    # 戦略インスタンス化
    strategy_class = get_strategy_class(strategy_name)
    strategy_instance = strategy_class(**params)
    
    # 必須: 実際のbacktest()実行
    result = strategy_instance.backtest(stock_data)
    
    # 必須: シグナル検証（基本理念違反検出）
    if 'Entry_Signal' not in result.columns or 'Exit_Signal' not in result.columns:
        raise ValueError(f"Strategy {strategy_name} violates backtest principle: Entry_Signal/Exit_Signal missing")
    
    # 必須: 取引数検証
    total_trades = (result['Entry_Signal'] == 1).sum() + (result['Exit_Signal'] == 1).sum()
    if total_trades == 0:
        logger.warning(f"Strategy {strategy_name}: No trades generated - verify strategy logic")
    
    return result

# ❌ 禁止: メタデータのみ・ハードコード・モックシグナル
def execute_strategy_flow_invalid(self, stock_data, strategy_name, params):
    # 基本理念違反: 実際のbacktest()なし
    return {"signal": "hold", "confidence": 0.7}  # TODO(tag:backtest_execution, rationale:violates backtest principle)
```

### **統合システム実装必須要件** 🧪
```python
# MultiStrategyManager / DSSMS統合システム必須実装パターン
class MultiStrategyManager:
    def execute_multi_strategy_flow(self, market_data, available_strategies):
        """
        バックテスト基本理念遵守: 統合システムでも実際のbacktest()実行必須
        """
        integrated_results = pd.DataFrame()
        strategy_performances = {}
        
        for strategy_name in available_strategies:
            # ✅ 必須: 実際の戦略backtest()実行
            strategy_result = self._execute_actual_strategy_backtest(
                market_data, strategy_name, self.optimized_params[strategy_name]
            )
            
            # ✅ 必須: シグナル・取引履歴検証
            self._validate_backtest_output(strategy_result, strategy_name)
            
            # 統合処理
            integrated_results = self._integrate_strategy_signals(integrated_results, strategy_result)
            strategy_performances[strategy_name] = self._calculate_performance_metrics(strategy_result)
        
        # ✅ 必須: 統合後もExcel出力可能な形式で返す
        return self._format_for_excel_output(integrated_results, strategy_performances)
    
    def _validate_backtest_output(self, result, strategy_name):
        """基本理念違反検出"""
        required_columns = ['Entry_Signal', 'Exit_Signal', 'Close', 'Position']
        missing_columns = [col for col in required_columns if col not in result.columns]
        
        if missing_columns:
            raise ValueError(f"Strategy {strategy_name} violates backtest principle: missing {missing_columns}")
        
        total_trades = (result['Entry_Signal'] == 1).sum() + (result['Exit_Signal'] == 1).sum()
        if total_trades == 0:
            logger.warning(f"Strategy {strategy_name}: Zero trades - potential backtest principle violation")
```

### **DSSMS統合バックテスト必須要件** 📊
```python
# DSSMS バックテスト基本理念遵守パターン
class DSSMSBacktester:
    def execute_dssms_backtest(self, stock_universe, strategies):
        """
        DSSMS統合でもバックテスト基本理念遵守必須
        """
        # ✅ 必須: 実際の銘柄選択・戦略実行
        for period in self.backtest_periods:
            # DSSMS銘柄選択
            selected_stocks = self.dssms_core.select_stocks(stock_universe, period)
            
            for stock_code in selected_stocks:
                stock_data = self.get_stock_data(stock_code, period)
                
                # ✅ 必須: 各戦略で実際のbacktest()実行
                for strategy_name in strategies:
                    strategy_result = self._execute_strategy_backtest(stock_data, strategy_name)
                    
                    # ✅ 必須: DSSMS切替判定でもシグナル検証
                    self._validate_dssms_signals(strategy_result, stock_code, strategy_name)
                    
                    # DSSMS特有の切替記録
                    switch_events = self._record_dssms_switches(strategy_result, stock_code)
        
        # ✅ 必須: DSSMS統合結果もExcel出力可能形式
        return self._format_dssms_results_for_excel()
```

### **基本理念違反検出システム** 🚨
```python
# 基本理念違反自動検出
def validate_backtest_principle_compliance(result_data, component_name):
    """バックテスト基本理念違反の自動検出"""
    violations = []
    
    # 1. シグナル列存在チェック
    required_signal_columns = ['Entry_Signal', 'Exit_Signal']
    missing_signals = [col for col in required_signal_columns if col not in result_data.columns]
    if missing_signals:
        violations.append(f"Missing signal columns: {missing_signals}")
    
    # 2. 取引数チェック
    if 'Entry_Signal' in result_data.columns and 'Exit_Signal' in result_data.columns:
        total_trades = (result_data['Entry_Signal'] == 1).sum() + (result_data['Exit_Signal'] == 1).sum()
        if total_trades == 0:
            violations.append("Zero trades generated - potential strategy logic issue")
    
    # 3. データ完整性チェック
    if len(result_data) == 0:
        violations.append("Empty result data")
    
    # 4. Excel出力要件チェック
    excel_required_columns = ['Close', 'Position', 'Portfolio_Value']
    missing_excel_columns = [col for col in excel_required_columns if col not in result_data.columns]
    if missing_excel_columns:
        violations.append(f"Excel output columns missing: {missing_excel_columns}")
    
    # 違反検出時の処理
    if violations:
        error_msg = f"Backtest principle violations in {component_name}: {'; '.join(violations)}"
        logger.error(error_msg)
        raise ValueError(f"{error_msg} TODO(tag:backtest_execution, rationale:fix principle violations)")
    
    return True
```

## 主要ワークフロー
- バックテストは`main.py`がエントリーポイント。`python main.py`で実行。
- データ取得・前処理は`data_fetcher.py`・`data_processor.py`で行い、戦略ごとに`backtest()`メソッドを呼び出します。
- 統合システム（`config/multi_strategy_manager.py`）が利用可能な場合はそちらを優先。**ただし統合システムでも実際のbacktest()実行は必須**。
- 結果は`output/simulation_handler.py`経由でExcel等に出力。
- パラメータ最適化や重み学習は`analysis/`や`config/weight_learning_config/`配下の専用モジュールを参照。

## コーディング規約・パターン
- **戦略クラスは`backtest()`メソッドを持ち、`Entry_Signal`/`Exit_Signal`列でシグナルを返すこと（基本理念遵守）**。
- シグナル統合は優先度順で、既存シグナルがなければ上書き。
- パラメータは必ず承認済み（またはデフォルト）を使う。未承認時は`get_default_parameters()`で補完。
- ロギングは`config/logger_config.py`の`setup_logger`を利用。
- 設定ファイル（特に重み学習やリスク最適化）はJSON形式で、変更時は妥当性検証を推奨。
- **バックテスト基本理念違反時は`TODO(tag:backtest_execution, rationale:violation description)`でマーク必須**。

## 重要ファイル・ディレクトリ
- `main.py`：全体の統合・実行フロー **（バックテスト基本理念の最終責任者）**
- `strategies/`：各戦略クラス（例：`VWAPBreakoutStrategy`等）
- `config/`：パラメータ・リスク管理・統合管理
- `analysis/`：リスク調整最適化・パフォーマンス評価
- `output/`：結果出力・レポート生成
- `data_fetcher.py`/`data_processor.py`：データ取得・前処理

## テスト・デバッグ
- テストは`conftest.py`や`*_test.py`で実装。pytest推奨。
- ログは`logs/`配下に出力。エラー時は詳細ログを確認。
- **戦略追加時は必ず`backtest()`の出力形式（シグナル列）を守ること（基本理念遵守）**。
- **取引数0件やシグナル欠損は基本理念違反として即座に調査・修正**。

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
- **DSSMS統合でもバックテスト基本理念遵守必須：実際の戦略backtest()実行・シグナル生成・Excel出力対応**

## Error Severity Policy (フォールバック対応版更新)
- CRITICAL: SystemFallbackPolicy.handle_component_failure() → raise (PRODUCTION mode)
- ERROR: SystemFallbackPolicy.handle_component_failure() → 明示的フォールバック (DEVELOPMENT mode)
- WARNING: ログのみ、フォールバック使用記録
- INFO/DEBUG: 状態追跡、フォールバック統計
- **バックテスト基本理念違反: CRITICAL扱い（PRODUCTION mode即停止、DEVELOPMENT mode明示的修正要求）**

## DSSMS フォールバック・テストデータポリシー (2025年10月追加)

### システムモード管理
- PRODUCTION: フォールバック禁止、エラーで即停止、**バックテスト基本理念違反で即停止**
- DEVELOPMENT: 明示的フォールバック許可、ログ必須、**基本理念違反時は修正要求**
- TESTING: モック/テストデータ許可、**ただし基本理念（シグナル生成・取引実行）は遵守**

### フォールバック実装必須パターン
```python
# ✅ 必須: 統一フォールバック処理（バックテスト基本理念遵守版）
from src.config.system_modes import SystemFallbackPolicy, ComponentType, SystemMode

def component_method(self):
    try:
        # ✅ 基本理念遵守: 実際のbacktest実行
        result = primary_backtest_function()
        validate_backtest_principle_compliance(result, "ComponentName")
        return result
    except Exception as e:
        return self.fallback_policy.handle_component_failure(
            component_type=ComponentType.STRATEGY_ENGINE,
            component_name="ComponentName", 
            error=e,
            fallback_func=lambda: explicit_backtest_fallback()  # フォールバックでも基本理念遵守
        )

# ❌ 禁止: バックテスト基本理念違反フォールバック
def component_method_invalid(self):
    try:
        return primary_function()
    except:
        return {"trades": 0, "signals": None}  # 基本理念違反: シグナルなし
```

### コンポーネントタイプ分類
- `DSSMS_CORE`: ランキング、スコアリング、銘柄選択 **（基本理念遵守：選択後は実際の戦略実行必須）**
- `STRATEGY_ENGINE`: 個別戦略 (VWAP, Bollinger等) **（基本理念の中核：backtest()実行必須）**
- `DATA_FETCHER`: yfinance, データ取得
- `RISK_MANAGER`: リスク管理、ポジション管理
- `MULTI_STRATEGY`: 統合システム、戦略統合 **（基本理念遵守：統合でもbacktest()実行必須）**

### TODOタグ必須規約
- フォールバック関数: `TODO(tag:phase2, rationale:eliminate after X integration)`
- モック/テストデータ: `TODO(tag:mock, rationale:replace with real data)`
- 暫定実装: `TODO(tag:temporary, rationale:improve performance)`
- **基本理念違反: `TODO(tag:backtest_execution, rationale:implement actual backtest execution)`**

### モック・テストデータ識別必須
- プレフィックス必須: `MOCK_`, `TEST_`, `DEMO_`
- ファイル名規約: `*_mock.py`, `*_test_data.py`
- 実データとの混在禁止
- TESTING modeでのみ使用許可
- **テストデータでもバックテスト基本理念（シグナル生成・取引実行）は遵守必須**

### フォールバック除去計画
1. Phase 1: 明示的フォールバックに変更 (TODO-FB-001~008)
2. Phase 2: 段階的フォールバック除去 (TODO-DSSMS-001~005)
3. Phase 3: Production readiness確認 (TODO-QG-001~002)
4. **Phase 4: バックテスト基本理念完全遵守確認 (TODO-BE-001~003)**

### バックテスト品質ゲート（基本理念統合版）
- **Production mode**: フォールバック使用量 = 0 必須、**取引数 > 0必須、シグナル生成必須**
- **Development mode**: フォールバック使用記録・監視必須、**基本理念違反時は修正要求**
- **Testing mode**: モック/テストデータのみ許可、**基本理念遵守は必須**

## KPI メタデータ出力
- 50銘柄ランキング処理時間(ms)
- 不要切替判定結果 (pending / evaluated)
- Excel export hash (内容一意性)
- **バックテスト品質指標: 総取引数、シグナル生成率、Excel出力完整性**

## 切替評価
- 不要切替: 10営業日後 (p_after - p_before)/p_before - cost ≤ 0
- cost デフォルト: 0.2% (設定値で上書き可)
- **切替評価でも基本理念遵守：実際のbacktest結果に基づく評価必須**

## 方針
- kabu 実行タイミング固定しない
- 強化学習: 現時点導入予定なし
- **バックテスト基本理念は全てに優先：いかなる統合・最適化でも実際のbacktest実行は必須**

## AI エージェント応答品質ガイド 🤖

### **正確性・誠実性必須原則**
- **存在しない機能・段階・計画への言及禁止**: 実装されていない機能やフェーズを即座に作り出して回答することは禁止
- **ドキュメント準拠必須**: 回答は既存のドキュメント・実装・計画に基づく内容のみとし、その場での創作は行わない
- **不明・未実装時の正直な回答**: 「存在しません」「実装されていません」「計画されていません」を明確に伝える

### **回答検証必須パターン** 🔍
```python
# ✅ 正しい回答パターン
def answer_about_feature(feature_name):
    # 1. 実際のドキュメント・コード確認
    if not exists_in_documentation(feature_name):
        return f"{feature_name} は現在実装されていません"
    
    # 2. 既存情報に基づく回答
    return get_documented_information(feature_name)

# ❌ 禁止: その場での創作回答
def answer_about_feature_invalid(feature_name):
    # 基本方針違反: 存在しない機能を即座に創作
    return f"{feature_name} の実装方針は..." # 実際には存在しない計画
```

### **未実装・不明時の必須対応**
- **「存在しません」を明確に伝達**: 曖昧な表現や回避ではなく、直接的な事実確認
- **必要に応じた提案**: 「実装が必要でしたら新たに計画いたします」等の建設的提案
- **TODO化**: 新規要求は `TODO(tag:new_requirement, rationale:user requested feature)` でマーク

### **品質保証チェックリスト**
- [ ] 既存ドキュメントで確認済み
- [ ] 実装コードで動作確認済み  
- [ ] 計画書で予定確認済み
- [ ] 未実装の場合は正直に回答済み
- [ ] 創作・推測での回答を避けている

### **違反時の対処**
```python
# 創作回答検出時の修正パターン
if detected_fabricated_response():
    logger.error("Fabricated response detected - correcting")
    return correct_factual_response_or_admit_unknown()
    # TODO(tag:response_quality, rationale:prevent fabricated answers)
```

---

**このガイドラインにより、AI エージェントは実装されていない機能について即座に創作回答することなく、正確で誠実な対応を行います。**

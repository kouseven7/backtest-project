#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Output problem solving roadmap2.md の表記変更
Task → Problem/Resolution 形式に変更
"""

# ロードマップファイルの更新
roadmap_content = '''
# DSSMS 出力 / 構造問題 解決ロードマップ（目的再定義版）

## 0. ---

## 3. 解決策の評価・実装フレームワーク

### 3.1 有効な評価基準

1. **目的整合性**: 解決策が---

## 5. 問題別解決策（Problem/Resolution）

### Problem 1: 切替判定ロジック劣化（117回→3回の激減）

#### 問題詳細
- **現象**: 切替数が117回から3回に激減
- **重要度**: Critical Priority 🚨
- **効率値**: 31.7（改善効果95%/工数3.0）
- **カテゴリ**: A. 切替メカニズム劣化
- **関連KPI**: 適正切替頻度、不要切替率<20%

#### 根本原因

1. **設定過度制限**: `randomness_control.switching.enable_probabilistic: false`により確率的切替が完全無効化
2. **判定硬直化**: `switch_criteria.score_difference_threshold: 0.15`の閾値が過度に厳格で現実的でない
3. **ノイズ抑制**: `randomness_control.scoring.enable_noise: false`により微細な優位性変化を検出不能

#### 解決策 (Resolution 1)

**【実装概要】**: 決定論的モード設定の段階的緩和による切替判定柔軟性回復

**【変更対象】**: `config/dssms/dssms_backtester_config.json`

**【実装内容】**:
```json
{
  "randomness_control": {
    "switching": {
      "enable_probabilistic": true,        // False → True (確率的切替有効化)
      "strict_threshold_mode": false       // True → False (厳格モード解除)
    },
    "scoring": {
      "enable_noise": true,               // False → True (微細変動検出)
      "noise_level": 0.02                 // 0.0 → 0.02 (2%ノイズ許容)
    }
  },
  "switch_criteria": {
    "score_difference_threshold": 0.08,   // 0.15 → 0.08 (閾値緩和)
    "confidence_threshold": 0.5,          // 0.6 → 0.5 (信頼度緩和)
    "minimum_holding_period_hours": 2     // 4 → 2 (保有期間短縮)
  },
  "risk_control": {
    "max_daily_switches": 5,              // 3 → 5 (日次切替上限緩和)
    "max_weekly_switches": 20             // 10 → 20 (週次切替上限緩和)
  }
}
```

**【期待効果】**: 切替数3回→90-120回レベル回復、適正切替頻度KPI達成、不要切替率<20%維持

#### 検証計画

1. **設定変更前ベースライン取得** - 現状3回切替の詳細ログ保存・分析
2. **段階的設定変更テスト** - enable_probabilistic有効化→閾値緩和→ノイズ有効化の順で実施
3. **30日間切替数測定** - 目標レンジ90-120回への回復確認、不要切替率<20%維持確認
4. **再現性確認** - seed固定での2回実行で切替タイミング・回数の一致性確認
5. **パフォーマンス影響評価** - 総リターン・シャープレシオへの正当な改善効果確認

#### 完了条件

- **定量的条件1**: 30日間テスト期間での切替数90-120回レンジ達成
- **定量的条件2**: 不要切替率≤20%（10営業日後収益性による評価）
- **定量的条件3**: 決定論的再現性維持（seed固定2回実行で±5%以内の切替数差異）
- **定性的条件**: 85.0点エンジンの出力品質維持、統一出力システムとの整合性確保

#### 実装状況
- [✓] 解決策の検討・確定
- [ ] コード変更の実装
- [ ] テスト実行
- [ ] KPI評価
- [ ] 完了確認

---

### Problem 12/3（統合）: 決定論的モード設定問題

#### 問題詳細
- **現象**: 過度な決定論的モード設定による切替判定の硬直化
- **重要度**: High Priority
- **効率値**: 170.0（改善効果85%/工数0.5）
- **カテゴリ**: A. 切替メカニズム劣化, F. 再現性設定
- **関連KPI**: 適正切替頻度、決定論差分=ゼロ

#### 根本原因

1. **設定統合問題**: Problem 1と同一の`config/dssms/dssms_backtester_config.json`設定により、確率的切替とノイズが同時無効化
2. **再現性過重視**: `deterministic: true`、`enable_reproducible_results: true`の組み合わせで過度な決定論的動作を強制
3. **統合設計不備**: 再現性要件と実用性要件のバランス調整が未実装で、DSSMS Core機能の柔軟性が損なわれている

#### 解決策 (Resolution 12/3)

**【実装概要】**: Problem 1統合による決定論的モード最適化と再現性バランス調整

**【変更対象】**: `config/dssms/dssms_backtester_config.json`（Problem 1と共通設定）

**【実装内容】**:
```json
{
  "execution_mode": {
    "deterministic": true,                    // 維持（seed固定再現性確保）
    "random_seed": 42,                        // 維持（決定論差分=ゼロ保証）
    "enable_reproducible_results": true       // 維持（KPI要件準拠）
  },
  "randomness_control": {
    "scoring": {
      "enable_noise": true,                   // False → True（微細変動検出有効化）
      "noise_level": 0.02                     // 0.0 → 0.02（2%ノイズで柔軟性確保）
    },
    "ranking": {
      "tie_breaking": "deterministic"         // 維持（再現性保証）
    },
    "switching": {
      "enable_probabilistic": true,           // 追加（確率的切替有効化）
      "switching_probability": 0.9,           // 追加（90%確率で適度な柔軟性）
      "strict_threshold_mode": false          // True → False（厳格モード解除）
    }
  },
  "performance_calculation": {
    "use_fixed_execution_price": false,       // True → False（価格変動考慮）
    "execution_delay_simulation": false,      // 維持
    "slippage_simulation": false,             // 維持
    "fixed_commission_rate": 0.001            // 維持
  }
}
```

**【期待効果】**: 
- 適正切替頻度KPI達成（切替数回復）
- 決定論差分=ゼロ維持（seed固定再現性確保）
- Problem 1との相乗効果による統合最適化

#### 検証計画

1. **統合動作確認** - Problem 1解決策との同時適用で相互影響なし確認
2. **再現性検証** - seed固定2回実行で決定論差分=ゼロ維持確認
3. **切替頻度測定** - 適正切替頻度KPI達成（90-120回レンジ）確認
4. **ノイズレベル調整** - noise_level 0.02の効果測定と必要に応じた微調整
5. **統合パフォーマンス評価** - Problem 1との統合効果による総合改善確認

#### 完了条件

- **定量的条件1**: seed固定2回実行で決定論差分=ゼロ維持（KPI要件）
- **定量的条件2**: 適正切替頻度90-120回レンジ達成（Problem 1と共通目標）
- **定量的条件3**: noise_level 0.02での微細変動検出機能正常動作確認
- **定性的条件**: 85.0点エンジン出力品質維持、既存SwitchEvent JSON互換性保証

#### 実装状況
- [✓] 解決策の検討・確定
- [ ] コード変更の実装
- [ ] テスト実行
- [ ] KPI評価
- [ ] 完了確認

---

### Problem 6: データフロー/ポートフォリオ処理混乱

#### 問題詳細
- **現象**: ポートフォリオデータの複雑な参照・処理構造
- **重要度**: High Priority
- **効率値**: 24.0（改善効果60%/工数2.5）
- **カテゴリ**: B. アーキ/データフロー混乱
- **関連KPI**: 出力整合、構造健全

#### 根本原因

1. **データフロー分散**: `DSSMSBacktester.portfolio_values`が27箇所で直接参照・操作され、データ整合性管理が困難
2. **変換ロジック不統一**: v1,v2,v4エンジン間で実装差異があり、データ形式変換処理が統一されていない
3. **日付処理散在**: v4エンジンで8箇所の`pd.to_datetime`使用による処理重複と保守性劣化

#### 解決策 (Resolution 6)

**【実装概要】**: PortfolioDataManagerによるデータフロー一元管理と処理統合

**【変更対象】**: 
- `src/dssms/dssms_backtester.py`（主要変更）
- `output/dssms_unified_output_engine.py`（連携部分）
- `src/dssms/portfolio_data_manager.py`（新規作成）

**【実装内容】**:
```python
class PortfolioDataManager:
    """ポートフォリオデータの一元管理・処理統合"""
    
    def __init__(self, backtester_instance):
        self._backtester = backtester_instance
        self._portfolio_cache = {}
        self._date_processor = DateProcessor()
        
    def get_portfolio_values(self, date=None, format_type='standard'):
        """統一ポートフォリオ値取得"""
        # TODO(tag:phase2, rationale:DSSMS Core focus): キャッシュ最適化実装
        if date:
            date = self._date_processor.normalize_date(date)
        return self._backtester.portfolio_values.get(date, {})
        
    def update_portfolio_values(self, date, values, validation=True):
        """統一ポートフォリオ値更新（整合性チェック付き）"""
        if validation:
            self._validate_portfolio_data(values)
        self._backtester.portfolio_values[date] = values
        
    def convert_engine_format(self, data, source_engine, target_engine):
        """エンジン間データ形式変換統一"""
        # v1,v2,v4の実装差異を吸収
        pass  # TODO(tag:phase2, rationale:統一変換ロジック実装)

# filepath: src/dssms/dssms_backtester.py
class DSSMSBacktester:
    def __init__(self):
        # ...existing code...
        self.portfolio_manager = PortfolioDataManager(self)  # 新規追加
        
    def _update_portfolio_performance(self, date, symbol_data):
        """27箇所参照の統一化"""
        # 変更前: self.portfolio_values[date] = symbol_data
        # 変更後: 
        self.portfolio_manager.update_portfolio_values(date, symbol_data)
```

**【期待効果】**: 
- 27箇所データ参照の統一化により出力整合性確保
- エンジン間変換ロジック統一による構造健全性向上
- 日付処理統合（8箇所→2-3箇所）による保守性改善

#### 検証計画

1. **データ整合性テスト** - 変更前後でportfolio_values内容の完全一致確認
2. **27箇所参照動作確認** - 全参照箇所でPortfolioDataManager経由の正常動作検証
3. **エンジン間変換テスト** - v1,v2,v4間でのデータ形式変換正常性確認
4. **日付処理統合検証** - 8箇所→2-3箇所削減の機能影響確認
5. **パフォーマンステスト** - 50銘柄ランキング処理時間<30s維持確認

#### 完了条件

- **定量的条件1**: portfolio_values参照箇所の90%以上をPortfolioDataManager経由に統一
- **定量的条件2**: v1,v2,v4エンジン間データ変換の100%正常動作確認
- **定量的条件3**: 日付処理箇所を8箇所から3箇所以下に削減
- **定性的条件**: 85.0点エンジン出力品質維持、既存SwitchEvent構造の非破壊保証

#### 実装状況
- [✓] 解決策の検討・確定
- [ ] コード変更の実装
- [ ] テスト実行
- [ ] KPI評価
- [ ] 完了確認

---

### Problem 13: エンジン競合解決

#### 問題詳細
- **現象**: 103個のエンジン重複による混乱
- **重要度**: High Priority
- **効率値**: 13.0（改善効果52%/工数4.0）
- **カテゴリ**: B. アーキ/データフロー混乱
- **関連KPI**: 構造健全（重複エンジン整理率>90%）

#### 根本原因

1. **エンジン管理方針不在**: DSSMS統合過程で段階的追加されたエンジンの採用/アーカイブ基準が未確立、開発履歴とプロダクション品質の境界が曖昧
2. **品質評価体系欠如**: 103個エンジンの機能・性能・コード品質の体系的評価が未実施で、85.0点エンジン以外の相対品質が不明、重複・劣化エンジンが残存
3. **アーカイブ戦略未確立**: 旧バージョンや実験的実装の整理手順が未定義で、`output/`配下に開発段階の成果物が無秩序蓄積

#### 解決策 (Resolution 13)

**【実装概要】**: 品質評価ベース段階的エンジン整理による構造健全化

**【変更対象】**: 
- `output/` 配下の全エンジンファイル（103個）
- `src/dssms/` 配下の関連エンジン
- `archive/engines/` （新規ディレクトリ作成）
- `docs/engine_management.md` （新規作成）

**【実装内容】**:
```python
"""エンジン監査・整理スクリプト"""

class EngineAuditManager:
    """エンジン品質評価・整理管理"""
    
    def __init__(self):
        self.engine_registry = {}
        self.quality_metrics = {}
        
    def audit_all_engines(self):
        """全エンジンの品質評価実行"""
        # TODO(tag:phase2, rationale:DSSMS Core focus): 品質評価実装
        engines = self._discover_engines()
        for engine in engines:
            self.quality_metrics[engine] = self._evaluate_engine_quality(engine)
            
    def classify_engines(self):
        """エンジン分類（採用/アーカイブ/削除）"""
        classification = {
            'adopted': [],      # 85.0点エンジン等の高品質
            'archived': [],     # 履歴保存価値あり
            'deprecated': []    # 削除対象
        }
        
        # 85.0点エンジンを標準採用
        classification['adopted'].append('dssms_unified_output_engine.py')
        
        return classification
        
    def execute_reorganization(self, classification):
        """エンジン再編成実行"""
        # アーカイブディレクトリ作成
        os.makedirs('archive/engines/', exist_ok=True)
        
        # 分類に基づく移動処理
        for category, engines in classification.items():
            if category == 'archived':
                self._move_to_archive(engines)
            elif category == 'deprecated':
                self._safe_remove(engines)
```

**【期待効果】**: 
- 103個→5個以下エンジン削減（重複エンジン整理率>90%達成）
- 85.0点エンジン標準化による品質統一
- アーキテクチャ簡素化によるメンテナンス性向上

#### 検証計画

1. **エンジン使用状況調査** - 各エンジンの実際の参照・使用箇所を全ファイルから検索・特定
2. **品質評価実行** - 85.0点エンジンとの比較による相対品質評価（機能・性能・コード品質）
3. **依存関係分析** - 各エンジンの相互依存関係と影響範囲の完全マッピング
4. **段階的移行テスト** - アーカイブ対象エンジンの段階的無効化による動作影響確認
5. **統合動作確認** - 整理後のDSSMS全体実行による機能正常性確認

#### 完了条件

- **定量的条件1**: エンジンファイル数を103個から10個以下に削減（整理率>90%）
- **定量的条件2**: 採用エンジンの品質評価全てが80点以上を維持
- **定量的条件3**: アーカイブ・削除されたエンジンの依存関係エラー0件
- **定性的条件**: 85.0点エンジンの出力品質維持、DSSMS Core機能の非破壊保証、アーカイブ戦略文書化完了

#### 実装状況
- [✓] 解決策の検討・確定
- [ ] コード変更の実装
- [ ] テスト実行
- [ ] KPI評価
- [ ] 完了確認

---

### Problem 10: 数学的エラー修正

#### 問題詳細
- **現象**: 計算式エラー率160%の高い数学的不整合
- **重要度**: Medium Priority
- **効率値**: 45.0（改善効果90%/工数2.0）
- **カテゴリ**: D. 統計/数式実装欠損
- **関連KPI**: 数式健全（計算式エラー率<5%）

#### 根本原因

1. **分母欠如エラー**: 勝率・ProfitFactor計算で取引数=0時の分母チェック未実装、ZeroDivisionError頻発
2. **計算式実装誤り**: 平均損益・最大ドローダウン・シャープレシオで数学的定義と実装の乖離、160%エラー率の主因
3. **データ型不整合**: 浮動小数点演算とNaN処理の不統一により、pandas計算結果との差分が蓄積

#### 解決策 (Resolution 10)

**【実装概要】**: 重要統計指標の数式修正と分母チェック強化による計算精度向上

**【変更対象】**: 
- `src/dssms/dssms_backtester.py`（主要変更）
- `output/dssms_unified_output_engine.py`（統計計算部分）
- `analysis/performance_metrics.py`（新規作成推奨）

**【実装内容】**:
```python
class DSSMSBacktester:
    def calculate_win_rate(self, trades_data):
        """勝率計算（分母欠如エラー修正）"""
        if not trades_data or len(trades_data) == 0:
            return 0.0  # TODO(tag:phase2, rationale:DSSMS Core focus): デフォルト値ポリシー検討
        
        winning_trades = len([t for t in trades_data if t.get('profit', 0) > 0])
        total_trades = len(trades_data)
        
        return (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
    
    def calculate_profit_factor(self, trades_data):
        """ProfitFactor計算（数式修正）"""
        if not trades_data:
            return 0.0
            
        gross_profit = sum(t.get('profit', 0) for t in trades_data if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t.get('profit', 0) for t in trades_data if t.get('profit', 0) < 0))
        
        # 分母チェック強化
        return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
    
    def calculate_sharpe_ratio(self, returns_series):
        """シャープレシオ計算（標準偏差0対応）"""
        import numpy as np
        
        if len(returns_series) < 2:
            return 0.0
            
        mean_return = np.mean(returns_series)
        std_return = np.std(returns_series, ddof=1)  # 修正: 不偏標準偏差使用
        
        return mean_return / std_return if std_return > 0 else 0.0
    
    def calculate_max_drawdown(self, portfolio_values):
        """最大ドローダウン計算（累積値ベース修正）"""
        if not portfolio_values:
            return 0.0
            
        cumulative_values = list(portfolio_values.values())
        peak = cumulative_values[0]
        max_dd = 0.0
        
        for value in cumulative_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
            
        return max_dd * 100  # パーセンテージ表示
```

**【期待効果】**: 
- 計算式エラー率160%→5%以下削減（KPI達成）
- 統計指標の数学的精度向上
- ZeroDivisionError・NaN発生の完全抑制

#### 検証計画

1. **数式単体テスト** - 各統計指標の境界値テスト（取引数0、負の値、NaN入力等）
2. **既知データ検証** - 手計算可能な小規模データセットでの計算結果一致確認
3. **エラー率測定** - 修正前後での計算式エラー検出率比較（目標：160%→5%以下）
4. **pandas整合性確認** - pandas標準関数との計算結果差分測定（許容誤差±0.01%）
5. **統合動作確認** - DSSMS全体実行での統計指標正常出力確認

#### 完了条件

- **定量的条件1**: 計算式エラー率5%以下達成（KPI基準値）
- **定量的条件2**: 主要統計指標（勝率・ProfitFactor・シャープレシオ・最大DD）の単体テスト100%成功
- **定量的条件3**: ZeroDivisionError・NaN例外の完全抑制（エラーログ0件）
- **定性的条件**: 85.0点エンジン出力品質維持、既存バックテスト結果との数学的整合性確保

#### 実装状況
- [✓] 解決策の検討・確定
- [ ] コード変更の実装
- [ ] テスト実行
- [ ] KPI評価
- [ ] 完了確認

---

### Problem 9: エンジン品質統一

#### 問題詳細
- **現象**: エンジン間の品質不均一・機能重複
- **重要度**: Medium Priority
- **効率値**: 30.0（改善効果75%/工数2.5）
- **カテゴリ**: B. アーキ/データフロー混乱
- **関連KPI**: 構造健全（重複エンジン整理率>90%）

#### 根本原因

1. **品質基準未統一**: 85.0点エンジン以外の品質評価・改善基準が未確立で、出力品質・コード品質・機能完成度にばらつき
2. **機能重複放置**: 同一機能を複数エンジンが異なる実装で提供し、品質差・保守性劣化・使用判断困難を招いている
3. **統合品質管理欠如**: DSSMS統合過程で段階的追加されたエンジンの品質チェック・標準化プロセスが未実装

#### 解決策 (Resolution 9)

**【実装概要】**: 85.0点エンジン標準による品質統一とエンジン機能統合

**【変更対象】**: 
- `output/` 配下の品質未達エンジン（特定後リスト化）
- `src/dssms/dssms_backtester.py`（エンジン選択ロジック）
- `docs/engine_quality_standards.md`（新規作成）
- `scripts/quality_assessment.py`（新規作成）

**【実装内容】**:
```python
"""エンジン品質評価・統一スクリプト"""

class EngineQualityManager:
    """エンジン品質評価・統一管理"""
    
    def __init__(self):
        self.quality_standards = {
            'output_accuracy': 85.0,    # 85.0点エンジン基準
            'code_quality': 80.0,       # 静的解析スコア
            'performance': 75.0,        # 処理速度基準
            'completeness': 90.0        # 機能完成度
        }
        self.reference_engine = 'dssms_unified_output_engine.py'
        
    def evaluate_engine_quality(self, engine_path):
        """エンジン品質評価実行"""
        # TODO(tag:phase2, rationale:DSSMS Core focus): 品質評価実装
        scores = {
            'output_accuracy': self._assess_output_accuracy(engine_path),
            'code_quality': self._assess_code_quality(engine_path),
            'performance': self._assess_performance(engine_path),
            'completeness': self._assess_completeness(engine_path)
        }
        return scores
        
    def standardize_engine(self, engine_path, target_standards):
        """85.0点エンジン基準による品質統一"""
        improvements = []
        
        # 出力精度改善
        if self._needs_output_improvement(engine_path):
            improvements.append(self._apply_output_standards(engine_path))
            
        # コード品質改善
        if self._needs_code_improvement(engine_path):
            improvements.append(self._apply_coding_standards(engine_path))
            
        return improvements
        
    def consolidate_duplicate_functions(self, engine_list):
        """重複機能統合"""
        function_map = self._analyze_function_overlap(engine_list)
        
        # 85.0点エンジンの実装を標準として採用
        standard_implementations = self._extract_standard_functions(
            self.reference_engine
        )
        
        return self._merge_functions(function_map, standard_implementations)

# filepath: src/dssms/dssms_backtester.py
class DSSMSBacktester:
    def __init__(self):
        # ...existing code...
        self.output_engine = self._select_optimal_engine()  # 統一エンジン選択
        
    def _select_optimal_engine(self):
        """品質基準による最適エンジン選択"""
        # 85.0点エンジンを標準採用
        return DSSMSUnifiedOutputEngine()  # 品質統一済みエンジン
```

**【期待効果】**: 
- エンジン品質の85.0点基準統一達成
- 重複機能整理による保守性向上（>90%整理率）
- DSSMS出力品質の一貫性確保

#### 検証計画

1. **品質ベンチマーク実行** - 全エンジンの品質評価（出力精度・コード品質・性能・完成度）
2. **85.0点基準適合テスト** - 改善対象エンジンの品質統一後の基準達成確認
3. **機能重複分析** - 重複機能の特定・統合効果測定（整理率>90%）
4. **統合動作検証** - 品質統一後のDSSMS全体実行による出力品質確認
5. **パフォーマンス影響評価** - 50銘柄ランキング処理時間<30s維持確認

#### 完了条件

- **定量的条件1**: 採用エンジンの品質評価全てが85.0点基準以上達成
- **定量的条件2**: 重複機能整理率>90%達成（機能統合による重複解消）
- **定量的条件3**: 品質統一後のDSSMS出力一貫性100%確認（複数実行での結果一致）
- **定性的条件**: 85.0点エンジン出力品質維持、DSSMS Core機能の非破壊保証、品質標準文書化完了

#### 実装状況
- [✓] 解決策の検討・確定
- [ ] コード変更の実装
- [ ] テスト実行
- [ ] KPI評価
- [ ] 完了確認

---

### Problem 11: ISM統合カバレッジ向上

#### 問題詳細
- **現象**: 切替判定ロジック分散・標準化不足
- **重要度**: Medium Priority
- **効率値**: 18.6（改善効果65%/工数3.5）
- **カテゴリ**: A. 切替メカニズム劣化
- **関連KPI**: 切替品質（不要切替率<20%）

#### 根本原因

1. **切替判定分散**: `DSSMSBacktester._evaluate_switch_decision()`とISMの切替ロジックが並行存在し、判定基準・閾値・タイミングが統一されていない
2. **統合カバレッジ不足**: ISM導入が段階的だったため、daily/weekly/emergency切替の一部がISM管理外に残存し、品質不均一を招いている
3. **標準化プロセス欠如**: 切替品質評価・改善のフィードバックループがISMに集約されておらず、不要切替率の体系的改善が困難

#### 解決策 (Resolution 11)

**【解決策概要】**: IntelligentSwitchManagerへの切替判定ロジック完全統合によって、分散した切替判定機能を一元化し、品質管理を強化する。すべての切替判定をISM経由で実行することで、統一基準での切替品質向上と不要切替率20%以下達成を目標とする。

**【変更対象】**: 
- `config/dssms/dssms_backtester_config.json`: ISM統合設定追加
- `IntelligentSwitchManager`: 切替判定ロジック統合インターフェース追加
- `DSSMSBacktester`: 分散切替判定ロジックのISM委譲
- `PortfolioDataManager`: 直接切替判定の廃止とISM経由実行
- 切替判定独立関数群: ISMサブモジュール化または統合

**【実装内容】**:

```python
class IntelligentSwitchManager:
    def __init__(self, config):
        self.switch_criteria = config['switch_criteria']
        self.quality_thresholds = config['quality_thresholds']
        self.unified_logic = UnifiedSwitchLogic()
        self.quality_tracker = SwitchQualityTracker()
        
    def evaluate_all_switches(self, portfolio_data, market_context):
        """全切替判定の統一エントリーポイント"""
        return self.unified_logic.process(portfolio_data, market_context)
        
    def get_switch_quality_metrics(self):
        """切替品質指標の統一取得"""
        return self.quality_tracker.get_metrics()
        
class UnifiedSwitchLogic:
    def process(self, portfolio_data, market_context):
        # 分散していた判定ロジックの統一実装
        criteria_results = self._evaluate_criteria(portfolio_data)
        quality_check = self._quality_assessment(criteria_results)
        return self._make_unified_decision(quality_check)
        
    def _evaluate_criteria(self, portfolio_data):
        """統一基準による切替判定評価"""
        # TODO(tag:phase1, rationale:DSSMS Core focus): DSSMSBacktester._evaluate_switch_decision統合
        return {
            'daily_criteria': self._daily_switch_check(portfolio_data),
            'weekly_criteria': self._weekly_switch_check(portfolio_data),
            'emergency_criteria': self._emergency_switch_check(portfolio_data)
        }
```

設定統合:
```json
{
    "intelligent_switch_manager": {
        "unified_switching": true,
        "integration_coverage": 100,
        "quality_target": {
            "unnecessary_switch_rate": 0.20,
            "consistency_rate": 0.95
        },
        "switch_consolidation": {
            "daily_ism_routing": true,
            "weekly_ism_routing": true,
            "emergency_ism_routing": true
        }
    }
}
```

**【期待効果】**: 
- 切替判定統合率: 40% → 100% (ISM経由完全統合)
- 不要切替率: 現状値 → <20% (品質管理強化)
- 切替判定一貫率: 向上率95%以上 (統一基準適用)
- 切替品質向上: 85.0点エンジン品質基準での品質保証体制確立
- 保守性向上: 切替ロジック保守箇所の一元化による開発効率改善

#### 検証計画

1. **統合前ベースライン取得** - 現在の切替パフォーマンス（切替数・不要切替率・判定一貫性）の詳細測定
2. **段階的統合テスト** - daily/weekly/emergency切替の順次ISM統合と動作確認
3. **切替品質評価** - 統合後の不要切替率測定と20%以下達成確認
4. **判定一貫性確認** - 同条件での切替判定結果の一貫性95%以上確認
5. **パフォーマンス影響評価** - ISM統合による処理時間・メモリ使用量への影響測定

#### 完了条件

- **定量的条件1**: 全切替判定100%ISM統合完了（直接切替判定の完全廃止）
- **定量的条件2**: 不要切替率≤20%達成（KPI基準値）
- **定量的条件3**: 切替判定一貫率95%以上（同条件での判定結果統一）
- **定性的条件**: 85.0点エンジン品質維持、SwitchEvent JSON互換性保証、切替品質改善サイクル文書化

#### 実装状況
- [✓] 解決策の検討・確定
- [ ] コード変更の実装
- [ ] テスト実行
- [ ] KPI評価
- [ ] 完了確認

---

### Problem 8: 実行ランタイム最適化

#### 問題詳細
- **現象**: 処理効率・リソース使用の非最適化
- **重要度**: Medium Priority
- **効率値**: 16.7（改善効果50%/工数3.0）
- **カテゴリ**: D. 統計/数式実装欠損
- **関連KPI**: 数式健全

#### 根本原因

1. **重複計算処理**: `portfolio_values`27箇所参照で同一データの重複アクセス・計算が発生し、処理効率が大幅に劣化している
   - DSSMSBacktester内で同一ポートフォリオデータへの重複アクセスが頻発
   - Perfect Order判定、スコア計算、統計処理で類似計算の重複実行

2. **非効率データアクセス**: Perfect Order判定・スコア計算で毎回データベース的アクセスが発生し、キャッシュ機能が未実装
   - 50銘柄ランキング生成時の反復的データ読み込み
   - MA値計算の重複実行（timeframe別で同一基礎データから再計算）

3. **メモリ使用非最適**: 50銘柄ランキング処理時の一時データ保持・ガベージコレクション効率が悪く、メモリ圧迫によるパフォーマンス劣化
   - 大量の中間データオブジェクトが適切に解放されない
   - データ構造の効率化不足によるメモリ断片化

#### 解決策 (Resolution 8)

**【実装概要】**: データアクセス効率化とキャッシュ機能による実行ランタイム最適化

**【変更対象】**: 
- `src/dssms/dssms_backtester.py`（主要変更）
- `src/dssms/performance_optimizer.py`（新規作成）
- `output/dssms_unified_output_engine.py`（最適化適用）
- `config/dssms/performance_config.json`（新規作成）

**【実装内容】**:
```python
"""DSSMS実行ランタイム最適化モジュール"""

class PerformanceOptimizer:
    """実行時パフォーマンス最適化管理"""
    
    def __init__(self, config):
        self.cache_config = config.get('cache', {})
        self.data_cache = {}
        self.calculation_cache = {}
        self.access_metrics = {}
        
    def optimize_portfolio_access(self, backtester_instance):
        """portfolio_values アクセス最適化"""
        # TODO(tag:phase2, rationale:DSSMS Core focus): キャッシュ戦略実装
        original_portfolio_values = backtester_instance.portfolio_values
        
        # キャッシュレイヤー追加
        cached_portfolio = CachedPortfolioManager(original_portfolio_values)
        backtester_instance.portfolio_values = cached_portfolio
        
        return cached_portfolio
        
    def optimize_ranking_calculation(self, ranking_data):
        """50銘柄ランキング計算最適化"""
        cache_key = self._generate_cache_key(ranking_data)
        
        if cache_key in self.calculation_cache:
            return self.calculation_cache[cache_key]
            
        # 計算量削減アルゴリズム適用
        optimized_result = self._efficient_ranking_algorithm(ranking_data)
        
        # キャッシュ保存（設定に基づく）
        if self.cache_config.get('enable_ranking_cache', True):
            self.calculation_cache[cache_key] = optimized_result
            
        return optimized_result

class CachedPortfolioManager:
    """portfolio_values アクセス最適化"""
    
    def __init__(self, original_portfolio):
        self._original = original_portfolio
        self._cache = {}
        self._access_count = {}
        
    def __getitem__(self, key):
        """キャッシュ機能付きアクセス"""
        if key not in self._cache:
            self._cache[key] = self._original[key]
            
        self._access_count[key] = self._access_count.get(key, 0) + 1
        return self._cache[key]
        
    def get_access_metrics(self):
        """アクセス統計取得（パフォーマンス監視用）"""
        return {
            'total_accesses': sum(self._access_count.values()),
            'unique_keys': len(self._access_count),
            'cache_hit_rate': len(self._cache) / max(1, sum(self._access_count.values()))
        }

# filepath: src/dssms/dssms_backtester.py
class DSSMSBacktester:
    def __init__(self):
        # ...existing code...
        from src.dssms.performance_optimizer import PerformanceOptimizer
        self.performance_optimizer = PerformanceOptimizer(self.config)
        self.portfolio_values = self.performance_optimizer.optimize_portfolio_access(self)
        
    def generate_ranking(self, symbols_data):
        """50銘柄ランキング生成（最適化版）"""
        # パフォーマンス測定開始
        start_time = time.time()
        
        # 最適化されたランキング計算
        ranking_result = self.performance_optimizer.optimize_ranking_calculation(symbols_data)
        
        # パフォーマンス監視
        execution_time = time.time() - start_time
        if execution_time > 30.0:  # KPI基準
            logger.warning(f"Ranking calculation exceeded 30s: {execution_time:.2f}s")
            
        return ranking_result
        
    def _efficient_perfect_order_calculation(self, symbol_data):
        """Perfect Order判定の計算効率化"""
        # 重複計算削減
        ma_values = self._calculate_ma_values_once(symbol_data)  # 1回計算・再利用
        
        return {
            'timeframe_5m': self._check_perfect_order_cached(ma_values['5m']),
            'timeframe_15m': self._check_perfect_order_cached(ma_values['15m']),
            'timeframe_1h': self._check_perfect_order_cached(ma_values['1h']),
            'timeframe_4h': self._check_perfect_order_cached(ma_values['4h'])
        }
```

設定統合:
```json
{
    "cache": {
        "enable_ranking_cache": true,
        "enable_portfolio_cache": true,
        "cache_size_limit": 1000,
        "cache_ttl_seconds": 3600
    },
    "performance_monitoring": {
        "enable_timing": true,
        "warning_threshold_seconds": 30,
        "log_slow_operations": true
    },
    "memory_optimization": {
        "enable_gc_optimization": true,
        "large_data_threshold_mb": 100
    }
}
```

**【期待効果】**: 
- 50銘柄ランキング処理時間<30s達成（KPI基準）
- portfolio_values重複アクセス削減（27箇所→効率化）
- メモリ使用量最適化による安定性向上

#### 検証計画

1. **ベースライン測定** - 最適化前の処理時間・メモリ使用量・CPU利用率の詳細測定
2. **段階的最適化テスト** - キャッシュ→計算効率化→メモリ最適化の順で効果測定
3. **50銘柄ランキング性能テスト** - 複数回実行での処理時間安定性確認（目標<30s）
4. **メモリプロファイリング** - 最適化前後でのメモリ使用量比較・リーク検出
5. **統合動作確認** - 最適化によるDSSMS全体機能への影響確認（出力品質維持）

#### 完了条件

- **定量的条件1**: 50銘柄ランキング処理時間≤30秒達成（KPI基準値）
- **定量的条件2**: portfolio_valuesアクセス効率化による処理時間20%以上短縮
- **定量的条件3**: メモリ使用量15%以上削減（ガベージコレクション効率化）
- **定性的条件**: 85.0点エンジン出力品質維持、既存機能の非破壊保証、パフォーマンス監視機能統合

#### 実装状況
- [✓] 解決策の検討・確定
- [ ] コード変更の実装
- [ ] テスト実行
- [ ] KPI評価
- [ ] 完了確認

---

### Problem 18: ファイル管理最適化

#### 問題詳細
- **現象**: ファイル構造の肥大化・整理不足
- **重要度**: Low Priority
- **効率値**: 20.0（改善効果20%/工数1.0）
- **カテゴリ**: G. 管理/品質標準
- **関連KPI**: 構造健全

#### 根本原因

1. **ファイルライフサイクル管理不在**: DSSMS統合過程で一時ファイル・実験ファイル・バックアップファイルの整理基準が未確立で、不要ファイルが蓄積
   - 開発段階で作成された検証用ファイル・テストファイルが削除されずに残存
   - backup_*、analyze_*、check_*等の命名パターンファイルが大量に散在

2. **重複ファイル放置**: 同一機能の異なるバージョンファイルが並存し、active/deprecated/archiveの区別が曖昧で管理負荷増大
   - 類似名称のファイルが複数存在（例：dssms_excel_exporter系の複数バージョン）
   - 機能統合後の旧ファイルがアーカイブされずに放置

3. **定期クリーンアップ不在**: プロジェクト成長に伴う自然なファイル肥大化に対する定期的な整理手順が未実装で、継続的な構造健全性確保が困難
   - ログファイル・一時ファイル・キャッシュファイルの自動削除機能不在
   - ファイル管理ポリシーの未文書化による整理基準の不明確性

#### 解決策 (Resolution 18)

**【実装概要】**: 段階的ファイル整理とライフサイクル管理による構造健全化

**【変更対象】**: 
- プロジェクトルート全体（特に`output`、`archive`、`logs`）
- `scripts/file_cleanup.py`（新規作成）
- `.gitignore`（更新）
- `docs/file_management_policy.md`（新規作成）

**【実装内容】**:
```python
"""プロジェクトファイル整理・管理スクリプト"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

class FileCleanupManager:
    """ファイル整理・ライフサイクル管理"""
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.cleanup_policy = {
            'temp_files': {'pattern': '*.tmp', 'retention_days': 7},
            'log_files': {'pattern': '*.log', 'retention_days': 30},
            'backup_files': {'pattern': '*_backup_*', 'retention_days': 90},
            'cache_files': {'pattern': '__pycache__/*', 'retention_days': 0}
        }
        
    def identify_cleanup_candidates(self):
        """整理対象ファイル特定"""
        candidates = {
            'safe_delete': [],      # 安全削除可能
            'review_required': [],  # 要確認
            'archive_target': []    # アーカイブ対象
        }
        
        # TODO(tag:phase2, rationale:DSSMS Core focus): 詳細分析実装
        # 明らかな不要ファイル特定
        candidates['safe_delete'].extend(self._find_temp_files())
        candidates['safe_delete'].extend(self._find_cache_files())
        
        # 要確認ファイル特定
        candidates['review_required'].extend(self._find_duplicate_files())
        candidates['review_required'].extend(self._find_large_files())
        
        return candidates
        
    def execute_safe_cleanup(self, candidates):
        """安全なファイル削除実行"""
        deleted_files = []
        backup_location = self.project_root / 'archive' / 'deleted_files' / datetime.now().strftime('%Y%m%d')
        backup_location.mkdir(parents=True, exist_ok=True)
        
        for file_path in candidates['safe_delete']:
            if self._is_safe_to_delete(file_path):
                # バックアップ作成
                shutil.copy2(file_path, backup_location)
                # 削除実行
                os.remove(file_path)
                deleted_files.append(file_path)
                
        return deleted_files
        
    def _find_temp_files(self):
        """一時ファイル検出"""
        temp_patterns = ['*.tmp', '*.temp', '*~', '.DS_Store']
        temp_files = []
        
        for pattern in temp_patterns:
            temp_files.extend(self.project_root.rglob(pattern))
            
        return temp_files
        
    def _find_cache_files(self):
        """キャッシュファイル検出"""
        cache_dirs = ['__pycache__', '.pytest_cache', '.coverage']
        cache_files = []
        
        for cache_dir in cache_dirs:
            cache_paths = list(self.project_root.rglob(cache_dir))
            for cache_path in cache_paths:
                if cache_path.is_dir():
                    cache_files.extend(cache_path.rglob('*'))
                    
        return cache_files
        
    def generate_cleanup_report(self):
        """整理レポート生成"""
        report = {
            'total_files': len(list(self.project_root.rglob('*'))),
            'cleanup_candidates': self.identify_cleanup_candidates(),
            'disk_usage': self._calculate_disk_usage(),
            'recommendations': self._generate_recommendations()
        }
        return report
```

ファイル管理ポリシー:
```markdown
# ファイル管理ポリシー

## ファイル分類
- **Core files**: DSSMS Core機能に必須
- **Archive files**: 履歴保存価値あり
- **Temp files**: 7日後自動削除対象
- **Cache files**: 即座削除可能

## 定期クリーンアップ
- 週次: 一時ファイル削除
- 月次: ログファイル整理
- 四半期: 重複ファイル確認
```

**【期待効果】**: 
- プロジェクトファイル数20%削減による構造健全性向上
- 不要ファイル削除による管理効率改善
- ファイルライフサイクル管理体制確立

#### 検証計画

1. **ファイル数ベースライン取得** - 整理前のプロジェクト全体ファイル数・サイズ・構成の詳細測定
2. **安全削除確認** - 削除対象ファイルの機能影響確認（DSSMS動作テスト実行）
3. **整理効果測定** - 削除後のファイル数・ディスク使用量削減効果測定
4. **バックアップ整合性確認** - 削除ファイルのバックアップ完全性確認
5. **DSSMS動作確認** - 整理後の85.0点エンジン動作・出力品質維持確認

#### 完了条件

- **定量的条件1**: プロジェクトファイル数20%以上削減達成
- **定量的条件2**: 明らかな不要ファイル（*.tmp、__pycache__等）100%削除
- **定量的条件3**: ディスク使用量15%以上削減（不要ファイル除去効果）
- **定性的条件**: 85.0点エンジン出力品質維持、DSSMS Core機能の非破壊保証、ファイル管理ポリシー文書化完了

#### 実装状況
- [✓] 解決策の検討・確定
- [ ] コード変更の実装
- [ ] テスト実行
- [ ] KPI評価
- [ ] 完了確認

---

### Problem 戦略統計: シート品質改善

#### 問題詳細
- **現象**: 戦略統計シートの軽微な品質問題
- **重要度**: Low Priority
- **効率値**: 15.0（改善効果15%/工数1.0）
- **カテゴリ**: E. 出力一貫性
- **関連KPI**: 出力整合

#### 根本原因

1. **統計項目欠損**: 戦略別の勝率・ProfitFactor・平均損益等の基本統計項目が一部未実装または計算精度不足
   - 戦略統計シートで表示される統計項目に欠損・不完全な計算結果が含まれる
   - Problem 10で修正された計算式が戦略統計シートに未反映

2. **フォーマット不統一**: 戦略統計シートの数値フォーマット・日付表示・単位表記が他シートと不整合
   - 勝率表示、金額表示、パーセンテージ表示の形式が統一されていない
   - 日付・時刻フォーマットが他の出力シートと異なる表記を使用

3. **網羅性不足**: DSSMS統合過程で戦略統計出力の優先度が低く、完全性検証が後回しになり軽微な品質問題が残存
   - 85.0点エンジン使用確認後の最終仕上げ対象として位置付けられた
   - 戦略統計シートの品質基準が他の出力シートより低い状態で放置

#### 解決策 (Resolution 戦略統計)

**【実装概要】**: 戦略統計シートの統計項目補完とフォーマット統一による出力完全性向上

**【変更対象】**: 
- `output/dssms_unified_output_engine.py`（戦略統計部分）
- `src/dssms/strategy_statistics_calculator.py`（新規作成推奨）
- 戦略統計シート出力テンプレート

**【実装内容】**:
```python
class DSSMSUnifiedOutputEngine:
    def generate_strategy_statistics_sheet(self, strategy_results):
        """戦略統計シート生成（品質改善版）"""
        # TODO(tag:phase2, rationale:DSSMS Core focus): 統計項目補完実装
        
        strategy_stats = {}
        for strategy_name, results in strategy_results.items():
            strategy_stats[strategy_name] = self._calculate_complete_statistics(results)
            
        # フォーマット統一適用
        formatted_stats = self._apply_unified_formatting(strategy_stats)
        
        return self._create_strategy_statistics_sheet(formatted_stats)
        
    def _calculate_complete_statistics(self, strategy_results):
        """完全な戦略統計計算"""
        if not strategy_results or len(strategy_results) == 0:
            return self._get_default_statistics()
            
        statistics = {
            # 基本統計（必須項目）
            'total_trades': len(strategy_results),
            'win_rate': self._calculate_win_rate(strategy_results),
            'profit_factor': self._calculate_profit_factor(strategy_results),
            'average_profit': self._calculate_average_profit(strategy_results),
            'max_drawdown': self._calculate_max_drawdown(strategy_results),
            
            # 追加統計（品質向上項目）
            'sharpe_ratio': self._calculate_sharpe_ratio(strategy_results),
            'total_return': self._calculate_total_return(strategy_results),
            'average_holding_period': self._calculate_average_holding_period(strategy_results),
            
            # メタデータ
            'calculation_timestamp': datetime.now().isoformat(),
            'data_completeness': self._assess_data_completeness(strategy_results)
        }
        
        return statistics
        
    def _apply_unified_formatting(self, strategy_stats):
        """統一フォーマット適用"""
        formatted = {}
        
        for strategy_name, stats in strategy_stats.items():
            formatted[strategy_name] = {
                # 数値フォーマット統一
                '勝率(%)': f"{stats['win_rate']:.2f}%",
                'プロフィットファクター': f"{stats['profit_factor']:.3f}",
                '平均損益(円)': f"¥{stats['average_profit']:,.0f}",
                '最大ドローダウン(%)': f"{stats['max_drawdown']:.2f}%",
                'シャープレシオ': f"{stats['sharpe_ratio']:.3f}",
                '総取引数': f"{stats['total_trades']:,}回",
                '総リターン(%)': f"{stats['total_return']:.2f}%",
                '平均保有期間(時間)': f"{stats['average_holding_period']:.1f}h"
            }
            
        return formatted
        
    def _get_default_statistics(self):
        """デフォルト統計値（データ不足時）"""
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_return': 0.0,
            'average_holding_period': 0.0,
            'calculation_timestamp': datetime.now().isoformat(),
            'data_completeness': 0.0
        }
```

専用統計計算モジュール:
```python
# filepath: src/dssms/strategy_statistics_calculator.py
"""戦略統計計算専用モジュール（新規作成）"""

class StrategyStatisticsCalculator:
    """戦略別統計計算の専門クラス"""
    
    def __init__(self):
        self.calculation_methods = {
            'win_rate': self._win_rate_calculation,
            'profit_factor': self._profit_factor_calculation,
            'sharpe_ratio': self._sharpe_ratio_calculation
        }
        
    def _win_rate_calculation(self, trades):
        """勝率計算（Problem 10対応済み）"""
        if not trades:
            return 0.0
        winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
        return (winning_trades / len(trades)) * 100
        
    def _profit_factor_calculation(self, trades):
        """プロフィットファクター計算（Problem 10対応済み）"""
        if not trades:
            return 0.0
        gross_profit = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
        gross_loss = abs(sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
```

**【期待効果】**: 
- 戦略統計シートの完全性100%達成（全統計項目実装）
- 出力フォーマット統一による出力整合性向上
- 85.0点エンジン品質基準での戦略統計品質保証

#### 検証計画

1. **統計項目網羅性確認** - 全戦略の基本統計項目（8項目）の計算・出力確認
2. **フォーマット整合性テスト** - 他シートとの数値フォーマット・単位表記統一確認
3. **計算精度検証** - Problem 10対応済み計算式での統計値精度確認
4. **データ欠損対応テスト** - 戦略データ不足時のデフォルト値表示確認
5. **85.0点エンジン品質維持確認** - 変更後の全体出力品質維持確認

#### 完了条件

- **定量的条件1**: 戦略統計シートの全統計項目（8項目）100%実装・出力確認
- **定量的条件2**: 出力フォーマット統一率100%（他シートとの整合性確保）
- **定量的条件3**: 統計計算精度100%確認（Problem 10対応済み計算式使用）
- **定性的条件**: 85.0点エンジン出力品質維持、DSSMS Core機能の非破壊保証、戦略統計シート完全性達成

#### 実装状況
- [✓] 解決策の検討・確定
- [ ] コード変更の実装
- [ ] テスト実行
- [ ] KPI評価
- [ ] 完了確認

## 6. 重要成果サマリー

### 6.1 **最高品質エンジン(85.0点)使用確認完了**
**Phase 1調査結果**: Problem 17の記述に誤認があり、実際は85.0点の高品質エンジンが使用されていることを確認。これによりDSSMS出力品質の根本的な懸念が解消され、**切替メカニズム復旧に集中**できる状況が整いました。

**決定的証拠**: `src/dssms/dssms_backtester.py`が`dssms_unified_output_engine.py` (Task 4.2で85.0点) を正しく使用中

---
**最終更新**: 2025年9月18日  
**状態**: 切替メカニズム復旧への集中戦略確定、問題別解決策フレームワーク追加完了次で最適銘柄選択・リスク調整後利益獲得）に沿っているか
2. **方法の適切性**: 現状の方法で十分か、より効果的な手法があるか
3. **整合性チェック**: 矛盾・新たなエラー要因がないか
4. **根本解決**: 緊急対応ではなく根本原因に対処しているか
5. **明確性**: 解決策が具体的かつ実装可能な形で明示されているか
6. **検証計画**: 実装後の動作確認手順が含まれているか
7. **コード管理**: 新旧ファイルの管理方法が適切か（コメントアウト、リファクタリング等）

### 3.2 調整アプローチ

1. **アプローチ**: 
   - 根本解決を行うが修正の工数が多い場合は段階的に修正していく
   - 特にDSSMSの切替メカニズム劣化は根本修正をおこなうが段階的に修正を行う

2. **リファクタリングのバランス**:
   - 「似たようなファイルを作らない」という原則は基本としつつ
   - 必要に応じて一時的な並行運用を許容（特に検証フェーズ）
   - 古いファイル削除前の移行期間を設定し、安全性を確保

### 3.3 追加評価基準

1. **KPI連動性**: 解決策が「0.3 成功判定KPI」に直接貢献するか明確化
2. **優先度整合**: 「0.5 優先順位ルール」に従った解決順序になっているか
3. **完了定義**: 「0.7 完了判定条件」に基づいた検証方法が含まれているか
4. **リソース効率**: 最小工数で最大効果が得られる順序になっているか

### 3.4 実装・検証フロー

各修正後には以下の検証ステップを組み込む：

1. **コンパイルチェック**: 基本的な構文エラーがないか確認
2. **ユニットテスト**: 修正した機能の個別テスト実行
3. **統合テスト**: python "src\dssms\dssms_backtester.py"での全体実行
4. **比較テスト**: 修正前後の出力差分を計測（期待通りの変化か）
5. **KPI評価**: 定義済みKPIの達成度確認

### 3.5 実装ベストプラクティス

1. **段階的実装**: 
   - 一度に全てを修正せず、安全な単位で区切って実装
   - 各ステップで動作確認を行う

2. **安全対策**:
   - 修正前に必ず作業コピーを保存
   - 重要な設定ファイルはバックアップを取得
   - 変更のロールバック計画を事前に策定

3. **ドキュメント化**:
   - 各変更とその意図を変更ログとして文書化
   - 重要な決定事項と理由を記録

4. **再現性管理**:
   - 同一設定での複数回実行で結果一致を確認
   - seed値固定テストケースの作成と保存

## 4. 重要成果サマリー

### 4.1 **最高品質エンジン(85.0点)使用確認完了**
**Phase 1調査結果**: Problem 17の記述に誤認があり、実際は85.0点の高品質エンジンが使用されていることを確認。これによりDSSMS出力品質の根本的な懸念が解消され、**切替メカニズム復旧に集中**できる状況が整いました。

**決定的証拠**: `src/dssms/dssms_backtester.py`が`dssms_unified_output_engine.py` (Task 4.2で85.0点) を正しく使用中

---
**最終更新**: 2025年9月18日  
**状態**: 切替メカニズム復旧への集中戦略確定、解決策評価フレームワーク追加完了再定義）

### 0.1 Ultimate Business Goal
日次で日経225から最適上昇トレンド銘柄を動的選択し、単一集中＋マルチ戦略適用 + kabu発注でリスク調整後利益を継続獲得。

### 0.2 Purpose Hierarchy
1. Business: 動的最適銘柄集中運用で資本効率最大化  
2. System Core: 50銘柄ランキング → 最適 + バックアップ抽出 → 切替判定 → 学習フィードバック  
3. Technical: 処理<30s / 再現性 / 切替根拠メタデータ / 出力一意性  
4. Operational: 日次成功率>99% / 不要切替率<20% / Excel一致率100%  
5. Improvement Loop: SwitchEvent → 10営業日検証 → 不要切替率 → スコア/閾値調整

### 0.3 成功判定 KPI（計測仕様）
| カテゴリ | KPI | 定義 | 現状課題との関係 |
|----------|-----|------|------------------|
| 切替性能 | 適正切替頻度 | 期間内想定レンジ（過去再現値117回基準で乖離分析） | Problem 1,3,12 |
| 切替品質 | 不要切替率<20% | 10営業日後 (p_after - p_before - cost) ≤ 0 | Problem 7,11 |
| トレンド精度 | Perfect Order 判定一致率 99% | 手動サンプル比較 | Problem 4,6,17 |
| 出力整合 | Excel再計算一致率 100% | 内部再計算 vs 出力 | Problem 2,5,8,10 |
| 数式健全 | 計算式エラー率<5% | 分母欠如/誤式検出 | Problem 8,9,10 |
| 構造健全 | 重複エンジン整理率>90% | 存在→採用/アーカイブ分類 | Problem 13,17,18 |
| 再現性 | 決定論差分=ゼロ | seed/設定同一時の出力差分 | Problem 3,12 |

### 0.4 問題領域カテゴリ（分類基準）
| カテゴリ | 説明 | 含まれる既存Problem |
|----------|------|----------------------|
| A. 切替メカニズム劣化 | 頻度/判定ロジック/保有期間 | 1,3,11,12 |
| B. アーキ/データフロー混乱 | エンジン多重・日付/portfolio流 | 4,6,13,17,18 |
| C. 保有期間/イベント整合性 | 実測 vs 固定24h 乖離 | 7 |
| D. 統計/数式実装欠損 | 勝率/ProfitFactor等未実装 & 誤式 | 8,9,10 |
| E. 出力一貫性 | Excel日付ループ/欠損/戦略統計不足 | 2,5,8,10 |
| F. 再現性設定 | 過度決定論・ノイズ抑止 | 3,12 |
| G. 管理/品質標準 | バージョン/ファイル肥大・未統合 | 17,18 |
| H. 低影響補助 | 軽微データ品質 | 14 |

### 0.5 優先順位ルール
1. Core Blocking (A) → 出力より先に切替機能回復  
2. Integrity Before Analytics: 正しい計算 > 表示/書式  
3. Single Source: 出力エンジンの単一化後に最適化  
4. Determinism Balance: 過度固定化設定は段階解除  
5. Measurable Closure: すべての修復は KPI で完了条件明示  

### 0.6 解決順序（再確認）
1. A: 切替復旧（1,3,12）  
2. D: 数式健全化（8,9,10）  
3. B: エンジン統合/フロー正常化（4,6,13,17,18）  
4. C/E: 保有期間 & 出力最終整合（7,2,5,戦略統計）  
5. F: 再現性微調整（残差）  
6. G/H: 管理最適化 & 軽微改善  

### 0.7 完了判定条件（Definition of Done）
| 項目 | DoD |
|------|-----|
| 切替機能 | 任意テスト期間で目標レンジ回復 + 再現性差分0 |
| 数式 | 全指標ユニットテスト緑 (勝率/ProfitFactor/平均損益/総取引/最大DD) |
| 出力 | 任意3期間で内部計算 vs Excel 差分=0 |
| エンジン統合 | 採用=1 / アーカイブ=明示 / 重複=0 |
| 保有期間 | 全SwitchEventで実測時間>0 且つ 24h固定値消滅 |
| 再現性 | seed固定で2連続実行ハッシュ一致 |
| 管理 | 不要ファイル/空v3削除、READMEと整合 |

---

## 1. 問題サマリー（簡潔版）

**過去の問題分析は `docs/dssms/Problem_details_archive.md` に移動しました、一部最新情報と齟齬あり注意**

### 🎯 Critical Priority（即座解決が必要）
- **Problem 1**: 切替判定ロジック劣化（117回→3回の激減）
- **Problem 12/3統合**: 決定論的モード設定問題（同一根本原因）

### ✅ Resolved（解決済み）
- **Problem 15**: エンジン品質評価一致確認完了
- **Problem 17**: エンジン使用状況誤認訂正（85.0点エンジン使用確認済み）

### 🔧 High Priority（システム効率化）
- **Problem 6**: データフロー/ポートフォリオ処理混乱
  - DSSMSBacktester.portfolio_values: 27箇所で参照・操作される複雑なデータフロー
  - エンジン間変換ロジック不統一（v1,v2,v4で実装差異）
  - 日付修正機能分散（v4で8箇所のpd.to_datetime使用）
  - 日付ループ問題（2023-12-31 → 2023-01-01）
- **Problem 13**: エンジン競合解決（103個エンジン重複）
- **Problem 10**: 数学的エラー修正（計算式エラー率160%）
- **Problem 9**: エンジン品質統一
- **Problem 11**: ISM統合カバレッジ向上

### 📁 Medium Priority（管理改善）
- **Problem 18**: ファイル管理最適化（実害なし）
- **Problem 14**: データ品質改善（7.3%軽微影響）
- **戦略統計シート**: 軽微問題修正

---

## 2. 現在の実行戦略

### 🔄 Task 6 Phase C: 修復優先順位戦略（最終目的達成のためのコア優先）

#### Phase 1: コア機能復旧（切替メカニズム）【Critical Priority】
**所要時間**: 2-4時間
**対象問題**: Problem 1, Problem 3
**目的**: DSSMS主機能の切替メカニズム復旧（117回→3回の激減問題解決）
**理論的根拠**: システム中核機能回復、最終目的（正常動作）に直結

#### Phase 2: データフロー最適化（出力品質向上）【Medium Priority】
**所要時間**: 3-5時間
**対象問題**: Problem 6, Problem 2
**目的**: 正しい計算結果の正しい出力保証（残存問題解決）
**理論的根拠**: 最終目的（正常出力）の完全達成

#### Phase 3: インフラ整理（管理改善）【Low Priority】
**所要時間**: 1-2時間
**対象問題**: Problem 18, Problem 17
**目的**: 管理・維持性向上（品質への実害なし）
**理論的根拠**: 長期維持管理効率化

#### Phase 4: 詳細調整（完全性向上）【Low Priority】
**所要時間**: 2-3時間
**対象問題**: Problem 8-14残存分（戦略統計等）
**目的**: 全出力要素の完全性確保
**理論的根拠**: 最終品質到達

### 📋 戦略原則

1. **コア優先**: 切替メカニズム（DSSMS主機能）を最優先で修復
2. **修正重複回避**: 機能復旧完了後に最適化・管理改善を実施
3. **段階的検証**: 各Phase完了時に動作確認を実施
4. **最小影響**: 既存の正常動作部分（90/100品質）への影響を最小化

**根拠**: Task 6.3で85.0点エンジンが実際に高品質出力（90/100）を生成しているため、エンジン問題よりも切替メカニズム劣化が主要課題

### � Task 6依存関係分析による科学的効率分析表更新版

**分析基準**: Task 6.3結果による状況変化を反映した再評価

| Problem | 改善効果 | 実装コスト | 効率値 | 優先順位 | 根拠 |
|---------|----------|------------|--------|----------|------|
| **Problem 1** | **95%** | **3.0工数** | **31.7** | **🚨切替機能最優先** | **切替数117→3激減、DSSMS中核機能** |
| **Problem 12/3（統合）** | **85%** | **0.5工数** | **170.0** | **🥇効率1位** | **決定論的モード統一問題、同一根本原因** |
| **Problem 6** | **60%** | **2.5工数** | **24.0** | **🥈2位** | **データフロー混乱、27箇所portfolio_values** |
| **Problem 13** | **52%** | **4.0工数** | **13.0** | **🥉3位** | **103エンジン重複、Critical影響** |
| Problem 10 | 90% | 2.0工数 | 45.0 | 4位 | 数学的エラー160%削減 |
| Problem 9 | 75% | 2.5工数 | 30.0 | 5位 | 85点品質統一 |
| Problem 11 | 65% | 3.5工数 | 18.6 | 6位 | ISM統合カバレッジ向上 |
| Problem 8 | 50% | 3.0工数 | 16.7 | 7位 | 実行ランタイム最適化 |
| **Problem 17** | **解決済み** | **0工数** | **N/A** | **✅完了** | **Phase 1調査で誤認確認** |
| **Problem 18** | **20%** | **1.0工数** | **20.0** | **8位** | **ファイル整理、実害なし** |
| **Problem 戦略統計** | **15%** | **1.0工数** | **15.0** | **9位** | **Task 6.3発見、軽微問題** |
| **Problem 15** | **解決済み** | **0工数** | **N/A** | **✅完了** | **Phase 1調査で品質一致確認** |
| ~~Problem 16~~ | ~~60%~~ | ~~1.5工数~~ | ~~40.0~~ | **❌無効化** | **評価システム正常確認** |

**🚨 Problem 12/3統合による修正実行計画**:

#### 最優先Phase: 切替メカニズム復旧（同一問題統合処理）
1. **Problem 1解決**: 切替判定ロジック復旧（即座実行）
   - 24時間保有期間制約緩和
   - 決定論的モード過度設定修正
   - **期待効果**: 切替数3回→117回レベル回復
   - **実行時間**: 1時間

2. **Problem 12/3統合解決**: 決定論的モード統一最適化
   - `enable_score_noise`, `enable_switching_probability`調整
   - 同一根本原因による共通対応
   - **期待効果**: 切替判定柔軟性回復
   - **実行時間**: 30分
   - **評価基準**: 問題明確化後に改善策検討（先送り）

### 🔧 Phase A実装詳細（重要要素抜粋）:

#### 決定論的モード最適化（Resolution 2.9）
**設定変更対象**:
```json
{
  "randomness_control": {
    "switching": {
      "enable_probabilistic": true,        // False → True
      "switching_probability": 0.9
    },
    "scoring": {
      "enable_noise": true,               // False → True  
      "noise_factor": 0.05
    }
  },
  "performance_calculation": {
    "use_fixed_execution_price": false    // True → False
  }
}
```
**対象ファイル**: `config/dssms/dssms_backtester_config.json`

#### ISM統合標準化
**統合パターン**:
- `_evaluate_switch_decision()` → `self.switch_manager.should_switch()`
- 主要バックテスター（v1,v2,v2_updated）での必須統合
- 統一APIインターフェース確立

#### データフロー統一化
- **portfolio_values処理**: 27箇所参照の整理・統合
- **変換ロジック統一**: v1,v2,v4の実装差異解消
- **日付処理最適化**: v4の8箇所pd.to_datetime → 必要最小限（2-3箇所）に削減

#### 第2Phase: 効率最適化
3. **Problem 12/3統合解決**: 設定最適化（効率170.0、同一問題統合処理）
4. **Problem 13解決**: エンジン競合解決

#### 第3優先: 管理改善（低優先度）
5. **Problem 17-18解決**: ファイル管理最適化
6. **戦略統計シート**: 軽微問題修正

---

## � 重要成果サマリー

### � **最高品質エンジン(85.0点)使用確認完了**
**Phase 1調査結果**: Problem 17の記述に誤認があり、実際は85.0点の高品質エンジンが使用されていることを確認。これによりDSSMS出力品質の根本的な懸念が解消され、**切替メカニズム復旧に集中**できる状況が整いました。

**決定的証拠**: `src/dssms/dssms_backtester.py`が`dssms_unified_output_engine.py` (Task 4.2で85.0点) を正しく使用中

---
**最終更新**: 2025年9月18日  
**状態**: 切替メカニズム復旧への集中戦略確定、220行以下重複内容アーカイブ移動完了

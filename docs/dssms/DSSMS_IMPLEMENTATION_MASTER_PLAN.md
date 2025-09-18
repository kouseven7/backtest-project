# Dynamic Stock Selection Multi-Strategy System (DSSMS)
## 動的株式選択マルチ戦略システム - 実装マスタープラン

**プロジェクト**: my_backtest_project  
**作成日**: 2025年8月16日  
**更新日**: 2025年8月16日  
**バージョン**: 1.0  

---

## 🎯 **システム概要**

### **DSSMS（Dynamic Stock Selection Multi-Strategy System）とは**
- **目的**: 複数銘柄から上昇トレンド銘柄を動的に選択し、マルチ戦略システムで売買を実行
- **コンセプト**: 単一銘柄での戦略実行 → 最適銘柄選択 + マルチ戦略実行
- **エッジ**: バックテストでパーフェクトオーダー上昇トレンド銘柄は必ず+の実績

### **システム特徴**
1. **動的銘柄選択**: 50銘柄から最適銘柄を自動選択
2. **階層的優先判定**: パーフェクトオーダー全時間軸 → 月週軸 → その他
3. **統合マルチ戦略**: 既存戦略システムを選択銘柄に適用
4. **リスク管理統合**: 資金管理・ドローダウン対応・市場停止判定

---

## 🔍 **技術要件定義**

### **パーフェクトオーダー判定基準**
```python
# 判定ロジック定義
timeframes = {
    "daily": {"short": 5, "medium": 25, "long": 75},      # SBI証券準拠
    "weekly": {"short": 13, "medium": 26, "long": 52},    # SBI証券準拠  
    "monthly": {"short": 9, "medium": 24, "long": 60}     # SBI証券準拠
}

# 判定条件: SMA_short > SMA_medium > SMA_long かつ 現在価格 > SMA_short
```

### **銘柄選定フィルタ**
```python
# 前提条件フィルタ
screening_filters = {
    "universe": "nikkei225_constituents",
    "min_price": 500,                    # 円
    "min_market_cap": 10_000_000_000,    # 100億円（仕手株除外）
    "min_shares_affordable": 100,        # 100株購入可能
    "drawdown_adjustment": True          # ドローダウン時資金調整
}
```

### **優先順位ロジック**
```python
# 階層的優先判定
priority_hierarchy = {
    "level_1": {
        "condition": "all_timeframes_perfect_order",
        "priority": 1,
        "selection_method": "comprehensive_scoring"
    },
    "level_2": {
        "condition": "monthly_weekly_perfect_order", 
        "priority": 2,
        "selection_method": "comprehensive_scoring"
    },
    "level_3": {
        "condition": "others",
        "priority": 3,
        "selection_method": "best_available"
    }
}
```

### **スコアリング重み配分**
```python
# 同一優先レベル内でのスコアリング
scoring_weights = {
    "fundamental": 0.40,      # 業績（営業利益黒字、連続増益等）
    "technical": 0.30,        # RSI、MACD、モメンタム
    "volume": 0.20,           # 出来高・流動性
    "volatility": 0.10        # ボラティリティ適正度
}
```

---

## 🏗️ **アーキテクチャ設計**

### **システム構成図**
```
DSSMS System Architecture
┌─────────────────────────────────────────────────────────┐
│                  DSSMS Core Engine                      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Screening   │  │ Scoring     │  │ Selection   │     │
│  │ Engine      │→ │ Engine      │→ │ Manager     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│           ↑                ↑                ↑           │
├───────────┼────────────────┼────────────────┼───────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Perfect     │  │ Fundamental │  │ Market      │     │
│  │ Order       │  │ Analyzer    │  │ Condition   │     │
│  │ Detector    │  │             │  │ Monitor     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│                Integration Layer                        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ kabu_api    │  │ Multi       │  │ Risk        │     │
│  │ Integration │  │ Strategy    │  │ Management  │     │
│  │             │  │ Adapter     │  │             │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### **データフロー**
```
1. Nikkei225 Symbols (225) 
   ↓ [price/market_cap/affordability filters]
2. Filtered Symbols (50)
   ↓ [perfect_order_detection]  
3. Categorized by Priority (Level 1/2/3)
   ↓ [comprehensive_scoring]
4. Ranked Symbols by Priority Group
   ↓ [intelligent_selection]
5. Selected Symbol (1) + Backup Candidates (5)
   ↓ [multi_strategy_application]
6. Strategy Execution + Real-time Monitoring
   ↓ [switch_condition_monitoring]
7. Dynamic Symbol Switching (as needed)
```

---

## 📋 **実装フェーズ計画**

### **Phase 1: コアエンジン開発** ⏱️ *3-4週間*

#### **Task 1.1: パーフェクトオーダー検出エンジン**
**ファイル**: `src/dssms/perfect_order_detector.py`
```python
class PerfectOrderDetector:
    """SBI証券準拠のマルチタイムフレーム・パーフェクトオーダー検出"""
    
    def __init__(self):
        self.timeframes = {
            "daily": {"short": 5, "medium": 25, "long": 75},
            "weekly": {"short": 13, "medium": 26, "long": 52},
            "monthly": {"short": 9, "medium": 24, "long": 60}
        }
    
    def detect_perfect_order(self, data: pd.DataFrame, timeframe: str) -> Dict[str, bool]
    def check_multi_timeframe_perfect_order(self, symbol: str) -> Dict[str, Any]
    def calculate_perfect_order_priority(self, results: Dict) -> int  # 1=全軸, 2=月週, 3=その他
```

**実装ポイント**:
- `indicators/unified_trend_detector.py`を拡張活用
- SBI証券のMA期間設定に準拠
- 複数時間軸データの効率的処理

#### **Task 1.2: 日経225スクリーニングエンジン**
**ファイル**: `src/dssms/nikkei225_screener.py`
```python
class Nikkei225Screener:
    """日経225銘柄の多段階フィルタリング"""
    
    def fetch_nikkei225_symbols(self) -> List[str]  # 225銘柄取得
    def apply_price_filter(self, symbols: List[str], min_price: float = 500) -> List[str]
    def apply_market_cap_filter(self, symbols: List[str], min_cap: float = 1e10) -> List[str]
    def apply_affordability_filter(self, symbols: List[str], available_funds: float) -> List[str]
    def get_filtered_symbols(self, available_funds: float) -> List[str]  # 最終50銘柄
```

**実装ポイント**:
- Yahoo Finance API活用（reference_docs/Yahoo Finance API/yfinance参照）
- 時価総額・流動性データ取得
- ドローダウン時の資金調整ロジック

#### **Task 1.3: 業績データ取得・分析器**
**ファイル**: `src/dssms/fundamental_analyzer.py`
```python
class FundamentalAnalyzer:
    """Yahoo Finance APIによる業績データ分析"""
    
    def fetch_financial_data(self, symbol: str) -> Dict[str, Any]
    def check_operating_profit_positive(self, symbol: str) -> bool
    def check_consecutive_growth(self, symbol: str, quarters: int = 3) -> bool
    def check_consensus_beat(self, symbol: str) -> bool  # 予想超え判定
    def calculate_fundamental_score(self, symbol: str) -> float  # 0-1スコア
```

**実装ポイント**:
- Yahoo Finance APIの業績データ活用
- 営業利益黒字判定
- 3四半期連続増益チェック
- コンセンサス予想比較

### **Phase 2: 統合スコアリングシステム** ⏱️ *2-3週間*

#### **Task 2.1: 階層的銘柄ランキングシステム**
**ファイル**: `src/dssms/hierarchical_ranking_system.py`
```python
class HierarchicalRankingSystem:
    """優先度ベースの階層的ランキング"""
    
    def categorize_by_perfect_order_priority(self, symbols: List[str]) -> Dict[int, List[str]]
    def rank_within_priority_group(self, symbols: List[str]) -> List[Tuple[str, float]]
    def get_top_candidate(self, available_funds: float) -> Optional[str]
    def get_backup_candidates(self, n: int = 5) -> List[str]
```

#### **Task 2.2: 総合スコアリングエンジン**
**ファイル**: `src/dssms/comprehensive_scoring_engine.py`
```python
class ComprehensiveScoringEngine:
    """優先グループ内での詳細スコアリング"""
    
    def __init__(self):
        self.weights = {
            "fundamental": 0.40,    # 業績関連
            "technical": 0.30,      # RSI, MACD等
            "volume": 0.20,         # 出来高・流動性
            "volatility": 0.10      # ボラティリティ
        }
    
    def calculate_technical_score(self, symbol: str) -> float
    def calculate_volume_score(self, symbol: str) -> float  
    def calculate_volatility_score(self, symbol: str) -> float
    def calculate_composite_score(self, symbol: str) -> float
```

### **Phase 3: 動的選択・切替システム** ⏱️ *2-3週間*

#### **Task 3.1: 市場全体監視システム**
**ファイル**: `src/dssms/market_condition_monitor.py`
```python
class MarketConditionMonitor:
    """日経225指数ベースの市場監視"""
    
    def analyze_nikkei225_trend(self) -> Dict[str, Any]
    def check_market_perfect_order(self) -> bool
    def should_halt_trading(self) -> Tuple[bool, str]  # 売買停止判定
    def get_market_health_score(self) -> float
```

#### **Task 3.2: インテリジェント銘柄切替管理**
**ファイル**: `src/dssms/intelligent_switch_manager.py`
```python
class IntelligentSwitchManager:
    """高度な銘柄切替ロジック"""
    
    def evaluate_current_position(self, symbol: str) -> Dict[str, Any]
    def check_perfect_order_breakdown(self, symbol: str) -> Dict[str, Any]
    def should_immediate_switch(self, current: str, candidate: str) -> bool
    def execute_switch_with_risk_control(self, from_symbol: str, to_symbol: str) -> bool
    def update_available_funds_after_drawdown(self) -> float
```

### **Phase 4: kabu_api統合・実行システム** ⏱️ *3週間*

#### **Task 4.1: kabu_api統合マネージャー**
**ファイル**: `src/dssms/kabu_integration_manager.py`
```python
class KabuIntegrationManager:
    """kabu_apiとの完全統合"""
    
    def register_screening_symbols(self, symbols: List[str]) -> bool  # 50銘柄登録
    def get_realtime_data_for_selected(self, symbol: str) -> pd.DataFrame
    def execute_dynamic_orders(self, switch_data: Dict) -> Dict[str, Any]
    def monitor_position_status(self) -> Dict[str, Any]
```

#### **Task 4.2: DSSMS実行スケジューラー**
**ファイル**: `src/dssms/dssms_scheduler.py`
```python
class DSSMSScheduler:
    """前場後場スケジューリング"""
    
    def run_morning_screening(self) -> str  # 09:30実行
    def run_afternoon_screening(self) -> str  # 12:30実行
    def start_selected_symbol_monitoring(self, symbol: str) -> None
    def handle_emergency_switch_check(self) -> None  # パーフェクトオーダー崩れ時
```

### **Phase 5: バックテスト・分析システム** ⏱️ *3-4週間*

#### **Task 5.1: DSSMS専用バックテスター**
**ファイル**: `src/dssms/dssms_backtester.py`
```python
class DSSMSBacktester:
    """動的銘柄選択バックテスト"""
    
    def simulate_dynamic_selection(self, start_date: str, end_date: str) -> Dict
    def track_symbol_switches(self) -> List[Dict]  # 切替履歴
    def calculate_dssms_performance(self) -> Dict[str, float]
    def compare_with_static_strategy(self) -> Dict[str, Any]
```

#### **Task 5.2: DSSMS分析システム**
**ファイル**: `src/dssms/dssms_analyzer.py`
```python
class DSSMSAnalyzer:
    """DSSMS専用分析システム"""
    
    def analyze_symbol_selection_accuracy(self) -> Dict[str, float]
    def optimize_switching_parameters(self) -> Dict[str, Any]
    def generate_performance_report(self) -> Dict[str, Any]
```

---

## ⚙️ **設定ファイル構造**

### **メイン設定**: `config/dssms/dssms_config.json`
```json
{
  "screening": {
    "nikkei225_filters": {
      "min_price": 500,
      "min_market_cap": 10000000000,
      "min_trading_volume": 100000,
      "max_symbols": 50
    },
    "perfect_order": {
      "daily": {"short": 5, "medium": 25, "long": 75},
      "weekly": {"short": 13, "medium": 26, "long": 52},
      "monthly": {"short": 9, "medium": 24, "long": 60}
    },
    "priority_logic": {
      "level_1": "all_timeframes_perfect_order",
      "level_2": "monthly_weekly_perfect_order", 
      "level_3": "others"
    }
  },
  "scoring": {
    "weights_within_priority": {
      "fundamental": 0.40,
      "technical": 0.30,
      "volume": 0.20,
      "volatility": 0.10
    },
    "fundamental_criteria": {
      "operating_profit_positive": true,
      "consecutive_growth_quarters": 3,
      "consensus_beat_preferred": true
    },
    "technical_indicators": {
      "rsi_period": 14,
      "macd_fast": 12,
      "macd_slow": 26,
      "bollinger_period": 20
    }
  },
  "risk_management": {
    "affordability_check": true,
    "min_shares_purchasable": 100,
    "drawdown_fund_adjustment": true,
    "max_position_size_ratio": 0.1
  },
  "market_halt": {
    "nikkei225_trend_required": true,
    "min_viable_candidates": 1,
    "halt_on_market_crash": true
  },
  "switching": {
    "immediate_switch_conditions": [
      "weekly_perfect_order_breakdown",
      "monthly_perfect_order_breakdown",
      "no_position_and_better_candidate"
    ],
    "observation_conditions": [
      "daily_perfect_order_breakdown_only",
      "minor_score_degradation"
    ],
    "observation_period_days": 3,
    "profit_protection": {
      "min_profit_threshold": 0.02,
      "trailing_stop_percentage": 0.05
    }
  },
  "execution": {
    "screening_schedule": {
      "morning_session": "09:30",
      "afternoon_session": "12:30"
    },
    "monitoring_interval_minutes": 5,
    "emergency_check_interval_minutes": 1
  }
}
```

---

## 🚀 **実装開始手順**

### **Step 1: プロジェクト構造準備**
```powershell
# DSSMSディレクトリ構造作成
mkdir src\dssms src\dssms\screening src\dssms\scoring src\dssms\switching config\dssms tests\dssms logs\dssms

# 設定ファイル作成
New-Item -Path "config\dssms\dssms_config.json" -ItemType File
New-Item -Path "src\dssms\__init__.py" -ItemType File

# ログディレクトリ作成
mkdir logs\dssms\screening logs\dssms\ranking logs\dssms\switching logs\dssms\performance
```

### **Step 2: Phase 1開始**
```powershell
# Task 1.1開始: パーフェクトオーダー検出器
New-Item -Path "src\dssms\perfect_order_detector.py" -ItemType File

# Task 1.2開始: 日経225スクリーナー
New-Item -Path "src\dssms\nikkei225_screener.py" -ItemType File

# Task 1.3開始: 業績分析器
New-Item -Path "src\dssms\fundamental_analyzer.py" -ItemType File
```

### **Step 3: 設定ファイル初期化**
```powershell
# 設定ファイルテンプレートコピー
Copy-Item "config\dssms\dssms_config_template.json" "config\dssms\dssms_config.json"

# ログ設定初期化
python -c "from src.utils.logger_setup import setup_logger; setup_logger('dssms')"
```

---

## 🔁 Purpose Hierarchy
| レベル | 説明 |
|--------|------|
| Business | 日次動的選択 + 戦略適用 + kabu発注によるリスク調整後利益最大化 |
| System | Perfect Order階層選別→スコア→最適/バックアップ→切替判定→学習 |
| Technical | 処理 <30s / 再現性 / 監査可能切替ログ / 出力一意性 |
| Operational | 日次成功率>99% / 不要切替率<20% / Excel一致率100% |
| Improvement Loop | 切替後10日検証→不要切替率→適応スコア補正更新 |

## 📊 KPI 定義
- 不要切替: 切替後10営業日累積利益率 ≤ 取引コスト(往復0.2%)  
- 評価バッチ: 週次集計 + 月次リセット指標  
- 再現性トリガ: 乱数種 / 決定論フラグ / キャッシュキー / バージョンID

## 🧱 Error Severity Matrix
| Severity | 例 | 処理 |
|----------|----|------|
| CRITICAL | 日付ループ/価格NaN連鎖/注文拒否 | 停止+通知 |
| ERROR | 単一銘柄指標欠損 | 代替/除外 |
| WARNING | 欠損補完/再試行成功 | 継続 |
| INFO | 正常終了 | 記録 |
| DEBUG | 内部スコア詳細 | 開発時のみ |

## 🔄 切替評価ワークフロー
1. SwitchEvent記録 (scores, reasons, expected_alpha)
2. 10営業日後 outcome attach
3. 不要切替判定 → 集計
4. Adaptive補正へフィードバック

---

## 📚 **関連ドキュメント**

- [`docs/dssms/DSSMS_PHASE_BREAKDOWN.md`](docs/dssms/DSSMS_PHASE_BREAKDOWN.md) - フェーズ別詳細実装
- [`docs/dssms/DSSMS_CONFIG_TEMPLATES.md`](docs/dssms/DSSMS_CONFIG_TEMPLATES.md) - 設定ファイルテンプレート
- [`docs/dssms/DSSMS_QUICK_START_GUIDE.md`](docs/dssms/DSSMS_QUICK_START_GUIDE.md) - クイックスタート
- [`docs/operation_manual.md`](docs/operation_manual.md) - 既存運用マニュアル

---

## 🤝 **コントリビューション**

### **開発ガイドライン**
1. コーディング規約は `.github/copilot-instructions.md` に準拠
2. 新機能は必ず単体テストを含める
3. ログ出力は `config/logger_config.py` を使用
4. 設定変更時は妥当性検証を実施

### **レビュープロセス**
1. コードレビュー必須
2. パフォーマンステスト実行
3. バックテスト結果検証
4. ドキュメント更新

---

**最終更新**: 2025年8月16日  
**バージョン**: 1.0  
**作成者**: GitHub Copilot + Human Collaboration  
**次回レビュー予定**: Phase 1完了時
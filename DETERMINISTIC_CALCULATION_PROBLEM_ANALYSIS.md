# 決定論的計算による切替回数激減問題 - 解決戦略書

**作成日時**: 2025年9月25日  
**対象システム**: DSSMS (Dynamic Stock Selection and Management System)  
**問題**: 切替回数117回→3-1回への激減  
**根本原因**: 決定論的計算の導入による動的判定機能の停止  

## 🚨 **Critical Discovery: 決定論的計算が切替機能を破壊**

### **問題の核心**
```
決定論的計算導入前:
├─ 実データ変動 → 大きなスコア差 → 頻繁な切替判定 → 117回切替
├─ 市場変動反映 → 動的ランキング変更 → アクティブな銘柄選択
└─ ノイズ含有 → 不完全だが活発な判定システム

決定論的計算導入後:
├─ 固定スコア範囲(0.3-0.7) → 小さなスコア差 → 切替判定不成立
├─ 静的ランキング → ランキング変更なし → 1-3回切替のみ
└─ ノイズ排除 → 過度な安定化 → 判定機能の実質停止
```

### **技術的根拠**
```python
# 問題のある決定論的計算
def _generate_deterministic_score(self, symbol: str, date: datetime) -> float:
    symbol_hash = hash(f"{symbol}_{date.strftime('%Y-%m-%d')}")
    base_score = ((symbol_hash % 1000) / 1000) * 0.4 + 0.3  # 0.3-0.7固定範囲
    return base_score

# 問題点:
# 1. 同じ銘柄・日付で常に同じスコア（変動なし）
# 2. スコア差が0.4範囲に制限（判定しきい値未達）
# 3. 市場の実際の変動を完全に無視
# 4. 切替判定アルゴリズムの前提条件を破壊
```

## 🎯 **解決戦略: 高度な実データ分析システム復旧**

### **Phase 1: 決定論的計算完全除去 (優先度: CRITICAL)**

#### **Step 1.1: ComprehensiveScoringEngine実データ復活**
```python
# TODO(tag:phase1, rationale:決定論除去): src/dssms/comprehensive_scoring.py
class ComprehensiveScoringEngine:
    def __init__(self, data_manager=None):
        self.data_manager = data_manager
        self.use_deterministic = False  # 🔧 完全に無効化
        
    def calculate_composite_score(self, symbol: str, date: datetime = None) -> float:
        """実データベースの動的スコア計算"""
        try:
            # 実データ取得 (yfinance正常化済み)
            market_data = self._get_market_data(symbol)
            if market_data is None or len(market_data) == 0:
                return self._calculate_fallback_score(symbol, date)
            
            # 実際の市場指標計算
            technical_score = self._calculate_real_technical_score(market_data)
            volume_score = self._calculate_real_volume_score(market_data)
            volatility_score = self._calculate_real_volatility_score(market_data)
            momentum_score = self._calculate_momentum_score(market_data)
            
            # 動的重み付け合成
            composite_score = (
                technical_score * 0.3 +
                volume_score * 0.25 + 
                volatility_score * 0.2 +
                momentum_score * 0.25
            )
            
            return max(0.1, min(0.9, composite_score))
            
        except Exception as e:
            self.logger.warning(f"実データ計算失敗 {symbol}: {e}")
            return self._calculate_fallback_score(symbol, date)
```

#### **Step 1.2: 実データ技術指標計算実装**
```python
# TODO(tag:phase1, rationale:実データ分析): 市場データベース指標
def _calculate_real_technical_score(self, data: pd.DataFrame) -> float:
    """実際の技術指標によるスコア算出"""
    try:
        closes = data['Close'].values
        
        # RSI計算 (Relative Strength Index)
        rsi = self._calculate_rsi(closes, period=14)
        rsi_score = self._normalize_rsi_score(rsi)
        
        # MACD計算 (Moving Average Convergence Divergence)
        macd_line, signal_line = self._calculate_macd(closes)
        macd_score = self._normalize_macd_score(macd_line, signal_line)
        
        # ボリンジャーバンド位置
        bb_position = self._calculate_bb_position(closes)
        bb_score = self._normalize_bb_score(bb_position)
        
        # 複合技術スコア
        tech_score = (rsi_score * 0.4 + macd_score * 0.3 + bb_score * 0.3)
        
        return tech_score
        
    except Exception as e:
        self.logger.warning(f"技術指標計算エラー: {e}")
        return 0.5

def _calculate_real_volume_score(self, data: pd.DataFrame) -> float:
    """実際の出来高データによるスコア"""
    try:
        volumes = data['Volume'].values
        avg_volume = np.mean(volumes[-20:])  # 20日平均出来高
        recent_volume = np.mean(volumes[-5:])  # 直近5日平均出来高
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # 出来高急増でスコア向上（市場注目度反映）
        if volume_ratio > 1.5:
            return 0.8  # 活発な取引
        elif volume_ratio > 1.2:
            return 0.7  # やや活発
        elif volume_ratio > 0.8:
            return 0.6  # 通常範囲
        else:
            return 0.4  # 低調
            
    except Exception as e:
        return 0.5

def _calculate_real_volatility_score(self, data: pd.DataFrame) -> float:
    """実際の価格変動率によるスコア"""
    try:
        closes = data['Close'].values
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # 年率換算ボラティリティ
        
        # 適度なボラティリティを高評価
        if 0.15 <= volatility <= 0.35:
            return 0.8  # 適度な変動（投資機会あり）
        elif 0.10 <= volatility <= 0.50:
            return 0.6  # 許容範囲
        else:
            return 0.4  # 過度に安定または不安定
            
    except Exception as e:
        return 0.5
```

### **Phase 2: 動的ランキングシステム復旧**

#### **Step 2.1: リアルタイム市場反応システム**
```python
# TODO(tag:phase2, rationale:動的判定): src/dssms/dynamic_ranking.py
class DynamicRankingEngine:
    """動的・リアルタイム対応ランキングエンジン"""
    
    def __init__(self):
        self.market_sensitivity = 0.8  # 市場感度（高いほど変動敏感）
        self.ranking_memory = {}  # 前回ランキング記憶
        
    def generate_dynamic_ranking(self, symbols: List[str], date: datetime) -> Dict[str, float]:
        """市場変動を反映した動的ランキング生成"""
        current_scores = {}
        
        for symbol in symbols:
            # 実データベーススコア取得
            base_score = self.scoring_engine.calculate_composite_score(symbol, date)
            
            # 市場変動調整
            market_adjustment = self._calculate_market_adjustment(symbol, date)
            
            # 相対的位置調整
            relative_adjustment = self._calculate_relative_position_adjustment(symbol, symbols, date)
            
            # 動的合成スコア
            dynamic_score = base_score * (1 + market_adjustment) * (1 + relative_adjustment)
            current_scores[symbol] = dynamic_score
            
        # ランキング変動分析
        ranking_change = self._analyze_ranking_change(current_scores)
        
        # 記憶更新
        self.ranking_memory[date] = current_scores
        
        return current_scores
        
    def _calculate_market_adjustment(self, symbol: str, date: datetime) -> float:
        """市場全体の変動に基づく調整"""
        try:
            # 市場指数取得（例：日経平均）
            market_data = self._get_market_index_data(date)
            symbol_data = self._get_symbol_data(symbol, date)
            
            if market_data is None or symbol_data is None:
                return 0.0
                
            # 市場との相関・ベータ値計算
            correlation = self._calculate_correlation(market_data, symbol_data)
            
            # 市場強度に基づく調整
            market_strength = self._calculate_market_strength(market_data)
            
            return correlation * market_strength * self.market_sensitivity
            
        except Exception:
            return 0.0
```

### **Phase 3: 高度な切替判定ロジック復旧**

#### **Step 3.1: 多次元切替判定システム**
```python
# TODO(tag:phase3, rationale:高度判定): src/dssms/advanced_switching.py
class AdvancedSwitchingLogic:
    """高度な切替判定システム"""
    
    def should_switch_symbol(self, current_symbol: str, top_ranked_symbol: str, 
                           ranking_scores: Dict[str, float], date: datetime) -> bool:
        """多次元的切替判定"""
        
        # 1. スコア差分析
        score_gap = ranking_scores.get(top_ranked_symbol, 0) - ranking_scores.get(current_symbol, 0)
        score_switch = score_gap > self.score_threshold
        
        # 2. トレンド分析
        trend_switch = self._analyze_trend_switch_signal(current_symbol, top_ranked_symbol, date)
        
        # 3. 市場状況分析  
        market_switch = self._analyze_market_condition_switch(date)
        
        # 4. リスク調整分析
        risk_switch = self._analyze_risk_adjusted_switch(current_symbol, top_ranked_symbol)
        
        # 5. 統合判定
        switch_factors = {
            'score_gap': score_switch,
            'trend_signal': trend_switch, 
            'market_condition': market_switch,
            'risk_assessment': risk_switch
        }
        
        # 複数要因による総合判定
        positive_factors = sum(switch_factors.values())
        
        # 2/4以上で切替実行
        final_decision = positive_factors >= 2
        
        self.logger.info(f"切替判定 {current_symbol}→{top_ranked_symbol}: {switch_factors} → {final_decision}")
        
        return final_decision
```

## 🔧 **実装優先順位・タイムライン**

### **即座に実行すべき修正**
```python
# TODO(tag:immediate, rationale:決定論除去): src/dssms/dssms_backtester.py
class DSSMSBacktester:
    def _update_symbol_ranking(self, date: datetime, symbols: List[str]) -> Dict[str, Any]:
        # 🚨 決定論的計算を完全除去
        use_integration_patch = True  # 強制的に有効化
        use_deterministic_scoring = False  # 強制的に無効化
        
        # ComprehensiveScoringEngineによる実データ分析
        real_scores = {}
        for symbol in symbols:
            score = self.scoring_engine.calculate_composite_score(symbol, date)
            real_scores[symbol] = score
            
        # 動的ランキング生成
        sorted_symbols = sorted(real_scores.keys(), key=lambda s: real_scores[s], reverse=True)
        top_symbol = sorted_symbols[0] if sorted_symbols else None
        
        return {
            'date': date,
            'rankings': real_scores,
            'top_symbol': top_symbol,
            'method': 'real_data_analysis',
            'deterministic': False  # 明示的に非決定論
        }
```

### **実装タイムライン**
```
Hour 1: 決定論的計算完全除去
Hour 2-3: ComprehensiveScoringEngine実データ復活
Hour 4: 動的ランキングシステム実装
Hour 5: 高度切替判定ロジック実装
Hour 6: 統合テスト・検証
```

## 📊 **期待される効果**

### **切替回数回復**
```
現状: 1-3回 (決定論的計算による停滞)
目標: 50-117回 (実データ変動による活発な判定)
根拠: 市場の実際の変動がスコア計算に反映される
```

### **ランキング精度向上**
```
現状: 固定スコア (0.3-0.7範囲) → 銘柄間差異なし
目標: 動的スコア (0.1-0.9範囲) → 明確な優劣判定
根拠: RSI、MACD、出来高等の実データ指標活用
```

### **投資判断の市場連動性復活**
```
現状: 機械的・静的判定 → 市場変動無視
目標: 市場連動・動的判定 → リアルタイム市場反映
根拠: 市場指数との相関・ベータ値による調整
```

## 🚨 **リスク評価・対策**

### **実装リスク**
- **Technical Risk**: 実データAPI依存によるエラー増加可能性
- **対策**: 堅牢なエラーハンドリング・フォールバック機構実装

### **パフォーマンスリスク**
- **Performance Risk**: 実データ計算による処理速度低下
- **対策**: 並列処理・キャッシュ機構・段階的実装

### **投資リスク**
- **Investment Risk**: 切替回数増加による取引コスト増
- **対策**: 切替判定しきい値の最適化・リスク調整機構

## 📋 **Success Criteria（成功判定基準）**

### **Technical Success**
- ✅ 決定論的計算の完全除去確認
- ✅ ComprehensiveScoringEngine実データ取得成功率 >90%
- ✅ 動的スコア計算の正常動作確認

### **Functional Success**  
- ✅ 切替回数: 3回 → 30回以上への回復
- ✅ top_symbol=None問題の完全解決
- ✅ ランキング変動の確認（日次変化率 >10%）

### **Investment Success**
- ✅ バックテスト結果の市場連動性向上
- ✅ 117回レベルへの段階的回復達成
- ✅ 投資戦略の実用性確認

## 🎯 **Next Actions (エージェントモード実行用)**

### **Phase 1実行コマンド**
```bash
# Step 1: 決定論的計算除去
python -c "
from src.dssms.dssms_backtester import DSSMSBacktester
backtester = DSSMSBacktester()
backtester.use_deterministic_scoring = False
print('決定論的計算を無効化しました')
"

# Step 2: 実データシステム復活テスト
python src/dssms/comprehensive_scoring.py

# Step 3: 切替回数確認テスト
python src/dssms/dssms_backtester.py
```

### **検証コマンド**
```bash
# 切替回数カウント
grep -o "切替実行" logs/*.log | wc -l

# スコア変動確認  
grep "composite_score" logs/*.log | head -20

# エラー監視
grep -i "error\|exception" logs/*.log
```

---

## 📑 **結論**

**決定論的計算は切替回数激減の根本原因**。市場の実際の変動を無視した固定スコア範囲により、切替判定アルゴリズムが機能停止した。

**解決策**: 決定論的計算を完全除去し、ComprehensiveScoringEngineによる実データ分析システムを復活させることで、117回レベルの切替回数復旧が期待できる。

**実装順序**: 決定論除去 → 実データ復活 → 動的判定復旧の3段階で段階的に実装。

この戦略により、本来のDSSMSシステムの動的・活発な銘柄選択機能を完全復旧できる。
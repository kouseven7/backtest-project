"""
Performance Data Collector for 4-3-2 Dashboard
既存システムからの戦略パフォーマンス・配分データ収集

統合対象:
- StrategySelector (戦略選択状況)
- PortfolioWeightCalculator (配分比率)
- StrategyScoreCalculator (パフォーマンス評価)
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field

# プロジェクトパス追加
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 既存システムインポート
try:
    from config.strategy_selector import StrategySelector, SelectionCriteria
    from config.portfolio_weight_calculator import PortfolioWeightCalculator, WeightAllocationConfig
    from config.strategy_scoring_model import StrategyScoreCalculator, StrategyScoreManager
    from visualization.data_aggregator import VisualizationDataAggregator
except ImportError as e:
    logging.getLogger(__name__).warning(f"Import error: {e}")

logger = logging.getLogger(__name__)

@dataclass
class PerformanceSnapshot:
    """パフォーマンススナップショット"""
    timestamp: datetime
    ticker: str
    strategy_allocations: Dict[str, float]
    strategy_scores: Dict[str, float]
    total_performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    market_context: Dict[str, Any]
    alerts: List[str] = field(default_factory=list)

class PerformanceDataCollector:
    """パフォーマンスデータ収集器"""
    
    def __init__(self, output_dir: str = "logs/dashboard/performance_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 既存システム初期化（エラー耐性付き）
        try:
            self.strategy_selector = StrategySelector()
        except Exception as e:
            logger.warning(f"StrategySelector initialization failed: {e}")
            self.strategy_selector = None
            
        try:
            self.weight_calculator = PortfolioWeightCalculator()
        except Exception as e:
            logger.warning(f"PortfolioWeightCalculator initialization failed: {e}")
            self.weight_calculator = None
            
        try:
            self.score_manager = StrategyScoreManager()
        except Exception as e:
            logger.warning(f"StrategyScoreManager initialization failed: {e}")
            self.score_manager = None
            
        try:
            self.data_aggregator = VisualizationDataAggregator()
        except Exception as e:
            logger.warning(f"VisualizationDataAggregator initialization failed: {e}")
            self.data_aggregator = None
        
        # 履歴管理
        self.data_retention_days = 90  # 3ヶ月
        
        logger.info("PerformanceDataCollector initialized")
    
    def collect_current_snapshot(self, ticker: str, market_data: pd.DataFrame = None) -> PerformanceSnapshot:
        """現在のパフォーマンススナップショットを収集"""
        try:
            timestamp = datetime.now()
            
            # サンプル市場データ生成（引数がない場合）
            if market_data is None:
                market_data = self._generate_sample_market_data()
            
            # 1. 戦略選択情報の取得
            strategy_allocations = {}
            try:
                if self.strategy_selector:
                    selection_criteria = SelectionCriteria(max_strategies=5, min_score_threshold=0.3)
                    strategy_selection = self.strategy_selector.select_strategies(
                        market_data, ticker, selection_criteria
                    )
                    if hasattr(strategy_selection, 'selected_strategies'):
                        # 選択された戦略から配分を作成
                        selected_strategies = strategy_selection.selected_strategies
                        if selected_strategies:
                            weight_per_strategy = 1.0 / len(selected_strategies)
                            strategy_allocations = {
                                strategy: weight_per_strategy 
                                for strategy in selected_strategies
                            }
            except Exception as e:
                logger.warning(f"Strategy selection failed: {e}")
                strategy_allocations = self._generate_sample_allocations()
            
            if not strategy_allocations:
                strategy_allocations = self._generate_sample_allocations()
            
            # 2. ポートフォリオ重み情報の取得
            try:
                if self.weight_calculator:
                    weight_config = WeightAllocationConfig()
                    allocation_result = self.weight_calculator.calculate_portfolio_weights(
                        ticker, market_data, weight_config
                    )
                    if hasattr(allocation_result, 'strategy_weights') and allocation_result.strategy_weights:
                        strategy_allocations.update(allocation_result.strategy_weights)
            except Exception as e:
                logger.warning(f"Portfolio weight calculation failed: {e}")
            
            # 3. 戦略スコア情報の取得
            strategy_scores = {}
            try:
                if self.score_manager:
                    available_strategies = list(strategy_allocations.keys())[:10]
                    for strategy in available_strategies:
                        score = self.score_manager.calculator.calculate_strategy_score(strategy, ticker)
                        if score:
                            strategy_scores[strategy] = score.total_score
            except Exception as e:
                logger.warning(f"Strategy score calculation failed: {e}")
                
            if not strategy_scores:
                strategy_scores = self._generate_sample_scores(list(strategy_allocations.keys()))
            
            # 4. パフォーマンス指標の計算
            total_performance = self._calculate_total_performance(
                strategy_allocations, strategy_scores, market_data
            )
            
            # 5. リスク指標の計算
            risk_metrics = self._calculate_risk_metrics(
                strategy_allocations, market_data
            )
            
            # 6. 市場コンテキストの構築
            market_context = self._build_market_context(market_data)
            
            # 7. アラートの生成
            alerts = self._generate_alerts(
                strategy_allocations, risk_metrics, total_performance
            )
            
            snapshot = PerformanceSnapshot(
                timestamp=timestamp,
                ticker=ticker,
                strategy_allocations=strategy_allocations,
                strategy_scores=strategy_scores,
                total_performance=total_performance,
                risk_metrics=risk_metrics,
                market_context=market_context,
                alerts=alerts
            )
            
            # ファイル保存
            self._save_snapshot(snapshot)
            
            logger.info(f"Performance snapshot collected for {ticker}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to collect snapshot for {ticker}: {e}")
            return self._create_empty_snapshot(ticker, e)
    
    def _generate_sample_market_data(self) -> pd.DataFrame:
        """サンプル市場データ生成"""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        prices = 110 + np.random.randn(30).cumsum() * 0.5
        volumes = np.random.randint(1000, 10000, 30)
        
        return pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Volume': volumes
        })
    
    def _generate_sample_allocations(self) -> Dict[str, float]:
        """サンプル戦略配分生成"""
        strategies = ["VWAPBounce", "Momentum", "Breakout", "MeanReversion", "TrendFollowing"]
        weights = np.random.dirichlet(np.ones(len(strategies)), size=1)[0]
        return {strategy: weight for strategy, weight in zip(strategies, weights)}
    
    def _generate_sample_scores(self, strategies: List[str]) -> Dict[str, float]:
        """サンプル戦略スコア生成"""
        return {strategy: np.random.uniform(0.4, 0.9) for strategy in strategies}
    
    def _calculate_total_performance(self, allocations: Dict[str, float], 
                                   scores: Dict[str, float], market_data: pd.DataFrame) -> Dict[str, float]:
        """総合パフォーマンスの計算"""
        try:
            # 基本指標計算
            weighted_score = sum(allocations.get(strategy, 0) * scores.get(strategy, 0) 
                               for strategy in allocations.keys())
            
            portfolio_return = weighted_score * 10.0  # スコアベースのリターン推定
            portfolio_risk = max(5.0, 20.0 * (1 - weighted_score))  # リスク推定
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            performance = {
                "portfolio_return": portfolio_return,
                "portfolio_risk": portfolio_risk,
                "sharpe_ratio": sharpe_ratio,
                "diversification_ratio": len(allocations) / 10.0  # 分散化指標
            }
            
            # 市場データベースの追加指標
            if not market_data.empty and 'Close' in market_data.columns:
                returns = market_data['Close'].pct_change().dropna()
                if len(returns) > 0:
                    performance.update({
                        "daily_volatility": float(returns.std() * 100),
                        "max_daily_return": float(returns.max() * 100),
                        "min_daily_return": float(returns.min() * 100)
                    })
            
            return performance
            
        except Exception as e:
            logger.warning(f"Performance calculation error: {e}")
            return {"portfolio_return": 2.5, "portfolio_risk": 12.0, "sharpe_ratio": 0.6}
    
    def _calculate_risk_metrics(self, allocations: Dict[str, float], market_data: pd.DataFrame) -> Dict[str, float]:
        """リスク指標の計算"""
        try:
            # 集中度リスク（ハーフィンダール指数）
            hhi = sum(w**2 for w in allocations.values()) if allocations else 0
            concentration_risk = hhi * 100
            
            # VaR推定
            portfolio_volatility = 15.0  # デフォルトボラティリティ
            if not market_data.empty and 'Close' in market_data.columns:
                returns = market_data['Close'].pct_change().dropna()
                if len(returns) > 0:
                    portfolio_volatility = float(returns.std() * 100 * np.sqrt(252))
            
            var_95 = portfolio_volatility * 1.65  # 95% VaR
            
            risk_metrics = {
                "var_95": var_95,
                "max_drawdown": min(25.0, portfolio_volatility * 0.8),
                "risk_score": max(0, 100 - concentration_risk - var_95 * 2),
                "concentration_risk": concentration_risk,
                "portfolio_volatility": portfolio_volatility
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.warning(f"Risk calculation error: {e}")
            return {"var_95": 15.0, "max_drawdown": 10.0, "risk_score": 70.0, "concentration_risk": 30.0}
    
    def _build_market_context(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """市場コンテキストの構築"""
        context = {
            "trend": "sideways",
            "trend_confidence": 0.6,
            "trend_strength": 0.5,
            "market_volatility": 0.15,
            "data_quality": "good" if len(market_data) > 20 else "limited"
        }
        
        # 価格動向分析
        if not market_data.empty and 'Close' in market_data.columns:
            prices = market_data['Close']
            if len(prices) >= 2:
                context["price_change_1d"] = float((prices.iloc[-1] / prices.iloc[-2] - 1) * 100)
                
                # 簡易トレンド判定
                if len(prices) >= 10:
                    recent_trend = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10]
                    if recent_trend > 0.02:
                        context["trend"] = "uptrend"
                        context["trend_confidence"] = min(0.9, abs(recent_trend) * 10)
                    elif recent_trend < -0.02:
                        context["trend"] = "downtrend"
                        context["trend_confidence"] = min(0.9, abs(recent_trend) * 10)
                        
            if len(prices) >= 20:
                context["price_change_20d"] = float((prices.iloc[-1] / prices.iloc[-20] - 1) * 100)
        
        return context
    
    def _generate_alerts(self, allocations: Dict[str, float], 
                        risk_metrics: Dict[str, float], performance: Dict[str, float]) -> List[str]:
        """アラートの生成"""
        alerts = []
        
        # 高リスクアラート
        if risk_metrics.get("risk_score", 100) < 50:
            alerts.append("リスクレベル: 高")
        
        # 低パフォーマンスアラート  
        if performance.get("sharpe_ratio", 0) < 0.5:
            alerts.append("シャープレシオ: 低水準")
        
        # 集中度アラート
        if risk_metrics.get("concentration_risk", 0) > 50:
            alerts.append("集中度リスク: 高")
            
        # ボラティリティアラート
        if risk_metrics.get("portfolio_volatility", 0) > 25:
            alerts.append("ボラティリティ: 高水準")
        
        return alerts
    
    def _save_snapshot(self, snapshot: PerformanceSnapshot):
        """スナップショットの保存"""
        try:
            timestamp_str = snapshot.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_{snapshot.ticker}_{timestamp_str}.json"
            filepath = self.output_dir / filename
            
            # JSON形式で保存
            snapshot_data = {
                "timestamp": snapshot.timestamp.isoformat(),
                "ticker": snapshot.ticker,
                "strategy_allocations": snapshot.strategy_allocations,
                "strategy_scores": snapshot.strategy_scores,
                "total_performance": snapshot.total_performance,
                "risk_metrics": snapshot.risk_metrics,
                "market_context": snapshot.market_context,
                "alerts": snapshot.alerts
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Snapshot saved: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
    def load_historical_snapshots(self, ticker: str, days: int = 30) -> List[PerformanceSnapshot]:
        """履歴スナップショットの読み込み"""
        snapshots = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            for file in self.output_dir.glob(f"performance_{ticker}_*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    timestamp = datetime.fromisoformat(data["timestamp"])
                    if timestamp >= cutoff_date:
                        snapshot = PerformanceSnapshot(
                            timestamp=timestamp,
                            ticker=data["ticker"],
                            strategy_allocations=data["strategy_allocations"],
                            strategy_scores=data["strategy_scores"],
                            total_performance=data["total_performance"],
                            risk_metrics=data["risk_metrics"],
                            market_context=data["market_context"],
                            alerts=data["alerts"]
                        )
                        snapshots.append(snapshot)
                        
                except Exception as e:
                    logger.warning(f"Failed to load snapshot {file}: {e}")
            
            # 時系列順にソート
            snapshots.sort(key=lambda x: x.timestamp)
            logger.info(f"Loaded {len(snapshots)} historical snapshots for {ticker}")
            
        except Exception as e:
            logger.error(f"Failed to load historical snapshots: {e}")
        
        return snapshots
    
    def cleanup_old_data(self):
        """古いデータのクリーンアップ"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
            removed_count = 0
            
            for file in self.output_dir.glob("performance_*.json"):
                try:
                    # ファイル名から日付を抽出
                    parts = file.stem.split('_')
                    if len(parts) >= 3:
                        date_str = parts[2]
                        file_date = datetime.strptime(date_str, "%Y%m%d")
                        
                        if file_date < cutoff_date:
                            file.unlink()
                            removed_count += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to process file {file}: {e}")
            
            logger.info(f"Cleaned up {removed_count} old performance files")
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def _create_empty_snapshot(self, ticker: str, error: Exception) -> PerformanceSnapshot:
        """エラー時の空スナップショット作成"""
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            ticker=ticker,
            strategy_allocations={"DefaultStrategy": 1.0},
            strategy_scores={"DefaultStrategy": 0.5},
            total_performance={"portfolio_return": 0.0, "portfolio_risk": 15.0, "sharpe_ratio": 0.0},
            risk_metrics={"var_95": 15.0, "max_drawdown": 10.0, "risk_score": 50.0, "concentration_risk": 100.0},
            market_context={"trend": "unknown", "data_quality": "error"},
            alerts=[f"データ収集エラー: {str(error)}"]
        )

if __name__ == "__main__":
    # テスト用
    collector = PerformanceDataCollector()
    snapshot = collector.collect_current_snapshot("USDJPY")
    print(f"Snapshot collected: {len(snapshot.strategy_allocations)} strategies")
    print(f"Portfolio return: {snapshot.total_performance.get('portfolio_return', 0):.2f}%")
    print(f"Alerts: {len(snapshot.alerts)}")

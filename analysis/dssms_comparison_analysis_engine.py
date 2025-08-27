"""
DSSMS Phase 3 Task 3.2: 比較分析機能向上
統合比較分析エンジン
File: dssms_comparison_analysis_engine.py

DSSMS統合比較分析エンジン
戦略パフォーマンス比較、市場レジーム分析、株式選択効果分析を統合実行

Author: imega (Agent Mode Implementation)
Created: 2025-01-22
Based on: Previous conversation design specifications
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

# 既存システム統合
try:
    from analysis.strategy_switching.strategy_switching_analyzer import (
        StrategySwitchingAnalyzer, SwitchingAnalysisResult, SwitchingEvent, MarketRegime
    )
    from output.simple_excel_exporter import SimpleExcelExporter
    HAS_CORE_INTEGRATION = True
except ImportError as e:
    logging.warning(f"Core integration modules not available: {e}")
    HAS_CORE_INTEGRATION = False

# ロギング設定
logger = logging.getLogger(__name__)

@dataclass
class ComparisonResult:
    """比較分析結果データクラス"""
    analysis_id: str
    timestamp: datetime
    
    # 戦略パフォーマンス比較
    strategy_performance: Dict[str, Dict[str, float]]
    strategy_rankings: Dict[str, List[str]]
    performance_differences: Dict[str, float]
    
    # 市場レジーム分析
    regime_analysis: Dict[str, Any]
    regime_performance: Dict[str, Dict[str, float]]
    regime_transitions: List[Dict[str, Any]]
    
    # 株式選択効果分析
    stock_selection_effects: Dict[str, Dict[str, float]]
    sector_analysis: Dict[str, Any]
    correlation_analysis: Dict[str, Any]
    
    # 統合分析結果
    overall_recommendations: List[str]
    risk_adjusted_rankings: Dict[str, List[str]]
    optimization_suggestions: List[Dict[str, Any]]
    
    # メタデータ
    analysis_period: Tuple[datetime, datetime]
    data_quality_score: float
    confidence_level: float
    analysis_mode: str = "comprehensive"

@dataclass
class AnalysisConfiguration:
    """分析設定データクラス"""
    # 分析モード設定
    analysis_mode: str = "comprehensive"  # comprehensive, quick_summary, deep_dive
    enable_regime_analysis: bool = True
    enable_stock_selection_analysis: bool = True
    enable_correlation_analysis: bool = True
    
    # 分析期間設定
    lookback_period: int = 252  # 1年間
    min_data_points: int = 30
    regime_detection_window: int = 20
    
    # パフォーマンス閾値
    significance_threshold: float = 0.05
    performance_difference_threshold: float = 0.02
    correlation_threshold: float = 0.7
    
    # レポート設定
    output_format: str = "excel"  # excel, html, both
    include_detailed_charts: bool = True
    export_raw_data: bool = False

class AnalysisMode(Enum):
    """分析モード"""
    COMPREHENSIVE = "comprehensive"
    QUICK_SUMMARY = "quick_summary"
    DEEP_DIVE = "deep_dive"

class DSSMSComparisonAnalysisEngine:
    """DSSMS統合比較分析エンジン"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Parameters:
            config_path: 設定ファイルパス
        """
        self.config = self._load_configuration(config_path)
        self.analysis_history = []
        self.cache = {}
        
        # 既存システム統合
        self._initialize_integrations()
        
        # パフォーマンストラッキング
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'average_processing_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
        logger.info("DSSMS比較分析エンジン初期化完了")

    def _load_configuration(self, config_path: Optional[str] = None) -> AnalysisConfiguration:
        """設定読み込み"""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "config", "comparison_config.json"
            )
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return AnalysisConfiguration(**config_data)
            else:
                logger.warning(f"設定ファイルが見つかりません: {config_path}. デフォルト設定を使用")
                return AnalysisConfiguration()
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            return AnalysisConfiguration()

    def _initialize_integrations(self):
        """既存システム統合初期化"""
        self.switching_analyzer = None
        self.excel_exporter = None
        
        if HAS_CORE_INTEGRATION:
            try:
                self.switching_analyzer = StrategySwitchingAnalyzer()
                self.excel_exporter = SimpleExcelExporter()
                logger.info("既存システム統合完了")
            except Exception as e:
                logger.warning(f"既存システム統合失敗: {e}")

    def run_comprehensive_analysis(
        self,
        data: pd.DataFrame,
        strategies: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        analysis_mode: Optional[str] = None
    ) -> ComparisonResult:
        """
        包括的比較分析実行
        
        Parameters:
            data: 分析対象データ
            strategies: 戦略リスト
            start_date: 分析開始日
            end_date: 分析終了日
            analysis_mode: 分析モード
            
        Returns:
            比較分析結果
        """
        start_time = datetime.now()
        
        try:
            # パラメータ設定
            if analysis_mode is None:
                analysis_mode = self.config.analysis_mode
            
            # データ前処理
            processed_data = self._preprocess_data(data, start_date, end_date)
            
            # 分析ID生成
            analysis_id = f"dssms_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 戦略パフォーマンス比較分析
            strategy_performance = self._analyze_strategy_performance(
                processed_data, strategies, analysis_mode
            )
            
            # 市場レジーム分析
            regime_analysis = None
            if self.config.enable_regime_analysis:
                regime_analysis = self._analyze_market_regimes(
                    processed_data, strategies, analysis_mode
                )
            
            # 株式選択効果分析
            stock_selection_analysis = None
            if self.config.enable_stock_selection_analysis:
                stock_selection_analysis = self._analyze_stock_selection_effects(
                    processed_data, strategies, analysis_mode
                )
            
            # 相関分析
            correlation_analysis = None
            if self.config.enable_correlation_analysis:
                correlation_analysis = self._analyze_correlations(
                    processed_data, strategies, analysis_mode
                )
            
            # 統合分析
            integrated_results = self._integrate_analysis_results(
                strategy_performance, regime_analysis, 
                stock_selection_analysis, correlation_analysis
            )
            
            # 結果生成
            result = ComparisonResult(
                analysis_id=analysis_id,
                timestamp=datetime.now(),
                strategy_performance=strategy_performance,
                strategy_rankings=integrated_results['rankings'],
                performance_differences=integrated_results['differences'],
                regime_analysis=regime_analysis or {},
                regime_performance=regime_analysis.get('regime_performance', {}) if regime_analysis else {},
                regime_transitions=regime_analysis.get('transitions', []) if regime_analysis else [],
                stock_selection_effects=stock_selection_analysis or {},
                sector_analysis=stock_selection_analysis.get('sector_analysis', {}) if stock_selection_analysis else {},
                correlation_analysis=correlation_analysis or {},
                overall_recommendations=integrated_results['recommendations'],
                risk_adjusted_rankings=integrated_results['risk_adjusted_rankings'],
                optimization_suggestions=integrated_results['optimization_suggestions'],
                analysis_period=(
                    processed_data.index.min(), 
                    processed_data.index.max()
                ),
                data_quality_score=self._calculate_data_quality_score(processed_data),
                confidence_level=integrated_results['confidence_level'],
                analysis_mode=analysis_mode
            )
            
            # パフォーマンス更新
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(True, processing_time)
            
            # 結果保存
            self.analysis_history.append(result)
            
            logger.info(f"比較分析完了: {analysis_id} ({processing_time:.2f}秒)")
            return result
            
        except Exception as e:
            logger.error(f"比較分析エラー: {e}")
            self._update_performance_metrics(False, 0)
            raise

    def _preprocess_data(
        self, 
        data: pd.DataFrame, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """データ前処理"""
        try:
            # データコピー
            processed_data = data.copy()
            
            # 日付インデックス確認・変換
            if not isinstance(processed_data.index, pd.DatetimeIndex):
                if 'Date' in processed_data.columns:
                    processed_data.set_index('Date', inplace=True)
                elif 'date' in processed_data.columns:
                    processed_data.set_index('date', inplace=True)
                processed_data.index = pd.to_datetime(processed_data.index)
            
            # 期間フィルタリング
            if start_date:
                processed_data = processed_data[processed_data.index >= start_date]
            if end_date:
                processed_data = processed_data[processed_data.index <= end_date]
            
            # 最小データポイント確認
            if len(processed_data) < self.config.min_data_points:
                raise ValueError(f"データポイント不足: {len(processed_data)} < {self.config.min_data_points}")
            
            # 欠損値処理
            processed_data = processed_data.fillna(method='ffill').fillna(0)
            
            # 基本指標計算
            processed_data = self._calculate_basic_indicators(processed_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"データ前処理エラー: {e}")
            raise

    def _calculate_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """基本指標計算"""
        try:
            # リターン計算
            if 'Close' in data.columns:
                data['Daily_Return'] = data['Close'].pct_change()
                data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()
            
            # ボラティリティ計算
            if 'Daily_Return' in data.columns:
                data['Volatility_20'] = data['Daily_Return'].rolling(20).std()
                data['Volatility_60'] = data['Daily_Return'].rolling(60).std()
            
            # 移動平均
            if 'Close' in data.columns:
                data['MA_20'] = data['Close'].rolling(20).mean()
                data['MA_60'] = data['Close'].rolling(60).mean()
            
            return data
            
        except Exception as e:
            logger.error(f"基本指標計算エラー: {e}")
            return data

    def _analyze_strategy_performance(
        self, 
        data: pd.DataFrame, 
        strategies: List[str], 
        analysis_mode: str
    ) -> Dict[str, Dict[str, float]]:
        """戦略パフォーマンス分析"""
        try:
            strategy_performance = {}
            
            for strategy in strategies:
                # 戦略列確認
                strategy_columns = [col for col in data.columns if strategy.lower() in col.lower()]
                if not strategy_columns:
                    logger.warning(f"戦略 {strategy} の列が見つかりません")
                    continue
                
                # パフォーマンス指標計算
                performance_metrics = self._calculate_strategy_metrics(
                    data, strategy_columns, analysis_mode
                )
                
                strategy_performance[strategy] = performance_metrics
            
            return strategy_performance
            
        except Exception as e:
            logger.error(f"戦略パフォーマンス分析エラー: {e}")
            return {}

    def _calculate_strategy_metrics(
        self, 
        data: pd.DataFrame, 
        strategy_columns: List[str], 
        analysis_mode: str
    ) -> Dict[str, float]:
        """戦略指標計算"""
        try:
            metrics = {}
            
            # 基本メトリクス
            if len(strategy_columns) > 0:
                main_column = strategy_columns[0]
                
                if main_column in data.columns:
                    strategy_data = data[main_column].dropna()
                    
                    if len(strategy_data) > 0:
                        # リターン系指標
                        metrics['total_return'] = strategy_data.iloc[-1] / strategy_data.iloc[0] - 1 if len(strategy_data) > 1 else 0
                        metrics['annual_return'] = metrics['total_return'] * (252 / len(strategy_data))
                        
                        # リスク指標
                        returns = strategy_data.pct_change().dropna()
                        if len(returns) > 0:
                            metrics['volatility'] = returns.std() * np.sqrt(252)
                            metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
                            metrics['max_drawdown'] = self._calculate_max_drawdown(strategy_data)
                        
                        # 詳細モード用追加指標
                        if analysis_mode == "comprehensive" or analysis_mode == "deep_dive":
                            metrics['skewness'] = returns.skew() if len(returns) > 0 else 0
                            metrics['kurtosis'] = returns.kurtosis() if len(returns) > 0 else 0
                            metrics['var_95'] = returns.quantile(0.05) if len(returns) > 0 else 0
                            metrics['calmar_ratio'] = abs(metrics['annual_return'] / metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"戦略指標計算エラー: {e}")
            return {}

    def _calculate_max_drawdown(self, data: pd.Series) -> float:
        """最大ドローダウン計算"""
        try:
            cumulative = (1 + data.pct_change()).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            return drawdown.min()
        except Exception:
            return 0.0

    def _analyze_market_regimes(
        self, 
        data: pd.DataFrame, 
        strategies: List[str], 
        analysis_mode: str
    ) -> Dict[str, Any]:
        """市場レジーム分析"""
        try:
            if not self.switching_analyzer:
                logger.warning("戦略切替アナライザーが利用できません")
                return self._fallback_regime_analysis(data, strategies)
            
            # 既存システム統合分析
            switching_analysis = self.switching_analyzer.analyze_switching_performance(
                data=data,
                strategies=strategies,
                analysis_period=(data.index.min(), data.index.max())
            )
            
            # レジーム分析結果変換
            regime_analysis = {
                'regime_performance': switching_analysis.regime_analysis,
                'transitions': self._extract_regime_transitions(switching_analysis),
                'regime_stability': self._calculate_regime_stability(data),
                'optimal_strategies_by_regime': self._identify_optimal_strategies_by_regime(
                    data, strategies, switching_analysis
                )
            }
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"市場レジーム分析エラー: {e}")
            return self._fallback_regime_analysis(data, strategies)

    def _fallback_regime_analysis(self, data: pd.DataFrame, strategies: List[str]) -> Dict[str, Any]:
        """フォールバック レジーム分析"""
        try:
            # シンプルなレジーム検出
            if 'Daily_Return' not in data.columns:
                return {}
            
            returns = data['Daily_Return'].dropna()
            
            # ボラティリティベースレジーム分類
            vol_threshold_high = returns.std() * 1.5
            vol_threshold_low = returns.std() * 0.5
            
            regimes = []
            for i, ret in enumerate(returns):
                vol = returns.iloc[max(0, i-20):i+1].std()
                if vol > vol_threshold_high:
                    regimes.append('high_volatility')
                elif vol < vol_threshold_low:
                    regimes.append('low_volatility')
                else:
                    regimes.append('normal')
            
            regime_analysis = {
                'regime_performance': {
                    'high_volatility': {'count': regimes.count('high_volatility')},
                    'low_volatility': {'count': regimes.count('low_volatility')},
                    'normal': {'count': regimes.count('normal')}
                },
                'transitions': [],
                'regime_stability': 0.7,  # デフォルト値
                'optimal_strategies_by_regime': {}
            }
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"フォールバック レジーム分析エラー: {e}")
            return {}

    def _analyze_stock_selection_effects(
        self, 
        data: pd.DataFrame, 
        strategies: List[str], 
        analysis_mode: str
    ) -> Dict[str, Any]:
        """株式選択効果分析"""
        try:
            selection_effects = {}
            
            # 戦略別選択効果
            for strategy in strategies:
                strategy_columns = [col for col in data.columns if strategy.lower() in col.lower()]
                if strategy_columns:
                    effects = self._calculate_selection_effect(data, strategy_columns[0])
                    selection_effects[strategy] = effects
            
            # セクター分析
            sector_analysis = self._analyze_sector_effects(data, strategies)
            
            # 結果統合
            result = {
                'strategy_selection_effects': selection_effects,
                'sector_analysis': sector_analysis,
                'overall_selection_quality': self._calculate_overall_selection_quality(selection_effects)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"株式選択効果分析エラー: {e}")
            return {}

    def _calculate_selection_effect(self, data: pd.DataFrame, strategy_column: str) -> Dict[str, float]:
        """選択効果計算"""
        try:
            if strategy_column not in data.columns:
                return {}
            
            strategy_data = data[strategy_column].dropna()
            
            # ベンチマーク比較（市場全体との比較）
            if 'Close' in data.columns:
                market_return = data['Close'].pct_change().dropna()
                strategy_return = strategy_data.pct_change().dropna()
                
                # 共通期間抽出
                common_index = market_return.index.intersection(strategy_return.index)
                if len(common_index) > 0:
                    market_common = market_return.loc[common_index]
                    strategy_common = strategy_return.loc[common_index]
                    
                    # 選択効果指標
                    excess_return = strategy_common.mean() - market_common.mean()
                    tracking_error = (strategy_common - market_common).std()
                    information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
                    
                    return {
                        'excess_return': excess_return * 252,  # 年換算
                        'tracking_error': tracking_error * np.sqrt(252),
                        'information_ratio': information_ratio,
                        'hit_rate': (strategy_common > market_common).mean()
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"選択効果計算エラー: {e}")
            return {}

    def _analyze_correlations(
        self, 
        data: pd.DataFrame, 
        strategies: List[str], 
        analysis_mode: str
    ) -> Dict[str, Any]:
        """相関分析"""
        try:
            strategy_data = {}
            
            # 戦略データ収集
            for strategy in strategies:
                strategy_columns = [col for col in data.columns if strategy.lower() in col.lower()]
                if strategy_columns:
                    strategy_data[strategy] = data[strategy_columns[0]].dropna()
            
            if len(strategy_data) < 2:
                return {}
            
            # 相関行列計算
            correlation_df = pd.DataFrame(strategy_data).corr()
            
            # 高相関ペア特定
            high_correlations = []
            for i in range(len(correlation_df.columns)):
                for j in range(i+1, len(correlation_df.columns)):
                    corr_value = correlation_df.iloc[i, j]
                    if abs(corr_value) > self.config.correlation_threshold:
                        high_correlations.append({
                            'strategy_1': correlation_df.columns[i],
                            'strategy_2': correlation_df.columns[j],
                            'correlation': corr_value
                        })
            
            result = {
                'correlation_matrix': correlation_df.to_dict(),
                'high_correlations': high_correlations,
                'diversification_score': self._calculate_diversification_score(correlation_df),
                'principal_components': self._perform_pca_analysis(strategy_data) if analysis_mode == "deep_dive" else {}
            }
            
            return result
            
        except Exception as e:
            logger.error(f"相関分析エラー: {e}")
            return {}

    def _integrate_analysis_results(
        self, 
        strategy_performance: Dict[str, Dict[str, float]], 
        regime_analysis: Optional[Dict[str, Any]], 
        stock_selection_analysis: Optional[Dict[str, Any]], 
        correlation_analysis: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析結果統合"""
        try:
            # 戦略ランキング生成
            rankings = self._generate_strategy_rankings(strategy_performance)
            
            # パフォーマンス差分計算
            differences = self._calculate_performance_differences(strategy_performance)
            
            # 推奨事項生成
            recommendations = self._generate_recommendations(
                strategy_performance, regime_analysis, 
                stock_selection_analysis, correlation_analysis
            )
            
            # リスク調整ランキング
            risk_adjusted_rankings = self._generate_risk_adjusted_rankings(strategy_performance)
            
            # 最適化提案
            optimization_suggestions = self._generate_optimization_suggestions(
                strategy_performance, correlation_analysis
            )
            
            # 信頼度レベル計算
            confidence_level = self._calculate_confidence_level(
                strategy_performance, regime_analysis, correlation_analysis
            )
            
            return {
                'rankings': rankings,
                'differences': differences,
                'recommendations': recommendations,
                'risk_adjusted_rankings': risk_adjusted_rankings,
                'optimization_suggestions': optimization_suggestions,
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            logger.error(f"分析結果統合エラー: {e}")
            return {
                'rankings': {},
                'differences': {},
                'recommendations': ["分析結果統合中にエラーが発生しました"],
                'risk_adjusted_rankings': {},
                'optimization_suggestions': [],
                'confidence_level': 0.5
            }

    def _generate_strategy_rankings(self, strategy_performance: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """戦略ランキング生成"""
        try:
            rankings = {}
            
            # 複数指標でのランキング
            metrics = ['total_return', 'sharpe_ratio', 'annual_return']
            
            for metric in metrics:
                if all(metric in perf for perf in strategy_performance.values()):
                    sorted_strategies = sorted(
                        strategy_performance.keys(),
                        key=lambda s: strategy_performance[s].get(metric, 0),
                        reverse=True
                    )
                    rankings[metric] = sorted_strategies
            
            return rankings
            
        except Exception as e:
            logger.error(f"戦略ランキング生成エラー: {e}")
            return {}

    def _calculate_performance_differences(self, strategy_performance: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """パフォーマンス差分計算"""
        try:
            if len(strategy_performance) < 2:
                return {}
            
            strategies = list(strategy_performance.keys())
            differences = {}
            
            # 上位戦略と他戦略の差分
            for metric in ['total_return', 'sharpe_ratio']:
                if all(metric in perf for perf in strategy_performance.values()):
                    values = [strategy_performance[s].get(metric, 0) for s in strategies]
                    differences[f'{metric}_range'] = max(values) - min(values)
                    differences[f'{metric}_std'] = np.std(values)
            
            return differences
            
        except Exception as e:
            logger.error(f"パフォーマンス差分計算エラー: {e}")
            return {}

    def _generate_recommendations(
        self, 
        strategy_performance: Dict[str, Dict[str, float]], 
        regime_analysis: Optional[Dict[str, Any]], 
        stock_selection_analysis: Optional[Dict[str, Any]], 
        correlation_analysis: Optional[Dict[str, Any]]
    ) -> List[str]:
        """推奨事項生成"""
        try:
            recommendations = []
            
            # パフォーマンスベース推奨
            if strategy_performance:
                best_strategy = max(
                    strategy_performance.keys(),
                    key=lambda s: strategy_performance[s].get('sharpe_ratio', 0)
                )
                recommendations.append(f"最適戦略: {best_strategy} (シャープレシオ基準)")
            
            # 相関ベース推奨
            if correlation_analysis and 'high_correlations' in correlation_analysis:
                high_corr_count = len(correlation_analysis['high_correlations'])
                if high_corr_count > 0:
                    recommendations.append(f"高相関戦略ペア {high_corr_count} 組を発見。分散化改善を推奨")
            
            # レジームベース推奨
            if regime_analysis and 'optimal_strategies_by_regime' in regime_analysis:
                recommendations.append("市場レジーム別最適戦略の適用を推奨")
            
            # 基本推奨事項
            if not recommendations:
                recommendations.append("継続的なモニタリングと定期的な再分析を推奨")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"推奨事項生成エラー: {e}")
            return ["分析エラーのため推奨事項を生成できませんでした"]

    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """データ品質スコア計算"""
        try:
            total_points = len(data)
            missing_points = data.isnull().sum().sum()
            completeness = 1 - (missing_points / (total_points * len(data.columns)))
            
            # その他の品質指標も含める場合はここで追加
            
            return max(0.0, min(1.0, completeness))
            
        except Exception:
            return 0.5

    def _update_performance_metrics(self, success: bool, processing_time: float):
        """パフォーマンスメトリクス更新"""
        self.performance_metrics['total_analyses'] += 1
        if success:
            self.performance_metrics['successful_analyses'] += 1
        
        # 平均処理時間更新
        total = self.performance_metrics['total_analyses']
        avg = self.performance_metrics['average_processing_time']
        self.performance_metrics['average_processing_time'] = (avg * (total - 1) + processing_time) / total

    # 以下、追加メソッドのスタブ実装
    def _extract_regime_transitions(self, switching_analysis) -> List[Dict[str, Any]]:
        """レジーム遷移抽出"""
        return []

    def _calculate_regime_stability(self, data: pd.DataFrame) -> float:
        """レジーム安定性計算"""
        return 0.75

    def _identify_optimal_strategies_by_regime(self, data: pd.DataFrame, strategies: List[str], switching_analysis) -> Dict[str, str]:
        """レジーム別最適戦略特定"""
        return {}

    def _analyze_sector_effects(self, data: pd.DataFrame, strategies: List[str]) -> Dict[str, Any]:
        """セクター効果分析"""
        return {}

    def _calculate_overall_selection_quality(self, selection_effects: Dict[str, Dict[str, float]]) -> float:
        """全体選択品質計算"""
        return 0.7

    def _calculate_diversification_score(self, correlation_df: pd.DataFrame) -> float:
        """分散化スコア計算"""
        try:
            avg_correlation = correlation_df.values[np.triu_indices_from(correlation_df.values, k=1)].mean()
            return 1 - abs(avg_correlation)
        except Exception:
            return 0.5

    def _perform_pca_analysis(self, strategy_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """主成分分析実行"""
        return {}

    def _generate_risk_adjusted_rankings(self, strategy_performance: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """リスク調整ランキング生成"""
        try:
            if 'sharpe_ratio' in next(iter(strategy_performance.values()), {}):
                sorted_strategies = sorted(
                    strategy_performance.keys(),
                    key=lambda s: strategy_performance[s].get('sharpe_ratio', 0),
                    reverse=True
                )
                return {'risk_adjusted': sorted_strategies}
            return {}
        except Exception:
            return {}

    def _generate_optimization_suggestions(
        self, 
        strategy_performance: Dict[str, Dict[str, float]], 
        correlation_analysis: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """最適化提案生成"""
        suggestions = []
        
        try:
            # 基本的な最適化提案
            if strategy_performance:
                suggestions.append({
                    'type': 'performance_optimization',
                    'description': '上位戦略への重み集中を検討',
                    'priority': 'medium'
                })
            
            if correlation_analysis and correlation_analysis.get('high_correlations'):
                suggestions.append({
                    'type': 'diversification',
                    'description': '高相関戦略の統合または除外を検討',
                    'priority': 'high'
                })
        
        except Exception as e:
            logger.error(f"最適化提案生成エラー: {e}")
        
        return suggestions

    def _calculate_confidence_level(
        self, 
        strategy_performance: Dict[str, Dict[str, float]], 
        regime_analysis: Optional[Dict[str, Any]], 
        correlation_analysis: Optional[Dict[str, Any]]
    ) -> float:
        """信頼度レベル計算"""
        try:
            confidence_factors = []
            
            # データ量ベース信頼度
            if strategy_performance:
                confidence_factors.append(min(1.0, len(strategy_performance) / 5))
            
            # 分析完了度ベース信頼度
            analysis_completeness = sum([
                1 if strategy_performance else 0,
                1 if regime_analysis else 0,
                1 if correlation_analysis else 0
            ]) / 3
            confidence_factors.append(analysis_completeness)
            
            return np.mean(confidence_factors) if confidence_factors else 0.5
            
        except Exception:
            return 0.5

    def get_analysis_summary(self) -> Dict[str, Any]:
        """分析サマリー取得"""
        return {
            'total_analyses': len(self.analysis_history),
            'performance_metrics': self.performance_metrics,
            'latest_analysis': self.analysis_history[-1].analysis_id if self.analysis_history else None,
            'configuration': {
                'analysis_mode': self.config.analysis_mode,
                'lookback_period': self.config.lookback_period,
                'output_format': self.config.output_format
            }
        }

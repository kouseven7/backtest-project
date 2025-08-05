"""
拡張ウォークフォワードエグゼキューター：市場分類対応のウォークフォワードテスト実行
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import os

from .classification_integration import ClassificationIntegration
from .market_aware_analyzer import MarketAwareAnalyzer
from ..walkforward_executor import WalkforwardExecutor
from ..walkforward_result_analyzer import WalkforwardResultAnalyzer

logger = logging.getLogger(__name__)


class EnhancedWalkforwardExecutor:
    """市場分類対応の拡張ウォークフォワードエグゼキューター"""
    
    def __init__(self, config_path: str, enhanced_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config_path: 基本設定ファイルのパス
            enhanced_config: 拡張設定（市場分類など）
        """
        self.config_path = config_path
        self.base_config = self._load_base_config(config_path)
        self.enhanced_config = enhanced_config or {}
        
        # 分類統合システム
        self.classification_integration = ClassificationIntegration(config_path)
        
        # 市場対応分析器
        self.market_analyzer = MarketAwareAnalyzer(self.classification_integration)
        
        # 基本ウォークフォワードエグゼキューター
        self.base_executor = WalkforwardExecutor(config_path)
        
        # 結果格納
        self.execution_results: Dict[str, Any] = {}
        self.market_classifications: Dict[str, Any] = {}
        self.enhanced_analysis: Dict[str, Any] = {}
        
    def _load_base_config(self, config_path: str) -> Dict[str, Any]:
        """基本設定の読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def execute_enhanced_walkforward(self, 
                                   data: pd.DataFrame,
                                   mode: str = "hybrid") -> Dict[str, Any]:
        """拡張ウォークフォワードテストの実行"""
        logger.info("Starting enhanced walkforward execution with market classification")
        
        try:
            # 1. 基本設定の拡張
            enhanced_config = self.classification_integration.enhance_walkforward_config(self.base_config)
            
            # 2. 市場分類の実行
            logger.info("Performing market classification for all test scenarios")
            self.market_classifications = self._classify_all_scenarios(data, mode)
            
            # 3. 拡張されたシナリオでウォークフォワードテスト実行
            logger.info("Executing walkforward tests with market-aware scenarios")
            self.execution_results = self._execute_market_aware_walkforward(data, enhanced_config)
            
            # 4. 市場対応分析の実行
            logger.info("Performing market-aware analysis")
            self.enhanced_analysis = self.market_analyzer.analyze_strategy_performance_by_market(
                self.execution_results, self.market_classifications
            )
            
            # 5. 結果の統合
            final_results = self._integrate_results()
            
            logger.info("Enhanced walkforward execution completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"Enhanced walkforward execution failed: {e}")
            return {
                'error': str(e),
                'execution_results': self.execution_results,
                'market_classifications': self.market_classifications,
                'enhanced_analysis': self.enhanced_analysis
            }
    
    def _classify_all_scenarios(self, data: pd.DataFrame, mode: str) -> Dict[str, Any]:
        """全シナリオの市場分類"""
        classifications = {}
        
        try:
            test_scenarios = self.base_config.get('test_scenarios', [])
            symbols = self.base_config.get('symbols', [])
            
            for scenario in test_scenarios:
                scenario_name = scenario.get('name', 'Unknown')
                start_date = scenario.get('start_date')
                end_date = scenario.get('end_date')
                
                logger.info(f"Classifying market for scenario: {scenario_name}")
                
                scenario_classifications = {}
                
                for symbol in symbols:
                    try:
                        # シンボルデータの抽出
                        symbol_data = self.classification_integration._extract_symbol_data(data, symbol)
                        
                        if not symbol_data.empty:
                            # 市場分類の実行
                            classification_result = self.classification_integration.classify_market_for_period(
                                symbol_data, symbol, start_date, end_date, mode
                            )
                            
                            scenario_classifications[symbol] = {
                                'simple_condition': classification_result.simple_condition.value,
                                'detailed_condition': classification_result.detailed_condition.value,
                                'confidence': classification_result.confidence,
                                'metrics': {
                                    'trend_strength': classification_result.metrics.trend_strength,
                                    'volatility': classification_result.metrics.volatility,
                                    'momentum': classification_result.metrics.momentum,
                                    'volume_trend': classification_result.metrics.volume_trend,
                                    'price_momentum': classification_result.metrics.price_momentum,
                                    'risk_level': classification_result.metrics.risk_level
                                }
                            }
                        else:
                            logger.warning(f"No data found for symbol {symbol} in scenario {scenario_name}")
                            
                    except Exception as e:
                        logger.error(f"Classification error for {symbol} in {scenario_name}: {e}")
                        continue
                
                # シナリオ全体の市場状況判定
                if scenario_classifications:
                    overall_condition = self.classification_integration._determine_overall_market_condition(
                        scenario_classifications
                    )
                    scenario_classifications['overall'] = overall_condition
                
                classifications[scenario_name] = scenario_classifications
            
        except Exception as e:
            logger.error(f"Error in scenario classification: {e}")
            classifications['error'] = str(e)
        
        return classifications
    
    def _execute_market_aware_walkforward(self, 
                                        data: pd.DataFrame, 
                                        enhanced_config: Dict[str, Any]) -> Dict[str, Any]:
        """市場分類を考慮したウォークフォワードテスト実行"""
        try:
            # 基本的なウォークフォワードテストを実行
            # （実際の実装では既存のwalkforward_executorを使用）
            base_results = self._execute_base_walkforward(data)
            
            # 市場分類に基づく戦略推奨の適用
            enhanced_results = self._apply_market_based_recommendations(base_results)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in market-aware walkforward execution: {e}")
            return {'error': str(e)}
    
    def _execute_base_walkforward(self, data: pd.DataFrame) -> Dict[str, Any]:
        """基本ウォークフォワードテストの実行"""
        try:
            # 既存のウォークフォワードエグゼキューターを使用
            # ここでは簡略化した実装
            results = {
                'scenarios': {},
                'strategies': {},
                'summary': {}
            }
            
            strategies = self.base_config.get('strategies', [])
            test_scenarios = self.base_config.get('test_scenarios', [])
            
            for scenario in test_scenarios:
                scenario_name = scenario.get('name', 'Unknown')
                scenario_results = {}
                
                for strategy_config in strategies:
                    strategy_name = strategy_config.get('name', 'Unknown')
                    
                    # 戦略実行のシミュレーション（実際の実装が必要）
                    strategy_result = self._simulate_strategy_execution(
                        data, strategy_name, scenario
                    )
                    
                    scenario_results[strategy_name] = strategy_result
                
                results['scenarios'][scenario_name] = scenario_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error in base walkforward execution: {e}")
            return {'error': str(e)}
    
    def _simulate_strategy_execution(self, 
                                   data: pd.DataFrame, 
                                   strategy_name: str, 
                                   scenario: Dict[str, Any]) -> Dict[str, Any]:
        """戦略実行のシミュレーション（簡略化）"""
        # 実際の実装では戦略クラスを呼び出す
        return {
            'total_trades': np.random.randint(10, 100),
            'winning_trades': np.random.randint(5, 60),
            'total_return': np.random.uniform(-0.1, 0.3),
            'max_drawdown': np.random.uniform(0.0, 0.2),
            'sharpe_ratio': np.random.uniform(-1.0, 2.0),
            'trades': [
                {'return': np.random.uniform(-0.05, 0.1)} 
                for _ in range(np.random.randint(5, 50))
            ]
        }
    
    def _apply_market_based_recommendations(self, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """市場分類に基づく推奨事項の適用"""
        enhanced_results = base_results.copy()
        
        try:
            # 各シナリオに対して市場に基づく調整を適用
            for scenario_name, scenario_results in base_results.get('scenarios', {}).items():
                if scenario_name in self.market_classifications:
                    market_info = self.market_classifications[scenario_name]
                    overall_condition = market_info.get('overall', {})
                    
                    # 市場状況に基づく戦略調整
                    adjusted_results = self._adjust_strategy_results_for_market(
                        scenario_results, overall_condition
                    )
                    
                    enhanced_results['scenarios'][scenario_name] = adjusted_results
                    
                    # 市場情報を結果に追加
                    enhanced_results['scenarios'][scenario_name]['market_info'] = market_info
            
        except Exception as e:
            logger.error(f"Error applying market-based recommendations: {e}")
            enhanced_results['recommendation_error'] = str(e)
        
        return enhanced_results
    
    def _adjust_strategy_results_for_market(self, 
                                          strategy_results: Dict[str, Any], 
                                          market_condition: Dict[str, Any]) -> Dict[str, Any]:
        """市場状況に基づく戦略結果の調整"""
        adjusted_results = strategy_results.copy()
        
        try:
            overall_simple = market_condition.get('overall_simple_condition', 'sideways')
            overall_confidence = market_condition.get('overall_confidence', 0.5)
            
            # 市場状況に基づく調整係数
            adjustment_factors = {
                'trending_bull': {'return_mult': 1.2, 'risk_mult': 0.9},
                'trending_bear': {'return_mult': 0.8, 'risk_mult': 1.3},
                'sideways': {'return_mult': 1.0, 'risk_mult': 1.0},
                'volatile': {'return_mult': 0.9, 'risk_mult': 1.5},
                'recovery': {'return_mult': 1.1, 'risk_mult': 1.1}
            }
            
            factor = adjustment_factors.get(overall_simple, {'return_mult': 1.0, 'risk_mult': 1.0})
            
            # 信頼度に基づく調整の強さ
            confidence_mult = overall_confidence
            
            for strategy_name, strategy_data in adjusted_results.items():
                if isinstance(strategy_data, dict) and 'total_return' in strategy_data:
                    # リターンの調整
                    original_return = strategy_data['total_return']
                    adjusted_return = original_return * (
                        1 + (factor['return_mult'] - 1) * confidence_mult
                    )
                    strategy_data['adjusted_total_return'] = adjusted_return
                    
                    # リスク指標の調整
                    if 'max_drawdown' in strategy_data:
                        original_dd = strategy_data['max_drawdown']
                        adjusted_dd = original_dd * (
                            1 + (factor['risk_mult'] - 1) * confidence_mult
                        )
                        strategy_data['adjusted_max_drawdown'] = adjusted_dd
                    
                    # 調整情報の記録
                    strategy_data['market_adjustment'] = {
                        'market_condition': overall_simple,
                        'confidence': overall_confidence,
                        'return_multiplier': factor['return_mult'],
                        'risk_multiplier': factor['risk_mult']
                    }
        
        except Exception as e:
            logger.error(f"Error adjusting strategy results: {e}")
            adjusted_results['adjustment_error'] = str(e)
        
        return adjusted_results
    
    def _integrate_results(self) -> Dict[str, Any]:
        """全結果の統合"""
        integrated_results = {
            'execution_timestamp': datetime.now().isoformat(),
            'config_used': {
                'base_config': self.base_config,
                'enhanced_config': self.enhanced_config
            },
            'market_classifications': self.market_classifications,
            'execution_results': self.execution_results,
            'enhanced_analysis': self.enhanced_analysis,
            'summary': {
                'total_scenarios': len(self.market_classifications),
                'classification_mode': self.enhanced_config.get('market_classification', {}).get('mode', 'hybrid'),
                'analysis_completed': bool(self.enhanced_analysis)
            }
        }
        
        return integrated_results
    
    def save_results(self, output_dir: str):
        """結果の保存"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. 統合結果の保存
            results_path = os.path.join(output_dir, f'enhanced_walkforward_results_{timestamp}.json')
            integrated_results = self._integrate_results()
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(integrated_results, f, indent=2, ensure_ascii=False)
            
            # 2. 市場分析レポートの生成
            report_path = os.path.join(output_dir, f'market_analysis_report_{timestamp}.md')
            report_content = self.market_analyzer.generate_market_analysis_report(report_path)
            
            # 3. 市場分類結果のエクスポート
            classification_path = os.path.join(output_dir, f'market_classifications_{timestamp}.csv')
            self.classification_integration.export_classifications(classification_path)
            
            # 4. 可視化の保存
            viz_path = os.path.join(output_dir, f'market_performance_comparison_{timestamp}.png')
            self.market_analyzer.plot_market_performance_comparison(viz_path)
            
            logger.info(f"Enhanced walkforward results saved to {output_dir}")
            
            return {
                'results_file': results_path,
                'report_file': report_path,
                'classification_file': classification_path,
                'visualization_file': viz_path
            }
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return {'error': str(e)}
    
    def get_strategy_recommendations_for_current_market(self, 
                                                      current_data: pd.DataFrame,
                                                      symbols: List[str]) -> Dict[str, Any]:
        """現在の市場状況に基づく戦略推奨"""
        try:
            current_classifications = {}
            
            # 各シンボルの現在の市場分類
            for symbol in symbols:
                symbol_data = self.classification_integration._extract_symbol_data(current_data, symbol)
                if not symbol_data.empty:
                    classification = self.classification_integration.classifier.classify(
                        symbol_data, symbol, mode="hybrid"
                    )
                    current_classifications[symbol] = classification
            
            # 全体的な推奨事項の生成
            overall_recommendations = {
                'timestamp': datetime.now().isoformat(),
                'symbol_classifications': {},
                'strategy_recommendations': {},
                'risk_recommendations': {},
                'market_summary': {}
            }
            
            # シンボル別分類情報
            for symbol, classification in current_classifications.items():
                overall_recommendations['symbol_classifications'][symbol] = classification.to_dict()
                
                # 戦略推奨の取得
                strategy_rec = self.classification_integration.get_strategy_recommendations(classification)
                overall_recommendations['strategy_recommendations'][symbol] = strategy_rec
            
            # 全体的なマーケットサマリー
            if current_classifications:
                simple_conditions = [c.simple_condition.value for c in current_classifications.values()]
                from collections import Counter
                condition_counter = Counter(simple_conditions)
                
                overall_recommendations['market_summary'] = {
                    'dominant_condition': condition_counter.most_common(1)[0][0] if condition_counter else 'unknown',
                    'condition_distribution': dict(condition_counter),
                    'consensus_level': max(condition_counter.values()) / len(simple_conditions) if simple_conditions else 0.0,
                    'analyzed_symbols': len(current_classifications)
                }
            
            return overall_recommendations
            
        except Exception as e:
            logger.error(f"Error generating current market recommendations: {e}")
            return {'error': str(e)}
    
    def compare_with_historical_performance(self, 
                                          current_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """現在の推奨と過去のパフォーマンスとの比較"""
        try:
            if not self.enhanced_analysis:
                return {'error': 'No historical analysis available'}
            
            comparison = {
                'current_market_condition': current_recommendations.get('market_summary', {}),
                'historical_performance': {},
                'recommendations': {}
            }
            
            # 現在の主要な市場状況を取得
            dominant_condition = current_recommendations.get('market_summary', {}).get('dominant_condition')
            
            if dominant_condition and 'by_simple_condition' in self.enhanced_analysis:
                historical_perf = self.enhanced_analysis['by_simple_condition'].get(dominant_condition, {})
                comparison['historical_performance'] = historical_perf
                
                # 推奨戦略の特定
                if 'best_strategy' in historical_perf:
                    best_strategy = historical_perf['best_strategy']
                    comparison['recommendations']['primary_strategy'] = best_strategy
                
                # リスク調整の推奨
                avg_return = historical_perf.get('avg_return', 0)
                max_drawdown = historical_perf.get('max_drawdown', 0)
                
                if avg_return > 0.05:  # 5%以上のリターン
                    comparison['recommendations']['position_sizing'] = 'aggressive'
                elif avg_return < -0.02:  # -2%以下のリターン
                    comparison['recommendations']['position_sizing'] = 'conservative'
                else:
                    comparison['recommendations']['position_sizing'] = 'moderate'
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error in historical comparison: {e}")
            return {'error': str(e)}

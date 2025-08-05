"""
分類統合モジュール：既存システムと市場分類の統合
"""
import pandas as pd
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from ..market_classification.market_classifier import MarketClassifier
from ..market_classification.market_conditions import (
    SimpleMarketCondition, DetailedMarketCondition, ClassificationResult, MarketMetrics
)

logger = logging.getLogger(__name__)


class ClassificationIntegration:
    """市場分類と既存システムの統合クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.classifier = MarketClassifier()
        self.classification_cache: Dict[str, ClassificationResult] = {}
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """設定ファイルの読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 分類器のパラメータを設定
            classifier_config = config.get('market_classification', {})
            self.classifier = MarketClassifier(
                lookback_periods=classifier_config.get('lookback_periods', 20),
                volatility_threshold=classifier_config.get('volatility_threshold', 0.02),
                trend_threshold=classifier_config.get('trend_threshold', 0.001),
                confidence_threshold=classifier_config.get('confidence_threshold', 0.6)
            )
            
            logger.info(f"Loaded classification config from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            # デフォルト設定を使用
            self.classifier = MarketClassifier()
    
    def enhance_walkforward_config(self, original_config: Dict[str, Any]) -> Dict[str, Any]:
        """既存のウォークフォワード設定を拡張"""
        enhanced_config = original_config.copy()
        
        # 市場分類設定を追加
        enhanced_config['market_classification'] = {
            'enabled': True,
            'mode': 'hybrid',  # simple, detailed, hybrid
            'lookback_periods': 20,
            'volatility_threshold': 0.02,
            'trend_threshold': 0.001,
            'confidence_threshold': 0.6,
            'update_on_each_period': True,
            'cache_classifications': True
        }
        
        # 分類対応の分析設定を追加
        if 'analysis' not in enhanced_config:
            enhanced_config['analysis'] = {}
        
        enhanced_config['analysis']['market_classification_analysis'] = {
            'enabled': True,
            'generate_classification_report': True,
            'plot_classification_distribution': True,
            'analyze_strategy_performance_by_market': True,
            'export_classification_results': True
        }
        
        return enhanced_config
    
    def classify_market_for_period(self, data: pd.DataFrame, 
                                 symbol: str, 
                                 period_start: str, 
                                 period_end: str,
                                 mode: str = "hybrid") -> ClassificationResult:
        """特定期間の市場分類を実行"""
        cache_key = f"{symbol}_{period_start}_{period_end}_{mode}"
        
        # キャッシュチェック
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        try:
            # 期間データの抽出
            period_data = data[
                (data.index >= period_start) & 
                (data.index <= period_end)
            ].copy()
            
            if period_data.empty:
                logger.warning(f"No data found for {symbol} in period {period_start} to {period_end}")
                # デフォルト結果を返す
                result = ClassificationResult(
                    simple_condition=SimpleMarketCondition.SIDEWAYS,
                    detailed_condition=DetailedMarketCondition.NEUTRAL_SIDEWAYS,
                    confidence=0.1,
                    metrics=self.classifier.calculate_market_metrics(data, symbol),
                    timestamp=datetime.now().isoformat(),
                    symbol=symbol,
                    classification_reason={'error': 'No data in period'}
                )
            else:
                # 分類実行
                result = self.classifier.classify(period_data, symbol, mode)
                result.classification_reason['period'] = f"{period_start} to {period_end}"
            
            # キャッシュに保存
            self.classification_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Classification error for {symbol} in period {period_start}-{period_end}: {e}")
            # エラー時のデフォルト結果
            return ClassificationResult(
                simple_condition=SimpleMarketCondition.SIDEWAYS,
                detailed_condition=DetailedMarketCondition.NEUTRAL_SIDEWAYS,
                confidence=0.0,
                metrics=self.classifier.calculate_market_metrics(data.tail(20), symbol),
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                classification_reason={'error': str(e)}
            )
    
    def enhance_test_scenario(self, scenario: Dict[str, Any], 
                            data: pd.DataFrame, 
                            symbols: List[str]) -> Dict[str, Any]:
        """テストシナリオを市場分類で拡張"""
        enhanced_scenario = scenario.copy()
        
        # 期間情報の取得
        start_date = scenario['start_date']
        end_date = scenario['end_date']
        
        # 各シンボルの市場分類を実行
        market_classifications = {}
        for symbol in symbols:
            if symbol in data.columns or 'Close' in data.columns:
                symbol_data = self._extract_symbol_data(data, symbol)
                if not symbol_data.empty:
                    classification = self.classify_market_for_period(
                        symbol_data, symbol, start_date, end_date
                    )
                    market_classifications[symbol] = {
                        'simple_condition': classification.simple_condition.value,
                        'detailed_condition': classification.detailed_condition.value,
                        'confidence': classification.confidence,
                        'metrics': classification.metrics.__dict__
                    }
        
        # シナリオに分類情報を追加
        enhanced_scenario['market_classifications'] = market_classifications
        
        # 全体的な市場状況の判定
        if market_classifications:
            overall_classification = self._determine_overall_market_condition(market_classifications)
            enhanced_scenario['overall_market_condition'] = overall_classification
        
        return enhanced_scenario
    
    def get_strategy_recommendations(self, classification_result: ClassificationResult) -> Dict[str, Any]:
        """市場分類に基づく戦略推奨"""
        recommendations = {
            'primary_strategies': [],
            'secondary_strategies': [],
            'risk_adjustments': {},
            'parameter_suggestions': {}
        }
        
        simple_condition = classification_result.simple_condition
        detailed_condition = classification_result.detailed_condition
        confidence = classification_result.confidence
        
        # シンプル分類に基づく基本推奨
        if simple_condition == SimpleMarketCondition.TRENDING_BULL:
            recommendations['primary_strategies'] = ['MomentumInvestingStrategy', 'BreakoutStrategy']
            recommendations['secondary_strategies'] = ['VWAPBreakoutStrategy']
            recommendations['risk_adjustments'] = {'position_size_multiplier': 1.2}
            
        elif simple_condition == SimpleMarketCondition.TRENDING_BEAR:
            recommendations['primary_strategies'] = ['VWAPBounceStrategy']
            recommendations['secondary_strategies'] = ['GCStrategy']
            recommendations['risk_adjustments'] = {'position_size_multiplier': 0.8}
            
        elif simple_condition == SimpleMarketCondition.SIDEWAYS:
            recommendations['primary_strategies'] = ['VWAPBounceStrategy', 'GCStrategy']
            recommendations['secondary_strategies'] = ['VWAPBreakoutStrategy']
            recommendations['risk_adjustments'] = {'stop_loss_multiplier': 0.8}
            
        elif simple_condition == SimpleMarketCondition.VOLATILE:
            recommendations['primary_strategies'] = ['VWAPBounceStrategy']
            recommendations['secondary_strategies'] = ['GCStrategy']
            recommendations['risk_adjustments'] = {'position_size_multiplier': 0.6, 'stop_loss_multiplier': 1.5}
            
        elif simple_condition == SimpleMarketCondition.RECOVERY:
            recommendations['primary_strategies'] = ['MomentumInvestingStrategy', 'BreakoutStrategy']
            recommendations['secondary_strategies'] = ['VWAPBreakoutStrategy']
            recommendations['risk_adjustments'] = {'position_size_multiplier': 1.0}
        
        # 詳細分類による微調整
        if detailed_condition == DetailedMarketCondition.STRONG_BULL:
            recommendations['risk_adjustments']['position_size_multiplier'] = 1.5
        elif detailed_condition == DetailedMarketCondition.STRONG_BEAR:
            recommendations['risk_adjustments']['position_size_multiplier'] = 0.5
        
        # 信頼度による調整
        if confidence < 0.6:
            # 信頼度が低い場合は保守的な設定
            recommendations['risk_adjustments']['position_size_multiplier'] = \
                recommendations['risk_adjustments'].get('position_size_multiplier', 1.0) * 0.8
        
        return recommendations
    
    def _extract_symbol_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """データから特定シンボルのデータを抽出"""
        try:
            # マルチインデックスの場合
            if isinstance(data.columns, pd.MultiIndex):
                symbol_data = data.xs(symbol, axis=1, level=1)
            # シングルインデックスでシンボルが含まれる場合
            elif any(symbol in col for col in data.columns):
                symbol_cols = [col for col in data.columns if symbol in col]
                symbol_data = data[symbol_cols].copy()
                # カラム名を標準化
                new_columns = {}
                for col in symbol_cols:
                    if 'Close' in col or 'close' in col:
                        new_columns[col] = 'Close'
                    elif 'High' in col or 'high' in col:
                        new_columns[col] = 'High'
                    elif 'Low' in col or 'low' in col:
                        new_columns[col] = 'Low'
                    elif 'Volume' in col or 'volume' in col:
                        new_columns[col] = 'Volume'
                symbol_data = symbol_data.rename(columns=new_columns)
            # デフォルトのOHLCV列を使用
            else:
                required_cols = ['Close', 'High', 'Low', 'Volume']
                available_cols = [col for col in required_cols if col in data.columns]
                if available_cols:
                    symbol_data = data[available_cols].copy()
                else:
                    return pd.DataFrame()
            
            return symbol_data.dropna()
            
        except Exception as e:
            logger.error(f"Error extracting data for symbol {symbol}: {e}")
            return pd.DataFrame()
    
    def _determine_overall_market_condition(self, 
                                          market_classifications: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """複数シンボルから全体的な市場状況を判定"""
        simple_conditions = [
            cls['simple_condition'] for cls in market_classifications.values()
        ]
        detailed_conditions = [
            cls['detailed_condition'] for cls in market_classifications.values()
        ]
        confidences = [
            cls['confidence'] for cls in market_classifications.values()
        ]
        
        # 最も多い分類を全体分類とする
        from collections import Counter
        simple_counter = Counter(simple_conditions)
        detailed_counter = Counter(detailed_conditions)
        
        overall_simple = simple_counter.most_common(1)[0][0] if simple_counter else 'sideways'
        overall_detailed = detailed_counter.most_common(1)[0][0] if detailed_counter else 'neutral_sideways'
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return {
            'overall_simple_condition': overall_simple,
            'overall_detailed_condition': overall_detailed,
            'overall_confidence': overall_confidence,
            'consensus_level': simple_counter[overall_simple] / len(simple_conditions) if simple_conditions else 0.0,
            'symbol_count': len(market_classifications)
        }
    
    def export_classifications(self, output_path: str):
        """分類結果をファイルにエクスポート"""
        try:
            export_data = []
            for result in self.classification_cache.values():
                export_data.append(result.to_dict())
            
            export_df = pd.DataFrame(export_data)
            export_df.to_csv(output_path, index=False)
            logger.info(f"Classifications exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export classifications: {e}")
    
    def clear_cache(self):
        """分類キャッシュをクリア"""
        self.classification_cache.clear()
        logger.info("Classification cache cleared")

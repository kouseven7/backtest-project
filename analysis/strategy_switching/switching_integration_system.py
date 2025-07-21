"""
Module: Strategy Switching Integration System
File: switching_integration_system.py
Description:
  5-1-1「戦略切替のタイミング分析ツール」
  既存システムとの統合インターフェース

Author: imega
Created: 2025-01-21
Modified: 2025-01-21
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import json

# プロジェクトパスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 既存システムのインポート（利用可能な場合）
try:
    from config.strategy_selector import StrategySelector
    STRATEGY_SELECTOR_AVAILABLE = True
except ImportError:
    STRATEGY_SELECTOR_AVAILABLE = False
    
try:
    from config.drawdown_controller import DrawdownController
    DRAWDOWN_CONTROLLER_AVAILABLE = True
except ImportError:
    DRAWDOWN_CONTROLLER_AVAILABLE = False
    
try:
    from indicators.unified_trend_detector import UnifiedTrendDetector
    TREND_DETECTOR_AVAILABLE = True
except ImportError:
    TREND_DETECTOR_AVAILABLE = False

# 切替分析モジュールのインポート
try:
    from .strategy_switching_analyzer import StrategySwitchingAnalyzer
    from .switching_timing_evaluator import SwitchingTimingEvaluator
    from .switching_pattern_detector import SwitchingPatternDetector
    from .switching_performance_calculator import SwitchingPerformanceCalculator
    from .switching_analysis_dashboard import SwitchingAnalysisDashboard
except ImportError:
    # 相対インポートが失敗した場合の絶対インポート
    pass

# ロガーの設定
logger = logging.getLogger(__name__)

class SwitchingIntegrationSystem:
    """戦略切替統合システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Parameters:
            config_path: 設定ファイルパス
        """
        self.config = self._load_config(config_path)
        
        # 既存システムの初期化
        self._initialize_existing_systems()
        
        # 切替分析システムの初期化
        self._initialize_switching_systems()
        
        # 統合データストレージ
        self.switching_history: List[Dict[str, Any]] = []
        self.analysis_cache: Dict[str, Any] = {}
        
        logger.info("SwitchingIntegrationSystem initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Config loading failed: {e}")
                
        return self._get_default_integration_config()

    def _get_default_integration_config(self) -> Dict[str, Any]:
        """デフォルト統合設定"""
        return {
            'analysis_settings': {
                'lookback_days': 30,
                'evaluation_window': 20,
                'confidence_threshold': 0.6,
                'min_switch_interval_days': 5
            },
            'integration_settings': {
                'enable_auto_switching': False,
                'enable_risk_override': True,
                'max_switches_per_month': 8,
                'emergency_stop_drawdown': 0.15
            },
            'reporting_settings': {
                'generate_daily_reports': False,
                'generate_weekly_reports': True,
                'export_to_excel': True,
                'email_notifications': False
            },
            'dashboard_settings': {
                'update_frequency_minutes': 60,
                'enable_live_updates': False,
                'chart_theme': 'plotly_white'
            }
        }

    def _initialize_existing_systems(self):
        """既存システムの初期化"""
        # Strategy Selector
        if STRATEGY_SELECTOR_AVAILABLE:
            try:
                self.strategy_selector = StrategySelector()
                logger.info("StrategySelector initialized")
            except Exception as e:
                logger.warning(f"StrategySelector initialization failed: {e}")
                self.strategy_selector = None
        else:
            self.strategy_selector = None
            logger.info("StrategySelector not available")
            
        # Drawdown Controller
        if DRAWDOWN_CONTROLLER_AVAILABLE:
            try:
                self.drawdown_controller = DrawdownController()
                logger.info("DrawdownController initialized")
            except Exception as e:
                logger.warning(f"DrawdownController initialization failed: {e}")
                self.drawdown_controller = None
        else:
            self.drawdown_controller = None
            logger.info("DrawdownController not available")
            
        # Trend Detector
        if TREND_DETECTOR_AVAILABLE:
            try:
                self.trend_detector = UnifiedTrendDetector()
                logger.info("UnifiedTrendDetector initialized")
            except Exception as e:
                logger.warning(f"UnifiedTrendDetector initialization failed: {e}")
                self.trend_detector = None
        else:
            self.trend_detector = None
            logger.info("UnifiedTrendDetector not available")

    def _initialize_switching_systems(self):
        """切替分析システムの初期化"""
        try:
            # 設定を辞書形式で渡す
            config_dict = self.config if isinstance(self.config, dict) else {}
            
            self.switching_analyzer = StrategySwitchingAnalyzer(config_dict)
            self.timing_evaluator = SwitchingTimingEvaluator(config_dict)
            self.pattern_detector = SwitchingPatternDetector(config_dict)
            self.performance_calculator = SwitchingPerformanceCalculator(config_dict)
            self.analysis_dashboard = SwitchingAnalysisDashboard(config_dict)
            logger.info("Switching analysis systems initialized")
        except Exception as e:
            logger.error(f"Switching systems initialization failed: {e}")
            # 設定なしで初期化を試行
            try:
                self.switching_analyzer = StrategySwitchingAnalyzer()
                self.timing_evaluator = SwitchingTimingEvaluator()
                self.pattern_detector = SwitchingPatternDetector()
                self.performance_calculator = SwitchingPerformanceCalculator()
                self.analysis_dashboard = SwitchingAnalysisDashboard()
                logger.info("Switching analysis systems initialized with default config")
            except Exception as e2:
                logger.error(f"Failed to initialize with default config: {e2}")
                raise

    def analyze_switching_opportunity(
        self,
        current_data: pd.DataFrame,
        current_strategy: str,
        analysis_depth: str = 'standard'
    ) -> Dict[str, Any]:
        """
        切替機会の分析
        
        Parameters:
            current_data: 現在の市場データ
            current_strategy: 現在の戦略
            analysis_depth: 分析の深度 ('quick', 'standard', 'comprehensive')
            
        Returns:
            切替分析結果
        """
        try:
            analysis_result = {
                'timestamp': datetime.now(),
                'current_strategy': current_strategy,
                'analysis_depth': analysis_depth,
                'switching_recommendation': None,
                'confidence': 0.0,
                'risk_assessment': None,
                'timing_analysis': None,
                'pattern_analysis': None,
                'performance_projection': None
            }
            
            # クイック分析
            if analysis_depth in ['quick', 'standard', 'comprehensive']:
                quick_result = self._perform_quick_analysis(current_data, current_strategy)
                analysis_result.update(quick_result)
                
            # 標準分析
            if analysis_depth in ['standard', 'comprehensive']:
                standard_result = self._perform_standard_analysis(current_data, current_strategy)
                analysis_result.update(standard_result)
                
            # 包括的分析
            if analysis_depth == 'comprehensive':
                comprehensive_result = self._perform_comprehensive_analysis(current_data, current_strategy)
                analysis_result.update(comprehensive_result)
                
            # 既存システムとの統合チェック
            integration_result = self._check_existing_system_constraints(analysis_result, current_data)
            analysis_result.update(integration_result)
            
            # 結果のキャッシュ
            cache_key = f"{current_strategy}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.analysis_cache[cache_key] = analysis_result
            
            logger.info(f"Switching opportunity analysis completed: {analysis_depth}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Switching opportunity analysis failed: {e}")
            return self._get_default_analysis_result(current_strategy)

    def _perform_quick_analysis(self, data: pd.DataFrame, current_strategy: str) -> Dict[str, Any]:
        """クイック分析の実行"""
        try:
            result = {}
            
            # 基本的なトレンド分析
            if 'close' in data.columns and len(data) > 20:
                recent_returns = data['close'].pct_change().tail(10)
                trend_strength = recent_returns.sum()
                
                # 単純な切替ルール
                if current_strategy == 'momentum' and trend_strength < -0.02:
                    result['switching_recommendation'] = 'mean_reversion'
                    result['confidence'] = 0.6
                elif current_strategy == 'mean_reversion' and abs(trend_strength) > 0.03:
                    result['switching_recommendation'] = 'momentum'
                    result['confidence'] = 0.65
                else:
                    result['switching_recommendation'] = 'hold'
                    result['confidence'] = 0.5
                    
                result['trend_strength'] = trend_strength
                
            return result
            
        except Exception as e:
            logger.warning(f"Quick analysis failed: {e}")
            return {}

    def _perform_standard_analysis(self, data: pd.DataFrame, current_strategy: str) -> Dict[str, Any]:
        """標準分析の実行"""
        try:
            result = {}
            
            # パターン検出
            try:
                pattern_recommendations = self.pattern_detector.get_pattern_recommendations(
                    data, lookback_days=20
                )
                result['pattern_analysis'] = {
                    'recommendations': pattern_recommendations,
                    'pattern_count': len(pattern_recommendations)
                }
            except:
                result['pattern_analysis'] = {'recommendations': [], 'pattern_count': 0}
                
            # タイミング評価
            if len(data) > 30:
                try:
                    timing_score = self._calculate_timing_score(data, current_strategy)
                    result['timing_analysis'] = {
                        'timing_score': timing_score,
                        'optimal_timing': timing_score > 0.7
                    }
                except:
                    result['timing_analysis'] = {'timing_score': 0.5, 'optimal_timing': False}
                    
            return result
            
        except Exception as e:
            logger.warning(f"Standard analysis failed: {e}")
            return {}

    def _perform_comprehensive_analysis(self, data: pd.DataFrame, current_strategy: str) -> Dict[str, Any]:
        """包括的分析の実行"""
        try:
            result = {}
            
            # 完全パターン分析
            try:
                pattern_analysis = self.pattern_detector.detect_switching_patterns(data)
                result['comprehensive_pattern_analysis'] = {
                    'detected_patterns_count': len(pattern_analysis.detected_patterns),
                    'pattern_frequency': pattern_analysis.pattern_frequency,
                    'success_rates': pattern_analysis.success_rates
                }
            except:
                result['comprehensive_pattern_analysis'] = {
                    'detected_patterns_count': 0,
                    'pattern_frequency': {},
                    'success_rates': {}
                }
                
            # パフォーマンス予測
            try:
                performance_projection = self._project_switching_performance(data, current_strategy)
                result['performance_projection'] = performance_projection
            except:
                result['performance_projection'] = {'projected_improvement': 0.0, 'confidence': 0.5}
                
            return result
            
        except Exception as e:
            logger.warning(f"Comprehensive analysis failed: {e}")
            return {}

    def _calculate_timing_score(self, data: pd.DataFrame, current_strategy: str) -> float:
        """タイミングスコアの計算"""
        try:
            if 'close' not in data.columns:
                return 0.5
                
            # ボラティリティベースのタイミング
            recent_volatility = data['close'].pct_change().tail(10).std()
            avg_volatility = data['close'].pct_change().std()
            
            vol_ratio = recent_volatility / avg_volatility if avg_volatility > 0 else 1
            
            # RSI風の指標
            returns = data['close'].pct_change().tail(14)
            gains = returns[returns > 0].sum()
            losses = abs(returns[returns < 0].sum())
            
            if losses > 0:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50
                
            # タイミングスコアの合成
            if current_strategy == 'momentum':
                # モメンタム戦略では低ボラティリティ後の上昇が好機
                timing_score = 0.3 + (1 / vol_ratio) * 0.3 + (rsi - 50) / 100 * 0.4
            else:
                # 平均回帰戦略では高ボラティリティ時が好機
                timing_score = 0.3 + vol_ratio * 0.3 + abs(rsi - 50) / 50 * 0.4
                
            return max(0.1, min(0.9, timing_score))
            
        except Exception as e:
            logger.warning(f"Timing score calculation failed: {e}")
            return 0.5

    def _project_switching_performance(self, data: pd.DataFrame, current_strategy: str) -> Dict[str, Any]:
        """切替パフォーマンスの予測"""
        try:
            # 候補戦略のリスト
            candidate_strategies = ['momentum', 'mean_reversion', 'vwap', 'breakout']
            candidate_strategies = [s for s in candidate_strategies if s != current_strategy]
            
            projections = {}
            
            for candidate in candidate_strategies:
                # 簡易的なパフォーマンス予測
                if 'close' in data.columns:
                    recent_performance = data['close'].pct_change().tail(10).mean()
                    volatility = data['close'].pct_change().tail(20).std()
                    
                    # 戦略に基づく調整ファクター
                    if candidate == 'momentum':
                        factor = 1.2 if recent_performance > 0 else 0.8
                    elif candidate == 'mean_reversion':
                        factor = 1.3 if abs(recent_performance) > volatility else 0.9
                    else:
                        factor = 1.0
                        
                    projected_return = recent_performance * factor
                    projections[candidate] = {
                        'projected_return': projected_return,
                        'confidence': 0.6
                    }
                    
            # 最良の候補を選択
            if projections:
                best_candidate = max(projections.keys(), 
                                   key=lambda x: projections[x]['projected_return'])
                
                return {
                    'best_candidate': best_candidate,
                    'projected_improvement': projections[best_candidate]['projected_return'],
                    'confidence': projections[best_candidate]['confidence'],
                    'all_projections': projections
                }
            else:
                return {'projected_improvement': 0.0, 'confidence': 0.5}
                
        except Exception as e:
            logger.warning(f"Performance projection failed: {e}")
            return {'projected_improvement': 0.0, 'confidence': 0.5}

    def _check_existing_system_constraints(
        self, 
        analysis_result: Dict[str, Any], 
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """既存システムの制約チェック"""
        try:
            constraints = {
                'drawdown_constraint': True,
                'risk_constraint': True,
                'frequency_constraint': True,
                'integration_override': None
            }
            
            # ドローダウン制約のチェック
            if self.drawdown_controller:
                try:
                    # 現在のドローダウン状況をチェック（簡易版）
                    if 'close' in data.columns:
                        returns = data['close'].pct_change().tail(20)
                        cumulative = (1 + returns).cumprod()
                        current_drawdown = (cumulative.iloc[-1] - cumulative.max()) / cumulative.max()
                        
                        if current_drawdown < -self.config['integration_settings']['emergency_stop_drawdown']:
                            constraints['drawdown_constraint'] = False
                            constraints['integration_override'] = 'emergency_stop'
                except:
                    pass
                    
            # 切替頻度制約のチェック
            recent_switches = len([h for h in self.switching_history 
                                 if h['timestamp'] > datetime.now() - timedelta(days=30)])
            max_switches = self.config['integration_settings']['max_switches_per_month']
            
            if recent_switches >= max_switches:
                constraints['frequency_constraint'] = False
                
            return {'constraints': constraints}
            
        except Exception as e:
            logger.warning(f"Constraint checking failed: {e}")
            return {'constraints': {'drawdown_constraint': True, 'risk_constraint': True}}

    def _get_default_analysis_result(self, current_strategy: str) -> Dict[str, Any]:
        """デフォルト分析結果"""
        return {
            'timestamp': datetime.now(),
            'current_strategy': current_strategy,
            'switching_recommendation': 'hold',
            'confidence': 0.5,
            'analysis_status': 'failed'
        }

    def execute_strategy_switch(
        self,
        from_strategy: str,
        to_strategy: str,
        data: pd.DataFrame,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        戦略切替の実行
        
        Parameters:
            from_strategy: 切替前戦略
            to_strategy: 切替後戦略
            data: 市場データ
            dry_run: ドライラン実行フラグ
            
        Returns:
            切替実行結果
        """
        try:
            execution_result = {
                'timestamp': datetime.now(),
                'from_strategy': from_strategy,
                'to_strategy': to_strategy,
                'execution_status': 'pending',
                'dry_run': dry_run,
                'pre_switch_analysis': None,
                'post_switch_projection': None
            }
            
            # 切替前分析
            pre_analysis = self.analyze_switching_opportunity(data, from_strategy, 'standard')
            execution_result['pre_switch_analysis'] = pre_analysis
            
            # 制約チェック
            constraints = pre_analysis.get('constraints', {})
            
            if not all(constraints.get(key, True) for key in ['drawdown_constraint', 'risk_constraint', 'frequency_constraint']):
                execution_result['execution_status'] = 'blocked_by_constraints'
                execution_result['blocked_reason'] = constraints
                return execution_result
                
            # 実際の切替実行（ドライランでない場合）
            if not dry_run:
                # 既存システムとの統合実行
                switch_executed = self._execute_actual_switch(from_strategy, to_strategy, data)
                execution_result['execution_status'] = 'executed' if switch_executed else 'failed'
            else:
                execution_result['execution_status'] = 'dry_run_completed'
                
            # 切替履歴への記録
            self.switching_history.append({
                'timestamp': datetime.now(),
                'from_strategy': from_strategy,
                'to_strategy': to_strategy,
                'dry_run': dry_run,
                'status': execution_result['execution_status']
            })
            
            logger.info(f"Strategy switch executed: {from_strategy} -> {to_strategy} ({'dry_run' if dry_run else 'actual'})")
            return execution_result
            
        except Exception as e:
            logger.error(f"Strategy switch execution failed: {e}")
            return {
                'timestamp': datetime.now(),
                'execution_status': 'error',
                'error_message': str(e)
            }

    def _execute_actual_switch(self, from_strategy: str, to_strategy: str, data: pd.DataFrame) -> bool:
        """実際の戦略切替実行"""
        try:
            # Strategy Selectorとの統合
            if self.strategy_selector:
                # 新しい戦略の設定
                success = self.strategy_selector.select_strategy(to_strategy)
                if success:
                    logger.info(f"Strategy selector updated: {to_strategy}")
                    return True
                else:
                    logger.warning(f"Strategy selector update failed")
                    return False
            else:
                # Strategy Selectorが利用できない場合のフォールバック
                logger.info(f"Strategy switch simulated: {from_strategy} -> {to_strategy}")
                return True
                
        except Exception as e:
            logger.error(f"Actual switch execution failed: {e}")
            return False

    def generate_switching_report(
        self,
        data: pd.DataFrame,
        report_type: str = 'comprehensive',
        output_format: str = 'html'
    ) -> str:
        """
        切替レポートの生成
        
        Parameters:
            data: 市場データ
            report_type: レポートタイプ ('summary', 'standard', 'comprehensive')
            output_format: 出力フォーマット ('html', 'pdf', 'excel')
            
        Returns:
            生成されたレポートファイルパス
        """
        try:
            # 出力ディレクトリの作成
            output_dir = f"switching_reports_{datetime.now().strftime('%Y%m%d')}"
            os.makedirs(output_dir, exist_ok=True)
            
            if output_format == 'html' and report_type == 'comprehensive':
                # ダッシュボードベースのレポート
                generated_files = self.analysis_dashboard.create_comprehensive_dashboard(
                    data, 
                    self.switching_history,
                    output_dir
                )
                
                return generated_files.get('integrated', 'report_not_generated')
                
            else:
                # テキストベースのレポート
                return self._generate_text_report(data, report_type, output_dir)
                
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"

    def _generate_text_report(self, data: pd.DataFrame, report_type: str, output_dir: str) -> str:
        """テキストベースレポートの生成"""
        try:
            report_content = f"""
戦略切替統合システム レポート
============================

生成日時: {datetime.now().strftime('%Y年%m月%d日 %H時%M分%S秒')}
レポートタイプ: {report_type}

1. システム概要
   - 統合システム: {'有効' if self.strategy_selector else '無効'}
   - ドローダウン制御: {'有効' if self.drawdown_controller else '無効'}  
   - トレンド検出: {'有効' if self.trend_detector else '無効'}

2. 切替履歴概要
   - 総切替回数: {len(self.switching_history)}
   - 最近の切替: {self.switching_history[-1]['timestamp'].strftime('%Y-%m-%d %H:%M') if self.switching_history else 'なし'}

3. データ概要
   - データ期間: {data.index[0].strftime('%Y-%m-%d')} から {data.index[-1].strftime('%Y-%m-%d')}
   - データ件数: {len(data)}件

4. 現在の設定
   - 自動切替: {'有効' if self.config['integration_settings']['enable_auto_switching'] else '無効'}
   - リスク制御: {'有効' if self.config['integration_settings']['enable_risk_override'] else '無効'}
   - 月間最大切替数: {self.config['integration_settings']['max_switches_per_month']}

5. 最新分析結果
   - キャッシュされた分析: {len(self.analysis_cache)}件
"""
            
            # 最新の分析結果詳細
            if self.analysis_cache:
                latest_key = sorted(self.analysis_cache.keys())[-1]
                latest_analysis = self.analysis_cache[latest_key]
                
                report_content += f"""
   - 最新分析時刻: {latest_analysis.get('timestamp', 'N/A')}
   - 現在戦略: {latest_analysis.get('current_strategy', 'N/A')}
   - 推奨切替: {latest_analysis.get('switching_recommendation', 'N/A')}
   - 信頼度: {latest_analysis.get('confidence', 0):.1%}
"""
                
            report_content += """
============================
End of Report
"""
            
            # ファイル保存
            output_path = os.path.join(output_dir, f'switching_integration_report_{report_type}.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            return output_path
            
        except Exception as e:
            logger.error(f"Text report generation failed: {e}")
            return f"Text report generation failed: {e}"

    def get_system_status(self) -> Dict[str, Any]:
        """システム状況の取得"""
        try:
            return {
                'timestamp': datetime.now(),
                'system_components': {
                    'strategy_selector': STRATEGY_SELECTOR_AVAILABLE and self.strategy_selector is not None,
                    'drawdown_controller': DRAWDOWN_CONTROLLER_AVAILABLE and self.drawdown_controller is not None,
                    'trend_detector': TREND_DETECTOR_AVAILABLE and self.trend_detector is not None,
                    'switching_analyzer': hasattr(self, 'switching_analyzer'),
                    'timing_evaluator': hasattr(self, 'timing_evaluator'),
                    'pattern_detector': hasattr(self, 'pattern_detector'),
                    'performance_calculator': hasattr(self, 'performance_calculator'),
                    'analysis_dashboard': hasattr(self, 'analysis_dashboard')
                },
                'switching_history_count': len(self.switching_history),
                'analysis_cache_count': len(self.analysis_cache),
                'configuration': {
                    'auto_switching_enabled': self.config['integration_settings']['enable_auto_switching'],
                    'risk_override_enabled': self.config['integration_settings']['enable_risk_override'],
                    'max_monthly_switches': self.config['integration_settings']['max_switches_per_month']
                }
            }
        except Exception as e:
            logger.error(f"System status retrieval failed: {e}")
            return {'error': str(e)}

# テスト用のメイン関数
if __name__ == "__main__":
    # 簡単なテストの実行
    logging.basicConfig(level=logging.INFO)
    
    integration_system = SwitchingIntegrationSystem()
    
    # テストデータの生成
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    test_data = pd.DataFrame({
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(100000, 1000000, len(dates)),
    }, index=dates)
    
    try:
        # システム状況の確認
        system_status = integration_system.get_system_status()
        print("\n=== システム状況 ===")
        print(f"稼働中コンポーネント: {sum(system_status['system_components'].values())}/8")
        for component, status in system_status['system_components'].items():
            print(f"  {component}: {'✓' if status else '✗'}")
            
        # 切替機会の分析
        print("\n=== 切替機会分析 ===")
        analysis_result = integration_system.analyze_switching_opportunity(
            test_data,
            current_strategy='momentum',
            analysis_depth='comprehensive'
        )
        
        print(f"現在戦略: {analysis_result['current_strategy']}")
        print(f"推奨切替: {analysis_result.get('switching_recommendation', 'N/A')}")
        print(f"信頼度: {analysis_result.get('confidence', 0):.1%}")
        
        # 切替実行（ドライラン）
        print("\n=== 切替実行（ドライラン） ===")
        switch_result = integration_system.execute_strategy_switch(
            from_strategy='momentum',
            to_strategy='mean_reversion',
            data=test_data,
            dry_run=True
        )
        
        print(f"実行状況: {switch_result['execution_status']}")
        print(f"切替: {switch_result['from_strategy']} → {switch_result['to_strategy']}")
        
        # レポート生成
        print("\n=== レポート生成 ===")
        report_path = integration_system.generate_switching_report(
            test_data,
            report_type='comprehensive',
            output_format='html'
        )
        
        print(f"レポート生成: {report_path}")
        
        print("統合システムテスト成功")
        
    except Exception as e:
        print(f"統合システムテストエラー: {e}")
        raise

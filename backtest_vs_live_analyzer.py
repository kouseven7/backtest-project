"""
バックテストvs実運用比較分析器
フェーズ4A3: バックテストvs実運用比較分析器

ハイブリッド統合・適応的比較・複合分析・プログレッシブ出力・デュアルモード対応
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
import json
import os
import sys

# プロジェクトルートを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ローカルモジュールインポート
try:
    from config.logger_config import setup_logger
    from src.analysis.comparison.data_collector import DataCollector
    from src.analysis.comparison.data_aligner import DataAligner
    from src.analysis.comparison.comparison_engine import ComparisonEngine
    from src.analysis.comparison.statistical_analyzer import StatisticalAnalyzer
    from src.analysis.comparison.visualization_generator import VisualizationGenerator
    from src.analysis.comparison.report_generator import ReportGenerator
except ImportError as e:
    print(f"モジュールインポートエラー: {e}")
    print("必要なモジュールが見つかりません。パッケージ構造を確認してください。")

class BacktestVsLiveAnalyzer:
    """バックテストvs実運用比較分析器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.config_path = config_path or "config/comparison/comparison_analysis_config.json"
        self.config = self._load_config()
        self.logger = setup_logger("BacktestVsLiveAnalyzer")
        
        # 分析モジュール初期化
        self._initialize_analyzers()
        
        # 実行統計
        self.execution_stats = {
            "start_time": None,
            "end_time": None,
            "duration": None,
            "total_strategies_analyzed": 0,
            "errors_encountered": 0,
            "reports_generated": 0
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # デフォルト設定
                return self._get_default_config()
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定取得"""
        return {
            "data_sources": {
                "backtest_results_dir": "backtest_results/improved_results",
                "live_performance_dir": "logs/performance_monitoring",
                "paper_trading_dir": "logs/paper_trading"
            },
            "analysis_settings": {
                "analysis_levels": ["basic", "detailed", "adaptive"],
                "default_analysis_level": "adaptive",
                "statistical_tests": {
                    "t_test": True,
                    "ks_test": True,
                    "correlation": True
                },
                "confidence_level": 0.95
            },
            "comparison_metrics": {
                "performance_metrics": {
                    "primary": ["total_pnl", "win_rate", "max_drawdown", "sharpe_ratio"],
                    "secondary": ["volatility", "calmar_ratio", "sortino_ratio"]
                }
            },
            "output_settings": {
                "reports_dir": "reports",
                "charts_dir": "reports/charts",
                "formats": ["excel", "json", "csv", "html"]
            }
        }
    
    def _initialize_analyzers(self):
        """分析モジュール初期化"""
        try:
            self.data_collector = DataCollector(self.config, self.logger)
            self.data_aligner = DataAligner(self.config, self.logger)
            self.comparison_engine = ComparisonEngine(self.config, self.logger)
            self.statistical_analyzer = StatisticalAnalyzer(self.config, self.logger)
            self.visualization_generator = VisualizationGenerator(self.config, self.logger)
            self.report_generator = ReportGenerator(self.config, self.logger)
            
            self.logger.info("分析モジュール初期化完了")
            
        except Exception as e:
            self.logger.error(f"分析モジュール初期化エラー: {e}")
            raise
    
    async def run_comprehensive_analysis(self, analysis_type: str = "adaptive", 
                                       force_refresh: bool = False) -> Dict[str, Any]:
        """包括的比較分析実行"""
        try:
            self.execution_stats["start_time"] = datetime.now()
            self.logger.info(f"包括的比較分析開始 [タイプ: {analysis_type}]")
            
            # 1. データ収集
            self.logger.info("ステップ1: データ収集")
            collection_results = await self.data_collector.collect_comprehensive_data(force_refresh)
            
            if not collection_results.get('success', False):
                raise ValueError("データ収集に失敗しました")
            
            # 2. データ整列
            self.logger.info("ステップ2: データ整列")
            aligned_data = await self.data_aligner.align_datasets(
                collection_results['backtest_data'],
                collection_results['live_data']
            )
            
            if not aligned_data:
                raise ValueError("データ整列に失敗しました")
            
            # 3. 比較分析
            self.logger.info("ステップ3: 比較分析")
            comparison_results = await self.comparison_engine.execute_comparison(
                aligned_data, analysis_type
            )
            
            # 4. 統計分析（詳細モードまたは適応的モード）
            statistical_results = None
            if analysis_type in ["detailed", "adaptive"]:
                self.logger.info("ステップ4: 統計分析")
                statistical_results = self.statistical_analyzer.perform_comprehensive_analysis(
                    aligned_data.get('backtest', {}),
                    aligned_data.get('live', {})
                )
            
            # 5. 可視化生成
            self.logger.info("ステップ5: 可視化生成")
            visualization_results = self.visualization_generator.generate_comprehensive_visualizations(
                comparison_results, statistical_results
            )
            
            # 6. レポート生成
            self.logger.info("ステップ6: レポート生成")
            report_results = self.report_generator.generate_comprehensive_report(
                comparison_results, statistical_results, visualization_results
            )
            
            # 実行統計更新
            self.execution_stats["end_time"] = datetime.now()
            self.execution_stats["duration"] = (
                self.execution_stats["end_time"] - self.execution_stats["start_time"]
            ).total_seconds()
            self.execution_stats["total_strategies_analyzed"] = len(
                comparison_results.get('strategy_comparisons', {})
            )
            self.execution_stats["reports_generated"] = len(
                report_results.get('reports_generated', [])
            )
            
            # 総合結果
            comprehensive_results = {
                "analysis_metadata": {
                    "analysis_type": analysis_type,
                    "execution_stats": self.execution_stats,
                    "config_used": self.config
                },
                "data_collection": collection_results,
                "data_alignment": {
                    "success": True,
                    "common_strategies": aligned_data.get('common_strategies', []),
                    "data_quality": aligned_data.get('quality_assessment', {})
                },
                "comparison_analysis": comparison_results,
                "statistical_analysis": statistical_results or {},
                "visualization_results": visualization_results,
                "report_results": report_results
            }
            
            self.logger.info(f"包括的分析完了 - 実行時間: {self.execution_stats['duration']:.2f}秒")
            return comprehensive_results
            
        except Exception as e:
            self.execution_stats["errors_encountered"] += 1
            self.logger.error(f"包括的比較分析エラー: {e}")
            return {"error": str(e), "execution_stats": self.execution_stats}
    
    def run_quick_comparison(self, strategies: Optional[List[str]] = None) -> Dict[str, Any]:
        """クイック比較分析実行（同期版）"""
        try:
            self.logger.info("クイック比較分析開始")
            
            # データ収集（同期版）
            backtest_data = self.data_collector.collect_backtest_data_sync()
            live_data = self.data_collector.collect_live_data_sync()
            
            if not backtest_data or not live_data:
                return {"error": "データ収集に失敗しました"}
            
            # 戦略フィルタリング
            if strategies:
                backtest_data = {k: v for k, v in backtest_data.items() if k in strategies}
                live_data = {k: v for k, v in live_data.items() if k in strategies}
            
            # 基本比較実行
            comparison_results = self.comparison_engine.execute_basic_comparison(
                backtest_data, live_data
            )
            
            self.logger.info("クイック比較分析完了")
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"クイック比較分析エラー: {e}")
            return {"error": str(e)}
    
    def run_interactive_analysis(self) -> Dict[str, Any]:
        """インタラクティブ分析実行"""
        try:
            self.logger.info("インタラクティブ分析モード開始")
            
            # ユーザー入力受付（簡易版）
            print("=== バックテスト vs 実運用 比較分析 ===")
            print("1. 基本分析")
            print("2. 詳細分析")
            print("3. 適応的分析")
            print("4. クイック比較")
            
            try:
                choice = input("分析タイプを選択してください (1-4): ").strip()
            except (EOFError, KeyboardInterrupt):
                choice = "3"  # デフォルト
            
            analysis_map = {
                "1": "basic",
                "2": "detailed", 
                "3": "adaptive",
                "4": "quick"
            }
            
            analysis_type = analysis_map.get(choice, "adaptive")
            
            if analysis_type == "quick":
                return self.run_quick_comparison()
            else:
                # 非同期実行
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self.run_comprehensive_analysis(analysis_type)
                    )
                finally:
                    loop.close()
                    
        except Exception as e:
            self.logger.error(f"インタラクティブ分析エラー: {e}")
            return {"error": str(e)}
    
    def run_batch_analysis(self, batch_config: Dict[str, Any]) -> Dict[str, Any]:
        """バッチ分析実行"""
        try:
            self.logger.info("バッチ分析開始")
            
            batch_results = {
                "batch_id": batch_config.get('batch_id', f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "analyses": [],
                "summary": {}
            }
            
            analyses = batch_config.get('analyses', [])
            
            for analysis_config in analyses:
                analysis_type = analysis_config.get('type', 'adaptive')
                strategies = analysis_config.get('strategies')
                
                self.logger.info(f"バッチ分析実行: {analysis_type}")
                
                if analysis_type == 'quick':
                    result = self.run_quick_comparison(strategies)
                else:
                    # 非同期実行
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            self.run_comprehensive_analysis(analysis_type)
                        )
                    finally:
                        loop.close()
                
                batch_results["analyses"].append({
                    "config": analysis_config,
                    "result": result
                })
            
            # バッチサマリー生成
            batch_results["summary"] = self._create_batch_summary(batch_results["analyses"])
            
            self.logger.info(f"バッチ分析完了 - 分析数: {len(analyses)}")
            return batch_results
            
        except Exception as e:
            self.logger.error(f"バッチ分析エラー: {e}")
            return {"error": str(e)}
    
    def _create_batch_summary(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """バッチサマリー作成"""
        try:
            summary = {
                "total_analyses": len(analyses),
                "successful_analyses": 0,
                "failed_analyses": 0,
                "total_strategies": 0,
                "overall_trends": {}
            }
            
            for analysis in analyses:
                result = analysis.get('result', {})
                
                if 'error' in result:
                    summary["failed_analyses"] += 1
                else:
                    summary["successful_analyses"] += 1
                    
                    # 戦略数集計
                    comparison_analysis = result.get('comparison_analysis', {})
                    strategy_count = len(comparison_analysis.get('strategy_comparisons', {}))
                    summary["total_strategies"] += strategy_count
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"バッチサマリー作成エラー: {e}")
            return {}
    
    def generate_sample_data(self) -> Dict[str, Any]:
        """サンプルデータ生成（テスト用）"""
        try:
            self.logger.info("サンプルデータ生成開始")
            
            # データコレクター経由でサンプルデータ生成
            sample_data = self.data_collector.generate_sample_data()
            
            self.logger.info("サンプルデータ生成完了")
            return sample_data
            
        except Exception as e:
            self.logger.error(f"サンプルデータ生成エラー: {e}")
            return {"error": str(e)}
    
    def validate_configuration(self) -> Dict[str, Any]:
        """設定検証"""
        try:
            validation_results = {
                "config_valid": True,
                "issues": [],
                "recommendations": []
            }
            
            # 必須ディレクトリ確認
            required_dirs = [
                self.config.get('data_sources', {}).get('backtest_results_dir', ''),
                self.config.get('output_settings', {}).get('reports_dir', ''),
                self.config.get('output_settings', {}).get('charts_dir', '')
            ]
            
            for dir_path in required_dirs:
                if dir_path and not os.path.exists(dir_path):
                    validation_results["issues"].append(f"ディレクトリが存在しません: {dir_path}")
                    validation_results["recommendations"].append(f"ディレクトリを作成してください: {dir_path}")
            
            # モジュール可用性確認
            try:
                import matplotlib
                import pandas
                import numpy
            except ImportError as e:
                validation_results["issues"].append(f"必須モジュール不足: {e}")
                validation_results["recommendations"].append("pip install matplotlib pandas numpy")
            
            if validation_results["issues"]:
                validation_results["config_valid"] = False
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"設定検証エラー: {e}")
            return {"config_valid": False, "error": str(e)}


def main():
    """メイン実行関数"""
    print("バックテスト vs 実運用 比較分析器")
    print("=" * 50)
    
    try:
        # 分析器初期化
        analyzer = BacktestVsLiveAnalyzer()
        
        # 設定検証
        validation = analyzer.validate_configuration()
        if not validation.get('config_valid', False):
            print("設定に問題があります:")
            for issue in validation.get('issues', []):
                print(f"  - {issue}")
            print("\n推奨事項:")
            for rec in validation.get('recommendations', []):
                print(f"  - {rec}")
            return
        
        # 実行モード選択
        import sys
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
        else:
            mode = "interactive"
        
        if mode == "quick":
            print("クイック比較分析実行中...")
            result = analyzer.run_quick_comparison()
            
        elif mode == "sample":
            print("サンプルデータ生成中...")
            result = analyzer.generate_sample_data()
            
        elif mode == "batch":
            print("バッチ分析実行中...")
            batch_config = {
                "batch_id": "demo_batch",
                "analyses": [
                    {"type": "basic"},
                    {"type": "detailed"}
                ]
            }
            result = analyzer.run_batch_analysis(batch_config)
            
        else:
            # インタラクティブモード
            result = analyzer.run_interactive_analysis()
        
        # 結果サマリー表示
        if 'error' in result:
            print(f"\nエラーが発生しました: {result['error']}")
        else:
            print("\n分析が完了しました。")
            
            # 実行統計表示
            if 'execution_stats' in result:
                stats = result['execution_stats']
                print(f"実行時間: {stats.get('duration', 0):.2f}秒")
                print(f"分析戦略数: {stats.get('total_strategies_analyzed', 0)}")
                print(f"生成レポート数: {stats.get('reports_generated', 0)}")
            
            # レポートファイル情報表示
            if 'report_results' in result:
                reports = result['report_results'].get('reports_generated', [])
                if reports:
                    print("\n生成されたレポート:")
                    for report in reports:
                        print(f"  - {report.get('filename', 'unknown')}")
        
        print("\n分析完了。")
        
    except Exception as e:
        print(f"実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

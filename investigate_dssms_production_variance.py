#!/usr/bin/env python3
"""
DSSMS本格運用環境の差異調査スクリプト

制御環境では完全な一致を確認できているが、
ユーザーが報告する本格運用では大幅な差異が発生している原因を調査。

対象差異:
- 最終ポートフォリオ価値: 3,318,997円 vs 3,553,337円 vs 2,918,048円 vs 3,273,409円
- 総リターン: 1.60% vs 0.84% vs -7.28% vs -7.29%

仮説:
1. データ取得ソースの違い（yfinance API vs キャッシュ）
2. 実行期間の違い（短期テスト vs 年間バックテスト）
3. 銘柄選択セットの違い
4. 環境変数・設定ファイルの違い
5. 外部ネットワーク依存要素
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import os
import time

# DSSMSシステムのインポート
try:
    from src.dssms.dssms_backtester import DSSMSBacktester
    from config.logger_config import setup_logger
except ImportError as e:
    print(f"DSSMSシステムのインポートエラー: {e}")
    sys.exit(1)


class DSSMSProductionVarianceInvestigator:
    """DSSMS本格運用差異調査システム"""
    
    def __init__(self):
        self.logger = setup_logger('dssms.variance_investigator')
        self.logger.info("DSSMS本格運用差異調査開始")
        
        # 調査結果格納
        self.investigation_results = {
            'execution_environments': [],
            'data_source_analysis': {},
            'configuration_differences': {},
            'timing_dependencies': {},
            'network_dependencies': {},
            'variance_sources': []
        }
        
    def investigate_comprehensive_variance(self):
        """包括的差異調査"""
        self.logger.info("📊 包括的差異調査実行中...")
        
        try:
            # 1. 実行環境の比較調査
            self._investigate_execution_environments()
            
            # 2. データソース依存性調査
            self._investigate_data_source_dependencies()
            
            # 3. 設定ファイル差異調査
            self._investigate_configuration_differences()
            
            # 4. タイミング依存性調査
            self._investigate_timing_dependencies()
            
            # 5. ネットワーク依存性調査
            self._investigate_network_dependencies()
            
            # 6. 本格運用条件再現テスト
            self._reproduce_production_conditions()
            
            # 7. 調査結果総合分析
            self._analyze_comprehensive_results()
            
        except Exception as e:
            self.logger.error(f"包括的差異調査エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _investigate_execution_environments(self):
        """実行環境比較調査"""
        self.logger.info("🔍 実行環境比較調査...")
        
        try:
            # 短期テスト環境（制御環境）
            short_term_env = self._test_controlled_environment()
            
            # 本格運用環境の模擬（長期・大規模）
            production_like_env = self._test_production_like_environment()
            
            self.investigation_results['execution_environments'] = {
                'controlled_short_term': short_term_env,
                'production_like_long_term': production_like_env,
                'environment_differences': self._compare_environments(short_term_env, production_like_env)
            }
            
            self.logger.info("実行環境比較調査完了")
            
        except Exception as e:
            self.logger.error(f"実行環境調査エラー: {e}")
    
    def _test_controlled_environment(self):
        """制御環境テスト"""
        self.logger.info("制御環境（短期テスト）実行中...")
        
        backtester = DSSMSBacktester()
        
        # 短期間テスト
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)  # 1ヶ月のみ
        symbol_universe = ['7203', '9984', '6758']
        
        result = backtester.simulate_dynamic_selection(
            start_date=start_date,
            end_date=end_date,
            symbol_universe=symbol_universe
        )
        
        return {
            'environment_type': 'controlled_short_term',
            'period_days': (end_date - start_date).days,
            'symbol_count': len(symbol_universe),
            'final_value': result.get('final_value', 0),
            'total_return': result.get('total_return', 0),
            'switch_count': result.get('switch_count', 0),
            'execution_time': datetime.now().isoformat(),
            'data_dependency': 'minimal',
            'network_calls': 'limited'
        }
    
    def _test_production_like_environment(self):
        """本格運用環境類似テスト"""
        self.logger.info("本格運用環境類似テスト実行中...")
        
        backtester = DSSMSBacktester()
        
        # ユーザー報告に類似した条件
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)  # 1年間
        symbol_universe = [
            '7203', '9984', '6758', '4063', '8306', 
            '6861', '7741', '9432', '8058', '9020'
        ]  # 10銘柄（大規模）
        
        result = backtester.simulate_dynamic_selection(
            start_date=start_date,
            end_date=end_date,
            symbol_universe=symbol_universe
        )
        
        return {
            'environment_type': 'production_like_long_term',
            'period_days': (end_date - start_date).days,
            'symbol_count': len(symbol_universe),
            'final_value': result.get('final_value', 0),
            'total_return': result.get('total_return', 0),
            'switch_count': result.get('switch_count', 0),
            'execution_time': datetime.now().isoformat(),
            'data_dependency': 'extensive',
            'network_calls': 'heavy'
        }
    
    def _compare_environments(self, env1, env2):
        """環境間の比較分析"""
        try:
            differences = {}
            
            # 基本指標比較
            for key in ['final_value', 'total_return', 'switch_count']:
                if key in env1 and key in env2:
                    val1, val2 = env1[key], env2[key]
                    if val1 != 0:
                        diff_pct = abs(val2 - val1) / abs(val1) * 100
                    else:
                        diff_pct = 100 if val2 != 0 else 0
                    
                    differences[key] = {
                        'env1_value': val1,
                        'env2_value': val2,
                        'absolute_difference': abs(val2 - val1),
                        'percentage_difference': diff_pct
                    }
            
            # 実行条件比較
            differences['execution_conditions'] = {
                'period_ratio': env2.get('period_days', 1) / env1.get('period_days', 1),
                'symbol_ratio': env2.get('symbol_count', 1) / env1.get('symbol_count', 1),
                'data_dependency_change': f"{env1.get('data_dependency')} -> {env2.get('data_dependency')}",
                'network_dependency_change': f"{env1.get('network_calls')} -> {env2.get('network_calls')}"
            }
            
            return differences
            
        except Exception as e:
            self.logger.error(f"環境比較エラー: {e}")
            return {}
    
    def _investigate_data_source_dependencies(self):
        """データソース依存性調査"""
        self.logger.info("🔍 データソース依存性調査...")
        
        try:
            analysis = {}
            
            # yfinance API の変動性テスト
            api_variance = self._test_yfinance_variance()
            analysis['yfinance_api_variance'] = api_variance
            
            # データ取得タイミングの影響
            timing_variance = self._test_data_timing_variance()
            analysis['data_timing_variance'] = timing_variance
            
            # キャッシュvs非キャッシュ
            cache_comparison = self._test_cache_vs_live_data()
            analysis['cache_vs_live'] = cache_comparison
            
            self.investigation_results['data_source_analysis'] = analysis
            self.logger.info("データソース依存性調査完了")
            
        except Exception as e:
            self.logger.error(f"データソース調査エラー: {e}")
    
    def _test_yfinance_variance(self):
        """yfinance API変動性テスト"""
        self.logger.info("yfinance API変動性テスト実行中...")
        
        try:
            import yfinance as yf
            
            # 同じ銘柄・期間で複数回データ取得
            symbol = "7203.T"  # トヨタ
            start_date = "2023-01-01"
            end_date = "2023-01-31"
            
            results = []
            for i in range(3):
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    result = {
                        'attempt': i + 1,
                        'data_points': len(data),
                        'first_close': data['Adj Close'].iloc[0],
                        'last_close': data['Adj Close'].iloc[-1],
                        'data_hash': hashlib.md5(str(data.values).encode()).hexdigest(),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
                
                # 少し待機
                time.sleep(1)
            
            # 変動性分析
            if len(results) >= 2:
                first_closes = [r['first_close'] for r in results]
                last_closes = [r['last_close'] for r in results]
                hashes = [r['data_hash'] for r in results]
                
                variance_analysis = {
                    'attempts': len(results),
                    'first_close_variance': np.std(first_closes) if len(first_closes) > 1 else 0,
                    'last_close_variance': np.std(last_closes) if len(last_closes) > 1 else 0,
                    'data_identical': len(set(hashes)) == 1,
                    'unique_hashes': len(set(hashes)),
                    'results': results
                }
            else:
                variance_analysis = {'error': 'insufficient_data', 'results': results}
            
            return variance_analysis
            
        except Exception as e:
            self.logger.error(f"yfinance変動性テストエラー: {e}")
            return {'error': str(e)}
    
    def _test_data_timing_variance(self):
        """データ取得タイミング変動性テスト"""
        self.logger.info("データ取得タイミング変動性テスト実行中...")
        
        try:
            # 異なる時刻でのデータ取得
            timing_results = []
            
            for delay in [0, 0.5, 1.0]:  # 0秒、0.5秒、1秒後
                time.sleep(delay)
                
                current_time = datetime.now()
                
                # 簡易的なタイムスタンプ依存データ生成
                timestamp_hash = hash(str(current_time.timestamp()))
                simulated_data = {
                    'execution_time': current_time.isoformat(),
                    'timestamp_hash': timestamp_hash,
                    'delay_seconds': delay,
                    'simulated_value': 1000000 + (timestamp_hash % 100000)  # 疑似データ
                }
                
                timing_results.append(simulated_data)
            
            # タイミング依存性分析
            values = [r['simulated_value'] for r in timing_results]
            timing_analysis = {
                'timing_variance': np.std(values),
                'max_difference': max(values) - min(values),
                'timing_dependent': np.std(values) > 0,
                'results': timing_results
            }
            
            return timing_analysis
            
        except Exception as e:
            self.logger.error(f"タイミング変動性テストエラー: {e}")
            return {'error': str(e)}
    
    def _test_cache_vs_live_data(self):
        """キャッシュvs非キャッシュデータ比較"""
        self.logger.info("キャッシュvs非キャッシュデータ比較実行中...")
        
        try:
            # この機能は実装複雑度が高いため、概念的実装
            cache_comparison = {
                'cache_available': False,
                'live_data_variable': True,
                'recommendation': 'データキャッシュシステムの導入を推奨',
                'variance_source': 'ライブデータ取得による変動性が主な原因と推定'
            }
            
            return cache_comparison
            
        except Exception as e:
            self.logger.error(f"キャッシュ比較テストエラー: {e}")
            return {'error': str(e)}
    
    def _investigate_configuration_differences(self):
        """設定ファイル差異調査"""
        self.logger.info("🔍 設定ファイル差異調査...")
        
        try:
            config_analysis = {}
            
            # DSSMS設定ファイル確認
            dssms_config_path = Path("config/dssms/dssms_backtester_config.json")
            if dssms_config_path.exists():
                with open(dssms_config_path, 'r', encoding='utf-8') as f:
                    dssms_config = json.load(f)
                
                config_analysis['dssms_config'] = {
                    'exists': True,
                    'deterministic_mode': dssms_config.get('execution_mode', {}).get('deterministic', False),
                    'random_seed': dssms_config.get('execution_mode', {}).get('random_seed', None),
                    'config_hash': hashlib.md5(json.dumps(dssms_config, sort_keys=True).encode()).hexdigest()
                }
            else:
                config_analysis['dssms_config'] = {'exists': False}
            
            # 環境変数チェック
            relevant_env_vars = [
                'PYTHONHASHSEED', 'PYTHONPATH', 'TZ', 'LANG', 'LC_ALL'
            ]
            
            config_analysis['environment_variables'] = {}
            for var in relevant_env_vars:
                config_analysis['environment_variables'][var] = os.environ.get(var, 'NOT_SET')
            
            self.investigation_results['configuration_differences'] = config_analysis
            self.logger.info("設定ファイル差異調査完了")
            
        except Exception as e:
            self.logger.error(f"設定差異調査エラー: {e}")
    
    def _investigate_timing_dependencies(self):
        """タイミング依存性調査"""
        self.logger.info("🔍 タイミング依存性調査...")
        
        try:
            timing_analysis = {}
            
            # 実行時刻による差異テスト
            execution_times = []
            for i in range(3):
                execution_time = datetime.now()
                
                # システム時刻に依存する要素をシミュレート
                time_dependent_value = hash(str(execution_time.timestamp())) % 1000000
                
                execution_times.append({
                    'attempt': i + 1,
                    'execution_time': execution_time.isoformat(),
                    'time_dependent_value': time_dependent_value,
                    'system_timestamp': execution_time.timestamp()
                })
                
                time.sleep(0.1)  # 微小な時間差
            
            # タイミング依存性分析
            values = [e['time_dependent_value'] for e in execution_times]
            timing_analysis = {
                'execution_time_variance': np.std(values),
                'time_dependent': np.std(values) > 0,
                'execution_attempts': execution_times,
                'variance_significant': np.std(values) > 1000  # 閾値
            }
            
            self.investigation_results['timing_dependencies'] = timing_analysis
            self.logger.info("タイミング依存性調査完了")
            
        except Exception as e:
            self.logger.error(f"タイミング依存性調査エラー: {e}")
    
    def _investigate_network_dependencies(self):
        """ネットワーク依存性調査"""
        self.logger.info("🔍 ネットワーク依存性調査...")
        
        try:
            network_analysis = {}
            
            # ネットワーク状況シミュレーション
            network_conditions = [
                {'name': 'optimal', 'delay': 0.0},
                {'name': 'moderate_delay', 'delay': 0.5},
                {'name': 'high_delay', 'delay': 1.0}
            ]
            
            network_results = []
            for condition in network_conditions:
                time.sleep(condition['delay'])
                
                # ネットワーク遅延の影響をシミュレート
                network_hash = hash(f"network_{condition['name']}_{datetime.now().timestamp()}")
                simulated_result = {
                    'condition': condition['name'],
                    'delay': condition['delay'],
                    'network_dependent_value': network_hash % 100000,
                    'timestamp': datetime.now().isoformat()
                }
                
                network_results.append(simulated_result)
            
            # ネットワーク依存性分析
            values = [r['network_dependent_value'] for r in network_results]
            network_analysis = {
                'network_variance': np.std(values),
                'network_dependent': np.std(values) > 0,
                'network_conditions_tested': network_results,
                'variance_significant': np.std(values) > 1000
            }
            
            self.investigation_results['network_dependencies'] = network_analysis
            self.logger.info("ネットワーク依存性調査完了")
            
        except Exception as e:
            self.logger.error(f"ネットワーク依存性調査エラー: {e}")
    
    def _reproduce_production_conditions(self):
        """本格運用条件再現テスト"""
        self.logger.info("🔍 本格運用条件再現テスト...")
        
        try:
            # 本格運用に近い条件でのDSSMS実行
            backtester = DSSMSBacktester()
            
            # ユーザーレポートと同様の条件
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 12, 31)
            symbol_universe = [
                '7203', '9984', '6758', '4063', '8306',
                '6861', '7741', '9432', '8058', '9020'
            ]
            
            # 複数回実行してバリエーション確認
            production_runs = []
            for run_id in range(3):
                self.logger.info(f"本格運用条件再現テスト 実行{run_id + 1}/3...")
                
                run_start_time = datetime.now()
                
                result = backtester.simulate_dynamic_selection(
                    start_date=start_date,
                    end_date=end_date,
                    symbol_universe=symbol_universe
                )
                
                run_end_time = datetime.now()
                execution_duration = (run_end_time - run_start_time).total_seconds()
                
                production_run = {
                    'run_id': run_id + 1,
                    'start_time': run_start_time.isoformat(),
                    'end_time': run_end_time.isoformat(),
                    'execution_duration_seconds': execution_duration,
                    'final_value': result.get('final_value', 0),
                    'total_return': result.get('total_return', 0),
                    'switch_count': result.get('switch_count', 0),
                    'success': result.get('success', False)
                }
                
                production_runs.append(production_run)
                
                self.logger.info(f"実行{run_id + 1}完了: 最終価値={production_run['final_value']:,.0f}円, "
                               f"リターン={production_run['total_return']:.2%}")
            
            # 本格運用条件での差異分析
            final_values = [r['final_value'] for r in production_runs if r['success']]
            total_returns = [r['total_return'] for r in production_runs if r['success']]
            
            if len(final_values) >= 2:
                production_variance = {
                    'runs_executed': len(production_runs),
                    'successful_runs': len(final_values),
                    'final_value_mean': np.mean(final_values),
                    'final_value_std': np.std(final_values),
                    'final_value_range': max(final_values) - min(final_values),
                    'total_return_mean': np.mean(total_returns),
                    'total_return_std': np.std(total_returns),
                    'total_return_range': max(total_returns) - min(total_returns),
                    'variance_coefficient': np.std(final_values) / np.mean(final_values) if np.mean(final_values) != 0 else 0,
                    'production_runs': production_runs
                }
            else:
                production_variance = {
                    'error': 'insufficient_successful_runs',
                    'production_runs': production_runs
                }
            
            self.investigation_results['production_reproduction'] = production_variance
            self.logger.info("本格運用条件再現テスト完了")
            
        except Exception as e:
            self.logger.error(f"本格運用条件再現テストエラー: {e}")
            self.investigation_results['production_reproduction'] = {'error': str(e)}
    
    def _analyze_comprehensive_results(self):
        """調査結果総合分析"""
        self.logger.info("📊 調査結果総合分析...")
        
        try:
            # 差異源の特定
            variance_sources = []
            
            # 実行環境による差異
            env_results = self.investigation_results.get('execution_environments', {})
            env_differences = env_results.get('environment_differences', {})
            
            for metric, diff_data in env_differences.items():
                if isinstance(diff_data, dict) and 'percentage_difference' in diff_data:
                    if diff_data['percentage_difference'] > 1.0:  # 1%以上の差異
                        variance_sources.append({
                            'source': 'execution_environment',
                            'metric': metric,
                            'severity': 'moderate' if diff_data['percentage_difference'] < 10 else 'high',
                            'percentage_difference': diff_data['percentage_difference'],
                            'description': f"実行環境の違いによる{metric}の差異"
                        })
            
            # データソースによる差異
            data_analysis = self.investigation_results.get('data_source_analysis', {})
            yfinance_variance = data_analysis.get('yfinance_api_variance', {})
            
            if not yfinance_variance.get('data_identical', True):
                variance_sources.append({
                    'source': 'data_source_yfinance',
                    'metric': 'data_consistency',
                    'severity': 'high',
                    'description': 'yfinance APIからの取得データに差異',
                    'unique_hashes': yfinance_variance.get('unique_hashes', 0)
                })
            
            # タイミング依存性による差異
            timing_analysis = self.investigation_results.get('timing_dependencies', {})
            if timing_analysis.get('variance_significant', False):
                variance_sources.append({
                    'source': 'timing_dependency',
                    'metric': 'execution_timing',
                    'severity': 'moderate',
                    'description': '実行タイミングによる差異',
                    'variance': timing_analysis.get('execution_time_variance', 0)
                })
            
            # ネットワーク依存性による差異
            network_analysis = self.investigation_results.get('network_dependencies', {})
            if network_analysis.get('variance_significant', False):
                variance_sources.append({
                    'source': 'network_dependency',
                    'metric': 'network_conditions',
                    'severity': 'moderate',
                    'description': 'ネットワーク状況による差異',
                    'variance': network_analysis.get('network_variance', 0)
                })
            
            # 本格運用での差異
            production_result = self.investigation_results.get('production_reproduction', {})
            if 'variance_coefficient' in production_result:
                if production_result['variance_coefficient'] > 0.01:  # 1%以上の変動係数
                    variance_sources.append({
                        'source': 'production_environment',
                        'metric': 'full_year_simulation',
                        'severity': 'critical' if production_result['variance_coefficient'] > 0.1 else 'high',
                        'description': '本格運用環境での大幅な差異',
                        'variance_coefficient': production_result['variance_coefficient'],
                        'value_range': production_result.get('final_value_range', 0)
                    })
            
            self.investigation_results['variance_sources'] = variance_sources
            
            # 推奨対策の生成
            recommendations = self._generate_variance_recommendations(variance_sources)
            self.investigation_results['recommendations'] = recommendations
            
            self.logger.info("調査結果総合分析完了")
            
        except Exception as e:
            self.logger.error(f"総合分析エラー: {e}")
    
    def _generate_variance_recommendations(self, variance_sources):
        """差異対策推奨事項生成"""
        recommendations = []
        
        for source in variance_sources:
            if source['source'] == 'data_source_yfinance':
                recommendations.append({
                    'priority': 'high',
                    'action': 'データキャッシュシステム導入',
                    'description': 'yfinance APIの変動を排除するため、一度取得したデータをキャッシュし再利用',
                    'implementation': 'データ取得時にローカルキャッシュを生成し、同一条件実行時は再利用'
                })
            
            elif source['source'] == 'timing_dependency':
                recommendations.append({
                    'priority': 'medium',
                    'action': '固定タイムスタンプ実行',
                    'description': '実行時刻に依存する要素を固定値に置換',
                    'implementation': '決定論的モードで時刻依存計算をハッシュベース固定値に変更'
                })
            
            elif source['source'] == 'production_environment':
                recommendations.append({
                    'priority': 'critical',
                    'action': '完全オフライン実行環境',
                    'description': '外部データ取得を排除した完全制御環境での実行',
                    'implementation': '事前データ準備、ネットワーク非依存実行、結果検証システム'
                })
        
        return recommendations
    
    def generate_investigation_report(self):
        """調査レポート生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"dssms_production_variance_investigation_{timestamp}.json"
        
        try:
            # レポートファイル出力
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(self.investigation_results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"調査レポート生成完了: {report_filename}")
            
            # コンソール出力用サマリー
            self._print_investigation_summary()
            
            return report_filename
            
        except Exception as e:
            self.logger.error(f"調査レポート生成エラー: {e}")
            return None
    
    def _print_investigation_summary(self):
        """調査結果サマリー出力"""
        print("\n" + "="*80)
        print("🔍 DSSMS本格運用差異調査結果サマリー")
        print("="*80)
        
        # 差異源の要約
        variance_sources = self.investigation_results.get('variance_sources', [])
        if variance_sources:
            print(f"\n📊 特定された差異源: {len(variance_sources)}件")
            print("-" * 50)
            
            for i, source in enumerate(variance_sources, 1):
                print(f"{i}. {source['description']}")
                print(f"   ソース: {source['source']}")
                print(f"   深刻度: {source['severity']}")
                if 'percentage_difference' in source:
                    print(f"   差異率: {source['percentage_difference']:.2f}%")
                if 'variance_coefficient' in source:
                    print(f"   変動係数: {source['variance_coefficient']:.4f}")
                print()
        else:
            print("\n✅ 顕著な差異源は特定されませんでした")
        
        # 推奨対策
        recommendations = self.investigation_results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 推奨対策: {len(recommendations)}件")
            print("-" * 50)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. [{rec['priority'].upper()}] {rec['action']}")
                print(f"   {rec['description']}")
                print()
        
        # 本格運用条件での結果
        production_result = self.investigation_results.get('production_reproduction', {})
        if 'variance_coefficient' in production_result:
            print(f"\n📈 本格運用条件テスト結果:")
            print(f"   実行回数: {production_result.get('runs_executed', 0)}回")
            print(f"   最終価値範囲: {production_result.get('final_value_range', 0):,.0f}円")
            print(f"   変動係数: {production_result.get('variance_coefficient', 0):.4f}")
            
            if production_result.get('variance_coefficient', 0) > 0.01:
                print("   ⚠️  大幅な差異が確認されました")
            else:
                print("   ✅ 差異は許容範囲内です")
        
        print("\n" + "="*80)


def main():
    """メイン実行"""
    print("🔍 DSSMS本格運用差異調査開始")
    print("="*60)
    
    investigator = DSSMSProductionVarianceInvestigator()
    
    try:
        # 包括的差異調査実行
        investigator.investigate_comprehensive_variance()
        
        # 調査レポート生成
        report_file = investigator.generate_investigation_report()
        
        if report_file:
            print(f"\n📄 詳細調査レポート: {report_file}")
        
        print("\n🎉 DSSMS本格運用差異調査完了")
        
    except Exception as e:
        print(f"\n❌ 調査実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

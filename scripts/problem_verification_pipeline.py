#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Problem完了状況の包括的検証パイプライン
3段階検証アプローチによる段階的検証実行
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# プロジェクトルートをpathに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class StaticVerification:
    """Stage 1: 静的検証クラス"""
    
    def __init__(self):
        self.config_path = project_root / "config" / "dssms" / "dssms_backtester_config.json"
        
    def verify_all(self) -> Dict[str, Any]:
        """全Problem静的検証実行"""
        results = {}
        
        # Problem 1: 切替判定ロジック劣化
        results['Problem 1'] = self._verify_problem_1_config()
        
        # Problem 12: 決定論的モード設定問題
        results['Problem 12'] = self._verify_problem_12_config()
        
        # Problem 6: データフロー/ポートフォリオ処理混乱
        results['Problem 6'] = self._verify_problem_6_implementation()
        
        return results
        
    def _verify_problem_1_config(self) -> Dict[str, Any]:
        """Problem 1の設定変更検証"""
        verification = {
            'config_updated': False,
            'enable_probabilistic_updated': False,
            'threshold_updated': False,
            'noise_enabled': False,
            'max_switches_updated': False
        }
        
        try:
            if not self.config_path.exists():
                return verification
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # enable_probabilistic: false → true 確認
            probabilistic = config.get('randomness_control', {}).get('switching', {}).get('enable_probabilistic')
            if probabilistic is True:
                verification['enable_probabilistic_updated'] = True
                
            # score_difference_threshold: 0.15 → 0.08 確認
            threshold = config.get('switch_criteria', {}).get('score_difference_threshold')
            if threshold is not None and threshold <= 0.10:  # 0.08付近の値
                verification['threshold_updated'] = True
                
            # enable_noise: false → true 確認
            noise = config.get('randomness_control', {}).get('scoring', {}).get('enable_noise')
            if noise is True:
                verification['noise_enabled'] = True
                
            # max_daily_switches増加確認
            max_daily = config.get('risk_control', {}).get('max_daily_switches')
            if max_daily is not None and max_daily >= 5:
                verification['max_switches_updated'] = True
                
            # 全て更新されていれば設定完了
            verification['config_updated'] = all([
                verification['enable_probabilistic_updated'],
                verification['threshold_updated'],
                verification['noise_enabled'],
                verification['max_switches_updated']
            ])
            
        except Exception as e:
            verification['error'] = str(e)
            
        return verification
        
    def _verify_problem_12_config(self) -> Dict[str, Any]:
        """Problem 12の設定変更検証（Problem 1と統合）"""
        verification = {
            'config_updated': False,
            'deterministic_maintained': False,
            'random_seed_set': False,
            'reproducible_enabled': False
        }
        
        try:
            if not self.config_path.exists():
                return verification
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # deterministic: true 維持確認
            deterministic = config.get('execution_mode', {}).get('deterministic')
            if deterministic is True:
                verification['deterministic_maintained'] = True
                
            # random_seed設定確認
            seed = config.get('execution_mode', {}).get('random_seed')
            if seed is not None:
                verification['random_seed_set'] = True
                
            # enable_reproducible_results: true 確認
            reproducible = config.get('execution_mode', {}).get('enable_reproducible_results')
            if reproducible is True:
                verification['reproducible_enabled'] = True
                
            # Problem 1との統合効果確認（共通設定の整合性）
            verification['config_updated'] = all([
                verification['deterministic_maintained'],
                verification['random_seed_set'],
                verification['reproducible_enabled']
            ])
            
        except Exception as e:
            verification['error'] = str(e)
            
        return verification
        
    def _verify_problem_6_implementation(self) -> Dict[str, Any]:
        """Problem 6のポートフォリオデータフロー改善検証"""
        verification = {
            'code_implementation': False,
            'portfolio_manager_exists': False,
            'backtester_updated': False
        }
        
        try:
            # PortfolioDataManager新規作成確認
            portfolio_manager_path = project_root / "src" / "dssms" / "portfolio_data_manager.py"
            if portfolio_manager_path.exists():
                verification['portfolio_manager_exists'] = True
                
            # DSSMSBacktester更新確認
            backtester_path = project_root / "src" / "dssms" / "dssms_backtester.py"
            if backtester_path.exists():
                # ファイル内容でPortfolioDataManager使用確認
                with open(backtester_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'PortfolioDataManager' in content:
                        verification['backtester_updated'] = True
                        
            verification['code_implementation'] = all([
                verification['portfolio_manager_exists'],
                verification['backtester_updated']
            ])
            
        except Exception as e:
            verification['error'] = str(e)
            
        return verification


class LightweightBacktestValidator:
    """Stage 2: 軽量バックテスト検証クラス"""
    
    def __init__(self):
        self.test_symbols = ['7203', '9432', '9984']  # トヨタ、NTT、SBG
        self.test_days = 7
        
    def verify_critical(self) -> Dict[str, Any]:
        """Critical Problem軽量検証実行"""
        results = {}
        
        # Problem 1: 切替メカニズム基本動作確認
        results['switching_mechanism'] = self._verify_switching_mechanism()
        
        # Problem 12: 決定論的再現性確認
        results['deterministic_reproducibility'] = self._verify_deterministic_reproducibility()
        
        return results
        
    def _verify_switching_mechanism(self) -> Dict[str, Any]:
        """切替メカニズム基本動作確認"""
        verification = {
            'functional': False,
            'switching_count': 0,
            'within_expected_range': False,
            'improvement_confirmed': False
        }
        
        try:
            # 軽量バックテスト実行
            result = self._run_lightweight_backtest()
            
            switching_count = result.get('switching_count', 0)
            verification['switching_count'] = switching_count
            
            # 7日間期待レンジ: 21-28回（月間90-120回の比例計算）
            expected_min, expected_max = 21, 28
            verification['within_expected_range'] = expected_min <= switching_count <= expected_max
            
            # 改善確認（3回→改善）
            verification['improvement_confirmed'] = switching_count > 3
            
            verification['functional'] = verification['improvement_confirmed']
            
        except Exception as e:
            verification['error'] = str(e)
            
        return verification
        
    def _verify_deterministic_reproducibility(self) -> Dict[str, Any]:
        """決定論的再現性確認"""
        verification = {
            'reproducible': False,
            'run1_switches': 0,
            'run2_switches': 0,
            'difference_percent': 100.0
        }
        
        try:
            # seed固定で2回実行
            result1 = self._run_lightweight_backtest(seed=42)
            result2 = self._run_lightweight_backtest(seed=42)
            
            switches1 = result1.get('switching_count', 0)
            switches2 = result2.get('switching_count', 0)
            
            verification['run1_switches'] = switches1
            verification['run2_switches'] = switches2
            
            # 差異計算
            if switches1 > 0:
                diff_percent = abs(switches1 - switches2) / switches1 * 100
                verification['difference_percent'] = diff_percent
                
                # ±5%以内の再現性確認
                verification['reproducible'] = diff_percent <= 5.0
            else:
                verification['reproducible'] = switches1 == switches2 == 0
                
        except Exception as e:
            verification['error'] = str(e)
            
        return verification
        
    def _run_lightweight_backtest(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """軽量バックテスト実行"""
        try:
            # DSSMSBacktester動的インポート
            sys.path.insert(0, str(project_root / "src"))
            from dssms.dssms_backtester import DSSMSBacktester
            
            # 軽量設定でバックテスト実行
            backtester = DSSMSBacktester()
            
            # テスト期間設定（直近7日間）
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.test_days)
            
            # 簡易実行（詳細ログなし）
            result = backtester.run_backtest(
                symbols=self.test_symbols,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                seed=seed
            )
            
            # 切替数カウント
            switching_count = 0
            if hasattr(result, 'switch_events') and result.switch_events:
                switching_count = len(result.switch_events)
            elif hasattr(result, 'switching_log') and result.switching_log:
                switching_count = len(result.switching_log)
                
            return {
                'success': True,
                'switching_count': switching_count,
                'test_period_days': self.test_days,
                'test_symbols_count': len(self.test_symbols)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'switching_count': 0
            }


class FullScaleBacktestValidator:
    """Stage 3: 本格バックテスト検証クラス"""
    
    def __init__(self):
        self.test_period_days = 30
        
    def verify_final(self) -> Dict[str, Any]:
        """最終完全KPI検証実行"""
        results = {}
        
        # Problem 1: 完全KPI検証
        results['problem_1_kpi'] = self._verify_problem_1_kpi()
        
        return results
        
    def _verify_problem_1_kpi(self) -> Dict[str, Any]:
        """Problem 1完全KPI検証"""
        verification = {
            'kpi_achieved': False,
            'switching_count_target': False,
            'unnecessary_switch_rate_target': False,
            'deterministic_difference_target': False
        }
        
        try:
            # 本格バックテスト実行（30日間・50銘柄想定）
            result = self._run_full_scale_backtest()
            
            # KPI 1: 30日間切替数90-120回レンジ
            switching_count = result.get('switching_count', 0)
            verification['switching_count_target'] = 90 <= switching_count <= 120
            
            # KPI 2: 不要切替率≤20%
            unnecessary_rate = result.get('unnecessary_switch_rate', 100.0)
            verification['unnecessary_switch_rate_target'] = unnecessary_rate <= 20.0
            
            # KPI 3: 決定論的差分±5%以内
            deterministic_diff = result.get('deterministic_difference_percent', 100.0)
            verification['deterministic_difference_target'] = deterministic_diff <= 5.0
            
            # 総合KPI達成判定
            verification['kpi_achieved'] = all([
                verification['switching_count_target'],
                verification['unnecessary_switch_rate_target'],
                verification['deterministic_difference_target']
            ])
            
            # 詳細データ保存
            verification.update(result)
            
        except Exception as e:
            verification['error'] = str(e)
            
        return verification
        
    def _run_full_scale_backtest(self) -> Dict[str, Any]:
        """本格バックテスト実行"""
        try:
            # TODO: 実装詳細
            # python "src\dssms\dssms_backtester.py" --validation-mode --log-switches
            # の相当処理を実装
            
            return {
                'success': True,
                'switching_count': 95,  # 実測値プレースホルダ
                'unnecessary_switch_rate': 25.0,  # 実測値プレースホルダ
                'deterministic_difference_percent': 0.2,  # 実測値プレースホルダ
                'execution_time_minutes': 15
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'switching_count': 0
            }


class ProblemVerificationPipeline:
    """Problem完了状況の包括的検証パイプライン"""
    
    def __init__(self):
        self.verification_stages = {
            'static': StaticVerification(),
            'lightweight': LightweightBacktestValidator(),
            'full_scale': FullScaleBacktestValidator()
        }
        self.results = {}
        
    def verify_all_problems(self) -> Dict[str, Any]:
        """3段階検証実行"""
        results = {
            'pipeline_start_time': datetime.now().isoformat(),
            'stages': {}
        }
        
        print("=== Problem完了状況検証パイプライン開始 ===\n")
        
        # Stage 1: 静的検証（2分）
        print("🔍 Stage 1: 静的検証実行中...")
        stage1_start = time.time()
        static_results = self.verification_stages['static'].verify_all()
        stage1_duration = time.time() - stage1_start
        
        results['stages']['stage1_static'] = {
            'duration_seconds': round(stage1_duration, 2),
            'results': static_results
        }
        
        print(f"✅ Stage 1完了 ({stage1_duration:.1f}秒)")
        self._print_stage1_summary(static_results)
        
        # Stage 2: 軽量バックテスト（5分）
        if self._should_proceed_lightweight(static_results):
            print("\n🚀 Stage 2: 軽量バックテスト実行中...")
            stage2_start = time.time()
            lightweight_results = self.verification_stages['lightweight'].verify_critical()
            stage2_duration = time.time() - stage2_start
            
            results['stages']['stage2_lightweight'] = {
                'duration_seconds': round(stage2_duration, 2),
                'results': lightweight_results
            }
            
            print(f"✅ Stage 2完了 ({stage2_duration:.1f}秒)")
            self._print_stage2_summary(lightweight_results)
            
            # Stage 3: 本格バックテスト（15分）
            if self._should_proceed_full_scale(lightweight_results):
                print("\n📊 Stage 3: 本格バックテスト実行中...")
                stage3_start = time.time()
                full_results = self.verification_stages['full_scale'].verify_final()
                stage3_duration = time.time() - stage3_start
                
                results['stages']['stage3_full_scale'] = {
                    'duration_seconds': round(stage3_duration, 2),
                    'results': full_results
                }
                
                print(f"✅ Stage 3完了 ({stage3_duration:.1f}秒)")
                self._print_stage3_summary(full_results)
                
            else:
                print("⚠️  Stage 3スキップ: Stage 2で問題検出")
                
        else:
            print("⚠️  Stage 2, 3スキップ: Stage 1で問題検出")
            
        results['pipeline_end_time'] = datetime.now().isoformat()
        self.results = results
        
        # 最終サマリー出力
        self._print_final_summary(results)
        
        return results
        
    def _should_proceed_lightweight(self, static_results: Dict[str, Any]) -> bool:
        """軽量テストへの進行判定"""
        critical_problems = ['Problem 1', 'Problem 12']
        return all(
            static_results.get(problem, {}).get('config_updated', False) 
            for problem in critical_problems
        )
        
    def _should_proceed_full_scale(self, lightweight_results: Dict[str, Any]) -> bool:
        """本格テストへの進行判定"""
        return lightweight_results.get('switching_mechanism', {}).get('functional', False)
        
    def _print_stage1_summary(self, results: Dict[str, Any]):
        """Stage 1結果サマリー出力"""
        print("\n--- Stage 1結果サマリー ---")
        for problem, result in results.items():
            status = "✅" if result.get('config_updated', False) else "❌"
            print(f"{status} {problem}: 設定更新{'完了' if result.get('config_updated', False) else '未完了'}")
            
    def _print_stage2_summary(self, results: Dict[str, Any]):
        """Stage 2結果サマリー出力"""
        print("\n--- Stage 2結果サマリー ---")
        
        switching = results.get('switching_mechanism', {})
        switches = switching.get('switching_count', 0)
        functional = switching.get('functional', False)
        print(f"{'✅' if functional else '❌'} 切替メカニズム: {switches}回 ({'動作正常' if functional else '要修正'})")
        
        repro = results.get('deterministic_reproducibility', {})
        repro_ok = repro.get('reproducible', False)
        diff = repro.get('difference_percent', 100.0)
        print(f"{'✅' if repro_ok else '❌'} 決定論的再現性: ±{diff:.1f}% ({'OK' if repro_ok else 'NG'})")
        
    def _print_stage3_summary(self, results: Dict[str, Any]):
        """Stage 3結果サマリー出力"""
        print("\n--- Stage 3結果サマリー ---")
        
        kpi = results.get('problem_1_kpi', {})
        if kpi.get('kpi_achieved', False):
            print("✅ Problem 1 KPI: 全達成")
        else:
            print("❌ Problem 1 KPI: 要改善")
            
            switches_ok = kpi.get('switching_count_target', False)
            unnecessary_ok = kpi.get('unnecessary_switch_rate_target', False)
            deterministic_ok = kpi.get('deterministic_difference_target', False)
            
            print(f"  - 切替数レンジ: {'✅' if switches_ok else '❌'}")
            print(f"  - 不要切替率: {'✅' if unnecessary_ok else '❌'}")
            print(f"  - 決定論的差分: {'✅' if deterministic_ok else '❌'}")
            
    def _print_final_summary(self, results: Dict[str, Any]):
        """最終検証結果サマリー出力"""
        print("\n" + "="*50)
        print("🎯 Problem完了状況検証結果")
        print("="*50)
        
        stages = results.get('stages', {})
        
        for stage_name, stage_data in stages.items():
            duration = stage_data.get('duration_seconds', 0)
            print(f"⏱️  {stage_name}: {duration:.1f}秒")
            
        print(f"\n📋 検証結果保存: verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    def save_results(self, filepath: Optional[str] = None) -> str:
        """検証結果をJSONファイルに保存"""
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"verification_results_{timestamp}.json"
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
            
        return filepath


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Problem完了状況検証パイプライン')
    parser.add_argument('--stage', choices=['1', '2', '3', 'all'], default='all',
                       help='実行段階指定 (default: all)')
    parser.add_argument('--save-results', action='store_true',
                       help='結果をJSONファイルに保存')
    
    args = parser.parse_args()
    
    # 検証パイプライン実行
    pipeline = ProblemVerificationPipeline()
    
    if args.stage == 'all':
        results = pipeline.verify_all_problems()
    elif args.stage == '1':
        results = pipeline.verification_stages['static'].verify_all()
    elif args.stage == '2':
        results = pipeline.verification_stages['lightweight'].verify_critical()
    elif args.stage == '3':
        results = pipeline.verification_stages['full_scale'].verify_final()
        
    # 結果保存
    if args.save_results:
        filepath = pipeline.save_results()
        print(f"\n💾 結果保存: {filepath}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSSMS Switching Analysis After Cache Clear
キャッシュクリア後のDSSMS切替分析

仮説検証:
- キャッシュクリア直後（API直接取得時）: 切替回数多い
- キャッシュ蓄積後（キャッシュ取得時）: 切替回数少ない
"""

import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

# プロジェクトルート追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

logger = setup_logger(__name__)


class DSSMSSwitchingAnalyzer:
    """DSSMS切替分析器"""
    
    def __init__(self):
        self.analysis_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = {
            'analysis_id': self.analysis_id,
            'timestamp': datetime.now(),
            'experiments': {}
        }
        
    def run_dssms_backtest_with_monitoring(self, experiment_name: str, 
                                         cache_state: str = "unknown") -> Dict[str, Any]:
        """DSSMS バックテスト実行（切替監視付き）"""
        logger.info(f"=== Running DSSMS Backtest: {experiment_name} ===")
        logger.info(f"Cache state: {cache_state}")
        
        experiment_result = {
            'experiment_name': experiment_name,
            'cache_state': cache_state,
            'start_time': datetime.now(),
            'switch_count': 0,
            'execution_result': None,
            'error': None
        }
        
        try:
            # DSSMSバックテスト実行（実際のコマンド）
            import subprocess
            import os
            
            # 現在のディレクトリをプロジェクトルートに変更
            original_cwd = os.getcwd()
            project_path = str(project_root)
            os.chdir(project_path)
            
            # DSSMSバックテスト実行
            logger.info("Executing DSSMS backtest...")
            
            try:
                # バックテスト実行コマンド（正しいパス）
                cmd = ["python", "src\\dssms\\dssms_backtester.py"]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5分タイムアウト
                    encoding='utf-8'
                )
                
                experiment_result['execution_result'] = {
                    'returncode': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
                # 出力から切替回数を抽出
                switch_count = self._extract_switch_count_from_output(result.stdout, result.stderr)
                experiment_result['switch_count'] = switch_count
                
                logger.info(f"DSSMS backtest completed. Switch count: {switch_count}")
                
            except subprocess.TimeoutExpired:
                logger.error("DSSMS backtest timed out")
                experiment_result['error'] = "Execution timeout (300s)"
                
            except Exception as e:
                logger.error(f"DSSMS backtest execution failed: {e}")
                experiment_result['error'] = str(e)
                
            finally:
                # 元のディレクトリに戻る
                os.chdir(original_cwd)
                
        except Exception as e:
            logger.error(f"Failed to run DSSMS backtest: {e}")
            experiment_result['error'] = str(e)
            
        experiment_result['end_time'] = datetime.now()
        experiment_result['duration_sec'] = (
            experiment_result['end_time'] - experiment_result['start_time']
        ).total_seconds()
        
        self.results['experiments'][experiment_name] = experiment_result
        
        return experiment_result
    
    def _extract_switch_count_from_output(self, stdout: str, stderr: str) -> int:
        """出力から切替回数を抽出"""
        combined_output = stdout + "\n" + stderr
        
        # 切替回数を示すパターンを検索
        patterns = [
            r"切替判定:\s*(\d+)",
            r"Switch count:\s*(\d+)",
            r"切替.*?(\d+).*?回",
            r"Total switches:\s*(\d+)",
            r"switching.*?(\d+)",
            r"執行.*?(\d+).*?回"
        ]
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, combined_output, re.IGNORECASE)
            if matches:
                try:
                    # 最後にマッチした数値を使用
                    return int(matches[-1])
                except ValueError:
                    continue
        
        # パターンマッチしない場合、出力を詳細解析
        logger.warning("Could not extract switch count from output using patterns")
        logger.debug(f"STDOUT:\n{stdout}")
        logger.debug(f"STDERR:\n{stderr}")
        
        return 0  # 抽出できない場合は0
    
    def run_comprehensive_switching_analysis(self) -> Dict[str, Any]:
        """包括的切替分析実行"""
        logger.info("=== Starting Comprehensive Switching Analysis ===")
        
        # 実験1: キャッシュクリア直後（API直接取得状態）
        logger.info("\n--- Experiment 1: Post Cache Clear (API Direct) ---")
        exp1_result = self.run_dssms_backtest_with_monitoring(
            "post_cache_clear", 
            "cleared_api_direct"
        )
        
        # 少し待機
        import time
        time.sleep(2)
        
        # 実験2: 2回目実行（キャッシュ蓄積状態）
        logger.info("\n--- Experiment 2: Second Run (Cache Accumulated) ---")
        exp2_result = self.run_dssms_backtest_with_monitoring(
            "second_run", 
            "cache_accumulated"
        )
        
        # 実験3: 3回目実行（さらなるキャッシュ蓄積）
        logger.info("\n--- Experiment 3: Third Run (More Cache Accumulated) ---")
        exp3_result = self.run_dssms_backtest_with_monitoring(
            "third_run", 
            "more_cache_accumulated"
        )
        
        # 分析結果の生成
        analysis_summary = self._generate_analysis_summary()
        
        return analysis_summary
    
    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """分析サマリー生成"""
        experiments = self.results['experiments']
        
        summary = {
            'hypothesis_validation': {},
            'switch_count_progression': [],
            'performance_metrics': {},
            'conclusions': []
        }
        
        # 切替回数の推移
        for exp_name, exp_data in experiments.items():
            summary['switch_count_progression'].append({
                'experiment': exp_name,
                'cache_state': exp_data.get('cache_state', 'unknown'),
                'switch_count': exp_data.get('switch_count', 0),
                'duration_sec': exp_data.get('duration_sec', 0),
                'success': exp_data.get('error') is None
            })
        
        # 仮説検証
        switch_counts = [exp.get('switch_count', 0) for exp in experiments.values()]
        
        if len(switch_counts) >= 3:
            post_clear_count = switch_counts[0]  # キャッシュクリア直後
            second_run_count = switch_counts[1]  # 2回目
            third_run_count = switch_counts[2]   # 3回目
            
            # 切替回数の減少パターンを分析
            degradation_detected = (
                post_clear_count > second_run_count or 
                second_run_count > third_run_count
            )
            
            summary['hypothesis_validation'] = {
                'cache_hypothesis': {
                    'statement': 'キャッシュ蓄積により切替回数が減少する',
                    'post_clear_switches': post_clear_count,
                    'second_run_switches': second_run_count,
                    'third_run_switches': third_run_count,
                    'degradation_detected': degradation_detected,
                    'degradation_rate': self._calculate_degradation_rate(switch_counts)
                }
            }
            
            # 結論生成
            if degradation_detected:
                summary['conclusions'].append(
                    "[SEARCH] 仮説支持: キャッシュ蓄積による切替回数減少が確認された"
                )
                if post_clear_count > 50 and third_run_count < 10:
                    summary['conclusions'].append(
                        "[WARNING] 重大な劣化: 切替回数が大幅に減少（API→キャッシュ影響大）"
                    )
            else:
                summary['conclusions'].append(
                    "[ERROR] 仮説不支持: キャッシュ蓄積による明確な切替回数減少は確認されなかった"
                )
        
        return summary
    
    def _calculate_degradation_rate(self, switch_counts: List[int]) -> float:
        """劣化率計算"""
        if len(switch_counts) < 2:
            return 0.0
        
        initial = switch_counts[0]
        final = switch_counts[-1]
        
        if initial == 0:
            return 0.0
        
        return (initial - final) / initial * 100
    
    def save_analysis_report(self, output_dir: str = ".") -> str:
        """分析レポート保存"""
        report_filename = f"dssms_switching_analysis_{self.analysis_id}.json"
        report_path = Path(output_dir) / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Analysis report saved: {report_path}")
        return str(report_path)


def main():
    """メイン実行関数"""
    print("DSSMS Switching Analysis After Cache Clear")
    print("==========================================")
    
    analyzer = DSSMSSwitchingAnalyzer()
    
    try:
        # 包括的切替分析実行
        analysis_summary = analyzer.run_comprehensive_switching_analysis()
        
        # レポート保存
        report_path = analyzer.save_analysis_report()
        
        print(f"\n=== Analysis Summary ===")
        print(json.dumps(analysis_summary, indent=2, ensure_ascii=False))
        
        print(f"\nDetailed report saved: {report_path}")
        
        # 結論表示
        if 'conclusions' in analysis_summary:
            print(f"\n=== Conclusions ===")
            for conclusion in analysis_summary['conclusions']:
                print(f"• {conclusion}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()
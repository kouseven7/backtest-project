#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSSMS vs Ultra Simple 差分解析ツール
目的: 切替回数117→3回劣化の根本原因特定
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

# プロジェクトルートを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# UltraSimple版をインポート
from ultra_simple_ranking_test import UltraSimpleRanking

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DSSMSAnalyzer:
    """DSSMS vs UltraSimple 差分解析"""
    
    def __init__(self):
        self.comparison_results = {}
        
    def analyze_dssms_switching_logic(self) -> Dict[str, Any]:
        """DSSMSの切替ロジックを解析"""
        
        logger.info("=== DSSMS切替ロジック解析開始 ===")
        
        try:
            # DSSMS backtesterをインポート（エラーハンドリング付き）
            try:
                from src.dssms.dssms_backtester import DSSMSBacktester
                logger.info("[OK] DSSMSBacktester インポート成功")
            except ImportError as e:
                logger.error(f"[ERROR] DSSMSBacktester インポート失敗: {e}")
                return {
                    'status': 'import_failed',
                    'error': str(e),
                    'analysis': 'DSSMSモジュールが見つからない'
                }
            
            # 基本設定（UltraSimpleと同等）
            symbols = ['6758', '7203', '8306', '9984']
            backtester = DSSMSBacktester()
            
            # DSSMS実行（短縮版・100日）
            logger.info("DSSMS 100日間実行開始...")
            start_time = datetime.now()
            
            # 実際の実行は重いので、まず既存ログファイルを確認
            return self._analyze_existing_dssms_results()
            
        except Exception as e:
            logger.error(f"DSSMS解析エラー: {e}")
            return {
                'status': 'analysis_failed',
                'error': str(e)
            }
    
    def _analyze_existing_dssms_results(self) -> Dict[str, Any]:
        """既存のDSSMS実行結果を解析"""
        
        # 最新のDSSMS実行結果を探す
        result_patterns = [
            'dssms_detailed_log_*.log',
            'dssms_backtester_*.log', 
            'switching_log.txt',
            'baseline_switching_log.txt'
        ]
        
        found_files = []
        for pattern in result_patterns:
            import glob
            matches = glob.glob(pattern.replace('*', '*'))
            found_files.extend(matches)
            
        logger.info(f"発見されたログファイル: {found_files}")
        
        if not found_files:
            return {
                'status': 'no_logs_found',
                'message': 'DSSMSの実行ログが見つかりません',
                'recommendation': '新規DSSMS実行が必要'
            }
        
        # 最新ファイルを解析
        latest_file = max(found_files, key=os.path.getmtime)
        logger.info(f"最新ログファイル分析: {latest_file}")
        
        return self._extract_switching_info_from_log(latest_file)
    
    def _extract_switching_info_from_log(self, log_file: str) -> Dict[str, Any]:
        """ログファイルから切替情報を抽出"""
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 切替パターンを検索
            switch_patterns = [
                'switching from',
                'switching to', 
                'Symbol changed',
                'top_symbol changed',
                'Selected symbol:'
            ]
            
            switches = []
            lines = content.split('\n') 
            
            for i, line in enumerate(lines):
                for pattern in switch_patterns:
                    if pattern.lower() in line.lower():
                        switches.append({
                            'line_number': i + 1,
                            'content': line.strip(),
                            'pattern': pattern
                        })
                        
            return {
                'status': 'log_analyzed',
                'log_file': log_file,
                'total_lines': len(lines),
                'switch_candidates': len(switches),
                'switches': switches[:10],  # 最初の10件
                'analysis': f'{len(switches)}個の切替候補を発見'
            }
            
        except Exception as e:
            logger.error(f"ログ解析エラー: {e}")
            return {
                'status': 'log_analysis_failed',
                'error': str(e)
            }
    
    def compare_ultra_vs_dssms(self) -> Dict[str, Any]:
        """UltraSimple vs DSSMS の詳細比較"""
        
        logger.info("=== Ultra Simple vs DSSMS 比較分析 ===")
        
        # Step 1: UltraSimple実行（ベースライン）
        ultra_simple = UltraSimpleRanking(['6758', '7203', '8306', '9984'])
        ultra_results = ultra_simple.simulate_switches(100)
        
        logger.info(f"[OK] UltraSimple: {ultra_results['switch_count']}回切替")
        
        # Step 2: DSSMS解析
        dssms_analysis = self.analyze_dssms_switching_logic()
        
        # Step 3: 比較結果
        comparison = {
            'ultra_simple': {
                'switches': ultra_results['switch_count'],
                'status': 'success' if ultra_results['success'] else 'failed',
                'pattern': 'deterministic_10day_cycle'
            },
            'dssms': dssms_analysis,
            'difference_analysis': self._analyze_differences(ultra_results, dssms_analysis)
        }
        
        return comparison
    
    def _analyze_differences(self, ultra_results: Dict, dssms_analysis: Dict) -> Dict[str, Any]:
        """差分詳細解析"""
        
        if dssms_analysis.get('status') != 'log_analyzed':
            return {
                'type': 'comparison_impossible',
                'reason': 'DSSMS解析データが不完全',
                'ultra_switches': ultra_results['switch_count'],
                'dssms_data': 'insufficient'
            }
        
        # 切替回数比較
        ultra_switches = ultra_results['switch_count']  # 期待: 10回
        dssms_candidates = dssms_analysis.get('switch_candidates', 0)
        
        return {
            'type': 'switch_count_analysis',
            'ultra_switches': ultra_switches,
            'dssms_switch_candidates': dssms_candidates,
            'ratio': dssms_candidates / ultra_switches if ultra_switches > 0 else 0,
            'severity': 'critical' if dssms_candidates < ultra_switches * 0.5 else 'normal',
            'recommendations': self._generate_recommendations(ultra_switches, dssms_candidates)
        }
    
    def _generate_recommendations(self, ultra_switches: int, dssms_candidates: int) -> List[str]:
        """修正推奨事項を生成"""
        
        recommendations = []
        
        if dssms_candidates < ultra_switches * 0.3:
            recommendations.append("[ALERT] CRITICAL: 切替回数が70%以上減少")
            recommendations.append("→ DSSMS切替判定ロジックに根本的問題")
            recommendations.append("→ UltraSimpleの決定論的ロジック適用を検討")
            
        if dssms_candidates == 0:
            recommendations.append("[FIRE] EMERGENCY: 切替が全く発生していない")
            recommendations.append("→ top_symbol=None または無限ループの可能性")
            
        recommendations.append("[OK] 詳細調査項目:")
        recommendations.append("  1. DSSMSのshould_switch()ロジック確認")
        recommendations.append("  2. top_symbol選択プロセス検証")
        recommendations.append("  3. 切替条件の複雑性簡素化")
        
        return recommendations

def main():
    """メイン解析実行"""
    
    print("[SEARCH] DSSMS vs Ultra Simple 差分解析")
    print("=" * 60)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    analyzer = DSSMSAnalyzer()
    
    try:
        # 比較分析実行
        comparison_results = analyzer.compare_ultra_vs_dssms()
        
        # 結果表示
        print("[CHART] 比較結果:")
        print("-" * 40)
        
        # UltraSimple結果
        ultra = comparison_results['ultra_simple']
        print(f"[OK] Ultra Simple: {ultra['switches']}回切替 ({ultra['status']})")
        print(f"   パターン: {ultra['pattern']}")
        
        # DSSMS結果  
        dssms = comparison_results['dssms']
        print(f"\n[SEARCH] DSSMS分析: {dssms.get('status', 'unknown')}")
        
        if dssms.get('status') == 'log_analyzed':
            print(f"   ログファイル: {dssms['log_file']}")
            print(f"   切替候補: {dssms['switch_candidates']}個")
            
        # 差分解析
        diff = comparison_results['difference_analysis']
        print(f"\n⚖️  差分解析:")
        print(f"   Ultra Simple: {diff.get('ultra_switches', 'N/A')}回")
        print(f"   DSSMS候補: {diff.get('dssms_switch_candidates', 'N/A')}個")
        
        if 'ratio' in diff:
            ratio_pct = diff['ratio'] * 100
            print(f"   比率: {ratio_pct:.1f}%")
            print(f"   重要度: {diff.get('severity', 'unknown')}")
        
        # 推奨事項
        if 'recommendations' in diff:
            print(f"\n[IDEA] 推奨事項:")
            for rec in diff['recommendations']:
                print(f"   {rec}")
        
        return comparison_results
        
    except Exception as e:
        print(f"[ERROR] 解析エラー: {e}")
        logger.error(f"メイン解析失敗: {e}")
        return None

if __name__ == "__main__":
    results = main()
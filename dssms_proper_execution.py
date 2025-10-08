#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSSMS 正しい引数による実行版
目的: simulate_dynamic_selectionを正しい引数で実行して切替解析
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

# UltraSimple版をインポート
from ultra_simple_ranking_test import UltraSimpleRanking

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DSSMSProperExecution:
    """DSSMS正しい引数による実行"""
    
    def __init__(self):
        self.symbols = ['6758', '7203', '8306', '9984']
        
    def execute_dssms_with_proper_args(self) -> Dict[str, Any]:
        """正しい引数でDSSMS実行"""
        
        logger.info("=== DSSMS正しい引数による実行 ===")
        
        try:
            from src.dssms.dssms_backtester import DSSMSBacktester
            
            backtester = DSSMSBacktester()
            logger.info("[OK] DSSMSBacktester初期化完了")
            
            # 正しい引数設定
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 15)  # 10営業日程度
            symbol_universe = self.symbols
            strategies = None  # オプショナル
            
            logger.info(f"実行設定:")
            logger.info(f"  start_date: {start_date}")
            logger.info(f"  end_date: {end_date}")
            logger.info(f"  symbol_universe: {symbol_universe}")
            logger.info(f"  strategies: {strategies}")
            
            # DSSMS実行
            logger.info("[ROCKET] DSSMS実行開始...")
            start_time = datetime.now()
            
            result = backtester.simulate_dynamic_selection(
                start_date=start_date,
                end_date=end_date,
                symbol_universe=symbol_universe,
                strategies=strategies
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[OK] DSSMS実行完了: {execution_time:.2f}秒")
            
            # 結果解析
            switch_analysis = self._analyze_dssms_result(result)
            
            return {
                'status': 'success',
                'execution_time': execution_time,
                'result_type': type(result).__name__,
                'result_keys': list(result.keys()) if isinstance(result, dict) else None,
                'switch_analysis': switch_analysis
            }
            
        except Exception as e:
            logger.error(f"DSSMS実行エラー: {e}")
            return {
                'status': 'execution_failed',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def _analyze_dssms_result(self, result) -> Dict[str, Any]:
        """DSSMS結果から切替情報を解析"""
        
        logger.info("=== DSSMS結果解析開始 ===")
        
        if not isinstance(result, dict):
            return {
                'status': 'invalid_result_type',
                'type': type(result).__name__,
                'message': '結果が辞書型ではありません'
            }
        
        # 辞書のキーを調査
        keys = list(result.keys())
        logger.info(f"結果キー: {keys}")
        
        # 切替関連情報を探す
        switch_candidates = []
        for key in keys:
            if any(switch_keyword in key.lower() for switch_keyword in 
                   ['switch', 'selection', 'symbol', 'position', 'trade']):
                switch_candidates.append(key)
        
        logger.info(f"切替候補キー: {switch_candidates}")
        
        # 各候補キーの内容を調査
        switch_data = {}
        for key in switch_candidates[:5]:  # 最初の5個
            try:
                value = result[key]
                switch_data[key] = {
                    'type': type(value).__name__,
                    'content': self._summarize_value(value)
                }
                logger.info(f"  {key}: {type(value).__name__}")
            except Exception as e:
                logger.warning(f"  {key}: 読み取りエラー - {e}")
        
        # 切替回数推定
        estimated_switches = self._estimate_switch_count(result, switch_data)
        
        return {
            'status': 'analyzed',
            'total_keys': len(keys),
            'switch_candidate_keys': switch_candidates,
            'switch_data': switch_data,
            'estimated_switches': estimated_switches
        }
    
    def _summarize_value(self, value) -> Dict[str, Any]:
        """値の要約"""
        
        if value is None:
            return {'summary': 'None'}
        elif isinstance(value, (int, float, str, bool)):
            return {'summary': str(value)[:100]}
        elif isinstance(value, list):
            return {
                'length': len(value),
                'first_items': value[:3] if len(value) > 0 else [],
                'sample': str(value[:3])
            }
        elif isinstance(value, dict):
            return {
                'keys': list(value.keys())[:5],
                'size': len(value)
            }
        else:
            return {
                'type': type(value).__name__,
                'str_repr': str(value)[:100]
            }
    
    def _estimate_switch_count(self, result: Dict, switch_data: Dict) -> Dict[str, Any]:
        """切替回数推定"""
        
        # 直接的な切替カウントを探す
        for key in result.keys():
            if 'switch' in key.lower() and isinstance(result[key], (int, float)):
                return {
                    'method': 'direct_count',
                    'key': key,
                    'count': result[key]
                }
        
        # 選択履歴から推定
        for key in result.keys():
            if any(pattern in key.lower() for pattern in ['selection', 'position', 'symbol']):
                value = result[key]
                if isinstance(value, list) and len(value) > 1:
                    # リスト内の変化をカウント
                    changes = 0
                    for i in range(1, len(value)):
                        try:
                            if value[i] != value[i-1]:
                                changes += 1
                        except:
                            continue
                    
                    if changes > 0:
                        return {
                            'method': 'history_analysis',
                            'key': key,
                            'count': changes,
                            'total_entries': len(value)
                        }
        
        return {
            'method': 'unknown',
            'count': 'unable_to_determine'
        }
    
    def run_complete_comparison(self) -> Dict[str, Any]:
        """完全な比較実行"""
        
        logger.info("=== 完全比較実行 ===")
        
        # Step 1: UltraSimple実行
        ultra_simple = UltraSimpleRanking(self.symbols)
        ultra_results = ultra_simple.simulate_switches(10)
        
        logger.info(f"[OK] UltraSimple: {ultra_results['switch_count']}回切替")
        
        # Step 2: DSSMS実行
        dssms_results = self.execute_dssms_with_proper_args()
        
        # Step 3: 比較分析
        comparison = self._perform_final_comparison(ultra_results, dssms_results)
        
        return {
            'ultra_simple': {
                'switches': ultra_results['switch_count'],
                'success': ultra_results['success'],
                'expected': 1
            },
            'dssms': dssms_results,
            'comparison': comparison
        }
    
    def _perform_final_comparison(self, ultra_results: Dict, dssms_results: Dict) -> Dict[str, Any]:
        """最終比較分析"""
        
        ultra_switches = ultra_results['switch_count']
        
        if dssms_results['status'] != 'success':
            return {
                'verdict': 'DSSMS_FAILED',
                'message': f"DSSMS実行失敗: {dssms_results.get('error', 'unknown')}",
                'ultra_switches': ultra_switches,
                'dssms_switches': None
            }
        
        switch_analysis = dssms_results.get('switch_analysis', {})
        estimated = switch_analysis.get('estimated_switches', {})
        
        if estimated.get('count') == 'unable_to_determine':
            return {
                'verdict': 'ANALYSIS_INCOMPLETE',
                'message': 'DSSMS結果から切替回数を特定できません',
                'ultra_switches': ultra_switches,
                'dssms_switches': None,
                'available_keys': switch_analysis.get('switch_candidate_keys', [])
            }
        
        dssms_switches = estimated.get('count', 0)
        
        return {
            'verdict': 'COMPARISON_COMPLETE',
            'ultra_switches': ultra_switches,
            'dssms_switches': dssms_switches,
            'difference': abs(ultra_switches - dssms_switches),
            'ratio': dssms_switches / ultra_switches if ultra_switches > 0 else 0,
            'analysis_method': estimated.get('method', 'unknown'),
            'success': dssms_switches == ultra_switches
        }

def main():
    """メイン実行"""
    
    print("[TARGET] DSSMS 正しい引数による実行版")
    print("=" * 50)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目的: simulate_dynamic_selectionを正しい引数で実行")
    print()
    
    executor = DSSMSProperExecution()
    
    try:
        # 完全比較実行
        results = executor.run_complete_comparison()
        
        # 結果表示
        print("[CHART] 完全比較結果:")
        print("-" * 30)
        
        # UltraSimple結果
        ultra = results['ultra_simple']
        print(f"[OK] Ultra Simple: {ultra['switches']}回切替 (期待: {ultra['expected']}回)")
        print(f"   成功: {ultra['success']}")
        
        # DSSMS結果
        dssms = results['dssms']
        print(f"\n[SEARCH] DSSMS実行: {dssms['status']}")
        
        if dssms['status'] == 'success':
            print(f"   実行時間: {dssms['execution_time']:.2f}秒")
            print(f"   結果型: {dssms['result_type']}")
            
            if 'result_keys' in dssms and dssms['result_keys']:
                print(f"   結果キー数: {len(dssms['result_keys'])}")
                print(f"   主要キー: {dssms['result_keys'][:5]}")
            
            switch_analysis = dssms.get('switch_analysis', {})
            if switch_analysis.get('status') == 'analyzed':
                estimated = switch_analysis.get('estimated_switches', {})
                print(f"   推定切替回数: {estimated.get('count', 'N/A')}")
                print(f"   解析手法: {estimated.get('method', 'N/A')}")
        else:
            print(f"   エラー: {dssms.get('error', 'unknown')}")
        
        # 比較結果
        comparison = results['comparison']
        print(f"\n⚖️  最終比較:")
        print(f"   判定: {comparison['verdict']}")
        print(f"   メッセージ: {comparison.get('message', 'N/A')}")
        
        if 'difference' in comparison:
            print(f"   Ultra Simple: {comparison['ultra_switches']}回")
            print(f"   DSSMS: {comparison['dssms_switches']}回")
            print(f"   差分: {comparison['difference']}回")
            print(f"   成功: {comparison['success']}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] 実行エラー: {e}")
        logger.error(f"メイン実行失敗: {e}")
        return None

if __name__ == "__main__":
    results = main()
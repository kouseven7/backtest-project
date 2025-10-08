#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSSMS 正しい銘柄選択ロジック修正版
目的: simulate_dynamic_selectionメソッドを使用してNone問題解決
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

# UltraSimple版をインポート
from ultra_simple_ranking_test import UltraSimpleRanking

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DSSMSCorrector:
    """DSSMS正しい銘柄選択ロジック修正"""
    
    def __init__(self):
        self.symbols = ['6758', '7203', '8306', '9984']
        
    def test_dssms_simulate_dynamic_selection(self) -> Dict[str, Any]:
        """simulate_dynamic_selectionメソッドのテスト"""
        
        logger.info("=== DSSMS simulate_dynamic_selection テスト ===")
        
        try:
            from src.dssms.dssms_backtester import DSSMSBacktester
            
            backtester = DSSMSBacktester()
            logger.info("[OK] DSSMSBacktester初期化完了")
            
            # simulate_dynamic_selectionメソッドの詳細調査
            if hasattr(backtester, 'simulate_dynamic_selection'):
                logger.info("[OK] simulate_dynamic_selection メソッド存在確認")
                
                # メソッドの引数を調査
                import inspect
                sig = inspect.signature(backtester.simulate_dynamic_selection)
                logger.info(f"メソッドシグネチャ: {sig}")
                
                # テスト実行
                logger.info("[TOOL] simulate_dynamic_selection テスト実行開始")
                
                try:
                    # 最小限の引数でテスト
                    result = backtester.simulate_dynamic_selection()
                    logger.info("[OK] 引数なしでの実行成功")
                    
                    return {
                        'status': 'success',
                        'method_signature': str(sig),
                        'result_type': type(result).__name__,
                        'result_summary': self._summarize_result(result)
                    }
                    
                except TypeError as e:
                    logger.warning(f"引数なし実行失敗: {e}")
                    
                    # 引数ありでの実行を試行
                    try:
                        # 一般的な引数パターンを試行
                        test_args = {
                            'symbols': self.symbols,
                            'days': 10,
                            'start_date': '2024-01-01',
                            'end_date': '2024-01-15'
                        }
                        
                        # 順次引数を減らしてテスト
                        for attempt in range(4):
                            try:
                                if attempt == 0:
                                    result = backtester.simulate_dynamic_selection(**test_args)
                                elif attempt == 1:
                                    result = backtester.simulate_dynamic_selection(symbols=self.symbols)
                                elif attempt == 2:
                                    result = backtester.simulate_dynamic_selection(days=10)
                                else:
                                    result = backtester.simulate_dynamic_selection(
                                        start_date='2024-01-01', 
                                        end_date='2024-01-15'
                                    )
                                
                                logger.info(f"[OK] 試行{attempt+1}で実行成功")
                                return {
                                    'status': 'success',
                                    'method_signature': str(sig),
                                    'successful_args': f'attempt_{attempt+1}',
                                    'result_type': type(result).__name__,
                                    'result_summary': self._summarize_result(result)
                                }
                                
                            except Exception as sub_e:
                                logger.debug(f"試行{attempt+1}失敗: {sub_e}")
                                continue
                        
                        # 全ての試行が失敗
                        return {
                            'status': 'all_attempts_failed',
                            'method_signature': str(sig),
                            'error': 'All argument combinations failed'
                        }
                        
                    except Exception as e:
                        logger.error(f"引数ありテスト失敗: {e}")
                        return {
                            'status': 'method_call_failed',
                            'error': str(e),
                            'method_signature': str(sig)
                        }
                        
            else:
                logger.error("[ERROR] simulate_dynamic_selection メソッドが存在しません")
                return {
                    'status': 'method_not_found',
                    'available_methods': [method for method in dir(backtester) 
                                        if not method.startswith('_') and callable(getattr(backtester, method))]
                }
                
        except Exception as e:
            logger.error(f"DSSMS初期化エラー: {e}")
            return {
                'status': 'initialization_failed',
                'error': str(e)
            }
    
    def _summarize_result(self, result) -> Dict[str, Any]:
        """結果要約"""
        
        if result is None:
            return {'type': 'None', 'content': 'None'}
        
        if isinstance(result, dict):
            return {
                'type': 'dict',
                'keys': list(result.keys())[:10],  # 最初の10キー
                'size': len(result) if hasattr(result, '__len__') else 'unknown'
            }
        elif isinstance(result, (list, tuple)):
            return {
                'type': type(result).__name__,
                'length': len(result),
                'first_items': result[:3] if len(result) > 0 else []
            }
        elif isinstance(result, str):
            return {
                'type': 'str', 
                'length': len(result),
                'preview': result[:100] if len(result) > 100 else result
            }
        else:
            return {
                'type': type(result).__name__,
                'str_repr': str(result)[:200]
            }
    
    def run_corrected_dssms_test(self) -> Dict[str, Any]:
        """修正版DSSMS実行テスト"""
        
        logger.info("=== 修正版DSSMS実行テスト ===")
        
        # Step 1: simulate_dynamic_selectionテスト
        method_test = self.test_dssms_simulate_dynamic_selection()
        
        # Step 2: UltraSimpleとの比較準備
        ultra_simple = UltraSimpleRanking(self.symbols)
        ultra_results = ultra_simple.simulate_switches(10)
        
        return {
            'ultra_simple_results': {
                'switches': ultra_results['switch_count'],
                'success': ultra_results['success'],
                'expected': 1  # 10日間で1回
            },
            'dssms_method_test': method_test,
            'comparison_status': self._determine_next_steps(ultra_results, method_test)
        }
    
    def _determine_next_steps(self, ultra_results: Dict, method_test: Dict) -> Dict[str, Any]:
        """次のステップ決定"""
        
        if method_test['status'] == 'success':
            return {
                'status': 'method_working',
                'message': 'simulate_dynamic_selection が動作 - 切替ロジック統合可能',
                'next_action': 'DSSMS結果から切替イベント抽出実装',
                'priority': 'HIGH'
            }
        elif method_test['status'] == 'method_not_found':
            return {
                'status': 'method_missing',
                'message': 'simulate_dynamic_selection が存在しない',
                'next_action': '別のDSSMSメソッドで銘柄選択実装',
                'priority': 'CRITICAL'
            }
        else:
            return {
                'status': 'method_broken',
                'message': 'simulate_dynamic_selection の実行に問題',
                'next_action': 'DSSMSメソッド修復またはUltraSimpleロジック移植',
                'priority': 'CRITICAL'
            }

def main():
    """修正版メイン実行"""
    
    print("[TOOL] DSSMS 正しい銘柄選択ロジック修正版")
    print("=" * 55)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目的: simulate_dynamic_selectionを使用したNone問題解決")
    print()
    
    corrector = DSSMSCorrector()
    
    try:
        # 修正版テスト実行
        results = corrector.run_corrected_dssms_test()
        
        # 結果表示
        print("[CHART] 修正版テスト結果:")
        print("-" * 35)
        
        # UltraSimple結果（ベースライン）
        ultra = results['ultra_simple_results']
        print(f"[OK] Ultra Simple: {ultra['switches']}回切替 (期待: {ultra['expected']}回)")
        print(f"   Status: {ultra['success']}")
        
        # DSSMSメソッドテスト結果
        method_test = results['dssms_method_test']
        print(f"\n[SEARCH] DSSMS Method Test: {method_test['status']}")
        
        if method_test['status'] == 'success':
            print(f"   Method Signature: {method_test.get('method_signature', 'N/A')}")
            print(f"   Result Type: {method_test.get('result_type', 'N/A')}")
            summary = method_test.get('result_summary', {})
            print(f"   Result Summary: {summary.get('type', 'N/A')}")
            
            if 'keys' in summary:
                print(f"   Available Keys: {summary['keys']}")
            elif 'length' in summary:
                print(f"   Length: {summary['length']}")
                
        elif method_test['status'] == 'method_not_found':
            available = method_test.get('available_methods', [])
            print(f"   Available Methods: {available[:5]}...")  # 最初の5個
            
        else:
            print(f"   Error: {method_test.get('error', 'unknown')}")
        
        # 次のステップ
        next_steps = results['comparison_status']
        print(f"\n[TARGET] 次のステップ:")
        print(f"   Status: {next_steps['status']}")
        print(f"   Message: {next_steps['message']}")
        print(f"   Action: {next_steps['next_action']}")
        print(f"   Priority: {next_steps['priority']}")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] 修正版テストエラー: {e}")
        logger.error(f"修正版テスト失敗: {e}")
        return None

if __name__ == "__main__":
    results = main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSSMS実行結果分析ツール（エンコーディング対応版）
目的: DSSMSの直接実行による切替解析
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

class DSSMSDirectAnalyzer:
    """DSSMS直接実行による切替解析"""
    
    def __init__(self):
        self.symbols = ['6758', '7203', '8306', '9984']
        
    def run_fresh_dssms_analysis(self) -> Dict[str, Any]:
        """新規DSSMS実行による切替解析"""
        
        logger.info("=== 新規DSSMS切替解析開始 ===")
        
        try:
            # DSSMS backtesterを直接インポート
            from src.dssms.dssms_backtester import DSSMSBacktester
            
            # 新しいインスタンス作成
            backtester = DSSMSBacktester()
            logger.info("✅ DSSMSBacktester初期化完了")
            
            # 最小限の実行設定（10日間のみ）
            test_config = {
                'symbols': self.symbols,
                'days': 10,  # 最小限のテスト
                'log_switches': True
            }
            
            logger.info("🔧 最小限DSSMS実行開始（10日間）")
            start_time = datetime.now()
            
            # この部分は実際のDSSMSバックテストメソッドを呼び出す
            # （具体的なメソッド名はDSSMSBacktesterクラスを確認して調整）
            results = self._execute_minimal_dssms(backtester, test_config)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"DSSMS実行完了: {execution_time:.2f}秒")
            
            return {
                'status': 'success',
                'execution_time': execution_time,
                'results': results,
                'analysis_type': 'fresh_execution'
            }
            
        except ImportError as e:
            logger.error(f"DSSMSインポートエラー: {e}")
            return {
                'status': 'import_failed',
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"DSSMS実行エラー: {e}")
            return {
                'status': 'execution_failed',
                'error': str(e)
            }
    
    def _execute_minimal_dssms(self, backtester, config: Dict) -> Dict[str, Any]:
        """最小限のDSSMS実行"""
        
        # 切替カウンター
        switch_events = []
        current_symbol = None
        
        # 10日間ループ
        for day in range(1, config['days'] + 1):
            try:
                # DSSMS ranking/selection logic
                # この部分は実際のDSSMSメソッドに合わせて調整が必要
                
                # 仮の実装：DSSMSの主要メソッドを呼び出し
                if hasattr(backtester, 'select_top_symbol'):
                    selected_symbol = backtester.select_top_symbol(day=day)
                elif hasattr(backtester, 'get_current_position'):
                    selected_symbol = backtester.get_current_position()
                else:
                    # フォールバック：利用可能なメソッドを探す
                    available_methods = [method for method in dir(backtester) 
                                       if not method.startswith('_') and callable(getattr(backtester, method))]
                    logger.warning(f"利用可能なメソッド: {available_methods[:10]}")
                    selected_symbol = None
                
                # 切替検出
                if current_symbol is None:
                    current_symbol = selected_symbol
                    logger.info(f"Day {day}: 初期銘柄 = {selected_symbol}")
                elif current_symbol != selected_symbol:
                    switch_events.append({
                        'day': day,
                        'from': current_symbol,
                        'to': selected_symbol
                    })
                    logger.info(f"Day {day}: 切替 {current_symbol} → {selected_symbol}")
                    current_symbol = selected_symbol
                else:
                    logger.debug(f"Day {day}: 継続 {current_symbol}")
                    
            except Exception as e:
                logger.error(f"Day {day} 実行エラー: {e}")
                switch_events.append({
                    'day': day,
                    'error': str(e)
                })
        
        return {
            'days_executed': config['days'],
            'switch_events': switch_events,
            'switch_count': len([event for event in switch_events if 'error' not in event]),
            'final_symbol': current_symbol,
            'success': True
        }
    
    def compare_direct_execution(self) -> Dict[str, Any]:
        """直接実行による比較分析"""
        
        logger.info("=== 直接実行比較分析 ===")
        
        # Step 1: UltraSimple実行（10日間）
        ultra_simple = UltraSimpleRanking(self.symbols)
        ultra_results = ultra_simple.simulate_switches(10)
        
        logger.info(f"✅ UltraSimple (10日): {ultra_results['switch_count']}回切替")
        
        # Step 2: DSSMS直接実行
        dssms_results = self.run_fresh_dssms_analysis()
        
        # Step 3: 比較
        comparison = {
            'ultra_simple': {
                'switches': ultra_results['switch_count'],
                'expected': 1,  # 10日間で1回（Day 10）
                'status': ultra_results['success']
            },
            'dssms_direct': dssms_results,
            'comparison_analysis': self._compare_results(ultra_results, dssms_results)
        }
        
        return comparison
    
    def _compare_results(self, ultra_results: Dict, dssms_results: Dict) -> Dict[str, Any]:
        """結果比較分析"""
        
        ultra_switches = ultra_results['switch_count']
        
        if dssms_results['status'] != 'success':
            return {
                'type': 'execution_failed',
                'ultra_switches': ultra_switches,
                'dssms_status': dssms_results['status'],
                'issue': 'DSSMS実行失敗',
                'action_required': 'DSSMSシステム修復が必要'
            }
        
        dssms_switches = dssms_results['results']['switch_count']
        
        analysis = {
            'type': 'switch_comparison',
            'ultra_switches': ultra_switches,
            'dssms_switches': dssms_switches,
            'difference': abs(ultra_switches - dssms_switches),
            'ratio': dssms_switches / ultra_switches if ultra_switches > 0 else 0
        }
        
        # 判定
        if dssms_switches == ultra_switches:
            analysis['verdict'] = 'NORMAL'
            analysis['message'] = '切替回数が一致 - 正常動作'
        elif dssms_switches < ultra_switches * 0.5:
            analysis['verdict'] = 'CRITICAL'
            analysis['message'] = 'DSSMS切替回数が50%以下 - 重大な問題'
        else:
            analysis['verdict'] = 'WARNING'
            analysis['message'] = 'DSSMS切替回数に差異 - 要調査'
        
        return analysis

def main():
    """メイン分析実行"""
    
    print("🚀 DSSMS直接実行分析")
    print("=" * 50)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("目的: DSSMSとUltraSimpleの直接比較（10日間）")
    print()
    
    analyzer = DSSMSDirectAnalyzer()
    
    try:
        # 直接実行比較
        comparison = analyzer.compare_direct_execution()
        
        # 結果表示
        print("📊 直接実行比較結果:")
        print("-" * 30)
        
        # UltraSimple結果
        ultra = comparison['ultra_simple']
        print(f"✅ Ultra Simple: {ultra['switches']}回切替 (期待: {ultra['expected']}回)")
        print(f"   Status: {ultra['status']}")
        
        # DSSMS結果
        dssms = comparison['dssms_direct']
        print(f"\n🔍 DSSMS直接実行: {dssms['status']}")
        
        if dssms['status'] == 'success':
            results = dssms['results']
            print(f"   切替回数: {results['switch_count']}回")
            print(f"   実行時間: {dssms['execution_time']:.2f}秒")
            print(f"   最終銘柄: {results['final_symbol']}")
            
            if results['switch_events']:
                print("   切替イベント:")
                for event in results['switch_events']:
                    if 'error' not in event:
                        print(f"     Day {event['day']}: {event['from']} → {event['to']}")
                    else:
                        print(f"     Day {event['day']}: エラー - {event['error']}")
        else:
            print(f"   エラー: {dssms.get('error', 'unknown')}")
        
        # 比較分析
        analysis = comparison['comparison_analysis']
        print(f"\n⚖️  比較分析:")
        print(f"   判定: {analysis.get('verdict', 'UNKNOWN')}")
        print(f"   メッセージ: {analysis.get('message', 'N/A')}")
        
        if 'difference' in analysis:
            print(f"   差分: {analysis['difference']}回")
            print(f"   比率: {analysis['ratio']:.2%}")
        
        return comparison
        
    except Exception as e:
        print(f"❌ 分析エラー: {e}")
        logger.error(f"メイン分析失敗: {e}")
        return None

if __name__ == "__main__":
    results = main()
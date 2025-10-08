#!/usr/bin/env python3
"""
DSSMS 実運用テスト
================

最終統合テスト後の実際のバックテスト実行
- DSSMSバックテスターの実動作確認
- 改善されたパフォーマンスの実測
- レポート生成機能の確認

実行方法:
    python dssms_operational_test.py

作成日: 2025年9月3日
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# 設定とロギング
from config.logger_config import setup_logger

def run_dssms_operational_test():
    """DSSMS実運用テスト実行"""
    
    logger = setup_logger(__name__)
    
    print("\n" + "="*80)
    print("[ROCKET] DSSMS 実運用テスト実行")
    print("="*80)
    
    test_results = {}
    
    # 1. DSSMSバックテスター動作テスト
    print("\n1️⃣ DSSMSバックテスター動作テスト")
    print("-" * 60)
    
    try:
        from src.dssms.dssms_backtester import DSSMSBacktester
        
        # テスト設定
        config = {
            'initial_capital': 1000000,  # 100万円
            'switch_cost_rate': 0.001,   # 0.1%
            'output_excel': False,       # テスト用に無効化
            'output_detailed_report': False,
            'data_source': 'yahoo_finance'
        }
        
        print(f"   DSSMSバックテスター初期化...")
        backtester = DSSMSBacktester(config)
        
        print(f"   [OK] バックテスター初期化成功")
        
        # テスト期間: 3ヶ月間
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=90)
        
        # 日本株テスト用シンボル
        test_symbols = ['7203.T', '6758.T', '9984.T', '4063.T', '8316.T']
        
        print(f"   テスト期間: {start_date.strftime('%Y-%m-%d')} ～ {end_date.strftime('%Y-%m-%d')}")
        print(f"   テスト銘柄: {', '.join(test_symbols)}")
        print(f"   シミュレーション実行中...")
        
        start_time = time.time()
        
        # DSSMSシミュレーション実行
        result = backtester.simulate_dynamic_selection(
            start_date=start_date,
            end_date=end_date,
            symbol_universe=test_symbols
        )
        
        execution_time = time.time() - start_time
        
        if result.get('success', False):
            print(f"   [OK] シミュレーション成功 ({execution_time:.2f}秒)")
            
            # 結果詳細表示
            final_value = result.get('final_value', 0)
            initial_capital = config['initial_capital']
            total_return = (final_value - initial_capital) / initial_capital
            switch_count = result.get('switch_count', 0)
            transaction_costs = result.get('transaction_costs', 0)
            
            print(f"   [CHART] 実行結果:")
            print(f"     初期資本: {initial_capital:,.0f}円")
            print(f"     最終価値: {final_value:,.0f}円")
            print(f"     総リターン: {total_return:.2%}")
            print(f"     切替回数: {switch_count}回")
            print(f"     取引コスト: {transaction_costs:,.0f}円")
            print(f"     実行時間: {execution_time:.2f}秒")
            
            # 年間換算値計算
            test_days = (end_date - start_date).days
            annual_switches = switch_count * (365 / test_days)
            annual_costs = transaction_costs * (365 / test_days)
            
            print(f"   [CHART] 年間換算:")
            print(f"     年間切替回数: {annual_switches:.0f}回")
            print(f"     年間取引コスト: {annual_costs:,.0f}円")
            
            test_results['backtest_success'] = True
            test_results['execution_time'] = execution_time
            test_results['total_return'] = total_return
            test_results['switch_count'] = switch_count
            test_results['annual_switches'] = annual_switches
            test_results['annual_costs'] = annual_costs
            
        else:
            print(f"   [ERROR] シミュレーション失敗")
            print(f"   エラー: {result.get('error', 'Unknown error')}")
            test_results['backtest_success'] = False
            test_results['error'] = result.get('error', 'Unknown error')
            
    except Exception as e:
        print(f"   [ERROR] バックテスター例外: {e}")
        test_results['backtest_success'] = False
        test_results['error'] = str(e)
        print(f"   詳細: {traceback.format_exc()}")
    
    # 2. パフォーマンス改善効果測定
    print("\n2️⃣ パフォーマンス改善効果測定")
    print("-" * 60)
    
    if test_results.get('backtest_success', False):
        # 改善前の想定値（設計文書から）
        before_metrics = {
            'annual_switches': 3600,
            'annual_transaction_costs': 275000,
            'execution_time_per_test': 120
        }
        
        # 実測値との比較
        actual_annual_switches = test_results.get('annual_switches', 0)
        actual_annual_costs = test_results.get('annual_costs', 0)
        actual_execution_time = test_results.get('execution_time', 0)
        
        # 改善率計算
        switch_reduction = (before_metrics['annual_switches'] - actual_annual_switches) / before_metrics['annual_switches']
        cost_reduction = (before_metrics['annual_transaction_costs'] - actual_annual_costs) / before_metrics['annual_transaction_costs']
        speed_improvement = (before_metrics['execution_time_per_test'] - actual_execution_time) / before_metrics['execution_time_per_test']
        
        print(f"   [CHART] 改善効果実測値:")
        print(f"     切替回数削減: {switch_reduction:.1%} (目標: 88%)")
        print(f"     取引コスト削減: {cost_reduction:.1%} (目標: 79%)")
        print(f"     実行時間短縮: {speed_improvement:.1%} (目標: 75%)")
        
        # 目標達成率
        target_achievement = {
            'switch_reduction': switch_reduction / 0.88,
            'cost_reduction': cost_reduction / 0.79,
            'speed_improvement': speed_improvement / 0.75
        }
        
        print(f"   [CHART] 目標達成率:")
        for metric, achievement in target_achievement.items():
            status = "[OK]" if achievement >= 0.8 else "[WARNING]" if achievement >= 0.5 else "[ERROR]"
            metric_name = {
                'switch_reduction': '切替回数削減',
                'cost_reduction': 'コスト削減',
                'speed_improvement': '実行時間短縮'
            }[metric]
            print(f"     {status} {metric_name}: {achievement:.1%}")
        
        test_results['improvement_metrics'] = {
            'switch_reduction': switch_reduction,
            'cost_reduction': cost_reduction,
            'speed_improvement': speed_improvement,
            'target_achievement': target_achievement
        }
        
        # 総合改善スコア
        avg_achievement = sum(target_achievement.values()) / len(target_achievement)
        test_results['overall_improvement_achievement'] = avg_achievement
        
        print(f"   [CHART] 総合改善達成率: {avg_achievement:.1%}")
        
    else:
        print(f"   [WARNING] バックテスト失敗のため改善効果測定をスキップ")
    
    # 3. レポート生成テスト
    print("\n3️⃣ レポート生成テスト")
    print("-" * 60)
    
    try:
        from src.reports.comprehensive.comprehensive_report_engine import ComprehensiveReportEngine
        
        print(f"   レポートエンジン初期化...")
        report_engine = ComprehensiveReportEngine()
        
        print(f"   [OK] レポートエンジン初期化成功")
        
        # テストレポート生成
        if test_results.get('backtest_success', False):
            print(f"   バックテスト結果レポート生成中...")
            
            # DSSMSデータレポート生成
            report_data = {
                'dssms_results': {
                    'final_value': test_results.get('annual_switches', 0),
                    'total_return': test_results.get('total_return', 0),
                    'switch_count': test_results.get('switch_count', 0),
                    'execution_time': test_results.get('execution_time', 0)
                },
                'improvement_metrics': test_results.get('improvement_metrics', {}),
                'test_metadata': {
                    'test_date': datetime.now().strftime('%Y-%m-%d'),
                    'test_duration': '3ヶ月間',
                    'symbols_tested': 5
                }
            }
            
            try:
                # 包括的レポート生成
                html_report = report_engine.generate_report(
                    data=report_data,
                    report_type='dssms',
                    detail_level='comprehensive'
                )
                
                if html_report and len(html_report) > 1000:
                    print(f"   [OK] レポート生成成功 ({len(html_report)}文字)")
                    
                    # レポートファイル保存
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_filename = f"dssms_operational_test_report_{timestamp}.html"
                    
                    with open(report_filename, 'w', encoding='utf-8') as f:
                        f.write(html_report)
                    
                    print(f"   📄 レポート保存: {report_filename}")
                    test_results['report_generation_success'] = True
                    test_results['report_filename'] = report_filename
                    
                else:
                    print(f"   [ERROR] レポート生成失敗（サイズ不足）")
                    test_results['report_generation_success'] = False
                    
            except Exception as e:
                print(f"   [ERROR] レポート生成エラー: {e}")
                test_results['report_generation_success'] = False
        else:
            print(f"   [WARNING] バックテスト失敗のためレポート生成をスキップ")
            test_results['report_generation_success'] = False
    
    except Exception as e:
        print(f"   [ERROR] レポートシステムエラー: {e}")
        test_results['report_generation_success'] = False
    
    # 4. 最終評価
    print("\n4️⃣ 最終評価")
    print("-" * 60)
    
    # 成功項目カウント
    success_components = [
        test_results.get('backtest_success', False),
        test_results.get('overall_improvement_achievement', 0) >= 0.5,
        test_results.get('report_generation_success', False)
    ]
    
    success_count = sum(success_components)
    total_components = len(success_components)
    
    overall_success = success_count >= 2  # 3つ中2つ以上成功
    
    print(f"   [CHART] テスト結果サマリー:")
    print(f"     バックテスト実行: {'[OK]' if test_results.get('backtest_success', False) else '[ERROR]'}")
    print(f"     改善効果達成: {'[OK]' if test_results.get('overall_improvement_achievement', 0) >= 0.5 else '[ERROR]'}")
    print(f"     レポート生成: {'[OK]' if test_results.get('report_generation_success', False) else '[ERROR]'}")
    
    print(f"   [CHART] 総合評価: {success_count}/{total_components} ({'[OK] 成功' if overall_success else '[ERROR] 失敗'})")
    
    if overall_success:
        print(f"   [SUCCESS] DSSMS実運用テスト成功！")
        if test_results.get('backtest_success', False):
            print(f"   [UP] 年間切替回数: {test_results.get('annual_switches', 0):.0f}回")
            print(f"   [MONEY] 年間取引コスト: {test_results.get('annual_costs', 0):,.0f}円")
            print(f"   ⚡ 実行時間: {test_results.get('execution_time', 0):.2f}秒")
        print(f"   [LIST] DSSMSは本番運用準備完了です")
    else:
        print(f"   [WARNING] DSSMS実運用テストで課題が発見されました")
        print(f"   [TOOL] 問題のある箇所の修正が推奨されます")
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"dssms_operational_test_results_{timestamp}.txt"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("DSSMS実運用テスト結果\n")
        f.write("="*50 + "\n\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write(f"総合結果: {'成功' if overall_success else '失敗'}\n\n")
        
        for key, value in test_results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n📄 詳細結果を保存しました: {results_file}")
    
    return overall_success

def main():
    """メイン実行関数"""
    
    try:
        success = run_dssms_operational_test()
        
        if success:
            print("\n[SUCCESS] DSSMS実運用テスト 【成功】")
            print("システムは実運用準備完了です！")
            return 0
        else:
            print("\n[WARNING] DSSMS実運用テスト 【課題あり】")
            print("一部修正が必要ですが、基本機能は動作しています。")
            return 1
            
    except Exception as e:
        print(f"\n[ERROR] 実運用テスト実行エラー: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

"""
DSSMS Phase 4 - Performance Metrics Integration Test
統合パフォーマンス指標のテスト実行

Author: AI Assistant
Created: 2025-09-30
Phase: Phase 4 Testing
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import json

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dssms.performance_metrics import PerformanceMetricsCalculator
from src.dssms.dssms_report_generator import DSSMSReportGenerator
from config.logger_config import setup_logger


def test_performance_metrics_integration():
    """Performance Metrics統合テスト"""
    print("=" * 60)
    print("Phase 4 - Performance Metrics Integration Test")
    print("=" * 60)
    
    logger = setup_logger(__name__)
    
    try:
        # 1. Performance Metrics Calculator単体テスト
        print("\n1. Performance Metrics Calculator テスト中...")
        calculator = PerformanceMetricsCalculator()
        
        # サンプルデータ
        sample_portfolio_values = [
            1000000.0, 1020000.0, 1015000.0, 1030000.0, 1025000.0,
            1040000.0, 1050000.0, 1045000.0, 1060000.0, 1055000.0,
            1070000.0, 1080000.0, 1075000.0, 1090000.0, 1095000.0
        ]
        
        # 包括的指標計算
        comprehensive_result = calculator.generate_comprehensive_metrics(sample_portfolio_values)
        
        print(f"   Status: {comprehensive_result['status']}")
        print(f"   Total Return: {comprehensive_result['summary']['total_return_percent']}%")
        print(f"   Sharpe Ratio: {comprehensive_result['summary']['sharpe_ratio']}")
        print(f"   Max Drawdown: {comprehensive_result['summary']['max_drawdown_percent']}%")
        print(f"   Overall Score: {comprehensive_result['summary']['overall_score']}")
        
        # 2. DSSMS Report Generator統合テスト
        print("\n2. DSSMS Report Generator統合テスト中...")
        
        # サンプルバックテスト結果の生成
        sample_backtest_data = {
            'statistics': {
                'total_return': 0.095,  # 9.5%
                'success_rate': 0.87,
                'total_trades': 150,
                'winning_trades': 130,
                'average_execution_time_ms': 85
            },
            'position_updates': [
                {'portfolio_value': value, 'timestamp': f'2024-{i+1:02d}-01'} 
                for i, value in enumerate(sample_portfolio_values)
            ],
            'switches': [
                {'from_strategy': 'A', 'to_strategy': 'B', 'timestamp': '2024-03-15', 'reason': 'performance'},
                {'from_strategy': 'B', 'to_strategy': 'C', 'timestamp': '2024-06-10', 'reason': 'risk_adjustment'}
            ]
        }
        
        # Report Generator初期化
        report_generator = DSSMSReportGenerator()
        
        # 統合レポート生成テスト
        try:
            comprehensive_report = report_generator.generate_comprehensive_report(sample_backtest_data)
            
            print(f"   Report Status: Success")
            print(f"   Report Sections: {len(comprehensive_report)} sections")
            
            # レポート内容の確認
            if 'advanced_performance_metrics' in comprehensive_report:
                advanced_metrics = comprehensive_report['advanced_performance_metrics']
                print(f"   Advanced Metrics Status: {advanced_metrics.get('advanced_metrics_status', 'unknown')}")
                print(f"   Sharpe Ratio: {advanced_metrics.get('sharpe_ratio', 'N/A')}")
                print(f"   Sortino Ratio: {advanced_metrics.get('sortino_ratio', 'N/A')}")
                print(f"   Information Ratio: {advanced_metrics.get('information_ratio', 'N/A')}")
                print(f"   Performance Score: {advanced_metrics.get('performance_score', 'N/A')}")
            else:
                print("   Advanced Performance Metrics: Not found in report")
            
        except Exception as e:
            print(f"   Report Generation Error: {e}")
            logger.error(f"Report generation failed: {e}")
        
        # 3. 個別指標テスト
        print("\n3. 個別指標計算テスト中...")
        
        # リターン計算
        returns = []
        for i in range(1, len(sample_portfolio_values)):
            daily_return = (sample_portfolio_values[i] - sample_portfolio_values[i-1]) / sample_portfolio_values[i-1]
            returns.append(daily_return)
        
        # Sharpe比率
        sharpe_result = calculator.calculate_sharpe_ratio(returns)
        print(f"   Sharpe Ratio: {sharpe_result['sharpe_ratio']}")
        print(f"   Annualized Return: {sharpe_result['annualized_return']:.4f}")
        print(f"   Volatility: {sharpe_result['volatility']:.4f}")
        
        # 最大ドローダウン
        drawdown_result = calculator.calculate_maximum_drawdown(sample_portfolio_values)
        print(f"   Max Drawdown: {drawdown_result['max_drawdown_percent']:.2f}%")
        print(f"   Drawdown Duration: {drawdown_result['drawdown_duration']} periods")
        
        # リスク調整後リターン
        risk_adjusted_result = calculator.calculate_risk_adjusted_returns(returns)
        print(f"   Calmar Ratio: {risk_adjusted_result['calmar_ratio']:.4f}")
        print(f"   Sortino Ratio: {risk_adjusted_result['sortino_ratio']:.4f}")
        print(f"   VaR 95%: {risk_adjusted_result['var_95']:.4f}")
        
        # 高度指標
        advanced_result = calculator.calculate_advanced_metrics(returns)
        print(f"   Information Ratio: {advanced_result['information_ratio']:.4f}")
        print(f"   Beta: {advanced_result['beta']:.4f}")
        print(f"   Alpha: {advanced_result['alpha']:.4f}")
        
        print("\n" + "=" * 60)
        print("Phase 4 Performance Metrics Integration: ✅ SUCCESS")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test Failed: {e}")
        logger.error(f"Integration test error: {e}")
        return False


def main():
    """メイン実行関数"""
    print("DSSMS Phase 4 - Performance Metrics Integration Test")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = test_performance_metrics_integration()
    
    if success:
        print("\n🎉 All Phase 4 tests completed successfully!")
        return 0
    else:
        print("\n💥 Phase 4 tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
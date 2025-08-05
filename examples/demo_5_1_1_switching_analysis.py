"""
Demo Script: Strategy Switching Timing Analysis Tool
File: demo_5_1_1_switching_analysis.py
Description:
  5-1-1「戦略切替のタイミング分析ツール」のデモンストレーション
  
Author: imega
Created: 2025-01-21
Modified: 2025-01-21
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# プロジェクトパスの追加
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 警告を抑制
warnings.filterwarnings('ignore')

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_demo_data(start_date: str = '2023-01-01', end_date: str = '2023-12-31') -> pd.DataFrame:
    """デモ用データの作成"""
    try:
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # 複数のトレンド期間を含むリアルな価格データを生成
        np.random.seed(42)
        
        # ベース価格の設定
        base_price = 100.0
        prices = [base_price]
        
        # 複数のレジーム期間を定義
        regimes = [
            {'start': 0, 'end': 80, 'trend': 0.0005, 'vol': 0.015, 'type': 'sideways'},
            {'start': 80, 'end': 150, 'trend': 0.002, 'vol': 0.012, 'type': 'bull_trend'},
            {'start': 150, 'end': 200, 'trend': -0.0015, 'vol': 0.025, 'type': 'correction'},
            {'start': 200, 'end': 280, 'trend': 0.001, 'vol': 0.018, 'type': 'recovery'},
            {'start': 280, 'end': n, 'trend': 0.0003, 'vol': 0.020, 'type': 'volatile_sideways'}
        ]
        
        # 価格生成
        for i in range(1, n):
            current_regime = None
            for regime in regimes:
                if regime['start'] <= i < regime['end']:
                    current_regime = regime
                    break
                    
            if current_regime:
                trend = current_regime['trend']
                volatility = current_regime['vol']
            else:
                trend = 0.0005
                volatility = 0.015
                
            # ランダムウォーク + トレンド
            random_component = np.random.normal(0, volatility)
            price_change = trend + random_component
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 10))  # 最低価格保証
            
        # ボリューム生成
        base_volume = 500000
        volume = []
        for i, price in enumerate(prices):
            # ボラティリティが高いときにボリューム増加
            if i > 0:
                price_change = abs((price - prices[i-1]) / prices[i-1])
                volume_factor = 1 + price_change * 5  # ボラティリティベースのボリューム調整
            else:
                volume_factor = 1
                
            daily_volume = int(base_volume * volume_factor * (0.8 + np.random.random() * 0.4))
            volume.append(daily_volume)
            
        # DataFrame作成
        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'close': prices,
            'volume': volume
        }, index=dates)
        
        # 微調整
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        logger.info(f"Demo data created: {len(data)} records from {start_date} to {end_date}")
        return data
        
    except Exception as e:
        logger.error(f"Demo data creation failed: {e}")
        raise

def create_sample_switching_events(data: pd.DataFrame) -> list:
    """サンプル切替イベントの作成"""
    try:
        events = []
        
        # データの期間を分析して適切な切替ポイントを設定
        strategies = ['momentum', 'mean_reversion', 'vwap', 'breakout']
        
        # 戦略切替のサンプルイベント
        switch_dates = [
            ('2023-02-15', 'momentum', 'mean_reversion', 'Trend reversal detected'),
            ('2023-04-20', 'mean_reversion', 'momentum', 'Strong trend emergence'),
            ('2023-06-10', 'momentum', 'vwap', 'High volatility period'),
            ('2023-07-25', 'vwap', 'breakout', 'Breakout pattern detected'),
            ('2023-09-05', 'breakout', 'mean_reversion', 'Momentum exhaustion'),
            ('2023-10-30', 'mean_reversion', 'momentum', 'New trend formation'),
            ('2023-12-01', 'momentum', 'vwap', 'Market stabilization')
        ]
        
        for date_str, from_strategy, to_strategy, reason in switch_dates:
            try:
                switch_date = pd.to_datetime(date_str)
                if switch_date in data.index:
                    events.append({
                        'timestamp': switch_date,
                        'from_strategy': from_strategy,
                        'to_strategy': to_strategy,
                        'reason': reason,
                        'price': data.loc[switch_date, 'close'],
                        'volume': data.loc[switch_date, 'volume']
                    })
            except:
                continue
                
        logger.info(f"Sample switching events created: {len(events)} events")
        return events
        
    except Exception as e:
        logger.error(f"Sample switching events creation failed: {e}")
        return []

def demonstrate_basic_analysis():
    """基本分析のデモンストレーション"""
    print("\n" + "="*60)
    print("5-1-1「戦略切替のタイミング分析ツール」 - 基本分析デモ")
    print("="*60)
    
    try:
        # 分析モジュールのインポート（個別テスト）
        from analysis.strategy_switching.strategy_switching_analyzer import StrategySwitchingAnalyzer
        from analysis.strategy_switching.switching_timing_evaluator import SwitchingTimingEvaluator
        from analysis.strategy_switching.switching_pattern_detector import SwitchingPatternDetector
        from analysis.strategy_switching.switching_performance_calculator import SwitchingPerformanceCalculator
        
        print("✓ 分析モジュールのインポート成功")
        
        # デモデータの作成
        demo_data = create_demo_data()
        switching_events = create_sample_switching_events(demo_data)
        
        print(f"✓ デモデータ作成完了: {len(demo_data)} 日分、{len(switching_events)} 回の切替イベント")
        
        # 1. 戦略切替分析
        print("\n--- 1. 戦略切替分析 ---")
        analyzer = StrategySwitchingAnalyzer()
        
        # 分析期間の設定
        analysis_start = demo_data.index[50]
        analysis_end = demo_data.index[-50]
        
        switching_analysis = analyzer.analyze_switching_performance(
            data=demo_data,
            switching_events=switching_events,
            analysis_period=(analysis_start, analysis_end)
        )
        
        print(f"分析期間: {analysis_start.strftime('%Y-%m-%d')} - {analysis_end.strftime('%Y-%m-%d')}")
        print(f"総切替回数: {switching_analysis.total_switches}")
        print(f"成功率: {switching_analysis.success_rate:.1%}")
        print(f"平均改善度: {switching_analysis.average_improvement:.2%}")
        
        # 2. タイミング評価
        print("\n--- 2. タイミング評価 ---")
        timing_evaluator = SwitchingTimingEvaluator()
        
        if switching_events:
            sample_event = switching_events[2]  # 中間のイベントを使用
            timing_result = timing_evaluator.evaluate_switching_timing(
                data=demo_data,
                timestamp=sample_event['timestamp'],
                current_strategy=sample_event['from_strategy'],
                candidate_strategies=[sample_event['to_strategy']]
            )
            
            print(f"評価対象切替: {sample_event['from_strategy']} → {sample_event['to_strategy']}")
            print(f"タイミングスコア: {timing_result.timing_score:.2f}")
            print(f"信頼度: {timing_result.confidence_level:.2f}")
            print(f"最適化オフセット: {timing_result.optimal_timing_offset} 日")
            print(f"評価要因: {len(timing_result.evaluation_factors)} 項目")
        
        # 3. パターン検出
        print("\n--- 3. パターン検出 ---")
        pattern_detector = SwitchingPatternDetector()
        
        pattern_analysis = pattern_detector.detect_switching_patterns(demo_data)
        
        print(f"検出パターン数: {len(pattern_analysis.detected_patterns)}")
        
        if pattern_analysis.pattern_frequency:
            print("パターン種別分布:")
            for pattern_type, count in pattern_analysis.pattern_frequency.items():
                success_rate = pattern_analysis.success_rates.get(pattern_type, 0.5)
                print(f"  {pattern_type.value}: {count} 件 (成功率: {success_rate:.1%})")
        
        # 4. パフォーマンス計算
        print("\n--- 4. パフォーマンス計算 ---")
        performance_calculator = SwitchingPerformanceCalculator()
        
        if switching_events:
            sample_event = switching_events[1]
            performance_result = performance_calculator.calculate_switching_performance(
                data=demo_data,
                switch_timestamp=sample_event['timestamp'],
                from_strategy=sample_event['from_strategy'],
                to_strategy=sample_event['to_strategy']
            )
            
            print(f"計算対象切替: {sample_event['from_strategy']} → {sample_event['to_strategy']}")
            print(f"切替前リターン: {performance_result.pre_switch_metrics.total_return:.2%}")
            print(f"切替後リターン: {performance_result.post_switch_metrics.total_return:.2%}")
            print(f"切替コスト: {performance_result.switching_cost:.4f}")
            print(f"純利益: {performance_result.net_benefit:.4f}")
            print(f"成功判定: {'成功' if performance_result.success else '失敗'}")
            print(f"信頼度: {performance_result.confidence_score:.1%}")
        
        print("\n✓ 基本分析デモンストレーション完了")
        return True
        
    except Exception as e:
        print(f"✗ 基本分析デモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_dashboard_creation():
    """ダッシュボード作成のデモンストレーション"""
    print("\n" + "="*60)
    print("ダッシュボード作成デモ")
    print("="*60)
    
    try:
        from analysis.strategy_switching.switching_analysis_dashboard import SwitchingAnalysisDashboard
        
        print("✓ ダッシュボードモジュールインポート成功")
        
        # デモデータの準備
        demo_data = create_demo_data()
        switching_events = create_sample_switching_events(demo_data)
        
        # ダッシュボードの作成
        dashboard = SwitchingAnalysisDashboard()
        
        print("ダッシュボード生成中...")
        generated_files = dashboard.create_comprehensive_dashboard(
            data=demo_data,
            switching_events=switching_events,
            output_dir="demo_dashboard_5_1_1"
        )
        
        print("\n--- 生成されたファイル ---")
        for section, file_path in generated_files.items():
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"✓ {section}: {file_path} ({file_size:,} bytes)")
            else:
                print(f"✗ {section}: {file_path} (ファイル未作成)")
        
        # 統合レポートの確認
        if 'integrated' in generated_files:
            integrated_path = generated_files['integrated']
            if os.path.exists(integrated_path):
                print(f"\n🎯 統合レポートが生成されました:")
                print(f"   ファイルパス: {integrated_path}")
                print(f"   ブラウザで開いて内容を確認してください。")
            else:
                print("✗ 統合レポートの生成に失敗しました")
        
        print("\n✓ ダッシュボード作成デモンストレーション完了")
        return True
        
    except Exception as e:
        print(f"✗ ダッシュボード作成デモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_integration_system():
    """統合システムのデモンストレーション"""
    print("\n" + "="*60)
    print("統合システムデモ")
    print("="*60)
    
    try:
        from analysis.strategy_switching.switching_integration_system import SwitchingIntegrationSystem
        
        print("✓ 統合システムモジュールインポート成功")
        
        # 統合システムの初期化
        integration_system = SwitchingIntegrationSystem()
        
        # システム状況の確認
        system_status = integration_system.get_system_status()
        
        print("\n--- システム構成 ---")
        components = system_status['system_components']
        active_components = sum(components.values())
        total_components = len(components)
        
        print(f"アクティブコンポーネント: {active_components}/{total_components}")
        
        for component, status in components.items():
            status_icon = "✓" if status else "✗"
            print(f"  {status_icon} {component}")
        
        # デモデータの準備
        demo_data = create_demo_data()
        
        # 切替機会の分析
        print("\n--- 切替機会分析 ---")
        current_strategy = 'momentum'
        
        analysis_result = integration_system.analyze_switching_opportunity(
            current_data=demo_data,
            current_strategy=current_strategy,
            analysis_depth='comprehensive'
        )
        
        print(f"現在の戦略: {analysis_result['current_strategy']}")
        print(f"推奨切替: {analysis_result.get('switching_recommendation', 'hold')}")
        print(f"信頼度: {analysis_result.get('confidence', 0.5):.1%}")
        
        # タイミング分析結果
        timing_analysis = analysis_result.get('timing_analysis', {})
        if timing_analysis:
            print(f"タイミングスコア: {timing_analysis.get('timing_score', 0.5):.2f}")
            print(f"最適タイミング: {'Yes' if timing_analysis.get('optimal_timing', False) else 'No'}")
        
        # パターン分析結果
        pattern_analysis = analysis_result.get('pattern_analysis', {})
        if pattern_analysis:
            recommendations = pattern_analysis.get('recommendations', [])
            print(f"検出パターン数: {len(recommendations)}")
            
            if recommendations:
                top_recommendation = recommendations[0]
                print(f"トップ推奨: {top_recommendation.get('recommended_action', 'N/A')}")
        
        # 切替実行（ドライラン）
        print("\n--- 切替実行シミュレーション ---")
        recommended_strategy = analysis_result.get('switching_recommendation', 'mean_reversion')
        
        if recommended_strategy != 'hold':
            switch_result = integration_system.execute_strategy_switch(
                from_strategy=current_strategy,
                to_strategy=recommended_strategy,
                data=demo_data,
                dry_run=True
            )
            
            print(f"切替実行: {switch_result['from_strategy']} → {switch_result['to_strategy']}")
            print(f"実行状況: {switch_result['execution_status']}")
            print(f"ドライラン: {'Yes' if switch_result['dry_run'] else 'No'}")
        else:
            print("現在の戦略を継続推奨")
        
        # レポート生成
        print("\n--- レポート生成 ---")
        try:
            report_path = integration_system.generate_switching_report(
                data=demo_data,
                report_type='comprehensive',
                output_format='html'
            )
            
            if os.path.exists(report_path):
                file_size = os.path.getsize(report_path)
                print(f"✓ レポート生成成功: {report_path} ({file_size:,} bytes)")
            else:
                print(f"レポートパス: {report_path}")
        except Exception as e:
            print(f"レポート生成エラー: {e}")
        
        print("\n✓ 統合システムデモンストレーション完了")
        return True
        
    except Exception as e:
        print(f"✗ 統合システムデモエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインデモンストレーション実行"""
    print("5-1-1「戦略切替のタイミング分析ツール」デモンストレーション開始")
    print(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H時%M分%S秒')}")
    
    results = []
    
    # 1. 基本分析デモ
    results.append(("基本分析", demonstrate_basic_analysis()))
    
    # 2. ダッシュボード作成デモ
    results.append(("ダッシュボード作成", demonstrate_dashboard_creation()))
    
    # 3. 統合システムデモ  
    results.append(("統合システム", demonstrate_integration_system()))
    
    # 結果サマリー
    print("\n" + "="*60)
    print("デモンストレーション結果サマリー")
    print("="*60)
    
    success_count = 0
    total_count = len(results)
    
    for demo_name, success in results:
        status = "成功" if success else "失敗"
        icon = "✓" if success else "✗"
        print(f"{icon} {demo_name}: {status}")
        
        if success:
            success_count += 1
    
    print(f"\n総合結果: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("🎉 全てのデモンストレーションが成功しました！")
        print("\n次のステップ:")
        print("1. demo_dashboard_5_1_1/integrated_report.html をブラウザで開く")
        print("2. 生成されたレポートファイルを確認")
        print("3. 実際のデータでのテストを実行")
    else:
        print("⚠️  一部のデモンストレーションが失敗しました。")
        print("ログを確認して問題を解決してください。")
    
    return success_count == total_count

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        
        print(f"\nデモンストレーション終了 (終了コード: {exit_code})")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\nデモンストレーションがユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\nデモンストレーション実行エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

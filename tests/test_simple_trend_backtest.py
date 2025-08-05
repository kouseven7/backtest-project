#!/usr/bin/env python3
"""
簡易版トレンド限定バックテスト実行スクリプト
型エラーを回避しながら実際にバックテストを実行
"""
import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import traceback
sys.path.append(r'C:\Users\imega\Documents\my_backtest_project')

def run_simple_trend_backtest():
    """簡易版でトレンド限定バックテストを実行"""
    print("=== 簡易版トレンド限定バックテスト実行 ===")
    
    try:
        # データ取得
        from data_fetcher import get_parameters_and_data
        from analysis.trend_limited_backtest import TrendLimitedBacktester
        from strategies.VWAP_Bounce import VWAPBounceStrategy
        
        # データ準備
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        test_data = stock_data.iloc[-100:].copy()  # 100日間のテストデータ
        print(f"✓ データ準備完了: {len(test_data)}日間")
        
        # バックテスターの作成
        backtester = TrendLimitedBacktester(test_data)
        print(f"✓ バックテスター作成完了: {len(backtester.labeled_data)}日間のラベリングデータ")
        
        # トレンド期間の抽出
        uptrend_periods = backtester.extract_trend_periods("uptrend", min_period_length=5, min_confidence=0.7)
        downtrend_periods = backtester.extract_trend_periods("downtrend", min_period_length=5, min_confidence=0.7)
        range_periods = backtester.extract_trend_periods("range-bound", min_period_length=5, min_confidence=0.7)
        
        print(f"✓ トレンド期間抽出完了:")
        print(f"  - 上昇トレンド: {len(uptrend_periods)}期間")
        print(f"  - 下降トレンド: {len(downtrend_periods)}期間")
        print(f"  - レンジ相場: {len(range_periods)}期間")
        
        # 戦略パラメータ（緩和設定）
        strategy_params = {
            'vwap_lower_threshold': 0.99,
            'vwap_upper_threshold': 1.01,
            'volume_increase_threshold': 1.1,
            'bullish_candle_min_pct': 0.001,
            'trend_filter_enabled': False,
            'stop_loss': 0.02,
            'take_profit': 0.04
        }
        
        # 各トレンド環境での簡易バックテスト実行
        results = {}
        
        for trend_type, periods in [("uptrend", uptrend_periods), ("downtrend", downtrend_periods), ("range-bound", range_periods)]:
            if not periods:
                print(f"  {trend_type}: 期間なし、スキップ")
                continue
                
            print(f"\n--- {trend_type} バックテスト実行 ---")
            trend_results = []
            
            for i, (start_date, end_date, period_data) in enumerate(periods):
                try:
                    print(f"  期間 {i+1}: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')} ({len(period_data)}日間)")
                    
                    # 戦略インスタンスを作成して実行
                    strategy = VWAPBounceStrategy(period_data, params=strategy_params)
                    backtest_result = strategy.backtest()
                    
                    # シグナル数を集計
                    entry_signals = int((backtest_result['Entry_Signal'] == 1).sum()) if 'Entry_Signal' in backtest_result.columns else 0
                    exit_signals = int((backtest_result['Exit_Signal'] == -1).sum()) if 'Exit_Signal' in backtest_result.columns else 0
                    
                    # 簡易収益計算（シンプルバージョン）
                    if entry_signals > 0 and len(period_data) > 0:
                        # 期間の最初と最後の価格で簡易計算
                        start_price = float(period_data.iloc[0]['Adj Close'])
                        end_price = float(period_data.iloc[-1]['Adj Close'])
                        period_return = (end_price - start_price) / start_price
                    else:
                        period_return = 0.0
                    
                    period_result = {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "period_length": len(period_data),
                        "entry_signals": entry_signals,
                        "exit_signals": exit_signals,
                        "period_return": period_return
                    }
                    
                    trend_results.append(period_result)
                    print(f"    エントリー: {entry_signals}回, イグジット: {exit_signals}回, 期間収益: {period_return:.3f}")
                    
                except Exception as e:
                    print(f"    エラー: {str(e)}")
                    trend_results.append({
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "error": str(e)
                    })
            
            # トレンド別サマリー
            successful_periods = [r for r in trend_results if "error" not in r]
            total_entries = sum(r.get("entry_signals", 0) for r in successful_periods)
            total_exits = sum(r.get("exit_signals", 0) for r in successful_periods)
            avg_return = np.mean([r.get("period_return", 0) for r in successful_periods]) if successful_periods else 0.0
            
            results[trend_type] = {
                "trend_type": trend_type,
                "periods_tested": len(successful_periods),
                "total_periods": len(periods),
                "total_entry_signals": total_entries,
                "total_exit_signals": total_exits,
                "average_period_return": float(avg_return),
                "success_rate": len(successful_periods) / len(periods) if periods else 0,
                "period_details": trend_results
            }
            
            print(f"  サマリー: {len(successful_periods)}/{len(periods)} 期間成功, 合計エントリー: {total_entries}回, 平均収益: {avg_return:.3f}")
        
        # 全体結果の保存
        final_result = {
            "strategy": "VWAPBounceStrategy",
            "test_timestamp": datetime.now().isoformat(),
            "data_period": f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}",
            "strategy_params": strategy_params,
            "trend_comparison": results
        }
        
        # JSONファイルに保存
        output_dir = "logs"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"simple_trend_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 結果をJSONファイルに保存: {filepath}")
        
        # 結果サマリーの表示
        print("\n=== 結果サマリー ===")
        for trend_type, result in results.items():
            print(f"{trend_type}:")
            print(f"  - 期間数: {result['periods_tested']}/{result['total_periods']}")
            print(f"  - エントリー: {result['total_entry_signals']}回")
            print(f"  - 平均収益: {result['average_period_return']:.3f}")
            print(f"  - 成功率: {result['success_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"✗ エラーが発生しました: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_simple_trend_backtest()
    if success:
        print("\n✓ 簡易版トレンド限定バックテスト完了!")
    else:
        print("\n✗ バックテスト実行に失敗しました")

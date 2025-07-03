#!/usr/bin/env python3
"""
VWAP特化型トレンド判定とゴールデンクロス特化型トレンド判定の強化モジュール
既存のトレンド判定機能を強化するために作成されました。
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 設定
logger = logging.getLogger(__name__)

def run_enhanced_trend_analysis():
    """強化されたトレンド判定のワークフロー実行"""
    print("=" * 80)
    print("■ 強化されたトレンドシステムのテスト実行")
    print("=" * 80)
    
    try:
        # データ取得
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        print(f"■ データ取得成功:")
        print(f"   銘柄: {ticker}")
        print(f"   期間: {start_date} から {end_date}")
        print(f"   データ数: {len(stock_data)} 行")
        
        # 強化されたトレンド判定
        from indicators.trend_analysis import (
            detect_trend_with_confidence,
            detect_vwap_trend,
            detect_golden_cross_trend
        )
        
        # VWAPの計算
        from indicators.basic_indicators import calculate_vwap
        stock_data = stock_data.copy()
        if "VWAP" not in stock_data.columns:
            stock_data["VWAP"] = calculate_vwap(stock_data, "Adj Close", "Volume")
            print("■ VWAP計算完了")
        
        # SMAの計算
        stock_data["SMA_25"] = stock_data["Adj Close"].rolling(window=25).mean()
        stock_data["SMA_75"] = stock_data["Adj Close"].rolling(window=75).mean()
        print("■ SMA計算完了")
        
        # 各種トレンド判定の実行
        print("\n■ トレンド判定結果:")
        
        # 通常のトレンド判定
        base_trend, base_conf = detect_trend_with_confidence(stock_data)
        print(f"   通常トレンド: {base_trend} (信頼度: {base_conf:.1%})")
        
        # VWAP特化型トレンド判定
        vwap_trend, vwap_conf = detect_vwap_trend(stock_data)
        print(f"   VWAP特化: {vwap_trend} (信頼度: {vwap_conf:.1%})")
        
        # ゴールデンクロス特化型トレンド判定
        gc_trend, gc_conf = detect_golden_cross_trend(stock_data)
        print(f"   GC特化: {gc_trend} (信頼度: {gc_conf:.1%})")
        
        # 精度検証
        from indicators.trend_accuracy_validator import TrendAccuracyValidator
        
        validator = TrendAccuracyValidator(stock_data, "Adj Close")
        
        # トレンド検証用関数
        def base_trend_detector(data):
            trend, _ = detect_trend_with_confidence(data)
            return trend
            
        def vwap_trend_detector(data):
            data = data.copy()
            if "VWAP" not in data.columns:
                data["VWAP"] = calculate_vwap(data, "Adj Close", "Volume")
            trend, _ = detect_vwap_trend(data)
            return trend
            
        def gc_trend_detector(data):
            data = data.copy()
            if "SMA_25" not in data.columns:
                data["SMA_25"] = data["Adj Close"].rolling(window=25).mean()
            if "SMA_75" not in data.columns:
                data["SMA_75"] = data["Adj Close"].rolling(window=75).mean()
            trend, _ = detect_golden_cross_trend(data)
            return trend
        
        # トレンド判定器の辞書
        trend_detectors = {
            "標準トレンド判定": base_trend_detector,
            "VWAP特化型": vwap_trend_detector,
            "GC特化型": gc_trend_detector
        }
        
        # 精度検証の実行
        print("\n■ トレンド判定精度検証中...")
        
        results_df = validator.run_comprehensive_validation(
            trend_detectors, 
            validation_windows=[5, 10, 15, 20]
        )
        
        if results_df.empty:
            print("×× 検証結果が取得できませんでした")
        else:
            print("\n■ 検証結果:")
            print(results_df.round(3).to_string(index=False))
            
            # ベスト結果の表示
            best_result = results_df.loc[results_df['overall_accuracy'].idxmax()]
            best_accuracy = best_result['overall_accuracy']
            
            print(f"\n■ 最高精度結果:")
            print(f"   手法: {best_result['detector']}")
            print(f"   検証期間: {best_result['validation_window']} 日")
            print(f"   全体精度: {best_accuracy:.1%}")
            
            # 戦略別推奨事項
            print(f"\n■ 戦略別推奨事項:")
            
            # VWAP戦略
            vwap_results = results_df[results_df['detector'] == 'VWAP特化型']
            if not vwap_results.empty:
                vwap_best = vwap_results.loc[vwap_results['overall_accuracy'].idxmax()]
                print(f"   VWAP戦略: {vwap_best['overall_accuracy']:.1%} 精度")
                if vwap_best['overall_accuracy'] > best_accuracy * 0.95:
                    print("      → VWAP戦略は高精度です！")
                else:
                    print("      → 標準より精度が落ちます")
            
            # GC戦略
            gc_results = results_df[results_df['detector'] == 'GC特化型']
            if not gc_results.empty:
                gc_best = gc_results.loc[gc_results['overall_accuracy'].idxmax()]
                print(f"   GC戦略: {gc_best['overall_accuracy']:.1%} 精度")
                if gc_best['overall_accuracy'] > best_accuracy * 0.95:
                    print("      → GC戦略は高精度です！")
                else:
                    print("      → 標準より精度が落ちます")
            
        print("\n■ テスト完了")
            
    except ImportError as e:
        print(f"×× インポートエラー: {e}")
        return None
    except Exception as e:
        print(f"×× 予期しないエラー: {e}")
        logger.exception("テスト実行中にエラーが発生しました")
        return None

if __name__ == "__main__":
    # ワークフローの実行
    run_enhanced_trend_analysis()

#!/usr/bin/env python3
"""
トレンド判定精度のテストスクリプト
既存の戦略に影響を与えることなく、トレンド判定の精度を検証します。
"""

import sys
import os
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_trend_accuracy():
    """トレンド判定精度のテスト"""
    print("=" * 60)
    print("■ トレンド判定精度テスト開始")
    print("=" * 60)
    
    try:
        # データ取得
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        print(f"■ データ取得完了:")
        print(f"   銘柄: {ticker}")
        print(f"   期間: {start_date} to {end_date}")
        print(f"   データ数: {len(stock_data)} 行")
        
        # トレンド判定モジュールのインポート
        from indicators.trend_accuracy_validator import TrendAccuracyValidator
        from indicators.trend_analysis import (
            detect_trend, 
            detect_trend_with_confidence, 
            detect_vwap_trend,
            detect_golden_cross_trend,
            simple_sma_trend_detector
        )
        
        # 検証器の初期化
        validator = TrendAccuracyValidator(stock_data, "Adj Close")
        
        # テスト対象のトレンド判定器
        def enhanced_trend_detector(data):
            """強化されたトレンド判定器"""
            trend, confidence = detect_trend_with_confidence(data)
            return trend
        
        def simple_trend_detector(data):
            """シンプルなトレンド判定器"""
            return detect_trend(data)
        
        def vwap_trend_detector(data):
            """VWAP専用トレンド判定器"""
            # VWAPがない場合は基本指標で計算
            if "VWAP" not in data.columns:
                from indicators.basic_indicators import calculate_vwap
                data = data.copy()
                try:
                    data["VWAP"] = calculate_vwap(data, "Adj Close", "Volume")
                except:
                    # VWAPが計算できない場合は通常のトレンド判定
                    return detect_trend(data)
            
            trend, confidence = detect_vwap_trend(data)
            return trend
        
        def golden_cross_trend_detector(data):
            """ゴールデンクロス専用トレンド判定器"""
            trend, confidence = detect_golden_cross_trend(data)
            return trend
        
        trend_detectors = {
            "Enhanced": enhanced_trend_detector,
            "Simple": simple_trend_detector,
            "VWAP_Specialized": vwap_trend_detector,
            "GoldenCross_Specialized": golden_cross_trend_detector,
            "Basic_SMA": simple_sma_trend_detector
        }
        
        print(f"\n■ 検証対象: {len(trend_detectors)} 種類のトレンド判定器")
        for name in trend_detectors.keys():
            print(f"   - {name}")
        
        # 包括的検証の実行
        print(f"\n▼ 包括的検証実行中...")
        results_df = validator.run_comprehensive_validation(trend_detectors)
        
        if results_df.empty:
            print("×× 検証結果が取得できませんでした")
            return None
        
        print(f"\n■ 検証結果")
        print("=" * 80)
        print(results_df.round(3).to_string(index=False))
        
        # 9割精度のチェック
        best_result = results_df.loc[results_df['overall_accuracy'].idxmax()]
        best_accuracy = best_result['overall_accuracy']
        
        print(f"\n★ 最高精度結果:")
        print(f"   手法: {best_result['detector']}")
        print(f"   検証期間: {best_result['validation_window']} 日")
        print(f"   全体精度: {best_accuracy:.1%}")
        print(f"   上昇トレンド精度: {best_result['uptrend_accuracy']:.1%}")
        print(f"   下降トレンド精度: {best_result['downtrend_accuracy']:.1%}")
        print(f"   レンジ相場精度: {best_result['range_accuracy']:.1%}")
        print(f"   サンプル数: {best_result['total_samples']}")
        
        # 9割精度達成チェック
        target_accuracy = 0.9
        if best_accuracy >= target_accuracy:
            print(f"\n○○ 目標精度 {target_accuracy:.0%} を達成！")
            print("   現在のトレンド判定は十分な精度を持っています。")
        else:
            print(f"\n△△ 目標精度 {target_accuracy:.0%} に届きませんでした（現在: {best_accuracy:.1%}）")
            print("   改善の余地があります。")
        
        # 戦略別推奨事項
        print(f"\n■ 戦略別推奨事項:")
        
        # VWAP戦略向け
        vwap_results = results_df[results_df['detector'] == 'VWAP_Specialized']
        if not vwap_results.empty:
            vwap_best = vwap_results.loc[vwap_results['overall_accuracy'].idxmax()]
            print(f"   ▼ VWAP戦略: {vwap_best['overall_accuracy']:.1%} 精度")
            if vwap_best['overall_accuracy'] >= 0.85:
                print("      → 現在の設定で良好な精度")
            else:
                print("      → パラメータ調整が推奨")
        
        # ゴールデンクロス戦略向け
        gc_results = results_df[results_df['detector'] == 'GoldenCross_Specialized']
        if not gc_results.empty:
            gc_best = gc_results.loc[gc_results['overall_accuracy'].idxmax()]
            print(f"   ▼ ゴールデンクロス戦略: {gc_best['overall_accuracy']:.1%} 精度")
            if gc_best['overall_accuracy'] >= 0.85:
                print("      → 現在の設定で良好な精度")
            else:
                print("      → MA期間の調整が推奨")
        
        # 結果の保存
        output_file = f"trend_accuracy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n■ 結果を {output_file} に保存しました")
        
        return results_df
        
    except ImportError as e:
        print(f"×× インポートエラー: {e}")
        print("必要なモジュールが不足している可能性があります")
        return None
    except Exception as e:
        print(f"×× 予期しないエラー: {e}")
        logger.exception("テスト実行中にエラーが発生しました")
        return None

def test_trend_confidence():
    """信頼度付きトレンド判定のテスト"""
    print(f"\n" + "=" * 60)
    print("■ 信頼度付きトレンド判定テスト")
    print("=" * 60)
    
    try:
        from data_fetcher import get_parameters_and_data
        from indicators.trend_analysis import detect_trend_with_confidence
        
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        # 最新のトレンド判定
        latest_trend, confidence = detect_trend_with_confidence(stock_data)
        
        print(f"■ 最新のトレンド判定結果:")
        print(f"   銘柄: {ticker}")
        print(f"   判定日: {stock_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   トレンド: {latest_trend}")
        print(f"   信頼度: {confidence:.1%}")
        
        # 信頼度に基づく推奨アクション
        if confidence >= 0.8:
            print(f"   ★ 高信頼度 - 強いトレンドシグナル")
        elif confidence >= 0.6:
            print(f"   ○ 中信頼度 - 注意深く判断")
        elif confidence >= 0.4:
            print(f"   △ 低信頼度 - 追加確認推奨")
        else:
            print(f"   × 極低信頼度 - トレンド不明瞭")
        
        # 過去1ヶ月のトレンド推移
        print(f"\n▼ 過去1ヶ月のトレンド推移:")
        recent_data = stock_data.tail(20)  # 最新20日
        
        for i in range(-5, 0):  # 最新5日分
            if len(recent_data) + i > 0:
                test_data = stock_data.iloc[:len(stock_data) + i]
                trend, conf = detect_trend_with_confidence(test_data)
                date = test_data.index[-1].strftime('%Y-%m-%d')
                print(f"   {date}: {trend:12} (信頼度: {conf:.1%})")
        
    except Exception as e:
        print(f"×× エラー: {e}")
        logger.exception("信頼度テスト中にエラーが発生しました")

if __name__ == "__main__":
    # メインテストの実行
    results = test_trend_accuracy()
    
    # 信頼度テストの実行
    test_trend_confidence()
    
    print(f"\n" + "=" * 60)
    print("■ テスト完了")
    print("=" * 60)

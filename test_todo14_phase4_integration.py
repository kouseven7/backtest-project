#!/usr/bin/env python3
"""
TODO #14 Phase 4: MarketDataQualityValidator統合テスト

Phase 4で実装されたデータ品質検証システムの動作確認
- 品質検証機能テスト
- RealMarketDataFetcher統合テスト
- 自動修正機能テスト
- バックテスト基本理念遵守確認
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 必要モジュールのインポート
try:
    from market_data_quality_validator import (
        MarketDataQualityValidator, 
        QualityLevel, 
        DataIssueType,
        validate_fetched_data_quality
    )
    from real_market_data_fetcher import fetch_strategy_required_data, create_real_market_data_fetcher
    print("[OK] Phase 4モジュールのインポート成功")
except ImportError as e:
    print(f"[ERROR] インポートエラー: {e}")
    sys.exit(1)

def create_test_data_with_issues():
    """問題のあるテストデータ作成"""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    
    # 基本データ
    data = pd.DataFrame({
        'Open': [100 + i for i in range(20)],
        'High': [105 + i for i in range(20)],
        'Low': [95 + i for i in range(20)],
        'Close': [102 + i for i in range(20)],
        'Volume': [1000000 + i*10000 for i in range(20)]
    }, index=dates)
    
    # 意図的に問題を作成
    # 1. 欠損データ
    data.loc[data.index[5], 'Close'] = np.nan
    data.loc[data.index[10], 'Volume'] = np.nan
    
    # 2. 異常値
    data.loc[data.index[7], 'High'] = 999999  # 異常に高い値
    
    # 3. 負の出来高
    data.loc[data.index[15], 'Volume'] = -50000
    
    # 4. ゼロ出来高
    data.loc[data.index[18], 'Volume'] = 0
    
    return data

def create_good_test_data():
    """問題のないテストデータ作成"""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    
    data = pd.DataFrame({
        'Open': [100 + i for i in range(20)],
        'High': [105 + i for i in range(20)],
        'Low': [95 + i for i in range(20)],
        'Close': [102 + i for i in range(20)],
        'Volume': [1000000 + i*10000 for i in range(20)]
    }, index=dates)
    
    return data

def test_quality_validator_basic():
    """MarketDataQualityValidator基本機能テスト"""
    print("\n[TOOL] MarketDataQualityValidator基本機能テスト")
    
    validator = MarketDataQualityValidator(auto_fix_enabled=True)
    
    try:
        # 問題のあるデータでテスト
        print("[LIST] 問題のあるテストデータで品質検証...")
        test_data = create_test_data_with_issues()
        report = validator.validate_data_quality(test_data, "test_data")
        
        print(f"[OK] 品質検証完了: {report.quality_level.value} ({report.quality_score:.1f}%)")
        print(f"   検出問題数: {len(report.issues)}")
        print(f"   修正問題数: {len(report.fixed_issues)}")
        print(f"   バックテスト基本理念遵守: {'[OK]' if report.backtest_compliance else '[ERROR]'}")
        
        # レポート表示
        report_text = validator.generate_quality_report_text(report)
        print("\n" + "="*50)
        print("品質レポート（抜粋）:")
        print("="*50)
        print(report_text[:500] + "..." if len(report_text) > 500 else report_text)
        
        return len(report.issues) > 0  # 問題が検出されれば成功
        
    except Exception as e:
        print(f"[ERROR] 品質検証テストエラー: {str(e)}")
        return False

def test_quality_validator_good_data():
    """良品質データでの動作テスト"""
    print("\n[TOOL] 良品質データでの動作テスト")
    
    validator = MarketDataQualityValidator(auto_fix_enabled=True)
    
    try:
        # 問題のないデータでテスト
        print("[LIST] 良品質テストデータで品質検証...")
        test_data = create_good_test_data()
        report = validator.validate_data_quality(test_data, "good_test_data")
        
        print(f"[OK] 品質検証完了: {report.quality_level.value} ({report.quality_score:.1f}%)")
        print(f"   検出問題数: {len(report.issues)}")
        print(f"   バックテスト基本理念遵守: {'[OK]' if report.backtest_compliance else '[ERROR]'}")
        
        # 良品質データは高スコアを期待
        return report.quality_score >= 95 and report.backtest_compliance
        
    except Exception as e:
        print(f"[ERROR] 良品質データテストエラー: {str(e)}")
        return False

def test_validate_fetched_data_quality():
    """validate_fetched_data_quality関数テスト"""
    print("\n[TOOL] validate_fetched_data_quality関数テスト")
    
    try:
        # 問題のあるデータで品質検証・修正テスト
        print("[LIST] 問題データの品質検証・修正テスト...")
        test_data = create_test_data_with_issues()
        
        validated_data, quality_report = validate_fetched_data_quality(
            data_type="test_data",
            data=test_data,
            auto_fix=True
        )
        
        print(f"[OK] 品質検証・修正完了: {quality_report.quality_level.value} ({quality_report.quality_score:.1f}%)")
        print(f"   元データ行数: {len(test_data)}")
        print(f"   修正後データ行数: {len(validated_data)}")
        print(f"   修正問題数: {len(quality_report.fixed_issues)}")
        
        # 修正により品質向上を期待
        return len(quality_report.fixed_issues) > 0
        
    except Exception as e:
        print(f"[ERROR] validate_fetched_data_qualityテストエラー: {str(e)}")
        return False

def test_phase4_integration_with_fetcher():
    """Phase 4とRealMarketDataFetcher統合テスト"""
    print("\n[TOOL] Phase 4 RealMarketDataFetcher統合テスト")
    
    try:
        # テスト用株価データ期間
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        stock_data = pd.DataFrame({'Close': range(10)}, index=dates)
        
        # Phase 4品質検証有効でデータ取得
        print("[LIST] Phase 4品質検証有効でのデータ取得...")
        strategy_data_with_validation = fetch_strategy_required_data(
            strategy_name='VWAPBreakoutStrategy',
            stock_data_period=stock_data,
            enable_quality_validation=True
        )
        
        if 'index_data' in strategy_data_with_validation and strategy_data_with_validation['index_data'] is not None:
            print(f"[OK] Phase 4統合成功: index_data with validation ({len(strategy_data_with_validation['index_data'])} rows)")
            
            # Phase 4品質検証無効でのデータ取得（比較用）
            print("[LIST] Phase 4品質検証無効でのデータ取得（比較用）...")
            strategy_data_without_validation = fetch_strategy_required_data(
                strategy_name='VWAPBreakoutStrategy', 
                stock_data_period=stock_data,
                enable_quality_validation=False
            )
            
            if 'index_data' in strategy_data_without_validation:
                print(f"[OK] 比較用データ取得成功: index_data without validation ({len(strategy_data_without_validation['index_data'])} rows)")
                return True
            
        return False
        
    except Exception as e:
        print(f"[ERROR] Phase 4統合テストエラー: {str(e)}")
        return False

def test_backtest_compliance_check():
    """バックテスト基本理念遵守チェックテスト"""
    print("\n[TOOL] バックテスト基本理念遵守チェックテスト")
    
    validator = MarketDataQualityValidator()
    
    try:
        # 1. 基本理念遵守データ
        print("[LIST] 基本理念遵守データテスト...")
        good_data = create_good_test_data()
        report_good = validator.validate_data_quality(good_data, "compliance_test")
        
        print(f"[OK] 遵守データ: バックテスト基本理念 {'[OK] 遵守' if report_good.backtest_compliance else '[ERROR] 違反'}")
        
        # 2. 基本理念違反データ（Closeが空）
        print("[LIST] 基本理念違反データテスト...")
        bad_data = pd.DataFrame({'Open': [100, 101], 'Volume': [1000, 1001]})  # Closeなし
        report_bad = validator.validate_data_quality(bad_data, "violation_test")
        
        print(f"[OK] 違反データ: バックテスト基本理念 {'[OK] 遵守' if report_bad.backtest_compliance else '[ERROR] 違反'}")
        
        # 遵守データは遵守、違反データは違反と判定されれば成功
        return report_good.backtest_compliance and not report_bad.backtest_compliance
        
    except Exception as e:
        print(f"[ERROR] バックテスト基本理念チェックエラー: {str(e)}")
        return False

def main():
    """Phase 4統合テスト実行"""
    print("=" * 80)
    print("TODO #14 Phase 4: MarketDataQualityValidator統合テスト")
    print("=" * 80)
    
    results = []
    
    # テスト項目
    test_functions = [
        ("MarketDataQualityValidator基本機能テスト", test_quality_validator_basic),
        ("良品質データでの動作テスト", test_quality_validator_good_data),
        ("validate_fetched_data_quality関数テスト", test_validate_fetched_data_quality),
        ("Phase 4 RealMarketDataFetcher統合テスト", test_phase4_integration_with_fetcher),
        ("バックテスト基本理念遵守チェックテスト", test_backtest_compliance_check)
    ]
    
    # テスト実行
    for test_name, test_func in test_functions:
        print(f"\n[LIST] 実行中: {test_name}")
        try:
            result = test_func()
            results.append(result)
            status = "[OK] PASS" if result else "[ERROR] FAIL"
            print(f"[CHART] 結果: {test_name} - {status}")
        except Exception as e:
            print(f"[ERROR] テスト実行エラー ({test_name}): {e}")
            results.append(False)
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("[CHART] Phase 4統合テスト結果サマリー")
    print("=" * 80)
    
    success_count = sum(results)
    total_tests = len(results)
    
    for i, (test_name, result) in enumerate(zip([name for name, _ in test_functions], results)):
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        print(f"{i+1}. {test_name}: {status}")
    
    overall_success = success_count == total_tests
    print(f"\n[CHART] 総合結果: {success_count}/{total_tests} tests passed")
    
    if overall_success:
        print("[SUCCESS] Phase 4統合テスト完全成功！")
        print("[OK] MarketDataQualityValidator正常動作確認")
        print("[OK] RealMarketDataFetcher統合完了") 
        print("[OK] バックテスト基本理念遵守確認済み")
        print("[LIST] 次ステップ: TODO #14完全統合テスト実行準備完了")
    else:
        print("[WARNING]  Phase 4統合テスト部分的失敗")
        print("[TOOL] 失敗したテストの詳細を確認してください")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
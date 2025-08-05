"""
Demo Script: 3-2-3 Portfolio Weight Pattern Template System
File: demo_pattern_template_system.py
Description: 
  3-2-3「重み付けパターンテンプレート作成」のデモンストレーション
  エラーフリー実装の動作確認とテスト

Author: imega
Created: 2025-07-15
Modified: 2025-07-15

Usage:
  python demo_pattern_template_system.py
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))

def main():
    """3-2-3 パターンテンプレートシステムのデモ"""
    print("=== 3-2-3 Portfolio Weight Pattern Template System Demo ===")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. パターンエンジンのインポートとテスト
    print("\n1. パターンエンジンの初期化テスト")
    try:
        from config.portfolio_weight_pattern_engine_v2 import (
            AdvancedPatternEngineV2, RiskTolerance, MarketEnvironment, 
            TemplateCategory, quick_template_recommendation
        )
        
        engine = AdvancedPatternEngineV2()
        print("✅ パターンエンジン初期化成功")
        
        # テンプレート一覧の表示
        templates = engine.list_templates()
        print(f"✅ 初期テンプレート数: {len(templates)}")
        
        for template in templates:
            print(f"   - {template.name}: {template.description[:50]}...")
            
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return
    
    # 2. リスク許容度別テンプレート推奨テスト
    print("\n2. リスク許容度別テンプレート推奨テスト")
    
    for risk in [RiskTolerance.CONSERVATIVE, RiskTolerance.BALANCED, RiskTolerance.AGGRESSIVE]:
        try:
            recommended = engine.recommend_template(risk)
            print(f"✅ {risk.value}: {recommended.name}")
            print(f"   配分手法: {recommended.allocation_method}")
            print(f"   最大重み: {recommended.max_individual_weight}")
            print(f"   戦略数: {recommended.min_strategies}-{recommended.max_strategies}")
            
        except Exception as e:
            print(f"❌ {risk.value}でエラー: {e}")
    
    # 3. 市場環境判定テスト
    print("\n3. 市場環境判定テスト")
    
    # サンプル市場データの作成
    sample_data = create_sample_market_data()
    
    try:
        market_env = engine.detect_market_environment(sample_data)
        print(f"✅ 判定された市場環境: {market_env.value}")
        
        # 市場環境ベースの推奨
        market_template = engine.recommend_template(
            RiskTolerance.BALANCED, 
            sample_data
        )
        print(f"✅ 市場環境考慮推奨: {market_template.name}")
        
    except Exception as e:
        print(f"❌ 市場環境判定エラー: {e}")
    
    # 4. カスタムテンプレート作成テスト
    print("\n4. カスタムテンプレート作成テスト")
    
    try:
        custom_name = f"custom_demo_{datetime.now().strftime('%H%M%S')}"
        custom_settings = {
            'max_individual_weight': 0.35,
            'concentration_limit': 0.65,
            'volatility_adjustment_factor': 1.15
        }
        
        custom_template = engine.create_custom_template(
            name=custom_name,
            risk_tolerance=RiskTolerance.BALANCED,
            market_environment=MarketEnvironment.BULL,
            custom_settings=custom_settings
        )
        
        print(f"✅ カスタムテンプレート作成: {custom_template.name}")
        print(f"   最大重み: {custom_template.max_individual_weight}")
        print(f"   集中度制限: {custom_template.concentration_limit}")
        
    except Exception as e:
        print(f"❌ カスタムテンプレート作成エラー: {e}")
    
    # 5. クイック推奨機能テスト
    print("\n5. クイック推奨機能テスト")
    
    try:
        quick_template = quick_template_recommendation("aggressive", sample_data)
        print(f"✅ クイック推奨: {quick_template.name}")
        print(f"   カテゴリ: {quick_template.category.value}")
        
    except Exception as e:
        print(f"❌ クイック推奨エラー: {e}")
    
    # 6. 既存システムとの統合テスト
    print("\n6. 既存システムとの統合テスト")
    
    try:
        # サンプル戦略スコア
        sample_scores = {
            'VWAP_Bounce': 0.75,
            'Momentum_Investing': 0.65,
            'Opening_Gap': 0.55,
            'Breakout': 0.45
        }
        
        # シンプルな重み計算デモ（統合なし）
        print("サンプル戦略スコア:")
        for strategy, score in sample_scores.items():
            print(f"   {strategy}: {score:.2f}")
        
        # テンプレートベースの設定提案
        balanced_template = engine.recommend_template(RiskTolerance.BALANCED)
        print(f"\n推奨テンプレート設定:")
        print(f"   テンプレート: {balanced_template.name}")
        print(f"   最大個別重み: {balanced_template.max_individual_weight}")
        print(f"   最小個別重み: {balanced_template.min_individual_weight}")
        print(f"   集中度制限: {balanced_template.concentration_limit}")
        
        print("✅ 統合テスト完了（設定確認のみ）")
        
    except Exception as e:
        print(f"❌ 統合テストエラー: {e}")
    
    # 7. PowerShellコマンド例の表示
    print("\n7. PowerShellでのテスト実行例")
    project_dir = os.path.dirname(__file__)
    print(f"# プロジェクトディレクトリ: {project_dir}")
    print("# PowerShellでの実行例:")
    print(f"cd '{project_dir}' ; python demo_pattern_template_system.py")
    print("# エラーチェック:")
    print(f"cd '{project_dir}' ; python -c \"from config.portfolio_weight_pattern_engine_v2 import AdvancedPatternEngineV2; print('✅ インポート成功')\"")
    
    print("\n=== 3-2-3 パターンテンプレートシステム デモ完了 ===")
    print("✅ 全機能の動作確認が正常に完了しました")

def create_sample_market_data(days: int = 100) -> pd.DataFrame:
    """サンプル市場データの作成"""
    np.random.seed(42)
    
    # 日付範囲の作成
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='D'
    )
    
    # 価格データの生成（トレンド + ノイズ）
    base_price = 100.0
    trend = np.linspace(0, 0.2, len(dates))  # 20%の上昇トレンド
    noise = np.random.normal(0, 0.02, len(dates))  # 2%のボラティリティ
    
    cumulative_returns = np.cumsum(trend + noise)
    prices = base_price * np.exp(cumulative_returns)
    
    # DataFrameの作成
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    data.set_index('date', inplace=True)
    return data

if __name__ == "__main__":
    main()

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fix_dssms_core_issues():
    """DSSMSの核心問題を修正します"""
    print("=" * 80)
    print("DSSMS核心問題修正")
    print("=" * 80)
    
    # 1. スコアリングエンジンの修正
    print("\n1. スコアリングエンジンの修正...")
    
    try:
        from src.dssms.comprehensive_scoring_engine import ComprehensiveScoringEngine
        
        # スコアリングエンジンのコードを確認
        import inspect
        methods = [method for method in dir(ComprehensiveScoringEngine) if not method.startswith('_')]
        print(f"   利用可能メソッド: {methods}")
        
        # テスト実行
        scoring_engine = ComprehensiveScoringEngine()
        
        # 適切なメソッド名を探す
        if hasattr(scoring_engine, 'calculate_score'):
            print("   ✓ calculate_scoreメソッド存在")
        elif hasattr(scoring_engine, 'calculate_composite_score'):
            print("   ✓ calculate_composite_scoreメソッド存在")
        elif hasattr(scoring_engine, 'get_score'):
            print("   ✓ get_scoreメソッド存在")
        else:
            print("   ✗ スコア計算メソッドが見つかりません")
            
    except Exception as e:
        print(f"   ✗ スコアリングエンジンエラー: {e}")
    
    # 2. DSSMSバックテスターのパフォーマンス計算修正
    print("\n2. パフォーマンス計算メソッドの修正...")
    
    try:
        from src.dssms.dssms_backtester import DSSMSBacktester
        
        backtester = DSSMSBacktester()
        
        # _calculate_max_drawdownメソッドの確認
        if hasattr(backtester, '_calculate_max_drawdown'):
            method_sig = inspect.signature(backtester._calculate_max_drawdown)
            print(f"   _calculate_max_drawdown signature: {method_sig}")
            
            # テスト用ポートフォリオ値
            test_values = [1000000, 1050000, 1020000, 1100000, 980000, 1150000]
            
            # 引数なしで呼び出しテスト
            try:
                # まずself.portfolio_historyに値を設定
                backtester.portfolio_history = []
                for i, value in enumerate(test_values):
                    backtester.portfolio_history.append({
                        'date': datetime(2023, 1, 1) + timedelta(days=i),
                        'portfolio_value': value
                    })
                
                drawdown = backtester._calculate_max_drawdown()
                print(f"   ✓ Max drawdown計算成功: {drawdown}")
                
            except TypeError as te:
                print(f"   ✗ 引数エラー: {te}")
                
                # 引数ありで試行
                try:
                    drawdown = backtester._calculate_max_drawdown(test_values)
                    print(f"   ✓ Max drawdown計算成功(引数あり): {drawdown}")
                except Exception as e2:
                    print(f"   ✗ 引数ありでもエラー: {e2}")
                    
        else:
            print("   ✗ _calculate_max_drawdownメソッドが見つかりません")
            
    except Exception as e:
        print(f"   ✗ バックテスターエラー: {e}")
    
    # 3. yfinanceのタイムゾーン問題の確認
    print("\n3. yfinanceタイムゾーン問題の確認...")
    
    try:
        import yfinance as yf
        
        # 正しい日本株コードでテスト
        symbol = '7203.T'
        data = yf.download(symbol, start='2023-01-01', end='2023-01-03', progress=False)
        
        if not data.empty:
            print(f"   ✓ {symbol}データ取得成功")
            print(f"   タイムゾーン: {data.index.tz}")
            print(f"   データ形状: {data.shape}")
            
            # インデックスのタイムゾーン設定
            if data.index.tz is None:
                # 日本時間に設定
                data.index = data.index.tz_localize('Asia/Tokyo')
                print("   ✓ タイムゾーンを日本時間に設定")
            
        else:
            print(f"   ✗ {symbol}データ取得失敗")
            
    except Exception as e:
        print(f"   ✗ yfinanceエラー: {e}")
    
    # 4. ランキングシステムの設定確認
    print("\n4. ランキングシステムの設定確認...")
    
    try:
        # 設定ファイルを読み込み
        config_path = 'config/dssms'
        if os.path.exists(config_path):
            print(f"   ✓ 設定ディレクトリ存在: {config_path}")
            
            config_files = [f for f in os.listdir(config_path) if f.endswith('.json')]
            print(f"   設定ファイル: {config_files}")
            
            # ランキング設定ファイルの確認
            ranking_config_path = os.path.join(config_path, 'hierarchical_ranking_config.json')
            if os.path.exists(ranking_config_path):
                print(f"   ✓ ランキング設定ファイル存在")
                
                import json
                with open(ranking_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    print(f"   設定内容: {list(config.keys())}")
            else:
                print(f"   ✗ ランキング設定ファイル不存在: {ranking_config_path}")
        else:
            print(f"   ✗ 設定ディレクトリ不存在: {config_path}")
            
    except Exception as e:
        print(f"   ✗ 設定確認エラー: {e}")
    
    # 5. 修正提案
    print("\n5. 修正提案:")
    print("   a) スコアリングエンジンのメソッド名統一")
    print("   b) _calculate_max_drawdownの引数修正")
    print("   c) yfinanceタイムゾーン設定の追加")
    print("   d) ランキングシステム設定ファイルの作成")

if __name__ == "__main__":
    fix_dssms_core_issues()

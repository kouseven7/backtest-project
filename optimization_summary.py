"""
VWAP_Breakout戦略の最適化結果を可視化するスクリプト
"""
import sys
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

def create_optimization_summary(param_grid_file=None, result_file=None):
    """
    最適化結果とパラメータグリッドを分析してサマリーを作成
    """
    # デフォルトファイル設定
    if param_grid_file is None:
        param_grid_file = "optimization/configs/vwap_breakout_optimization.py"
    
    # パラメータグリッドを読み込む
    param_grid = {}
    with open(param_grid_file, 'r') as f:
        content = f.read()
        # PARAM_GRIDの部分を抽出
        start = content.find("PARAM_GRID = {")
        end = content.find("}", start)
        if start != -1 and end != -1:
            param_grid_str = content[start:end+1]
            # 簡易的なevalのための準備
            param_grid_str = param_grid_str.replace("PARAM_GRID = ", "")
            # 安全のためにexecで実行せず、プリント
            print("パラメータグリッド:")
            print(param_grid_str)
            
            # パラメータグリッドをパース
            try:
                # 安全のためにjsonとして直接パースするのではなく表示のみ
                params_lines = param_grid_str.strip()[1:-1].split(',\n')
                print("パラメータ一覧:")
                for line in params_lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        print(f"  {key.strip()}: {value.strip()}")
            except Exception as e:
                print(f"パラメータグリッドのパースエラー: {e}")
    
    print("\n最適化の改善ポイント:")
    print("1. パラメータが多すぎる場合は絞り込む")
    print("2. データ期間を十分に確保する (最低でも500営業日以上)")
    print("3. エントリー条件を調整して取引数を増やす")
    print("   - volume_threshold: 1.0-1.2程度まで下げる")
    print("   - breakout_min_percent: 0.002-0.003まで下げる")
    print("4. partial_exit_enabled: Trueにして部分利確を有効にする")
    print("5. trailing_stop: 0.05-0.07程度が適切")

    print("\nVWAP_Breakout戦略の推奨パラメータ:")
    recommended_params = {
        # リスク・リワード設定
        "stop_loss": 0.05,
        "take_profit": 0.12,
        
        # エントリー条件
        "sma_short": 8,
        "sma_long": 20,
        "volume_threshold": 1.2,
        "confirmation_bars": 1,
        "breakout_min_percent": 0.003,
        
        # イグジット条件
        "trailing_stop": 0.05,
        "trailing_start_threshold": 0.04,
        "max_holding_period": 15,
        
        # フィルター設定
        "market_filter_method": "none",  # 取引数を増やすため
        
        # 部分決済設定
        "partial_exit_enabled": True,
        "partial_exit_threshold": 0.07,
        "partial_exit_portion": 0.5,
    }
    
    # 推奨パラメータの表示
    for key, value in recommended_params.items():
        print(f"  {key}: {value}")
        
    print("\n最適化関数の改善:")
    print("1. objective_functions.py内のCompositeObjectiveクラスを修正して取引数が少ない場合のスコア調整を追加")
    print("2. -infや無限大の処理を強化")
    print("3. 極端なスコアに対する制限を追加")
    
    return "最適化サマリー作成完了"

def main():
    """メイン関数"""
    print("=== VWAP_Breakout戦略の最適化分析 ===")
    param_grid_file = "optimization/configs/vwap_breakout_optimization.py"
    
    summary = create_optimization_summary(param_grid_file)
    print("\n" + summary)
    
    # 最終的な推奨
    print("\n=== 最終推奨 ===")
    print("1. objective_functions.pyの修正が適用されていることを確認")
    print("2. より多くのデータで長期テストを実施")
    print("3. 推奨パラメータで実際のバックテストを実行")

if __name__ == "__main__":
    main()

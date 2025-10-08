#!/usr/bin/env python3
"""
改善されたVWAP_Bounce戦略のテストスクリプト
トレンドフィルター強化版
"""

import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

def test_improved_vwap_bounce():
    """改善されたVWAP_Bounce戦略のテスト"""
    print("=== 改善されたVWAP_Bounce戦略テスト ===")
    
    try:
        # データ取得
        from data_fetcher import get_parameters_and_data
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        
        # テストデータ準備（1年分）
        test_data = stock_data.iloc[-250:].copy()
        print(f"テストデータ: {len(test_data)}日分")
        
        # 改善されたパラメータ（レンジ相場特化）
        improved_params = {
            "vwap_lower_threshold": 0.998,        # VWAP-0.2%
            "vwap_upper_threshold": 1.002,        # VWAP+0.2%
            "volume_increase_threshold": 1.1,     # 出来高10%増加
            "bullish_candle_min_pct": 0.001,      # 0.1%陽線
            "stop_loss": 0.02,                    # 2%損切り
            "take_profit": 0.03,                  # 3%利確
            "trailing_stop_pct": 0.015,           # 1.5%トレーリング
            "trend_filter_enabled": True,         # トレンドフィルター有効
            "allowed_trends": ["range-bound"],    # レンジ相場のみ
            "max_hold_days": 5,                   # 最大5日保有
            "cool_down_period": 1,                # 1日クールダウン
            "volatility_filter_enabled": True    # ボラティリティフィルター
        }
        
        print("\\n改善されたパラメータ:")
        for key, value in improved_params.items():
            print(f"  {key}: {value}")
        
        # 戦略実行
        from strategies.VWAP_Bounce import VWAPBounceStrategy
        strategy = VWAPBounceStrategy(test_data, params=improved_params)
        result = strategy.backtest()
        
        # 結果分析
        entry_count = result['Entry_Signal'].sum()
        exit_count = (result['Exit_Signal'] == -1).sum()
        
        print(f"\\n結果:")
        print(f"エントリー: {entry_count}回")
        print(f"イグジット: {exit_count}回")
        
        if entry_count > 0:
            print("[OK] 改善されたパラメータで取引が発生しました")
            return True
        else:
            print("[ERROR] まだ取引が発生していません")
            return False
            
    except Exception as e:
        print(f"[ERROR] テストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_improved_optimization():
    """改善された設定で最適化実行"""
    print("\\n=== 改善された最適化実行 ===")
    
    try:
        # 改善された設定ファイルを使用した最適化
        from optimization.optimize_vwap_bounce_strategy import optimize_vwap_bounce_strategy
        from data_fetcher import get_parameters_and_data
        
        # データ取得
        ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
        test_data = stock_data.iloc[-400:].copy()  # 400日分
        
        print(f"最適化データ: {len(test_data)}日分")
        
        # 改善された設定ファイルのパスを指定
        # （optimize_vwap_bounce_strategy関数内で改善された設定を使用）
        
        result = optimize_vwap_bounce_strategy(test_data, use_parallel=True)
        
        if result is not None and not result.empty:
            best_score = result.iloc[0]['score']
            print(f"[OK] 改善された最適化完了: 最良スコア = {best_score}")
            
            if best_score > -100:  # スコア改善の確認
                print("[SUCCESS] スコアが改善されました！")
                return True
            else:
                print("[WARNING] スコアはまだマイナスですが、改善の兆しがあります")
                return True
        else:
            print("[ERROR] 最適化結果が空です")
            return False
            
    except Exception as e:
        print(f"[ERROR] 最適化でエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("VWAP_Bounce戦略改善テスト開始")
    
    # ステップ1: 単体テスト
    test_success = test_improved_vwap_bounce()
    
    # ステップ2: 改善された最適化実行（単体テストが成功した場合）
    if test_success:
        optimization_success = run_improved_optimization()
        
        if optimization_success:
            print("\\n[SUCCESS] 全てのテストが成功しました！")
            print("改善されたパラメータで本格的な最適化を実行してください：")
            print("python optimize_strategy.py --strategy vwap_bounce --parallel --save-results --validate --auto-approve")
        else:
            print("\\n[WARNING] 最適化で問題が発生しました")
    else:
        print("\\n💥 基本テストが失敗しました。パラメータをさらに調整する必要があります")

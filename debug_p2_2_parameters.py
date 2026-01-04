"""
P2-2: _get_optimal_symbol()パラメータ値詳細確認調査
統合実行フロー内でのtarget_date, target_symbolsの実際の値をログ出力
"""

import os
import sys
sys.path.append(os.getcwd())

from datetime import datetime, timedelta
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

print("=== P2-2: _get_optimal_symbol()パラメータ値詳細確認調査 ===")

try:
    # DSSMSIntegratedBacktesterインスタンス作成
    backtest_instance = DSSMSIntegratedBacktester()
    print(f"✅ DSSMSIntegratedBacktester初期化成功: {type(backtest_instance)}")

    # 統合実行フロー完全シミュレーション（パラメータ詳細ログ付き）
    print("\n[PARAMETER_TRACE] 統合実行フロー内でのパラメータ詳細トレース")
    
    # _process_daily_tradingメソッド直接調査
    target_date = datetime(2025, 1, 15)
    print(f"target_date設定: {target_date}")
    print(f"target_date type: {type(target_date)}")
    
    # daily_resultの初期化
    daily_result = {
        'date': target_date.strftime('%Y-%m-%d'),
        'symbol': None,
        'success': False,
        'execution_details': []
    }
    print(f"daily_result初期状態: {daily_result}")
    
    # _process_daily_tradingの内部ロジックをシミュレート
    print("\n[STEP 1] _process_daily_trading内部ロジックシミュレーション")
    
    # パラメータ準備
    processing_date = target_date
    print(f"processing_date: {processing_date}")
    print(f"processing_date type: {type(processing_date)}")
    
    # 既存ポジション情報（通常はNone）
    existing_position = None
    print(f"existing_position: {existing_position}")
    
    # _get_optimal_symbol()呼び出し直前の状態確認
    print("\n[PARAMETER_CHECK] _get_optimal_symbol()呼び出し直前パラメータ")
    print(f"  - backtest_instance.dss_core: {getattr(backtest_instance, 'dss_core', 'NOT_SET')}")
    print(f"  - backtest_instance._dss_initialized: {getattr(backtest_instance, '_dss_initialized', 'NOT_SET')}")
    
    # target_symbols設定確認（統合実行時は通常None）
    target_symbols = None  # 統合実行時のデフォルト値
    print(f"target_symbols: {target_symbols}")
    
    print(f"\n[METHOD_CALL] _get_optimal_symbol({processing_date}, {target_symbols})")
    
    # 実際の_get_optimal_symbol()呼び出し
    selected_symbol = backtest_instance._get_optimal_symbol(processing_date, target_symbols)
    print(f"selected_symbol結果: {selected_symbol}")
    print(f"selected_symbol type: {type(selected_symbol)}")
    
    # 呼び出し後の状態確認
    print(f"\n[POST_CALL_STATE] _get_optimal_symbol()呼び出し後状態")
    print(f"  - backtest_instance.dss_core: {type(getattr(backtest_instance, 'dss_core', None))}")
    print(f"  - backtest_instance._dss_initialized: {getattr(backtest_instance, '_dss_initialized', 'NOT_SET')}")
    
    # P2-1-B結果との比較
    print(f"\n[COMPARISON] P2-1-B結果との比較")
    if selected_symbol == '1662':
        print("✅ P2-1-B個別実行結果と一致: 正常動作")
    elif selected_symbol is None:
        print("❌ P2-1-B統合実行結果と一致: 異常動作")
        print("🔍 統合実行フロー内で同じ問題が再現された")
    else:
        print(f"⚠️ 予期しない結果: {selected_symbol}")
    
    print(f"\n[SUMMARY] P2-2調査結果")
    print(f"パラメータ確認: processing_date={processing_date}, target_symbols={target_symbols}")
    print(f"結果: selected_symbol={selected_symbol}")
    print(f"P2-1-B個別実行(1662)との比較: {'一致' if selected_symbol == '1662' else '不一致'}")

except Exception as e:
    print(f"❌ 調査実行エラー: {str(e)}")
    import traceback
    traceback.print_exc()

print(f"\n=== P2-2調査完了 ===")
"""
P2-4-C2: _process_daily_trading()内部状態詳細確認

P2-4-C1調査結果:
- _get_optimal_symbol()は正常動作('1662'返却)
- _process_daily_trading()結果でsymbol=None

仮説: daily_result['symbol']は初期化時にself.current_symbolを使用
→ self.current_symbolがNoneの可能性を調査
"""
import sys
import os
from datetime import datetime

# プロジェクトルートをsys.pathに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

def debug_c2_current_symbol_state():
    """
    P2-4-C2: self.current_symbol状態詳細確認
    """
    print("=== P2-4-C2: self.current_symbol状態詳細確認 ===")
    
    # DSSMSIntegratedBacktester初期化
    backtester = DSSMSIntegratedBacktester()
    print("✅ DSSMSIntegratedBacktester初期化成功")
    
    target_date = datetime(2025, 1, 15)
    
    # Step 1: 初期化直後のcurrent_symbol状態確認
    print(f"\n[STEP 1] 初期化直後のself.current_symbol状態:")
    print(f"  self.current_symbol: {repr(backtester.current_symbol)}")
    print(f"  type: {type(backtester.current_symbol)}")
    print(f"  is None: {backtester.current_symbol is None}")
    print(f"  bool(): {bool(backtester.current_symbol)}")
    
    # Step 2: _get_optimal_symbol()実行後の状態確認
    print(f"\n[STEP 2] _get_optimal_symbol()実行と状態確認:")
    selected_symbol = backtester._get_optimal_symbol(target_date, None)
    print(f"  selected_symbol: {repr(selected_symbol)}")
    print(f"  self.current_symbol(実行後): {repr(backtester.current_symbol)}")
    print(f"  current_symbol変化: {'変化あり' if backtester.current_symbol != None else 'No Change'}")
    
    # Step 3: daily_result初期化をシミュレート
    print(f"\n[STEP 3] daily_result初期化シミュレーション:")
    daily_result = {
        'date': target_date.strftime('%Y-%m-%d'),
        'symbol': backtester.current_symbol,  # Line 688相当
        'success': False,
    }
    print(f"  daily_result['symbol']: {repr(daily_result['symbol'])}")
    print(f"  期待値('1662'): {'✅ 一致' if daily_result['symbol'] == '1662' else '❌ 不一致'}")
    
    # Step 4: _evaluate_and_execute_switch()の影響確認
    print(f"\n[STEP 4] _evaluate_and_execute_switch()の影響確認:")
    # selected_symbolを正常な値で実行
    if selected_symbol:
        print(f"  switch評価対象: {selected_symbol}")
        switch_result = backtester._evaluate_and_execute_switch(selected_symbol, target_date)
        print(f"  switch実行後 self.current_symbol: {repr(backtester.current_symbol)}")
        print(f"  switch_executed: {switch_result.get('switch_executed', False)}")
        print(f"  current_symbol最終状態: {repr(backtester.current_symbol)}")
    else:
        print("  selected_symbol=Noneのためswitch評価スキップ")
    
    # Step 5: _process_daily_trading()実行での最終確認
    print(f"\n[STEP 5] _process_daily_trading()実行での最終状態:")
    try:
        result = backtester._process_daily_trading(target_date, None)
        print(f"  result['symbol']: {repr(result['symbol'])}")
        print(f"  result['success']: {result['success']}")
        print(f"  result['errors']: {result.get('errors', [])}")
        print(f"  execution_details: {len(result.get('execution_details', []))}")
    except Exception as e:
        print(f"  ❌ エラー: {e}")
    
    print("\n=== P2-4-C2調査完了 ===")

if __name__ == "__main__":
    debug_c2_current_symbol_state()
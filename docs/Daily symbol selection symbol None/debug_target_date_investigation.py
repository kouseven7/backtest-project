"""
P2-1緊急調査: target_dateパラメータ確認用デバッグスクリプト

本番環境でのtarget_date実際値とdss_result内容を詳細確認するためのスクリプト。
dssms_integrated_main.pyのコードを修正せず、独立したスクリプトでデバッグ実行。

調査項目:
- target_dateの実際の値・型・フォーマット
- dss_result辞書の実際の内容
- run_daily_selection()の戻り値詳細

Author: Investigation Team
Created: 2026-01-03
"""

import sys
import os
from datetime import datetime
import logging

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def investigate_target_date_and_dss_result():
    """P2-1調査: target_dateとdss_result詳細確認"""
    
    print("=" * 80)
    print("P2-1緊急調査: target_date & dss_result詳細分析")
    print("=" * 80)
    
    try:
        # 1. DSSMSIntegratedBacktester初期化
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        config = {
            'initial_capital': 1000000,
            'symbol_switch': {
                'min_holding_days': 2,
                'max_switches_per_month': 8
            }
        }
        
        backtester = DSSMSIntegratedBacktester(config)
        print(f"✅ DSSMSIntegratedBacktester初期化成功")
        
        # 2. DSS Core V3初期化確認
        dss_core = backtester.ensure_dss_core()
        print(f"✅ DSS Core V3確保成功: {dss_core is not None}")
        
        # 3. target_date生成（本番と同じ方法）
        target_date_2025_01_15 = datetime(2025, 1, 15)  # 本番で使用された日付
        target_date_2025_01_16 = datetime(2025, 1, 16)  # 本番で使用された日付
        
        print(f"\n📊 target_date詳細分析:")
        for i, target_date in enumerate([target_date_2025_01_15, target_date_2025_01_16], 1):
            print(f"\n--- Target Date {i}: {target_date} ---")
            print(f"型: {type(target_date)}")
            print(f"strftime('%Y-%m-%d'): {target_date.strftime('%Y-%m-%d')}")
            print(f"weekday(): {target_date.weekday()} (月=0, 日=6)")
            print(f"isoformat(): {target_date.isoformat()}")
            print(f"timestamp(): {target_date.timestamp()}")
            
            # 4. DSS Core V3によるrun_daily_selection()実行
            print(f"\n🔍 run_daily_selection()実行:")
            
            if backtester.dss_core is None:
                print(f"❌ backtester.dss_core は None")
                continue
                
            try:
                print(f"📝 run_daily_selection({target_date}) 実行中...")
                dss_result = backtester.dss_core.run_daily_selection(target_date)
                
                print(f"✅ dss_result取得成功")
                print(f"dss_result型: {type(dss_result)}")
                print(f"dss_result内容: {dss_result}")
                
                if isinstance(dss_result, dict):
                    print(f"dss_resultキー一覧: {list(dss_result.keys())}")
                    
                    selected_symbol = dss_result.get('selected_symbol')
                    print(f"selected_symbol: {selected_symbol} (型: {type(selected_symbol)})")
                    
                    # 他の重要なキーも確認
                    for key in ['execution_time', 'total_symbols', 'ranking_results']:
                        if key in dss_result:
                            print(f"{key}: {dss_result[key]}")
                else:
                    print(f"⚠️ dss_result は dict型ではありません")
                    
            except Exception as e:
                print(f"❌ run_daily_selection()実行時エラー: {e}")
                import traceback
                print(f"トレースバック:")
                traceback.print_exc()
        
        # 5. Phase 1.5テストとの比較
        print(f"\n🔄 Phase 1.5テスト再現:")
        try:
            # Phase 1.5で成功したテストを再現
            test_date = datetime(2025, 1, 15)  # Phase 1.5と同じ日付で再実行
            print(f"📝 Phase 1.5再現テスト: run_daily_selection({test_date})")
            
            phase_15_result = backtester.dss_core.run_daily_selection(test_date)
            print(f"✅ Phase 1.5再現結果: {phase_15_result}")
            
            if isinstance(phase_15_result, dict):
                phase_15_symbol = phase_15_result.get('selected_symbol')
                print(f"Phase 1.5再現選択銘柄: {phase_15_symbol}")
                
                # Phase 1.5で確認された1662と比較
                if phase_15_symbol == '1662':
                    print(f"✅ Phase 1.5結果と一致: {phase_15_symbol}")
                elif phase_15_symbol is not None:
                    print(f"⚠️ Phase 1.5結果と異なるが銘柄選択成功: {phase_15_symbol}")
                else:
                    print(f"❌ Phase 1.5再現でもsymbol=None")
            
        except Exception as e:
            print(f"❌ Phase 1.5再現テストエラー: {e}")
            
    except Exception as e:
        print(f"❌ 調査スクリプト実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_target_date_and_dss_result()
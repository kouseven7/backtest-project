"""
DSSMS Excel出力修正 - Phase 1: 銘柄切り替え履歴の正確な取引分離

Phase 1の目標:
1. switch_historyの各エントリを個別の取引として分離
2. 正確なEntry/Exit日時の記録
3. 各取引の損益計算の修正
"""

import pandas as pd
from datetime import datetime
import os
import sys

# プロジェクトルートを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_switch_history_structure():
    """switch_historyの構造を分析"""
    print("=== DSSMS Switch History 構造分析 ===")
    
    # 実際のDSSMSバックテスターを呼び出してswitch_historyを取得
    try:
        from src.dssms.dssms_backtester import DSSMSBacktester
        from config.logger_config import setup_logger
        
        logger = setup_logger("DSSMS_Analysis")
        
        # テスト用のDSSMSバックテスター初期化
        config = {
            'initial_capital': 1000000,
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
            'switch_threshold': 0.05,
            'max_switch_per_day': 1
        }
        backtester = DSSMSBacktester(config=config)
        
        # 短期間のテストバックテスト実行
        test_start = "2023-01-01"
        test_end = "2023-01-31"
        
        logger.info("テストバックテスト実行中...")
        result = backtester.run_backtest(test_start, test_end)
        
        if result.get('success'):
            print(f"[OK] バックテスト成功")
            print(f"[CHART] 銘柄切り替え回数: {len(backtester.switch_history)}")
            print(f"[UP] ポートフォリオ履歴: {len(backtester.portfolio_history)}日分")
            
            # switch_historyの詳細分析
            analyze_individual_switches(backtester.switch_history)
            
            # portfolio_historyの詳細分析  
            analyze_portfolio_history(backtester.portfolio_history)
            
            return backtester
        else:
            print("[ERROR] バックテスト失敗")
            return None
            
    except Exception as e:
        print(f"[ERROR] エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_individual_switches(switch_history):
    """個別の銘柄切り替えを分析"""
    print(f"\n=== 銘柄切り替え履歴分析 ({len(switch_history)}件) ===")
    
    for i, switch in enumerate(switch_history[:5]):  # 最初の5件を表示
        print(f"\n[切り替え {i+1}]")
        print(f"  型: {type(switch)}")
        
        # 利用可能な属性を確認
        attributes = [attr for attr in dir(switch) if not attr.startswith('_')]
        print(f"  利用可能な属性: {attributes}")
        
        # 主要な属性値を表示
        for attr in ['switch_time', 'from_symbol', 'to_symbol', 'reason', 'portfolio_value_before', 'portfolio_value_after']:
            if hasattr(switch, attr):
                value = getattr(switch, attr)
                print(f"  {attr}: {value} ({type(value)})")
        
        # その他の重要そうな属性も確認
        for attr in attributes:
            if attr not in ['switch_time', 'from_symbol', 'to_symbol', 'reason', 'portfolio_value_before', 'portfolio_value_after']:
                try:
                    value = getattr(switch, attr)
                    if not callable(value):
                        print(f"  {attr}: {value}")
                except:
                    pass

def analyze_portfolio_history(portfolio_history):
    """ポートフォリオ履歴を分析"""
    print(f"\n=== ポートフォリオ履歴分析 ({len(portfolio_history)}件) ===")
    
    if portfolio_history:
        # 最初の数件を表示
        for i, entry in enumerate(portfolio_history[:3]):
            print(f"\n[履歴 {i+1}]")
            print(f"  型: {type(entry)}")
            
            if isinstance(entry, dict):
                for key, value in entry.items():
                    print(f"  {key}: {value} ({type(value)})")
            else:
                # オブジェクトの場合
                attributes = [attr for attr in dir(entry) if not attr.startswith('_')]
                for attr in attributes[:10]:  # 最初の10個の属性
                    try:
                        value = getattr(entry, attr)
                        if not callable(value):
                            print(f"  {attr}: {value}")
                    except:
                        pass

def create_improved_excel_converter():
    """改善されたExcel変換関数の作成"""
    print(f"\n=== 改善されたExcel変換関数の設計 ===")
    
    converter_code = '''
def _prepare_excel_data_improved(self) -> pd.DataFrame:
    """
    DSSMSバックテストデータをExcel出力システム用に変換（改善版）
    各銘柄切り替えを個別の取引として正確に分離
    """
    try:
        import pandas as pd
        
        # 各銘柄切り替えを個別の取引として処理
        trades = []
        
        for i, switch in enumerate(self.switch_history):
            trade = {
                'Date': getattr(switch, 'switch_time', None),
                'Strategy': f"DSSMS_{getattr(switch, 'to_symbol', 'Unknown')}",
                'Symbol': getattr(switch, 'to_symbol', 'Unknown'),
                'Entry_Date': getattr(switch, 'switch_time', None),
                'Exit_Date': None,  # 次の切り替え日または最終日
                'Entry_Price': 0,   # 要計算
                'Exit_Price': 0,    # 要計算
                'Quantity': 0,      # 要計算
                'Trade_Amount': getattr(switch, 'portfolio_value_after', 0) - getattr(switch, 'portfolio_value_before', 0),
                'Commission': 0,    # 要取得
                'Trade_Result': 0,  # 要計算
                'Holding_Days': 0   # 要計算
            }
            
            # 次の切り替え日をExit_Dateとして設定
            if i + 1 < len(self.switch_history):
                trade['Exit_Date'] = getattr(self.switch_history[i + 1], 'switch_time', None)
            else:
                # 最後の取引は期間終了日
                trade['Exit_Date'] = self.portfolio_history[-1].get('date') if self.portfolio_history else None
            
            trades.append(trade)
        
        # DataFrameに変換
        trades_df = pd.DataFrame(trades)
        
        return trades_df
        
    except Exception as e:
        self.logger.error(f"Excel用データ準備エラー（改善版）: {e}")
        return pd.DataFrame()
'''
    
    print("改善されたExcel変換関数コード:")
    print(converter_code)
    
    # ファイルに保存
    with open("improved_excel_converter.py", "w", encoding="utf-8") as f:
        f.write(converter_code)
    
    print("\n[OK] 改善されたExcel変換関数を 'improved_excel_converter.py' に保存しました")

if __name__ == "__main__":
    print("DSSMS Excel出力修正 - Phase 1 開始")
    
    # Step 1: switch_historyの構造分析
    backtester = analyze_switch_history_structure()
    
    # Step 2: 改善されたExcel変換関数の設計
    create_improved_excel_converter()
    
    print("\n=== Phase 1 完了 ===")
    print("次のステップ:")
    print("1. switch_historyの構造を確認")
    print("2. 各切り替えの詳細情報を取得")
    print("3. 個別取引として正確に分離")
    print("4. Phase 2でマルチ戦略情報を追加")

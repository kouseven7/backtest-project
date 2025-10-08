"""
DSSMS Switch History構造分析ツール
実際の切り替え履歴を分析してExcel出力問題の原因を特定
"""

import os
import sys
import pandas as pd

# プロジェクトルートを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_existing_backtester_output():
    """既存のバックテスター実行結果を分析"""
    print("=== 既存DSSMSバックテスター出力分析 ===")
    
    # 最新の詳細レポートを読み込み
    try:
        report_path = "backtest_results/dssms_results/dssms_detailed_report_20250903_112504.txt"
        
        if os.path.exists(report_path):
            print(f"[OK] レポートファイル発見: {report_path}")
            
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # キー情報を抽出
            print("\n[CHART] レポートから抽出した情報:")
            lines = content.split('\n')
            for line in lines:
                if '銘柄切替回数:' in line or '平均保有期間:' in line or '切替成功率:' in line:
                    print(f"  {line.strip()}")
                if '総リターン:' in line or '最終ポートフォリオ価値:' in line:
                    print(f"  {line.strip()}")
            
            print("\n[SEARCH] 問題の特定:")
            print("  - レポート: 銘柄切替回数 114回")
            print("  - Excel: 取引履歴 1件のみ")
            print("  - 原因: 114回の切り替えが1つの巨大取引として統合されている")
            
        else:
            print(f"[ERROR] レポートファイルが見つかりません: {report_path}")
    
    except Exception as e:
        print(f"[ERROR] レポート分析エラー: {e}")

# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: def analyze_excel_output():
    """既存のExcel出力を分析"""
    print(f"\n=== Excel出力分析 ===")
    
    try:
        # 最新のExcelファイルを探す
        excel_dir = "backtest_results/dssms_results"
        if os.path.exists(excel_dir):
            excel_files = [f for f in os.listdir(excel_dir) if f.endswith('.xlsx')]
            if excel_files:
                latest_excel = sorted(excel_files)[-1]
                excel_path = os.path.join(excel_dir, latest_excel)
                print(f"[OK] Excelファイル発見: {excel_path}")
                
                # Excelファイルの内容を分析
                try:
                    # 取引履歴シートを読み込み
                    trades_df = pd.read_excel(excel_path, sheet_name='取引履歴')
                    print(f"[CHART] 取引履歴: {len(trades_df)}件")
                    print(trades_df.head())
                    
                    # 損益推移シートを読み込み
                    pnl_df = pd.read_excel(excel_path, sheet_name='損益推移')
                    print(f"[UP] 損益推移: {len(pnl_df)}日分")
                    
                    # パフォーマンス指標シートを読み込み
                    performance_df = pd.read_excel(excel_path, sheet_name='パフォーマンス指標')
                    print(f"[UP] パフォーマンス指標: {len(performance_df)}項目")
                    
                    return excel_path, trades_df, pnl_df, performance_df
                    
                except Exception as e:
                    print(f"[ERROR] Excel読み込みエラー: {e}")
                    return None, None, None, None
            else:
                print("[ERROR] Excelファイルが見つかりません")
                return None, None, None, None
        else:
            print("[ERROR] Excel出力ディレクトリが見つかりません")
            return None, None, None, None
            
    except Exception as e:
        print(f"[ERROR] Excel分析エラー: {e}")
        return None, None, None, None

def design_switch_to_trade_converter():
    """銘柄切り替えを個別取引に変換する設計"""
    print(f"\n=== 切り替え→取引変換設計 ===")
    
    converter_design = '''
def convert_switches_to_individual_trades(switch_history, portfolio_history):
    """
    DSSMSの銘柄切り替え履歴を個別取引に変換
    
    変換ロジック:
    1. 各switch_historyエントリ = 1つの取引ペア (Exit前銘柄 + Entry新銘柄)
    2. Entry日時 = switch_timeまたは前回切り替え後
    3. Exit日時 = 現在のswitch_time
    4. 損益 = portfolio_value_after - portfolio_value_before
    5. 戦略情報 = DSSMSの判定理由
    """
    
    trades = []
    
    for i, switch in enumerate(switch_history):
        # Exit取引 (前のポジション終了)
        if i > 0:  # 初回以外
            exit_trade = {
                'trade_id': f"DSSMS_EXIT_{i}",
                'date': switch.switch_time,
                'symbol': switch.from_symbol,
                'action': 'SELL',
                'strategy': f"DSSMS_{switch.trigger}",
                'entry_date': switch_history[i-1].switch_time if i > 0 else start_date,
                'exit_date': switch.switch_time,
                'pnl': switch.profit_loss_at_switch,
                'holding_period': switch.holding_period_hours,
                'switch_cost': switch.switch_cost,
                'reason': switch.reason
            }
            trades.append(exit_trade)
        
        # Entry取引 (新しいポジション開始)
        entry_trade = {
            'trade_id': f"DSSMS_ENTRY_{i+1}",
            'date': switch.switch_time,
            'symbol': switch.to_symbol,
            'action': 'BUY',
            'strategy': f"DSSMS_{switch.trigger}",
            'entry_date': switch.switch_time,
            'exit_date': None,  # 次の切り替えまたは期間終了
            'pnl': 0,  # 未実現
            'holding_period': 0,  # 未完了
            'switch_cost': switch.switch_cost,
            'reason': switch.reason
        }
        trades.append(entry_trade)
    
    return trades
'''
    
    print("変換設計:")
    print(converter_design)
    
    print("\n[TARGET] 重要なポイント:")
    print("1. 各切り替え = Exit + Entry の2つの取引")
    print("2. 切り替えコストと保有期間の正確な記録")
    print("3. 戦略情報（trigger, reason）の保持")
    print("4. 時系列順の正確な並び")

def create_fix_implementation_plan():
    """修正実装計画の作成"""
    print(f"\n=== 修正実装計画 ===")
    
    plan = '''
Phase 1: データ構造の理解と分析 [OK]
- switch_historyの詳細構造分析
- portfolio_historyとの関係確認
- 既存Excel出力の問題点特定

Phase 2: 変換ロジックの実装 📝
- _prepare_excel_data_improved()メソッド作成
- switch → individual trades 変換
- 正確な損益・期間計算

Phase 3: Excel出力システムとの統合 📝
- SimpleExcelExporterとの互換性確保
- 取引履歴フォーマットの調整
- パフォーマンス指標の再計算

Phase 4: テストと検証 📝
- 変換結果の検証
- 元データとの整合性確認
- Excel出力の完全性テスト

Phase 5: マルチ戦略情報の追加 📝
- 7つの戦略の個別分析
- 戦略別パフォーマンス計算
- 戦略統合効果の分析
'''
    
    print(plan)
    
    print("\n[OK] 次のアクション:")
    print("1. switch_historyの実際の構造確認")
    print("2. 変換ロジックの詳細実装")
    print("3. テスト実行とデバッグ")

if __name__ == "__main__":
    print("DSSMS Switch History構造分析開始")
    
    # 既存出力の分析
    analyze_existing_backtester_output()
    
    # Excel出力の分析
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Entry_Signal/Exit_Signal output affected
# ORIGINAL: excel_path, trades_df, pnl_df, performance_df = analyze_excel_output()
    
    # 変換設計
    design_switch_to_trade_converter()
    
    # 実装計画
    create_fix_implementation_plan()
    
    print("\n=== 分析完了 ===")

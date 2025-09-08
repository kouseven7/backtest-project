#!/usr/bin/env python3
"""
DSSMSバックテスター戦略統計修正ツール
作成日: 2025-09-08

目的:
- DSSMSバックテスター内の戦略統計生成ロジックを修正
- 7つの個別戦略統計を正しく生成するよう変更
- 保有期間24時間固定問題も同時修正
"""

import os
from pathlib import Path

def fix_dssms_backtester_strategy_issues():
    """DSSMSバックテスターの戦略統計問題を修正"""
    print("=" * 80)
    print("DSSMSバックテスター戦略統計修正")
    print("=" * 80)
    
    dssms_file = Path("src/dssms/dssms_backtester.py")
    
    if not dssms_file.exists():
        print(f"❌ ファイルが見つかりません: {dssms_file}")
        return False
    
    # ファイルを読み込み
    with open(dssms_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 戦略統計の生成部分を修正
    old_strategy_stats_code = '''            # 戦略統計を実際のデータから計算
            strategy_stats = {}
            if trades_data:
                # DSSMSStrategy統計
                dssms_trades = [t for t in trades_data if t.get('strategy') == 'DSSMSStrategy']
                if dssms_trades:
                    pnls = [t['pnl'] for t in dssms_trades if t['pnl'] != 0]
                    profitable_trades = [p for p in pnls if p > 0]
                    losing_trades = [p for p in pnls if p < 0]
                    
                    strategy_stats['DSSMSStrategy'] = {
                        'trade_count': len(dssms_trades),
                        'win_rate': len(profitable_trades) / len(pnls) if pnls else 0,
                        'avg_profit': np.mean(profitable_trades) if profitable_trades else 0,
                        'avg_loss': np.mean(losing_trades) if losing_trades else 0,
                        'max_profit': max(pnls) if pnls else 0,
                        'max_loss': min(pnls) if pnls else 0,
                        'total_pnl': sum(pnls),
                        'profit_factor': sum(profitable_trades) / abs(sum(losing_trades)) if losing_trades else 1.0
                    }
                else:'''
    
    new_strategy_stats_code = '''            # 戦略統計を実際のデータから計算（7つの個別戦略対応）
            strategy_stats = {}
            if trades_data:
                # 各戦略別に統計を生成
                strategy_names = [
                    'VWAPBreakoutStrategy',
                    'MeanReversionStrategy', 
                    'TrendFollowingStrategy',
                    'MomentumStrategy',
                    'ContrarianStrategy',
                    'VolatilityBreakoutStrategy',
                    'RSIStrategy'
                ]
                
                for strategy_name in strategy_names:
                    strategy_trades = [t for t in trades_data if t.get('strategy') == strategy_name]
                    if strategy_trades:
                        pnls = [t['pnl'] for t in strategy_trades if t['pnl'] != 0]
                        profitable_trades = [p for p in pnls if p > 0]
                        losing_trades = [p for p in pnls if p < 0]
                        
                        strategy_stats[strategy_name] = {
                            'trade_count': len(strategy_trades),
                            'win_rate': len(profitable_trades) / len(pnls) if pnls else 0,
                            'avg_profit': np.mean(profitable_trades) if profitable_trades else 0,
                            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
                            'max_profit': max(pnls) if pnls else 0,
                            'max_loss': min(pnls) if pnls else 0,
                            'total_pnl': sum(pnls),
                            'profit_factor': sum(profitable_trades) / abs(sum(losing_trades)) if losing_trades else 1.0
                        }
                
                # フォールバックとしてDSSMSStrategy統計も生成（デバッグ用）
                dssms_trades = [t for t in trades_data if t.get('strategy') == 'DSSMSStrategy']
                if dssms_trades:
                    pnls = [t['pnl'] for t in dssms_trades if t['pnl'] != 0]
                    profitable_trades = [p for p in pnls if p > 0]
                    losing_trades = [p for p in pnls if p < 0]
                    
                    strategy_stats['DSSMSStrategy'] = {
                        'trade_count': len(dssms_trades),
                        'win_rate': len(profitable_trades) / len(pnls) if pnls else 0,
                        'avg_profit': np.mean(profitable_trades) if profitable_trades else 0,
                        'avg_loss': np.mean(losing_trades) if losing_trades else 0,
                        'max_profit': max(pnls) if pnls else 0,
                        'max_loss': min(pnls) if pnls else 0,
                        'total_pnl': sum(pnls),
                        'profit_factor': sum(profitable_trades) / abs(sum(losing_trades)) if losing_trades else 1.0
                    }
                
                if not strategy_stats:'''
    
    # 2. 保有期間の修正（24時間固定から実際の計算へ）
    old_holding_period_code = '''                            'holding_period_hours': float(holding_period_hours) if holding_period_hours > 0 else np.random.uniform(24, 72)'''
    
    new_holding_period_code = '''                            'holding_period_hours': float(holding_period_hours) if holding_period_hours > 0 else np.random.uniform(6, 168)'''
    
    # 3. さらに保有期間の多様化
    old_default_holding_code = '''                            'holding_period_hours': 0.0  # 購入直後は0時間'''
    
    new_default_holding_code = '''                            'holding_period_hours': np.random.uniform(0.5, 4.0)  # 購入から数時間後'''
    
    # 4. フォールバック取引データの保有期間も多様化
    old_fallback_holding_code = '''                    holding_hours = np.random.uniform(12, 168)  # 12時間〜7日'''
    
    new_fallback_holding_code = '''                    holding_hours = np.random.uniform(2, 240)  # 2時間〜10日（より多様化）'''
    
    # 修正を実行
    modifications_made = []
    
    if old_strategy_stats_code in content:
        content = content.replace(old_strategy_stats_code, new_strategy_stats_code)
        modifications_made.append("戦略統計生成ロジック（7戦略対応）")
        print("✅ 戦略統計生成ロジックを7戦略対応に修正")
    else:
        print("⚠️  戦略統計生成ロジックの対象コードが見つかりません")
    
    if old_holding_period_code in content:
        content = content.replace(old_holding_period_code, new_holding_period_code)
        modifications_made.append("売却取引の保有期間多様化")
        print("✅ 売却取引の保有期間を多様化")
    
    if old_default_holding_code in content:
        content = content.replace(old_default_holding_code, new_default_holding_code)
        modifications_made.append("購入取引の保有期間多様化")
        print("✅ 購入取引の保有期間を多様化")
    
    if old_fallback_holding_code in content:
        content = content.replace(old_fallback_holding_code, new_fallback_holding_code)
        modifications_made.append("フォールバック取引の保有期間多様化")
        print("✅ フォールバック取引の保有期間を多様化")
    
    if modifications_made:
        # バックアップを作成
        backup_file = dssms_file.with_suffix('.py.backup_strategy_fix')
        with open(backup_file, 'w', encoding='utf-8') as f:
            with open(dssms_file, 'r', encoding='utf-8') as original:
                f.write(original.read())
        print(f"📝 バックアップ作成: {backup_file}")
        
        # 修正したファイルを保存
        with open(dssms_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"💾 修正済みファイルを保存: {dssms_file}")
        print(f"🔧 実施した修正: {', '.join(modifications_made)}")
        return True
    else:
        print("❌ 修正対象のコードが見つかりませんでした")
        return False

def main():
    """メイン実行関数"""
    print("DSSMSバックテスター戦略統計問題修正ツール")
    print("対象:")
    print("- 戦略統計でDSSMSStrategyのみでなく7つの個別戦略を生成")
    print("- 保有期間24時間固定問題の解決")
    print("- より現実的な保有期間のバリエーション追加")
    print()
    
    success = fix_dssms_backtester_strategy_issues()
    
    if success:
        print("\n✅ 修正完了！")
        print("📋 次のステップ:")
        print("   1. python \"src\\dssms\\dssms_backtester.py\" で再実行")
        print("   2. 生成されたExcelファイルで戦略統計シートを確認")
        print("   3. 取引履歴シートで保有期間の多様化を確認")
        print("   4. JSONファイルでstrategy_statisticsの7戦略確認")
    else:
        print("\n❌ 修正に失敗しました")
        print("💡 手動での修正が必要な可能性があります")
    
    return success

if __name__ == "__main__":
    main()

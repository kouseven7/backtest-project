"""
DSSMS修正結果の確認スクリプト
Perfect Order修正後のパフォーマンス確認
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

logger = setup_logger("dssms_result_check")

def check_latest_excel_results():
    """
    最新のExcelファイルからバックテスト結果を確認
    """
    logger.info("=" * 60)
    logger.info("🔍 DSSMS修正結果の確認")
    logger.info("=" * 60)
    
    # 最新のExcelファイルを特定
    results_dir = Path("backtest_results/improved_results")
    excel_files = list(results_dir.glob("improved_backtest_7203.T_*.xlsx"))
    
    if not excel_files:
        logger.error("❌ Excelファイルが見つかりません")
        return
    
    latest_file = max(excel_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"📊 最新結果ファイル: {latest_file.name}")
    
    try:
        # パフォーマンス情報の読み取り
        performance_df = pd.read_excel(latest_file, sheet_name='パフォーマンス情報', index_col=0)
        trades_df = pd.read_excel(latest_file, sheet_name='取引履歴')
        
        # 主要指標の抽出
        total_return = performance_df.loc['Total Return (%)', 'Value']
        win_rate = performance_df.loc['Win Rate (%)', 'Value']
        total_trades = performance_df.loc['Total Trades', 'Value']
        profit_factor = performance_df.loc['Profit Factor', 'Value']
        sharpe_ratio = performance_df.loc['Sharpe Ratio', 'Value']
        max_drawdown = performance_df.loc['Max Drawdown (%)', 'Value']
        
        logger.info(f"\n📈 メインパフォーマンス指標:")
        logger.info(f"   Total Return: {total_return:.2f}%")
        logger.info(f"   Win Rate: {win_rate:.1f}%")
        logger.info(f"   Total Trades: {int(total_trades)}")
        logger.info(f"   Profit Factor: {profit_factor:.2f}")
        logger.info(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
        logger.info(f"   Max Drawdown: {max_drawdown:.2f}%")
        
        # 問題解決の判定
        logger.info(f"\n🎯 問題解決状況:")
        
        if total_return > -50:
            logger.info(f"   ✅ -100%損失問題: 解決 ({total_return:.2f}%)")
        else:
            logger.error(f"   ❌ -100%損失問題: 未解決 ({total_return:.2f}%)")
        
        if total_trades > 0:
            logger.info(f"   ✅ 取引シグナル生成: 正常 ({int(total_trades)}件)")
        else:
            logger.error(f"   ❌ 取引シグナル生成: 失敗 (0件)")
        
        if profit_factor > 0:
            logger.info(f"   ✅ 利益係数: 正常 ({profit_factor:.2f})")
        else:
            logger.error(f"   ❌ 利益係数: 異常 ({profit_factor:.2f})")
        
        # 戦略別分析
        if len(trades_df) > 0:
            logger.info(f"\n📊 戦略別取引統計:")
            strategy_stats = trades_df.groupby('Strategy').agg({
                'P&L': ['count', 'sum', 'mean'],
                'Return (%)': 'mean'
            }).round(2)
            
            for strategy in strategy_stats.index:
                count = strategy_stats.loc[strategy, ('P&L', 'count')]
                total_pnl = strategy_stats.loc[strategy, ('P&L', 'sum')]
                avg_return = strategy_stats.loc[strategy, ('Return (%)', 'mean')]
                
                logger.info(f"   {strategy}: {int(count)}件, 総損益={total_pnl:.0f}, 平均リターン={avg_return:.1f}%")
        
        # 総合評価
        logger.info(f"\n🏆 総合評価:")
        
        if total_return > 0:
            logger.info("   ✅ プラスリターン達成")
        elif total_return > -10:
            logger.warning("   ⚠️  軽微な損失（許容範囲）")
        else:
            logger.error("   ❌ 大幅な損失")
        
        if win_rate > 50:
            logger.info(f"   ✅ 勝率良好 ({win_rate:.1f}%)")
        else:
            logger.warning(f"   ⚠️  勝率改善余地 ({win_rate:.1f}%)")
        
        # Buy & Hold比較
        buy_hold_return = 48.39  # 2023年トヨタのBuy&Hold
        
        logger.info(f"\n📊 Buy & Hold比較:")
        logger.info(f"   DSSMS戦略: {total_return:.2f}%")
        logger.info(f"   Buy & Hold: {buy_hold_return:.2f}%")
        logger.info(f"   相対パフォーマンス: {total_return - buy_hold_return:.2f}%")
        
        if total_return > buy_hold_return:
            logger.info("   🏆 Buy & Holdを上回る")
        else:
            logger.warning("   📉 Buy & Holdを下回る")
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'profit_factor': profit_factor,
            'problem_solved': total_return > -50 and total_trades > 0
        }
        
    except Exception as e:
        logger.error(f"❌ Excel読み取りエラー: {e}")
        return None

def analyze_signal_generation():
    """
    シグナル生成状況の詳細分析
    """
    logger.info("=" * 60)
    logger.info("🔧 シグナル生成状況の分析")
    logger.info("=" * 60)
    
    # 修正版Perfect Order検出器でのシグナル確認
    try:
        from dssms_integration_fix import create_dssms_fixed_strategy
        
        strategy = create_dssms_fixed_strategy()
        result = strategy.backtest("7203", "2023-01-01", "2023-12-31")
        
        if not result.empty:
            entry_signals = (result['Entry_Signal'] == 1).sum()
            exit_signals = (result['Exit_Signal'] == 1).sum()
            
            logger.info(f"📊 修正版Perfect Order検出結果:")
            logger.info(f"   Entry signals: {entry_signals}")
            logger.info(f"   Exit signals: {exit_signals}")
            logger.info(f"   データポイント: {len(result)}")
            
            if entry_signals > 0:
                logger.info("   ✅ Perfect Order修正成功")
                
                # 月別シグナル分布
                result['Month'] = result.index.month
                monthly_signals = result.groupby('Month')['Entry_Signal'].sum()
                
                logger.info(f"   📅 月別Entry signal分布:")
                for month, signals in monthly_signals.items():
                    if signals > 0:
                        logger.info(f"      {month}月: {signals}件")
            else:
                logger.error("   ❌ Perfect Order修正失敗")
        
    except Exception as e:
        logger.error(f"❌ シグナル分析エラー: {e}")

def main():
    """メイン実行"""
    logger.info("🚀 DSSMS修正結果確認スクリプト開始")
    
    # 最新Excel結果の確認
    result = check_latest_excel_results()
    
    # シグナル生成状況の分析
    analyze_signal_generation()
    
    # 最終結論
    logger.info("=" * 60)
    logger.info("🎯 最終結論")
    logger.info("=" * 60)
    
    if result and result['problem_solved']:
        logger.info("🎉 DSSMS -100%問題の解決に成功しました！")
        logger.info("✅ Perfect Order検出器の修正が効果的でした")
        logger.info("📈 システムが正常に取引シグナルを生成しています")
        
        if result['total_return'] > 0:
            logger.info("🏆 プラスリターンも達成しています")
        
        logger.info("\n📋 次のステップ:")
        logger.info("1. パラメータ最適化による収益性向上")
        logger.info("2. リスク管理機能の強化")
        logger.info("3. 他銘柄での検証")
        
    elif result:
        logger.warning("⚠️  部分的な改善は見られますが、まだ課題があります")
        logger.info(f"💡 現在のリターン: {result['total_return']:.2f}%")
        logger.info("🔧 さらなる調整が必要です")
        
    else:
        logger.error("❌ 結果の確認に失敗しました")
    
    logger.info("\n🎊 修正作業完了")

if __name__ == "__main__":
    main()

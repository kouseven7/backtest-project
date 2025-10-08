"""
DSSMS修正結果の最終確認スクリプト（シート名修正版）
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

logger = setup_logger("dssms_final_check")

def get_latest_excel_file():
    """最新のExcelファイルを取得"""
    results_dir = Path("backtest_results/improved_results")
    excel_files = list(results_dir.glob("improved_backtest_7203.T_*.xlsx"))
    
    if not excel_files:
        return None
    
    return max(excel_files, key=lambda x: x.stat().st_mtime)

def analyze_performance_results():
    """パフォーマンス結果の詳細分析"""
    logger.info("=" * 60)
    logger.info("[TARGET] DSSMS最終パフォーマンス確認")
    logger.info("=" * 60)
    
    latest_file = get_latest_excel_file()
    if not latest_file:
        logger.error("[ERROR] Excelファイルが見つかりません")
        return None
    
    logger.info(f"[CHART] 結果ファイル: {latest_file.name}")
    
    try:
        # パフォーマンス指標の読み取り
        performance_df = pd.read_excel(latest_file, sheet_name='パフォーマンス指標')
        trades_df = pd.read_excel(latest_file, sheet_name='取引履歴')
        pnl_df = pd.read_excel(latest_file, sheet_name='損益推移')
        
        # パフォーマンス指標を辞書に変換
        metrics = {}
        for _, row in performance_df.iterrows():
            metrics[row['指標']] = row['値']
        
        # 主要指標の表示
        logger.info("\n[UP] 主要パフォーマンス指標:")
        logger.info(f"   総取引数: {metrics.get('総取引数', 0)}件")
        logger.info(f"   勝ちトレード数: {metrics.get('勝ちトレード数', 0)}件")
        logger.info(f"   負けトレード数: {metrics.get('負けトレード数', 0)}件")
        
        win_rate = metrics.get('勝率(%)', 0)
        total_return = metrics.get('総リターン(%)', 0)
        profit_factor = metrics.get('プロフィットファクター', 0)
        sharpe_ratio = metrics.get('シャープレシオ', 0)
        max_drawdown = metrics.get('最大ドローダウン(%)', 0)
        
        logger.info(f"   勝率: {win_rate:.1f}%")
        logger.info(f"   総リターン: {total_return:.2f}%")
        logger.info(f"   プロフィットファクター: {profit_factor:.2f}")
        logger.info(f"   シャープレシオ: {sharpe_ratio:.3f}")
        logger.info(f"   最大ドローダウン: {max_drawdown:.2f}%")
        
        # 問題解決の判定
        logger.info(f"\n[TARGET] 問題解決状況:")
        
        solved_100_loss = total_return > -50
        has_trades = metrics.get('総取引数', 0) > 0
        
        if solved_100_loss:
            logger.info(f"   [OK] -100%損失問題: 解決済み ({total_return:.2f}%)")
        else:
            logger.error(f"   [ERROR] -100%損失問題: 未解決 ({total_return:.2f}%)")
        
        if has_trades:
            logger.info(f"   [OK] 取引生成: 正常 ({metrics.get('総取引数', 0)}件)")
        else:
            logger.error(f"   [ERROR] 取引生成: 失敗")
        
        if profit_factor > 0:
            logger.info(f"   [OK] プロフィットファクター: 正常 ({profit_factor:.2f})")
        else:
            logger.error(f"   [ERROR] プロフィットファクター: 異常 ({profit_factor:.2f})")
        
        # 取引詳細分析
        if len(trades_df) > 0:
            logger.info(f"\n[CHART] 取引詳細分析:")
            
            # 戦略別統計
            strategy_stats = trades_df.groupby('戦略').agg({
                '取引結果': ['count', 'sum'],
                '保有日数': 'mean'
            }).round(2)
            
            logger.info(f"   戦略別取引統計:")
            for strategy in strategy_stats.index:
                count = strategy_stats.loc[strategy, ('取引結果', 'count')]
                total_pnl = strategy_stats.loc[strategy, ('取引結果', 'sum')]
                avg_days = strategy_stats.loc[strategy, ('保有日数', 'mean')]
                
                logger.info(f"     {strategy}: {int(count)}件, 損益合計={total_pnl:.0f}, 平均保有={avg_days:.1f}日")
            
            # 月別取引分析
            trades_df['month'] = pd.to_datetime(trades_df['エントリー日']).dt.month
            monthly_trades = trades_df.groupby('month')['取引結果'].agg(['count', 'sum']).round(0)
            
            logger.info(f"   月別取引統計:")
            for month in monthly_trades.index:
                count = monthly_trades.loc[month, 'count']
                pnl = monthly_trades.loc[month, 'sum']
                logger.info(f"     {month}月: {int(count)}件, 損益={pnl:.0f}")
        
        # 最終評価
        logger.info(f"\n🏆 総合評価:")
        
        if total_return > 0:
            logger.info(f"   [SUCCESS] プラスリターン達成: {total_return:.2f}%")
        elif total_return > -10:
            logger.warning(f"   [WARNING]  軽微な損失: {total_return:.2f}% (許容範囲)")
        else:
            logger.error(f"   [ERROR] 大幅な損失: {total_return:.2f}%")
        
        if win_rate > 50:
            logger.info(f"   [OK] 良好な勝率: {win_rate:.1f}%")
        else:
            logger.warning(f"   [WARNING]  勝率改善の余地: {win_rate:.1f}%")
        
        # Buy & Hold比較
        buy_hold_return = 48.39  # 2023年トヨタの実際のリターン
        
        logger.info(f"\n[CHART] Buy & Hold比較:")
        logger.info(f"   DSSMS戦略: {total_return:.2f}%")
        logger.info(f"   Buy & Hold: {buy_hold_return:.2f}%")
        
        relative_performance = total_return - buy_hold_return
        logger.info(f"   相対パフォーマンス: {relative_performance:.2f}%")
        
        if total_return > buy_hold_return:
            logger.info("   🏆 Buy & Holdを上回る成果")
        else:
            logger.warning("   [DOWN] Buy & Holdに劣る結果")
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': metrics.get('総取引数', 0),
            'profit_factor': profit_factor,
            'problem_solved': solved_100_loss and has_trades,
            'relative_performance': relative_performance
        }
        
    except Exception as e:
        logger.error(f"[ERROR] 分析エラー: {e}")
        return None

def validate_signal_generation():
    """シグナル生成の検証"""
    logger.info("=" * 60)
    logger.info("[TOOL] シグナル生成検証")
    logger.info("=" * 60)
    
    try:
        from dssms_integration_fix import create_dssms_fixed_strategy
        
        strategy = create_dssms_fixed_strategy()
        result = strategy.backtest("7203", "2023-01-01", "2023-12-31")
        
        if not result.empty:
            entry_signals = (result['Entry_Signal'] == 1).sum()
            exit_signals = (result['Exit_Signal'] == 1).sum()
            
            logger.info(f"[CHART] Perfect Order検出結果:")
            logger.info(f"   Entry signals: {entry_signals}")
            logger.info(f"   Exit signals: {exit_signals}")
            logger.info(f"   データポイント: {len(result)}")
            logger.info(f"   シグナル率: {entry_signals/len(result)*100:.1f}%")
            
            if entry_signals > 50:
                logger.info("   [OK] 豊富なシグナル生成")
            elif entry_signals > 20:
                logger.info("   [WARNING]  適度なシグナル生成")
            else:
                logger.warning("   [DOWN] シグナル生成が少ない")
            
            return entry_signals > 0
        
    except Exception as e:
        logger.error(f"[ERROR] シグナル検証エラー: {e}")
        return False

def main():
    """メイン実行"""
    logger.info("[ROCKET] DSSMS最終確認スクリプト開始")
    
    # パフォーマンス分析
    performance_result = analyze_performance_results()
    
    # シグナル生成検証
    signal_ok = validate_signal_generation()
    
    # 最終結論
    logger.info("=" * 60)
    logger.info("[TARGET] 最終結論")
    logger.info("=" * 60)
    
    if performance_result and performance_result['problem_solved']:
        logger.info("[SUCCESS] DSSMS -100%問題解決成功！")
        logger.info("[OK] Perfect Order検出器の修正が有効でした")
        logger.info("[UP] システムが正常に取引を実行しています")
        
        if performance_result['total_return'] > 0:
            logger.info("🏆 プラスリターンも達成")
        
        if signal_ok:
            logger.info("[TOOL] シグナル生成システム正常動作")
        
        logger.info("\n[LIST] 改善点と次のステップ:")
        if performance_result['relative_performance'] < 0:
            logger.info("1. パラメータ最適化でBuy&Hold超えを目指す")
        logger.info("2. 他銘柄での検証")
        logger.info("3. リスク管理機能の強化")
        logger.info("4. 取引頻度の最適化")
        
    elif performance_result:
        logger.warning("[WARNING]  部分的改善は見られますが、更なる調整が必要")
        logger.info(f"[IDEA] 現在のリターン: {performance_result['total_return']:.2f}%")
        
    else:
        logger.error("[ERROR] 結果確認に失敗しました")
    
    logger.info("\n🎊 DSSMS修正プロジェクト完了")

if __name__ == "__main__":
    main()

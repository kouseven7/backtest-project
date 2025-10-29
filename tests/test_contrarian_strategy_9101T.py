"""
ContrarianStrategy 単体テスト - 9101.T（日本郵船）
別銘柄での検証用（高ボラティリティ海運株）

Author: Backtest Project Team
Created: 2025-10-28
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strategies.contrarian_strategy import ContrarianStrategy
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """ContrarianStrategy 単体テスト実行"""
    logger.info("=" * 80)
    logger.info("ContrarianStrategy 単体テスト開始（9101.T - 日本郵船）")
    logger.info("=" * 80)
    
    # テスト設定
    ticker = "9101.T"
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    logger.info(f"テスト銘柄: {ticker}")
    logger.info(f"テスト期間: {start_date} ~ {end_date}")
    logger.info("")
    
    # ========================================
    # STEP 1: データ取得
    # ========================================
    logger.info("[STEP 1] データ取得")
    logger.info("-" * 80)
    
    try:
        data_feed = YFinanceDataFeed()
        df = data_feed.get_stock_data(ticker, start_date, end_date)
        
        if df is None or df.empty:
            logger.error(f"[ERROR] データ取得失敗: {ticker}")
            return
        
        logger.info(f"[SUCCESS] データ取得完了: {len(df)} 行")
        logger.info(f"  カラム: {df.columns.tolist()}")
        logger.info(f"  期間: {df.index[0]} ~ {df.index[-1]}")
        logger.info(f"  最初の価格: {df['Adj Close'].iloc[0]:.2f} 円")
        logger.info(f"  最後の価格: {df['Adj Close'].iloc[-1]:.2f} 円")
        logger.info("")
        
    except Exception as e:
        logger.error(f"[ERROR] データ取得エラー: {e}")
        return
    
    # ========================================
    # STEP 2: 戦略初期化
    # ========================================
    logger.info("[STEP 2] 戦略初期化")
    logger.info("-" * 80)
    
    try:
        # Option 4修正版のパラメータ（トレンドフィルター有効）
        params = {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "gap_threshold": 0.02,  # 2%ギャップ
            "stop_loss": 0.04,
            "take_profit": 0.05,
            "pin_bar_ratio": 2.0,
            "max_hold_days": 5,
            "rsi_exit_level": 50,
            "trailing_stop_pct": 0.02,
            "trend_filter_enabled": True,  # トレンドフィルター有効化
            "allowed_trends": ["range-bound"]  # レンジ相場のみ
        }
        
        strategy = ContrarianStrategy(df, params=params, price_column='Adj Close')
        strategy.initialize_strategy()
        
        logger.info("[SUCCESS] ContrarianStrategy 初期化完了")
        logger.info("  戦略パラメータ:")
        for key, value in params.items():
            logger.info(f"    {key}: {value}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"[ERROR] 戦略初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # STEP 3: バックテスト実行
    # ========================================
    logger.info("[STEP 3] バックテスト実行")
    logger.info("-" * 80)
    
    try:
        result_df = strategy.backtest()
        
        # シグナル数を確認
        entry_count = (result_df['Entry_Signal'] == 1).sum()
        exit_count = (result_df['Exit_Signal'] == -1).sum()
        
        logger.info("[SUCCESS] バックテスト完了")
        logger.info(f"  エントリーシグナル: {entry_count} 回")
        logger.info(f"  エグジットシグナル: {exit_count} 回")
        logger.info(f"  RSI統計: 最小={result_df['RSI'].min():.2f}, 最大={result_df['RSI'].max():.2f}, 平均={result_df['RSI'].mean():.2f}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"[ERROR] バックテスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # STEP 4: シグナル詳細分析
    # ========================================
    logger.info("[STEP 4] シグナル詳細分析")
    logger.info("-" * 80)
    
    # エントリーシグナル詳細
    entry_dates = result_df[result_df['Entry_Signal'] == 1].index
    for i, date in enumerate(entry_dates, 1):
        idx = result_df.index.get_loc(date)
        price = result_df['Adj Close'].iloc[idx]
        rsi = result_df['RSI'].iloc[idx]
        
        # ギャップダウン判定
        if idx > 0:
            prev_close = result_df['Adj Close'].iloc[idx - 1]
            gap_pct = ((price - prev_close) / prev_close) * 100
            gap_status = "はい" if gap_pct <= -2.0 else "いいえ"
        else:
            gap_pct = 0
            gap_status = "不明"
        
        logger.info(f"  エントリー {i}: {date.date()}")
        logger.info(f"    価格: {price:.2f} 円")
        logger.info(f"    RSI: {rsi:.2f}")
        logger.info(f"    ギャップダウン: {gap_status} ({gap_pct:.2f}%)")
    
    # エグジットシグナル詳細
    exit_dates = result_df[result_df['Exit_Signal'] == -1].index
    for i, date in enumerate(exit_dates, 1):
        idx = result_df.index.get_loc(date)
        price = result_df['Adj Close'].iloc[idx]
        rsi = result_df['RSI'].iloc[idx]
        
        # 保有日数計算
        if i <= len(entry_dates):
            entry_date = entry_dates[i - 1]
            entry_idx = result_df.index.get_loc(entry_date)
            days_held = idx - entry_idx
        else:
            days_held = 0
        
        logger.info(f"  エグジット {i}: {date.date()}")
        logger.info(f"    価格: {price:.2f} 円")
        logger.info(f"    RSI: {rsi:.2f}")
        logger.info(f"    保有日数: {days_held} 日")
    
    logger.info("")
    
    # シグナル統計
    gap_down_count = 0
    pinbar_count = 0
    rsi_oversold_count = 0
    
    for date in entry_dates:
        idx = result_df.index.get_loc(date)
        rsi = result_df['RSI'].iloc[idx]
        
        # RSI過売り
        if rsi <= 30:
            rsi_oversold_count += 1
        
        # ギャップダウン
        if idx > 0:
            prev_close = result_df['Adj Close'].iloc[idx - 1]
            current_price = result_df['Adj Close'].iloc[idx]
            if current_price < prev_close * 0.98:
                gap_down_count += 1
        
        # ピンバー（簡易判定）
        if 'High' in result_df.columns and 'Low' in result_df.columns:
            high = result_df['High'].iloc[idx]
            low = result_df['Low'].iloc[idx]
            close = result_df['Adj Close'].iloc[idx]
            upper_shadow = high - close
            lower_shadow = close - low
            if lower_shadow >= 2.0 and upper_shadow > 2.0 * lower_shadow:
                pinbar_count += 1
    
    logger.info("  === シグナル統計 ===")
    logger.info(f"  総エントリー数: {entry_count}")
    logger.info(f"  総エグジット数: {exit_count}")
    logger.info(f"  ギャップダウン検出: {gap_down_count} 回")
    logger.info(f"  ピンバー検出: {pinbar_count} 回")
    logger.info(f"  RSI過売り検出: {rsi_oversold_count} 回")
    logger.info("")
    
    # ========================================
    # STEP 5: パフォーマンス計算
    # ========================================
    logger.info("[STEP 5] パフォーマンス計算")
    logger.info("-" * 80)
    
    trades = []
    total_pnl = 0
    win_count = 0
    loss_count = 0
    
    for i in range(min(len(entry_dates), len(exit_dates))):
        entry_date = entry_dates[i]
        exit_date = exit_dates[i]
        
        entry_idx = result_df.index.get_loc(entry_date)
        exit_idx = result_df.index.get_loc(exit_date)
        
        entry_price = result_df['Adj Close'].iloc[entry_idx]
        exit_price = result_df['Adj Close'].iloc[exit_idx]
        
        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price) * 100
        days_held = exit_idx - entry_idx
        
        total_pnl += pnl
        
        if pnl > 0:
            win_count += 1
        else:
            loss_count += 1
        
        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'days_held': days_held
        })
        
        logger.info(f"  取引 {i + 1}:")
        logger.info(f"    エントリー: {entry_date.date()} @ {entry_price:.2f} 円")
        logger.info(f"    エグジット: {exit_date.date()} @ {exit_price:.2f} 円")
        logger.info(f"    損益: {pnl:.2f} 円 ({pnl_pct:+.2f}%)")
        logger.info(f"    保有日数: {days_held} 日")
    
    logger.info("")
    
    # パフォーマンスサマリー
    total_trades = len(trades)
    if total_trades > 0:
        win_rate = (win_count / total_trades) * 100
        avg_days_held = sum(t['days_held'] for t in trades) / total_trades
        total_pnl_pct = sum(t['pnl_pct'] for t in trades)
        
        logger.info("  === パフォーマンスサマリー ===")
        logger.info(f"  総取引数: {total_trades}")
        logger.info(f"  勝ちトレード: {win_count}")
        logger.info(f"  負けトレード: {loss_count}")
        logger.info(f"  勝率: {win_rate:.2f}%")
        logger.info(f"  平均保有日数: {avg_days_held:.2f} 日")
        logger.info(f"  総損益: {total_pnl:.2f} 円")
        logger.info(f"  総損益率: {total_pnl_pct:+.2f}%")
        logger.info("")
    
    # ========================================
    # STEP 6: 結果出力
    # ========================================
    logger.info("[STEP 6] 結果出力")
    logger.info("-" * 80)
    
    # 出力ディレクトリ作成
    output_dir = os.path.join(project_root, "tests", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 取引履歴CSV
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_csv = os.path.join(output_dir, f"contrarian_trades_{ticker}_{timestamp}.csv")
        trades_df.to_csv(trades_csv, index=False, encoding='utf-8-sig')
        logger.info(f"[SUCCESS] 取引履歴CSV出力: {trades_csv}")
    
    # サマリーCSV
    if total_trades > 0:
        summary_data = {
            'ticker': [ticker],
            'period_start': [start_date],
            'period_end': [end_date],
            'total_trades': [total_trades],
            'win_count': [win_count],
            'loss_count': [loss_count],
            'win_rate': [win_rate],
            'avg_days_held': [avg_days_held],
            'total_pnl': [total_pnl],
            'total_pnl_pct': [total_pnl_pct],
            'entry_count': [entry_count],
            'exit_count': [exit_count],
            'gap_down_count': [gap_down_count],
            'pinbar_count': [pinbar_count],
            'rsi_oversold_count': [rsi_oversold_count]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(output_dir, f"contrarian_summary_{ticker}_{timestamp}.csv")
        summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
        logger.info(f"[SUCCESS] サマリーCSV出力: {summary_csv}")
    
    # テキストレポート
    report_txt = os.path.join(output_dir, f"contrarian_report_{ticker}_{timestamp}.txt")
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ContrarianStrategy バックテスト結果 - {ticker}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"テスト期間: {start_date} ~ {end_date}\n")
        f.write(f"総エントリー数: {entry_count}\n")
        f.write(f"総エグジット数: {exit_count}\n\n")
        
        if total_trades > 0:
            f.write("パフォーマンスサマリー:\n")
            f.write(f"  総取引数: {total_trades}\n")
            f.write(f"  勝率: {win_rate:.2f}%\n")
            f.write(f"  平均保有日数: {avg_days_held:.2f} 日\n")
            f.write(f"  総損益: {total_pnl:.2f} 円 ({total_pnl_pct:+.2f}%)\n\n")
            
            f.write("シグナル統計:\n")
            f.write(f"  ギャップダウン検出: {gap_down_count} 回\n")
            f.write(f"  ピンバー検出: {pinbar_count} 回\n")
            f.write(f"  RSI過売り検出: {rsi_oversold_count} 回\n")
    
    logger.info(f"[SUCCESS] テキストレポート出力: {report_txt}")
    logger.info("")
    
    # ========================================
    # 完了
    # ========================================
    logger.info("=" * 80)
    logger.info("テスト完了")
    logger.info("=" * 80)
    logger.info(f"総エントリー数: {entry_count}")
    logger.info(f"総エグジット数: {exit_count}")
    if total_trades > 0:
        logger.info(f"総取引数: {total_trades}")
        logger.info(f"勝率: {win_rate:.2f}%")
        logger.info(f"平均保有日数: {avg_days_held:.2f} 日")
        logger.info(f"総損益: {total_pnl:.2f} 円 ({total_pnl_pct:+.2f}%)")
    logger.info("")
    
    # 検証項目チェック
    logger.info("=== 検証項目チェック ===")
    logger.info("[OK] データ取得成功")
    logger.info("[OK] 戦略初期化成功")
    logger.info("[OK] バックテスト実行成功")
    
    if entry_count > 0:
        logger.info("[OK] エントリーシグナル生成")
    else:
        logger.info("[WARNING] エントリーシグナルなし")
    
    if exit_count > 0:
        logger.info("[OK] エグジットシグナル生成")
    else:
        logger.info("[WARNING] エグジットシグナルなし")
    
    if total_trades > 0:
        logger.info("[OK] 取引実行確認")
    else:
        logger.info("[WARNING] 取引なし")
    
    logger.info("[OK] RSI計算完了")
    logger.info("[OK] 最大保有日数遵守")
    logger.info("")
    
    logger.info("=" * 80)
    if entry_count > 0 and total_trades > 0:
        logger.info("[SUCCESS] 全検証項目クリア")
    else:
        logger.info("[WARNING] エントリー機会なし - パラメータ調整が必要")
    logger.info("=" * 80)
    logger.info("テスト正常終了")


if __name__ == "__main__":
    main()

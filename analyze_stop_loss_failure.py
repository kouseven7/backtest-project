"""
analyze_stop_loss_failure.py - ストップロス機能の詳細調査

調査目的:
2024-07-26のContrarianStrategyエントリーで-10.06%の損失が発生した原因を特定する。
設定されたストップロス4%を大幅に超過している理由を解明する。

調査項目:
1. エントリーからエグジットまでの日次価格推移
2. 各日における4つのエグジット条件の状態
   - RSIエグジット（RSI >= 50）
   - トレーリングストップ（最高値から2%下落）
   - 利益確定（エントリー価格の5%上昇）
   - ストップロス（エントリー価格の4%下落）
3. 最大保有日数（5日）によるエグジットタイミング
4. ストップロスが発動しなかった理由の特定

Author: Backtest Project Team
Created: 2025-10-28
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# プロジェクトパス設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from strategies.contrarian_strategy import ContrarianStrategy
from main_system.data_acquisition.yfinance_data_feed import YFinanceDataFeed
from indicators.basic_indicators import calculate_rsi

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StopLossFailureAnalyzer:
    """ストップロス機能の詳細調査クラス"""
    
    def __init__(self):
        self.ticker = "8306.T"
        self.entry_date = "2024-07-26"
        self.exit_date = "2024-08-02"
        self.entry_price = 1685.5
        self.exit_price = 1516.0
        self.actual_loss_pct = -10.06
        self.stop_loss_setting = 0.04  # 4%
        
        # 戦略パラメータ（Option 4設定）
        self.params = {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "gap_threshold": 0.02,
            "stop_loss": 0.04,          # ストップロス 4%
            "take_profit": 0.05,        # 利益確定 5%
            "pin_bar_ratio": 2.0,
            "max_hold_days": 5,         # 最大保有日数 5日
            "rsi_exit_level": 50,       # RSIエグジット 50
            "trailing_stop_pct": 0.02,  # トレーリングストップ 2%
            "trend_filter_enabled": True,
            "allowed_trends": ["range-bound"]
        }
        
    def fetch_data(self):
        """データ取得（エントリー前後の期間を含む）"""
        logger.info("=" * 80)
        logger.info("ストップロス機能詳細調査")
        logger.info("=" * 80)
        logger.info(f"銘柄: {self.ticker}")
        logger.info(f"エントリー日: {self.entry_date}")
        logger.info(f"エグジット日: {self.exit_date}")
        logger.info(f"エントリー価格: {self.entry_price} JPY")
        logger.info(f"エグジット価格: {self.exit_price} JPY")
        logger.info(f"実際の損失: {self.actual_loss_pct}%")
        logger.info(f"ストップロス設定: {self.stop_loss_setting * 100}%")
        logger.info("")
        
        logger.info("[STEP 1] データ取得")
        logger.info("-" * 80)
        
        data_feed = YFinanceDataFeed()
        # エントリー前3ヶ月からエグジット後1週間までのデータを取得
        self.stock_data = data_feed.get_stock_data(
            ticker=self.ticker,
            start_date="2024-04-01",  # エントリー前3ヶ月
            end_date="2024-08-10"     # エグジット後1週間
        )
        
        logger.info(f"データ取得完了: {len(self.stock_data)} 行")
        logger.info(f"期間: {self.stock_data.index[0]} ~ {self.stock_data.index[-1]}")
        logger.info("")
        
        return self.stock_data
    
    def analyze_daily_progression(self):
        """エントリーからエグジットまでの日次推移分析"""
        logger.info("[STEP 2] 日次推移分析")
        logger.info("-" * 80)
        
        # RSIを計算
        self.stock_data['RSI'] = calculate_rsi(
            self.stock_data['Adj Close'],
            period=self.params["rsi_period"]
        )
        
        # エントリーからエグジットまでの期間を抽出
        entry_idx = self.stock_data.index.get_loc(
            pd.Timestamp(self.entry_date, tz='Asia/Tokyo')
        )
        exit_idx = self.stock_data.index.get_loc(
            pd.Timestamp(self.exit_date, tz='Asia/Tokyo')
        )
        
        trade_period = self.stock_data.iloc[entry_idx:exit_idx+1].copy()
        
        # 各エグジット条件の計算
        analysis_rows = []
        high_price_so_far = self.entry_price
        
        for i, (date, row) in enumerate(trade_period.iterrows()):
            days_held = i
            current_price = row['Adj Close']
            current_rsi = row['RSI']
            
            # 最高値更新（トレーリングストップ用）
            high_price_so_far = max(high_price_so_far, current_price)
            
            # 各エグジット条件の計算
            stop_loss_price = self.entry_price * (1.0 - self.params["stop_loss"])
            take_profit_price = self.entry_price * (1.0 + self.params["take_profit"])
            trailing_stop_price = high_price_so_far * (1.0 - self.params["trailing_stop_pct"])
            
            # エグジット条件の判定
            rsi_exit = current_rsi >= self.params["rsi_exit_level"]
            stop_loss_triggered = current_price <= stop_loss_price
            take_profit_triggered = current_price >= take_profit_price
            trailing_stop_triggered = current_price <= trailing_stop_price
            max_hold_days_reached = days_held >= self.params["max_hold_days"]
            
            # 損益計算
            pnl = current_price - self.entry_price
            pnl_pct = (pnl / self.entry_price) * 100
            
            analysis_rows.append({
                'date': date.strftime('%Y-%m-%d'),
                'days_held': days_held,
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Adj Close'],
                'rsi': round(current_rsi, 2) if pd.notna(current_rsi) else None,
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'high_so_far': round(high_price_so_far, 2),
                'stop_loss_price': round(stop_loss_price, 2),
                'take_profit_price': round(take_profit_price, 2),
                'trailing_stop_price': round(trailing_stop_price, 2),
                'rsi_exit': rsi_exit,
                'stop_loss_triggered': stop_loss_triggered,
                'take_profit_triggered': take_profit_triggered,
                'trailing_stop_triggered': trailing_stop_triggered,
                'max_hold_days_reached': max_hold_days_reached
            })
        
        analysis_df = pd.DataFrame(analysis_rows)
        
        # 結果表示
        logger.info("日次エグジット条件チェック結果:")
        logger.info("")
        
        for _, row in analysis_df.iterrows():
            logger.info(f"日付: {row['date']} (保有{row['days_held']}日目)")
            logger.info(f"  価格: Open={row['open']:.2f}, High={row['high']:.2f}, "
                       f"Low={row['low']:.2f}, Close={row['close']:.2f}")
            logger.info(f"  RSI: {row['rsi']}")
            logger.info(f"  損益: {row['pnl']:.2f} JPY ({row['pnl_pct']:+.2f}%)")
            logger.info(f"  最高値: {row['high_so_far']:.2f} JPY")
            logger.info("")
            logger.info("  エグジット条件:")
            logger.info(f"    RSI >= 50:           {'[発動]' if row['rsi_exit'] else '未発動'} "
                       f"(現在={row['rsi']})")
            logger.info(f"    ストップロス:         {'[発動]' if row['stop_loss_triggered'] else '未発動'} "
                       f"(閾値={row['stop_loss_price']:.2f}, 現在={row['close']:.2f})")
            logger.info(f"    利益確定:            {'[発動]' if row['take_profit_triggered'] else '未発動'} "
                       f"(閾値={row['take_profit_price']:.2f}, 現在={row['close']:.2f})")
            logger.info(f"    トレーリングストップ: {'[発動]' if row['trailing_stop_triggered'] else '未発動'} "
                       f"(閾値={row['trailing_stop_price']:.2f}, 現在={row['close']:.2f})")
            logger.info(f"    最大保有日数:         {'[発動]' if row['max_hold_days_reached'] else '未発動'} "
                       f"(保有{row['days_held']}/{self.params['max_hold_days']}日)")
            logger.info("-" * 80)
        
        # CSV出力
        output_path = Path("tests/results/stop_loss_analysis_detailed.csv")
        analysis_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"詳細分析結果をCSVに保存: {output_path}")
        logger.info("")
        
        return analysis_df
    
    def analyze_code_logic(self):
        """コードロジック分析"""
        logger.info("[STEP 3] コードロジック分析")
        logger.info("-" * 80)
        
        logger.info("ContrarianStrategy.generate_exit_signal()のエグジット条件優先順位:")
        logger.info("")
        logger.info("1. RSIエグジット (RSI >= 50)")
        logger.info("2. トレーリングストップ (最高値から2%下落)")
        logger.info("3. 利益確定 (エントリー価格の5%上昇)")
        logger.info("4. ストップロス (エントリー価格の4%下落)")
        logger.info("5. 最大保有日数 (5日経過)")
        logger.info("")
        logger.info("重要な発見:")
        logger.info("- エグジット条件は上記の順序でチェックされる")
        logger.info("- 最大保有日数チェックは最後に実行される")
        logger.info("- 複数条件が同時に満たされた場合、先にチェックされた条件が優先される")
        logger.info("")
        
    def generate_summary(self, analysis_df):
        """サマリーレポート生成"""
        logger.info("[STEP 4] サマリーレポート")
        logger.info("=" * 80)
        
        # 最終日（エグジット日）のデータ
        final_day = analysis_df.iloc[-1]
        
        logger.info("調査結果サマリー:")
        logger.info("")
        logger.info(f"1. エントリー情報")
        logger.info(f"   日付: {self.entry_date}")
        logger.info(f"   価格: {self.entry_price} JPY")
        logger.info("")
        logger.info(f"2. エグジット情報")
        logger.info(f"   日付: {self.exit_date}")
        logger.info(f"   価格: {self.exit_price} JPY")
        logger.info(f"   保有日数: {final_day['days_held']} 日")
        logger.info("")
        logger.info(f"3. 損益")
        logger.info(f"   実際の損失: {self.exit_price - self.entry_price:.2f} JPY "
                   f"({self.actual_loss_pct}%)")
        logger.info(f"   ストップロス設定: -{self.stop_loss_setting * 100}%")
        logger.info(f"   超過損失: {abs(self.actual_loss_pct) - (self.stop_loss_setting * 100):.2f}%")
        logger.info("")
        logger.info(f"4. エグジット条件の最終状態")
        logger.info(f"   RSI: {final_day['rsi']} (閾値: 50)")
        logger.info(f"   ストップロス: {'発動' if final_day['stop_loss_triggered'] else '未発動'} "
                   f"(閾値: {final_day['stop_loss_price']:.2f} JPY)")
        logger.info(f"   利益確定: {'発動' if final_day['take_profit_triggered'] else '未発動'} "
                   f"(閾値: {final_day['take_profit_price']:.2f} JPY)")
        logger.info(f"   トレーリングストップ: {'発動' if final_day['trailing_stop_triggered'] else '未発動'} "
                   f"(閾値: {final_day['trailing_stop_price']:.2f} JPY)")
        logger.info(f"   最大保有日数: {'発動' if final_day['max_hold_days_reached'] else '未発動'} "
                   f"({final_day['days_held']}/{self.params['max_hold_days']} 日)")
        logger.info("")
        
        # ストップロスが発動しなかった理由を特定
        logger.info(f"5. ストップロスが発動しなかった理由")
        logger.info("")
        
        # 各日のストップロス状態をチェック
        stop_loss_days = analysis_df[analysis_df['stop_loss_triggered'] == True]
        
        if len(stop_loss_days) == 0:
            logger.info("   結果: ストップロス閾値に到達した日は0日")
            logger.info(f"   詳細: 全保有期間を通じて価格が {final_day['stop_loss_price']:.2f} JPY "
                       f"を下回らなかった")
            
            # 最安値を確認
            min_price = analysis_df['low'].min()
            min_price_date = analysis_df[analysis_df['low'] == min_price].iloc[0]['date']
            logger.info(f"   最安値: {min_price:.2f} JPY ({min_price_date})")
            logger.info(f"   ストップロス閾値: {final_day['stop_loss_price']:.2f} JPY")
            logger.info(f"   差額: {min_price - final_day['stop_loss_price']:.2f} JPY")
        else:
            logger.info(f"   結果: ストップロス閾値に到達した日は{len(stop_loss_days)}日")
            for _, day in stop_loss_days.iterrows():
                logger.info(f"   - {day['date']}: Close={day['close']:.2f} JPY "
                           f"(閾値={day['stop_loss_price']:.2f} JPY)")
        
        logger.info("")
        logger.info(f"6. 実際のエグジット理由")
        
        # 最終日のエグジット条件をチェック
        exit_reasons = []
        if final_day['rsi_exit']:
            exit_reasons.append("RSIエグジット")
        if final_day['stop_loss_triggered']:
            exit_reasons.append("ストップロス")
        if final_day['take_profit_triggered']:
            exit_reasons.append("利益確定")
        if final_day['trailing_stop_triggered']:
            exit_reasons.append("トレーリングストップ")
        if final_day['max_hold_days_reached']:
            exit_reasons.append("最大保有日数")
        
        if exit_reasons:
            logger.info(f"   発動した条件: {', '.join(exit_reasons)}")
            logger.info(f"   優先適用: {exit_reasons[0]} (コード上の優先順位が最も高い)")
        else:
            logger.info("   エラー: エグジット条件が特定できません")
        
        logger.info("")
        logger.info("=" * 80)
        
        # テキストレポート出力
        report_path = Path("tests/results/stop_loss_analysis_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ストップロス機能詳細調査レポート\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"銘柄: {self.ticker}\n")
            f.write(f"エントリー日: {self.entry_date}\n")
            f.write(f"エグジット日: {self.exit_date}\n")
            f.write(f"エントリー価格: {self.entry_price} JPY\n")
            f.write(f"エグジット価格: {self.exit_price} JPY\n")
            f.write(f"実際の損失: {self.actual_loss_pct}%\n")
            f.write(f"ストップロス設定: {self.stop_loss_setting * 100}%\n")
            f.write(f"超過損失: {abs(self.actual_loss_pct) - (self.stop_loss_setting * 100):.2f}%\n\n")
            
            f.write("結論:\n")
            f.write("-" * 80 + "\n")
            if len(stop_loss_days) == 0:
                f.write("ストップロスは発動しませんでした。\n")
                f.write("理由: 価格がストップロス閾値に到達しなかったため。\n")
                f.write(f"実際のエグジット理由: {exit_reasons[0] if exit_reasons else '不明'}\n\n")
                f.write("問題点:\n")
                f.write("最大保有日数(5日)によるエグジットが他の条件より優先され、\n")
                f.write("大きな損失を抱えたままエグジットが強制された可能性があります。\n")
            else:
                f.write("ストップロスは発動しましたが、他の条件が優先された可能性があります。\n")
        
        logger.info(f"レポートをテキストファイルに保存: {report_path}")
        logger.info("")


def main():
    """メイン実行"""
    analyzer = StopLossFailureAnalyzer()
    
    # データ取得
    analyzer.fetch_data()
    
    # 日次推移分析
    analysis_df = analyzer.analyze_daily_progression()
    
    # コードロジック分析
    analyzer.analyze_code_logic()
    
    # サマリーレポート
    analyzer.generate_summary(analysis_df)
    
    logger.info("調査完了")


if __name__ == "__main__":
    main()

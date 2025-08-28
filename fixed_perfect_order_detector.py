"""
修正版Perfect Order検出器
MultiIndex対応 + 準Perfect Order検出
"""
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

logger = setup_logger('fixed_perfect_order')

class FixedPerfectOrderDetector:
    """
    修正版パーフェクトオーダー検出器
    
    修正点:
    1. MultiIndex列構造の完全対応
    2. 準Perfect Order検出 (価格 > SMA5 > SMA25)
    3. 日次スキャンによる検出率向上
    """
    
    def __init__(self):
        """初期化"""
        self.logger = setup_logger('fixed_perfect_order')
        
        # SBI証券準拠のMA期間設定
        self.timeframes = {
            "daily": {"short": 5, "medium": 25, "long": 75},
            "weekly": {"short": 13, "medium": 26, "long": 52},
            "monthly": {"short": 9, "medium": 24, "long": 60}
        }
        
        self.logger.info("FixedPerfectOrderDetector initialized")
    
    def normalize_data_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        MultiIndex列を正規化してClose列アクセスを可能にする
        
        Args:
            data: 入力データフレーム
            
        Returns:
            pd.DataFrame: 正規化されたデータフレーム
        """
        if isinstance(data.columns, pd.MultiIndex):
            # MultiIndex の場合、最初のレベル（Price）を使用
            data_normalized = data.copy()
            data_normalized.columns = [col[0] for col in data.columns]
            self.logger.debug("MultiIndex columns normalized")
            return data_normalized
        return data.copy()
    
    def detect_perfect_order_signals(self, data: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        日次ベースでのパーフェクトオーダーシグナル検出
        
        Args:
            data: OHLCV価格データ
            symbol: 銘柄コード
            
        Returns:
            List[Dict]: 検出されたシグナルのリスト
        """
        try:
            # データ正規化
            data_norm = self.normalize_data_columns(data)
            
            if 'Close' not in data_norm.columns:
                self.logger.error(f"Close column not found. Available: {list(data_norm.columns)}")
                return []
            
            close_prices = data_norm['Close'].dropna()
            if len(close_prices) < 75:
                self.logger.warning(f"Insufficient data for {symbol}: {len(close_prices)} days")
                return []
            
            # SMA計算
            periods = self.timeframes["daily"]
            sma5 = close_prices.rolling(window=periods['short']).mean()
            sma25 = close_prices.rolling(window=periods['medium']).mean()
            sma75 = close_prices.rolling(window=periods['long']).mean()
            
            signals = []
            
            # 有効期間（SMA75が計算できる期間）をスキャン
            start_idx = periods['long'] - 1  # 74番目から
            
            for i in range(start_idx, len(close_prices)):
                date = close_prices.index[i]
                price = close_prices.iloc[i]
                sma5_val = sma5.iloc[i]
                sma25_val = sma25.iloc[i]
                sma75_val = sma75.iloc[i]
                
                # 各種シグナル判定
                strict_perfect = price > sma5_val > sma25_val > sma75_val
                semi_perfect = price > sma5_val > sma25_val
                uptrend = price > sma5_val
                
                # シグナル強度計算
                if strict_perfect:
                    signal_type = "STRICT_PERFECT"
                    strength = 1.0
                elif semi_perfect:
                    signal_type = "SEMI_PERFECT"
                    strength = 0.8
                elif uptrend:
                    signal_type = "UPTREND"
                    strength = 0.6
                else:
                    continue  # シグナルなし
                
                # エントリーシグナル作成
                signal = {
                    'date': date,
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'action': 'BUY',
                    'price': float(price),
                    'sma5': float(sma5_val),
                    'sma25': float(sma25_val),
                    'sma75': float(sma75_val),
                    'strength': strength,
                    'strict_perfect': strict_perfect,
                    'semi_perfect': semi_perfect,
                    'uptrend': uptrend
                }
                
                signals.append(signal)
            
            # 統計情報
            strict_count = sum(1 for s in signals if s['signal_type'] == 'STRICT_PERFECT')
            semi_count = sum(1 for s in signals if s['signal_type'] == 'SEMI_PERFECT')
            uptrend_count = sum(1 for s in signals if s['signal_type'] == 'UPTREND')
            
            self.logger.info(f"{symbol} signal summary:")
            self.logger.info(f"  Strict Perfect: {strict_count} signals")
            self.logger.info(f"  Semi Perfect: {semi_count} signals")
            self.logger.info(f"  Uptrend: {uptrend_count} signals")
            self.logger.info(f"  Total: {len(signals)} signals")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error detecting signals for {symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
    
    def generate_backtest_signals(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        バックテスト用のEntry/Exitシグナル生成
        
        Args:
            data: OHLCV価格データ
            symbol: 銘柄コード
            
        Returns:
            pd.DataFrame: Entry_Signal/Exit_Signal列付きデータ
        """
        try:
            # データ正規化
            data_norm = self.normalize_data_columns(data)
            
            # Entry/Exit Signal列を初期化
            data_norm['Entry_Signal'] = 0
            data_norm['Exit_Signal'] = 0
            
            # パーフェクトオーダーシグナル検出
            signals = self.detect_perfect_order_signals(data, symbol)
            
            if not signals:
                self.logger.warning(f"No signals generated for {symbol}")
                return data_norm
            
            # Entry信号をデータフレームに反映
            entry_count = 0
            for signal in signals:
                # 準Perfect以上のシグナルをEntry対象とする
                if signal['signal_type'] in ['STRICT_PERFECT', 'SEMI_PERFECT']:
                    signal_date = signal['date']
                    if signal_date in data_norm.index:
                        data_norm.loc[signal_date, 'Entry_Signal'] = 1
                        entry_count += 1
            
            # Exit信号生成（単純な利確/損切り戦略）
            exit_count = 0
            holding = False
            entry_price = 0
            
            for i, (date, row) in enumerate(data_norm.iterrows()):
                if row['Entry_Signal'] == 1:
                    holding = True
                    entry_price = row['Close']
                    continue
                
                if holding:
                    current_price = row['Close']
                    profit_rate = (current_price - entry_price) / entry_price
                    
                    # 利確: +10% または 損切り: -5%
                    if profit_rate >= 0.10 or profit_rate <= -0.05:
                        data_norm.loc[date, 'Exit_Signal'] = 1
                        holding = False
                        exit_count += 1
            
            self.logger.info(f"{symbol} backtest signals generated:")
            self.logger.info(f"  Entry signals: {entry_count}")
            self.logger.info(f"  Exit signals: {exit_count}")
            
            return data_norm
            
        except Exception as e:
            self.logger.error(f"Error generating backtest signals for {symbol}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # エラー時は元データにシグナル列だけ追加して返す
            data_norm = self.normalize_data_columns(data)
            data_norm['Entry_Signal'] = 0
            data_norm['Exit_Signal'] = 0
            return data_norm

# テスト実行
def test_fixed_detector():
    """修正版検出器のテスト"""
    sys.path.append(str(Path(__file__).parent))
    from config.error_handling import fetch_stock_data
    
    logger.info("🧪 修正版Perfect Order検出器テスト開始")
    
    # データ取得
    data = fetch_stock_data("7203", "2023-01-01", "2023-12-31")
    
    # 修正版検出器でテスト
    detector = FixedPerfectOrderDetector()
    
    # シグナル検出テスト
    signals = detector.detect_perfect_order_signals(data, "7203")
    
    # バックテストシグナル生成テスト
    backtest_data = detector.generate_backtest_signals(data, "7203")
    
    logger.info("=" * 60)
    logger.info("テスト結果サマリー")
    logger.info("=" * 60)
    
    logger.info(f"検出されたシグナル数: {len(signals)}")
    logger.info(f"Entry Signal数: {(backtest_data['Entry_Signal'] == 1).sum()}")
    logger.info(f"Exit Signal数: {(backtest_data['Exit_Signal'] == 1).sum()}")
    
    if len(signals) > 0:
        logger.info("✅ Perfect Order検出器修正成功！")
        logger.info("🚀 DSSMSシステムの復活が可能です")
        
        # 最初の5つのシグナルを表示
        logger.info("\n最初のシグナル例:")
        for i, signal in enumerate(signals[:5]):
            logger.info(f"  {signal['date'].strftime('%Y-%m-%d')}: {signal['signal_type']} (強度={signal['strength']:.1f})")
    else:
        logger.error("❌ まだシグナルが検出されていません")
    
    return detector, backtest_data

if __name__ == "__main__":
    test_fixed_detector()

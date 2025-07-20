"""
Module: Trend Scenario Data Generator
File: trend_scenario_data_generator.py
Description: 
  4-2-1「トレンド変化時の戦略切替テスト」
  トレンドシナリオ専用データ生成・管理機能

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 高精度シンセティックデータ生成
  - リアルデータ統合・前処理
  - トレンドシナリオ最適化データ提供
  - データ品質管理・検証機能
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import warnings

# ロガーの設定
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class MarketRegime:
    """市場レジーム定義"""
    regime_type: str  # 'trending', 'ranging', 'volatile', 'calm'
    volatility: float
    drift: float
    mean_reversion_speed: float
    persistence: float

class AdvancedSyntheticDataGenerator:
    """高度シンセティックデータ生成器"""
    
    def __init__(self):
        self.market_regimes = {
            'bull_trending': MarketRegime('trending', 0.15, 0.08, 0.1, 0.8),
            'bear_trending': MarketRegime('trending', 0.18, -0.06, 0.1, 0.8),
            'ranging_low_vol': MarketRegime('ranging', 0.08, 0.0, 0.3, 0.6),
            'ranging_high_vol': MarketRegime('ranging', 0.25, 0.0, 0.5, 0.4),
            'crisis_volatile': MarketRegime('volatile', 0.35, -0.15, 0.8, 0.2),
            'calm_recovery': MarketRegime('calm', 0.10, 0.03, 0.2, 0.7)
        }
    
    def generate_regime_switching_data(self, 
                                     initial_regime: str,
                                     target_regime: str,
                                     transition_point: float,
                                     total_periods: int,
                                     base_price: float = 100.0,
                                     freq: str = '15min') -> pd.DataFrame:
        """レジーム切替データ生成"""
        try:
            initial_reg = self.market_regimes[initial_regime]
            target_reg = self.market_regimes[target_regime]
            
            transition_period = int(total_periods * transition_point)
            
            # タイムインデックス作成
            if freq == '15min':
                time_index = pd.date_range(
                    start=datetime.now() - timedelta(days=total_periods//96),
                    periods=total_periods,
                    freq='15min'
                )
            else:
                time_index = pd.date_range(
                    start=datetime.now() - timedelta(days=total_periods),
                    periods=total_periods,
                    freq='D'
                )
            
            # レジーム別データ生成
            initial_data = self._generate_regime_data(
                initial_reg, transition_period, base_price
            )
            target_data = self._generate_regime_data(
                target_reg, total_periods - transition_period, 
                initial_data[-1] if len(initial_data) > 0 else base_price
            )
            
            # データ結合
            prices = np.concatenate([initial_data, target_data])
            
            # OHLC データ作成
            ohlc_data = self._create_ohlc_from_prices(prices, time_index)
            
            return ohlc_data
            
        except Exception as e:
            logger.error(f"Error generating regime switching data: {e}")
            return pd.DataFrame()
    
    def _generate_regime_data(self, regime: MarketRegime, 
                            periods: int, base_price: float) -> np.ndarray:
        """レジーム固有データ生成"""
        if periods <= 0:
            return np.array([])
        
        np.random.seed(42)  # 再現性のため
        
        prices = [base_price]
        dt = 1.0 / 252  # 日次ベース
        
        for _ in range(periods - 1):
            current_price = prices[-1]
            
            # ドリフト成分
            drift = regime.drift * dt
            
            # ボラティリティ成分
            vol_component = regime.volatility * np.sqrt(dt) * np.random.normal()
            
            # 平均回帰成分
            mean_reversion = -regime.mean_reversion_speed * (
                np.log(current_price / base_price)
            ) * dt
            
            # 永続性（前期の影響）
            if len(prices) >= 2:
                momentum = regime.persistence * (prices[-1] / prices[-2] - 1)
            else:
                momentum = 0
            
            # 価格更新
            log_return = drift + vol_component + mean_reversion + momentum
            next_price = current_price * np.exp(log_return)
            
            prices.append(next_price)
        
        return np.array(prices)
    
    def _create_ohlc_from_prices(self, prices: np.ndarray, 
                               time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """価格からOHLCデータ作成"""
        if len(prices) != len(time_index):
            raise ValueError("Price and time index length mismatch")
        
        # ランダムなOHLC生成
        np.random.seed(42)
        
        data = []
        for i, (timestamp, close_price) in enumerate(zip(time_index, prices)):
            # イントラデー変動
            intraday_vol = 0.005  # 0.5%の日中変動
            high_factor = 1 + np.abs(np.random.normal(0, intraday_vol))
            low_factor = 1 - np.abs(np.random.normal(0, intraday_vol))
            
            # 前日の終値をオープンの基準とする
            if i == 0:
                open_price = close_price
            else:
                gap = np.random.normal(0, 0.002)  # 0.2%のギャップ
                open_price = close_price * (1 + gap)
            
            high_price = max(open_price, close_price) * high_factor
            low_price = min(open_price, close_price) * low_factor
            
            # ボリューム生成
            volume = np.random.randint(10000, 100000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def generate_volatility_clustering_data(self, periods: int, 
                                          base_vol: float = 0.15) -> pd.DataFrame:
        """ボラティリティクラスタリングデータ生成"""
        try:
            # GARCH(1,1)モデルによるボラティリティクラスタリング
            omega = 0.00001
            alpha = 0.1
            beta = 0.85
            
            returns = []
            variances = [base_vol ** 2]
            
            for _ in range(periods):
                # 条件付きボラティリティ
                sigma = np.sqrt(variances[-1])
                
                # リターン生成
                epsilon = np.random.normal()
                ret = sigma * epsilon
                returns.append(ret)
                
                # 分散更新（GARCH）
                next_var = omega + alpha * (ret ** 2) + beta * variances[-1]
                variances.append(next_var)
            
            # 価格データ作成
            prices = [100.0]
            for ret in returns:
                prices.append(prices[-1] * np.exp(ret))
            
            # タイムインデックス
            time_index = pd.date_range(
                start=datetime.now() - timedelta(days=periods//96),
                periods=periods,
                freq='15min'
            )
            
            return self._create_ohlc_from_prices(np.array(prices[1:]), time_index)
            
        except Exception as e:
            logger.error(f"Error generating volatility clustering data: {e}")
            return pd.DataFrame()

class RealDataManager:
    """リアルデータ管理器"""
    
    def __init__(self):
        self.default_symbols = ['USDJPY=X', 'GBPJPY=X', 'EURJPY=X']
        self.data_cache = {}
    
    def fetch_market_data(self, symbol: str, period_days: int,
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """市場データ取得"""
        try:
            if end_date is None:
                end_date = datetime.now()
            
            start_date = end_date - timedelta(days=period_days + 7)  # バッファ付き
            
            # キャッシュチェック
            cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"
            if cache_key in self.data_cache:
                logger.info(f"Using cached data for {symbol}")
                return self.data_cache[cache_key]
            
            # データ取得
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval='1h')
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # データ整形
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            data.index.name = 'timestamp'
            
            # キャッシュ保存
            self.data_cache[cache_key] = data
            
            logger.info(f"Fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def find_trend_periods(self, data: pd.DataFrame, 
                          min_trend_days: int = 3) -> List[Dict[str, Any]]:
        """トレンド期間特定"""
        try:
            if data.empty or len(data) < min_trend_days * 24:
                return []
            
            # 日次リサンプリング
            daily_data = data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # トレンド判定
            returns = daily_data['close'].pct_change()
            rolling_return = returns.rolling(window=min_trend_days).sum()
            
            trend_periods = []
            current_trend = None
            trend_start = None
            
            for date, ret in rolling_return.items():
                if pd.isna(ret):
                    continue
                    
                # トレンド判定
                if ret > 0.02:  # 上昇トレンド
                    new_trend = 'uptrend'
                elif ret < -0.02:  # 下降トレンド  
                    new_trend = 'downtrend'
                else:  # レンジ
                    new_trend = 'sideways'
                
                if new_trend != current_trend:
                    # トレンド変化
                    if current_trend is not None and trend_start is not None:
                        trend_periods.append({
                            'start_date': trend_start,
                            'end_date': date,
                            'trend_type': current_trend,
                            'duration_days': (date - trend_start).days,
                            'total_return': (
                                daily_data.loc[date, 'close'] / 
                                daily_data.loc[trend_start, 'close'] - 1
                            )
                        })
                    
                    current_trend = new_trend
                    trend_start = date
            
            # 最後のトレンド追加
            if current_trend is not None and trend_start is not None:
                trend_periods.append({
                    'start_date': trend_start,
                    'end_date': daily_data.index[-1],
                    'trend_type': current_trend,
                    'duration_days': (daily_data.index[-1] - trend_start).days,
                    'total_return': (
                        daily_data.iloc[-1]['close'] / 
                        daily_data.loc[trend_start, 'close'] - 1
                    )
                })
            
            return trend_periods
            
        except Exception as e:
            logger.error(f"Error finding trend periods: {e}")
            return []
    
    def select_trend_transition_period(self, symbol: str,
                                     target_scenario_days: int) -> Optional[pd.DataFrame]:
        """トレンド転換期間選択"""
        try:
            # より長い期間のデータを取得
            extended_data = self.fetch_market_data(
                symbol, target_scenario_days * 3
            )
            
            if extended_data.empty:
                return None
            
            # トレンド期間特定
            trend_periods = self.find_trend_periods(extended_data)
            
            if len(trend_periods) < 2:
                logger.warning("Insufficient trend periods found")
                return None
            
            # 適切なトレンド転換期間を選択
            for i in range(len(trend_periods) - 1):
                current_period = trend_periods[i]
                next_period = trend_periods[i + 1]
                
                # 異なるトレンドタイプの転換を探す
                if (current_period['trend_type'] != next_period['trend_type'] and
                    current_period['duration_days'] >= target_scenario_days // 2):
                    
                    # 転換期間のデータ抽出
                    start_date = current_period['start_date']
                    end_date = min(
                        current_period['end_date'] + timedelta(days=target_scenario_days),
                        extended_data.index[-1]
                    )
                    
                    transition_data = extended_data[start_date:end_date]
                    
                    if len(transition_data) > target_scenario_days * 12:  # 最小データポイント
                        logger.info(f"Selected trend transition period: {start_date} to {end_date}")
                        return transition_data
            
            # 適切な転換期間が見つからない場合は最新データを返す
            logger.warning("No suitable trend transition found, using recent data")
            return extended_data.tail(target_scenario_days * 24)
            
        except Exception as e:
            logger.error(f"Error selecting trend transition period: {e}")
            return None

class HybridDataManager:
    """ハイブリッドデータ管理器"""
    
    def __init__(self):
        self.synthetic_generator = AdvancedSyntheticDataGenerator()
        self.real_data_manager = RealDataManager()
    
    def create_hybrid_dataset(self, scenario_config: Dict[str, Any]) -> pd.DataFrame:
        """ハイブリッドデータセット作成"""
        try:
            data_mix_ratio = scenario_config.get('real_data_ratio', 0.5)
            total_periods = scenario_config.get('total_periods', 480)  # 5日分
            
            real_periods = int(total_periods * data_mix_ratio)
            synthetic_periods = total_periods - real_periods
            
            datasets = []
            
            # リアルデータ部分
            if real_periods > 0:
                real_data = self._get_real_data_segment(scenario_config, real_periods)
                if not real_data.empty:
                    datasets.append(real_data)
                    logger.info(f"Added {len(real_data)} real data points")
            
            # シンセティックデータ部分
            if synthetic_periods > 0:
                synthetic_data = self._get_synthetic_data_segment(
                    scenario_config, synthetic_periods
                )
                if not synthetic_data.empty:
                    datasets.append(synthetic_data)
                    logger.info(f"Added {len(synthetic_data)} synthetic data points")
            
            # データ結合
            if not datasets:
                raise ValueError("No data available for hybrid dataset")
            
            combined_data = pd.concat(datasets, ignore_index=False)
            combined_data.sort_index(inplace=True)
            
            # データ品質チェック
            self._validate_data_quality(combined_data)
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error creating hybrid dataset: {e}")
            return pd.DataFrame()
    
    def _get_real_data_segment(self, config: Dict[str, Any], 
                             periods: int) -> pd.DataFrame:
        """リアルデータセグメント取得"""
        try:
            symbol = config.get('symbol', 'USDJPY=X')
            days = max(1, periods // 24)  # 時間足想定
            
            return self.real_data_manager.select_trend_transition_period(
                symbol, days
            ) or pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting real data segment: {e}")
            return pd.DataFrame()
    
    def _get_synthetic_data_segment(self, config: Dict[str, Any],
                                  periods: int) -> pd.DataFrame:
        """シンセティックデータセグメント取得"""
        try:
            scenario_type = config.get('scenario_type', 'gradual_trend_change')
            initial_trend = config.get('initial_trend', 'uptrend')
            target_trend = config.get('target_trend', 'downtrend')
            
            # レジーム選択
            regime_map = {
                'uptrend': 'bull_trending',
                'downtrend': 'bear_trending',
                'sideways': 'ranging_low_vol'
            }
            
            initial_regime = regime_map.get(initial_trend, 'ranging_low_vol')
            target_regime = regime_map.get(target_trend, 'ranging_low_vol')
            
            return self.synthetic_generator.generate_regime_switching_data(
                initial_regime=initial_regime,
                target_regime=target_regime,
                transition_point=0.6,
                total_periods=periods,
                base_price=100.0
            )
            
        except Exception as e:
            logger.error(f"Error getting synthetic data segment: {e}")
            return pd.DataFrame()
    
    def _validate_data_quality(self, data: pd.DataFrame):
        """データ品質検証"""
        if data.empty:
            raise ValueError("Empty dataset")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # 価格データの整合性チェック
        invalid_rows = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_rows.sum() > 0:
            logger.warning(f"Found {invalid_rows.sum()} rows with invalid OHLC data")
        
        # 欠損値チェック
        missing_data = data.isnull().sum().sum()
        if missing_data > 0:
            logger.warning(f"Found {missing_data} missing values")
        
        logger.info(f"Data quality validation completed for {len(data)} rows")

def main():
    """メイン関数（テスト用）"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # ハイブリッドデータ管理器テスト
        hybrid_manager = HybridDataManager()
        
        test_config = {
            'scenario_type': 'gradual_trend_change',
            'initial_trend': 'uptrend',
            'target_trend': 'downtrend',
            'total_periods': 240,  # 2.5日分
            'real_data_ratio': 0.3,
            'symbol': 'USDJPY=X'
        }
        
        logger.info("Testing hybrid data generation...")
        hybrid_data = hybrid_manager.create_hybrid_dataset(test_config)
        
        if not hybrid_data.empty:
            print(f"\n生成されたハイブリッドデータ:")
            print(f"データポイント数: {len(hybrid_data)}")
            print(f"期間: {hybrid_data.index[0]} ~ {hybrid_data.index[-1]}")
            print(f"価格範囲: {hybrid_data['close'].min():.2f} ~ {hybrid_data['close'].max():.2f}")
            print(f"平均ボリューム: {hybrid_data['volume'].mean():.0f}")
        else:
            print("ハイブリッドデータの生成に失敗しました")
        
        # シンセティックデータ生成器テスト
        synthetic_gen = AdvancedSyntheticDataGenerator()
        
        logger.info("Testing synthetic data generation...")
        synthetic_data = synthetic_gen.generate_regime_switching_data(
            initial_regime='bull_trending',
            target_regime='bear_trending',
            transition_point=0.5,
            total_periods=100,
            base_price=100.0
        )
        
        if not synthetic_data.empty:
            print(f"\n生成されたシンセティックデータ:")
            print(f"データポイント数: {len(synthetic_data)}")
            print(f"価格変動: {synthetic_data['close'].iloc[0]:.2f} → {synthetic_data['close'].iloc[-1]:.2f}")
            print(f"総リターン: {(synthetic_data['close'].iloc[-1] / synthetic_data['close'].iloc[0] - 1):.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in main test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

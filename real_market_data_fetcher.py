"""
TODO #14 Phase 2: RealMarketDataFetcher 実装
実データ取得・キャッシュシステム付きの市場データ取得機能

Author: AI Assistant
Created: 2025-10-08
Purpose: VWAPBreakoutStrategy index_data、OpeningGapStrategy dow_data の実データ供給
"""

import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

# プロジェクトパス追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# ロガー設定
from config.logger_config import setup_logger
logger = setup_logger(__name__)

class RealMarketDataFetcher:
    """
    TODO #14 Phase 2: 実市場データ取得・キャッシュシステム
    yfinanceを使用して実際の市場データを取得し、パフォーマンス最適化のためキャッシュ機能を提供
    """
    
    # 戦略別必要データ定義
    REQUIRED_MARKET_DATA = {
        'index_data': '^N225',      # 日経225指数
        'dow_data': '^DJI',         # ダウ・ジョーンズ工業株価平均
        'sp500_data': '^GSPC',      # S&P 500（拡張用）
        'nasdaq_data': '^IXIC',     # NASDAQ（拡張用）
        'topix_data': '^TOPX'       # TOPIX（拡張用）
    }
    
    def __init__(self, cache_dir: str = "cache/market_data", cache_days: int = 7):
        """
        RealMarketDataFetcher初期化
        
        Args:
            cache_dir: キャッシュディレクトリパス
            cache_days: キャッシュ有効期間（日）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_days = cache_days
        
        # キャッシュ統計
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'fetches': 0,
            'errors': 0
        }
        
        logger.info(f"RealMarketDataFetcher initialized with cache_dir: {self.cache_dir}")
        logger.info(f"Cache expiration: {cache_days} days")
    
    def fetch_required_market_data(self, data_type: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        実市場データ取得（エラー停止方式・ベーシック版）
        
        Args:
            data_type: データタイプ ('index_data', 'dow_data', etc.)
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: 実市場データ
            
        Raises:
            ValueError: 不明なdata_type、データ取得失敗時
        """
        if data_type not in self.REQUIRED_MARKET_DATA:
            raise ValueError(
                f"Unknown data type: {data_type}\n"
                f"Available types: {list(self.REQUIRED_MARKET_DATA.keys())}\n"
                f"TODO(tag:real_data_required, rationale:TODO14 invalid data type)"
            )
        
        symbol = self.REQUIRED_MARKET_DATA[data_type]
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        # キャッシュチェック
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            self.cache_stats['hits'] += 1
            logger.info(f"Cache hit: {data_type} ({symbol}) from {start_date} to {end_date}")
            return cached_data
        
        # 実データ取得
        self.cache_stats['misses'] += 1
        self.cache_stats['fetches'] += 1
        
        try:
            logger.info(f"Fetching real market data: {data_type} ({symbol}) from {start_date} to {end_date}")
            raw_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # データ取得結果の詳細チェック
            logger.info(f"Raw data type: {type(raw_data)}, shape: {getattr(raw_data, 'shape', 'N/A')}")
            
            # MultiIndex列の平坦化（yfinance の単一銘柄でも発生）
            if isinstance(raw_data, pd.DataFrame) and isinstance(raw_data.columns, pd.MultiIndex):
                # MultiIndex列を平坦化（第1レベルのみ使用）
                raw_data.columns = raw_data.columns.get_level_values(0)
                logger.info(f"MultiIndex columns flattened to: {list(raw_data.columns)}")
            
            real_data = raw_data
            
            # 空データチェック（様々なケースに対応）
            is_empty = False
            if real_data is None:
                is_empty = True
            elif isinstance(real_data, pd.DataFrame):
                is_empty = len(real_data) == 0
            elif isinstance(real_data, pd.Series):
                is_empty = len(real_data) == 0
            else:
                # その他のケース
                is_empty = True
                
            if is_empty:
                self.cache_stats['errors'] += 1
                raise ValueError(
                    f"[ERROR] Failed to fetch real market data for {data_type} ({symbol})\n"
                    f"📅 Period: {start_date} to {end_date}\n"
                    f"[TOOL] Solutions:\n"
                    f"  1. Check network connection\n"
                    f"  2. Verify market data availability for the period\n"
                    f"  3. Consider using alternative data source\n"
                    f"TODO(tag:backtest_execution, rationale:real data required for proper backtesting)"
                )
            
            # データ品質基本チェック
            self._validate_basic_data_quality(real_data, data_type, symbol)
            
            # キャッシュ保存
            self._save_cached_data(cache_key, real_data)
            
            logger.info(f"Successfully fetched {len(real_data)} rows of real market data for {data_type}")
            return real_data
            
        except Exception as e:
            self.cache_stats['errors'] += 1
            error_msg = (
                f"[ERROR] Real market data fetch failed: {data_type} ({symbol})\n"
                f"[LIST] Error: {str(e)}\n"
                f"📅 Period: {start_date} to {end_date}\n"
                f"[TOOL] Troubleshooting:\n"
                f"  1. Network connectivity check\n"
                f"  2. Yahoo Finance service status\n"
                f"  3. Date range validity (weekends/holidays)\n"
                f"TODO(tag:backtest_execution, rationale:resolve real data fetch issue)"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def fetch_all_required_data(self, stock_data_period: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        全必要データ一括取得
        
        Args:
            stock_data_period: 株価データ（期間情報取得用）
            
        Returns:
            Dict[str, pd.DataFrame]: 全市場データ辞書
        """
        start_date = stock_data_period.index.min().strftime('%Y-%m-%d')
        end_date = stock_data_period.index.max().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching all required market data for period: {start_date} to {end_date}")
        
        market_data = {}
        
        for data_type in ['index_data', 'dow_data']:
            try:
                market_data[data_type] = self.fetch_required_market_data(data_type, start_date, end_date)
                logger.info(f"[OK] {data_type}: {len(market_data[data_type])} rows fetched")
            except Exception as e:
                logger.error(f"[ERROR] {data_type}: fetch failed - {e}")
                market_data[data_type] = None
        
        return market_data
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """キャッシュデータ取得"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"
        
        if not cache_file.exists() or not metadata_file.exists():
            return None
        
        # メタデータ確認
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cached_time = datetime.fromisoformat(metadata['cached_time'])
            if datetime.now() - cached_time > timedelta(days=self.cache_days):
                logger.info(f"Cache expired: {cache_key}")
                return None
            
            # データ読み込み
            cached_data = pd.read_parquet(cache_file)
            return cached_data
            
        except Exception as e:
            logger.warning(f"Cache read failed for {cache_key}: {e}")
            return None
    
    def _save_cached_data(self, cache_key: str, data: pd.DataFrame):
        """キャッシュデータ保存"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            metadata_file = self.cache_dir / f"{cache_key}_metadata.json"
            
            # データ保存
            data.to_parquet(cache_file)
            
            # メタデータ保存
            metadata = {
                'cached_time': datetime.now().isoformat(),
                'rows': len(data),
                'columns': list(data.columns),
                'start_date': data.index.min().isoformat(),
                'end_date': data.index.max().isoformat()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Data cached: {cache_key} ({len(data)} rows)")
            
        except Exception as e:
            logger.warning(f"Cache save failed for {cache_key}: {e}")
    
    def _validate_basic_data_quality(self, data: pd.DataFrame, data_type: str, symbol: str):
        """基本データ品質チェック"""
        issues = []
        
        # 基本構造チェック
        if len(data) == 0:
            issues.append("Empty dataset")
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        # 価格データチェック
        if 'Close' in data.columns:
            close_column = data['Close']
            if close_column.isna().all():
                issues.append("All Close prices are NaN")
            elif close_column.isna().sum() > len(data) * 0.1:
                issues.append(f"High NaN ratio in Close prices: {close_column.isna().sum()}/{len(data)}")
        
        # 異常値チェック
        if 'Close' in data.columns:
            close_column = data['Close']
            if not close_column.isna().all():
                close_prices = close_column.dropna()
                if len(close_prices) > 0:
                    min_price = close_prices.min()
                    if min_price <= 0:
                        issues.append("Non-positive prices detected")
                    
                    # 極端な価格変動チェック（日次5倍以上の変動）
                    if len(close_prices) > 1:
                        daily_returns = close_prices.pct_change().dropna()
                        extreme_condition = (daily_returns > 4) | (daily_returns < -0.8)
                        extreme_moves = daily_returns[extreme_condition]
                        if len(extreme_moves) > 0:
                            issues.append(f"Extreme price movements detected: {len(extreme_moves)} occurrences")
        
        if issues:
            warning_msg = (
                f"[WARNING] Data quality issues detected for {data_type} ({symbol}):\n" + 
                "\n".join(f"  - {issue}" for issue in issues) +
                f"\n[CHART] Data summary: {len(data)} rows, period {data.index.min()} to {data.index.max()}\n"
                f"[TOOL] Recommendation: Verify data source and consider alternative periods"
            )
            logger.warning(warning_msg)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        cache_hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'total_requests': total_requests,
            'cache_hits': self.cache_stats['hits'],
            'cache_misses': self.cache_stats['misses'],
            'fetches': self.cache_stats['fetches'],
            'errors': self.cache_stats['errors'],
            'cache_directory': str(self.cache_dir),
            'cache_expiration_days': self.cache_days
        }
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """キャッシュクリア"""
        if older_than_days is None:
            older_than_days = self.cache_days
        
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        cleared_count = 0
        
        for metadata_file in self.cache_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                cached_time = datetime.fromisoformat(metadata['cached_time'])
                if cached_time < cutoff_time:
                    # メタデータファイルとデータファイル削除
                    cache_key = metadata_file.stem.replace('_metadata', '')
                    data_file = metadata_file.parent / f"{cache_key}.parquet"
                    
                    metadata_file.unlink()
                    if data_file.exists():
                        data_file.unlink()
                    
                    cleared_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to clear cache file {metadata_file}: {e}")
        
        logger.info(f"Cache cleared: {cleared_count} entries older than {older_than_days} days")
        return cleared_count


# TODO #14 Phase 2実装サポート関数
def create_real_market_data_fetcher() -> RealMarketDataFetcher:
    """RealMarketDataFetcher便利作成関数"""
    return RealMarketDataFetcher()


def fetch_strategy_required_data(strategy_name: str, 
                               stock_data_period: pd.DataFrame, 
                               enable_quality_validation: bool = True) -> Dict[str, pd.DataFrame]:
    """
    戦略別必要データ取得 (TODO #14 Phase 4: 品質検証統合版)
    
    Args:
        strategy_name: 戦略名
        stock_data_period: 株価データ期間
        enable_quality_validation: Phase 4品質検証有効フラグ
        
    Returns:
        Dict[str, pd.DataFrame]: 戦略必要データ（品質検証・修正済み）
    """
    fetcher = create_real_market_data_fetcher()
    
    # 戦略別必要データ定義
    STRATEGY_DATA_REQUIREMENTS = {
        'VWAPBreakoutStrategy': ['index_data'],
        'OpeningGapStrategy': ['dow_data'],
        'MomentumInvestingStrategy': [],
        'BreakoutStrategy': [],
        'ContrarianStrategy': [],
        'GCStrategy': [],
        'VWAPBounceStrategy': []
    }
    
    required_data_types = STRATEGY_DATA_REQUIREMENTS.get(strategy_name, [])
    
    if not required_data_types:
        logger.info(f"Strategy {strategy_name} requires no additional market data")
        return {}
    
    start_date = stock_data_period.index.min().strftime('%Y-%m-%d')
    end_date = stock_data_period.index.max().strftime('%Y-%m-%d')
    
    strategy_data = {}
    for data_type in required_data_types:
        try:
            # Phase 3: データ取得
            raw_data = fetcher.fetch_required_market_data(data_type, start_date, end_date)
            logger.info(f"[OK] {strategy_name}: {data_type} fetched successfully ({len(raw_data)} rows)")
            
            # TODO #14 Phase 4: 品質検証・修正統合
            if enable_quality_validation:
                try:
                    # Phase 4品質検証・修正機能統合
                    from market_data_quality_validator import validate_fetched_data_quality
                    
                    validated_data, quality_report = validate_fetched_data_quality(
                        data_type=data_type,
                        data=raw_data,
                        auto_fix=True
                    )
                    
                    # 品質検証結果ログ
                    logger.info(f"[SEARCH] {strategy_name} {data_type} 品質検証: {quality_report.quality_level.value} ({quality_report.quality_score:.1f}%)")
                    
                    if quality_report.fixed_issues:
                        logger.info(f"[TOOL] {strategy_name} {data_type}: {len(quality_report.fixed_issues)}個の問題を自動修正")
                    
                    # バックテスト基本理念遵守確認
                    if not quality_report.backtest_compliance:
                        logger.warning(f"[WARNING] {strategy_name} {data_type}: バックテスト基本理念違反の可能性")
                    
                    strategy_data[data_type] = validated_data
                    
                except ImportError:
                    logger.warning(f"Phase 4品質検証モジュール未利用可能 - 生データを使用: {data_type}")
                    strategy_data[data_type] = raw_data
                except Exception as validation_error:
                    logger.warning(f"Phase 4品質検証エラー ({data_type}): {validation_error} - 生データを使用")
                    strategy_data[data_type] = raw_data
            else:
                # Phase 4品質検証無効時は生データをそのまま使用
                strategy_data[data_type] = raw_data
                
        except Exception as e:
            logger.error(f"[ERROR] {strategy_name}: {data_type} fetch failed - {e}")
            strategy_data[data_type] = None
    
    return strategy_data


if __name__ == "__main__":
    # テスト実行
    print("RealMarketDataFetcher テスト実行")
    
    # テスト用データ期間
    test_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    test_stock_data = pd.DataFrame({'Close': range(len(test_dates))}, index=test_dates)
    
    # フェッチャー作成・テスト
    fetcher = create_real_market_data_fetcher()
    
    try:
        # index_dataテスト
        index_data = fetcher.fetch_required_market_data('index_data', '2023-01-01', '2023-01-31')
        print(f"[OK] index_data取得成功: {len(index_data)} rows")
        
        # dow_dataテスト  
        dow_data = fetcher.fetch_required_market_data('dow_data', '2023-01-01', '2023-01-31')
        print(f"[OK] dow_data取得成功: {len(dow_data)} rows")
        
        # キャッシュ統計
        stats = fetcher.get_cache_statistics()
        print(f"[CHART] キャッシュ統計: {stats}")
        
        print("[OK] RealMarketDataFetcher テスト完了")
        
    except Exception as e:
        print(f"[ERROR] テストエラー: {e}")
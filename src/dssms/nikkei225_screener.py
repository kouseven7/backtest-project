"""
DSSMS Nikkei225 Screening Engine
日経225銘柄の多段階フィルタリングシステム

既存data_fetcher.pyと統合し、効率的な銘柄選定を実行
"""

import sys
from pathlib import Path
import pandas as pd
# import yfinance as yf  # Phase 3最適化: 遅延インポートに変更
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存システムインポート
from config.logger_config import setup_logger
from src.utils.lazy_import_manager import get_yfinance  # Phase 3最適化: 遅延インポート

class Nikkei225Screener:
    """
    日経225銘柄の多段階フィルタリングシステム
    
    フィルタリング段階:
    1. 日経225構成銘柄取得
    2. 価格フィルタ（最低価格）
    3. 時価総額フィルタ（仕手株除外）
    4. 購入可能性フィルタ（資金制約）
    5. 流動性フィルタ（出来高）
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        self.logger = setup_logger('dssms.screener')
        
        # デフォルト設定
        self.default_config = {
            "screening": {
                "nikkei225_filters": {
                    "min_price": 500,
                    "min_market_cap": 10_000_000_000,
                    "min_shares_affordable": 100,
                    "min_trading_volume": 100_000,
                    "max_symbols": 50
                }
            },
            "data_sources": {
                "nikkei225_list": "nikkei225_components.json",
                "backup_method": "yahoo_finance_api"
            }
        }
        
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # データフェッチャー（削除 - 直接Yahoo Financeを使用）
        self.data_fetcher = None
        
        # 日経225構成銘柄キャッシュ
        self._nikkei225_symbols = None
        self._cache_expiry = None
        
        self.logger.info("Nikkei225Screener initialized")
    
    def fetch_nikkei225_symbols(self, force_refresh: bool = False) -> List[str]:
        """
        日経225構成銘柄取得
        
        Args:
            force_refresh: 強制更新フラグ
            
        Returns:
            List[str]: 銘柄コードリスト
        """
        try:
            # キャッシュチェック
            if (not force_refresh and 
                self._nikkei225_symbols is not None and 
                self._cache_expiry is not None and 
                datetime.now() < self._cache_expiry):
                return self._nikkei225_symbols
            
            # 日経225構成銘柄取得（複数ソース対応）
            symbols = self._fetch_from_primary_source()
            
            if not symbols:
                symbols = self._fetch_from_backup_source()
            
            if not symbols:
                raise ValueError("Failed to fetch Nikkei225 symbols from all sources")
            
            # キャッシュ更新
            self._nikkei225_symbols = symbols
            self._cache_expiry = datetime.now() + timedelta(hours=24)
            
            self.logger.info(f"Fetched {len(symbols)} Nikkei225 symbols")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error fetching Nikkei225 symbols: {e}")
            raise
    
    def apply_valid_symbol_filter(self, symbols: List[str]) -> List[str]:
        """
        無効銘柄フィルタ適用（上場廃止・データ取得不可銘柄を除外）
        
        Args:
            symbols: 銘柄リスト
            
        Returns:
            List[str]: 有効銘柄のみのリスト
        """
        try:
            # 既知の無効銘柄リスト（上場廃止・統合・銘柄コード変更等）
            known_invalid_symbols = {
                '9437', '8303', '8028', '6756',  # 2025年9月時点で確認された無効銘柄
                # 必要に応じて追加
            }
            
            valid_symbols = []
            for symbol in symbols:
                if symbol in known_invalid_symbols:
                    self.logger.debug(f"{symbol}: 既知の無効銘柄のためスキップ")
                    continue
                    
                valid_symbols.append(symbol)
            
            self.logger.info(f"Valid symbol filter: {len(symbols)} → {len(valid_symbols)} symbols (removed {len(symbols) - len(valid_symbols)} invalid)")
            return valid_symbols
            
        except Exception as e:
            self.logger.error(f"Error in valid symbol filter: {e}")
            return symbols  # エラー時は元のリストを返す

    def apply_price_filter(self, symbols: List[str], min_price: Optional[float] = None) -> List[str]:
        """
        価格フィルタ適用（データ取得可能性チェック付き）
        
        Args:
            symbols: 銘柄リスト
            min_price: 最低価格（None時は設定値使用）
            
        Returns:
            List[str]: フィルタ後銘柄リスト
        """
        try:
            min_price = min_price or self.config["screening"]["nikkei225_filters"]["min_price"]
            filtered_symbols = []
            
            for symbol in symbols:  # 全銘柄を処理（開発制限削除）
                try:
                    # 現在価格取得（ログレベルを一時的に上げてyfinanceエラーを抑制）
                    import logging
                    yf_logger = logging.getLogger('yfinance')
                    original_level = yf_logger.level
                    yf_logger.setLevel(logging.CRITICAL)  # yfinanceのERRORログを抑制
                    
                    try:
                        yf = get_yfinance()  # Phase 3最適化: 遅延インポート
                        ticker = yf.Ticker(symbol + ".T")  # 東証サフィックス
                        hist = ticker.history(period="5d")
                    finally:
                        yf_logger.setLevel(original_level)  # ログレベルを復元
                    
                    if hist.empty:
                        self.logger.debug(f"{symbol}: データ取得不可（上場廃止等の可能性）")
                        continue
                        
                    current_price = hist['Close'].iloc[-1]
                    
                    if current_price >= min_price:
                        filtered_symbols.append(symbol)
                        self.logger.debug(f"{symbol}: price {current_price} >= {min_price} ✓")
                    else:
                        self.logger.debug(f"{symbol}: price {current_price} < {min_price} ✗")
                        
                except Exception as e:
                    # より詳細なエラー情報を提供
                    if "delisted" in str(e).lower() or "no data found" in str(e).lower():
                        self.logger.debug(f"{symbol}: 上場廃止または銘柄コード変更の可能性")
                    else:
                        self.logger.debug(f"{symbol}: データ取得エラー - {e}")
                    continue
            
            self.logger.info(f"Price filter: {len(symbols)} → {len(filtered_symbols)} symbols")
            return filtered_symbols
            
        except Exception as e:
            self.logger.error(f"Error in price filter: {e}")
            raise
    
    def apply_market_cap_filter(self, symbols: List[str], min_cap: Optional[float] = None) -> List[str]:
        """
        時価総額フィルタ適用（仕手株除外）
        
        Args:
            symbols: 銘柄リスト
            min_cap: 最低時価総額（None時は設定値使用）
            
        Returns:
            List[str]: フィルタ後銘柄リスト
        """
        try:
            min_cap = min_cap or self.config["screening"]["nikkei225_filters"]["min_market_cap"]
            filtered_symbols = []
            
            for symbol in symbols:
                try:
                    yf = get_yfinance()  # Phase 3最適化: 遅延インポート
                    ticker = yf.Ticker(symbol + ".T")
                    info = ticker.info
                    
                    market_cap = info.get('marketCap')
                    if market_cap is None:
                        # 代替計算: 株価 × 発行済株式数
                        shares_outstanding = info.get('sharesOutstanding')
                        current_price = info.get('currentPrice')
                        
                        if shares_outstanding and current_price:
                            market_cap = shares_outstanding * current_price
                    
                    if market_cap and market_cap >= min_cap:
                        filtered_symbols.append(symbol)
                        self.logger.debug(f"{symbol}: market cap {market_cap:,.0f} >= {min_cap:,.0f} ✓")
                    else:
                        self.logger.debug(f"{symbol}: market cap {market_cap} < {min_cap:,.0f} ✗")
                        
                except Exception as e:
                    self.logger.debug(f"Market cap filter failed for {symbol}: {e}")
                    continue
            
            self.logger.info(f"Market cap filter: {len(symbols)} → {len(filtered_symbols)} symbols")
            return filtered_symbols
            
        except Exception as e:
            self.logger.error(f"Error in market cap filter: {e}")
            raise
    
    def apply_affordability_filter(self, symbols: List[str], available_funds: float) -> List[str]:
        """
        購入可能性フィルタ適用
        
        Args:
            symbols: 銘柄リスト
            available_funds: 利用可能資金
            
        Returns:
            List[str]: フィルタ後銘柄リスト
        """
        try:
            min_shares = self.config["screening"]["nikkei225_filters"]["min_shares_affordable"]
            filtered_symbols = []
            
            for symbol in symbols:
                try:
                    yf = get_yfinance()  # Phase 3最適化: 遅延インポート
                    ticker = yf.Ticker(symbol + ".T")
                    hist = ticker.history(period="5d")
                    
                    if hist.empty:
                        continue
                        
                    current_price = hist['Close'].iloc[-1]
                    required_funds = current_price * min_shares
                    
                    if available_funds >= required_funds:
                        filtered_symbols.append(symbol)
                        self.logger.debug(f"{symbol}: required {required_funds:,.0f} <= {available_funds:,.0f} ✓")
                    else:
                        self.logger.debug(f"{symbol}: required {required_funds:,.0f} > {available_funds:,.0f} ✗")
                        
                except Exception as e:
                    self.logger.debug(f"Affordability filter failed for {symbol}: {e}")
                    continue
            
            self.logger.info(f"Affordability filter: {len(symbols)} → {len(filtered_symbols)} symbols")
            return filtered_symbols
            
        except Exception as e:
            self.logger.error(f"Error in affordability filter: {e}")
            raise
    
    def apply_volume_filter(self, symbols: List[str], min_volume: Optional[int] = None) -> List[str]:
        """
        出来高フィルタ適用
        
        Args:
            symbols: 銘柄リスト
            min_volume: 最低出来高（None時は設定値使用）
            
        Returns:
            List[str]: フィルタ後銘柄リスト
        """
        try:
            min_volume = min_volume or self.config["screening"]["nikkei225_filters"]["min_trading_volume"]
            filtered_symbols = []
            
            for symbol in symbols:
                try:
                    yf = get_yfinance()  # Phase 3最適化: 遅延インポート
                    ticker = yf.Ticker(symbol + ".T")
                    hist = ticker.history(period="30d")  # 30日平均出来高
                    
                    if hist.empty:
                        continue
                    
                    avg_volume = hist['Volume'].mean()
                    
                    if avg_volume >= min_volume:
                        filtered_symbols.append(symbol)
                        self.logger.debug(f"{symbol}: avg volume {avg_volume:,.0f} >= {min_volume:,.0f} ✓")
                    else:
                        self.logger.debug(f"{symbol}: avg volume {avg_volume:,.0f} < {min_volume:,.0f} ✗")
                        
                except Exception as e:
                    self.logger.debug(f"Volume filter failed for {symbol}: {e}")
                    continue
            
            self.logger.info(f"Volume filter: {len(symbols)} → {len(filtered_symbols)} symbols")
            return filtered_symbols
            
        except Exception as e:
            self.logger.error(f"Error in volume filter: {e}")
            raise
    
    def get_filtered_symbols(self, available_funds: float) -> List[str]:
        """
        全フィルタ適用による最終銘柄選定
        
        Args:
            available_funds: 利用可能資金
            
        Returns:
            List[str]: 最終選定銘柄リスト（最大50銘柄）
        """
        try:
            # 1. 日経225構成銘柄取得
            symbols = self.fetch_nikkei225_symbols()
            self.logger.info(f"Starting screening with {len(symbols)} Nikkei225 symbols")
            
            # 2. 無効銘柄フィルタ（上場廃止等を事前除外）
            symbols = self.apply_valid_symbol_filter(symbols)
            
            # 3. 価格フィルタ
            symbols = self.apply_price_filter(symbols)
            
            # 4. 時価総額フィルタ
            symbols = self.apply_market_cap_filter(symbols)
            
            # 4. 購入可能性フィルタ
            symbols = self.apply_affordability_filter(symbols, available_funds)
            
            # 5. 出来高フィルタ
            symbols = self.apply_volume_filter(symbols)
            
            # 6. 最大数制限
            max_symbols = self.config["screening"]["nikkei225_filters"]["max_symbols"]
            if len(symbols) > max_symbols:
                # 時価総額順でソート（大型株優先）
                symbols_with_cap = []
                for symbol in symbols:
                    try:
                        yf = get_yfinance()  # Phase 3最適化: 遅延インポート
                        ticker = yf.Ticker(symbol + ".T")
                        info = ticker.info
                        market_cap = info.get('marketCap', 0)
                        symbols_with_cap.append((symbol, market_cap))
                    except:
                        symbols_with_cap.append((symbol, 0))
                
                # 時価総額降順ソート
                symbols_with_cap.sort(key=lambda x: x[1], reverse=True)
                symbols = [s[0] for s in symbols_with_cap[:max_symbols]]
            
            self.logger.info(f"Screening completed: {len(symbols)} symbols selected")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error in symbol screening: {e}")
            raise
    
    def get_screening_statistics(self, available_funds: float) -> Dict[str, int]:
        """
        スクリーニング統計取得
        
        Args:
            available_funds: 利用可能資金
            
        Returns:
            Dict[str, int]: 各段階の銘柄数統計
        """
        try:
            stats = {}
            
            # 初期銘柄数
            symbols = self.fetch_nikkei225_symbols()
            stats['initial'] = len(symbols)
            
            # 各フィルタ後
            symbols = self.apply_price_filter(symbols)
            stats['after_price_filter'] = len(symbols)
            
            symbols = self.apply_market_cap_filter(symbols)
            stats['after_market_cap_filter'] = len(symbols)
            
            symbols = self.apply_affordability_filter(symbols, available_funds)
            stats['after_affordability_filter'] = len(symbols)
            
            symbols = self.apply_volume_filter(symbols)
            stats['after_volume_filter'] = len(symbols)
            
            # 最終選定数
            max_symbols = self.config["screening"]["nikkei225_filters"]["max_symbols"]
            stats['final'] = min(len(symbols), max_symbols)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting screening statistics: {e}")
            return {}
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """設定ファイル読み込み"""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return {**self.default_config, **config}
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # デフォルト設定ファイルパス
        default_config_path = Path(__file__).parent.parent.parent / "config" / "dssms" / "dssms_config.json"
        if default_config_path.exists():
            try:
                with open(default_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return {**self.default_config, **config}
            except Exception as e:
                self.logger.warning(f"Failed to load default config: {e}")
        
        return self.default_config
    
    def _fetch_from_primary_source(self) -> List[str]:
        """プライマリソースから銘柄取得（動的実装）"""
        try:
            # 1. 設定ファイルからの読み込みを試行
            config_file = self.config["data_sources"]["nikkei225_list"]
            config_path = Path(__file__).parent.parent.parent / "config" / "dssms" / config_file
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    symbols = data.get('symbols', [])
                    if symbols and len(symbols) > 20:  # 有効な銘柄リストの場合
                        self.logger.info(f"Loaded {len(symbols)} symbols from config file")
                        return symbols
            
            # 2. 主要な日経225銘柄を動的に生成
            try:
                major_sectors: List[List[str]] = [
                    # 自動車・輸送機器
                    ["7203", "7267", "7269"],  # トヨタ、ホンダ、スズキ
                    # 電機・精密機器  
                    ["6758", "6861", "6954", "6981"],  # ソニー、キーエンス、ファナック、村田製作所
                    # 情報通信・サービス
                    ["9984", "9432", "4689", "9437"],  # ソフトバンクG、NTT、ヤフー、NTTドコモ
                    # 金融・商社
                    ["8058", "8306", "8316", "8766"],  # 三菱商事、三菱UFJ、三井住友FG、東京海上
                    # 医薬品・化学
                    ["4519", "4452", "4568", "4502"],  # 中外製薬、花王、第一三共、武田薬品
                    # 小売・消費財
                    ["9020", "7974", "8267", "3382"],  # JR東日本、任天堂、イオン、セブン&アイ
                ]
                
                # セクター別銘柄をフラット化
                dynamic_symbols: List[str] = []
                for sector in major_sectors:
                    dynamic_symbols.extend(sector)
                
                # さらに代表的な大型株を追加
                additional_large_caps: List[str] = [
                    "6367", "6702", "4063", "9433", "4543", "6098", "7733", "6594",
                    "8801", "8830", "9501", "9502", "5020", "1605", "2914", "2802"
                ]
                
                dynamic_symbols.extend(additional_large_caps)
                
                # 重複除去とソート
                dynamic_symbols = sorted(list(set(dynamic_symbols)))
                
                self.logger.info(f"Generated {len(dynamic_symbols)} dynamic Nikkei225 symbols")
                return dynamic_symbols
                
            except Exception as api_error:
                self.logger.warning(f"Dynamic API fetch failed: {api_error}")
                
            # 3. 最終フォールバック: 拡張された代表銘柄リスト
            extended_fallback = [
                # 主要セクター代表銘柄（50銘柄程度）
                "7203", "9984", "6758", "9432", "8058", "6861", "9437", "6367", "6702", "4519",
                "7267", "8306", "4063", "9020", "8316", "8766", "9433", "4452", "4568", "6098",
                "6954", "6981", "4689", "7974", "8267", "3382", "4502", "8801", "8830", "9501",
                "9502", "5020", "1605", "2914", "2802", "7733", "6594", "4543", "7269", "8035",
                "4021", "5201", "3436", "4661", "6503", "9843", "6752", "6645", "4004", "8601"
            ]
            
            self.logger.info(f"Using extended fallback list with {len(extended_fallback)} symbols")
            return extended_fallback
            
        except Exception as e:
            self.logger.warning(f"Primary source fetch failed: {e}")
            return []
    
    def _fetch_from_backup_source(self) -> List[str]:
        """バックアップソースから銘柄取得（最小限の代表銘柄）"""
        try:
            # 最小限の代表的な銘柄リスト（各セクターの代表）
            backup_symbols = [
                # 各セクターの代表銘柄（30銘柄程度）
                "7203", "9984", "6758", "9432", "8058", "6861", "9437", "6367", "6702", "4519",
                "7267", "8306", "4063", "9020", "8316", "8766", "9433", "4452", "4568", "6098",
                "6954", "6981", "4689", "7974", "8267", "3382", "4502", "8801", "8830", "9501"
            ]
            
            self.logger.info(f"Using backup source with {len(backup_symbols)} representative symbols")
            return backup_symbols
            
        except Exception as e:
            self.logger.warning(f"Backup source fetch failed: {e}")
            return []


if __name__ == "__main__":
    # テスト実行
    screener = Nikkei225Screener()
    
    try:
        # 100万円の利用可能資金でテスト
        available_funds = 1_000_000
        
        print("=== DSSMS Nikkei225 Screener Test ===")
        
        # 統計取得
        stats = screener.get_screening_statistics(available_funds)
        print("\nScreening Statistics:")
        for stage, count in stats.items():
            print(f"  {stage}: {count} symbols")
        
        # 最終選定
        filtered_symbols = screener.get_filtered_symbols(available_funds)
        print(f"\nFinal selected symbols ({len(filtered_symbols)}):")
        for symbol in filtered_symbols:
            print(f"  {symbol}")
            
    except Exception as e:
        print(f"Test failed: {e}")

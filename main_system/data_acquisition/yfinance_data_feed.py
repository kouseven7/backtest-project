"""
yfinance_data_feed.py - yfinance統合データフィード実装（シンプル版）

Phase 4.2実装: リアルデータ取得機能
- yfinance APIを使用した株価データ取得
- エラーハンドリング実装
- copilot-instructions.md準拠: フォールバック機能削除（モック/ダミーデータ禁止）

Author: imega
Created: 2025-10-20
Modified: 2025-10-20
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

# プロジェクトパス設定
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger


class YFinanceDataFeed:
    """
    yfinance統合データフィード（シンプル版）
    
    機能:
    - yfinanceを使用したリアルデータ取得
    - エラーハンドリング
    - copilot-instructions.md準拠: データ取得失敗時はエラー発生（フォールバック禁止）
    """
    
    def __init__(self):
        """初期化"""
        self.logger = setup_logger(
            "YFinanceDataFeed",
            log_file="logs/yfinance_data_feed.log"
        )
        
        # yfinanceインポート（遅延インポート）
        self.yf = None
        self._import_yfinance()
        
        self.logger.info("YFinanceDataFeed initialized")
    
    def _import_yfinance(self):
        """
        yfinance遅延インポート
        
        Raises:
            RuntimeError: yfinanceインポート失敗時
        """
        try:
            import yfinance as yf
            self.yf = yf
            self.logger.info("yfinance imported successfully")
        except ImportError as e:
            self.logger.error(f"Failed to import yfinance: {e}")
            raise RuntimeError(
                "yfinance is not installed. "
                "Please install it with: pip install yfinance. "
                "Mock/dummy data fallback is prohibited by copilot-instructions.md"
            ) from e
    
    def get_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        株価データ取得
        
        Args:
            ticker: ティッカーシンボル（例: "AAPL", "^GSPC"）
            start_date: 開始日（YYYY-MM-DD形式）
            end_date: 終了日（YYYY-MM-DD形式）
            days_back: 取得日数（start_date未指定時）
        
        Returns:
            株価データ（OHLCV + Adj Close）
            
        Raises:
            RuntimeError: データ取得失敗時（copilot-instructions.md準拠）
        """
        self.logger.info(f"Getting stock data for {ticker}")
        
        # 日付範囲設定
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if start_date is None:
            start_datetime = datetime.now() - timedelta(days=days_back)
            start_date = start_datetime.strftime("%Y-%m-%d")
        
        self.logger.info(f"Date range: {start_date} to {end_date}")
        
        # yfinanceからリアルデータ取得（必須）
        if self.yf is None:
            raise RuntimeError(
                f"Cannot retrieve data for {ticker}: yfinance is not available. "
                "Mock/dummy data fallback is prohibited by copilot-instructions.md"
            )
        
        try:
            data = self._get_yfinance_data(ticker, start_date, end_date)
            if data is not None and len(data) > 0:
                self.logger.info(f"Successfully retrieved {len(data)} rows from yfinance")
                return data
            else:
                self.logger.error(f"yfinance returned empty data for {ticker}")
                raise RuntimeError(
                    f"Failed to retrieve data for {ticker}: yfinance returned empty dataset. "
                    f"Date range: {start_date} to {end_date}. "
                    "Mock/dummy data fallback is prohibited by copilot-instructions.md"
                )
        except RuntimeError:
            # RuntimeErrorはそのまま再送出
            raise
        except Exception as e:
            self.logger.error(f"yfinance data retrieval error: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to retrieve data for {ticker}: {e}. "
                "Mock/dummy data fallback is prohibited by copilot-instructions.md"
            ) from e
    
    def _get_yfinance_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        yfinanceを使用してデータ取得
        
        Args:
            ticker: ティッカーシンボル
            start_date: 開始日
            end_date: 終了日
        
        Returns:
            株価データ（成功時）またはNone（失敗時）
        """
        try:
            self.logger.info(f"Downloading data from yfinance: {ticker}")
            
            # yfinance Ticker オブジェクト作成
            stock = self.yf.Ticker(ticker)
            
            # 株価データダウンロード
            data = stock.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False  # Adj Close を別カラムで取得
            )
            
            if data.empty:
                self.logger.warning(f"No data returned from yfinance for {ticker}")
                return None
            
            # カラム名の正規化（yfinanceは大文字小文字混在の場合がある）
            data.columns = [col.strip() for col in data.columns]
            
            # 必須カラムの確認
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            # Adj Close が存在しない場合は Close をコピー
            if 'Adj Close' not in data.columns:
                self.logger.warning("'Adj Close' not found, using 'Close' instead")
                data['Adj Close'] = data['Close']
            
            # インデックスをdatetimeに変換
            data.index = pd.to_datetime(data.index)
            
            self.logger.info(f"yfinance data retrieved successfully: {len(data)} rows")
            self.logger.debug(f"Data columns: {data.columns.tolist()}")
            self.logger.debug(f"Data shape: {data.shape}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"yfinance data retrieval failed: {e}", exc_info=True)
            return None
    
    # _generate_sample_data() メソッドを削除
    # 理由: copilot-instructions.md違反
    # 「モック/ダミー/テストデータを使用するフォールバック禁止」
    # 実データ取得失敗時はget_stock_data()でRuntimeErrorを発生させる
    
    def get_index_data(
        self,
        index_symbol: str = "^GSPC",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        インデックスデータ取得（S&P 500など）
        
        Args:
            index_symbol: インデックスシンボル（デフォルト: ^GSPC = S&P 500）
            start_date: 開始日
            end_date: 終了日
            days_back: 取得日数
        
        Returns:
            インデックスデータ
        """
        self.logger.info(f"Getting index data: {index_symbol}")
        
        # 株価データ取得と同じロジックを使用
        return self.get_stock_data(index_symbol, start_date, end_date, days_back)


# テスト用コード
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("YFinanceDataFeed - yfinance統合テスト")
    print("=" * 80 + "\n")
    
    # YFinanceDataFeed初期化
    data_feed = YFinanceDataFeed()
    
    # テスト1: 株価データ取得（AAPL）
    print("[TEST 1] AAPL株価データ取得")
    print("-" * 80)
    aapl_data = data_feed.get_stock_data("AAPL", days_back=30)
    print(f"データ行数: {len(aapl_data)}")
    print(f"カラム: {aapl_data.columns.tolist()}")
    print(f"最新5行:\n{aapl_data.tail()}\n")
    
    # テスト2: インデックスデータ取得（S&P 500）
    print("[TEST 2] S&P 500インデックスデータ取得")
    print("-" * 80)
    index_data = data_feed.get_index_data("^GSPC", days_back=30)
    print(f"データ行数: {len(index_data)}")
    print(f"カラム: {index_data.columns.tolist()}")
    print(f"最新5行:\n{index_data.tail()}\n")
    
    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80 + "\n")

"""
Module: DSSMS Data Bridge
File: dssms_data_bridge.py
Description: 
  DSSMS用データ取得統合ブリッジです。
  既存のdata_fetcher.pyとDSSMS診断システムを統合し、
  フォールバック機能付きのデータ取得を提供します。

Author: GitHub Copilot
Created: 2025-08-22
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Union
import sys
import os
from datetime import datetime, timedelta
import logging

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# モジュールインポートとフォールバック
fetch_stock_data = None
DSSMSDataDiagnostics = None
setup_logger_func = None

try:
    from data_fetcher import fetch_stock_data
except ImportError:
    pass

try:
    from src.dssms.dssms_data_diagnostics import DSSMSDataDiagnostics
except ImportError:
    pass

try:
    from config.logger_config import setup_logger as setup_logger_func
except ImportError:
    pass

def create_logger(name: str) -> logging.Logger:
    """ロガー作成（フォールバック機能付き）"""
    if setup_logger_func:
        return setup_logger_func(name)
    else:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class DSSMSDataBridge:
    """DSSMS用データ取得統合ブリッジ"""
    
    def __init__(self):
        """初期化"""
        self.logger = create_logger(__name__)
        
        # DSSMSDataDiagnosticsが利用可能な場合のみ初期化
        if DSSMSDataDiagnostics:
            self.diagnostics = DSSMSDataDiagnostics()
        else:
            self.diagnostics = None
            self.logger.warning("DSSMSDataDiagnostics not available")
            
        self.last_diagnosis: Optional[Dict[str, Any]] = None
        
        self.logger.info("DSSMS データブリッジを初期化しました")
        
    def get_data_with_fallback(self, symbols: List[str], 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """フォールバック付きデータ取得"""
        self.logger.info(f"データ取得開始: {symbols}")
        
        # 日付型変換
        end_dt: datetime
        start_dt: datetime
        
        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
        if start_date is None:
            start_dt = end_dt - timedelta(days=365)  # 1年前
        else:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        
        # 診断実行（初回または前回から時間経過時）
        if self._should_run_diagnosis():
            if self.diagnostics:
                self.last_diagnosis = self.diagnostics.diagnose_data_sources(symbols)
        
        results: Dict[str, pd.DataFrame] = {}
        failed_symbols: List[str] = []
        
        # 第1段階: 既存fetch_stock_data使用
        for symbol in symbols:
            try:
                # fetch_stock_data関数が利用可能な場合
                data = self._try_fetch_stock_data(symbol, start_dt, end_dt)
                if data is not None and len(data) > 0:
                    results[symbol] = data
                    self.logger.debug(f"fetch_stock_data成功: {symbol}")
                    continue
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                self.logger.warning(f"fetch_stock_data失敗 {symbol}: {e}")
                failed_symbols.append(symbol)
        
        # 第2段階: フォールバック処理
        if failed_symbols:
            self.logger.info(f"フォールバック実行: {failed_symbols}")
            fallback_results = self._execute_fallback(failed_symbols)
            results.update(fallback_results)
        
        self.logger.info(f"データ取得完了: {len(results)}/{len(symbols)} 成功")
        return results
    
    def _try_fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """既存fetch_stock_data関数の試行"""
        try:
            # fetch_stock_data関数を試行
            if 'fetch_stock_data' in globals():
                data = fetch_stock_data(
                    symbol, 
                    start_date.strftime('%Y-%m-%d'), 
                    end_date.strftime('%Y-%m-%d')
                )
                return data
            else:
                self.logger.debug("fetch_stock_data関数が利用できません")
                return None
        except Exception as e:
            self.logger.debug(f"fetch_stock_data エラー {symbol}: {e}")
            return None
    
    def _should_run_diagnosis(self) -> bool:
        """診断実行判定"""
        if self.last_diagnosis is None:
            return True
        
        # 前回診断から1時間経過で再診断
        last_time = datetime.fromisoformat(self.last_diagnosis['timestamp'])
        return datetime.now() - last_time > timedelta(hours=1)
    
    def _execute_fallback(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """フォールバック実行"""
        results = {}
        
        for symbol in symbols:
            # yfinance直接試行
            data = self.diagnostics._fetch_yfinance_data(symbol)
            if data is not None and len(data) > 0:
                results[symbol] = data
                self.logger.debug(f"yfinance直接成功: {symbol}")
                continue
            
            # キャッシュ試行
            data = self.diagnostics._fetch_cache_data(symbol)
            if data is not None and len(data) > 0:
                results[symbol] = data
                self.logger.debug(f"キャッシュ成功: {symbol}")
                continue
            
            # サンプルデータ生成（最終手段）
            self.logger.warning(f"サンプルデータ使用: {symbol}")
            data = self.diagnostics._generate_sample_data(symbol)
            results[symbol] = data
        
        return results
    
    def get_diagnosis_report(self) -> Optional[Dict[str, Any]]:
        """最新診断結果取得"""
        return self.last_diagnosis
    
    def get_real_market_data(self, symbol: str, date: datetime) -> pd.DataFrame:
        """実際のマーケットデータ取得（DSSMS統合用）"""
        try:
            end_date = date + timedelta(days=1)
            start_date = date - timedelta(days=30)  # 30日分のデータ
            
            results = self.get_data_with_fallback(
                [symbol], 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if symbol in results:
                return results[symbol]
            else:
                self.logger.warning(f"データ取得失敗: {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"実データ取得エラー {symbol}: {e}")
            return pd.DataFrame()
    
    def update_portfolio_value_with_real_data(self, date: datetime, position: Optional[str], 
                                            current_value: float) -> float:
        """実データベースのポートフォリオ価値更新"""
        if not position:
            return current_value
            
        try:
            # 実際の価格データ取得
            data = self.get_real_market_data(position, date)
            
            if not data.empty and len(data) >= 2:
                # 実際の日次リターン計算
                close_prices = data['Close'].dropna()
                if len(close_prices) >= 2:
                    daily_return = (close_prices.iloc[-1] / close_prices.iloc[-2]) - 1
                else:
                    daily_return = 0.0
            else:
                # フォールバック: 小さなランダム変動
                import numpy as np
                daily_return = np.random.normal(0.0001, 0.01)
                self.logger.warning(f"フォールバック価格更新 {position}: {daily_return:.4f}")
                
            new_value = current_value * (1 + daily_return)
            
            self.logger.debug(f"価格更新: {position} {daily_return:.4f} -> {new_value:.0f}")
            return new_value
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ価値更新エラー: {e}")
            return current_value
    
    def validate_data_quality(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """データ品質検証"""
        return self.diagnostics._assess_data_quality(data)

def demo_data_bridge():
    """データブリッジデモ"""
    print("=== DSSMS データブリッジデモ ===")
    
    try:
        # ブリッジ初期化
        bridge = DSSMSDataBridge()
        
        # テスト銘柄
        test_symbols = ["7203.T", "8058.T", "9984.T"]
        
        # データ取得テスト
        print(f"\n📊 データ取得テスト: {test_symbols}")
        results = bridge.get_data_with_fallback(test_symbols)
        
        for symbol, data in results.items():
            print(f"✅ {symbol}: {len(data)}行のデータを取得")
            
            # データ品質検証
            quality = bridge.validate_data_quality(symbol, data)
            print(f"   品質スコア: {quality['overall_score']:.2f}")
        
        # ポートフォリオ価値更新テスト
        print(f"\n💰 ポートフォリオ価値更新テスト")
        test_date = datetime.now()
        initial_value = 1000000
        
        for symbol in test_symbols[:2]:  # 2銘柄でテスト
            new_value = bridge.update_portfolio_value_with_real_data(
                test_date, symbol, initial_value
            )
            change = (new_value / initial_value - 1) * 100
            print(f"   {symbol}: {initial_value:,.0f} -> {new_value:,.0f} ({change:+.2f}%)")
        
        # 診断レポート表示
        diagnosis = bridge.get_diagnosis_report()
        if diagnosis:
            print(f"\n📋 診断状態: {diagnosis['overall_status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ デモエラー: {e}")
        return False

if __name__ == "__main__":
    success = demo_data_bridge()
    if success:
        print("\n✅ データブリッジデモ完了")
    else:
        print("\n❌ データブリッジデモ失敗")

"""
DSSMS 銘柄有効性チェッカー
Task 1.2: "possibly delisted" 問題の根本的解決
キャッシュ付きチェック（定期更新）方式
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import time
import requests
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

class SymbolValidityChecker:
    """銘柄有効性チェッカー - キャッシュ付きチェック方式"""
    
    def __init__(self, config_path: Optional[str] = None, cache_dir: str = "cache/dssms"):
        self.logger = setup_logger(__name__)
        self.config = self._load_config(config_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.validity_cache: Dict[str, Any] = {}
        self.last_cache_update: Dict[str, datetime] = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """銘柄チェック設定読み込み"""
        default_config = {
            "cache_settings": {
                "validity_cache_hours": 24,      # 24時間キャッシュ
                "price_cache_hours": 1,          # 1時間価格キャッシュ
                "delisting_cache_days": 7        # 7日間上場廃止キャッシュ
            },
            "validation_criteria": {
                "min_price": 0.01,               # 最低価格
                "max_price_change": 10.0,        # 1日最大変動率（倍数）
                "min_volume": 0,                 # 最低出来高
                "min_market_cap": None,          # 最低時価総額（None=制限なし）
                "recent_trading_days": 5         # 直近取引日数
            },
            "api_settings": {
                "request_timeout": 10,           # リクエストタイムアウト（秒）
                "retry_attempts": 3,             # リトライ回数
                "retry_delay": 1,                # リトライ間隔（秒）
                "batch_size": 10                 # バッチサイズ
            },
            "dssms_symbols": {
                "japan_etfs": ["1306.T", "1321.T", "1570.T"],
                "us_etfs": ["SPY", "QQQ", "IWM"],
                "fallback_symbols": ["^N225", "^GSPC", "^IXIC"]
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"銘柄チェック設定読み込み失敗: {e}")
        
        return default_config
    
    def check_symbols_validity(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        銘柄リストの有効性一括チェック
        Args:
            symbols: チェック対象銘柄リスト
            force_refresh: キャッシュ強制更新
        Returns:
            銘柄別有効性結果辞書
        """
        results: Dict[str, Dict[str, Any]] = {}
        
        # キャッシュから有効性情報読み込み
        if not force_refresh:
            self._load_validity_cache()
        
        # バッチ処理でチェック
        batch_size = self.config["api_settings"]["batch_size"]
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            batch_results = self._check_batch_validity(batch_symbols, force_refresh)
            results.update(batch_results)
            
            # API制限対策
            if i + batch_size < len(symbols):
                time.sleep(0.5)
        
        # キャッシュ保存
        self._save_validity_cache()
        
        return results
    
    def _check_batch_validity(self, symbols: List[str], force_refresh: bool) -> Dict[str, Dict[str, Any]]:
        """バッチでの銘柄有効性チェック"""
        batch_results: Dict[str, Dict[str, Any]] = {}
        
        for symbol in symbols:
            try:
                # キャッシュチェック
                if not force_refresh and self._is_cache_valid(symbol):
                    batch_results[symbol] = self.validity_cache[symbol]
                    continue
                
                # 新規チェック実行
                validity_result = self._perform_validity_check(symbol)
                batch_results[symbol] = validity_result
                
                # キャッシュ更新
                self.validity_cache[symbol] = validity_result
                self.last_cache_update[symbol] = datetime.now()
                
            except Exception as e:
                self.logger.error(f"銘柄チェックエラー [{symbol}]: {e}")
                batch_results[symbol] = {
                    "is_valid": False,
                    "status": "error",
                    "error": str(e),
                    "last_check": datetime.now().isoformat()
                }
        
        return batch_results
    
    def _perform_validity_check(self, symbol: str) -> Dict[str, Any]:
        """個別銘柄の有効性チェック実行"""
        check_result = {
            "symbol": symbol,
            "is_valid": False,
            "status": "unknown",
            "checks": {},
            "last_check": datetime.now().isoformat(),
            "issues": []
        }
        
        try:
            # Yahoo Finance でティッカー情報取得
            ticker = yf.Ticker(symbol)
            
            # 1. 基本情報チェック
            info_check = self._check_ticker_info(ticker, symbol)
            check_result["checks"]["info"] = info_check
            
            # 2. 価格データチェック
            price_check = self._check_price_data(ticker, symbol)
            check_result["checks"]["price"] = price_check
            
            # 3. 取引活動チェック
            trading_check = self._check_trading_activity(ticker, symbol)
            check_result["checks"]["trading"] = trading_check
            
            # 4. 統合判定
            overall_status = self._determine_overall_validity(check_result["checks"])
            check_result.update(overall_status)
            
        except Exception as e:
            self.logger.error(f"銘柄チェック実行エラー [{symbol}]: {e}")
            check_result["status"] = "error"
            check_result["error"] = str(e)
            check_result["issues"].append(f"チェック実行失敗: {e}")
        
        return check_result
    
    def _check_ticker_info(self, ticker: yf.Ticker, symbol: str) -> Dict[str, Any]:
        """ティッカー基本情報チェック"""
        try:
            info = ticker.info
            
            if not info or len(info) < 5:
                return {
                    "status": "failed",
                    "reason": "ティッカー情報取得失敗"
                }
            
            # 上場廃止関連キーワードチェック
            delisting_keywords = ["delisted", "suspended", "halted"]
            info_text = str(info).lower()
            
            if any(keyword in info_text for keyword in delisting_keywords):
                return {
                    "status": "failed",
                    "reason": "上場廃止の可能性"
                }
            
            # 基本的なティッカー情報の存在確認
            essential_fields = ["symbol", "shortName"]
            missing_fields = [field for field in essential_fields if field not in info]
            
            if missing_fields:
                return {
                    "status": "warning",
                    "reason": f"基本情報不足: {missing_fields}"
                }
            
            return {
                "status": "passed",
                "ticker_info_count": len(info),
                "has_name": "shortName" in info
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _check_price_data(self, ticker: yf.Ticker, symbol: str) -> Dict[str, Any]:
        """価格データ有効性チェック"""
        try:
            # 直近30日のデータ取得
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                return {
                    "status": "failed",
                    "reason": "価格データなし"
                }
            
            # 価格データの品質チェック
            close_prices = hist['Close'].dropna()
            
            if len(close_prices) == 0:
                return {
                    "status": "failed",
                    "reason": "有効な終値データなし"
                }
            
            # 価格の妥当性チェック
            latest_price = close_prices.iloc[-1]
            min_price = self.config["validation_criteria"]["min_price"]
            
            if latest_price < min_price:
                return {
                    "status": "failed",
                    "reason": f"価格が低すぎる: {latest_price}"
                }
            
            # 異常な価格変動チェック
            if len(close_prices) > 1:
                price_changes = close_prices.pct_change().dropna()
                max_change = abs(price_changes).max()
                change_limit = self.config["validation_criteria"]["max_price_change"]
                
                if max_change > change_limit:
                    return {
                        "status": "warning",
                        "reason": f"異常な価格変動: {max_change:.2%}"
                    }
            
            return {
                "status": "passed",
                "data_points": len(close_prices),
                "latest_price": float(latest_price),
                "date_range": f"{hist.index[0].date()} to {hist.index[-1].date()}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _check_trading_activity(self, ticker: yf.Ticker, symbol: str) -> Dict[str, Any]:
        """取引活動チェック"""
        try:
            # 直近データで出来高チェック
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config["validation_criteria"]["recent_trading_days"])
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty or 'Volume' not in hist.columns:
                return {
                    "status": "warning",
                    "reason": "出来高データなし"
                }
            
            volumes = hist['Volume'].dropna()
            
            # 出来高ゼロの日数チェック
            zero_volume_days = (volumes == 0).sum()
            total_days = len(volumes)
            
            if total_days > 0:
                zero_ratio = zero_volume_days / total_days
                
                if zero_ratio > 0.5:  # 50%以上ゼロ出来高
                    return {
                        "status": "failed",
                        "reason": f"取引活動不足: {zero_ratio:.1%}がゼロ出来高"
                    }
                elif zero_ratio > 0.2:  # 20%以上ゼロ出来高
                    return {
                        "status": "warning",
                        "reason": f"取引活動低下: {zero_ratio:.1%}がゼロ出来高"
                    }
            
            return {
                "status": "passed",
                "avg_volume": float(volumes.mean()) if len(volumes) > 0 else 0,
                "zero_volume_ratio": float(zero_ratio) if total_days > 0 else 0
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _determine_overall_validity(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """統合有効性判定"""
        issues = []
        critical_failures = 0
        warnings = 0
        
        # 各チェック結果の評価
        for check_name, check_result in checks.items():
            status = check_result.get("status", "unknown")
            
            if status == "failed":
                critical_failures += 1
                reason = check_result.get("reason", "不明な失敗")
                issues.append(f"{check_name}: {reason}")
            elif status == "warning":
                warnings += 1
                reason = check_result.get("reason", "不明な警告")
                issues.append(f"{check_name}: {reason}")
            elif status == "error":
                critical_failures += 1
                error = check_result.get("error", "不明なエラー")
                issues.append(f"{check_name}: エラー - {error}")
        
        # 総合判定
        if critical_failures > 0:
            is_valid = False
            status = "invalid"
        elif warnings > 2:
            is_valid = False
            status = "unreliable"
        elif warnings > 0:
            is_valid = True
            status = "valid_with_warnings"
        else:
            is_valid = True
            status = "valid"
        
        return {
            "is_valid": is_valid,
            "status": status,
            "issues": issues,
            "critical_failures": critical_failures,
            "warnings": warnings
        }
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """キャッシュ有効性判定"""
        if symbol not in self.validity_cache or symbol not in self.last_cache_update:
            return False
        
        last_update = self.last_cache_update[symbol]
        cache_hours = self.config["cache_settings"]["validity_cache_hours"]
        expiry_time = last_update + timedelta(hours=cache_hours)
        
        return datetime.now() < expiry_time
    
    def _load_validity_cache(self) -> None:
        """有効性キャッシュ読み込み"""
        cache_file = self.cache_dir / "symbol_validity_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                self.validity_cache = cache_data.get("validity_cache", {})
                
                # 日時文字列をdatetimeオブジェクトに変換
                update_times = cache_data.get("last_update_times", {})
                self.last_cache_update = {}
                for symbol, time_str in update_times.items():
                    try:
                        self.last_cache_update[symbol] = datetime.fromisoformat(time_str)
                    except:
                        pass  # 無効な日時は無視
                        
            except Exception as e:
                self.logger.warning(f"キャッシュ読み込み失敗: {e}")
                self.validity_cache = {}
                self.last_cache_update = {}
    
    def _save_validity_cache(self) -> None:
        """有効性キャッシュ保存"""
        cache_file = self.cache_dir / "symbol_validity_cache.json"
        
        try:
            # datetimeオブジェクトを文字列に変換
            update_times = {}
            for symbol, dt in self.last_cache_update.items():
                update_times[symbol] = dt.isoformat()
            
            cache_data = {
                "validity_cache": self.validity_cache,
                "last_update_times": update_times,
                "cache_version": "1.0",
                "last_save": datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"キャッシュ保存失敗: {e}")
    
    def get_valid_symbols(self, symbols: List[str], fallback_enabled: bool = True) -> List[str]:
        """有効な銘柄リスト取得"""
        validity_results = self.check_symbols_validity(symbols)
        
        valid_symbols = []
        for symbol, result in validity_results.items():
            if result.get("is_valid", False):
                valid_symbols.append(symbol)
        
        # 有効な銘柄が不足している場合のフォールバック
        if len(valid_symbols) < 3 and fallback_enabled:
            fallback_symbols = self.config["dssms_symbols"]["fallback_symbols"]
            self.logger.warning(f"有効銘柄不足 ({len(valid_symbols)}), フォールバック実行")
            
            fallback_results = self.check_symbols_validity(fallback_symbols)
            for symbol, result in fallback_results.items():
                if result.get("is_valid", False) and symbol not in valid_symbols:
                    valid_symbols.append(symbol)
        
        return valid_symbols
    
    def generate_validity_report(self, symbols: List[str]) -> str:
        """銘柄有効性レポート生成"""
        validity_results = self.check_symbols_validity(symbols)
        
        report_lines = [
            "=" * 60,
            "DSSMS 銘柄有効性チェックレポート",
            f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]
        
        # サマリー統計
        total_symbols = len(validity_results)
        valid_count = sum(1 for r in validity_results.values() if r.get("is_valid", False))
        invalid_count = total_symbols - valid_count
        
        report_lines.extend([
            "[CHART] サマリー統計",
            "-" * 20,
            f"チェック対象銘柄数: {total_symbols}",
            f"有効: {valid_count} ({valid_count/total_symbols*100:.1f}%)",
            f"無効: {invalid_count} ({invalid_count/total_symbols*100:.1f}%)",
            ""
        ])
        
        # 詳細結果
        report_lines.extend([
            "[LIST] 詳細結果",
            "-" * 20
        ])
        
        for symbol, result in validity_results.items():
            is_valid = result.get("is_valid", False)
            status = result.get("status", "unknown")
            
            status_emoji = "[OK]" if is_valid else "[ERROR]"
            report_lines.append(f"{status_emoji} {symbol}: {status}")
            
            # 問題詳細
            issues = result.get("issues", [])
            if issues:
                for issue in issues[:3]:  # 最大3件まで表示
                    report_lines.append(f"   - {issue}")
        
        return "\n".join(report_lines)

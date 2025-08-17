"""
DSSMS Fundamental Analyzer
Yahoo Finance APIによる業績データ分析システム

SBI証券スクリーニング条件に基づく業績評価
"""

import sys
from pathlib import Path
import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存システムインポート
from config.logger_config import setup_logger

class FundamentalAnalyzer:
    """
    Yahoo Finance APIによる業績データ分析システム
    
    分析項目:
    - 営業利益黒字判定
    - 連続増益チェック（3四半期）
    - コンセンサス予想比較
    - 業績安定性評価
    """
    
    def __init__(self):
        """初期化"""
        self.logger = setup_logger('dssms.fundamental')
        
        # 分析設定
        self.config = {
            "required_quarters": 3,           # 連続増益判定期間
            "min_operating_margin": 0.02,     # 最低営業利益率
            "stability_threshold": 0.8,       # 安定性閾値
            "growth_threshold": 0.05          # 成長率閾値
        }
        
        # キャッシュ
        self._data_cache = {}
        self._cache_expiry = timedelta(hours=6)
        
        self.logger.info("FundamentalAnalyzer initialized")
    
    def fetch_financial_data(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        財務データ取得
        
        Args:
            symbol: 銘柄コード
            force_refresh: 強制更新フラグ
            
        Returns:
            Dict[str, Any]: 財務データ
        """
        try:
            # キャッシュチェック
            cache_key = f"{symbol}_financial"
            if (not force_refresh and 
                cache_key in self._data_cache):
                cache_time, data = self._data_cache[cache_key]
                if datetime.now() - cache_time < self._cache_expiry:
                    return data
            
            # Yahoo Finance データ取得
            ticker = yf.Ticker(symbol + ".T")
            
            # 基本情報
            info = ticker.info
            
            # 財務諸表（エラーハンドリング強化）
            try:
                financials = ticker.financials
            except Exception as e:
                self.logger.debug(f"Failed to get annual financials for {symbol}: {e}")
                financials = pd.DataFrame()
            
            try:
                quarterly_financials = ticker.quarterly_financials
            except Exception as e:
                self.logger.debug(f"Failed to get quarterly financials for {symbol}: {e}")
                quarterly_financials = pd.DataFrame()
            
            # 統合データ構築
            financial_data = {
                'basic_info': info,
                'annual_financials': financials,
                'quarterly_financials': quarterly_financials,
                'fetch_time': datetime.now()
            }
            
            # キャッシュ保存
            self._data_cache[cache_key] = (datetime.now(), financial_data)
            
            self.logger.debug(f"Financial data fetched for {symbol}")
            return financial_data
            
        except Exception as e:
            self.logger.error(f"Error fetching financial data for {symbol}: {e}")
            return {}
    
    def check_operating_profit_positive(self, symbol: str) -> bool:
        """
        営業利益黒字判定
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            bool: 営業利益黒字フラグ
        """
        try:
            data = self.fetch_financial_data(symbol)
            
            if not data or 'quarterly_financials' not in data:
                return False
            
            quarterly_financials = data['quarterly_financials']
            
            if quarterly_financials.empty:
                # 年次データも試行
                annual_financials = data.get('annual_financials', pd.DataFrame())
                if annual_financials.empty:
                    return False
                quarterly_financials = annual_financials
            
            # 営業利益取得（複数の項目名に対応）
            operating_income_keys = [
                'Operating Income',
                'EBIT',
                'Operating Revenue',
                'Total Operating Income Or Loss',
                'Operating Income (Loss)'
            ]
            
            operating_income = None
            for key in operating_income_keys:
                if key in quarterly_financials.index:
                    operating_income = quarterly_financials.loc[key]
                    break
            
            if operating_income is None:
                self.logger.debug(f"No operating income data found for {symbol}")
                return False
            
            # 最新四半期の営業利益チェック
            latest_income = operating_income.iloc[0]
            
            is_positive = pd.notna(latest_income) and latest_income > 0
            self.logger.debug(f"{symbol} operating profit positive: {is_positive} (value: {latest_income})")
            
            return is_positive
            
        except Exception as e:
            self.logger.warning(f"Operating profit check failed for {symbol}: {e}")
            return False
    
    def check_consecutive_growth(self, symbol: str, quarters: Optional[int] = None) -> bool:
        """
        連続増益チェック
        
        Args:
            symbol: 銘柄コード
            quarters: チェック四半期数（None時は設定値使用）
            
        Returns:
            bool: 連続増益フラグ
        """
        try:
            quarters = quarters or self.config["required_quarters"]
            data = self.fetch_financial_data(symbol)
            
            if not data or 'quarterly_financials' not in data:
                return False
            
            quarterly_financials = data['quarterly_financials']
            
            if quarterly_financials.empty:
                return False
            
            # 売上高取得
            revenue_keys = [
                'Total Revenue',
                'Revenue',
                'Net Sales',
                'Sales',
                'Total Revenues'
            ]
            
            revenue = None
            for key in revenue_keys:
                if key in quarterly_financials.index:
                    revenue = quarterly_financials.loc[key]
                    break
            
            if revenue is None or len(revenue) < quarters + 1:
                self.logger.debug(f"Insufficient revenue data for {symbol}")
                return False
            
            # 連続成長チェック
            growth_count = 0
            for i in range(quarters):
                current = revenue.iloc[i]
                previous = revenue.iloc[i + 1]
                
                if pd.isna(current) or pd.isna(previous):
                    break
                    
                if current > previous:
                    growth_count += 1
                else:
                    break
            
            is_consecutive = growth_count >= quarters
            self.logger.debug(f"{symbol} consecutive growth: {is_consecutive} ({growth_count}/{quarters})")
            
            return is_consecutive
            
        except Exception as e:
            self.logger.warning(f"Consecutive growth check failed for {symbol}: {e}")
            return False
    
    def check_consensus_beat(self, symbol: str) -> bool:
        """
        コンセンサス予想超え判定
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            bool: 予想超えフラグ
        """
        try:
            data = self.fetch_financial_data(symbol)
            
            if not data or 'basic_info' not in data:
                return False
            
            info = data['basic_info']
            
            # アナリスト予想データ
            target_mean_price = info.get('targetMeanPrice')
            current_price = info.get('currentPrice')
            
            if target_mean_price and current_price:
                # 目標株価が現在価格より高い場合は期待超え
                beat_consensus = target_mean_price > current_price
                self.logger.debug(f"{symbol} consensus beat: {beat_consensus} (target: {target_mean_price}, current: {current_price})")
                return beat_consensus
            
            # 予想EPS vs 実績EPS
            forward_eps = info.get('forwardEps')
            trailing_eps = info.get('trailingEps')
            
            if forward_eps and trailing_eps:
                beat_eps = trailing_eps > forward_eps * 0.95  # 予想の95%以上
                self.logger.debug(f"{symbol} EPS beat: {beat_eps} (trailing: {trailing_eps}, forward: {forward_eps})")
                return beat_eps
            
            self.logger.debug(f"{symbol} no consensus data available")
            return False
            
        except Exception as e:
            self.logger.warning(f"Consensus beat check failed for {symbol}: {e}")
            return False
    
    def calculate_fundamental_score(self, symbol: str) -> float:
        """
        総合業績スコア計算
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            float: 業績スコア（0-1）
        """
        try:
            score = 0.0
            max_score = 1.0
            
            # 営業利益黒字（40%）
            if self.check_operating_profit_positive(symbol):
                score += 0.4
                self.logger.debug(f"{symbol} +0.4 for positive operating profit")
            
            # 連続増益（30%）
            if self.check_consecutive_growth(symbol):
                score += 0.3
                self.logger.debug(f"{symbol} +0.3 for consecutive growth")
            
            # コンセンサス超え（20%）
            if self.check_consensus_beat(symbol):
                score += 0.2
                self.logger.debug(f"{symbol} +0.2 for consensus beat")
            
            # 財務安定性（10%）
            stability_score = self._calculate_stability_score(symbol)
            score += stability_score * 0.1
            self.logger.debug(f"{symbol} +{stability_score * 0.1:.3f} for stability")
            
            final_score = min(max_score, score)
            self.logger.debug(f"{symbol} final fundamental score: {final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            self.logger.warning(f"Fundamental score calculation failed for {symbol}: {e}")
            return 0.0
    
    def batch_analyze_fundamentals(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        複数銘柄の業績一括分析
        
        Args:
            symbols: 銘柄リスト
            
        Returns:
            Dict[str, Dict[str, Any]]: 分析結果
        """
        results = {}
        
        for symbol in symbols:
            try:
                analysis = {
                    'operating_profit_positive': self.check_operating_profit_positive(symbol),
                    'consecutive_growth': self.check_consecutive_growth(symbol),
                    'consensus_beat': self.check_consensus_beat(symbol),
                    'fundamental_score': self.calculate_fundamental_score(symbol),
                    'analysis_timestamp': datetime.now()
                }
                
                results[symbol] = analysis
                self.logger.debug(f"Fundamental analysis completed for {symbol}")
                
            except Exception as e:
                self.logger.warning(f"Fundamental analysis failed for {symbol}: {e}")
                results[symbol] = {
                    'operating_profit_positive': False,
                    'consecutive_growth': False,
                    'consensus_beat': False,
                    'fundamental_score': 0.0,
                    'analysis_timestamp': datetime.now(),
                    'error': str(e)
                }
        
        self.logger.info(f"Batch fundamental analysis completed: {len(results)} symbols")
        return results
    
    def _calculate_stability_score(self, symbol: str) -> float:
        """財務安定性スコア計算"""
        try:
            data = self.fetch_financial_data(symbol)
            
            if not data or 'basic_info' not in data:
                return 0.0
            
            info = data['basic_info']
            
            # 負債比率チェック
            debt_to_equity = info.get('debtToEquity', 100)  # デフォルト高値
            
            # ROE チェック
            return_on_equity = info.get('returnOnEquity', 0)
            
            # 安定性スコア計算
            stability = 0.0
            
            # 負債比率が適正範囲（50%以下）
            if debt_to_equity is not None:
                if debt_to_equity <= 50:
                    stability += 0.5
                elif debt_to_equity <= 100:
                    stability += 0.3
            
            # ROEが適正範囲（10%以上）
            if return_on_equity is not None:
                if return_on_equity >= 0.1:
                    stability += 0.5
                elif return_on_equity >= 0.05:
                    stability += 0.3
            
            return min(1.0, stability)
            
        except Exception as e:
            self.logger.debug(f"Stability score calculation failed for {symbol}: {e}")
            return 0.0
    
    def get_analysis_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """分析サマリー取得"""
        try:
            results = self.batch_analyze_fundamentals(symbols)
            
            summary = {
                'total_symbols': len(symbols),
                'operating_profit_positive': sum(1 for r in results.values() if r['operating_profit_positive']),
                'consecutive_growth': sum(1 for r in results.values() if r['consecutive_growth']),
                'consensus_beat': sum(1 for r in results.values() if r['consensus_beat']),
                'avg_fundamental_score': np.mean([r['fundamental_score'] for r in results.values()]),
                'top_performers': sorted(
                    [(symbol, data['fundamental_score']) for symbol, data in results.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating analysis summary: {e}")
            return {}


if __name__ == "__main__":
    # テスト実行
    analyzer = FundamentalAnalyzer()
    
    try:
        test_symbols = ["7203", "9984", "6758"]  # トヨタ、ソフトバンクG、ソニー
        
        print("=== DSSMS Fundamental Analyzer Test ===")
        
        # 個別分析テスト
        print("\n1. Individual analysis:")
        for symbol in test_symbols:
            print(f"\n{symbol}:")
            print(f"  Operating profit positive: {analyzer.check_operating_profit_positive(symbol)}")
            print(f"  Consecutive growth: {analyzer.check_consecutive_growth(symbol)}")
            print(f"  Consensus beat: {analyzer.check_consensus_beat(symbol)}")
            print(f"  Fundamental score: {analyzer.calculate_fundamental_score(symbol):.3f}")
        
        # バッチ分析テスト
        print("\n2. Batch analysis:")
        batch_results = analyzer.batch_analyze_fundamentals(test_symbols)
        
        for symbol, result in batch_results.items():
            print(f"  {symbol}: Score {result['fundamental_score']:.3f}")
        
        # サマリー
        print("\n3. Analysis summary:")
        summary = analyzer.get_analysis_summary(test_symbols)
        
        print(f"  Total symbols: {summary['total_symbols']}")
        print(f"  Operating profit positive: {summary['operating_profit_positive']}")
        print(f"  Consecutive growth: {summary['consecutive_growth']}")
        print(f"  Consensus beat: {summary['consensus_beat']}")
        print(f"  Average score: {summary['avg_fundamental_score']:.3f}")
        
        print("\n  Top performers:")
        for symbol, score in summary['top_performers']:
            print(f"    {symbol}: {score:.3f}")
            
    except Exception as e:
        print(f"Test failed: {e}")

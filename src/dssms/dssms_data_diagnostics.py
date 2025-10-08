"""
Module: DSSMS Data Diagnostics
File: dssms_data_diagnostics.py
Description: 
  DSSMS専用データ取得診断・修正システムです。
  Task 1.1: データ取得問題の診断と修正を実装します。

Author: GitHub Copilot
Created: 2025-08-22
"""

import logging
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os
import json

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from data_fetcher import fetch_stock_data
    from config.logger_config import setup_logger
except ImportError as e:
    print(f"Import warning: {e}")
    # フォールバック用の最小限実装
    def setup_logger(name: str):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class DSSMSDataDiagnostics:
    """DSSMS専用データ取得診断・修正システム"""
    
    def __init__(self, config_path: str = None):
        """初期化"""
        self.logger = setup_logger(__name__)
        self.config = self._load_config(config_path)
        self.cache = {}
        self.fallback_sources = ['yfinance', 'cache', 'sample']
        
        self.logger.info("DSSMSデータ診断システムを初期化しました")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        default_config = {
            "symbols": ["7203.T", "8058.T", "6861.T", "6954.T", "9984.T"],
            "test_period_days": 30,
            "timeout_seconds": 30,
            "max_retries": 3,
            "cache_enabled": True,
            "sample_data_enabled": True
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    default_config.update(config.get('data_diagnostics', {}))
                self.logger.info(f"設定ファイル読み込み成功: {config_path}")
            except Exception as e:
                self.logger.warning(f"設定ファイル読み込み失敗: {e}")
        
        return default_config
    
    def diagnose_data_sources(self, symbols: List[str] = None) -> Dict[str, Any]:
        """データソース診断実行"""
        if symbols is None:
            symbols = self.config['symbols']
        
        self.logger.info(f"データソース診断開始: {symbols}")
        
        diagnosis_results = {
            'timestamp': datetime.now().isoformat(),
            'symbols_tested': symbols,
            'source_results': {},
            'overall_status': 'unknown',
            'recommendations': [],
            'fallback_data': {}
        }
        
        # 各データソースを診断
        for source in self.fallback_sources:
            try:
                result = self._diagnose_single_source(source, symbols)
                diagnosis_results['source_results'][source] = result
                self.logger.info(f"{source}診断完了: 成功率{result['success_rate']:.1%}")
            except Exception as e:
                self.logger.error(f"{source}診断エラー: {e}")
                diagnosis_results['source_results'][source] = {
                    'available': False,
                    'error': str(e),
                    'success_rate': 0.0
                }
        
        # 総合評価
        diagnosis_results = self._evaluate_overall_status(diagnosis_results)
        
        self.logger.info(f"診断完了: {diagnosis_results['overall_status']}")
        return diagnosis_results
    
    def _diagnose_single_source(self, source: str, symbols: List[str]) -> Dict[str, Any]:
        """単一データソース診断"""
        start_time = time.time()
        result = {
            'available': False,
            'success_rate': 0.0,
            'response_time_avg': 0.0,
            'data_quality': {},
            'errors': [],
            'symbol_results': {}
        }
        
        successful_fetches = 0
        response_times = []
        
        for symbol in symbols:
            symbol_start = time.time()
            try:
                if source == 'yfinance':
                    data = self._fetch_yfinance_data(symbol)
                elif source == 'cache':
                    data = self._fetch_cache_data(symbol)
                elif source == 'sample':
                    data = self._generate_sample_data(symbol)
                else:
                    continue
                
                symbol_time = time.time() - symbol_start
                response_times.append(symbol_time)
                
                if data is not None and len(data) > 0:
                    successful_fetches += 1
                    quality = self._assess_data_quality(data)
                    result['symbol_results'][symbol] = {
                        'success': True,
                        'response_time': symbol_time,
                        'data_length': len(data),
                        'quality_score': quality['overall_score']
                    }
                else:
                    result['symbol_results'][symbol] = {
                        'success': False,
                        'error': 'データ取得失敗'
                    }
                    
            except Exception as e:
                result['errors'].append(f"{symbol}: {str(e)}")
                result['symbol_results'][symbol] = {
                    'success': False,
                    'error': str(e)
                }
        
        result['available'] = successful_fetches > 0
        result['success_rate'] = successful_fetches / len(symbols)
        result['response_time_avg'] = sum(response_times) / len(response_times) if response_times else 0
        
        return result
    
    def _fetch_yfinance_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """yfinanceデータ取得"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config['test_period_days'])
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                self.logger.debug(f"yfinance: 空データ {symbol}")
                return None
            
            # キャッシュ保存
            if self.config['cache_enabled']:
                self.cache[symbol] = data.copy()
            
            self.logger.debug(f"yfinance成功: {symbol} ({len(data)}行)")
            return data
            
        except Exception as e:
            self.logger.debug(f"yfinance取得エラー {symbol}: {e}")
            return None
    
    def _fetch_cache_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """キャッシュデータ取得"""
        data = self.cache.get(symbol)
        if data is not None:
            self.logger.debug(f"キャッシュ命中: {symbol}")
        return data
    
    def _generate_sample_data(self, symbol: str) -> pd.DataFrame:
        """サンプルデータ生成（最終フォールバック）"""
        import numpy as np
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=self.config['test_period_days']),
            end=datetime.now(),
            freq='D'
        )
        
        # 基本価格（銘柄に応じて調整）
        base_price = 1000 if symbol.endswith('.T') else 100
        
        # ランダムウォークで価格生成
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # OHLCV データ作成
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': [int(np.random.lognormal(12, 0.5)) for _ in prices]
        }, index=dates)
        
        self.logger.debug(f"サンプルデータ生成: {symbol}")
        return data
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """データ品質評価"""
        quality = {
            'overall_score': 0.0,
            'completeness': 0.0,
            'consistency': 0.0,
            'recency': 0.0,
            'issues': []
        }
        
        try:
            total_points = len(data)
            
            # 完全性チェック
            missing_points = data.isnull().sum().sum()
            quality['completeness'] = max(0, (total_points * 5 - missing_points) / (total_points * 5))
            
            # 一貫性チェック
            consistency_issues = 0
            if 'High' in data.columns and 'Low' in data.columns:
                consistency_issues += (data['High'] < data['Low']).sum()
            if 'Open' in data.columns and 'Close' in data.columns:
                huge_gaps = abs((data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)) > 0.1
                consistency_issues += huge_gaps.sum()
            
            quality['consistency'] = max(0, 1 - consistency_issues / total_points)
            
            # 新しさチェック
            if not data.empty:
                last_date = data.index[-1]
                if hasattr(last_date, 'to_pydatetime'):
                    last_date = last_date.to_pydatetime()
                days_old = (datetime.now() - last_date).days
                quality['recency'] = max(0, 1 - days_old / 7)  # 1週間で0になる
            
            # 総合スコア
            quality['overall_score'] = (
                quality['completeness'] * 0.4 +
                quality['consistency'] * 0.4 +
                quality['recency'] * 0.2
            )
            
        except Exception as e:
            quality['issues'].append(f"品質評価エラー: {e}")
            self.logger.warning(f"データ品質評価エラー: {e}")
        
        return quality
    
    def _evaluate_overall_status(self, diagnosis_results: Dict[str, Any]) -> Dict[str, Any]:
        """総合状態評価"""
        source_results = diagnosis_results['source_results']
        
        # 各ソースの成功率
        yfinance_success = source_results.get('yfinance', {}).get('success_rate', 0)
        cache_success = source_results.get('cache', {}).get('success_rate', 0)
        sample_success = source_results.get('sample', {}).get('success_rate', 0)
        
        # 総合判定
        if yfinance_success >= 0.8:
            diagnosis_results['overall_status'] = 'excellent'
            diagnosis_results['recommendations'] = ['yfinanceが正常に動作しています']
        elif yfinance_success >= 0.5:
            diagnosis_results['overall_status'] = 'good'
            diagnosis_results['recommendations'] = [
                'yfinanceは概ね動作していますが、一部エラーがあります',
                'エラーが頻発する場合はネットワーク状況を確認してください'
            ]
        elif cache_success > 0:
            diagnosis_results['overall_status'] = 'degraded'
            diagnosis_results['recommendations'] = [
                'yfinanceに問題があります。キャッシュデータを使用してください',
                'ネットワーク接続とyfinanceの状況を確認してください'
            ]
        elif sample_success > 0:
            diagnosis_results['overall_status'] = 'fallback'
            diagnosis_results['recommendations'] = [
                '実データの取得に失敗しました。サンプルデータを使用します',
                'ネットワーク接続を確認してください',
                'yfinanceのサービス状況を確認してください'
            ]
        else:
            diagnosis_results['overall_status'] = 'critical'
            diagnosis_results['recommendations'] = [
                'すべてのデータソースが利用できません',
                'システム環境を確認してください'
            ]
        
        return diagnosis_results
    
    def generate_diagnosis_report(self, diagnosis_results: Dict[str, Any], 
                                output_path: str = None) -> str:
        """診断レポート生成"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"diagnosis_report_{timestamp}.md"
        
        report_lines = [
            "# DSSMS データ取得診断レポート",
            f"**生成日時**: {diagnosis_results['timestamp']}",
            f"**総合状態**: {diagnosis_results['overall_status']}",
            "",
            "## 診断結果サマリー",
            f"- 対象銘柄: {len(diagnosis_results['symbols_tested'])}銘柄",
            f"- 診断実行時刻: {diagnosis_results['timestamp']}",
            "",
            "## データソース別結果",
            ""
        ]
        
        # 各ソースの結果
        for source, result in diagnosis_results['source_results'].items():
            status = "[OK] 正常" if result['available'] else "[ERROR] 異常"
            success_rate = result.get('success_rate', 0) * 100
            avg_time = result.get('response_time_avg', 0)
            
            report_lines.extend([
                f"### {source}",
                f"- 状態: {status}",
                f"- 成功率: {success_rate:.1f}%",
                f"- 平均応答時間: {avg_time:.2f}秒",
                ""
            ])
            
            if result.get('errors'):
                report_lines.extend([
                    "**エラー詳細:**",
                    *[f"- {error}" for error in result['errors']],
                    ""
                ])
        
        # 推奨事項
        report_lines.extend([
            "## 推奨事項",
            *[f"- {rec}" for rec in diagnosis_results['recommendations']],
            "",
            "## 銘柄別詳細結果",
            ""
        ])
        
        # 銘柄別結果
        for source, result in diagnosis_results['source_results'].items():
            if 'symbol_results' in result:
                report_lines.extend([
                    f"### {source} - 銘柄別結果",
                    ""
                ])
                for symbol, symbol_result in result['symbol_results'].items():
                    if symbol_result['success']:
                        quality = symbol_result.get('quality_score', 0)
                        report_lines.append(
                            f"- {symbol}: [OK] 成功 (品質: {quality:.2f}, "
                            f"応答: {symbol_result['response_time']:.2f}s)"
                        )
                    else:
                        report_lines.append(f"- {symbol}: [ERROR] 失敗 - {symbol_result.get('error', '不明')}")
                report_lines.append("")
        
        # レポート保存
        try:
            output_dir = os.path.dirname(output_path) or "."
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"診断レポート生成完了: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"レポート生成エラー: {e}")
            return None

def run_diagnosis_demo():
    """診断デモ実行"""
    print("=== DSSMS データ取得診断デモ ===")
    
    try:
        # 診断システム初期化
        diagnostics = DSSMSDataDiagnostics()
        
        # 診断実行
        results = diagnostics.diagnose_data_sources()
        
        # 結果表示
        print(f"\n[CHART] 診断結果: {results['overall_status']}")
        print(f"📅 実行時刻: {results['timestamp']}")
        
        for source, result in results['source_results'].items():
            success_rate = result.get('success_rate', 0) * 100
            status = "[OK]" if result['available'] else "[ERROR]"
            print(f"{status} {source}: {success_rate:.1f}% 成功")
        
        # レポート生成
        report_path = diagnostics.generate_diagnosis_report(results)
        if report_path:
            print(f"\n📄 詳細レポート: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 診断エラー: {e}")
        return False

if __name__ == "__main__":
    success = run_diagnosis_demo()
    if success:
        print("\n[OK] 診断完了")
    else:
        print("\n[ERROR] 診断失敗")

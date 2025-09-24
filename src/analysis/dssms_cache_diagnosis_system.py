#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSSMS Cache vs API Diagnosis System
API/キャッシュ仮説検証用診断システム

Problem 1: 切替判定ロジック劣化（117回→3回の激減）
仮説: APIを直接取得するときはうまく切り替えられるが、キャッシュに切り替わると切替回数が激減する

実装: 推奨案 = 調査手段1（データソース明示的分離テスト）+ 調査手段2（キャッシュ状態監視・ログ追跡）
"""

import sys
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import time
import logging

# プロジェクトルート追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from data_fetcher import fetch_yahoo_data, get_cache_filepath

logger = setup_logger(__name__)


class DSSMSCacheDiagnosisSystem:
    """DSSMS キャッシュ vs API 診断システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.diagnosis_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = {
            'diagnosis_id': self.diagnosis_id,
            'timestamp': datetime.now(),
            'tests': {},
            'summary': {}
        }
        self.cache_events = []
        self.switch_events = []
        
        logger.info(f"DSSMS Cache Diagnosis System initialized: {self.diagnosis_id}")
    
    def log_cache_event(self, event_type: str, symbol: str, data_source: str, 
                       switch_count: int, additional_data: Dict[str, Any] = None):
        """キャッシュイベントログ記録"""
        event = {
            'timestamp': datetime.now(),
            'event_type': event_type,  # 'cache_hit', 'cache_miss', 'api_fetch', 'cache_clear'
            'symbol': symbol,
            'data_source': data_source,
            'switch_count_at_time': switch_count,
            'cache_age': self._get_cache_age(symbol),
            'additional_data': additional_data or {}
        }
        
        self.cache_events.append(event)
        logger.info(f"CACHE_DIAGNOSIS: {json.dumps(event, default=str)}")
        
    def log_switch_event(self, switch_type: str, symbol_from: str, symbol_to: str,
                        data_source: str, decision_data: Dict[str, Any] = None):
        """切替イベントログ記録"""
        event = {
            'timestamp': datetime.now(),
            'switch_type': switch_type,  # 'executed', 'skipped', 'evaluated'
            'symbol_from': symbol_from,
            'symbol_to': symbol_to,
            'data_source': data_source,
            'decision_data': decision_data or {}
        }
        
        self.switch_events.append(event)
        logger.info(f"SWITCH_DIAGNOSIS: {json.dumps(event, default=str)}")
    
    def _get_cache_age(self, symbol: str) -> Optional[float]:
        """キャッシュ年齢取得（秒）"""
        try:
            cache_filepath = get_cache_filepath(symbol, "2023-01-01", "2023-12-31")
            if Path(cache_filepath).exists():
                cache_time = datetime.fromtimestamp(Path(cache_filepath).stat().st_mtime)
                return (datetime.now() - cache_time).total_seconds()
            return None
        except Exception as e:
            logger.warning(f"Cache age calculation failed for {symbol}: {e}")
            return None
    
    def fetch_data_with_source_control(self, symbol: str, start_date: str, end_date: str,
                                     force_api: bool = False, force_cache: bool = False) -> pd.DataFrame:
        """データソース制御付きデータ取得"""
        start_time = time.time()
        
        try:
            if force_api:
                # API直接取得（キャッシュ無効化）
                logger.info(f"FORCE_API: Fetching {symbol} directly from API")
                data = yf.download(symbol, start=start_date, end=end_date)
                self.log_cache_event('api_force', symbol, 'yfinance_direct', 0, {
                    'fetch_time_sec': time.time() - start_time,
                    'data_points': len(data) if not data.empty else 0
                })
                return data
                
            elif force_cache:
                # キャッシュのみ使用（API無効化）
                logger.info(f"FORCE_CACHE: Fetching {symbol} from cache only")
                cache_filepath = get_cache_filepath(symbol, start_date, end_date)
                
                if Path(cache_filepath).exists():
                    data = pd.read_csv(cache_filepath, index_col=0, parse_dates=True)
                    self.log_cache_event('cache_force', symbol, 'cache_only', 0, {
                        'fetch_time_sec': time.time() - start_time,
                        'data_points': len(data),
                        'cache_file': cache_filepath
                    })
                    return data
                else:
                    logger.warning(f"Cache file not found for {symbol}, returning empty DataFrame")
                    self.log_cache_event('cache_miss_force', symbol, 'cache_only', 0, {
                        'error': 'cache_file_not_found',
                        'expected_file': cache_filepath
                    })
                    return pd.DataFrame()
                    
            else:
                # 通常動作（既存ロジック）
                logger.info(f"NORMAL: Fetching {symbol} with normal logic")
                data = fetch_yahoo_data(symbol, start_date, end_date)
                self.log_cache_event('normal_fetch', symbol, 'normal_logic', 0, {
                    'fetch_time_sec': time.time() - start_time,
                    'data_points': len(data) if not data.empty else 0
                })
                return data
                
        except Exception as e:
            logger.error(f"Data fetch failed for {symbol}: {e}")
            self.log_cache_event('fetch_error', symbol, 'error', 0, {
                'error': str(e),
                'fetch_time_sec': time.time() - start_time
            })
            return pd.DataFrame()
    
    def run_data_source_separation_test(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """データソース分離テスト実行"""
        logger.info("=== Starting Data Source Separation Test ===")
        
        test_results = {
            'test_name': 'data_source_separation',
            'symbols': symbols,
            'date_range': f"{start_date} to {end_date}",
            'api_direct': {},
            'cache_only': {},
            'normal_logic': {},
            'comparison': {}
        }
        
        for symbol in symbols:
            logger.info(f"Testing symbol: {symbol}")
            
            # 1. API直接取得
            api_data = self.fetch_data_with_source_control(symbol, start_date, end_date, force_api=True)
            test_results['api_direct'][symbol] = {
                'data_points': len(api_data) if not api_data.empty else 0,
                'data_hash': self._calculate_data_hash(api_data) if not api_data.empty else None
            }
            
            # 2. キャッシュのみ取得
            cache_data = self.fetch_data_with_source_control(symbol, start_date, end_date, force_cache=True)
            test_results['cache_only'][symbol] = {
                'data_points': len(cache_data) if not cache_data.empty else 0,
                'data_hash': self._calculate_data_hash(cache_data) if not cache_data.empty else None
            }
            
            # 3. 通常ロジック取得
            normal_data = self.fetch_data_with_source_control(symbol, start_date, end_date)
            test_results['normal_logic'][symbol] = {
                'data_points': len(normal_data) if not normal_data.empty else 0,
                'data_hash': self._calculate_data_hash(normal_data) if not normal_data.empty else None
            }
            
            # 4. データ比較
            comparison = self._compare_datasets(api_data, cache_data, normal_data)
            test_results['comparison'][symbol] = comparison
            
            # 少し待機（API制限回避）
            time.sleep(1)
        
        self.results['tests']['data_source_separation'] = test_results
        logger.info("=== Data Source Separation Test Completed ===")
        
        return test_results
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """データハッシュ計算"""
        if data.empty:
            return "empty_dataframe"
        return hashlib.md5(data.to_string().encode()).hexdigest()[:16]
    
    def _compare_datasets(self, api_data: pd.DataFrame, cache_data: pd.DataFrame, 
                         normal_data: pd.DataFrame) -> Dict[str, Any]:
        """データセット比較"""
        comparison = {
            'api_vs_cache': {},
            'api_vs_normal': {},
            'cache_vs_normal': {},
            'consistency_score': 0.0
        }
        
        # データ存在チェック
        datasets = {
            'api': not api_data.empty,
            'cache': not cache_data.empty,
            'normal': not normal_data.empty
        }
        
        comparison['data_availability'] = datasets
        
        # データ一致チェック
        if datasets['api'] and datasets['cache']:
            api_hash = self._calculate_data_hash(api_data)
            cache_hash = self._calculate_data_hash(cache_data)
            comparison['api_vs_cache'] = {
                'identical': api_hash == cache_hash,
                'api_hash': api_hash,
                'cache_hash': cache_hash
            }
        
        if datasets['api'] and datasets['normal']:
            api_hash = self._calculate_data_hash(api_data)
            normal_hash = self._calculate_data_hash(normal_data)
            comparison['api_vs_normal'] = {
                'identical': api_hash == normal_hash,
                'api_hash': api_hash,
                'normal_hash': normal_hash
            }
        
        if datasets['cache'] and datasets['normal']:
            cache_hash = self._calculate_data_hash(cache_data)
            normal_hash = self._calculate_data_hash(normal_data)
            comparison['cache_vs_normal'] = {
                'identical': cache_hash == normal_hash,
                'cache_hash': cache_hash,
                'normal_hash': normal_hash
            }
        
        # 一貫性スコア計算
        matches = 0
        total_comparisons = 0
        
        for comp_key in ['api_vs_cache', 'api_vs_normal', 'cache_vs_normal']:
            if comparison[comp_key] and 'identical' in comparison[comp_key]:
                total_comparisons += 1
                if comparison[comp_key]['identical']:
                    matches += 1
        
        comparison['consistency_score'] = matches / total_comparisons if total_comparisons > 0 else 0.0
        
        return comparison
    
    def clear_cache_experiment(self, cache_dir: str = None) -> Dict[str, Any]:
        """キャッシュクリア実験"""
        logger.info("=== Starting Cache Clear Experiment ===")
        
        if cache_dir is None:
            cache_dir = "C:\\Users\\imega\\Documents\\my_backtest_project\\data_cache"
        
        experiment_results = {
            'test_name': 'cache_clear_experiment',
            'cache_dir': cache_dir,
            'pre_clear_cache_count': 0,
            'post_clear_cache_count': 0,
            'cleared_files': []
        }
        
        try:
            # クリア前のキャッシュファイル数
            cache_path = Path(cache_dir)
            if cache_path.exists():
                cache_files = list(cache_path.glob("*.csv"))
                experiment_results['pre_clear_cache_count'] = len(cache_files)
                experiment_results['cleared_files'] = [str(f) for f in cache_files]
                
                # キャッシュファイル削除
                for cache_file in cache_files:
                    cache_file.unlink()
                    logger.info(f"Cleared cache file: {cache_file}")
                
                self.log_cache_event('cache_clear', 'ALL', 'experiment', 0, {
                    'cleared_files_count': len(cache_files),
                    'cache_dir': cache_dir
                })
                
                # クリア後の確認
                remaining_files = list(cache_path.glob("*.csv"))
                experiment_results['post_clear_cache_count'] = len(remaining_files)
                
                logger.info(f"Cache clear experiment: {len(cache_files)} files cleared")
            else:
                logger.warning(f"Cache directory not found: {cache_dir}")
                
        except Exception as e:
            logger.error(f"Cache clear experiment failed: {e}")
            experiment_results['error'] = str(e)
        
        self.results['tests']['cache_clear_experiment'] = experiment_results
        logger.info("=== Cache Clear Experiment Completed ===")
        
        return experiment_results
    
    def generate_diagnosis_report(self, output_dir: str = None) -> str:
        """診断レポート生成"""
        if output_dir is None:
            output_dir = "."
        
        report_filename = f"dssms_cache_diagnosis_report_{self.diagnosis_id}.json"
        report_path = Path(output_dir) / report_filename
        
        # サマリー生成
        self.results['summary'] = {
            'total_cache_events': len(self.cache_events),
            'total_switch_events': len(self.switch_events),
            'test_count': len(self.results['tests']),
            'diagnosis_duration_sec': (datetime.now() - self.results['timestamp']).total_seconds()
        }
        
        # レポート出力
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"Diagnosis report generated: {report_path}")
        return str(report_path)
    
    def run_comprehensive_diagnosis(self, symbols: List[str] = None, 
                                  start_date: str = "2023-01-01", 
                                  end_date: str = "2023-12-31") -> str:
        """包括的診断実行"""
        if symbols is None:
            symbols = ["9101.T", "6758.T", "7203.T"]  # デフォルトテスト銘柄
        
        logger.info(f"=== Starting Comprehensive DSSMS Cache Diagnosis ===")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        try:
            # 1. データソース分離テスト
            self.run_data_source_separation_test(symbols, start_date, end_date)
            
            # 2. キャッシュクリア実験
            self.clear_cache_experiment()
            
            # 3. レポート生成
            report_path = self.generate_diagnosis_report()
            
            logger.info(f"=== Comprehensive Diagnosis Completed ===")
            logger.info(f"Report: {report_path}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"Comprehensive diagnosis failed: {e}")
            raise


def main():
    """メイン実行関数"""
    print("DSSMS Cache vs API Diagnosis System")
    print("===================================")
    
    # 診断システム初期化
    diagnosis_system = DSSMSCacheDiagnosisSystem()
    
    # 包括的診断実行
    report_path = diagnosis_system.run_comprehensive_diagnosis(
        symbols=["9101.T", "6758.T"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print(f"\n診断完了: {report_path}")
    print("\n=== 次のステップ ===")
    print("1. 生成されたレポートを確認")
    print("2. API vs キャッシュの一貫性を分析") 
    print("3. 切替判定への影響を評価")
    print("4. 必要に応じてDSSMSバックテストを実行し、切替回数を測定")


if __name__ == "__main__":
    main()
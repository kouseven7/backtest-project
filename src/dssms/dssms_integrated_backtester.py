"""
DSSMS統合システム - メインコントローラー

このモジュールは、DSS Core V3（動的銘柄選択）とマルチ戦略システム（7戦略）を統合し、
動的銘柄選択による高度なバックテストシステムを提供します。

Classes:
    DSSMSIntegratedBacktester: メインコントローラークラス
    
Author: GitHub Copilot
Created: 2025-09-27
Phase: Phase 3 - Implementation
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time
import json
import logging

# プロジェクトルートを追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 既存システムのインポート
from config.logger_config import setup_logger

# DSSMS統合システムのカスタム例外
class DSSMSError(Exception):
    """DSSMS統合システム基底例外"""
    pass

class ConfigError(DSSMSError):
    """設定関連エラー"""
    pass

class DataError(DSSMSError):
    """データ関連エラー"""
    pass

class StrategyError(DSSMSError):
    """戦略実行エラー"""
    pass

class RiskError(DSSMSError):
    """リスク管理エラー"""
    pass

class PositionError(DSSMSError):
    """ポジション管理エラー"""
    pass

class CacheError(DSSMSError):
    """キャッシュ関連エラー"""
    pass

class SystemError(DSSMSError):
    """システムレベルエラー"""
    pass


class DSSMSIntegratedBacktester:
    """
    DSSMS統合バックテスター
    DSS Core V3の銘柄選択 + マルチ戦略実行を統合管理
    
    このクラスは、動的銘柄選択と既存の7戦略システムを統合し、
    日次で最適銘柄を選択しながら戦略を実行するバックテストを提供します。
    
    Attributes:
        config (Dict[str, Any]): 統合設定
        dss_core: DSS Core V3インスタンス（遅延初期化）
        strategy_adapter: マルチ戦略アダプター
        switch_manager: 銘柄切替管理
        position_manager: ポジション管理
        data_cache: データキャッシュ管理
        performance_tracker: パフォーマンス監視
        
        current_symbol (str): 現在選択中の銘柄
        portfolio_value (float): 現在のポートフォリオ価値
        daily_results (List[Dict]): 日次結果履歴
        switch_history (List[Dict]): 銘柄切替履歴
        
    Example:
        config = {
            'initial_capital': 1000000,
            'backtest_period': {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31'
            }
        }
        backtester = DSSMSIntegratedBacktester(config)
        result = backtester.run_dynamic_backtest(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        DSSMS統合バックテスターの初期化
        
        Parameters:
            config (Dict[str, Any]): 統合設定辞書
                Required keys:
                    - 'initial_capital' (float): 初期資本金 (>= 100000)
                    - 'backtest_period' (Dict): バックテスト期間設定
                        - 'start_date' (str): 開始日 'YYYY-MM-DD'
                        - 'end_date' (str): 終了日 'YYYY-MM-DD'
                Optional keys:
                    - 'switch_cost_rate' (float): 銘柄切替コスト率 (default: 0.001)
                    - 'performance_targets' (Dict): パフォーマンス目標値
                    - 'enable_cache' (bool): キャッシュ有効化 (default: True)
                    - 'log_level' (str): ログレベル (default: 'INFO')
        
        Raises:
            ValueError: 必須設定項目の不足・無効値
            ConfigError: 設定ファイル形式エラー
            SystemError: システム初期化失敗
        """
        # 設定バリデーション
        self._validate_config(config)
        self.config = config
        
        # ログ設定
        log_level = config.get('log_level', 'INFO')
        self.logger = setup_logger(f"{self.__class__.__name__}", level=log_level)
        
        # 基本設定
        self.initial_capital = config['initial_capital']
        self.switch_cost_rate = config.get('switch_cost_rate', 0.001)
        
        # 状態管理
        self.current_symbol = None
        self.portfolio_value = self.initial_capital
        self.daily_results = []
        self.switch_history = []
        self.is_running = False
        
        # コンポーネント初期化（遅延初期化で実装負荷軽減）
        self.dss_core = None
        self.strategy_adapter = None
        self.switch_manager = None
        self.position_manager = None
        self.data_cache = None
        self.performance_tracker = None
        
        # 初期化完了ログ
        self.logger.info(f"DSSMSIntegratedBacktester初期化完了 - 初期資本: {self.initial_capital:,}円")
        self.logger.info(f"切替コスト率: {self.switch_cost_rate:.1%}")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        設定辞書のバリデーション
        
        Parameters:
            config (Dict[str, Any]): 設定辞書
            
        Raises:
            ValueError: 必須項目不足・無効値
            ConfigError: 設定形式エラー
        """
        # 必須項目チェック
        required_keys = ['initial_capital']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"必須設定項目が不足: {key}")
        
        # 初期資本金チェック
        initial_capital = config['initial_capital']
        if not isinstance(initial_capital, (int, float)) or initial_capital < 100000:
            raise ValueError(f"初期資本金は100,000円以上である必要があります: {initial_capital}")
        
        # 切替コスト率チェック
        switch_cost_rate = config.get('switch_cost_rate', 0.001)
        if not isinstance(switch_cost_rate, (int, float)) or not (0 <= switch_cost_rate <= 0.1):
            raise ValueError(f"切替コスト率は0-10%の範囲である必要があります: {switch_cost_rate}")
    
    def _initialize_components(self):
        """
        コンポーネントの遅延初期化
        
        実際のバックテスト実行時に必要なコンポーネントを初期化します。
        これにより、クラスインスタンス化時の負荷を軽減します。
        
        Raises:
            SystemError: コンポーネント初期化失敗
        """
        try:
            self.logger.info("DSSMSコンポーネント初期化開始...")
            
            # TODO: Phase 3で段階的に実装
            # 現在はプレースホルダー実装
            
            # DSS Core V3 初期化
            # self.dss_core = DSSBacktesterV3()
            self.logger.info("DSS Core V3 初期化: TODO")
            
            # マルチ戦略アダプター初期化
            # self.strategy_adapter = MultiStrategyAdapter(self.config)
            self.logger.info("MultiStrategyAdapter 初期化: TODO")
            
            # 銘柄切替管理初期化
            # self.switch_manager = SymbolSwitchManager(self.config)
            self.logger.info("SymbolSwitchManager 初期化: TODO")
            
            # ポジション管理初期化
            # self.position_manager = PositionManager(self.config)
            self.logger.info("PositionManager 初期化: TODO")
            
            # データキャッシュ管理初期化
            if self.config.get('enable_cache', True):
                # self.data_cache = DataCacheManager(self.config)
                self.logger.info("DataCacheManager 初期化: TODO")
            
            # パフォーマンス監視初期化
            # self.performance_tracker = PerformanceTracker()
            self.logger.info("PerformanceTracker 初期化: TODO")
            
            self.logger.info("DSSMSコンポーネント初期化完了")
            
        except Exception as e:
            error_msg = f"コンポーネント初期化エラー: {e}"
            self.logger.error(error_msg)
            raise SystemError(error_msg) from e
    
    def run_dynamic_backtest(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        動的銘柄選択バックテストの実行
        
        DSS Core V3による日次銘柄選択と、選択銘柄での7戦略実行を統合した
        動的バックテストを実行します。
        
        Parameters:
            start_date (datetime): バックテスト開始日
            end_date (datetime): バックテスト終了日
        
        Returns:
            Dict[str, Any]: バックテスト結果
                {
                    'status': str,                    # 'success' | 'partial_success' | 'failed'
                    'execution_summary': {
                        'total_days': int,            # 総実行日数
                        'success_days': int,          # 成功日数
                        'switch_count': int,          # 銘柄切替回数
                        'total_execution_time_ms': float
                    },
                    'performance_metrics': {
                        'final_portfolio_value': float,
                        'total_return': float,        # 総収益率
                        'sharpe_ratio': float,        # シャープレシオ
                        'max_drawdown': float,        # 最大ドローダウン
                        'win_rate': float,            # 勝率
                        'switch_cost_total': float    # 総切替コスト
                    },
                    'daily_results': List[Dict],      # 日次結果配列
                    'switch_history': List[Dict],     # 切替履歴
                    'error_log': List[Dict]           # エラーログ
                }
        
        Raises:
            ValueError: 無効な日付範囲
            DataError: 必要データの取得失敗
            SystemError: システム実行エラー
        """
        # バリデーション
        if start_date >= end_date:
            raise ValueError(f"開始日は終了日より前である必要があります: {start_date} >= {end_date}")
        
        # 実行準備
        self.is_running = True
        backtest_start_time = time.time()
        total_execution_time = 0
        success_days = 0
        error_log = []
        
        self.logger.info(f"DSSMS統合バックテスト開始: {start_date.date()} → {end_date.date()}")
        
        try:
            # コンポーネント初期化
            self._initialize_components()
            
            # 日次処理ループ
            current_date = start_date
            total_days = 0
            
            while current_date <= end_date:
                try:
                    # 日次処理実行
                    daily_result = self._process_daily_trading(current_date)
                    self.daily_results.append(daily_result)
                    
                    # 統計更新
                    total_execution_time += daily_result.get('execution_time_ms', 0)
                    if daily_result.get('status') == 'success':
                        success_days += 1
                    
                    total_days += 1
                    
                    # パフォーマンス監視（週次チェック）
                    if self.performance_tracker and hasattr(self.performance_tracker, 'should_check_performance'):
                        if self.performance_tracker.should_check_performance(current_date):
                            self._check_performance_targets(current_date)
                    
                except Exception as e:
                    # 日次処理エラーログ記録
                    error_entry = {
                        'date': current_date,
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'severity': 'ERROR'
                    }
                    error_log.append(error_entry)
                    self.logger.error(f"日次処理エラー ({current_date.date()}): {e}")
                
                # 次の日へ
                current_date += timedelta(days=1)
            
            # 最終結果生成
            final_result = self._generate_final_results(
                total_days=total_days,
                success_days=success_days,
                total_execution_time=total_execution_time,
                backtest_start_time=backtest_start_time,
                error_log=error_log
            )
            
            self.logger.info(f"DSSMS統合バックテスト完了 - 最終ポートフォリオ価値: {self.portfolio_value:,.0f}円")
            return final_result
            
        except Exception as e:
            error_msg = f"バックテスト実行エラー: {e}"
            self.logger.error(error_msg)
            raise SystemError(error_msg) from e
        
        finally:
            self.is_running = False
    
    def _process_daily_trading(self, target_date: datetime) -> Dict[str, Any]:
        """
        日次取引処理（内部メソッド・テスト用公開）
        
        1日分の動的銘柄選択 + 戦略実行 + 切替処理を実行します。
        
        Parameters:
            target_date (datetime): 対象日付
            
        Returns:
            Dict[str, Any]: 日次処理結果
                {
                    'date': datetime,
                    'dss_result': Dict,               # DSS選択結果
                    'switch_result': Dict,            # 切替評価結果
                    'strategy_result': Dict,          # 戦略実行結果
                    'portfolio_value': float,         # 更新後ポートフォリオ価値
                    'execution_time_ms': float        # 実行時間
                }
                
        Note: 単体テスト・デバッグ用途での利用想定
        """
        start_time = time.time()
        
        try:
            # 1. DSS Core で最適銘柄選択
            dss_result = self._run_dss_selection(target_date)
            selected_symbol = dss_result.get('selected_symbol')
            
            # 2. 銘柄切替判定・実行
            switch_result = self._evaluate_and_execute_switch(
                from_symbol=self.current_symbol,
                to_symbol=selected_symbol,
                target_date=target_date
            )
            
            # 3. 現在銘柄でマルチ戦略実行
            strategy_result = self._execute_strategies_for_current_symbol(target_date)
            
            # 4. 日次結果生成
            execution_time = (time.time() - start_time) * 1000
            
            daily_result = {
                'date': target_date,
                'status': 'success',
                'dss_result': dss_result,
                'switch_result': switch_result,
                'strategy_result': strategy_result,
                'portfolio_value': self.portfolio_value,
                'execution_time_ms': execution_time
            }
            
            # パフォーマンス記録
            if hasattr(self.performance_tracker, 'record_daily_performance'):
                self.performance_tracker.record_daily_performance(daily_result)
            
            return daily_result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_result = {
                'date': target_date,
                'status': 'error',
                'error': str(e),
                'portfolio_value': self.portfolio_value,
                'execution_time_ms': execution_time
            }
            self.logger.error(f"日次処理エラー ({target_date.date()}): {e}")
            return error_result
    
    def _run_dss_selection(self, target_date: datetime) -> Dict[str, Any]:
        """DSS Core V3による銘柄選択実行"""
        # TODO: Phase 3後半で実装
        # 現在はモック実装
        mock_symbols = ['7203', '6758', '9984', '6861', '8306']
        selected_symbol = mock_symbols[target_date.day % len(mock_symbols)]
        
        return {
            'selected_symbol': selected_symbol,
            'selection_score': 0.85,
            'execution_time_ms': 50.0,
            'status': 'success'
        }
    
    def _evaluate_and_execute_switch(self, from_symbol: str, to_symbol: str, 
                                   target_date: datetime) -> Dict[str, Any]:
        """銘柄切替評価・実行"""
        # TODO: Phase 3で SymbolSwitchManager 実装後に詳細化
        
        # 初回設定
        if from_symbol is None:
            self.current_symbol = to_symbol
            return {
                'should_switch': True,
                'reason': 'initial_setup',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'switch_cost': 0.0
            }
        
        # 同一銘柄の場合
        if from_symbol == to_symbol:
            return {
                'should_switch': False,
                'reason': 'same_symbol',
                'from_symbol': from_symbol,
                'to_symbol': to_symbol
            }
        
        # 切替実行（簡略化実装）
        switch_cost = self.portfolio_value * self.switch_cost_rate
        self.portfolio_value -= switch_cost
        self.current_symbol = to_symbol
        
        switch_result = {
            'should_switch': True,
            'reason': 'dss_selection_changed',
            'from_symbol': from_symbol,
            'to_symbol': to_symbol,
            'switch_cost': switch_cost
        }
        
        self.switch_history.append(switch_result)
        return switch_result
    
    def _execute_strategies_for_current_symbol(self, target_date: datetime) -> Dict[str, Any]:
        """現在銘柄での戦略実行"""
        # TODO: Phase 3で MultiStrategyAdapter 実装後に詳細化
        
        if not self.current_symbol:
            return {'status': 'no_symbol', 'message': 'No symbol selected'}
        
        # モック戦略実行結果
        return {
            'status': 'success',
            'symbol': self.current_symbol,
            'date': target_date,
            'entry_signal': 1 if target_date.day % 3 == 0 else 0,
            'exit_signal': 0,
            'strategy': 'VWAPBreakoutStrategy',
            'execution_time_ms': 25.0
        }
    
    def _check_performance_targets(self, current_date: datetime) -> None:
        """パフォーマンス目標チェック"""
        # TODO: Phase 3で PerformanceTracker 実装後に詳細化
        self.logger.debug(f"パフォーマンスチェック実行: {current_date.date()}")
    
    def _generate_final_results(self, total_days: int, success_days: int, 
                              total_execution_time: float, backtest_start_time: float,
                              error_log: List[Dict]) -> Dict[str, Any]:
        """バックテスト最終結果生成"""
        total_backtest_time = (time.time() - backtest_start_time) * 1000
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        switch_cost_total = sum(switch.get('switch_cost', 0) for switch in self.switch_history)
        
        return {
            'status': 'success' if success_days / total_days >= 0.95 else 'partial_success',
            'execution_summary': {
                'total_days': total_days,
                'success_days': success_days,
                'success_rate': success_days / total_days if total_days > 0 else 0,
                'switch_count': len(self.switch_history),
                'total_execution_time_ms': total_execution_time,
                'total_backtest_time_ms': total_backtest_time,
                'avg_daily_execution_time_ms': total_execution_time / total_days if total_days > 0 else 0
            },
            'performance_metrics': {
                'initial_portfolio_value': self.initial_capital,
                'final_portfolio_value': self.portfolio_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'sharpe_ratio': 0.0,  # TODO: 計算実装
                'max_drawdown': 0.0,  # TODO: 計算実装
                'win_rate': 0.0,      # TODO: 計算実装
                'switch_cost_total': switch_cost_total,
                'switch_cost_pct': switch_cost_total / self.initial_capital * 100
            },
            'daily_results': self.daily_results,
            'switch_history': self.switch_history,
            'error_log': error_log
        }
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        現在のシステム状態を取得
        
        Returns:
            Dict[str, Any]: システム状態情報
                {
                    'is_running': bool,               # 実行中フラグ
                    'current_symbol': str,            # 現在の選択銘柄
                    'portfolio_value': float,         # 現在のポートフォリオ価値
                    'last_update': datetime,          # 最終更新時刻
                    'performance_status': str,        # パフォーマンス状態
                    'cache_status': {
                        'hit_rate': float,            # キャッシュヒット率
                        'used_memory_mb': float       # 使用メモリ量
                    },
                    'component_status': {
                        'dss_core': str,              # DSS Core状態
                        'strategy_adapter': str,      # 戦略アダプター状態
                        'switch_manager': str,        # 切替管理状態
                        'position_manager': str       # ポジション管理状態
                    }
                }
        """
        return {
            'is_running': self.is_running,
            'current_symbol': self.current_symbol,
            'portfolio_value': self.portfolio_value,
            'last_update': datetime.now(),
            'total_switches': len(self.switch_history),
            'daily_results_count': len(self.daily_results),
            'performance_status': 'unknown',  # TODO: 実装
            'cache_status': {
                'hit_rate': 0.0,     # TODO: 実装
                'used_memory_mb': 0.0  # TODO: 実装
            },
            'component_status': {
                'dss_core': 'not_initialized' if self.dss_core is None else 'initialized',
                'strategy_adapter': 'not_initialized' if self.strategy_adapter is None else 'initialized',
                'switch_manager': 'not_initialized' if self.switch_manager is None else 'initialized',
                'position_manager': 'not_initialized' if self.position_manager is None else 'initialized'
            }
        }
    
    def export_results(self, output_path: str, format: str = 'excel') -> bool:
        """
        バックテスト結果をエクスポート
        
        Parameters:
            output_path (str): 出力ファイルパス
            format (str): 出力形式 ('excel' | 'csv' | 'json')
        
        Returns:
            bool: エクスポート成功フラグ
        
        Raises:
            ValueError: 無効な出力形式
            IOError: ファイル出力エラー
            DataError: 結果データ不整合
        """
        try:
            if format == 'json':
                # JSONエクスポート（基本実装）
                export_data = {
                    'backtest_config': self.config,
                    'final_results': self._generate_final_results(
                        total_days=len(self.daily_results),
                        success_days=sum(1 for r in self.daily_results if r.get('status') == 'success'),
                        total_execution_time=sum(r.get('execution_time_ms', 0) for r in self.daily_results),
                        backtest_start_time=time.time(),
                        error_log=[]
                    )
                }
                
                # 修正: ディレクトリパス処理を改善
                output_dir = os.path.dirname(output_path)
                if output_dir:  # ディレクトリパスが存在する場合のみ作成
                    os.makedirs(output_dir, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                
                self.logger.info(f"結果エクスポート完了: {output_path}")
                return True
            
            else:
                # TODO: Excel/CSV エクスポート実装
                raise ValueError(f"未対応の出力形式: {format}")
                
        except Exception as e:
            self.logger.error(f"エクスポートエラー: {e}")
            return False


# モジュールレベル関数・定数
DEFAULT_CONFIG = {
    'initial_capital': 1000000,
    'switch_cost_rate': 0.001,
    'enable_cache': True,
    'log_level': 'INFO'
}

def create_default_backtester(initial_capital: float = 1000000) -> DSSMSIntegratedBacktester:
    """
    デフォルト設定でのDSSMSバックテスター作成
    
    Parameters:
        initial_capital (float): 初期資本金
        
    Returns:
        DSSMSIntegratedBacktester: 初期化済みバックテスター
    """
    config = DEFAULT_CONFIG.copy()
    config['initial_capital'] = initial_capital
    return DSSMSIntegratedBacktester(config)


if __name__ == "__main__":
    # 動作テスト
    import sys
    
    print("DSSMSIntegratedBacktester 動作テスト")
    print("=" * 50)
    
    try:
        # テスト設定
        test_config = {
            'initial_capital': 1000000,
            'switch_cost_rate': 0.001,
            'log_level': 'INFO'
        }
        
        # バックテスター作成
        backtester = DSSMSIntegratedBacktester(test_config)
        print(f"✅ バックテスター初期化成功")
        
        # 状態確認
        status = backtester.get_current_status()
        print(f"✅ 状態取得成功: {status['current_symbol']}")
        
        # 短期間テスト実行
        test_start = datetime(2023, 6, 1)
        test_end = datetime(2023, 6, 5)
        
        print(f"テストバックテスト実行: {test_start.date()} → {test_end.date()}")
        result = backtester.run_dynamic_backtest(test_start, test_end)
        
        print(f"✅ テスト実行成功")
        print(f"  - 実行日数: {result['execution_summary']['total_days']}")
        print(f"  - 成功率: {result['execution_summary']['success_rate']:.1%}")
        print(f"  - 銘柄切替回数: {result['execution_summary']['switch_count']}")
        print(f"  - 最終ポートフォリオ価値: {result['performance_metrics']['final_portfolio_value']:,.0f}円")
        
        # エクスポートテスト
        output_path = "test_dssms_result.json"
        if backtester.export_results(output_path, 'json'):
            print(f"✅ エクスポート成功: {output_path}")
        
        print("\n🎉 DSSMSIntegratedBacktester 実装完了！")
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        sys.exit(1)
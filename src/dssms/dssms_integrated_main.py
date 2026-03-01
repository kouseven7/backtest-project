"""
DSSMS統合メインモジュール - 動的銘柄選択とマルチ戦略バックテスト実行

日経225銘柄からパーフェクトオーダーの銘柄を動的に選択し、
複数の戦略を統合してバックテストを実行します。

主な機能:
- DSSMS銘柄選択（Screening/Ranking/Scoring/Symbol Switching）
- マルチ戦略実行制御（BaseStrategy派生クラス統合）
- ポジション管理（エントリー/エグジット/銘柄切替）
- バックテスト終了時の強制決済処理（重要: 削除禁止、Line 670-850付近）
- 日次データ取得とルックアヘッドバイアス防止
- 取引履歴の記録とレポート生成
- パフォーマンス統計の計算

統合コンポーネント:
- DSSMS Core: symbol_switch_manager_ultra_light（銘柄選択）
- Screener: nikkei225_screener.py（日経225銘柄フィルタリング）
- 戦略層: GCStrategy, VWAPBreakoutStrategy, BreakoutStrategy等
- データ層: data_fetcher.py, data_cache_manager.py
- 出力: CSV+JSON+TXT統一出力エンジン

セーフティ機能/注意事項:
- バックテスト終了時の強制決済処理は絶対に削除しないこと
  （Line 670-850付近、削除すると未決済ポジションが残る）
- ルックアヘッドバイアス防止（日次データ取得、翌日始値エントリー）
- yfinance auto_adjust=False 必須（Adj Close取得のため）
- 二重サフィックス防止（to_yfinance()関数使用）
- DSSMS日次判定 vs マルチ戦略全期間判定の整合性に注意

既知の問題と対策:
- Issue #2: 強制決済コードの削除 -> 削除禁止コメント追加済み
- Issue #5: API期間異常 -> Screener target_date対応必要
- Issue #1: 二重サフィックス問題 -> to_yfinance()関数で防止

Author: Backtest Project Team
Created: 2025-09-28
Last Modified: 2026-02-05
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from pathlib import Path
import json
import uuid
import argparse

# pandas, numpy loaded when needed via lazy_import
import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Direct file import (avoid heavy __init__.py chain)
import importlib.util
import os

# Task 1実装 (2026-01-13): 完全版SymbolSwitchManagerに切替
# ultra_light版から変更 - 保有期間制限・頻度削減ロジックを復元
# 目標: 85回 → 30回以下の切替、平均保有期間 3.9日 → 10日以上
# パフォーマンス影響: +0.1秒/11ヶ月（無視可能）
# DSSMS_Implementation_Plan.md Task 1 参照

try:
    from src.dssms.symbol_switch_manager import SymbolSwitchManager
except ImportError as e:
    import logging
    logging.error(f"[TASK1_ERROR] SymbolSwitchManager import failed: {e}")
    SymbolSwitchManager = None

from src.utils.symbol_utils import to_yfinance

# SystemFallbackPolicy利用可能性チェック(TODO-INTEGRATE-001対応)
try:
    from src.config.system_modes import get_fallback_policy, ComponentType
    fallback_policy_available = True
except ImportError:
    fallback_policy_available = False

# DSS Core V3利用可能性チェック(TODO-INTEGRATE-001対応)
try:
    from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
    dss_available = True
except ImportError:
    dss_available = False

try:
    import yfinance
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    DATA_FETCHER_AVAILABLE = False


class DSSMSIntegrationError(Exception):
    pass


class DSSMSIntegratedBacktester:
    """

    DSS Core V3の動的銘柄選択とマルチ戦略システムを統合し高度なバックテストシステムを提供

    Responsibilities:
    - DSS Core V3との連携(動的銘柄選択)
    - マルチ戦略システムとの統合
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        
        Args:
            config: 統合設定辞書
        
        Raises:
            DSSMSIntegrationError: 初期化失敗
        """
        try:
            # 設定初期化
            self.config = config or self._load_default_config()
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.setLevel(logging.INFO)
            
            self._file_logging_initialized = False
            
            self.dss_core = None
            self.advanced_ranking_engine = None
            self.risk_manager = None
            self._dss_initialized = False
            self._ranking_initialized = False
            self._risk_initialized = False
            
            self.switch_manager = None
            self.data_cache = None
            self.performance_tracker = None
            self.report_generator = None
            self.nikkei225_screener = None
            self._components_initialized = False
            
            # 実際の初期化は必要時に行う
            
            # システム状態
            # self.current_symbol = None  # Sprint 2削除: 複数銘柄保有対応
            # 代替: self.positions.keys()で保有銘柄リストを取得
            self.portfolio_value = self.config.get('initial_capital', 1000000)
            self.initial_capital = self.portfolio_value
            
            # Cycle 4-A Cycle 3改善: 利益保護用エントリー価格追跡
            self.last_entry_price = None  # 最後のエントリー価格（切替判定時に参照）
            
            # Cycle 4-2: 残高管理システム追加（2026-01-10）
            self.cash_balance = self.initial_capital  # 現金残高追跡
            self.logger.info(f"[CASH_MANAGEMENT] 残高管理システム初期化: cash_balance={self.cash_balance:,.0f}円")
            
            # [PRIORITY E] peak_value初期化追加（Priority D-2で発見された問題の修正）
            self.peak_value = self.portfolio_value  # 初期値はportfolio_valueと同じ
            
            # 修正案2: 最後に実行した戦略名を記録(December 14, 2025追加)
            # 目的: execution_detailsに実際の戦略名を記録する
            self.last_executed_strategy = None
            # 削除: self.position_size, self.position_entry_price初期化
            #   - portfolio_valueのみを追跡(main_new.pyのreturnを累積)
            # 影響: position_value/cash_balance計算を全額キャッシュ扱いに変更
            self.force_close_in_progress = False  # [Task11] DSSMS側ForceCloseフラグ
            
            # Option A実装(December 28, 2025): MainSystemController instance variable
            # 資金リセット問題解決のため,日次作成ではなく__init__で一度だけ作成
            # これにより,PaperBrokerの資金が日次でリセットされなくなる
            self.main_controller = None  # 遅延初期化(初回_execute_multi_strategies呼び出し時)
            
            # 累積期間方式用設定(December 6, 2025追加)
            self.warmup_days = 150  # Option A-2: 150暦日 × 68.5% ≒ 103営業日
            
            # Phase 3-C Day 12: MarketAnalyzer追加
            try:
                from main_system.market_analysis.market_analyzer import MarketAnalyzer
                self.market_analyzer = MarketAnalyzer()
                self.logger.info("MarketAnalyzer初期化成功")
            except ImportError as e:
                self.logger.warning(f"MarketAnalyzer初期化失敗: {e}, 簡易版を使用")
                self.market_analyzer = None
            
            # Phase 3-C Day 12: DynamicStrategySelector追加
            try:
                from main_system.strategy_selection.dynamic_strategy_selector import (
                    DynamicStrategySelector, StrategySelectionMode
                )
                self.strategy_selector = DynamicStrategySelector(
                    selection_mode=StrategySelectionMode.SINGLE_BEST,  # Phase 3-C: 単一戦略選択
                    min_confidence_threshold=0.35
                )
                self.logger.info("DynamicStrategySelector初期化成功")
            except ImportError as e:
                self.logger.warning(f"DynamicStrategySelector初期化失敗: {e}, 固定戦略を使用")
                self.strategy_selector = None
            
            # Phase 3-C Day 12: ポジション状態管理
            # Sprint 2: 複数銘柄保有対応 (2026-02-10)
            # self.current_position = None  # Sprint 2削除: positionsに統合
            # 削除理由: 複数銘柄保有対応により、単一ポジション管理は不要
            # 代替: self.positions辞書で複数ポジションを管理
            # 参照: MULTI_POSITION_IMPLEMENTATION_PLAN.md Task 2-2-1
            self.positions = {}  # {symbol: {strategy, entry_price, shares, entry_date, entry_idx}}
            self.max_positions = 4  # Sprint 2設定: 最大保有銘柄数（2026-03-01変更: 3→4）
            
            # Phase 2: equity_curve再構築用追跡変数
            self.cumulative_pnl = 0.0  # 累積損益追跡
            self.total_trades_count = 0  # 総取引数追跡
            self.previous_total_profit = 0.0  # 累積期間バックテスト重複計上対策(Phase 13)
            
            # 実行履歴
            self.daily_results = []
            self.switch_history = []
            self.strategy_statistics = {}
            
            self.performance_targets = {
                'max_daily_execution_time_ms': 1500,
                'min_success_rate': 0.95,
                'max_drawdown_limit': -0.15,
                'max_switch_cost_rate': 0.05
            }
            
            
            # Task 1実装 (2026-01-13): __init__()でコンポーネント初期化を実行
            # 理由: 完全版SymbolSwitchManagerを確実にロードするため
            self._initialize_components()
            
        except Exception as e:
            raise DSSMSIntegrationError(f"初期化失敗: {e}")

    # 直接初期化メソッド群(lazy_loader除去対応)
    def _initialize_dss_core(self):
        """DSS Core V3直接初期化"""
        if not self._dss_initialized:
            try:
                from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
                self.dss_core = DSSBacktesterV3()
                self.logger.info("DSS Core V3 直接初期化完了")
            except ImportError:
                self.dss_core = None
            self._dss_initialized = True
        return self.dss_core

    def _initialize_advanced_ranking(self):
        """AdvancedRankingEngine直接初期化"""
        if not self._ranking_initialized:
            try:
                from src.dssms.advanced_ranking_system.advanced_ranking_engine import AdvancedRankingEngine
                # Option C: 常にデフォルト設定を使用(実績あり,設定ファイル不在,実需なし)
                self.advanced_ranking_engine = AdvancedRankingEngine(None)
                self.logger.info("AdvancedRankingEngine 直接初期化完了")
            except (ImportError, Exception) as e:
                self.advanced_ranking_engine = None
                self.logger.warning(
                    f"理由: {e} | "
                )
            self._ranking_initialized = True
        return self.advanced_ranking_engine

    def _initialize_risk_management(self):
        """RiskManagement直接初期化"""
        if not self._risk_initialized:
            try:
                from config.risk_management import RiskManagement
                initial_capital = self.config.get('initial_capital', 1000000)
                self.risk_manager = RiskManagement(total_assets=initial_capital)
                self.logger.info("リスク管理システム 直接初期化完了")
            except (ImportError, Exception) as e:
                self.risk_manager = None
                self.logger.warning(f"リスク管理システム初期化失敗: {e} - デフォルト設定使用")
                self.risk_manager = None
                self.logger.warning("リスク管理システム使用不可 - デフォルト設定使用")
            self._risk_initialized = True
        return self.risk_manager

    def ensure_dss_core(self):
        """DSS Core確保(パブリックアクセス用)"""
        return self._initialize_dss_core()

    def ensure_advanced_ranking(self):
        """AdvancedRanking確保(パブリックアクセス用)"""
        return self._initialize_advanced_ranking()

    def ensure_risk_management(self):
        """RiskManagement確保(パブリックアクセス用)"""
        return self._initialize_risk_management()

    def _setup_file_logging(self, output_dir: str = None, run_id: str = None):
        '''
        詳細ログ出力を設定
        
        戦略選択・市場分析・FIFO決済の詳細ログをファイル出力します。
        
        Args:
            output_dir: ログ出力ディレクトリ(省略時はデフォルト)
            run_id: 実行ID(省略時は自動生成)
        
        Returns:
            None
        '''
        if self._file_logging_initialized:
            return
        
        try:
            # 2026-02-15改善: 詳細戦略ログ機能を使用
            from config.logger_config import setup_detailed_strategy_logger
            
            # output_dirが指定されていない場合はデフォルト
            if output_dir is None:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = f"output/dssms_integration/dssms_{timestamp}"
            
            # run_idが指定されていない場合は出力ディレクトリから抽出
            if run_id is None:
                import os
                run_id = os.path.basename(output_dir).replace('dssms_', '')
            
            # 詳細ログ設定(戦略選択、市場分析、FIFO決済を自動ファイル出力)
            self.logger = setup_detailed_strategy_logger(
                "DSSMS_Integrated",
                output_dir,
                run_id=run_id,
                enable_console=True
            )
            
            self._file_logging_initialized = True
            self.logger.info("詳細戦略ログ初期化完了（戦略選択・市場分析・FIFO決済ログ有効）")
            
        except Exception as e:
            # フォールバック: 従来のEnhanced Logger Manager使用
            self.logger.warning(f"詳細戦略ログ初期化失敗: {e}, Enhanced Logger Managerにフォールバック")
            try:
                from src.utils.logger_setup import get_logger_manager
                logger_manager = get_logger_manager()
                self.logger = logger_manager.get_strategy_logger("DSSMS_Integrated")
                self._file_logging_initialized = True
                self.logger.info("Enhanced Logger Manager初期化完了（ログローテーション・圧縮機能有効）")
            except Exception as e2:
                # 最終フォールバック: 基本ロガー
                self.logger.warning(f"Enhanced Logger Manager初期化も失敗: {e2}, 基本ロガー使用")
                self._file_logging_initialized = True

    def _initialize_components(self):
        if not self._components_initialized:
            try:
                # SymbolSwitchManager直接初期化(軽量版優先)
                switch_config = self.config.get('symbol_switch', {})
                self.switch_manager = SymbolSwitchManager(switch_config)
                self.logger.info(f"[OK] SymbolSwitchManager初期化完了: {type(self.switch_manager).__name__}")
                
                self._initialize_data_cache()
                self._initialize_performance_tracker()
                self._initialize_report_generator()
                self._initialize_nikkei225_screener()
                
                self._components_initialized = True
            except Exception as e:
                # Component initialization error handling - CRITICAL: DO NOT HIDE EXCEPTIONS
                self.logger.error(f"Component initialization failed: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                # Partial initialization is acceptable for fallback functionality
                self._components_initialized = True

    def _initialize_data_cache(self):
        try:
            from src.dssms.data_cache_manager import DataCacheManager
            cache_config = self.config.get('data_cache', {})
            self.data_cache = DataCacheManager(cache_config)
        except ImportError:
            self.data_cache = None

    def _initialize_performance_tracker(self):
        try:
            from src.dssms.performance_tracker import PerformanceTracker
            self.performance_tracker = PerformanceTracker()
        except ImportError:
            self.performance_tracker = None

    def _initialize_report_generator(self):
        try:
            from src.dssms.dssms_report_generator import DSSMSReportGenerator
            report_config = self.config.get('report_settings', {})
            self.report_generator = DSSMSReportGenerator(report_config)
        except ImportError:
            self.report_generator = None

    def _initialize_nikkei225_screener(self):
        try:
            from src.dssms.nikkei225_screener import Nikkei225Screener
            self.nikkei225_screener = Nikkei225Screener()
            self.logger.info("Nikkei225Screener直接初期化完了")
        except (ImportError, Exception) as e:
            self.nikkei225_screener = None
            self.logger.warning(f"Nikkei225Screener初期化失敗: {e}")

    def ensure_components(self):
        if not self._components_initialized:
            self._initialize_components()
        return self.switch_manager
    
    def _warm_fundamental_cache(self, symbols: List[str]):
        """
        Fundamental cache warming optimization:
        Sequential approach (60 seconds) -> Parallel approach (20 seconds with 3 threads)
        
        Args:
            symbols: Target symbol list for cache warming
        """
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            # FundamentalAnalyzerアクセス
            if not hasattr(self.advanced_ranking_engine, '_hierarchical_system'):
                return
            
            hierarchical_system = self.advanced_ranking_engine._hierarchical_system
            if not hasattr(hierarchical_system, 'fundamental_analyzer'):
                return
            
            fundamental_analyzer = hierarchical_system.fundamental_analyzer
            
            warm_start = time.time()
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(fundamental_analyzer.fetch_financial_data, symbol): symbol for symbol in symbols}
                
                # 完了待機
                for future in futures:
                    try:
                        future.result(timeout=30)  # 30秒タイムアウト
                    except Exception as e:
                        symbol = futures[future]
            
            warm_time = time.time() - warm_start
            
        except Exception as e:
            # Cache warming error handling
            pass
    
    def _normalize_stock_data_structure(self, stock_data: 'pd.DataFrame', symbol: str) -> 'pd.DataFrame':
        """
        株価データ構造の正規化（Phase 3-B Step B1）
        
        yfinanceのMultiIndex構造をフラットなDataFrameに変換し、
        VWAPBreakoutStrategy等の戦略が期待する形式に統一する。
        
        Args:
            stock_data (pd.DataFrame): 元データ（MultiIndex可能性あり）
            symbol (str): 銘柄コード
            
        Returns:
            pd.DataFrame: 正規化されたデータ
                - Columns: ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                - Index: DatetimeIndex
                
        copilot-instructions.md遵守:
        - 実データのみ使用（フォールバック禁止）
        - データ品質保証
        
        Author: Backtest Project Team
        Created: 2025-12-31
        """
        import pandas as pd
        
        # データのコピーを作成（元データへの影響防止）
        normalized_data = stock_data.copy()
        
        self.logger.debug(f"[NORMALIZE] 入力データ構造: columns={list(stock_data.columns)}, shape={stock_data.shape}")
        
        # MultiIndex構造のフラット化
        if isinstance(normalized_data.columns, pd.MultiIndex):
            self.logger.info(f"[NORMALIZE] MultiIndex検出 → フラット化実行")
            
            # MultiIndex構造の判定: (銘柄, カラム名) or (カラム名, 銘柄)
            first_col = normalized_data.columns[0]
            if isinstance(first_col, tuple) and len(first_col) >= 2:
                # サンプルチェック: どちらがカラム名か判定
                sample_col_0 = str(first_col[0])
                sample_col_1 = str(first_col[1])
                
                # 標準カラム名リスト
                standard_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                
                # カラム名位置の判定
                if sample_col_0 in standard_columns:
                    # ('Open', '9101.T') → 'Open'
                    flat_columns = [col[0] for col in normalized_data.columns]
                    self.logger.debug(f"[NORMALIZE] MultiIndex構造: (カラム名, 銘柄)")
                elif sample_col_1 in standard_columns:
                    # ('9101.T', 'Open') → 'Open'
                    flat_columns = [col[1] for col in normalized_data.columns]
                    self.logger.debug(f"[NORMALIZE] MultiIndex構造: (銘柄, カラム名)")
                else:
                    # どちらも標準カラム名ではない場合は最後の要素を使用
                    flat_columns = [col[-1] for col in normalized_data.columns]
                    self.logger.warning(f"[NORMALIZE] 非標準MultiIndex構造: 最後の要素を使用")
            else:
                # タプルではない場合（念のため）
                flat_columns = list(normalized_data.columns)
            
            normalized_data.columns = flat_columns
            self.logger.info(f"[NORMALIZE] フラット化完了: columns={list(normalized_data.columns)}")
        
        # 必須カラムの存在確認
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in normalized_data.columns]
        
        if missing_columns:
            self.logger.warning(f"[NORMALIZE] 不足カラム検出: {missing_columns}")
            
            # 'Adj Close'がなく'Close'がある場合は補完（copilot-instructions.md: 限定的フォールバック）
            if 'Adj Close' in missing_columns and 'Close' in normalized_data.columns:
                normalized_data['Adj Close'] = normalized_data['Close']
                self.logger.info(f"[NORMALIZE] 'Adj Close'を'Close'で補完")
                missing_columns.remove('Adj Close')
            
            # その他のカラムが不足している場合はエラー
            if missing_columns:
                raise ValueError(f"Required columns missing: {missing_columns}. Available: {list(normalized_data.columns)}")
        
        # データ型の確認・調整
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            if col in normalized_data.columns:
                if not pd.api.types.is_numeric_dtype(normalized_data[col]):
                    self.logger.warning(f"[NORMALIZE] カラム'{col}'の型変換: {normalized_data[col].dtype} → float64")
                    try:
                        normalized_data[col] = pd.to_numeric(normalized_data[col], errors='coerce')
                    except Exception as e:
                        self.logger.error(f"[NORMALIZE] カラム'{col}'の型変換失敗: {e}")
                        raise ValueError(f"Failed to convert column '{col}' to numeric")
        
        # インデックスがDatetimeIndexか確認
        if not isinstance(normalized_data.index, pd.DatetimeIndex):
            self.logger.warning(f"[NORMALIZE] インデックスがDatetimeIndexではない: {type(normalized_data.index)}")
            try:
                normalized_data.index = pd.to_datetime(normalized_data.index)
                self.logger.info(f"[NORMALIZE] インデックスをDatetimeIndexに変換")
            except Exception as e:
                self.logger.error(f"[NORMALIZE] インデックス変換失敗: {e}")
                raise ValueError(f"Failed to convert index to DatetimeIndex: {e}")
        
        self.logger.info(f"[NORMALIZE] データ正規化完了: shape={normalized_data.shape}, columns={list(normalized_data.columns)}")
        return normalized_data
    
    def _adjust_to_business_day(self, target_date: 'datetime', stock_data: 'pd.DataFrame') -> 'datetime':
        """
        業務日への調整（Phase 3-B Step B1）
        
        target_dateが休日・土日・データ範囲外の場合、
        最寄りの業務日（株式市場営業日）に調整する。
        
        Args:
            target_date (datetime): 対象日
            stock_data (pd.DataFrame): 株価データ（範囲確認用）
            
        Returns:
            datetime: 調整後の業務日
            
        copilot-instructions.md遵守:
        - 実データ範囲内での調整（フォールバック禁止）
        - 決定論保証（同じ入力で同じ出力）
        
        Author: Backtest Project Team
        Created: 2025-12-31
        """
        import pandas as pd
        from datetime import timedelta
        
        target_timestamp = pd.Timestamp(target_date)
        
        # データ範囲内に存在する場合はそのまま返す
        if target_timestamp in stock_data.index:
            self.logger.debug(f"[ADJUST_DATE] target_dateがデータ内に存在: {target_timestamp.strftime('%Y-%m-%d')}")
            return target_timestamp.to_pydatetime()
        
        self.logger.info(f"[ADJUST_DATE] target_dateがデータ範囲外: {target_timestamp.strftime('%Y-%m-%d')}")
        
        # timezone対応: stock_data.indexのtimezoneに合わせる
        data_dates = stock_data.index
        if hasattr(data_dates, 'tz') and data_dates.tz is not None:
            # target_timestampがすでにタイムゾーンを持っている場合は変換、そうでなければローカライズ
            if target_timestamp.tz is not None:
                target_timestamp = target_timestamp.tz_convert(data_dates.tz)
            else:
                target_timestamp = target_timestamp.tz_localize(data_dates.tz)
        else:
            # データがタイムゾーンなしの場合、target_timestampもタイムゾーンなしにする
            if target_timestamp.tz is not None:
                target_timestamp = target_timestamp.tz_localize(None)
        
        # 過去方向に最大10日間検索（業務日を優先）
        for days_back in range(10):
            candidate = target_timestamp - timedelta(days=days_back)
            
            # データ範囲内かつ平日（月〜金）の条件を満たす
            if candidate in data_dates and candidate.weekday() < 5:
                adjusted_date = candidate.to_pydatetime()
                if adjusted_date.tzinfo is not None:
                    adjusted_date = adjusted_date.replace(tzinfo=None)  # timezone情報を除去
                self.logger.info(f"[ADJUST_DATE] 業務日調整成功 (過去方向): {target_timestamp.strftime('%Y-%m-%d')} → {adjusted_date.strftime('%Y-%m-%d')}")
                return adjusted_date
        
        # 未来方向に最大5日間検索（過去で見つからない場合）
        for days_forward in range(1, 6):
            candidate = target_timestamp + timedelta(days=days_forward)
            
            if candidate in data_dates and candidate.weekday() < 5:
                adjusted_date = candidate.to_pydatetime()
                if adjusted_date.tzinfo is not None:
                    adjusted_date = adjusted_date.replace(tzinfo=None)  # timezone情報を除去
                self.logger.info(f"[ADJUST_DATE] 業務日調整成功 (未来方向): {target_timestamp.strftime('%Y-%m-%d')} → {adjusted_date.strftime('%Y-%m-%d')}")
                return adjusted_date
        
        # 最後の手段: データ範囲内の最も近い日を選択（曜日不問）
        if len(data_dates) > 0:
            # 過去方向で最も近い日
            past_dates = data_dates[data_dates <= target_timestamp]
            if len(past_dates) > 0:
                closest_past = past_dates[-1].to_pydatetime()
                if closest_past.tzinfo is not None:
                    closest_past = closest_past.replace(tzinfo=None)  # timezone情報を除去
                self.logger.warning(f"[ADJUST_DATE] 最終調整 (過去): {target_timestamp.strftime('%Y-%m-%d')} → {closest_past.strftime('%Y-%m-%d')}")
                return closest_past
            
            # 未来方向で最も近い日
            future_dates = data_dates[data_dates >= target_timestamp]
            if len(future_dates) > 0:
                closest_future = future_dates[0].to_pydatetime()
                if closest_future.tzinfo is not None:
                    closest_future = closest_future.replace(tzinfo=None)  # timezone情報を除去
                self.logger.warning(f"[ADJUST_DATE] 最終調整 (未来): {target_timestamp.strftime('%Y-%m-%d')} → {closest_future.strftime('%Y-%m-%d')}")
                return closest_future
        
        # 調整不可能な場合は元の日付を返す
        self.logger.error(f"[ADJUST_DATE] 業務日調整失敗: {target_timestamp.strftime('%Y-%m-%d')}")
        adjusted_date = target_timestamp.to_pydatetime()
        if adjusted_date.tzinfo is not None:
            adjusted_date = adjusted_date.replace(tzinfo=None)  # timezone情報を除去
        return adjusted_date
    
    def run_dynamic_backtest(self, start_date: datetime, end_date: datetime,
                           target_symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        動的銘柄選択バックテスト実行
        
        Args:
            start_date: バックテスト開始日
            end_date: バックテスト終了日
        
        Returns:
            Dict[str, Any]: 統合バックテスト結果
        
        Raises:
            DSSMSIntegrationError: バックテスト実行失敗
        """
        try:
            self.logger.info(f"DSSMS動的バックテスト開始: {start_date} -> {end_date}")
            
            # 修正案A: バックテスト開始日を保存(累積期間方式用)
            self.dssms_backtest_start_date = start_date
            
            # 2026-02-15修正: run_idを動的に生成してバックテストごとに新ファイル作成
            from config.logger_config import add_detailed_handlers_to_existing_loggers
            from datetime import datetime
            current_run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            temp_output_dir = f"output/dssms_integration/backtest_{current_run_id}"
            add_detailed_handlers_to_existing_loggers(temp_output_dir, current_run_id) 

            if target_symbols:
                self._warm_fundamental_cache(target_symbols)
            
            # 実行統計
            execution_start = time.time()
            total_trading_days = 0
            successful_days = 0
            
            self.logger.info(
                f"[PERIOD_SCOPE] DSSMS期間: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} "
                f"(平日のみ処理, 土日スキップ)"
            )
            
            current_date = start_date
            
            while current_date <= end_date:
                # 平日のみ処理(土日スキップ)
                if current_date.weekday() < 5:  # 月-金
                    daily_start = time.time()
                    
                    # 日次取引処理
                    daily_result = self._process_daily_trading(current_date, target_symbols)
                    
                    # 実行時間記録
                    daily_execution_time = (time.time() - daily_start) * 1000
                    daily_result['execution_time_ms'] = daily_execution_time
                    
                    # 日次結果記録
                    self.daily_results.append(daily_result)
                    
                    self.logger.info(
                        f"[DAILY_SUMMARY] {current_date.strftime('%Y-%m-%d')}: "
                        f"symbol={daily_result.get('symbol')}, "
                        f"execution_details={len(daily_result.get('execution_details', []))}, "
                        f"success={daily_result.get('success')}"
                    )
                    
                    if self.performance_tracker:
                        self.performance_tracker.record_daily_performance(daily_result)
                    else:
                        self.logger.debug(f"日次結果記録: {daily_result.get('date')} - 収益率: {daily_result.get('daily_return', 0):.3f}%")
                    
                    # 成功判定
                    if daily_result.get('success', False):
                        successful_days += 1
                    
                    total_trading_days += 1
                    
                    if daily_execution_time > self.performance_targets['max_daily_execution_time_ms']:
                        self.logger.warning(f"実行時間超過: {daily_execution_time:.0f}ms (目標: {self.performance_targets['max_daily_execution_time_ms']}ms)")
                
                current_date += timedelta(days=1)
            
            # ============================================================
            # [重要] このコードは絶対に削除しないでください
            # ============================================================
            # 【目的】
            # バックテスト終了時に保有中のポジションを強制決済し、
            # all_transactions.csvに正確なexit_date, exit_price, pnlを記録する
            #
            # 【削除してはいけない理由】
            # 1. 未決済ポジションが残ると、最終ポートフォリオ値が確定しない
            # 2. 取引履歴（all_transactions.csv）が不完全になる
            # 3. パフォーマンス計算（総損益、勝率等）が不正確になる
            #
            # 【複数銘柄保有対応との関係】
            # 複数銘柄保有対応を実装する際も、この処理は必須です。
            # current_position（単数形）を current_positions（複数形）に変更する際も、
            # 全ポジションをループで強制決済する処理を維持してください。
            #
            # 【過去の問題】
            # Issue #2: AI（VSCode Copilot）が削除して問題発生
            # -> 復元後、このコメントを追加（2026-02-05）
            #
            # 【参照】
            # - KNOWN_ISSUES_AND_PREVENTION.md Issue #2
            # - MULTI_POSITION_IMPLEMENTATION_PLAN.md Sprint 1
            # ============================================================
            
            # ==========================================
            # 期間終了時の強制決済（Sprint 2修正: 2026-02-10）
            # ==========================================
            # 目的: バックテスト終了時に保有中の全ポジションを強制決済し、
            #       all_transactions.csvに正確なexit_dateを記録する
            # Sprint 2修正: 複数銘柄保有対応（全ポジションをループで決済）
            
            if len(self.positions) > 0:
                self.logger.info(
                    f"[FINAL_CLOSE] バックテスト終了時の強制決済開始: "
                    f"{len(self.positions)}銘柄保有中"
                )
                
                final_execution_details = []
                
                # Sprint 2: 全ポジションをループで決済
                for symbol, position_data in list(self.positions.items()):
                    try:
                        shares = position_data['shares']
                        entry_price = position_data['entry_price']
                        
                        # デバッグログ追加 (2026-02-05)
                        self.logger.info(f"\n[DEBUG_PRICE] ========== 銘柄{symbol}の強制決済 ==========")
                        self.logger.info(f"[DEBUG_PRICE] entry_price: {entry_price}, shares: {shares}")
                        
                        # 最終日の終値を取得
                        final_price = None
                        stock_data, _ = self._get_symbol_data(symbol, end_date)
                        
                        # デバッグログ追加
                        if stock_data is not None and len(stock_data) > 0:
                            self.logger.info(f"[DEBUG_PRICE] 取得後: stock_data.shape = {stock_data.shape}")
                            self.logger.info(f"[DEBUG_PRICE] 取得後: stock_data.index[-5:] = {stock_data.index[-5:].tolist()}")
                            self.logger.info(f"[DEBUG_PRICE] 取得後: stock_data['Close'].iloc[-5:] = {stock_data['Close'].iloc[-5:].tolist()}")
                        else:
                            self.logger.info(f"[DEBUG_PRICE] 取得後: stock_data取得失敗(None or empty)")
                        
                        if stock_data is None or len(stock_data) == 0:
                            # エラーログ出力（フォールバックなし）
                            self.logger.error(
                                f"[FINAL_CLOSE] データ取得失敗: {symbol} - "
                                f"yfinance APIエラーまたはデータ不足"
                            )
                            # 強制決済レコードは生成するがfinal_priceはentry_priceを使用
                            # （最悪ケース: PnL=0として決済記録）
                            final_price = entry_price
                            
                            self.logger.info(f"[DEBUG_PRICE] final_price = entry_price: {final_price}")
                            
                            self.logger.warning(
                                f"[FINAL_CLOSE] {symbol}: データ取得失敗のため "
                                f"entry_price({entry_price:.2f})をfinal_priceとして使用 (PnL=0)"
                            )
                        else:
                            final_price = stock_data['Close'].iloc[-1]
                            
                            self.logger.info(f"[DEBUG_PRICE] final_price = Close最終値: {final_price}")
                            self.logger.info(f"[DEBUG_PRICE] entry_price = {entry_price}")
                        
                        # PnL計算
                        pnl = (final_price - entry_price) * shares
                        
                        self.logger.info(f"[DEBUG_PRICE] PnL: {pnl:.2f}円")
                        self.logger.info(f"[DEBUG_PRICE] ==========================================\n")
                        
                        # 決済記録（データ取得失敗でも記録）
                        exit_detail = {
                            'timestamp': end_date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': shares,
                            'price': final_price,
                            'total_value': final_price * shares,
                            'strategy': position_data.get('strategy', 'DSSMS_Integrated'),
                            'reason': 'backtest_end',
                            'status': 'force_closed',
                            'pnl': pnl,
                            'return_pct': (final_price - entry_price) / entry_price if entry_price > 0 else 0.0
                        }
                        
                        final_execution_details.append(exit_detail)
                        
                        # cash_balance更新（DSSMSのみ）
                        self.cash_balance += final_price * shares
                        
                        # ログ出力
                        self.logger.info(
                            f"[FINAL_CLOSE] {symbol}: {shares}株 @{final_price:.2f}円, "
                            f"PnL={pnl:+,.0f}円({exit_detail['return_pct']:+.2%})"
                        )
                    
                    except Exception as e:
                        self.logger.error(
                            f"[FINAL_CLOSE] 強制決済エラー: {symbol}, {e}",
                            exc_info=True
                        )
                
                # Sprint 2: 全ポジション削除
                self.positions.clear()
                self.logger.info(f"[FINAL_CLOSE] 全ポジション削除完了")
                
                # 最終日の日次結果にexecution_detailsを追加
                if self.daily_results and final_execution_details:
                    last_daily_result = self.daily_results[-1]
                    if 'execution_details' not in last_daily_result:
                        last_daily_result['execution_details'] = []
                    last_daily_result['execution_details'].extend(final_execution_details)
                    
                    self.logger.info(
                        f"[FINAL_CLOSE] 強制決済完了: {len(final_execution_details)}件の決済記録を追加"
                    )
                
                # portfolio_valueをcash_balanceに同期（全決済後は現金のみ）
                self.portfolio_value = self.cash_balance
                self.logger.info(
                    f"[FINAL_CLOSE] portfolio_value更新: {self.portfolio_value:,.2f}円（全決済後）"
                )
            
            self.logger.info(
                f"[BACKTEST_END] 保有銘柄リスト: {list(self.positions.keys())} "
                f"(保有数: {len(self.positions)}/{self.max_positions})"
            )
            
            # ==========================================
            # 最終結果生成
            # ==========================================
            total_execution_time = time.time() - execution_start
            final_results = self._generate_final_results(total_execution_time, total_trading_days, successful_days)
            
            self._generate_outputs(final_results)
            
            self.logger.info(f"DSSMS動的バックテスト完了: {total_trading_days}日処理,{successful_days}日成功")
            return final_results
            
        except Exception as e:
            raise DSSMSIntegrationError(f"バックテスト実行失敗: {e}")
    
    def _process_daily_trading(self, target_date: datetime, 
                             target_symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        日次取引処理
        
        Args:
            target_date: 対象日付
            target_symbols: 対象銘柄リスト
        
        Returns:
            Dict[str, Any]: 日次処理結果
        """
        try:
            daily_result = {
                'date': target_date.strftime('%Y-%m-%d'),
                'symbol': None,  # Sprint 2: selected_symbolで後ほど設定
                'success': False,
                'portfolio_value_start': self.portfolio_value,
                'daily_return': 0,
                'daily_pnl': 0,  # ポートフォリオ資産曲線用
                'daily_return_rate': 0,
                'strategy_results': {},
                'switch_executed': False,
                'errors': [],
                
                # Phase 2: equity_curve再構築用カラム
                # Phase 1 Stage 4-2変更(December 19, 2025):
                'cash_balance': self.portfolio_value,
                'position_value': 0,
                
                'peak_value': self.peak_value,
                'drawdown_pct': 0,  # 後程計算
                'cumulative_pnl': self.cumulative_pnl,
                'total_trades': self.total_trades_count,
                'active_positions': 0,  # Phase 1 Stage 4-2: 0固定(main_new.pyが管理)
                'risk_status': 'Normal',
                'blocked_trades': 0,
                'risk_action': '',
                'execution_details': []  # _convert_main_new_result()で設定
            }
            
            # 1. DSS Core V3による銘柄選択
            selected_symbol = self._get_optimal_symbol(target_date, target_symbols)
            
            if not selected_symbol:
                daily_result['errors'].append('銘柄選択失敗')
                return daily_result
            
            switch_result = self._evaluate_and_execute_switch(selected_symbol, target_date)
            
            # Sprint 2追加: force_close対象銘柄がある場合、先に決済
            force_close_symbol = switch_result.get('force_close_symbol')
            if force_close_symbol:
                self.logger.warning(
                    f"[FORCE_CLOSE] {force_close_symbol}を決済: "
                    f"reason={switch_result.get('reason')}"
                )
                # force_close対象銘柄のデータ取得
                close_stock_data, _ = self._get_symbol_data(force_close_symbol, target_date)
                if close_stock_data is not None:
                    # force_closeフラグ付きで戦略実行（決済のみ）
                    if force_close_symbol in self.positions:
                        # force_close用のexisting_positionを構築
                        force_close_position = self.positions[force_close_symbol].copy()
                        force_close_position['force_close'] = True  # 強制決済フラグ
                        force_close_position['entry_symbol'] = force_close_symbol
                        self.logger.info(
                            f"[FORCE_CLOSE] force_close_position構築: "
                            f"symbol={force_close_symbol}, force_close=True, "
                            f"entry_price={force_close_position.get('entry_price', 0):.2f}"
                        )
                        
                        # 決済処理実行（force_close_position付き）
                        close_result = self._execute_multi_strategies_daily(
                            target_date, 
                            force_close_symbol, 
                            close_stock_data,
                            force_close_position=force_close_position
                        )
                        
                        # 決済結果をdaily_resultに記録
                        if 'execution_details' in close_result:
                            if 'execution_details' not in daily_result:
                                daily_result['execution_details'] = []
                            daily_result['execution_details'].extend(close_result['execution_details'])
                        
                        # 決済完了確認（防御的チェック）
                        if force_close_symbol not in self.positions:
                            self.logger.info(
                                f"[FIFO_EXIT] FIFO決済成功: {force_close_symbol} "
                                f"削除完了（残り保有数: {len(self.positions)}/{self.max_positions}）"
                            )
                        else:
                            # 戦略がSELLを返さなかった場合: 直接ポジション削除（フォールバック）
                            self.logger.warning(
                                f"[FORCE_CLOSE_FALLBACK] 戦略が決済を返さなかったため直接削除: "
                                f"{force_close_symbol}"
                            )
                            # 当日終値で強制決済
                            try:
                                close_price = close_stock_data['Adj Close'].iloc[-1]
                                position = self.positions[force_close_symbol]
                                shares = position.get('shares', 0)
                                entry_price = position.get('entry_price', 0)
                                pnl = (close_price - entry_price) * shares
                                
                                # 残高更新
                                self.cash_balance += close_price * shares
                                
                                # execution_details記録
                                force_detail = {
                                    'timestamp': target_date.strftime('%Y-%m-%d %H:%M:%S'),
                                    'symbol': force_close_symbol,
                                    'action': 'SELL',
                                    'price': close_price,
                                    'shares': shares,
                                    'strategy': position.get('strategy', 'FIFO_FORCE_CLOSE'),
                                    'signal_strength': -1,
                                    'reason': 'FIFO強制決済（max_positions到達）',
                                    'status': 'executed'
                                }
                                if 'execution_details' not in daily_result:
                                    daily_result['execution_details'] = []
                                daily_result['execution_details'].append(force_detail)
                                
                                # ポジション削除
                                del self.positions[force_close_symbol]
                                self.logger.info(
                                    f"[FIFO_DETAIL] 直接決済完了: "
                                    f"{force_close_symbol} {shares}株 @{close_price:.2f}円, "
                                    f"PnL={pnl:+,.0f}円, "
                                    f"残り保有数: {len(self.positions)}/{self.max_positions}"
                                )
                            except Exception as e:
                                self.logger.error(
                                    f"[FORCE_CLOSE_FALLBACK_ERROR] 直接決済失敗: "
                                    f"{force_close_symbol}: {e}"
                                )
                    else:
                        self.logger.warning(
                            f"[FORCE_CLOSE_ERROR] {force_close_symbol}のポジションが存在しません"
                        )
                else:
                    self.logger.error(
                        f"[FORCE_CLOSE_ERROR] {force_close_symbol}のデータ取得失敗: "
                        f"date={target_date.strftime('%Y-%m-%d')}"
                    )
            
            # Sprint 2修正: switch_executedの処理（簡素化）
            if switch_result.get('switch_executed', False):
                # 銘柄切替が実行された場合
                daily_result['switch_executed'] = True
                daily_result['symbol'] = selected_symbol  # Sprint 2: 新銘柄を設定
                self.switch_history.append(switch_result)
                self.logger.info(
                    f"[SWITCH] 銘柄切替実行: {switch_result.get('from_symbol')} -> {selected_symbol}"
                )
            
            # ============================================================
            # 保有中銘柄のエグジットチェック（毎日・全保有銘柄を評価）
            # 修正 (2026-02-20): selected_symbolの有無に関わらず毎日実行
            # 理由: selected_symbol=Noneでも既存ポジションのstop loss等を確認する必要がある
            # ============================================================
            for held_symbol in list(self.positions.keys()):
                held_stock_data, _ = self._get_symbol_data(held_symbol, target_date)
                if held_stock_data is None or held_stock_data.empty:
                    self.logger.warning(
                        f"[EXIT_CHECK] {held_symbol}: データ取得失敗のためエグジットチェックをスキップ"
                    )
                    continue
                
                # エグジットチェックのみ実行（エントリーは行わない）
                # force_close_position=Noneで通常のエグジット条件（ストップロス等）を評価させる
                exit_check_result = self._execute_multi_strategies_daily(
                    target_date,
                    held_symbol,
                    held_stock_data,
                    force_close_position=None
                )
                
                self.logger.info(
                    f"[EXIT_CHECK] {held_symbol}: action={exit_check_result.get('action', 'hold')}"
                )
                
                # 修正 (2026-02-20): EXIT_CHECKの結果をdaily_resultに記録
                # 理由: エグジット判定でSELLが返されても、execution_detailsに記録されず未完了取引になっていた
                if 'execution_details' in exit_check_result and exit_check_result['execution_details']:
                    if 'execution_details' not in daily_result:
                        daily_result['execution_details'] = []
                    daily_result['execution_details'].extend(exit_check_result['execution_details'])
                    self.logger.info(
                        f"[EXIT_CHECK_RECORD] {held_symbol}: {len(exit_check_result['execution_details'])}件のexecution_detailsを記録"
                    )
            # ============================================================
            # 保有中銘柄エグジットチェック ここまで
            # ============================================================
            
            # 3. 選択銘柄でのマルチ戦略実行
            # Sprint 2: 新規エントリーまたは既存ポジション継続
            if selected_symbol:
                print(f"DEBUG positions={list(self.positions.keys())} selected={selected_symbol} date={target_date.strftime('%Y-%m-%d')}", flush=True)
                
                # Sprint 2修正: BUY実行前にmax_positionsチェック（防御的）
                # 選択銘柄が未保有の場合のみチェック（保有中なら継続判定なのでOK）
                if selected_symbol not in self.positions and len(self.positions) >= self.max_positions:
                    self.logger.warning(
                        f"[SAFETY_CHECK] BUY実行前: "
                        f"len={len(self.positions)} >= max={self.max_positions}, "
                        f"selected_symbol={selected_symbol} は未保有 → BUYスキップ"
                    )
                    # max_positions到達でBUY不可: この日の戦略実行をスキップ
                    strategy_result = {
                        'status': 'skipped',
                        'reason': f'max_positions reached ({len(self.positions)}/{self.max_positions})',
                        'symbol': selected_symbol,
                        'action': 'hold',
                        'signal': 0,
                        'price': 0.0,
                        'shares': 0
                    }
                    daily_result['strategy_results'] = strategy_result
                    # 以下のstrategy実行をスキップ
                else:
                    # Sprint 2: 選択銘柄でのマルチ戦略実行
                    self.logger.debug(
                        f"[STRATEGY_EXEC] {selected_symbol}でマルチ戦略実行: "
                        f"date={target_date.strftime('%Y-%m-%d')}, "
                        f"switch_executed={switch_result.get('switch_executed', False)}"
                    )
                    
                    # selected_symbolのデータ取得
                    stock_data, _ = self._get_symbol_data(selected_symbol, target_date)
                    if stock_data is not None:
                        # 確実に日次取引モード使用（重複エントリー防止）
                        strategy_result = self._execute_multi_strategies_daily(
                            target_date,
                            selected_symbol,
                            stock_data
                        )
                    else:
                        self.logger.error(
                            f"[DAILY_TRADING] データ取得失敗: symbol={selected_symbol}, "
                            f"date={target_date.strftime('%Y-%m-%d')}"
                        )
                        strategy_result = {
                            'status': 'error',
                            'reason': 'Data unavailable',
                            'symbol': selected_symbol,
                            'target_date': target_date
                        }
                daily_result['strategy_results'] = strategy_result
                
                # Cycle 10-6: cash_balance/position_valueをdaily_resultに引き継ぎ
                if 'cash_balance' in strategy_result:
                    daily_result['cash_balance'] = strategy_result['cash_balance']
                if 'position_value' in strategy_result:
                    daily_result['position_value'] = strategy_result['position_value']
                if 'total_portfolio_value' in strategy_result:
                    daily_result['total_portfolio_value'] = strategy_result['total_portfolio_value']
                
                # Phase 2優先度3: execution_details設定(詳細設計書3.1.3準拠)
                # December 9, 2025修正: 代入(上書き)からextend(追加)に変更
                # 理由: Line 546で追加した銘柄切替のSELL注文が消失する問題を修正
                if 'execution_details' in strategy_result:
                    if 'execution_details' not in daily_result:
                        daily_result['execution_details'] = []
                    if isinstance(strategy_result['execution_details'], list):
                        daily_result['execution_details'].extend(strategy_result['execution_details'])
                    else:
                        self.logger.warning(
                            f"[EXEC_DETAILS_TYPE_ERROR] strategy_result['execution_details']がリストではありません: "
                            f"type={type(strategy_result['execution_details'])}"
                        )
            
            if strategy_result.get('position_update'):
                position_return = strategy_result['position_update']['return']
                portfolio_value_before = self.portfolio_value
                self.portfolio_value += position_return
                daily_result['daily_return'] = position_return
                daily_result['daily_pnl'] = position_return  # ポートフォリオ資産曲線用
                daily_result['daily_return_rate'] = position_return / daily_result['portfolio_value_start']
                
                self.cumulative_pnl += position_return
                daily_result['cumulative_pnl'] = self.cumulative_pnl
                
                # Cycle 10-7修正: total_portfolio_valueをself.portfolio_valueに反映
                # 理由: cash_balance + position_valueの合計値を正確に追跡
                # 修正日: 2026-02-05
                # 修正者: Sprint 1 Task 1-10
                if 'total_portfolio_value' in strategy_result:
                    self.portfolio_value = strategy_result['total_portfolio_value']
                    self.logger.debug(
                        f"[PORTFOLIO_SYNC] {target_date.strftime('%Y-%m-%d')}: "
                        f"self.portfolio_value更新 -> {self.portfolio_value:,.0f}円 "
                        f"(cash={strategy_result.get('cash_balance', 0):,.0f}円 + "
                        f"position={strategy_result.get('position_value', 0):,.0f}円)"
                    )
            
            # 修正理由: switch実行日に取引がない場合でも,switch後のportfolio_valueを反映するため
            # switch処理でself.portfolio_valueが変更される可能性があるため,if position_update外で実行
            # Task 6調査で発見された問題の修正(December 8, 2025)
            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value
            
            daily_result['peak_value'] = self.peak_value
            daily_result['drawdown_pct'] = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0
            
            if strategy_result.get('position_update'):
                self.logger.info(
                    f"[PORTFOLIO_UPDATE] {target_date.strftime('%Y-%m-%d')} - "
                    f"Before: {portfolio_value_before:,.0f}円, "
                    f"Return: {position_return:+,.0f}円 ({daily_result['daily_return_rate']:+.2%}), "
                    f"After: {self.portfolio_value:,.0f}円, "
                    f"Drawdown: {daily_result['drawdown_pct']:.2%}"
                )
            
            # Cycle 10-5: Phase 1 Stage 4-2のcash_balance上書き処理を削除
            # 削除理由: unified_resultで設定したcash_balance/position_valueを保持
            # Cycle 10目標: ポジション価値の正確な評価、未決済ポジションの時価評価
            # 既存コメント（削除済み）:
            #   Phase 1 Stage 4-2変更(December 19, 2025):
            #   - cash_balance: portfolio_value全額(キャッシュ扱い)
            #   daily_result['cash_balance'] = self.portfolio_value
            #   daily_result['position_value'] = 0
            
            # Cycle 10-5: unified_resultからcash_balance/position_valueを維持
            # 既にstrategy_result（unified_result）に設定済み
            if 'cash_balance' not in daily_result:
                daily_result['cash_balance'] = self.cash_balance
            if 'position_value' not in daily_result:
                daily_result['position_value'] = 0
            
            self.logger.debug(
                f"[CASH_POSITION_CALC] {target_date.strftime('%Y-%m-%d')}: "
                f"cash={daily_result.get('cash_balance', 0):,.0f}円, "
                f"position={daily_result.get('position_value', 0):,.0f}円"
            )
            
            # 4. リスク管理チェック
            risk_result = self._check_risk_limits(daily_result)
            
            if risk_result.get('risk_violation'):
                daily_result['errors'].append(f"リスク制限違反: {risk_result['violation_type']}")
                self._handle_risk_violation(risk_result)
            
            # 最終結果設定
            daily_result['portfolio_value_end'] = self.portfolio_value
            
            expected_portfolio = daily_result.get('cash_balance', 0) + daily_result.get('position_value', 0)
            actual_portfolio = self.portfolio_value
            if abs(expected_portfolio - actual_portfolio) > 0.01:
                self.logger.warning(
                    f"[PORTFOLIO_MISMATCH] {target_date.strftime('%Y-%m-%d')}: "
                    f"expected={expected_portfolio:.2f}, actual={actual_portfolio:.2f}, "
                    f"diff={expected_portfolio - actual_portfolio:.2f}, "
                    f"cash={daily_result.get('cash_balance', 0):.2f}, "
                    f"position={daily_result.get('position_value', 0):.2f}"
                )
            
            daily_result['success'] = len(daily_result['errors']) == 0
            
            return daily_result
            
        except Exception as e:
            self.logger.error(f"[DAILY_TRADING_ERROR] {e}", exc_info=True)
            return {
                'date': target_date.strftime('%Y-%m-%d'),
                'symbol': None,  # Sprint 2: 複数銘柄保有対応
                'success': False,
                'portfolio_value_start': self.portfolio_value,
                'portfolio_value_end': self.portfolio_value,
                'daily_return': 0,
                'daily_return_rate': 0
            }
    
    def _advanced_ranking_selection(self, filtered_symbols: List[str], target_date: datetime) -> str:
        """
        TODO-DSSMS-004.2統合最適化適用版
        
        Args:
            filtered_symbols: フィルタ済み銘柄リスト
            target_date: 対象日付
            
        Returns:
        """
        if self.advanced_ranking_engine and len(filtered_symbols) > 0:
            try:
                # TODO-DSSMS-004.2: 統合最適化実装
                selected_symbol = self._integrated_ranking_selection_optimized(
                    filtered_symbols, target_date
                )
                
                if selected_symbol:
                    self.logger.info(
                        f"{len(filtered_symbols)}銘柄最適選択)"
                    )
                    return selected_symbol
                
                return self._legacy_advanced_ranking_selection(filtered_symbols, target_date)
                
            except Exception as e:
                return self._legacy_advanced_ranking_selection(filtered_symbols, target_date)
        
        raise RuntimeError(
            f"Failed to select optimal symbol from {len(filtered_symbols)} candidates. "
            f"All ranking methods failed. "
            f"Random selection fallback is prohibited by copilot-instructions.md. "
            f"Cannot proceed with symbol selection."
        )
    
    def _integrated_ranking_selection_optimized(self, filtered_symbols: List[str], target_date: datetime) -> Optional[str]:
        """
        
        
        Args:
            filtered_symbols: フィルタ済み銘柄リスト
            target_date: 対象日付
            
        Returns:
        """
        start_time = time.time()
        
        try:
            
            # Step 1: HierarchicalRankingSystem基盤計算(重複排除の基準)
            hierarchical_results = self._get_hierarchical_ranking_base(filtered_symbols)
            
            if not hierarchical_results:
                self.logger.warning("HierarchicalRankingSystem基盤計算失敗")
                return None
            
            # Step 2: AdvancedRankingEngine高度分析(基盤結果再利用)
            advanced_results = self._run_advanced_analysis_with_base_results(
                filtered_symbols, target_date, hierarchical_results
            )
            
            final_selection = self._calculate_integrated_optimal_selection(
                hierarchical_results, advanced_results
            )
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(f"[OK] 統合最適化完了: {execution_time:.2f}ms, 選択: {final_selection}")
            
            return final_selection
            
        except Exception as e:
            return None
    
    def _get_hierarchical_ranking_base(self, filtered_symbols: List[str]) -> Optional[Dict[str, Any]]:
        """
        HierarchicalRankingSystem基盤計算取得(キャッシュ活用)
        
        Args:
            filtered_symbols: 対象銘柄リスト
        
        Returns:
        """
        try:
            # HierarchicalRankingSystemアクセス
            if hasattr(self.advanced_ranking_engine, '_hierarchical_system') and \
               self.advanced_ranking_engine._hierarchical_system:
                
                hierarchical_system = self.advanced_ranking_engine._hierarchical_system
                
                # 優先度分類(基盤計算)
                priority_groups = hierarchical_system.categorize_by_perfect_order_priority(filtered_symbols)
                
                ranking_results = {}
                for priority_level, group_symbols in priority_groups.items():
                    if group_symbols:
                        group_ranking = hierarchical_system.rank_within_priority_group(group_symbols)
                        ranking_results[priority_level] = group_ranking
                
                base_results = {
                    'priority_groups': priority_groups,
                    'ranking_results': ranking_results,
                    'timestamp': time.time(),
                    'symbols_count': len(filtered_symbols)
                }
                
                return base_results
            
            return None
            
        except Exception as e:
            return None
    
    def _run_advanced_analysis_with_base_results(self, filtered_symbols: List[str], 
                                               target_date: datetime,
                                               hierarchical_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        AdvancedRankingEngine高度分析(基盤結果再利用版)
        
        Args:
            filtered_symbols: 対象銘柄リスト
            target_date: 対象日付
            hierarchical_results: HierarchicalRankingSystem基盤結果
            
        Returns:
            Optional[Dict[str, Any]]: 高度分析結果
        """
        try:
            market_data = self._prepare_market_data_for_analysis(filtered_symbols, target_date)
            
            if not market_data:
                return {'base_results_only': True, 'hierarchical_results': hierarchical_results}
            
            analysis_params = {
                'analysis_depth': 'comprehensive',
                'enable_parallel': len(filtered_symbols) > 5,
                'timeout_seconds': 20,
                'base_ranking_results': hierarchical_results,  # 基盤結果統合
                'reuse_calculations': True  # 重複計算回避
            }
            
            # 統合分析実行
            ranking_results = self._run_advanced_analysis_sync(filtered_symbols, market_data, analysis_params)
            
            if ranking_results:
                advanced_results = {
                    'advanced_ranking': ranking_results,
                    'market_data_available': True,
                    'base_integration': True,
                    'timestamp': time.time()
                }
                
                self.logger.info(f"[FIRE] 高度分析完了: {len(ranking_results)}結果 (基盤統合)")
                return advanced_results
            
            return None
            
        except Exception as e:
            return None
    
    def _calculate_integrated_optimal_selection(self, hierarchical_results: Dict[str, Any],
                                              advanced_results: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        TODO-DSSMS-004.2 Stage 3: 高度分析機能統合実装
        
        Args:
            hierarchical_results: HierarchicalRankingSystem基盤結果
            advanced_results: AdvancedRankingEngine高度分析結果
            
        Returns:
            Optional[str]: 統合最適選択銘柄
        """
        try:
            base_candidates = []
            ranking_results = hierarchical_results.get('ranking_results', {})
            
            # 優先度順に候補収集
            for priority_level in [1, 2, 3]:
                group_ranking = ranking_results.get(priority_level, [])
                if group_ranking:
                    # トップ候補を基盤候補として追加
                    top_candidate = group_ranking[0]  # (symbol, score) tuple
                    base_candidates.append({
                        'symbol': top_candidate[0],
                        'base_score': top_candidate[1],
                        'priority_level': priority_level
                    })
            
            if not base_candidates:
                self.logger.warning("統合選択: 基盤候補なし")
                return None
            
            # Stage 3: 高度分析機能統合強化
            enhanced_candidates = self._enhance_candidates_with_advanced_analysis(
                base_candidates, advanced_results
            )
            
            # 統合スコア最高の銘柄選択
            final_selection = self._select_optimal_from_enhanced_candidates(enhanced_candidates)
            
            return final_selection
            
        except Exception as e:
            return None
    
    def _enhance_candidates_with_advanced_analysis(self, base_candidates: List[Dict[str, Any]],
                                                 advanced_results: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 3: 高度分析機能による候補強化
        
        Args:
            base_candidates: 基盤候補リスト
            advanced_results: AdvancedRankingEngine高度分析結果
            
        Returns:
            List[Dict[str, Any]]: 強化済み候補リスト
        """
        enhanced_candidates = []
        
        for base_candidate in base_candidates:
            symbol = base_candidate['symbol']
            
            enhanced = base_candidate.copy()
            
            try:
                # Stage 3-1: テクニカル分析機能強化
                technical_analysis = self._get_enhanced_technical_analysis(symbol)
                enhanced.update(technical_analysis)
                
                fundamental_analysis = self._get_enhanced_fundamental_analysis(symbol)
                enhanced.update(fundamental_analysis)
                
                # Stage 3-3: MultiTimeframePerfectOrder高度判定統合
                perfect_order_analysis = self._get_enhanced_perfect_order_analysis(symbol)
                enhanced.update(perfect_order_analysis)
                
                # Stage 3-4: 高度分析結果統合
                if advanced_results and not advanced_results.get('base_results_only', False):
                    advanced_integration = self._integrate_advanced_ranking_results(
                        symbol, advanced_results
                    )
                    enhanced.update(advanced_integration)
                
                composite_score = self._calculate_composite_score_optimized(enhanced)
                enhanced['composite_score'] = composite_score
                
                enhanced_candidates.append(enhanced)
                
            except Exception as e:
                enhanced['composite_score'] = base_candidate.get('base_score', 0.0)
                enhanced_candidates.append(enhanced)
        
        self.logger.info(f"[FIRE] 高度分析強化完了: {len(enhanced_candidates)}候補")
        return enhanced_candidates
    
    def _get_enhanced_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Stage 3-1: テクニカル分析機能のフル活用実装
        
        Args:
            
        Returns:
            Dict[str, Any]: 強化テクニカル分析結果
        """
        technical_data = {
            'rsi_14': 0.0,
            'macd_signal': 'neutral',
            'bollinger_position': 0.5,
            'volume_trend': 'stable',
            'momentum_score': 0.0,
            'volatility_score': 0.0,
            'technical_strength': 0.0
        }
        
        try:
            # PerfectOrderDetectorの結果活用
            if hasattr(self.advanced_ranking_engine, '_hierarchical_system'):
                hierarchical_system = self.advanced_ranking_engine._hierarchical_system
                if hasattr(hierarchical_system, 'perfect_order_detector'):
                    perfect_detector = hierarchical_system.perfect_order_detector
                    
                    # MultiTimeframePerfectOrder取得
                    perfect_result = perfect_detector.detect_perfect_order_multi_timeframes(symbol, {})
                    
                    if perfect_result:
                        technical_data['perfect_order_daily'] = perfect_result.daily_result.is_perfect_order
                        technical_data['perfect_order_strength'] = perfect_result.daily_result.strength_score
                        technical_data['trend_duration'] = perfect_result.daily_result.trend_duration_days
                        technical_data['technical_strength'] = perfect_result.daily_result.strength_score * 0.7
            
            # テクニカル指標統合スコア
            technical_data['technical_strength'] = max(
                technical_data.get('technical_strength', 0.0),
                (technical_data['rsi_14'] / 100 + technical_data['bollinger_position']) / 2
            )
            
        except Exception as e:
            # Error handling
            pass
        
        return technical_data
    
    def _get_enhanced_fundamental_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        
        Args:
            
        Returns:
        """
        fundamental_data = {
            'per_ratio': 0.0,
            'pbr_ratio': 0.0,
            'roe_percent': 0.0,
            'operating_margin': 0.0,
            'revenue_growth': 0.0,
            'profit_stability': 0.0,
            'fundamental_score': 0.0
        }
        
        try:
            # FundamentalAnalyzerの結果活用
            if hasattr(self.advanced_ranking_engine, '_hierarchical_system'):
                hierarchical_system = self.advanced_ranking_engine._hierarchical_system
                if hasattr(hierarchical_system, 'fundamental_analyzer'):
                    fundamental_analyzer = hierarchical_system.fundamental_analyzer
                    
                    # 業績分析実行
                    analysis_result = fundamental_analyzer.analyze_financial_performance(symbol)
                    
                    if analysis_result and analysis_result.get('status') == 'success':
                        data = analysis_result.get('data', {})
                        
                        # 主要指標抽出
                        fundamental_data['operating_margin'] = data.get('operating_margin', 0.0)
                        fundamental_data['revenue_growth'] = data.get('revenue_growth_rate', 0.0)
                        fundamental_data['profit_stability'] = data.get('earnings_stability', 0.0)
                        
                        if 'valuation_metrics' in data:
                            valuation = data['valuation_metrics']
                            fundamental_data['per_ratio'] = valuation.get('pe_ratio', 0.0)
                            fundamental_data['pbr_ratio'] = valuation.get('pb_ratio', 0.0)
                            fundamental_data['roe_percent'] = valuation.get('roe', 0.0)
                        
                        fundamental_data['fundamental_score'] = self._calculate_fundamental_composite_score(
                            fundamental_data
                        )
            
        except Exception as e:
            # Error handling
            pass
        
        return fundamental_data
    
    def _get_enhanced_perfect_order_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Stage 3-3: MultiTimeframePerfectOrder高度判定統合
        
        Args:
            
        Returns:
        """
        perfect_order_data = {
            'perfect_order_daily': False,
            'perfect_order_weekly': False,
            'perfect_order_monthly': False,
            'multi_timeframe_strength': 0.0,
            'trend_consistency': 0.0,
            'perfect_order_composite': 0.0
        }
        
        try:
            # MultiTimeframePerfectOrder高度判定
            if hasattr(self.advanced_ranking_engine, '_hierarchical_system'):
                hierarchical_system = self.advanced_ranking_engine._hierarchical_system
                if hasattr(hierarchical_system, 'perfect_order_detector'):
                    perfect_detector = hierarchical_system.perfect_order_detector
                    
                    multi_result = perfect_detector.detect_perfect_order_multi_timeframes(symbol, {})
                    
                    if multi_result:
                        perfect_order_data['perfect_order_daily'] = multi_result.daily_result.is_perfect_order
                        
                        if hasattr(multi_result, 'weekly_result') and multi_result.weekly_result:
                            perfect_order_data['perfect_order_weekly'] = multi_result.weekly_result.is_perfect_order
                        
                        if hasattr(multi_result, 'monthly_result') and multi_result.monthly_result:
                            perfect_order_data['perfect_order_monthly'] = multi_result.monthly_result.is_perfect_order
                        
                        # 多時間軸強度計算
                        strength_scores = []
                        if multi_result.daily_result:
                            strength_scores.append(multi_result.daily_result.strength_score)
                        
                        if strength_scores:
                            perfect_order_data['multi_timeframe_strength'] = sum(strength_scores) / len(strength_scores)
                        
                        consistent_timeframes = sum([
                            perfect_order_data['perfect_order_daily'],
                            perfect_order_data['perfect_order_weekly'],
                            perfect_order_data['perfect_order_monthly']
                        ])
                        perfect_order_data['trend_consistency'] = consistent_timeframes / 3.0
                        
                        perfect_order_data['perfect_order_composite'] = (
                            perfect_order_data['multi_timeframe_strength'] * 0.6 +
                            perfect_order_data['trend_consistency'] * 0.4
                        )
            
        except Exception as e:
            # Error handling
            pass
        
        return perfect_order_data
    
    def _integrate_advanced_ranking_results(self, symbol: str, advanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3-4: 高度分析結果統合
        
        Args:
            advanced_results: AdvancedRankingEngine高度分析結果
            
        Returns:
            Dict[str, Any]: 統合高度分析結果
        """
        integration_data = {
            'advanced_total_score': 0.0,
            'advanced_ranking_position': 0,
            'multi_dimensional_score': 0.0,
            'confidence_level': 0.0
        }
        
        try:
            advanced_ranking = advanced_results.get('advanced_ranking', [])
            
            # 該当銘柄の高度分析結果を検索
            symbol_result = None
            for i, result in enumerate(advanced_ranking):
                result_symbol = None
                if hasattr(result, 'symbol'):
                    result_symbol = result.symbol
                elif isinstance(result, dict):
                    result_symbol = result.get('symbol')
                
                if result_symbol == symbol:
                    symbol_result = result
                    integration_data['advanced_ranking_position'] = i + 1
                    break
            
            if symbol_result:
                # 高度分析スコア抽出
                if hasattr(symbol_result, 'total_score'):
                    integration_data['advanced_total_score'] = symbol_result.total_score
                elif isinstance(symbol_result, dict):
                    integration_data['advanced_total_score'] = symbol_result.get('total_score', 0.0)
                
                # 多次元スコア計算
                dimension_scores = []
                if hasattr(symbol_result, 'perfect_order_score'):
                    dimension_scores.append(symbol_result.perfect_order_score)
                if hasattr(symbol_result, 'fundamental_score'):
                    dimension_scores.append(symbol_result.fundamental_score)
                if hasattr(symbol_result, 'technical_score'):
                    dimension_scores.append(symbol_result.technical_score)
                
                if dimension_scores:
                    integration_data['multi_dimensional_score'] = sum(dimension_scores) / len(dimension_scores)
                
                # 信頼度レベル計算
                total_rankings = len(advanced_ranking)
                if total_rankings > 0:
                    position_ratio = integration_data['advanced_ranking_position'] / total_rankings
                    integration_data['confidence_level'] = max(0.0, 1.0 - position_ratio)
            
        except Exception as e:
            # Error handling
            pass
        
        return integration_data
    
    def _calculate_fundamental_composite_score(self, fundamental_data: Dict[str, Any]) -> float:
        """
        
        Args:
            
        Returns:
            float: 複合スコア
        """
        try:
            scores = []
            
            # 営業利益率スコア
            operating_margin = fundamental_data.get('operating_margin', 0.0)
            if operating_margin > 0:
                margin_score = min(1.0, operating_margin / 0.1)  # 10%で満点
                scores.append(margin_score * 0.3)
            
            # 成長率スコア
            revenue_growth = fundamental_data.get('revenue_growth', 0.0)
            if revenue_growth > 0:
                growth_score = min(1.0, revenue_growth / 0.2)  # 20%で満点
                scores.append(growth_score * 0.3)
            
            # 安定性スコア
            stability = fundamental_data.get('profit_stability', 0.0)
            scores.append(stability * 0.2)
            
            # ROEスコア
            roe = fundamental_data.get('roe_percent', 0.0)
            if roe > 0:
                roe_score = min(1.0, roe / 15.0)  # 15%で満点
                scores.append(roe_score * 0.2)
            
            return sum(scores) if scores else 0.0
            
        except Exception as e:
            return 0.0
    
    def _calculate_composite_score_optimized(self, enhanced_candidate: Dict[str, Any]) -> float:
        """
        
        Args:
            
        Returns:
            float: 最適化複合スコア
        """
        try:
            # 基盤スコア
            base_score = enhanced_candidate.get('base_score', 0.0)
            priority_level = enhanced_candidate.get('priority_level', 3)
            
            # テクニカル分析スコア
            technical_strength = enhanced_candidate.get('technical_strength', 0.0)
            perfect_order_composite = enhanced_candidate.get('perfect_order_composite', 0.0)
            
            fundamental_score = enhanced_candidate.get('fundamental_score', 0.0)
            
            # 高度分析統合スコア
            advanced_total_score = enhanced_candidate.get('advanced_total_score', 0.0)
            confidence_level = enhanced_candidate.get('confidence_level', 0.0)
            
            priority_weights = {1: 0.4, 2: 0.3, 3: 0.2}
            priority_weight = priority_weights.get(priority_level, 0.1)
            
            # 複合スコア計算(最適化重み付け)
            composite_score = (
                base_score * priority_weight +                    # 基盤優先度
                technical_strength * 0.25 +                      # テクニカル強度
                advanced_total_score * 0.2                       # 高度分析
            )
            
            # スコア正規化(0-1範囲)
            normalized_score = max(0.0, min(1.0, composite_score))
            
            return normalized_score
            
        except Exception as e:
            return enhanced_candidate.get('base_score', 0.0)
    
    def _select_optimal_from_enhanced_candidates(self, enhanced_candidates: List[Dict[str, Any]]) -> Optional[str]:
        """
        強化候補から最適銘柄選択
        
        Args:
            enhanced_candidates: 強化済み候補リスト
            
        Returns:
            Optional[str]: 最適選択銘柄
        """
        try:
            if not enhanced_candidates:
                return None
            
            # 複合スコア最高の候補選択
            best_candidate = max(enhanced_candidates, key=lambda x: x.get('composite_score', 0.0))
            
            sorted_candidates = sorted(enhanced_candidates, key=lambda x: x.get('composite_score', 0.0), reverse=True)
            for idx, candidate in enumerate(sorted_candidates[:3], 1):
                self.logger.info(
                    f"[RANKING_RESULT]   {idx}位: {candidate.get('symbol', 'N/A')} "
                    f"(複合: {candidate.get('composite_score', 0.0):.4f}, "
                    f"テクニカル: {candidate.get('technical_strength', 0.0):.3f}, "
                    f"優先度: {candidate.get('priority_level', 3)})"
                )
            
            symbol = best_candidate['symbol']
            composite_score = best_candidate.get('composite_score', 0.0)
            priority_level = best_candidate.get('priority_level', 3)
            
            self.logger.info(
                f"[TARGET] 高度分析統合選択: {symbol} "
                f"(複合スコア: {composite_score:.4f}, 優先度: {priority_level})"
            )
            
            self.logger.info(
                f"  - テクニカル強度: {best_candidate.get('technical_strength', 0.0):.3f}"
            )
            self.logger.info(
            )
            self.logger.info(
                f"  - 高度分析スコア: {best_candidate.get('advanced_total_score', 0.0):.3f}"
            )
            
            return symbol
            
        except Exception as e:
            if enhanced_candidates:
                return enhanced_candidates[0].get('symbol')
            return None
    
    def _calculate_hybrid_scores(self, base_candidates: List[Dict[str, Any]], 
                               advanced_ranking: List[Any]) -> List[Dict[str, Any]]:
        """
        ハイブリッドスコア計算(基盤+高度分析統合)
        
        Args:
            base_candidates: 基盤候補リスト
            
        Returns:
            List[Dict[str, Any]]: 統合スコア計算結果
        """
        try:
            integrated_scores = []
            
            # 高度分析結果を辞書化(高速ルックアップ)
            advanced_dict = {}
            for result in advanced_ranking:
                if hasattr(result, 'symbol') and hasattr(result, 'total_score'):
                    advanced_dict[result.symbol] = result.total_score
                elif isinstance(result, dict):
                    symbol = result.get('symbol')
                    score = result.get('total_score', 0.0)
                    if symbol:
                        advanced_dict[symbol] = score
            
            # 基盤候補と高度分析の統合スコア計算
            for base_candidate in base_candidates:
                symbol = base_candidate['symbol']
                base_score = base_candidate['base_score']
                priority_level = base_candidate['priority_level']
                
                # 高度分析スコア取得
                advanced_score = advanced_dict.get(symbol, 0.0)
                
                # 統合スコア計算(重み付き)
                # 優先度が高いほど基盤スコアの重みを増加
                priority_weight = 0.8 if priority_level == 1 else 0.6 if priority_level == 2 else 0.4
                advanced_weight = 1.0 - priority_weight
                
                integrated_score = (base_score * priority_weight) + (advanced_score * advanced_weight)
                
                integrated_scores.append({
                    'symbol': symbol,
                    'base_score': base_score,
                    'advanced_score': advanced_score,
                    'priority_level': priority_level,
                    'integrated_score': integrated_score,
                    'priority_weight': priority_weight,
                    'advanced_weight': advanced_weight
                })
            
            self.logger.info(f"ハイブリッドスコア計算完了: {len(integrated_scores)}候補")
            return integrated_scores
            
        except Exception as e:
            return []
    
    def _legacy_advanced_ranking_selection(self, filtered_symbols: List[str], target_date: datetime) -> str:
        """
        
        Args:
            filtered_symbols: フィルタ済み銘柄リスト
            target_date: 対象日付
            
        Returns:
        """
        try:
            
            market_data = self._prepare_market_data_for_analysis(filtered_symbols, target_date)
            
            if market_data:
                # analyze_symbols_advanced()を同期実行
                ranking_results = self._run_advanced_analysis_sync(filtered_symbols, market_data)
                
                if ranking_results and len(ranking_results) > 0:
                    best_symbol = self._select_best_symbol_from_ranking(ranking_results)
                    
                    if best_symbol:
                        self.logger.info(
                        )
                        return best_symbol
            
            # 高度分析失敗時はシステム状態確認による暫定選択
            system_status = self.advanced_ranking_engine.get_system_status()
            if system_status.get('integration_status', {}).get('hierarchical_system', False):
                selected = filtered_symbols[0]
                self.logger.warning(
                )
                return selected
                
        except Exception as e:
            # Advanced ranking engine error - log and raise with context
            pass
        
        raise RuntimeError(
            f"AdvancedRankingEngine analysis failed for {len(filtered_symbols)} symbols. "
            f"Random selection fallback is prohibited by copilot-instructions.md. "
            f"Cannot proceed with symbol selection. "
            f"Advanced ranking available: {self.advanced_ranking_engine is not None}"
        )
    
    # _legacy_random_selection() メソッドを削除
    # 理由: copilot-instructions.md違反

    def _get_optimal_symbol(self, target_date: datetime, 
                          target_symbols: Optional[List[str]] = None) -> Optional[str]:
        """
        DSS Core V3による最適銘柄取得
        
        Args:
            target_date: 対象日付
            target_symbols: 対象銘柄リスト（固定銘柄モード時に使用）
        
        Returns:
            Optional[str]: 選択された銘柄（パーフェクトオーダーでない場合はNone）
        """
        try:
            self.ensure_components()
            self.ensure_advanced_ranking()  # AdvancedRankingEngine初期化
            self.ensure_dss_core()         # DSS Core V3初期化
            
            # 固定銘柄モード処理
            if target_symbols and len(target_symbols) == 1:
                fixed_symbol = target_symbols[0]
                self.logger.info(f"[FIXED-SYMBOL MODE] 固定銘柄モード: {fixed_symbol} @ {target_date.strftime('%Y-%m-%d')}")
                # NOTE: パーフェクトオーダーチェックは一時的に無効化（data_manager未初期化のため）
                # TODO: DSSMSDataManagerを初期化し、パーフェクトオーダーチェックを復元
                return fixed_symbol
            
            if self.dss_core and dss_available:
                # DSS Core V3による動的選択
                dss_result = self.dss_core.run_daily_selection(target_date)
                selected_symbol = dss_result.get('selected_symbol')
                
                if selected_symbol:
                    self.logger.debug(f"DSS選択結果: {selected_symbol} @ {target_date}")
                    return selected_symbol
            
            if self.nikkei225_screener:
                try:
                    self.logger.info(f"[SYMBOL_SELECTION] 銘柄選定開始: {target_date.strftime('%Y-%m-%d')}")
                    
                    available_funds = self.portfolio_value * 0.8
                    filtered_symbols = self.nikkei225_screener.get_filtered_symbols(
                        available_funds, 
                        target_date  # target_dateを渡してルックアヘッドバイアス防止
                    )
                    
                    
                    self.logger.debug(
                        f"[DEBUG] _get_optimal_symbol | filtered_symbols: {filtered_symbols[:10] if len(filtered_symbols) > 10 else filtered_symbols} "
                        f"(total: {len(filtered_symbols)})"
                    )
                    
                    if filtered_symbols:
                        if fallback_policy_available:
                            fallback_policy = get_fallback_policy()
                            selected = fallback_policy.handle_component_failure(
                                component_type=ComponentType.DSSMS_CORE,
                                component_name="DSSMSIntegratedBacktester._get_optimal_symbol",
                                error=RuntimeError("DSS Core V3 unavailable"),
                                fallback_func=lambda: self._advanced_ranking_selection(filtered_symbols, target_date),
                                context={
                                    "target_date": target_date.isoformat(),
                                    "available_symbols": len(filtered_symbols),
                                    "portfolio_value": self.portfolio_value
                                }
                            )
                        else:
                            selected = self._advanced_ranking_selection(filtered_symbols, target_date)
                        
                        self.logger.info(f"[SYMBOL_SELECTION] 最終選択: {selected} (候補: {len(filtered_symbols)}銘柄)")
                        self.logger.debug(f"[DEBUG] _get_optimal_symbol | selected_symbol: {selected}")
                        self.logger.info(
                            f"理由: DSS Core V3使用不可 | "
                            f"影響範囲: {len(filtered_symbols)}銘柄から選択 (選択: {selected})"
                        )
                        return selected
                except Exception as e:
                    # Error handling
                    pass
            
            
        except Exception as e:
            return None
    
    def _evaluate_and_execute_switch(self, selected_symbol: str, 
                                   target_date: datetime) -> Dict[str, Any]:
        """
        銘柄切替の評価と実行（Sprint 2: 複数銘柄保有対応）
        
        Sprint 2実装: FIFO方式による複数銘柄保有対応
        - max_positions到達時は最も古いポジションを決済（FIFO）
        - 4ケース分岐: 初回エントリー/銘柄継続/新規エントリー/銘柄切替
        
        Cycle 4-A実装: 利益中ポジション保護機能（保留）
        - Sprint 2では単純FIFO実装、利益保護はSprint 3で検討
        
        Args:
            selected_symbol: 選択された銘柄
            target_date: 対象日付
        
        Returns:
            Dict[str, Any]: 切替実行結果
                - switch_executed: bool（切替実行フラグ）
                - from_symbol: str | None（切替元銘柄）
                - to_symbol: str（切替先銘柄）
                - reason: str（理由）
                - force_close_symbol: str | None（決済対象銘柄、重要）
        """
        try:
            # Sprint 2: 複数銘柄保有対応 - 4ケース分岐
            self.logger.info(
                f"[SWITCH] 銘柄切替判定開始 | "
                f"target_date={target_date.strftime('%Y-%m-%d')}, "
                f"selected_symbol={selected_symbol}, "
                f"保有数={len(self.positions)}/{self.max_positions}, "
                f"保有銘柄={list(self.positions.keys())}"
            )
            
            # ===== デバッグログ追加 (2026-02-15) =====
            self.logger.info(
                f"[SWITCH_DEBUG] 切替判定: " 
                f"len(positions)={len(self.positions)}, "
                f"max_positions={self.max_positions}, "
                f"selected_symbol={selected_symbol}, "
                f"positions.keys()={list(self.positions.keys())}"
            )
            # ==========================================
            
            # ケース1: 初回エントリー
            if len(self.positions) == 0:
                self.logger.info(f"[SWITCH_DEBUG] ケース1実行: 初回エントリー selected_symbol={selected_symbol}")
                self.logger.info(f"[SWITCH] 初回エントリー: {selected_symbol}")
                return {
                    'date': target_date.strftime('%Y-%m-%d'),
                    'switch_executed': False,
                    'from_symbol': None,
                    'to_symbol': selected_symbol,
                    'reason': 'initial_entry',
                    'force_close_symbol': None,
                    'switch_cost': 0
                }
            
            # ケース2: 選択銘柄が既に保有中
            if selected_symbol in self.positions:
                self.logger.info(f"[SWITCH_DEBUG] ケース2実行: 継続保有 symbol={selected_symbol}")
                self.logger.debug(
                    f"[SWITCH] 銘柄継続: {selected_symbol} "
                    f"(entry_date={self.positions[selected_symbol].get('entry_date')})"
                )
                return {
                    'date': target_date.strftime('%Y-%m-%d'),
                    'switch_executed': False,
                    'from_symbol': selected_symbol,
                    'to_symbol': selected_symbol,
                    'reason': 'symbol_already_held',
                    'force_close_symbol': None,
                    'switch_cost': 0
                }
            
            # ケース3: max_positions未満（新規エントリー可能）
            if len(self.positions) < self.max_positions:
                self.logger.info(
                    f"[SWITCH_DEBUG] ケース3実行: 新規追加 "
                    f"len={len(self.positions)} < max={self.max_positions}, "
                    f"selected_symbol={selected_symbol}"
                )
                self.logger.info(
                    f"[SWITCH] 新規エントリー枠あり: {selected_symbol} "
                    f"(現在{len(self.positions)}/{self.max_positions}銘柄)"
                )
                return {
                    'date': target_date.strftime('%Y-%m-%d'),
                    'switch_executed': False,
                    'from_symbol': None,
                    'to_symbol': selected_symbol,
                    'reason': 'new_entry_available',
                    'force_close_symbol': None,
                    'switch_cost': 0
                }
            
            # ケース4: max_positions到達、銘柄切替判定
            # 最も古いポジションを決済候補とする（FIFO）
            oldest_symbol, oldest_position = min(
                self.positions.items(), 
                key=lambda x: x[1]['entry_date']
            )
            
            self.logger.info(
                f"[SWITCH_DEBUG] ケース4実行: FIFO決済 "
                f"len={len(self.positions)} >= max={self.max_positions}, "
                f"oldest_symbol={oldest_symbol}, "
                f"selected_symbol={selected_symbol}"
            )
            self.logger.info(
                f"[FIFO_EXIT] max_positions到達、FIFO決済候補: {oldest_symbol} "
                f"(entry_date={oldest_position['entry_date']}, "
                f"strategy={oldest_position.get('strategy', 'unknown')})"
            )
            
            # SymbolSwitchManagerで切替可否を判定
            self.ensure_components()  # 遅延初期化
            switch_evaluation = self.switch_manager.evaluate_symbol_switch(
                from_symbol=oldest_symbol,
                to_symbol=selected_symbol,
                target_date=target_date
            )
            
            should_switch = switch_evaluation.get('should_switch', False)
            self.logger.debug(
                f"[SWITCH] SymbolSwitchManager判定: should_switch={should_switch}, "
                f"reason={switch_evaluation.get('reason')}"
            )
            
            if should_switch:
                # 銘柄切替実行（force_closeフラグ付き）
                self.logger.warning(
                    f"[FIFO_EXIT] 銘柄切替実行: {oldest_symbol} -> {selected_symbol} "
                    f"(FIFO決済, entry_date={oldest_position['entry_date']})"
                )
                
                switch_result = {
                    'date': target_date.strftime('%Y-%m-%d'),
                    'switch_executed': True,
                    'from_symbol': oldest_symbol,
                    'to_symbol': selected_symbol,
                    'reason': switch_evaluation.get('reason', 'max_positions_fifo'),
                    'executed_date': target_date,
                    'force_close_symbol': oldest_symbol,  # 決済対象（重要）
                    'switch_cost': 0
                }
                
                # 切替履歴記録
                self.switch_manager.record_switch_executed(switch_result)
                
                return switch_result
            else:
                # 切替拒否（min_holding_days未満、max_switches_per_month超過等）
                self.logger.info(
                    f"[SWITCH] 銘柄切替拒否: {selected_symbol}, "
                    f"reason={switch_evaluation.get('reason')}"
                )
                return {
                    'date': target_date.strftime('%Y-%m-%d'),
                    'switch_executed': False,
                    'from_symbol': oldest_symbol,
                    'to_symbol': selected_symbol,
                    'reason': switch_evaluation.get('reason'),
                    'force_close_symbol': None,
                    'switch_cost': 0
                }
            
        except Exception as e:
            self.logger.error(f"[SWITCH] 銘柄切替評価エラー: {e}", exc_info=True)
            return {
                'date': target_date.strftime('%Y-%m-%d'),
                'switch_executed': False,
                'from_symbol': None,
                'to_symbol': selected_symbol,
                'reason': f'error: {e}',
                'force_close_symbol': None,
                'switch_cost': 0,
                'error': str(e)
            }
    
    def _execute_multi_strategies(self, symbol: str, target_date: datetime, force_close_on_entry: bool = False) -> Dict[str, Any]:
        """
        マルチ戦略実行(main_new.py統合版)
        
        Args:
            symbol: 対象銘柄
            target_date: 対象日付
            force_close_on_entry: 銘柄切替時のForceClose要求フラグ
        
        Returns:
            Dict[str, Any]: 戦略実行結果
        """
        try:
            # [Task11] ForceClose実行中はスキップ
            if self.force_close_in_progress:
                self.logger.warning(f"[DSSMS_FORCE_CLOSE_SUPPRESS] ForceClose実行中のため戦略評価をスキップ: symbol={symbol}, date={target_date.strftime('%Y-%m-%d')}")
                return {
                    'status': 'skipped',
                    'symbol': symbol,
                    'target_date': target_date
                }
            
            # 1. force_close_on_entry処理
            if force_close_on_entry:
                self.logger.info(
                    f"[DSSMS_FORCE_CLOSE_REQUEST] 銘柄切替によるForceClose要求: "
                    f"symbol={symbol}, date={target_date.strftime('%Y-%m-%d')}"
                )
            
            stock_data, index_data = self._get_symbol_data(symbol, target_date)
            
            if stock_data is None or stock_data.empty:
                return {
                    'status': 'data_unavailable',
                    'symbol': symbol,
                    'date': target_date.strftime('%Y-%m-%d')
                }
            
            # 2. MainSystemController初期化
            from main_new import MainSystemController
            
            config = {
                'execution': {
                    'execution_mode': 'simple',
                    'broker': {
                        'initial_cash': self.config.get('initial_capital', 1000000),
                        'commission_per_trade': 1.0
                    }
                },
                'risk_management': {
                    'use_enhanced_risk': False,
                    'max_drawdown_threshold': 0.15
                },
                'performance': {
                    'use_aggregator': False
                },
            }
            
            # Option A実装(December 28, 2025): MainSystemController日次作成削除
            # 修正2: 日次でのMainSystemController作成を削除(資金リセット防止)
            # 代わりに,self.main_controllerを初回のみ作成(遅延初期化)
            if self.main_controller is None:
                self.main_controller = MainSystemController(config)
                self.logger.info("[Option A] MainSystemController初回作成完了")
            
            # 3. バックテスト実行(累積期間方式復元)
            # Phase 1実装(December 30, 2025): 累積期間方式復元
            
            # 【Phase 1実装】December 30, 2025
            # 累積期間方式復元: dssms_backtest_start_date(固定開始日)からtarget_date(対象日)まで累積的にバックテスト
            backtest_start_date = self.dssms_backtest_start_date  # 固定開始日(例: 2025-01-15)
            backtest_end_date = target_date + timedelta(days=7)  # 期間延長: 7日間の取引期間を確保
            warmup_days = self.warmup_days  # クラス変数を使用(150日)
            
            period_days = (target_date - backtest_start_date).days
            self.logger.info(f"[DETERMINISM_MONITOR] 累積期間バックテスト: {backtest_start_date.strftime('%Y-%m-%d')} -> {target_date.strftime('%Y-%m-%d')} ({period_days}日間)")
            
            if self.main_controller and hasattr(self.main_controller, 'paper_broker'):
                broker = self.main_controller.paper_broker
                self.logger.info(f"[DETERMINISM_MONITOR] PaperBroker状態前: account_balance={broker.account_balance:.0f}, filled_orders_count={len(broker.filled_orders)}")
            
            # 優先度B対応: stock_dataとbacktest_start_dateを記録(_convert_main_new_resultで使用)
            self._last_stock_data = stock_data
            self._last_backtest_start_date = backtest_start_date
            
            self.logger.info(f"[DSSMS->main_new] バックテスト開始: {symbol}, {target_date}")
            self.logger.info(f"[DSSMS->main_new_DATA] 銘柄: {symbol}")
            self.logger.info(f"[DSSMS->main_new_DATA] 対象日: {target_date.strftime('%Y-%m-%d')}")
            self.logger.info(f"[DSSMS->main_new_DATA] trading_start_date: {backtest_start_date.strftime('%Y-%m-%d')} (Phase 1: 累積期間方式復元)")
            self.logger.info(f"[DSSMS->main_new_DATA] trading_end_date: {backtest_end_date.strftime('%Y-%m-%d')}")
            self.logger.info(f"[DSSMS->main_new_DATA] warmup_days: {warmup_days}")
            
            if stock_data is not None and len(stock_data) > 0:
                self.logger.info(f"[DSSMS->main_new_DATA] stock_data範囲: {stock_data.index[0]} ~ {stock_data.index[-1]} ({len(stock_data)}行)")
                
                # [CRITICAL_CHECK] trading_start_dateとstock_data範囲のギャップ検証
                stock_data_start_naive = pd.Timestamp(stock_data.index[0]).tz_localize(None)
                if backtest_start_date < stock_data_start_naive:
                    gap_days = (stock_data_start_naive - backtest_start_date).days
                    self.logger.warning(
                        f"[DATA_RANGE_MISMATCH] trading_start_date({backtest_start_date})がstock_data範囲外です."
                        f"stock_data開始日: {stock_data.index[0]}, ギャップ: {gap_days}日."
                        f"取引可能期間が{gap_days}日短縮されます."
                    )
            else:
                self.logger.warning(f"[DSSMS->main_new_DATA] stock_data: None または空")
            
            if index_data is not None and len(index_data) > 0:
                self.logger.info(f"[DSSMS->main_new_DATA] index_data範囲: {index_data.index[0]} ~ {index_data.index[-1]} ({len(index_data)}行)")
            else:
                # Index data not available
                pass
            
            # Option A実装(December 28, 2025): 修正4: instance variable使用
            # DSSMS専用：MainSystemController出力を無効化
            if hasattr(self.main_controller, 'reporter'):
                # ComprehensiveReporter出力を一時的に無効化
                original_output_dir = self.main_controller.reporter.output_base_dir
                # 出力先を一時ディレクトリに変更（後でクリーンアップ）
                temp_output_dir = Path("temp_dssms_disable_output")
                self.main_controller.reporter.output_base_dir = temp_output_dir
                
                result = self.main_controller.execute_comprehensive_backtest(
                    ticker=symbol,
                    stock_data=stock_data,
                    index_data=index_data,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                    warmup_days=warmup_days,
                    force_close_on_entry=force_close_on_entry
                )
                
                # 出力先を元に戻す
                self.main_controller.reporter.output_base_dir = original_output_dir
                
                # 一時ディレクトリをクリーンアップ
                if temp_output_dir.exists():
                    import shutil
                    try:
                        shutil.rmtree(temp_output_dir)
                        self.logger.info(f"[CLEANUP] 一時出力ディレクトリ削除: {temp_output_dir}")
                    except Exception as cleanup_error:
                        self.logger.warning(f"[CLEANUP] 一時ディレクトリ削除失敗: {cleanup_error}")
            else:
                result = self.main_controller.execute_comprehensive_backtest(
                    ticker=symbol,
                    stock_data=stock_data,
                    index_data=index_data,
                    backtest_start_date=backtest_start_date,
                    backtest_end_date=backtest_end_date,
                    warmup_days=warmup_days,
                    force_close_on_entry=force_close_on_entry
                )
            
            # 決定論破綻監視: MainSystemController実行後のPaperBroker状態チェック
            if self.main_controller and hasattr(self.main_controller, 'paper_broker'):
                broker = self.main_controller.paper_broker
                self.logger.info(f"[DETERMINISM_MONITOR] PaperBroker状態後: account_balance={broker.account_balance:.0f}, filled_orders_count={len(broker.filled_orders)}")
                
                if len(broker.filled_orders) > 0:
                    today_orders = [order for order in broker.filled_orders 
                                  if order.filled_at and order.filled_at.date() == target_date.date()]
                    if len(today_orders) > 1:
                        for i, order in enumerate(today_orders):
                            # Process each order
                            pass
            
            # 4. 結果変換
            return self._convert_main_new_result(result, symbol, target_date)
            
        except Exception as e:
            self.logger.error(f"_execute_multi_strategies error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbol': symbol,
                'date': target_date.strftime('%Y-%m-%d')
            }

    def _create_strategy_instance(self, strategy_name: str, data: pd.DataFrame):
        """
        戦略インスタンス動的生成 (Phase 3-C Day 12)
        
        Parameters:
            strategy_name (str): 戦略名（例: 'VWAPBreakoutStrategy'）
            data (pd.DataFrame): 株価データ
            
        Returns:
            BaseStrategy: 戦略インスタンス
            
        Raises:
            ValueError: 戦略名が不正、またはimport失敗
            
        Author: Backtest Project Team
        Created: 2025-12-31 (Phase 3-C Day 12)
        """
        strategy_map = {
            'VWAPBreakoutStrategy': ('strategies.VWAP_Breakout', 'VWAPBreakoutStrategy'),
            'MomentumInvestingStrategy': ('strategies.Momentum_Investing', 'MomentumInvestingStrategy'),
            'BreakoutStrategy': ('strategies.Breakout', 'BreakoutStrategy'),
            'BreakoutStrategyRelaxed': ('strategies.BreakoutStrategyRelaxed', 'BreakoutStrategyRelaxed'),
            'ContrarianStrategy': ('strategies.contrarian_strategy', 'ContrarianStrategy'),
            'GCStrategy': ('strategies.gc_strategy_signal', 'GCStrategy'),
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        module_name, class_name = strategy_map[strategy_name]
        
        try:
            # 動的import
            import importlib
            module = importlib.import_module(module_name)
            strategy_class = getattr(module, class_name)
            
            # インスタンス作成
            strategy = strategy_class(data)
            self.logger.debug(f"{strategy_name}初期化成功")
            
            return strategy
            
        except ImportError as e:
            raise ValueError(f"{strategy_name} import失敗: {e}")
        except Exception as e:
            raise ValueError(f"{strategy_name}初期化失敗: {e}")
    
    def _simple_market_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        簡易市場分析 (Phase 3-C Day 12)
        
        MarketAnalyzer未利用時のフォールバック。
        UnifiedTrendDetectorのみを使用した簡易版。
        
        copilot-instructions.md準拠:
        - モック/ダミーデータのフォールバック禁止
        - 実データのみ使用
        
        Parameters:
            data (pd.DataFrame): 株価データ
            
        Returns:
            Dict[str, Any]: 市場分析結果
            
        Author: Backtest Project Team
        Created: 2025-12-31 (Phase 3-C Day 12)
        """
        from indicators.unified_trend_detector import detect_unified_trend
        
        trend = detect_unified_trend(data, strategy='DSSMS', method='advanced')
        
        return {
            'market_regime': f"{trend}_trend",
            'confidence_score': 0.5,
            'trend': trend,
            'source': 'simple_fallback'
        }
    
    def _execute_multi_strategies_daily(self, target_date: datetime, symbol: str, stock_data, force_close_position: dict = None) -> Dict[str, Any]:
        """
        Phase 3-C 拡張版統合実行 - 市場分析・戦略選択・ポジション管理統合
        
        DSSMS日次判断とマルチ戦略の統合実行システム。
        Phase 3-C拡張: MarketAnalyzer・DynamicStrategySelector統合、ポジション状態管理追加。
        
        Args:
            target_date (datetime): 判定対象日
            symbol (str): 対象銘柄コード
            stock_data (pd.DataFrame): 株価データ（ウォームアップ期間含む）
            force_close_position (dict): FIFO強制決済用ポジション情報（force_close=True含む）。指定時はexisting_positionを上書き。
            
        Returns:
            Dict[str, Any]: {
                'status': 'success'|'error'|'hold',
                'strategy_name': str,
                'action': 'entry'|'exit'|'hold',
                'signal': int,
                'price': float,
                'shares': int,
                'reason': str,
                'execution_timestamp': datetime,
                'target_date': datetime,
                'symbol': str,
                'adjusted_target_date': datetime,
                'market_analysis': dict,  # Phase 3-C: 市場分析結果
                'strategy_selection': dict  # Phase 3-C: 戦略選択結果
            }
            
        Raises:
            None: エラーは内部で処理し、結果に含める
            
        copilot-instructions.md遵守:
        - バックテスト実行必須: strategy.backtest_daily()の実際実行
        - フォールバック禁止: 実データのみ使用
        - ルックアヘッドバイアス防止: 戦略内で前日データ判定
        
        Author: Backtest Project Team
        Created: 2025-12-31
        Modified: 2025-12-31 (Phase 3-C拡張)
        """
        execution_start = time.time()
        
        try:
            self.logger.info(f"[PHASE3-C-B1] _execute_multi_strategies_daily開始: symbol={symbol}, target_date={target_date.strftime('%Y-%m-%d')}")
            
            # Phase 3-B Step B1: データ構造変換処理
            processed_data = self._normalize_stock_data_structure(stock_data, symbol)
            
            # Phase 3-B Step B1: 休日判定・業務日調整
            adjusted_target_date = self._adjust_to_business_day(target_date, processed_data)
            
            if adjusted_target_date != target_date:
                self.logger.info(f"[PHASE3-C-B1] 業務日調整: {target_date.strftime('%Y-%m-%d')} → {adjusted_target_date.strftime('%Y-%m-%d')}")
            
            # Phase 3-C Day 12 Task 2-1: 市場分析ロジック追加
            if self.market_analyzer:
                try:
                    self.logger.debug(f"[PHASE3-C-B1] MarketAnalyzer.comprehensive_market_analysis()実行開始")
                    market_analysis = self.market_analyzer.comprehensive_market_analysis(
                        processed_data, index_data=None, ticker=symbol
                    )
                    self.logger.info(f"[PHASE3-C-B1] 市場分析完了: regime={market_analysis.get('market_regime')}, confidence={market_analysis.get('confidence_score')}")
                except Exception as e:
                    self.logger.warning(f"[PHASE3-C-B1] MarketAnalyzer実行エラー、簡易版にフォールバック: {e}")
                    market_analysis = self._simple_market_analysis(processed_data)
            else:
                self.logger.debug(f"[PHASE3-C-B1] MarketAnalyzer未初期化、簡易版使用")
                market_analysis = self._simple_market_analysis(processed_data)
            
            # Phase 3-C Day 12 Task 2-2: 戦略選択ロジック（DynamicStrategySelector使用）
            # Cycle 13修正: BreakoutStrategyRelaxed強制使用を削除し、実戦略選択を有効化
            # Cycle 14修正: volume_threshold=1.0に調整後、実戦略使用（BreakoutStrategy）
            # 2026-02-26修正: 既存ポジション時は戦略継続（ストップロス判定を正常化）
            # 既存ポジションがある場合はエントリー時の戦略を継続使用
            if symbol in self.positions and self.positions[symbol].get('strategy'):
                position_strategy = self.positions[symbol].get('strategy')
                best_strategy_name = position_strategy
                strategy_selection = {
                    'status': 'position_continuity',
                    'reason': f'Existing position: continuing {position_strategy}',
                    'selected_strategies': [position_strategy]
                }
                self.logger.info(
                    f"[STRATEGY_CONTINUITY] 既存ポジション戦略を継続: "
                    f"{best_strategy_name}（エントリー時戦略）"
                )
            else:
                # 新規エントリー候補: DynamicStrategySelector で選択
                if self.strategy_selector:
                    try:
                        self.logger.info(f"[PHASE3-C-B1] DynamicStrategySelector.select_optimal_strategies()実行開始")
                        # Cycle 23修正: select_best_strategy() → select_optimal_strategies()
                        strategy_selection_result = self.strategy_selector.select_optimal_strategies(
                            market_analysis=market_analysis, 
                            stock_data=processed_data, 
                            ticker=symbol
                        )
                        # Cycle 23修正: 'selected_strategy' → 'selected_strategies'（リスト）
                        selected_strategies = strategy_selection_result.get('selected_strategies', ['BreakoutStrategy'])
                        best_strategy_name = selected_strategies[0] if selected_strategies else 'BreakoutStrategy'
                        strategy_selection = strategy_selection_result
                        self.logger.info(
                            f"[PHASE3-C-B1] 戦略選択完了: {best_strategy_name}, "
                            f"全選択: {selected_strategies}"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"[PHASE3-C-B1] DynamicStrategySelector実行エラー、"
                            f"BreakoutStrategyにフォールバック: {e}"
                        )
                        best_strategy_name = 'BreakoutStrategy'
                        strategy_selection = {
                            'status': 'fallback',
                            'reason': f'Selector error: {str(e)}',
                            'selected_strategies': ['BreakoutStrategy']
                        }
                else:
                    self.logger.warning(f"[PHASE3-C-B1] DynamicStrategySelector未初期化、BreakoutStrategyをデフォルト使用")
                    best_strategy_name = 'BreakoutStrategy'
                    strategy_selection = {
                        'status': 'default',
                        'reason': 'Strategy selector not initialized',
                        'selected_strategies': ['BreakoutStrategy']
                    }
            
            # Phase 3-C Day 12 Task 2-2: 動的戦略インスタンス生成
            try:
                strategy = self._create_strategy_instance(best_strategy_name, processed_data)
                self.logger.debug(f"[PHASE3-C-B1] {best_strategy_name}初期化成功 (データ構造: {processed_data.shape})")
            except Exception as e:
                self.logger.error(f"[PHASE3-C-B1] {best_strategy_name}初期化失敗: {e}")
                return {
                    'status': 'error',
                    'strategy_name': best_strategy_name,
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': 0,
                    'reason': f'Strategy initialization failed: {e}',
                    'execution_timestamp': datetime.now(),
                    'target_date': target_date,
                    'adjusted_target_date': adjusted_target_date,
                    'symbol': symbol,
                    'market_analysis': market_analysis,
                    'strategy_selection': strategy_selection
                }
            
            # Phase 3-C Day 12 Task 2-3: existing_position判定ロジック追加
            # Sprint 2修正: self.current_position → self.positions[symbol]
            # Fix 2: force_close_positionが渡された場合、existing_positionを上書き
            if force_close_position is not None:
                existing_position = {
                    'entry_idx': force_close_position.get('entry_idx', 0),
                    'quantity': force_close_position.get('shares', 0),
                    'entry_price': force_close_position.get('entry_price', 0.0),
                    'entry_date': force_close_position.get('entry_date', None),
                    'strategy': force_close_position.get('strategy', best_strategy_name),
                    'entry_symbol': symbol,
                    'force_close': True
                }
                self.logger.info(
                    f"[FORCE_CLOSE] existing_position overridden: symbol={symbol}, "
                    f"entry_price={existing_position['entry_price']:.2f}, "
                    f"force_close=True"
                )
            elif symbol in self.positions:
                # 既に同じ銘柄でポジション保有中（戦略継続）
                existing_position = {
                    'entry_idx': self.positions[symbol].get('entry_idx', 0),  # 2026-02-26: BUY時に正しく保存されるはず
                    'quantity': self.positions[symbol].get('shares', 0),
                    'entry_price': self.positions[symbol].get('entry_price', 0.0),
                    'entry_date': self.positions[symbol].get('entry_date', None),
                    'strategy': self.positions[symbol].get('strategy', best_strategy_name),
                    'entry_symbol': symbol  # Cycle 7: エントリー銘柄コード
                }
                self.logger.info(
                    f"[MULTI_POSITION] 銘柄継続: {symbol}, "
                    f"既存戦略={existing_position['strategy']}, "
                    f"保有数={len(self.positions)}/{self.max_positions}"
                )
            else:
                # ポジションなし
                existing_position = None
                self.logger.debug(f"[MULTI_POSITION] 新規判定: {symbol}, existing_position=None")
            
            # backtest_daily()実行（copilot-instructions.md: バックテスト実行必須）
            # Cycle 7: force_close時にentry_symbolのデータを取得
            entry_symbol_data = None
            if existing_position and existing_position.get('force_close', False):
                entry_symbol = existing_position.get('entry_symbol', '')
                if entry_symbol:
                    self.logger.info(f"[CYCLE7] force_close時のentry_symbol={entry_symbol}のデータ取得開始")
                    try:
                        entry_symbol_data, _ = self._get_symbol_data(entry_symbol, adjusted_target_date)
                        if entry_symbol_data is not None and len(entry_symbol_data) > 0:
                            self.logger.info(f"[CYCLE7] entry_symbol={entry_symbol}のデータ取得成功: {len(entry_symbol_data)}行")
                        else:
                            self.logger.warning(f"[CYCLE7] entry_symbol={entry_symbol}のデータ取得失敗またはデータなし")
                    except Exception as e:
                        self.logger.error(f"[CYCLE7] entry_symbolデータ取得エラー: {e}", exc_info=True)
            
            try:
                # Cycle 7: entry_symbol_dataをkwargsで渡す
                kwargs = {}
                if entry_symbol_data is not None:
                    kwargs['entry_symbol_data'] = entry_symbol_data
                    
                self.logger.info(f"[PHASE3-C-B1] backtest_daily()実行開始: adjusted_target_date={adjusted_target_date.strftime('%Y-%m-%d')}, existing_position={existing_position}, kwargs={list(kwargs.keys())}")
                result = strategy.backtest_daily(adjusted_target_date, processed_data, existing_position=existing_position, **kwargs)
                self.logger.info(f"[PHASE3-C-B1] backtest_daily()実行完了: action={result['action']}, signal={result['signal']}")
                
                # 実際の実行結果を検証（copilot-instructions.md: 検証なしの報告禁止）
                if not isinstance(result, dict):
                    raise ValueError(f"Invalid result type: {type(result)}, expected dict")
                
                required_keys = ['action', 'signal', 'price', 'shares', 'reason']
                missing_keys = [key for key in required_keys if key not in result]
                if missing_keys:
                    raise ValueError(f"Missing required keys: {missing_keys}")
                
                # Cycle 21修正: action値の正規化（'entry'→'buy', 'exit'→'sell'）
                # 理由: BreakoutStrategyは'entry'/'exit'を返すが、DSSMSは'buy'/'sell'を期待
                if result['action'] == 'entry':
                    result['action'] = 'buy'
                    self.logger.info(f"[PHASE3-C-B1] action正規化: 'entry' → 'buy'")
                elif result['action'] == 'exit':
                    result['action'] = 'sell'
                    self.logger.info(f"[PHASE3-C-B1] action正規化: 'exit' → 'sell'")
                    
            except Exception as e:
                self.logger.error(f"[PHASE3-C-B1] backtest_daily()実行エラー: {e}", exc_info=True)
                return {
                    'status': 'error',
                    'strategy_name': best_strategy_name,
                    'action': 'hold',
                    'signal': 0,
                    'price': 0.0,
                    'shares': 0,
                    'reason': f'backtest_daily() execution failed: {e}',
                    'execution_timestamp': datetime.now(),
                    'target_date': target_date,
                    'adjusted_target_date': adjusted_target_date,
                    'symbol': symbol,
                    'market_analysis': market_analysis,
                    'strategy_selection': strategy_selection
                }
            
            # Phase 3-C Day 12 Task 2-4: ポジション状態更新ロジック追加
            # Cycle 10-10検証: result['action']の実際の値を確認
            self.logger.info(
                f"[DEBUG_ACTION] result['action']='{result.get('action')}', "
                f"signal={result.get('signal')}, "
                f"type={type(result.get('action'))}, "
                f"symbol={symbol}, target_date={adjusted_target_date.strftime('%Y-%m-%d')}"
            )
            
            # Sprint 2削除: 古いcurrent_position更新ロジック（Line 2537-2558）
            # 理由: Task 2-2-2で実装済みの self.positions ベースの処理に統合
            
            # 結果処理・ログ記録
            execution_time = time.time() - execution_start
            
            # Cycle 9: BUY前の残高チェック（「70万円損失後に96万円エントリー可能」問題対応）
            if result['action'] == 'buy' and result['signal'] != 0:
                required_cash = result['price'] * result['shares']
                
                if self.cash_balance < required_cash:
                    # 残高不足: 購入可能な最大株数に調整
                    affordable_shares_raw = int(self.cash_balance / result['price'])
                    affordable_shares = (affordable_shares_raw // 100) * 100  # 100株単位
                    
                    if affordable_shares > 0:
                        # 株数調整してエントリー
                        original_shares = result['shares']
                        result['shares'] = affordable_shares
                        adjusted_required_cash = result['price'] * affordable_shares
                        
                        self.logger.warning(
                            f"[BALANCE_CHECK] 残高不足によりBUY株数調整: "
                            f"現金残高={self.cash_balance:,.0f}円, 必要額={required_cash:,.0f}円, "
                            f"株数: {original_shares}株 → {affordable_shares}株（調整後必要額: {adjusted_required_cash:,.0f}円）"
                        )
                        
                        # Sprint 2削除: current_position['shares']更新は不要（実際のポジション追加は_process_daily_tradingで実行）
                        # 旧コード: if self.current_position is not None: self.current_position['shares'] = affordable_shares
                        
                        # 調整後のrequired_cashを更新
                        required_cash = adjusted_required_cash
                    else:
                        # 購入可能株数が0株（100株未満）: エントリーをスキップ
                        self.logger.warning(
                            f"[BALANCE_CHECK] 残高不足によりBUYスキップ: "
                            f"現金残高={self.cash_balance:,.0f}円, 必要額={required_cash:,.0f}円（購入可能株数0株）"
                        )
                        
                        # actionをholdに変更してスキップ
                        result['action'] = 'hold'
                        result['signal'] = 0
                        result['reason'] += ' (Balance insufficient, entry skipped)'
            
            # Phase 3-C Stage 2: execution_details生成（DSSMS取引0件問題修正）
            execution_details = []
            position_update = {'return': 0, 'cost': 0}
            self.logger.info(
                f"[CONDITION_CHECK] date={adjusted_target_date.strftime('%Y-%m-%d')}, "
                f"symbol={symbol}, action={result['action']}, signal={result['signal']}, "
                f"passes_condition={result['action'] in ['buy', 'sell'] and result['signal'] != 0}"
            )
            if result['action'] in ['buy', 'sell'] and result['signal'] != 0:
                # backtest_daily()の結果をexecution_details形式に変換
                execution_detail = {
                    'timestamp': adjusted_target_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'action': result['action'].upper(),  # 'buy' -> 'BUY'
                    'price': result['price'],
                    'shares': result['shares'],
                    'strategy': best_strategy_name,
                    'signal_strength': result['signal'],
                    'reason': result['reason'],
                    'status': 'executed'
                }
                execution_details.append(execution_detail)
                
                # 取引実行時のポートフォリオ更新（Cycle 4-2: 残高管理追加）
                if result['action'] == 'buy':
                    trade_cost = result['price'] * result['shares']
                    # Cycle 4-2: BUY時に現金残高を減少
                    self.cash_balance -= trade_cost
                    position_update = {'return': -trade_cost, 'cost': trade_cost}
                    
                    # 2026-02-26修正: entry_idxを正確に計算して保存
                    # current_idxを計算（ストップロス判定で使用）
                    try:
                        current_idx = processed_data.index.get_loc(adjusted_target_date)
                        if isinstance(current_idx, slice):
                            current_idx = current_idx.start
                    except Exception as e:
                        self.logger.warning(f"[ENTRY_IDX] adjusted_target_date={adjusted_target_date}のインデックス取得失敗: {e}, データ末尾を使用")
                        current_idx = len(processed_data) - 1
                    
                    # Sprint 2: self.positionsへの追加（2026-02-15修正、2026-02-26 entry_idx修正）
                    self.positions[symbol] = {
                        'strategy': best_strategy_name,
                        'entry_price': result['price'],
                        'shares': result['shares'],
                        'entry_date': adjusted_target_date,
                        'entry_idx': current_idx  # 2026-02-26修正: 正しいインデックスを保存
                    }
                    self.logger.info(
                        f"[POSITION_ADD] {symbol}: {result['shares']}株を追加, entry_idx={current_idx}, "
                        f"entry_date={adjusted_target_date.strftime('%Y-%m-%d')}, "
                        f"保有数: {len(self.positions)}/{self.max_positions}"
                    )
                    # ===== デバッグログ (2026-02-15, 2026-02-26更新) =====
                    self.logger.info(
                        f"[POSITION_DEBUG] BUY後: {adjusted_target_date.strftime('%Y-%m-%d')}, "
                        f"len(positions)={len(self.positions)}, "
                        f"保有銘柄={list(self.positions.keys())}"
                    )
                    # ========================================================
                    
                    self.logger.info(
                        f"[PORTFOLIO_TRADE] BUY執行: {symbol} {result['shares']}株 @ {result['price']:.2f}円, "
                        f"コスト: {trade_cost:,.0f}円, 残高: {self.cash_balance:,.0f}円"
                    )
                elif result['action'] == 'sell':
                    trade_profit = result['price'] * result['shares']
                    # Cycle 4-2: SELL時に現金残高を増加
                    self.cash_balance += trade_profit
                    position_update = {'return': trade_profit, 'cost': 0}
                    
                    # Sprint 2: self.positionsからの削除（2026-02-15修正）
                    if symbol in self.positions:
                        del self.positions[symbol]
                        self.logger.info(f"[POSITION_DELETE] {symbol}: ポジション削除（保有数: {len(self.positions)}/{self.max_positions}）")
                    else:
                        self.logger.warning(f"[POSITION_DELETE] {symbol}: ポジション未登録（削除スキップ）")
                    
                    self.logger.info(
                        f"[PORTFOLIO_TRADE] SELL執行: {symbol} {result['shares']}株 @ {result['price']:.2f}円, "
                        f"収益: {trade_profit:,.0f}円, 残高: {self.cash_balance:,.0f}円"
                    )
                
                self.logger.info(f"[PHASE3-C-B1] execution_details生成: action={result['action']}, price={result['price']}, shares={result['shares']}")
            
            # Cycle 10-4: ポジション価値計算修正（日次終値ベースの時価評価）
            # 注: portfolio_equity_curve.csvは「その日終了時点のポートフォリオ状態」を記録
            # Sprint 2修正: 現在の銘柄のポジション価値のみ計算（全体はループ外で計算）
            position_value = 0.0
            
            if symbol in self.positions:
                # ポジション保有中: 当日終値ベースで時価評価
                try:
                    current_price = processed_data['Adj Close'].iloc[-1] if len(processed_data) > 0 else 0
                    position_value = self.positions[symbol].get('shares', 0) * current_price
                    self.logger.debug(
                        f"[PORTFOLIO_VALUE] ポジション評価: {self.positions[symbol].get('shares')}株 × "
                        f"{current_price:.2f}円 = {position_value:,.2f}円"
                    )
                except Exception as e:
                    self.logger.warning(f"[PORTFOLIO_VALUE] 終値取得エラー: {e}")
                    position_value = 0.0
            
            total_portfolio_value = self.cash_balance + position_value
            
            self.logger.info(
                f"[PORTFOLIO_SNAPSHOT] {target_date.strftime('%Y-%m-%d')}: "
                f"cash={self.cash_balance:,.0f}円, position={position_value:,.0f}円, total={total_portfolio_value:,.0f}円"
            )
            
            unified_result = {
                'status': 'success',
                'strategy_name': best_strategy_name,
                'action': result['action'],
                'signal': result['signal'],
                'price': result['price'],
                'shares': result['shares'],
                'reason': result['reason'],
                'execution_timestamp': datetime.now(),
                'target_date': target_date,
                'adjusted_target_date': adjusted_target_date,
                'symbol': symbol,
                'execution_time_ms': round(execution_time * 1000, 2),
                'market_analysis': market_analysis,
                'strategy_selection': strategy_selection,
                'execution_details': execution_details,  # Stage 2: execution_details追加
                'position_update': position_update,  # 取引実行時のポートフォリオ更新データ
                # Cycle 10: ポートフォリオ管理データ追加
                'cash_balance': self.cash_balance,
                'position_value': position_value,
                'total_portfolio_value': total_portfolio_value
            }
            
            # 詳細ログ記録
            self.logger.info(
                f"[PHASE3-C-B1] 統合実行完了 - "
                f"Status: {unified_result['status']}, "
                f"Strategy: {unified_result['strategy_name']}, "
                f"Action: {unified_result['action']}, "
                f"Signal: {unified_result['signal']}, "
                f"Price: {unified_result['price']}, "
                f"Shares: {unified_result['shares']}, "
                f"ExecutionTime: {unified_result['execution_time_ms']}ms, "
                f"DateAdjusted: {target_date.strftime('%Y-%m-%d')} → {adjusted_target_date.strftime('%Y-%m-%d')}"
            )
            
            return unified_result
            
        except Exception as e:
            execution_time = time.time() - execution_start
            self.logger.error(f"[PHASE3-B-B1] 予期しないエラー: {e}", exc_info=True)
            
            return {
                'status': 'error',
                'strategy_name': 'VWAPBreakout',
                'action': 'hold',
                'signal': 0,
                'price': 0.0,
                'shares': 0,
                'reason': f'Unexpected error: {e}',
                'execution_timestamp': datetime.now(),
                'target_date': target_date,
                'adjusted_target_date': target_date,  # 調整失敗時は元の日付
                'symbol': symbol,
                'execution_time_ms': round(execution_time * 1000, 2)
            }
    
    def _convert_main_new_result(self, main_new_result: Dict, symbol: str, target_date: datetime) -> Dict[str, Any]:
        """
        main_new.pyの結果をDSSMS形式に変換
        
        Args:
            main_new_result: MainSystemControllerの実行結果
            target_date: 対象日付
        
        Returns:
            Dict[str, Any]: DSSMS形式の戦略実行結果
        """
        try:
            if main_new_result.get('status') != 'SUCCESS':
                return {
                    'status': 'error',
                    'error': main_new_result.get('error', 'Unknown error'),
                    'symbol': symbol,
                    'date': target_date.strftime('%Y-%m-%d')
                }
            
            performance_results = main_new_result.get('performance_results', {})
            execution_results = main_new_result.get('execution_results', {})
            
            # summary_statisticsから取得(main_new.pyの実際の構造に合わせる)
            summary_stats = performance_results.get('summary_statistics', {})
            
            # Phase 13修正: 累積期間バックテスト重複計上対策(増分計算)
            # December 3, 2025修正: 'total_return'(比率)ではなく'total_profit'(金額)を取得
            current_total_profit = summary_stats.get('total_profit', 0.0)
            if current_total_profit == 0.0:
                basic_perf = performance_results.get('basic_performance', {})
                current_total_profit = basic_perf.get('total_profit', 0.0)
                
                if current_total_profit == 0.0 and 'total_portfolio_value' in execution_results:
                    portfolio_value = execution_results.get('total_portfolio_value', 1000000.0)
                    current_total_profit = portfolio_value - 1000000.0
            
            # 増分計算: 前回のtotal_profitとの差分を算出
            incremental_profit = current_total_profit - self.previous_total_profit
            self.previous_total_profit = current_total_profit
            
            # 互換性のため,total_returnにはincremental_profitを代入
            total_return = incremental_profit
            
            # copilot-instructions.md準拠: 異常値チェック追加
            # -100%未満(資産が完全消失以上の損失)は異常値として扱う
            ABNORMAL_RETURN_THRESHOLD = -1.0  # -100%
            if total_return < ABNORMAL_RETURN_THRESHOLD:
                error_msg = (
                    f"閾値: {ABNORMAL_RETURN_THRESHOLD:.2%}未満は異常値として扱います."
                )
                self.logger.error(f"[ABNORMAL_VALUE] {error_msg}")
                return {
                    'status': 'abnormal_value',
                    'error': error_msg,
                    'symbol': symbol,
                    'date': target_date.strftime('%Y-%m-%d'),
                    'total_return': total_return,
                    'threshold': ABNORMAL_RETURN_THRESHOLD
                }
            
            # Phase 2優先度3: execution_details抽出(詳細設計書3.1.3準拠)
            execution_details = execution_results.get('execution_details', [])
            
            # stock_dataとbacktest_start_dateの取得(_execute_multi_strategiesから引き継ぎ)
            stock_data_info = "N/A"
            if hasattr(self, '_last_stock_data') and self._last_stock_data is not None and len(self._last_stock_data) > 0:
                stock_data_info = f"{self._last_stock_data.index[0]} ~ {self._last_stock_data.index[-1]}"
            backtest_start_info = getattr(self, '_last_backtest_start_date', 'N/A')
            
            self.logger.info(
                f"[EXEC_DETAILS_COUNT] {symbol} {target_date.strftime('%Y-%m-%d')}: "
                f"execution_details件数={len(execution_details)}, "
                f"stock_data範囲={stock_data_info}, "
                f"trading_start_date={backtest_start_info}"
            )
            
            if not execution_details:
                self.logger.warning(
                    f"[EXECUTION_DETAILS_MISSING] {symbol} {target_date.strftime('%Y-%m-%d')}: "
                    f"execution_detailsが空です.ComprehensiveReporter生成時に取引履歴が不足する可能性があります."
                )
            
            # DSSMS形式に変換
            dssms_result = {
                'status': 'success',
                'symbol': symbol,
                'date': target_date.strftime('%Y-%m-%d'),
                'strategy_results': {
                    'main_strategy': {
                        'total_return': total_return,
                        'sharpe_ratio': summary_stats.get('sharpe_ratio', 0.0),
                        'max_drawdown': summary_stats.get('max_drawdown', 0.0),
                        'total_trades': summary_stats.get('total_trades', 0),
                        'winning_trades': execution_results.get('winning_trades', 0),
                        'win_rate': summary_stats.get('win_rate', 0.0)
                    }
                },
                'summary': {
                    'execution_timestamp': main_new_result.get('execution_timestamp'),
                    'market_analysis': main_new_result.get('market_analysis', {}),
                    'strategy_selection': main_new_result.get('strategy_selection', {}),
                    'risk_assessment': main_new_result.get('risk_assessment', {})
                },
                'position_update': {
                    'return': total_return,  # Phase 13: 増分のみを返す(重複計上対策)
                    'balance': execution_results.get('total_portfolio_value', 1000000.0),
                    'total_trades': summary_stats.get('total_trades', 0)
                },
                # Phase 2優先度3: execution_details追加(詳細設計書3.1.3準拠)
                'execution_details': execution_details
            }
            
            return dssms_result
            
        except Exception as e:
            return {
                'status': 'conversion_error',
                'error': str(e),
                'symbol': symbol,
                'date': target_date.strftime('%Y-%m-%d')
            }
    
    def _execute_single_strategy(self, strategy_name: str, symbol: str, 
                               stock_data: Any, index_data: Any,
                               target_date: datetime) -> Dict[str, Any]:
        """
        単一戦略実行(簡略実装)
        
        Args:
            strategy_name: 戦略名
            target_date: 対象日付
        
        Returns:
            Dict[str, Any]: 戦略実行結果
        """
        try:
            
            return {'success': False, 'signal': 'HOLD', 'reason': 'insufficient_data'}
            
            # 基本指標計算
            close_prices = stock_data['Close'].tail(20)
            sma_5 = close_prices.tail(5).mean()
            sma_20 = close_prices.mean()
            current_price = close_prices.iloc[-1]
            volume = stock_data['Volume'].iloc[-1]
            
            # 修正案2: 実行された戦略名を記録(December 14, 2025追加)
            # _get_active_strategy_name()がこの値を使用して実際の戦略名を返す
            self.last_executed_strategy = strategy_name
            
            signal = 'HOLD'
            confidence = 0.5
            
            if strategy_name == 'VWAPBreakoutStrategy':
                vwap = (stock_data['Close'] * stock_data['Volume']).sum() / stock_data['Volume'].sum()
                if current_price > vwap * 1.02:
                    signal = 'BUY'
                    confidence = 0.7
                elif current_price < vwap * 0.98:
                    signal = 'SELL'
                    confidence = 0.6
                    
            elif strategy_name == 'MomentumInvestingStrategy':
                if sma_5 > sma_20 * 1.01:
                    signal = 'BUY'
                    confidence = 0.75
                elif sma_5 < sma_20 * 0.99:
                    signal = 'SELL'
                    confidence = 0.65
                    
            elif strategy_name == 'BreakoutStrategy':
                high_20 = stock_data['High'].tail(20).max()
                low_20 = stock_data['Low'].tail(20).min()
                if current_price > high_20 * 1.005:
                    signal = 'BUY'
                    confidence = 0.8
                elif current_price < low_20 * 0.995:
                    signal = 'SELL'
                    confidence = 0.7
            
            else:
                price_change = (current_price - close_prices.iloc[-2]) / close_prices.iloc[-2]
                if price_change > 0.02:
                    signal = 'BUY'
                    confidence = 0.6
                elif price_change < -0.02:
                    signal = 'SELL'
                    confidence = 0.6
            
            return {
                'success': True,
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'indicators': {
                    'sma_5': sma_5,
                    'sma_20': sma_20,
                    'volume': volume
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'signal': 'HOLD'
            }
    
    def _get_active_strategy_name(self, symbol: str) -> str:
        """
        修正案2: アクティブな基本戦略名を取得(December 14, 2025追加)
        
        目的:
        - execution_detailsに実際の戦略名(GCStrategy等)を記録する
        
        Args:
        
        Returns:
            str: 戦略名(例: 'GCStrategy', 'ContrarianStrategy')
                 戦略名不明時は'DSSMS_SymbolSwitch'を返す
        
        Examples:
            'GCStrategy'  # self.last_executed_strategy = 'GCStrategy'の場合
            >>> self._get_active_strategy_name('9101')
            'DSSMS_SymbolSwitch'  # self.last_executed_strategy = Noneの場合
        """
        # 最後に実行された戦略名を返す(記録済みの場合)
        if self.last_executed_strategy:
            return self.last_executed_strategy
        
        return 'DSSMS_SymbolSwitch'
    
    def _get_symbol_data(self, symbol: str, target_date: datetime):
        """
        
        Args:
            target_date: 対象日付
        
        Returns:
        """
        try:
            # yfinanceのhistory()はend_dateをexclusiveとして扱うため,
            end_date = target_date + timedelta(days=1)
            
            # 旧: DSSMS開始日 - warmup期間(累積期間方式)
            # main_new.pyの期待(backtest_start_date - warmup_days)と整合
            warmup_days = getattr(self, 'warmup_days', 90)  # デフォルト90日
            start_date = target_date - timedelta(days=warmup_days)  # target_dateから計算
            self.logger.info(
                f"- warmup({warmup_days}日) = 取得開始日({start_date.strftime('%Y-%m-%d')})"
            )
            
            
            # キャッシュから取得試行
            cached_data = self.data_cache.get_cached_data(symbol, start_date, end_date)
            
            data_source = "cache" if cached_data[0] is not None else "yfinance_api"
            self.logger.debug(
                f"[DEBUG] _get_symbol_data | symbol: {symbol} | source: {data_source} | "
                f"period: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
            )
            
            if cached_data[0] is not None:
                return cached_data
            
            if DATA_FETCHER_AVAILABLE:
                try:
                    from src.utils.lazy_import_manager import get_yfinance
                    yf = get_yfinance()
                    ticker = yf.Ticker(to_yfinance(symbol))
                    # auto_adjust=False指定でAdj Closeカラムを保証(VWAPBreakoutStrategy対応)
                    # Line 2880で既にtarget_date + 1日（exclusive対策）済み、追加の期間拡大は不要
                    stock_data = ticker.history(start=start_date, end=end_date, auto_adjust=False)
                    
                    if 'Adj Close' not in stock_data.columns and 'Close' in stock_data.columns:
                        stock_data['Adj Close'] = stock_data['Close']
                    
                    if not stock_data.empty:
                        data_last_date = stock_data.index[-1]
                        
                        # Bug fix (December 31, 2025): required_date未定義エラー修正
                        # 修正前: required_dateが未定義のまま使用されていた
                        # 修正後: target_dateを使用（メソッド引数として定義済み）
                        if hasattr(data_last_date, 'tz') and data_last_date.tz is not None:
                            if not hasattr(target_date, 'tz') or target_date.tz is None:
                                target_date_normalized = pd.Timestamp(target_date).tz_localize(data_last_date.tz)
                            else:
                                target_date_normalized = target_date
                        else:
                            target_date_normalized = target_date
                        
                        if data_last_date < target_date_normalized:
                            self.logger.warning(
                                f"必要日: {target_date_normalized.strftime('%Y-%m-%d')}, "
                                f"不足: {(target_date_normalized - data_last_date).days}日"
                            )
                        else:
                            self.logger.info(
                                f"target_date: {target_date.strftime('%Y-%m-%d')}"
                            )
                    
                    nikkei_ticker = yf.Ticker("^N225")
                    # Line 2880で既にtarget_date + 1日（exclusive対策）済み、追加の期間拡大は不要
                    index_data = nikkei_ticker.history(start=start_date, end=end_date, auto_adjust=False)
                    
                    if 'Adj Close' not in index_data.columns and 'Close' in index_data.columns:
                        index_data['Adj Close'] = index_data['Close']
                    
                    # キャッシュに保存
                    if self.data_cache and not stock_data.empty and not index_data.empty:
                        self.data_cache.store_cached_data(symbol, start_date, end_date, stock_data, index_data)
                    
                    return stock_data, index_data
                except ImportError as import_error:
                    raise RuntimeError(
                        f"Failed to retrieve real market data for {symbol}. "
                        f"yfinance is not available: {import_error}. "
                        f"Mock data generation is prohibited by copilot-instructions.md. "
                        f"Cannot proceed with backtest."
                    ) from import_error
            else:
                raise RuntimeError(
                    f"Failed to retrieve real market data for {symbol}. "
                    f"DATA_FETCHER_AVAILABLE is False. "
                    f"Mock data generation is prohibited by copilot-instructions.md. "
                    f"Cannot proceed with backtest."
                )
                
        except Exception as e:
            self.logger.error(
                f"[ERROR] _get_symbol_data failed for {symbol}: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            return None, None
    
    # _generate_mock_data() メソッドを削除
    # 理由: copilot-instructions.md違反
    
    # _calculate_position_update() メソッドを削除(December 19, 2025)
    # 理由: 未使用メソッド(呼び出し箇所なし)
    # 影響: なし(完全に未使用)
    
    # 削除: _close_position()と_open_position()メソッド本体(Line 2113-2273, 160行)
    # 削除理由: DSSMSが取引実行しない設計に準拠
    #   - execution_details生成もmain_new.py側で実施
    # 代替:
    #   - 決済: main_new.pyのForceCloseStrategy
    # 影響: 呼び出し箇所はStage 3-2で全削除済みのため影響なし
    # 備考:
    #   - Stage 3-1: _calculate_position_update()削除(116行,未使用)
    #   - Stage 3-2: 銘柄切替時取引実行削除(39行)+ ForceClose呼び出し削除(57行)
    #   - Stage 3-3: メソッド本体削除(160行,本削除)
    
    def _check_risk_limits(self, daily_result: Dict[str, Any]) -> Dict[str, Any]:
        """リスク制限チェック"""
        try:
            risk_result = {'risk_violation': False}
            
            current_drawdown = (self.portfolio_value - self.initial_capital) / self.initial_capital
            max_drawdown = self.performance_targets['max_drawdown_limit']
            
            if current_drawdown < max_drawdown:
                risk_result.update({
                    'risk_violation': True,
                    'violation_type': 'max_drawdown',
                    'current_drawdown': current_drawdown,
                    'limit': max_drawdown
                })
            
            return risk_result
            
        except Exception as e:
            return {'risk_violation': False, 'error': str(e)}
    
    def _handle_risk_violation(self, risk_result: Dict[str, Any]) -> None:
        """リスク制限違反時の処理"""
        # 削除: position_size半減処理(Line 2157-2158)
        # 代替: main_new.pyのリスク管理に移管
        try:
            violation_type = risk_result.get('violation_type')
            
            if violation_type == 'max_drawdown':
                pass
                    
        except Exception as e:
            # Error handling
            pass
    
    def _generate_final_results(self, execution_time: float, trading_days: int, 
                              successful_days: int) -> Dict[str, Any]:
        """最終結果生成"""
        try:
            self.logger.info(f"[FINAL_STATS] 最終結果生成開始 (取引日数: {trading_days}, 成功日数: {successful_days})")
            
            # 基本統計
            total_return = self.portfolio_value - self.initial_capital
            total_return_rate = total_return / self.initial_capital
            success_rate = successful_days / trading_days if trading_days > 0 else 0
            
            # [LOG#1] DSSMS収益計算詳細
            self.logger.info(
                f"[REVENUE_CALC_DETAIL] DSSMS収益計算: "
                f"portfolio_value({self.portfolio_value:,.0f}円) - initial_capital({self.initial_capital:,.0f}円) = "
                f"total_return({total_return:,.0f}円, {total_return_rate:+.4%})"
            )
            
            self.logger.info(
                f"[FINAL_STATS] 基本統計計算完了 - "
                f"成功率: {success_rate:.2%}"
            )
            
            daily_returns = [r.get('daily_return_rate', 0) for r in self.daily_results]
            
            if daily_returns:
                volatility = np.std(daily_returns) * np.sqrt(252)  # 年率化
                sharpe_ratio = (np.mean(daily_returns) * 252) / volatility if volatility > 0 else 0
                max_drawdown = min([r.get('daily_return_rate', 0) for r in self.daily_results])
                
                self.logger.info(
                    f"ボラティリティ: {volatility:.4f}, "
                    f"最大DD: {max_drawdown:.4f}"
                )
            else:
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
                self.logger.warning(f"[FINAL_STATS] daily_returns空 - デフォルト値使用")
            
            # 切替統計
            if self.switch_manager:
                switch_stats = self.switch_manager.get_switch_statistics()
                self.logger.info(
                    f"[FINAL_STATS] 切替統計取得完了 - "
                    f"総切替: {switch_stats.get('total_switches', 0)}回, "
                    f"成功率: {switch_stats.get('switch_success_rate', 0.0):.2%}"
                )
            else:
                switch_stats = {
                    'total_switches': 0,
                    'profitable_switches': 0,
                    'unprofitable_switches': 0,
                    'switch_success_rate': 0.0,
                    'average_switch_profit': 0.0
                }
                self.logger.warning(f"[FINAL_STATS] switch_manager未初期化 - デフォルト値使用")
            
            # 戦略統計
            strategy_stats = self._calculate_strategy_statistics()
            self.logger.info(f"[FINAL_STATS] 戦略統計計算完了 - {len(strategy_stats)}戦略")
            
            self.logger.info(f"[FINAL_STATS] 最終結果生成完了 - SUCCESS")
            
            # Cycle 11-2: 正確なfinal_capital計算（cash_balance + ポジション時価評価）
            # portfolio_valueは未決済ポジションを含む計算のため、cash_balance + position_valueを使用
            # Sprint 2修正: self.current_position → len(self.positions)チェック
            final_cash_balance = self.cash_balance
            final_position_value = 0.0
            if len(self.positions) > 0:
                try:
                    # 最終日の終値でポジション時価評価（全ポジションの合計）
                    if self.daily_results:
                        last_daily_result = self.daily_results[-1]
                        final_position_value = last_daily_result.get('position_value', 0.0)
                except Exception as e:
                    self.logger.warning(f"[FINAL_CAPITAL_CALC] ポジション時価評価失敗: {e}")
            
            final_capital_accurate = final_cash_balance + final_position_value
            
            self.logger.info(
                f"[FINAL_CAPITAL_CALC] 最終資本計算: "
                f"cash_balance({final_cash_balance:,.2f}円) + "
                f"position_value({final_position_value:,.2f}円) = "
                f"final_capital({final_capital_accurate:,.2f}円)"
            )
            
            return {
                'execution_metadata': {
                    'start_date': self.daily_results[0]['date'] if self.daily_results else None,
                    'end_date': self.daily_results[-1]['date'] if self.daily_results else None,
                    'total_execution_time_seconds': execution_time,
                    'trading_days': trading_days,
                    'successful_days': successful_days,
                    'generated_at': datetime.now()
                },
                'portfolio_performance': {
                    'initial_capital': self.initial_capital,
                    'final_capital': final_capital_accurate,  # Cycle 11-2: 正確な最終資本
                    'total_return': final_capital_accurate - self.initial_capital,  # Cycle 11-2: 修正
                    'total_return_rate': (final_capital_accurate - self.initial_capital) / self.initial_capital,  # Cycle 11-2: 修正
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'success_rate': success_rate
                },
                'daily_results': self.daily_results,
                'switch_history': self.switch_history,
                'switch_statistics': switch_stats,
                'strategy_statistics': strategy_stats,
                'performance_summary': self.performance_tracker.get_performance_summary() if self.performance_tracker else {
                    'execution': {'average_time_ms': 0},
                    'reliability': {'success_rate': 0.0}
                }
            }
            
            self.logger.info(
                f"[FINAL_REVENUE_BREAKDOWN] DSSMS最終結果 - "
                f"取引日数: {trading_days}, 成功日数: {successful_days}, "
                f"最終資産: {self.portfolio_value:,.0f}円, "
            )
            
        except Exception as e:
            return {
                'error': str(e),
                'execution_metadata': {
                    'start_date': self.daily_results[0]['date'] if self.daily_results else None,
                    'end_date': self.daily_results[-1]['date'] if self.daily_results else None,
                    'trading_days': trading_days if 'trading_days' in locals() else 0,
                    'successful_days': successful_days if 'successful_days' in locals() else 0,
                    'generated_at': datetime.now()
                },
                'portfolio_performance': {
                    'initial_capital': self.initial_capital,
                    'final_capital': self.portfolio_value,
                    'total_return': self.portfolio_value - self.initial_capital,
                    'total_return_rate': (self.portfolio_value - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0,
                    'success_rate': 0.0
                },
                'daily_results': self.daily_results,
                'switch_history': self.switch_history,
                'performance_summary': {
                    'execution': {'average_time_ms': 0},
                    'reliability': {'success_rate': 0.0}
                }
            }
    
    def _calculate_strategy_statistics(self) -> Dict[str, Any]:
        """戦略統計計算"""
        try:
            strategy_stats = {}
            
            strategies = [
                'VWAPBreakoutStrategy', 'MomentumInvestingStrategy', 'BreakoutStrategy',
                'VWAPBounceStrategy', 'OpeningGapStrategy', 'ContrarianStrategy', 'GCStrategy'
            ]
            
            for strategy_name in strategies:
                executions = 0
                successes = 0
                signals = 0
                
                for daily in self.daily_results:
                    strategy_result = daily.get('strategy_results', {}).get('strategy_results', {}).get(strategy_name, {})
                    
                    if strategy_result:
                        executions += 1
                        if strategy_result.get('success', False):
                            successes += 1
                        if strategy_result.get('signal', 'HOLD') != 'HOLD':
                            signals += 1
                
                strategy_stats[strategy_name] = {
                    'execution_count': executions,
                    'success_count': successes,
                    'success_rate': successes / executions if executions > 0 else 0,
                    'signal_count': signals,
                    'signal_rate': signals / executions if executions > 0 else 0
                }
            
            return strategy_stats
            
        except Exception as e:
            return {}
    
    def _rebuild_equity_curve(self, daily_results: List[Dict[str, Any]]) -> 'pd.DataFrame':
        """
        Rebuild equity_curve DataFrame from daily_results (Phase 2 Priority 2: Design Doc 3.1.4 compliant)
        
        Generate 13-column equity_curve DataFrame for ComprehensiveReporter.
        
        Modifications (December 8, 2025):
        - Changed daily_pnl calculation to day-over-day difference (portfolio_value delta)
        - Fixed issues discovered in Task 6 investigation
        
        Args:
            daily_results: DSSMS日次結果リスト
        
        Returns:
        
        Columns:
            - date: 日付(index)
            - cash_balance: 現金残高
            - cumulative_pnl: 累積损益
            - daily_pnl: 日次損益(前日比計算)
            - total_trades: 総取引数
        """
        try:
            import pandas as pd
            
            if not daily_results:
                self.logger.warning("[REBUILD_EQUITY_CURVE] daily_results空 - 空のDataFrame返却")
                return pd.DataFrame()
            
            equity_data = []
            previous_portfolio_value = self.config.get('initial_capital', 1000000)  # 初期資本
            
            for daily_result in daily_results:
                current_portfolio_value = daily_result.get('portfolio_value_end', 0)
                
                # daily_pnl計算: 前日からのportfolio_value変化(全要素を含む)
                # - 取引損益(daily_return)
                # - switch_cost
                daily_pnl = current_portfolio_value - previous_portfolio_value
                
                equity_data.append({
                    'date': daily_result.get('date'),
                    'portfolio_value': current_portfolio_value,
                    'cash_balance': daily_result.get('cash_balance', 0),
                    'position_value': daily_result.get('position_value', 0),
                    'peak_value': daily_result.get('peak_value', 0),
                    'drawdown_pct': daily_result.get('drawdown_pct', 0),
                    'cumulative_pnl': daily_result.get('cumulative_pnl', 0),
                    'daily_pnl': daily_pnl,  # 修正箇所: 前日比計算
                    'total_trades': daily_result.get('total_trades', 0),
                    'active_positions': daily_result.get('active_positions', 0),
                    'risk_status': daily_result.get('risk_status', ''),
                    'blocked_trades': daily_result.get('blocked_trades', 0),
                    'risk_action': daily_result.get('risk_action', '')
                })
                
                previous_portfolio_value = current_portfolio_value
            
            df = pd.DataFrame(equity_data)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            self.logger.info(f"[REBUILD_EQUITY_CURVE] equity_curve再構築完了: {len(df)}行,13カラム")
            return df
            
        except Exception as e:
            import pandas as pd
            return pd.DataFrame()
    
    def _generate_switch_history_csv(self, output_dir: Path) -> None:
        """
        銘柄切替履歴CSV生成(Phase 2優先度5: 詳細設計書4.2.1準拠)
        
        - switch_date: 切替日
        - from_symbol: 前の銘柄
        - to_symbol: 新銘柄
        - reason: 切替理由
        - switch_cost: コスト(金額)
        
        追加推奨カラム(3カラム):
        - ranking_score: DSS選択スコア
        
        Args:
            output_dir: 出力ディレクトリ
        """
        try:
            import pandas as pd
            
            if not self.switch_history:
                self.logger.warning("[SWITCH_HISTORY_CSV] switch_history空 - CSV生成スキップ")
                return
            
            # switch_historyから8カラムCSV生成
            csv_data = []
            for switch in self.switch_history:
                csv_data.append({
                    'switch_date': switch.get('date'),
                    'from_symbol': switch.get('from_symbol', ''),
                    'to_symbol': switch.get('to_symbol', ''),
                    'reason': switch.get('reason', ''),
                    'switch_cost': switch.get('switch_cost', 0.0),
                    'ranking_score': switch.get('ranking_score', 0.0),  # 推奨カラム
                    'portfolio_value_before': switch.get('portfolio_value_before', 0.0),  # 推奨カラム
                    'portfolio_value_after': switch.get('portfolio_value_after', 0.0)  # 推奨カラム
                })
            
            df = pd.DataFrame(csv_data)
            csv_path = output_dir / "dssms_switch_history.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(
                f"[SWITCH_HISTORY_CSV] switch_history.csv生成完了: "
                f"{csv_path}, {len(df)}行,8カラム"
            )
            
        except Exception as e:
            # Error handling
            pass
    
    def _convert_to_execution_format(self, final_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        DSSMS結果をmain_new.py形式に変換(Phase 2優先度3: 詳細設計書3.1準拠)
        修正案C: executed_price除外による重複除去実装(December 5, 2025)
        
        ComprehensiveReporter.generate_full_backtest_report()の入力形式に変換.
        
        Args:
            final_results: DSSMS最終結果
        
        Returns:
            Dict[str, Any]: main_new.py形式の実行結果(execution_results互換)
        
            - status: 'success' -> 'SUCCESS'
            - daily_results: execution_details統合 + 重複除去
            - total_return: total_portfolio_value計算
            - strategy_weights: 固定値(DSSMS_MultiStrategy: 1.0)
        
        """
        try:
            status = final_results.get('status', 'error')
            if status.lower() == 'success':
                status = 'SUCCESS'
            elif status.lower() == 'error':
                status = 'UNKNOWN'
            
            portfolio_perf = final_results.get('portfolio_performance', {})
            initial_capital = portfolio_perf.get('initial_capital', 1000000)
            final_capital = portfolio_perf.get('final_capital', 1000000)
            total_return = final_capital - initial_capital
            
            # execution_details統合 + 重複除去(修正案D: December 10, 2025 - 最終日のみ処理)
            if not final_results.get('daily_results'):
                self.logger.warning("[CONVERT_TO_EXECUTION_FORMAT] daily_results is empty")
                return {
                    'status': 'ERROR',
                    'total_portfolio_value': initial_capital,
                    'initial_capital': initial_capital,
                    'total_return': 0.0,
                    'execution_details': [],
                    'strategy_weights': {'DSSMS_MultiStrategy': 1.0},
                    'execution_results': [{
                        'status': 'ERROR',
                        'total_portfolio_value': initial_capital,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'execution_details': [],
                        'backtest_signals': None
                    }],
                    'equity_recorder': None
                }
            
            # 全日処理(修正案1完全実装: December 14, 2025 - すべての日のexecution_detailsを収集)
            details = []
            for daily_result in final_results['daily_results']:
                daily_execution_details = daily_result.get('execution_details', [])
                details.extend(daily_execution_details)
            
            # [DEBUG_EXEC_DETAILS] 全期間のexecution_details件数を出力
            self.logger.info(
                f"[DEBUG_EXEC_DETAILS] 全期間execution_details収集完了: "
                f"取引日数={len(final_results['daily_results'])}, "
                f"execution_details総数={len(details)}"
            )
            
            all_execution_details = []
            
            for detail_idx, detail in enumerate(details):
                    timestamp = detail.get('timestamp', '')
                    action = detail.get('action', '')
                    symbol = detail.get('symbol', '')
                    # Note: detailは'executed_price'を使用('price'ではない)
                    price = detail.get('executed_price', 0.0)
                    quantity = detail.get('quantity', 0)
                    strategy_name = detail.get('strategy_name', '')
                    
                    self.logger.info(
                        f"[DEBUG_EXEC_DETAILS]   detail[{detail_idx}]: "
                        f"action={action}, timestamp={timestamp}, "
                        f"price={price:.2f}, quantity={quantity}, symbol={symbol}, strategy={strategy_name}"
                    )
                    
                    if not all([timestamp, action, symbol, price > 0]):
                        skipped_invalid_count += 1
                        self.logger.warning(
                            f"[DEDUP_SKIP] 最終日, detail[{detail_idx}]: "
                            f"(timestamp={timestamp}, action={action}, symbol={symbol}, price={price})"
                        )
                        continue
                    
                    # December 9, 2025修正: timestamp+action+symbol+strategyの組み合わせでは
                    # 同じ日付の同じ銘柄の複数取引が重複と誤判定される問題を修正
                    order_id = detail.get('order_id')
                    if not order_id:
                        skipped_invalid_count += 1
                        self.logger.warning(
                            f"[DEDUP_SKIP] 最終日, detail[{detail_idx}]: "
                            f"order_id欠損のためスキップ "
                            f"(timestamp={timestamp}, action={action}, symbol={symbol})"
                        )
                        continue
                    
                    unique_key = order_id
                    
                    # 重複チェック
                    if unique_key in seen_keys:
                        duplicate_count += 1
                        self.logger.debug(
                            f"[DEDUP_SKIP] Duplicate execution_detail: "
                            f"order_id={order_id}, timestamp={timestamp}, action={action}, symbol={symbol}, "
                            f"price={price:.2f}, strategy={strategy_name}"
                        )
                        continue
                    
                    seen_keys.add(unique_key)
                    all_execution_details.append(detail)
            
            self.logger.info(
                f"[DEDUP_RESULT] execution_details重複除去完了: "
                f"総件数={len(all_execution_details)}件, 重複除去={duplicate_count}件, "
            )
            
            # main_new.py形式に変換
            execution_format = {
                'status': status,
                'total_portfolio_value': final_capital,
                'initial_capital': initial_capital,
                'total_return': total_return,
                'execution_details': all_execution_details,
                'strategy_weights': {
                    'DSSMS_MultiStrategy': 1.0  # DSSMS単一戦略として扱う
                },
                'execution_results': [{  # ComprehensiveReporter互換形式(リスト化)
                    'status': status,
                    'total_portfolio_value': final_capital,
                    'winning_trades': 0,  # 後でexecution_detailsから計算
                    'losing_trades': 0,
                    'execution_details': all_execution_details,  # Phase 2: 追加
                    'backtest_signals': None  # equity_curve再構築に不要(configで渡す)
                }],
                # equity_recorderモック(ComprehensiveReporterが使用)
                'equity_recorder': None  # _rebuild_equity_curve()で代替
            }
            
            self.logger.info(
                f"[CONVERT_TO_EXECUTION_FORMAT] DSSMS->main_new.py変換完了: "
                f"status={status}, execution_details={len(all_execution_details)}件"
            )
            
            return execution_format
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'execution_details': []
            }
    
    def _generate_outputs(self, final_results: Dict[str, Any]) -> None:
        """
        DSSMS専用出力生成 - 11ファイル統合出力（ComprehensiveReporter統合版）
        
        出力先: output/dssms_integration/dssms_{timestamp}/
        
        生成ファイル（11ファイル）:
        1. dssms_comprehensive_report.json - 包括分析レポート
        2. dssms_switch_history.csv - 銘柄切替履歴
        3. execution_results.json - 実行結果（銘柄プレフィックス削除）
        4. performance_metrics.json - パフォーマンス指標（銘柄プレフィックス削除）
        5. performance_summary.csv - パフォーマンス要約（銘柄プレフィックス削除）
        6. trade_analysis.json - トレード分析（銘柄プレフィックス削除）
        6.5. all_transactions.csv - 詳細取引データ（ComprehensiveReporter相当）
        7. portfolio_equity_curve.csv - ポートフォリオ資産曲線
        8. comprehensive_report.txt - 包括レポート（ComprehensiveReporter相当）
        9. summary.txt - サマリー（銘柄プレフィックス削除）
        10. dssms_execution_log.txt - 実行ログ
        
        2026-01-08: ComprehensiveReporter相当の詳細取引分析機能統合
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # DSSMS専用出力先ディレクトリ設定
            output_dir = Path(f"output/dssms_integration/dssms_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
                        
            self.logger.info(f"[DSSMS_OUTPUT] DSSMS専用10ファイル生成開始: {output_dir}")
            
            # 1. dssms_comprehensive_report.json - 包括分析レポート
            self._generate_comprehensive_report_json(output_dir, final_results)
            
            # 2. dssms_switch_history.csv - 銘柄切替履歴  
            self._generate_switch_history_csv(output_dir)
            
            # 3. execution_results.json - 実行結果（銘柄プレフィックス削除）
            self._generate_execution_results_json(output_dir, final_results)
            
            # 4. performance_metrics.json - パフォーマンス指標（銘柄プレフィックス削除）
            self._generate_performance_metrics_json(output_dir, final_results)
            
            # 5. performance_summary.csv - パフォーマンス要約（銘柄プレフィックス削除）
            self._generate_performance_summary_csv(output_dir, final_results)
            
            # 6. trade_analysis.json - トレード分析（銘柄プレフィックス削除）
            self._generate_trade_analysis_json(output_dir, final_results)
            
            # 6.5. all_transactions.csv - 詳細取引データ（ComprehensiveReporter相当）
            self._generate_all_transactions_csv(output_dir, final_results)
            
            # 7. portfolio_equity_curve.csv - ポートフォリオ資産曲線
            self._generate_portfolio_equity_curve_csv(output_dir, final_results)
            
            # 8. comprehensive_report.txt - 包括レポート（銘柄プレフィックス削除）
            self._generate_comprehensive_report_txt(output_dir, final_results)
            
            # 9. summary.txt - サマリー（銘柄プレフィックス削除）
            self._generate_summary_txt(output_dir, final_results)
            
            # 10. dssms_execution_log.txt - 実行ログ
            self._generate_execution_log_txt(output_dir, final_results)
            
            self.logger.info(f"[DSSMS_OUTPUT] DSSMS専用10ファイル生成完了: {output_dir}")
            
        except Exception as e:
            # Error handling - CRITICAL: Exception details must be logged for debugging
            import traceback
            self.logger.error(f"[GENERATE_OUTPUTS_ERROR] 出力ファイル生成で例外発生: {type(e).__name__}: {e}")
            self.logger.error(f"[GENERATE_OUTPUTS_STACK] スタックトレース: {traceback.format_exc()}")
            self.logger.error(f"[GENERATE_OUTPUTS_STATE] final_results keys: {list(final_results.keys()) if final_results else 'None'}")
            # DO NOT HIDE EXCEPTIONS - Let them be visible for debugging
            raise e
    
    def _load_default_config(self) -> Dict[str, Any]:
        """デフォルト設定読み込み（Task 1: 2026-01-13最適化）"""
        return {
            'initial_capital': 1000000,
            'symbol_switch': {
                # Sprint 1 検証用設定: 最小保有期間を5日に設定
                # 目的: システム動作の最低限確認
                # 修正日: 2026-02-09
                'min_holding_days': 5,  # 10 → 5に変更（Sprint 1検証用）
                'max_switches_per_month': 5,  # 10回 → 5回に変更
                'switch_cost_rate': 0.001,  # 0.1%（明示的に追加）
            },
            'data_cache': {
                'cache_size_mb': 100,
                'cache_retention_days': 30
            },
            'export_settings': {
                'include_charts': True,
                'output_directory': 'output/dssms_integration'
            },
            'report_settings': {
                'analysis_depth': 'comprehensive',
                'include_recommendations': True
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得（Sprint 2: 複数銘柄保有対応）"""
        try:
            dss_available = self._check_dss_availability()
            risk_available = self._check_risk_management_availability()
            data_available = self._check_data_fetcher_availability()
            
            return {
                'held_symbols': list(self.positions.keys()),  # Sprint 2: 保有銘柄リスト
                'positions_count': len(self.positions),
                'max_positions': self.max_positions,
                'portfolio_value': self.portfolio_value,
                # Phase 1 Stage 4-2: position_size削除(December 19, 2025)
                'daily_results_count': len(self.daily_results),
                'switch_history_count': len(self.switch_history),
                'dss_available': dss_available,
                'risk_management_available': risk_available,
                'data_fetcher_available': data_available,
                'performance_summary': self.performance_tracker.get_performance_summary() if self.performance_tracker else {}
            }
        except Exception as e:
            return {'error': str(e)}
            
    def _check_dss_availability(self) -> bool:
        """DSS Core V3可用性チェック"""
        try:
            dss_core = self._initialize_dss_core()
            return dss_core is not None
        except Exception:
            return False
    
    def _check_risk_management_availability(self) -> bool:
        """リスク管理可用性チェック"""
        try:
            risk_manager = self._initialize_risk_management()
            return risk_manager is not None
        except Exception:
            return False
    
    def _check_data_fetcher_availability(self) -> bool:
        try:
            import yfinance
            return True
        except ImportError:
            return False
    
    # DSSMS専用10ファイル生成メソッド群
    def _generate_comprehensive_report_json(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """1. dssms_comprehensive_report.json - 包括分析レポート生成"""
        try:
            report_data = {
                'backtest_results': final_results,
                'performance_data': final_results.get('performance_summary', {}),
                'switch_data': final_results.get('switch_statistics', {}),
                'system_status': self.get_system_status()
            }
            
            report_path = output_dir / "dssms_comprehensive_report.json"
            
            if self.report_generator:
                comprehensive_report = self.report_generator.generate_comprehensive_report(report_data, str(report_path))
            else:
                import json
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"[FILE 1/10] dssms_comprehensive_report.json 生成完了")
        except Exception as e:
            self.logger.error(f"[FILE 1/10] comprehensive_report.json 生成エラー: {e}")
    
    def _generate_execution_log_txt(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """3. dssms_execution_log.txt - 実行ログ生成"""
        try:
            log_path = output_dir / "dssms_execution_log.txt"
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("DSSMS 実行ログ\n")
                f.write("=" * 50 + "\n")
                f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"保有銘柄: {', '.join(self.positions.keys()) if self.positions else 'なし'} ({len(self.positions)}/{self.max_positions})\n")
                f.write(f"ポートフォリオ値: {self.portfolio_value:,.0f}円\n\n")
                
                # 日次結果ログ
                f.write("日次処理ログ:\n")
                f.write("-" * 30 + "\n")
                for idx, daily in enumerate(self.daily_results[-10:], 1):  # 最新10件
                    f.write(f"{idx}. {daily.get('date', 'N/A')}: {daily.get('symbol', 'N/A')} - ")
                    f.write(f"成功: {'Yes' if daily.get('success', False) else 'No'}\n")
            
            self.logger.info(f"[FILE 3/10] dssms_execution_log.txt 生成完了")
        except Exception as e:
            self.logger.error(f"[FILE 3/10] execution_log.txt 生成エラー: {e}")
    
    def _generate_performance_summary_json(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """4. dssms_performance_summary.json - パフォーマンスサマリー生成"""
        try:
            performance_data = {
                'portfolio_performance': final_results.get('portfolio_performance', {}),
                'execution_metadata': final_results.get('execution_metadata', {}),
                'performance_tracker': self.performance_tracker.get_performance_summary() if self.performance_tracker else {},
                'generated_at': datetime.now().isoformat()
            }
            
            summary_path = output_dir / "dssms_performance_summary.json"
            
            import json
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"[FILE 4/10] dssms_performance_summary.json 生成完了")
        except Exception as e:
            self.logger.error(f"[FILE 4/10] performance_summary.json 生成エラー: {e}")
    
    def _generate_symbol_analysis_csv(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """5. dssms_symbol_analysis.csv - 銘柄分析結果生成"""
        try:
            import pandas as pd
            
            # 日次結果から銘柄分析データ抽出
            analysis_data = []
            for daily in self.daily_results:
                analysis_data.append({
                    'date': daily.get('date'),
                    'symbol': daily.get('symbol'),
                    'success': daily.get('success', False),
                    'portfolio_value': daily.get('portfolio_value', 0),
                    'execution_time': daily.get('execution_time', 0)
                })
            
            df = pd.DataFrame(analysis_data)
            csv_path = output_dir / "dssms_symbol_analysis.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"[FILE 5/10] dssms_symbol_analysis.csv 生成完了 ({len(df)}行)")
        except Exception as e:
            self.logger.error(f"[FILE 5/10] symbol_analysis.csv 生成エラー: {e}")
    
    def _generate_risk_analysis_json(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """6. dssms_risk_analysis.json - リスク分析生成"""
        try:
            risk_data = {
                'portfolio_risk': {
                    'max_drawdown': 0.0,
                    'volatility': 0.0,
                    'var_95': 0.0
                },
                'switch_risk': {
                    'switch_count': len(self.switch_history),
                    'switch_frequency': len(self.switch_history) / max(1, len(self.daily_results))
                },
                'system_risk': {
                    'execution_failure_rate': 0.0,
                    'data_availability': 1.0
                },
                'generated_at': datetime.now().isoformat()
            }
            
            risk_path = output_dir / "dssms_risk_analysis.json"
            
            import json
            with open(risk_path, 'w', encoding='utf-8') as f:
                json.dump(risk_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"[FILE 6/10] dssms_risk_analysis.json 生成完了")
        except Exception as e:
            self.logger.error(f"[FILE 6/10] risk_analysis.json 生成エラー: {e}")
    
    def _generate_market_conditions_csv(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """7. dssms_market_conditions.csv - 市場環境データ生成"""
        try:
            import pandas as pd
            
            # 市場環境データをダミー生成（実装時は実データを使用）
            market_data = []
            for daily in self.daily_results[-30:]:  # 最新30日
                market_data.append({
                    'date': daily.get('date'),
                    'trend': 'upward',
                    'volatility': 0.15,
                    'volume_average': 1000000,
                    'market_sentiment': 'neutral'
                })
            
            df = pd.DataFrame(market_data)
            csv_path = output_dir / "dssms_market_conditions.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"[FILE 7/10] dssms_market_conditions.csv 生成完了 ({len(df)}行)")
        except Exception as e:
            self.logger.error(f"[FILE 7/10] market_conditions.csv 生成エラー: {e}")
    
    def _generate_backtest_details_json(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """8. dssms_backtest_details.json - バックテスト詳細生成"""
        try:
            details_data = {
                'backtest_configuration': {
                    'initial_capital': self.config.get('initial_capital', 1000000),
                    'symbol_switch_config': self.config.get('symbol_switch', {}),
                    'start_date': final_results.get('execution_metadata', {}).get('start_date'),
                    'end_date': final_results.get('execution_metadata', {}).get('end_date')
                },
                'execution_details': final_results.get('execution_metadata', {}),
                'daily_results_summary': {
                    'total_days': len(self.daily_results),
                    'successful_days': sum(1 for d in self.daily_results if d.get('success', False)),
                    'failed_days': sum(1 for d in self.daily_results if not d.get('success', False))
                },
                'generated_at': datetime.now().isoformat()
            }
            
            details_path = output_dir / "dssms_backtest_details.json"
            
            import json
            with open(details_path, 'w', encoding='utf-8') as f:
                json.dump(details_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"[FILE 8/10] dssms_backtest_details.json 生成完了")
        except Exception as e:
            self.logger.error(f"[FILE 8/10] backtest_details.json 生成エラー: {e}")
    
    def _generate_strategy_effectiveness_csv(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """9. dssms_strategy_effectiveness.csv - 戦略効果分析生成"""
        try:
            import pandas as pd
            
            # 戦略効果データをダミー生成（実装時は実データを使用）
            strategy_data = [
                {'strategy': 'VWAPBreakoutStrategy', 'effectiveness_score': 0.75, 'win_rate': 0.60},
                {'strategy': 'MomentumInvestingStrategy', 'effectiveness_score': 0.65, 'win_rate': 0.55},
                {'strategy': 'BreakoutStrategy', 'effectiveness_score': 0.70, 'win_rate': 0.58},
                {'strategy': 'ContrarianStrategy', 'effectiveness_score': 0.50, 'win_rate': 0.45},
                {'strategy': 'GCStrategy', 'effectiveness_score': 0.55, 'win_rate': 0.48}
            ]
            
            df = pd.DataFrame(strategy_data)
            csv_path = output_dir / "dssms_strategy_effectiveness.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"[FILE 9/10] dssms_strategy_effectiveness.csv 生成完了 ({len(df)}行)")
        except Exception as e:
            self.logger.error(f"[FILE 9/10] strategy_effectiveness.csv 生成エラー: {e}")
    
    def _generate_consolidated_report_txt(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """10. dssms_consolidated_report.txt - 統合レポート生成"""
        try:
            report_path = output_dir / "dssms_consolidated_report.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("DSSMS 統合レポート\n")
                f.write("=" * 60 + "\n")
                f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"レポートID: dssms_{datetime.now().strftime('%Y%m%d_%H%M%S')}\n\n")
                
                # 実行概要
                f.write("実行概要:\n")
                f.write("-" * 20 + "\n")
                exec_meta = final_results.get('execution_metadata', {})
                f.write(f"期間: {exec_meta.get('start_date', 'N/A')} - {exec_meta.get('end_date', 'N/A')}\n")
                f.write(f"取引日数: {exec_meta.get('trading_days', 0)}日\n")
                f.write(f"成功日数: {exec_meta.get('successful_days', 0)}日\n\n")
                
                # パフォーマンス概要
                f.write("パフォーマンス概要:\n")
                f.write("-" * 20 + "\n")
                portfolio_perf = final_results.get('portfolio_performance', {})
                f.write(f"最終資本: {portfolio_perf.get('final_capital', 0):,.0f}円\n")
                f.write(f"総収益率: {portfolio_perf.get('total_return_rate', 0):.2%}\n")
                f.write(f"成功率: {portfolio_perf.get('success_rate', 0):.1%}\n\n")
                
                # 銘柄切替概要
                f.write("銘柄切替概要:\n")
                f.write("-" * 20 + "\n")
                f.write(f"総切替回数: {len(self.switch_history)}回\n")
                if self.switch_history:
                    recent_switches = self.switch_history[-5:]
                    f.write("最新切替履歴:\n")
                    for switch in recent_switches:
                        f.write(f"  {switch.get('date', 'N/A')}: {switch.get('from_symbol', 'N/A')} -> {switch.get('to_symbol', 'N/A')}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("レポート終了\n")
            
            self.logger.info(f"[FILE 10/10] dssms_consolidated_report.txt 生成完了")
        except Exception as e:
            self.logger.error(f"[FILE 10/10] consolidated_report.txt 生成エラー: {e}")
    
    # 新規DSSMS専用10ファイル生成メソッド群（2026-01-08追加）
    
    def _generate_execution_results_json(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """3. execution_results.json - 実行結果（銘柄プレフィックス削除）"""
        try:
            execution_path = output_dir / "execution_results.json"
            
            execution_data = {
                "total_trades": 0,
                "successful_trades": 0,
                "failed_trades": 0,
                "execution_rate": 0.0
            }
            
            # final_resultsからexecution_resultsを抽出
            if 'execution_results' in final_results:
                exec_results = final_results['execution_results']
                if isinstance(exec_results, list) and exec_results:
                    execution_data = exec_results[0]  # 最初の結果を使用
                elif isinstance(exec_results, dict):
                    execution_data = exec_results
            
            with open(execution_path, 'w', encoding='utf-8') as f:
                json.dump(execution_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"[FILE 3/10] execution_results.json 生成完了")
        except Exception as e:
            self.logger.error(f"[FILE 3/10] execution_results.json 生成エラー: {e}")
    
    def _generate_performance_metrics_json(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """4. performance_metrics.json - パフォーマンス指標（銘柄プレフィックス削除）"""
        try:
            metrics_path = output_dir / "performance_metrics.json"
            
            metrics_data = {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0
            }
            
            # final_resultsからperformance_metricsを抽出
            portfolio_perf = final_results.get('portfolio_performance', {})
            if portfolio_perf:
                metrics_data.update({
                    "total_return": portfolio_perf.get('total_return_rate', 0.0),
                    "sharpe_ratio": portfolio_perf.get('sharpe_ratio', 0.0),
                    "max_drawdown": portfolio_perf.get('max_drawdown', 0.0),
                    "win_rate": portfolio_perf.get('success_rate', 0.0)
                })
            
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"[FILE 4/10] performance_metrics.json 生成完了")
        except Exception as e:
            self.logger.error(f"[FILE 4/10] performance_metrics.json 生成エラー: {e}")
    
    def _generate_performance_summary_csv(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """5. performance_summary.csv - パフォーマンス要約（銘柄プレフィックス削除）"""
        try:
            summary_path = output_dir / "performance_summary.csv"
            
            portfolio_perf = final_results.get('portfolio_performance', {})
            exec_meta = final_results.get('execution_metadata', {})
            
            summary_data = {
                'Metric': ['初期資本', '最終資本', '総収益率', '成功率', '取引日数', '成功日数'],
                'Value': [
                    portfolio_perf.get('initial_capital', 1000000),
                    portfolio_perf.get('final_capital', 1000000),
                    f"{portfolio_perf.get('total_return_rate', 0):.2%}",
                    f"{portfolio_perf.get('success_rate', 0):.1%}",
                    exec_meta.get('trading_days', 0),
                    exec_meta.get('successful_days', 0)
                ]
            }
            
            df = pd.DataFrame(summary_data)
            df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"[FILE 5/10] performance_summary.csv 生成完了")
        except Exception as e:
            self.logger.error(f"[FILE 5/10] performance_summary.csv 生成エラー: {e}")
    
    def _generate_trade_analysis_json(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """6. trade_analysis.json - トレード分析（銘柄プレフィックス削除）"""
        try:
            analysis_path = output_dir / "trade_analysis.json"
            
            analysis_data = {
                "total_trades": len(self.daily_results),
                "profitable_trades": 0,
                "losing_trades": 0,
                "break_even_trades": 0
            }
            
            # 日次結果から取引分析を算出
            profitable = 0
            losing = 0
            break_even = 0
            
            for daily_result in self.daily_results:
                pnl = daily_result.get('daily_pnl', 0)
                if pnl > 0:
                    profitable += 1
                elif pnl < 0:
                    losing += 1
                else:
                    break_even += 1
            
            analysis_data.update({
                "profitable_trades": profitable,
                "losing_trades": losing,
                "break_even_trades": break_even
            })
            
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"[FILE 6/10] trade_analysis.json 生成完了")
        except Exception as e:
            self.logger.error(f"[FILE 6/10] trade_analysis.json 生成エラー: {e}")
    
    def _generate_all_transactions_csv(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """6.5. all_transactions.csv - 詳細取引データ（ComprehensiveReporter相当）"""
        try:
            import csv
            
            transactions_path = output_dir / "all_transactions.csv"
            
            # デバッグ: self.daily_resultsの内容確認（2026-02-05追加）
            print(f"\n[DEBUG] ========== CSV出力前のデバッグ ==========")
            print(f"[DEBUG] self.daily_results の件数: {len(self.daily_results) if self.daily_results else 0}")
            
            # execution_detailsから詳細取引データを抽出
            all_execution_details = []
            for i, daily_result in enumerate(self.daily_results):
                daily_execution_details = daily_result.get('execution_details', [])
                if daily_execution_details:
                    print(f"[DEBUG] daily_results[{i}] ({daily_result.get('date', 'N/A')}): {len(daily_execution_details)}件のexecution_details")
                    for j, detail in enumerate(daily_execution_details):
                        print(f"[DEBUG]   [{j}] action={detail.get('action')}, symbol={detail.get('symbol')}, price={detail.get('price')}, shares={detail.get('shares')}")
                all_execution_details.extend(daily_execution_details)
            
            print(f"[DEBUG] all_execution_details 合計: {len(all_execution_details)}件")
            print(f"[DEBUG] CSV出力パス: {transactions_path}")
            print(f"[DEBUG] ==========================================\n")
            
            # BUY/SELLペアリングして取引レコードを作成
            trades = self._convert_execution_details_to_trades(all_execution_details)
            
            # Cycle 23デバッグ: tradesの内容確認
            self.logger.info(f"[CSV_DEBUG] trades type={type(trades)}, len={len(trades) if trades else 0}")
            if trades:
                self.logger.info(f"[CSV_DEBUG] trades[0]={trades[0]}")
            
            # CSV作成
            with open(transactions_path, 'w', newline='', encoding='utf-8') as f:
                if trades:
                    # ComprehensiveReporter相当のカラム
                    fieldnames = ['symbol', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 
                                'shares', 'pnl', 'return_pct', 'holding_period_days', 'strategy_name', 
                                'position_value', 'is_forced_exit']
                    
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for trade in trades:
                        writer.writerow(trade)
                    
                    self.logger.info(f"[FILE 6.5/11] all_transactions.csv 生成完了: {len(trades)}件の取引")
                else:
                    # 取引がない場合もヘッダーは生成
                    f.write("symbol,entry_date,entry_price,exit_date,exit_price,shares,pnl,return_pct,holding_period_days,strategy_name,position_value,is_forced_exit\n")
                    self.logger.info(f"[FILE 6.5/11] all_transactions.csv 生成完了: 0件の取引")
                    
        except Exception as e:
            self.logger.error(f"[FILE 6.5/11] all_transactions.csv 生成エラー: {e}")
    
    def _convert_execution_details_to_trades(self, execution_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """execution_detailsをBUY/SELLペアリングして取引レコード形式に変換"""
        try:
            import pandas as pd
            from collections import defaultdict
            from datetime import datetime, timedelta
            
            if not execution_details:
                return []
            
            # DSSMS修正: 銘柄切替時の跨銘柄ペアリング処理
            # 時系列順でBUY/SELLペアリング（異なる銘柄でも可）
            all_orders = []
            for detail in execution_details:
                action = detail.get('action', '').upper()
                ts = detail.get('timestamp')
                
                # timestamp型統一: str → datetime変換（2026-02-05 修正）
                if isinstance(ts, str):
                    try:
                        ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                        detail['timestamp'] = ts  # 元のdictも更新
                    except ValueError:
                        self.logger.warning(f"[TIMESTAMP_PARSE] 変換失敗: {ts}")
                
                self.logger.debug(f"[DEBUG_TRADE] execution_detail: action={action}, symbol={detail.get('symbol')}, price={detail.get('price')}, shares={detail.get('shares')}, timestamp={ts}, type={type(ts)}")
                if action in ['BUY', 'SELL']:
                    all_orders.append(detail)
            
            # 時系列順でソート（datetime vs str エラー修正: 2026-02-05）
            self.logger.debug(f"[SORT_DEBUG] ソート前: all_orders count={len(all_orders)}")
            for i, order in enumerate(all_orders[:3]):  # 最初の3件のみ
                ts = order.get('timestamp')
                self.logger.debug(f"[SORT_DEBUG] order[{i}]: timestamp={ts}, type={type(ts)}, value={repr(ts)}")
            
            all_orders.sort(key=lambda x: x.get('timestamp', datetime.min))
            
            # FIFO ペアリング（銘柄別管理）
            buy_stacks = {}  # {symbol: [buy_orders]}
            trades = []
            
            for order in all_orders:
                action = order.get('action', '').upper()
                symbol = order.get('symbol', order.get('ticker', ''))
                
                if action == 'BUY':
                    if symbol not in buy_stacks:
                        buy_stacks[symbol] = []
                    buy_stacks[symbol].append(order)
                    
                elif action == 'SELL':
                    if symbol in buy_stacks and buy_stacks[symbol]:  # ペアリング可能
                        buy_order = buy_stacks[symbol].pop(0)  # FIFO (同銘柄のみ)
                        
                        # 取引レコード作成
                        entry_date = buy_order.get('timestamp', '')
                        exit_date = order.get('timestamp', '')
                        entry_price = buy_order.get('executed_price', buy_order.get('price', 0.0))
                        
                        # Cycle 5修正: exit_priceは常にSELLのexecution_detailから取得
                        # 理由: backtest_daily()でforce_close時に正しい価格が設定されているはず
                        buy_symbol = buy_order.get('symbol', '')
                        sell_symbol = order.get('symbol', '')
                        
                        # SELLのexecution_detailから価格取得（跨銘柄切替でも同様）
                        exit_price = order.get('executed_price', order.get('price', 0.0))
                        
                        # ペアリング成功ログ
                        self.logger.info(
                            f"[TRADE_MATCH] {symbol}: "
                            f"BUY@{entry_price:.2f}({entry_date}) "
                            f"-> SELL@{exit_price:.2f}({exit_date})"
                        )
                        # ペアリング成功ログ
                        self.logger.info(
                            f"[TRADE_MATCH] {symbol}: "
                            f"BUY@{entry_price:.2f}({entry_date}) "
                            f"-> SELL@{exit_price:.2f}({exit_date})"
                        )
                        
                        if buy_symbol != sell_symbol:
                            # 跨銘柄切替: ログ記録のみ
                            self.logger.warning(
                                f"[CROSS_SYMBOL] 跨銘柄切替検出: BUY={buy_symbol}, SELL={sell_symbol}, "
                                f"entry_price={entry_price:.2f}, exit_price={exit_price:.2f}"
                            )
                        
                        shares = buy_order.get('quantity', buy_order.get('shares', 0))
                        strategy_name = buy_order.get('strategy_name', buy_order.get('strategy', ''))
                        
                        if entry_price > 0 and exit_price > 0 and shares > 0:
                            pnl = (exit_price - entry_price) * shares
                            return_pct = ((exit_price - entry_price) / entry_price) if entry_price > 0 else 0
                            position_value = entry_price * shares
                            
                            # 保有期間計算
                            try:
                                entry_dt = pd.to_datetime(entry_date)
                                exit_dt = pd.to_datetime(exit_date)
                                holding_days = (exit_dt - entry_dt).days
                            except:
                                holding_days = 0
                            
                            # 銘柄切替判定
                            is_cross_symbol = (buy_symbol != sell_symbol)
                            # 強制決済判定
                            is_forced_exit = order.get('status') == 'force_closed' or is_cross_symbol
                            
                            # 表示用銘柄（エントリー銘柄を採用）
                            display_symbol = buy_symbol
                            
                            trade_record = {
                                'symbol': display_symbol,
                                'entry_date': entry_date,
                                'entry_price': entry_price,
                                'exit_date': exit_date,
                                'exit_price': exit_price,
                                'shares': shares,
                                'pnl': pnl,
                                'return_pct': return_pct,
                                'holding_period_days': holding_days,
                                'strategy_name': strategy_name,
                                'position_value': position_value,
                                'is_forced_exit': is_forced_exit
                            }
                            
                            trades.append(trade_record)
                            
                            # 銘柄切替取引の特別ログ
                            if is_cross_symbol:
                                self.logger.info(f"[TRADE_CONVERSION] 銘柄切替取引: {buy_symbol}(BUY) -> {sell_symbol}(SELL), PnL={pnl:,.0f}円")
                            else:
                                self.logger.info(f"[TRADE_CONVERSION] 通常取引: {display_symbol}, PnL={pnl:,.0f}円")
                    else:
                        # ペアリング失敗（対応するBUYがない）
                        self.logger.warning(
                            f"[TRADE_MATCH] {symbol}: "
                            f"対応するBUY注文が見つかりません (SELL日:{order.get('timestamp','?')})"
                        )
            
            # 未決済BUY注文処理
            for symbol, symbol_buy_stack in buy_stacks.items():
                for buy in symbol_buy_stack:
                    entry_date = buy.get('timestamp', '')
                    entry_price = buy.get('executed_price', buy.get('price', 0.0))
                    shares = buy.get('quantity', buy.get('shares', 0))
                    strategy_name = buy.get('strategy_name', buy.get('strategy', ''))
                    symbol = buy.get('symbol', '')
                    
                    if entry_price > 0 and shares > 0:
                        position_value = entry_price * shares
                        
                        # 未決済取引レコード作成
                        trade_record = {
                            'symbol': symbol,
                            'entry_date': entry_date,
                            'entry_price': entry_price,
                            'exit_date': '',  # 未決済
                            'exit_price': 0.0,  # 未決済
                            'shares': shares,
                            'pnl': 0.0,  # 未決済のためPnL未確定
                            'return_pct': 0.0,  # 未決済のため収益率未確定
                            'holding_period_days': 0,
                            'strategy_name': strategy_name,
                            'position_value': position_value,
                            'is_forced_exit': False
                        }
                        
                        trades.append(trade_record)
                        self.logger.info(f"[TRADE_CONVERSION] 未決済BUY注文を取引レコードに追加: {symbol} {shares}株 @ {entry_price}")
            
            self.logger.info(f"[TRADE_CONVERSION] BUY={len([o for o in all_orders if o.get('action', '').upper() == 'BUY'])}, SELL={len([o for o in all_orders if o.get('action', '').upper() == 'SELL'])}")
            self.logger.info(f"[TRADE_CONVERSION] 生成された取引レコード: {len(trades)}件")
            return trades
            
        except Exception as e:
            self.logger.error(f"[TRADE_CONVERSION] エラー: {e}")
            return []

    def _generate_portfolio_equity_curve_csv(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """7. portfolio_equity_curve.csv - ポートフォリオ資産曲線（Cycle 10修正版）"""
        try:
            curve_path = output_dir / "portfolio_equity_curve.csv"
            
            # Cycle 10修正: daily_resultsからcash_balance, position_value, total_portfolio_valueを直接使用
            curve_data = []
            
            for daily_result in self.daily_results:
                # Cycle 10-2: 日付処理修正（文字列/datetime対応）
                date_value = daily_result.get('adjusted_target_date', daily_result.get('date', datetime.now()))
                if isinstance(date_value, str):
                    date = date_value  # すでに文字列の場合はそのまま使用
                else:
                    date = date_value.strftime('%Y-%m-%d')  # datetime型の場合はstrftimeでフォーマット
                
                # Cycle 10: 実際のcash_balance, position_value, total_portfolio_valueを使用
                cash_balance = daily_result.get('cash_balance', 1000000)
                position_value = daily_result.get('position_value', 0.0)
                total_value = daily_result.get('total_portfolio_value', cash_balance + position_value)
                
                curve_data.append({
                    'date': date,
                    'cash_balance': cash_balance,
                    'position_value': position_value,
                    'total_value': total_value,
                    'symbol': daily_result.get('symbol', ', '.join(self.positions.keys()) if self.positions else 'N/A')
                })
            
            if curve_data:
                df = pd.DataFrame(curve_data)
            else:
                # 空データの場合のフォールバック
                df = pd.DataFrame({
                    'date': [datetime.now().strftime('%Y-%m-%d')],
                    'cash_balance': [1000000],
                    'position_value': [0],
                    'total_value': [1000000],
                    'symbol': list(self.positions.keys())
                })
            
            df.to_csv(curve_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"[FILE 7/10] portfolio_equity_curve.csv 生成完了: {len(curve_data)}日分のデータ")
        except Exception as e:
            self.logger.error(f"[FILE 7/10] portfolio_equity_curve.csv 生成エラー: {e}")
    
    def _generate_comprehensive_report_txt(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """8. comprehensive_report.txt - 包括レポート（ComprehensiveReporter相当の詳細分析）"""
        try:
            report_path = output_dir / "comprehensive_report.txt"
            
            # 詳細取引データを取得
            all_execution_details = []
            for daily_result in self.daily_results:
                daily_execution_details = daily_result.get('execution_details', [])
                all_execution_details.extend(daily_execution_details)
            
            trades = self._convert_execution_details_to_trades(all_execution_details)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                # ヘッダー
                f.write("=" * 80 + "\n")
                f.write("DSSMS マルチ戦略動的バックテスト包括レポート\n")
                f.write("=" * 80 + "\n")
                f.write(f"保有銘柄: {', '.join(self.positions.keys()) if self.positions else 'なし'} ({len(self.positions)}/{self.max_positions})\n")
                f.write(f"レポート生成日時: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
                f.write(f"レポート種別: DSSMS 統合戦略実行結果\n\n")
                
                # 1. システム実行概要
                f.write("1. システム実行概要\n")
                f.write("-" * 40 + "\n")
                f.write(f"総取引回数: {len(trades)}\n")
                
                if self.daily_results:
                    start_date = self.daily_results[0].get('date', 'N/A')
                    end_date = self.daily_results[-1].get('date', 'N/A')
                    f.write(f"データ期間: {start_date} - {end_date}\n")
                    f.write(f"取引日数: {len(self.daily_results)}\n")
                
                # ポートフォリオ統計
                portfolio_perf = final_results.get('portfolio_performance', {})
                initial_capital = portfolio_perf.get('initial_capital', 1000000)
                final_capital = portfolio_perf.get('final_capital', 1000000)
                total_return_rate = portfolio_perf.get('total_return_rate', 0)
                
                f.write(f"初期資金: ¥{initial_capital:,}\n")
                f.write(f"最終ポートフォリオ値: ¥{final_capital:,}\n")
                f.write(f"総リターン: {total_return_rate:.2%}\n")
                
                # 勝率計算
                winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
                losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])
                win_rate = (winning_trades / max(1, len(trades))) * 100
                f.write(f"勝率: {win_rate:.2f}%\n\n")
                
                # 2. パフォーマンス統計
                f.write("2. パフォーマンス統計\n")
                f.write("-" * 40 + "\n")
                f.write(f"総取引数: {len(trades)}\n")
                f.write(f"勝ちトレード数: {winning_trades}\n")
                f.write(f"負けトレード数: {losing_trades}\n")
                f.write(f"勝率: {win_rate:.2f}%\n\n")
                
                # 損益統計
                if trades:
                    profits = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0]
                    losses = [t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]
                    
                    avg_profit = sum(profits) / max(1, len(profits)) if profits else 0
                    avg_loss = sum(losses) / max(1, len(losses)) if losses else 0
                    max_profit = max(profits) if profits else 0
                    max_loss = min(losses) if losses else 0
                    
                    total_profit = sum(profits)
                    total_loss = sum(losses)
                    net_profit = total_profit + total_loss  # lossは負の値
                    
                    f.write(f"平均利益: ¥{avg_profit:,.0f}\n")
                    f.write(f"平均損失: ¥{avg_loss:,.0f}\n")
                    f.write(f"最大利益: ¥{max_profit:,.0f}\n")
                    f.write(f"最大損失: ¥{max_loss:,.0f}\n\n")
                    
                    f.write(f"総利益: ¥{total_profit:,.0f}\n")
                    f.write(f"総損失: ¥{total_loss:,.0f}\n")
                    f.write(f"純利益: ¥{net_profit:,.0f}\n")
                    
                    profit_factor = abs(total_profit / total_loss) if total_loss != 0 else 0
                    f.write(f"プロフィットファクター: {profit_factor:.2f}\n\n")
                else:
                    f.write("取引データなし\n\n")
                
                # 3. 期待値分析
                f.write("3. 期待値分析\n")
                f.write("-" * 40 + "\n")
                if trades:
                    avg_pnl_per_trade = sum([t.get('pnl', 0) for t in trades]) / len(trades)
                    f.write(f"システム期待値 (1トレードあたり):\n")
                    f.write(f"  金額: ¥{avg_pnl_per_trade:,.0f}\n")
                    f.write(f"  基準: {len(trades)}取引の平均\n\n")
                    
                    # 戦略別期待値（簡略化）
                    strategy_stats = {}
                    for trade in trades:
                        strategy = trade.get('strategy_name', 'Unknown')
                        if strategy not in strategy_stats:
                            strategy_stats[strategy] = {'pnl': 0, 'count': 0, 'wins': 0}
                        strategy_stats[strategy]['pnl'] += trade.get('pnl', 0)
                        strategy_stats[strategy]['count'] += 1
                        if trade.get('pnl', 0) > 0:
                            strategy_stats[strategy]['wins'] += 1
                    
                    f.write("戦略別期待値:\n")
                    for strategy, stats in strategy_stats.items():
                        expected_value = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
                        win_rate = (stats['wins'] / stats['count']) * 100 if stats['count'] > 0 else 0
                        f.write(f"  {strategy}:\n")
                        f.write(f"    期待値: ¥{expected_value:,.0f}\n")
                        f.write(f"    取引数: {stats['count']}\n")
                        f.write(f"    勝率: {win_rate:.2f}%\n\n")
                    
                    # 期待値統計
                    daily_expected = avg_pnl_per_trade if len(self.daily_results) > 0 else 0
                    monthly_expected = daily_expected * 20  # 概算
                    yearly_expected = monthly_expected * 12
                    
                    f.write("期待値統計:\n")
                    f.write(f"  日次期待値: ¥{daily_expected:,.0f}\n")
                    f.write(f"  月次期待値: ¥{monthly_expected:,.0f}\n")
                    f.write(f"  年次期待値: ¥{yearly_expected:,.0f}\n\n")
                else:
                    f.write("取引データがないため期待値計算不可\n\n")
                
                # 4. 取引詳細
                f.write("4. 取引詳細\n")
                f.write("-" * 40 + "\n")
                f.write(f"総取引数: {len(trades)}\n\n")
                
                if trades:
                    f.write("取引履歴 (最初の10件):\n\n")
                    f.write("No.  戦略                   エントリー日       エグジット日       価格         価格         PnL         \n")
                    f.write("                                                    (エントリー)    (エグジット)    (円)         \n")
                    f.write("-" * 90 + "\n")
                    
                    for i, trade in enumerate(trades[:10], 1):
                        strategy = trade.get('strategy_name', 'Unknown')[:20]  # 戦略名を20文字で切り詰め
                        entry_date = trade.get('entry_date', '')[:10]  # 日付部分のみ
                        exit_date = trade.get('exit_date', '')[:10]
                        entry_price = trade.get('entry_price', 0)
                        exit_price = trade.get('exit_price', 0)
                        pnl = trade.get('pnl', 0)
                        
                        f.write(f"{i:<4} {strategy:<20} {entry_date:<14} {exit_date:<14} {entry_price:<10.2f} {exit_price:<10.2f} {pnl:<10.0f}\n")
                    
                    if len(trades) > 10:
                        f.write(f"... 他{len(trades) - 10}件の取引\n")
                else:
                    f.write("取引履歴なし\n")
                
                f.write("\n")
                
                # 5. DSSMS統計
                f.write("5. DSSMS統計\n")
                f.write("-" * 40 + "\n")
                f.write(f"銘柄切替回数: {len(self.switch_history)}回\n")
                f.write(f"平均実行時間: {self._calculate_average_execution_time():.0f}ms\n")
                if self.daily_results:
                    success_rate = (len([r for r in self.daily_results if r.get('success')]) / len(self.daily_results)) * 100
                    f.write(f"日次成功率: {success_rate:.1f}%\n")
                
                # 6. フッター
                f.write("\n" + "=" * 80 + "\n")
                f.write("レポート終了\n")
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"[FILE 8/11] comprehensive_report.txt 生成完了 (ComprehensiveReporter相当)")
        except Exception as e:
            self.logger.error(f"[FILE 8/11] comprehensive_report.txt 生成エラー: {e}")
    
    def _calculate_average_execution_time(self) -> float:
        """平均実行時間を計算"""
        if not self.daily_results:
            return 0.0
        
        execution_times = [r.get('execution_time_ms', 0) for r in self.daily_results]
        return sum(execution_times) / len(execution_times) if execution_times else 0.0
    
    def _generate_summary_txt(self, output_dir: Path, final_results: Dict[str, Any]) -> None:
        """9. summary.txt - サマリー（銘柄プレフィックス削除）"""
        try:
            summary_path = output_dir / "summary.txt"
            
            portfolio_perf = final_results.get('portfolio_performance', {})
            exec_meta = final_results.get('execution_metadata', {})
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("DSSMS 実行サマリー\n")
                f.write("=" * 30 + "\n")
                f.write(f"保有銘柄: {', '.join(self.positions.keys()) if self.positions else 'なし'} ({len(self.positions)}/{self.max_positions})\n")
                f.write(f"ポートフォリオ価値: {self.portfolio_value:,.0f}円\n")
                f.write(f"総収益率: {portfolio_perf.get('total_return_rate', 0):.2%}\n")
                f.write(f"成功率: {portfolio_perf.get('success_rate', 0):.1%}\n")
                f.write(f"取引日数: {exec_meta.get('trading_days', 0)}日\n")
                f.write(f"銘柄切替回数: {len(self.switch_history)}回\n")
                f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            self.logger.info(f"[FILE 9/10] summary.txt 生成完了")
        except Exception as e:
            self.logger.error(f"[FILE 9/10] summary.txt 生成エラー: {e}")
    
    # 既存メソッド継続
    
    def _prepare_market_data_for_analysis(self, symbols: List[str], target_date: datetime) -> Optional[Dict[str, Any]]:
        """
        
        Args:
            symbols: 対象銘柄リスト
            target_date: 対象日付
            
        Returns:
        """
        try:
            market_data = {}
            
            for symbol in symbols:
                stock_data, _ = self._get_symbol_data(symbol, target_date)
                if stock_data is not None and not stock_data.empty:
                    market_data[symbol] = stock_data
                    
            if len(market_data) > 0:
                return market_data
            else:
                return None
                
        except Exception as e:
            return None
    
    def _run_advanced_analysis_sync(self, symbols: List[str], market_data: Dict[str, Any], 
                                   analysis_params: Optional[Dict[str, Any]] = None) -> Optional[List[Any]]:
        """
        AdvancedRankingEngine分析の同期実行 (TODO-DSSMS-004.1)
        TODO-DSSMS-004.2統合最適化対応版
        
        Args:
            symbols: 対象銘柄リスト  
            
        Returns:
        """
        try:
            import asyncio
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if analysis_params is None:
                analysis_params = {
                    'analysis_depth': 'comprehensive',
                    'enable_parallel': len(symbols) > 5,
                    'timeout_seconds': 30
                }
            
            # TODO-DSSMS-004.2: 基盤結果統合処理
            if analysis_params.get('base_ranking_results') and analysis_params.get('reuse_calculations'):
                analysis_params['optimization_mode'] = 'integrated'
                analysis_params['base_calculations_reuse'] = True
            
            ranking_results = loop.run_until_complete(
                self.advanced_ranking_engine.analyze_symbols_advanced(
                    symbols, market_data, analysis_params
                )
            )
            
            integration_info = "統合最適化" if analysis_params.get('reuse_calculations') else "従来方式"
            self.logger.info(f"高度分析完了 ({integration_info}): {len(ranking_results) if ranking_results else 0}件の結果")
            return ranking_results
            
        except Exception as e:
            return None
    
    def _select_best_symbol_from_ranking(self, ranking_results: List[Any]) -> Optional[str]:
        """
        Select best symbol from AdvancedRankingEngine results.
        
        Args:
            ranking_results: Analysis results from AdvancedRankingEngine
            
        Returns:
            str: Best symbol code or None if no valid results
        """
        try:
            if not ranking_results or len(ranking_results) == 0:
                return None
            
            best_result = None
            best_score = -float('inf')
            
            for result in ranking_results:
                # AdvancedRankingResult型の属性アクセス
                try:
                    if hasattr(result, 'total_score'):
                        score = result.total_score
                        symbol = result.symbol
                    else:
                        # 辞書形式の場合
                        score = result.get('total_score', 0.0)
                        symbol = result.get('symbol', '')
                    
                    if score > best_score and symbol:
                        best_score = score
                        best_result = symbol
                        
                except Exception as e:
                    continue
            
            if best_result:
                self.logger.info(f"最優秀銘柄選択: {best_result} (スコア: {best_score:.4f})")
                return best_result
            else:
                self.logger.warning("有効な銘柄スコアが見つかりませんでした")
                return None
                
        except Exception as e:
            return None


def main():
    parser = argparse.ArgumentParser(description='DSSMS Integrated Backtest System')
    parser.add_argument('--start-date', type=str, help='開始日 (YYYY-MM-DD形式)', default='2023-01-01')
    parser.add_argument('--end-date', type=str, help='終了日 (YYYY-MM-DD形式)', default='December 31, 2023')
    parser.add_argument('--fixed-symbol', type=str, help='固定銘柄モード: 指定銘柄のみでバックテスト (例: 8306.T)', default=None)
    args = parser.parse_args()
    
    print("=" * 60)
    
    try:
        # 1. 初期化テスト
        
        # Task 1実装 (2026-01-13): 設定値を最適化
        # 重要: SymbolSwitchManagerは'switch_management'キーを期待
        config = {
            'initial_capital': 1000000,
            'symbol_switch': {
                'switch_management': {
                    'min_holding_days': 5,  # 10 → 5に変更（Sprint 1検証用、2026-02-09）
                    'max_switches_per_month': 5,  # 8回 → 5回に変更（Task 1）
                    'switch_cost_rate': 0.001  # 0.1%（明示的に追加）
                }
            }
        }
        
        backtester = DSSMSIntegratedBacktester(config)
        print("[SUCCESS] 初期化成功")
        
        # 2. システム状態確認
        print(f"\n[STATUS] システム状態確認:")
        status = backtester.get_system_status()
        print(f"[SUCCESS] システム状態取得成功:")
        print(f"  - DSS Core V3: {'利用可能' if status['dss_available'] else '使用不可'}")
        print(f"  - リスク管理: {'利用可能' if status['risk_management_available'] else '使用不可'}")
        print(f"  - 初期資本: {status['portfolio_value']:,.0f}円")
        
        # 3. カスタム期間バックテストテスト
        print(f"\n[BACKTEST] カスタム期間バックテストテスト:")
        
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError as e:
            print("正しい形式: YYYY-MM-DD (例: 2023-01-01)")
            return
        
        # 固定銘柄モード処理
        if args.fixed_symbol:
            print(f"\n[FIXED-SYMBOL MODE] 固定銘柄モード: {args.fixed_symbol}")
            target_symbols = [args.fixed_symbol]
        else:
            target_symbols = None  # 全銘柄(日経225自動選択)
        
        results = backtester.run_dynamic_backtest(start_date, end_date, target_symbols)
        
        if 'error' in results:
            print(f"  - 生成時刻: {results['execution_metadata'].get('generated_at', 'N/A')}")
            return
        
        print(f"[SUCCESS] バックテスト実行成功:")
        
        exec_meta = results.get('execution_metadata', {})
        portfolio_perf = results.get('portfolio_performance', {})
        
        print(f"  - 実行期間: {exec_meta.get('start_date', 'N/A')} -> {exec_meta.get('end_date', 'N/A')}")
        print(f"  - 取引日数: {exec_meta.get('trading_days', 0)}日")
        print(f"  - 成功日数: {exec_meta.get('successful_days', 0)}日")
        
        if portfolio_perf:
            print(f"  - 成功率: {portfolio_perf.get('success_rate', 0):.1%}")
            print(f"  - 最終資本: {portfolio_perf.get('final_capital', 0):,.0f}円")
            print(f"  - 総収益率: {portfolio_perf.get('total_return_rate', 0):.2%}")
        
        switch_history = results.get('switch_history', [])
        print(f"  - 銘柄切替: {len(switch_history)}回")
        
        perf_summary = results.get('performance_summary', {})
        
        if perf_summary:
            overall_status = perf_summary.get('overall', {}).get('status', 'データなし')
            exec_time = perf_summary.get('execution', {}).get('average_time_ms', 0)
            reliability = perf_summary.get('reliability', {}).get('success_rate', 0)
            
            print(f"  - 総合評価: {overall_status}")
            print(f"  - 平均実行時間: {exec_time:.0f}ms")
            print(f"  - システム信頼性: {reliability:.1%}")
        else:
            pass
        
        
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
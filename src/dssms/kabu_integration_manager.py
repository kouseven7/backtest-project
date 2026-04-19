"""
DSSMS Phase 4 Task 4.1: kabu_api統合マネージャー
kabu STATIONとの完全統合システム

既存DSSMSシステムとの統合を考慮した設計
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from enum import Enum
import pandas as pd

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 既存DSSMSコンポーネントのインポート
try:
    from .market_condition_monitor import MarketConditionMonitor
    from .hierarchical_ranking_system import HierarchicalRankingSystem
    from .comprehensive_scoring_engine import ComprehensiveScoringEngine
    from .intelligent_switch_manager import IntelligentSwitchManager
    from .nikkei225_screener import Nikkei225Screener
except ImportError:
    # 絶対パスでの再試行
    sys.path.append(str(project_root / "src" / "dssms"))
    from market_condition_monitor import MarketConditionMonitor
    from hierarchical_ranking_system import HierarchicalRankingSystem
    from comprehensive_scoring_engine import ComprehensiveScoringEngine
    from intelligent_switch_manager import IntelligentSwitchManager
    from nikkei225_screener import Nikkei225Screener

# 既存リスク管理のインポート
from config.risk_management import RiskManagement
from config.logger_config import setup_logger

class AuthenticationStatus(Enum):
    """認証状態"""
    NOT_AUTHENTICATED = "not_authenticated"
    AUTHENTICATED = "authenticated"
    TOKEN_EXPIRED = "token_expired"
    CONNECTION_ERROR = "connection_error"

class APILimitStatus(Enum):
    """API制限状態"""
    AVAILABLE = "available"
    APPROACHING_LIMIT = "approaching_limit"
    LIMIT_EXCEEDED = "limit_exceeded"

@dataclass
class SymbolPriorityInfo:
    """銘柄優先度情報"""
    symbol: str
    priority: int
    update_frequency: str
    confidence: str
    last_updated: datetime
    exchange: int = 1

@dataclass
class OrderExecutionResult:
    """注文実行結果"""
    success: bool
    order_id: Optional[str]
    executed_price: Optional[float]
    error_message: Optional[str]
    timestamp: datetime

class KabuAuthManager:
    """kabu STATION認証管理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_token = None
        self.token_expires_at = None
        self.logger = setup_logger(__name__)
        
    def authenticate(self) -> bool:
        """kabu STATION認証"""
        try:
            # 環境に応じたパスワード取得
            password = self._get_api_password()
            
            obj = {'APIPassword': password}
            json_data = json.dumps(obj).encode('utf8')

            base_url = self.config.get('base_url', 'http://localhost:18080/kabusapi')
            url = f"{base_url}/token"
            req = urllib.request.Request(url, json_data, method='POST')
            req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req, timeout=self.config['timeout_seconds']) as res:
                if res.status == 200:
                    content = json.loads(res.read())
                    self.current_token = content.get('Token')
                    # トークンの有効期限を設定（通常24時間）
                    self.token_expires_at = datetime.now() + timedelta(hours=23)
                    self.logger.info("kabu STATION認証成功")
                    return True
                else:
                    self.logger.error(f"認証失敗: HTTP {res.status}")
                    return False
                    
        except urllib.error.HTTPError as e:
            self.logger.error(f"認証HTTPエラー: {e}")
            try:
                content = json.loads(e.read())
                self.logger.error(f"エラー詳細: {content}")
            except:
                pass
            return False
        except Exception as e:
            self.logger.error(f"認証エラー: {e}")
            return False
    
    def _get_api_password(self) -> str:
        """API パスワード取得（ハイブリッド方式）"""
        # 優先度1: 環境変数
        password = os.getenv('KABU_API_PASSWORD')
        if password:
            return password

        # 優先度2: self.config に直接ある api_password
        password = self.config.get('api_password')
        if password:
            return password

        raise ValueError("API パスワードが設定されていません（環境変数 KABU_API_PASSWORD または設定ファイルの authentication.api_password を設定してください）")
    
    def is_token_valid(self) -> bool:
        """トークン有効性確認"""
        if not self.current_token or not self.token_expires_at:
            return False
        return datetime.now() < self.token_expires_at
    
    def get_current_token(self) -> Optional[str]:
        """現在のトークン取得"""
        if self.is_token_valid():
            return self.current_token
        return None

class DSSMSSymbolRegistry:
    """DSSMS階層ランキング統合50銘柄管理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_symbols = config.get('max_symbols', 50)
        self.registered_symbols: Dict[str, SymbolPriorityInfo] = {}
        
        # DSSMS統合コンポーネント
        self.dssms_ranking = None
        self.intelligent_switch = None
        self.screener = None
        
        self.logger = setup_logger(__name__)
    
    def initialize_dssms_components(self):
        """DSSMS既存コンポーネント初期化"""
        try:
            self.screener = Nikkei225Screener()
            self.dssms_ranking = HierarchicalRankingSystem(config={})
            self.intelligent_switch = IntelligentSwitchManager()
            self.logger.info("DSSMS統合コンポーネント初期化成功")
        except Exception as e:
            self.logger.error(f"DSSMS統合コンポーネント初期化エラー: {e}")
    
    def update_symbol_registration(self, available_funds: float) -> Dict[str, Any]:
        """DSSMS優先度に基づく銘柄登録更新"""
        try:
            if not self.dssms_ranking:
                self.initialize_dssms_components()
            
            # DSSMS階層ランキングから候補取得
            symbols = self.screener.get_filtered_symbols(available_funds, date.today()) if self.screener else []
            
            if not symbols:
                # テスト用銘柄を使用
                symbols = self.config.get('development_settings', {}).get('use_test_symbols', ['9433', '5401'])
            
            # 優先度別配分計算
            allocation = self._calculate_priority_allocation(len(symbols))
            new_registration = {}
            
            # 優先度レベル1: 高頻度更新
            level1_count = min(allocation['level1_count'], len(symbols))
            for i, symbol in enumerate(symbols[:level1_count]):
                new_registration[symbol] = SymbolPriorityInfo(
                    symbol=symbol,
                    priority=1,
                    update_frequency='high',
                    confidence='very_high',
                    last_updated=datetime.now(),
                    exchange=1
                )
            
            # 優先度レベル2: 中頻度更新
            level2_start = level1_count
            level2_end = level2_start + min(allocation['level2_count'], len(symbols) - level1_count)
            for symbol in symbols[level2_start:level2_end]:
                new_registration[symbol] = SymbolPriorityInfo(
                    symbol=symbol,
                    priority=2,
                    update_frequency='medium',
                    confidence='high',
                    last_updated=datetime.now(),
                    exchange=1
                )
            
            # 優先度レベル3: 低頻度更新
            level3_start = level2_end
            level3_end = min(level3_start + allocation['level3_count'], len(symbols))
            for symbol in symbols[level3_start:level3_end]:
                new_registration[symbol] = SymbolPriorityInfo(
                    symbol=symbol,
                    priority=3,
                    update_frequency='low',
                    confidence='medium',
                    last_updated=datetime.now(),
                    exchange=1
                )
            
            self.registered_symbols = new_registration
            self.logger.info(f"銘柄登録更新完了: {len(new_registration)}銘柄")
            
            return {
                'success': True,
                'registered_count': len(new_registration),
                'priority_distribution': allocation,
                'symbols': list(new_registration.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"銘柄登録更新エラー: {e}")
            return {
                'success': False,
                'error': str(e),
                'registered_count': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_priority_allocation(self, total_symbols: int) -> Dict[str, int]:
        """優先度別配分計算"""
        available_slots = min(total_symbols, self.max_symbols)
        
        level1_count = int(available_slots * 0.4)
        level2_count = int(available_slots * 0.35)
        level3_count = available_slots - level1_count - level2_count
        
        return {
            'level1_count': level1_count,
            'level2_count': level2_count,
            'level3_count': level3_count
        }
    
    def get_symbol_info(self, symbol: str) -> Optional[SymbolPriorityInfo]:
        """銘柄情報取得"""
        return self.registered_symbols.get(symbol)
    
    def get_all_registered_symbols(self) -> List[str]:
        """登録済み銘柄一覧取得"""
        return list(self.registered_symbols.keys())

class AdaptiveRealtimeClient:
    """適応的頻度調整リアルタイムデータクライアント"""
    
    def __init__(self, auth_manager: KabuAuthManager, config: Dict[str, Any]):
        self.auth_manager = auth_manager
        self.config = config
        self.data_cache = {}
        self.last_request_time = {}
        self.request_count = 0
        self.request_reset_time = datetime.now()
        
        self.logger = setup_logger(__name__)
    
    def get_realtime_data(self, symbol: str) -> pd.DataFrame:
        """適応的頻度でリアルタイムデータ取得"""
        try:
            # API制限チェック
            if not self._can_make_request():
                return self._get_cached_data(symbol)
            
            # トークン確認
            token = self.auth_manager.get_current_token()
            if not token:
                self.logger.warning("認証トークンが無効です")
                return self._get_cached_data(symbol)
            
            # ボードデータ取得
            board_data = self._fetch_board_data(symbol, token)
            
            if board_data:
                formatted_data = self._format_board_data(board_data, symbol)
                self._update_cache(symbol, formatted_data)
                return formatted_data
            else:
                return self._get_cached_data(symbol)
                
        except Exception as e:
            self.logger.error(f"リアルタイムデータ取得エラー {symbol}: {e}")
            return self._get_cached_data(symbol)
    
    def _fetch_board_data(self, symbol: str, token: str) -> Optional[Dict[str, Any]]:
        """kabu APIからボードデータ取得"""
        try:
            base_url = self.config.get('base_url', 'http://localhost:18080/kabusapi')
            url = f"{base_url}/board/{symbol}@1"
            req = urllib.request.Request(url, method='GET')
            req.add_header('Content-Type', 'application/json')
            req.add_header('X-API-KEY', token)
            
            with urllib.request.urlopen(req, timeout=self.config['timeout_seconds']) as res:
                if res.status == 200:
                    content = json.loads(res.read())
                    self._update_request_count()
                    return content
                else:
                    self.logger.error(f"ボードデータ取得失敗: HTTP {res.status}")
                    return None
                    
        except urllib.error.HTTPError as e:
            self.logger.error(f"ボードデータHTTPエラー {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"ボードデータ取得エラー {symbol}: {e}")
            return None
    
    def _format_board_data(self, board_data: Dict[str, Any], symbol: str) -> pd.DataFrame:
        """ボードデータをDataFrameに変換"""
        try:
            # 基本的な価格情報を抽出
            data = {
                'timestamp': [datetime.now()],
                'symbol': [symbol],
                'current_price': [board_data.get('CurrentPrice', 0)],
                'current_price_time': [board_data.get('CurrentPriceTime', '')],
                'ask_price': [board_data.get('Ask', [{}])[0].get('Price', 0) if board_data.get('Ask') else 0],
                'bid_price': [board_data.get('Bid', [{}])[0].get('Price', 0) if board_data.get('Bid') else 0],
                'volume': [board_data.get('TradingVolume', 0)],
                'vwap': [board_data.get('VWAP', 0)],
                'over_sell_under_buy': [board_data.get('OverSellUnderBuy', 0)]
            }
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"ボードデータフォーマットエラー {symbol}: {e}")
            return pd.DataFrame()
    
    def _can_make_request(self) -> bool:
        """API制限確認"""
        now = datetime.now()
        
        # 1分間のリクエスト数をリセット
        if (now - self.request_reset_time).total_seconds() > 60:
            self.request_count = 0
            self.request_reset_time = now
        
        # リクエスト制限チェック
        if self.request_count >= self.config.get('requests_per_minute', 100):
            return False
        
        return True
    
    def _update_request_count(self):
        """リクエスト数更新"""
        self.request_count += 1
    
    def _get_cached_data(self, symbol: str) -> pd.DataFrame:
        """キャッシュデータ取得"""
        cached = self.data_cache.get(symbol)
        if cached and isinstance(cached, pd.DataFrame) and not cached.empty:
            return cached
        return pd.DataFrame()
    
    def _update_cache(self, symbol: str, data: pd.DataFrame):
        """キャッシュ更新"""
        self.data_cache[symbol] = data
        self.last_request_time[symbol] = datetime.now()

class Phase4RiskManager:
    """Phase 4専用リスク管理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.daily_order_count = 0
        self.last_reset_date = datetime.now().date()
        self.active_positions = {}
        
        self.logger = setup_logger(__name__)
    
    def validate_order(self, switch_data: Dict[str, Any]) -> Dict[str, Any]:
        """注文リスク検証"""
        try:
            # 日次リミットチェック
            self._reset_daily_counters()
            
            if self.daily_order_count >= self.config.get('max_daily_orders', 10):
                return {
                    'allowed': False,
                    'reason': '日次注文数制限に達しました'
                }
            
            # ポジション集中度チェック
            symbol = switch_data.get('symbol', '')
            amount = switch_data.get('amount', 0)
            
            if self._check_concentration_limit(symbol, amount):
                return {
                    'allowed': False,
                    'reason': 'ポジション集中度制限に達しました'
                }
            
            return {
                'allowed': True,
                'reason': 'リスクチェック通過'
            }
            
        except Exception as e:
            self.logger.error(f"リスク検証エラー: {e}")
            return {
                'allowed': False,
                'reason': f'リスク検証エラー: {str(e)}'
            }
    
    def _reset_daily_counters(self):
        """日次カウンタリセット"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_order_count = 0
            self.last_reset_date = today
    
    def _check_concentration_limit(self, symbol: str, amount: float) -> bool:
        """集中度制限チェック"""
        concentration_limit = self.config.get('concentration_limit', 0.1)
        total_portfolio_value = sum(self.active_positions.values()) + amount
        
        if total_portfolio_value == 0:
            return False
        
        new_position_ratio = amount / total_portfolio_value
        return new_position_ratio > concentration_limit
    
    def update_position(self, symbol: str, amount: float):
        """ポジション更新"""
        self.active_positions[symbol] = amount
        self.daily_order_count += 1

class KabuOrderExecutor:
    """kabu API注文実行（段階的リスク統合）"""
    
    def __init__(self, auth_manager: KabuAuthManager, config: Dict[str, Any]):
        self.auth_manager = auth_manager
        self.config = config
        self.phase4_risk_manager = Phase4RiskManager(config.get('position_limits', {}))
        
        # 既存リスク管理システム
        try:
            self.existing_risk_manager = RiskManagement(total_assets=10000000.0)
        except Exception as e:
            self.existing_risk_manager = None
            
        self.integration_mode = 'phase4_only'  # 初期は独立運用
        self.logger = setup_logger(__name__)
    
    def execute_dynamic_order(self, switch_data: Dict[str, Any]) -> OrderExecutionResult:
        """動的注文実行"""
        try:
            # Phase 4専用リスク管理チェック
            phase4_check = self.phase4_risk_manager.validate_order(switch_data)
            if not phase4_check['allowed']:
                return OrderExecutionResult(
                    success=False,
                    order_id=None,
                    executed_price=None,
                    error_message=f"Phase4リスク制限: {phase4_check['reason']}",
                    timestamp=datetime.now()
                )
            
            # 統合モードの場合は既存リスク管理もチェック
            if self.integration_mode == 'integrated' and self.existing_risk_manager:
                # 既存リスク管理の簡易チェック
                if not self._check_existing_risk_management(switch_data):
                    return OrderExecutionResult(
                        success=False,
                        order_id=None,
                        executed_price=None,
                        error_message="既存リスク管理制限",
                        timestamp=datetime.now()
                    )
            
            # テスト環境では模擬実行
            if self.config.get('development_settings', {}).get('debug_mode', True):
                return self._execute_mock_order(switch_data)
            
            # 実際のkabu API注文実行
            return self._execute_kabu_order(switch_data)
            
        except Exception as e:
            self.logger.error(f"注文実行エラー: {e}")
            return OrderExecutionResult(
                success=False,
                order_id=None,
                executed_price=None,
                error_message=f"システムエラー: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _execute_mock_order(self, switch_data: Dict[str, Any]) -> OrderExecutionResult:
        """模擬注文実行（テスト用）"""
        self.logger.info(f"模擬注文実行: {switch_data}")
        
        # 模擬的な注文ID生成
        order_id = f"MOCK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 模擬実行価格
        mock_price = switch_data.get('price', 1000.0)
        
        return OrderExecutionResult(
            success=True,
            order_id=order_id,
            executed_price=mock_price,
            error_message=None,
            timestamp=datetime.now()
        )
    
    def _execute_kabu_order(self, switch_data: Dict[str, Any]) -> OrderExecutionResult:
        """実際のkabu API注文実行"""
        try:
            token = self.auth_manager.get_current_token()
            if not token:
                return OrderExecutionResult(
                    success=False,
                    order_id=None,
                    executed_price=None,
                    error_message="認証トークンが無効",
                    timestamp=datetime.now()
                )
            
            # 注文オブジェクト構築
            order_obj = self._build_order_object(switch_data)
            json_data = json.dumps(order_obj).encode('utf-8')

            base_url = self.config.get('base_url', 'http://localhost:18080/kabusapi')
            url = f"{base_url}/sendorder"
            req = urllib.request.Request(url, json_data, method='POST')
            req.add_header('Content-Type', 'application/json')
            req.add_header('X-API-KEY', token)
            
            with urllib.request.urlopen(req, timeout=self.config['timeout_seconds']) as res:
                if res.status == 200:
                    content = json.loads(res.read())
                    order_id = content.get('OrderId')
                    
                    return OrderExecutionResult(
                        success=True,
                        order_id=str(order_id),
                        executed_price=switch_data.get('price'),
                        error_message=None,
                        timestamp=datetime.now()
                    )
                else:
                    return OrderExecutionResult(
                        success=False,
                        order_id=None,
                        executed_price=None,
                        error_message=f"注文失敗: HTTP {res.status}",
                        timestamp=datetime.now()
                    )
                    
        except Exception as e:
            self.logger.error(f"kabu API注文実行エラー: {e}")
            return OrderExecutionResult(
                success=False,
                order_id=None,
                executed_price=None,
                error_message=f"注文実行エラー: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _build_order_object(self, switch_data: Dict[str, Any]) -> Dict[str, Any]:
        """注文オブジェクト構築"""
        order_type = self.config.get('order_types', {}).get('cash_buy', {})
        
        return {
            'Symbol': str(switch_data.get('symbol', '9433')),
            'Exchange': switch_data.get('exchange', 1),
            'SecurityType': order_type.get('security_type', 1),
            'Side': str(switch_data.get('side', '2')),  # 2:買い、1:売り
            'CashMargin': order_type.get('cash_margin', 1),
            'DelivType': order_type.get('deliv_type', 2),
            'FundType': order_type.get('fund_type', 'AA'),
            'AccountType': order_type.get('account_type', 2),
            'Qty': int(switch_data.get('quantity', 100)),
            'FrontOrderType': 30,  # 成行注文
            'Price': 0,  # 成行の場合は0
            'ExpireDay': 0
        }
    
    def _check_existing_risk_management(self, switch_data: Dict[str, Any]) -> bool:
        """既存リスク管理チェック"""
        if not self.existing_risk_manager:
            return True
        
        try:
            # 既存リスク管理の基本的なチェック
            symbol = switch_data.get('symbol', '')
            return self.existing_risk_manager.check_position_size("kabu_integration", symbol)
        except Exception as e:
            self.logger.warning(f"既存リスク管理チェックエラー: {e}")
            return True
    
    def enable_full_integration(self):
        """完全統合モード有効化"""
        self.integration_mode = 'integrated'
        self.logger.info("既存リスク管理との完全統合モードを有効化")

class KabuIntegrationManager:
    """kabu_apiとの完全統合マネージャー"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = setup_logger(__name__)
        
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # 認証・接続管理
        self.auth_manager = KabuAuthManager(self.config['authentication'])
        
        # 50銘柄管理
        self.symbol_registry = DSSMSSymbolRegistry(self.config['registration_strategy'])
        
        # データ取得
        self.realtime_client = AdaptiveRealtimeClient(
            self.auth_manager, 
            self.config['authentication']
        )
        
        # 注文実行
        self.order_executor = KabuOrderExecutor(
            self.auth_manager,
            {**self.config['order_settings'], **self.config}
        )
        
        # 初期化状態
        self.is_initialized = False
        
        self.logger.info("KabuIntegrationManager 初期化完了")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            """overrideの値でbaseを再帰的に上書きする。"""
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    _deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        defaults: Dict[str, Any] = {
            'authentication': {
                'base_url': 'http://localhost:18080/kabusapi',
                'timeout_seconds': 30
            },
            'registration_strategy': {
                'max_symbols': 50
            },
            'order_settings': {
                'default_order_type': 'market'
            }
        }

        try:
            configs: Dict[str, Any] = {
                'authentication': dict(defaults['authentication']),
                'registration_strategy': dict(defaults['registration_strategy']),
                'order_settings': dict(defaults['order_settings'])
            }

            if config_path is None:
                config_dir = project_root / "config" / "kabu_api"
                
                # 各設定ファイル読み込み
                config_files = [
                    "kabu_connection_config.json",
                    "symbol_registration_config.json", 
                    "order_execution_config.json"
                ]
                
                for config_file in config_files:
                    file_path = config_dir / config_file
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_config = json.load(f)
                            _deep_merge(configs, file_config)
                
                return configs
            else:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                return _deep_merge(configs, file_config)
                    
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー: {e}")
            # デフォルト設定を返す
            return defaults
    
    def initialize(self) -> bool:
        """システム初期化"""
        try:
            # kabu STATION認証
            if not self.auth_manager.authenticate():
                self.logger.error("kabu STATION認証に失敗しました")
                return False
            
            # DSSMS統合コンポーネント初期化
            self.symbol_registry.initialize_dssms_components()
            
            self.is_initialized = True
            self.logger.info("KabuIntegrationManager システム初期化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"システム初期化エラー: {e}")
            return False
    
    def register_screening_symbols(self, symbols: List[str]) -> bool:
        """50銘柄登録（階層化優先度管理）"""
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return False
            
            # DSSMS優先度管理による銘柄登録
            available_funds = 10000000.0  # デフォルト利用可能資金
            
            registration_result = self.symbol_registry.update_symbol_registration(available_funds)
            
            if registration_result['success']:
                # kabu APIへの実際の銘柄登録
                success = self._register_symbols_to_kabu_api(registration_result['symbols'])
                
                if success:
                    self.logger.info(f"銘柄登録成功: {registration_result['registered_count']}銘柄")
                    return True
                else:
                    self.logger.error("kabu API銘柄登録に失敗しました")
                    return False
            else:
                self.logger.error(f"銘柄登録失敗: {registration_result.get('error', '不明なエラー')}")
                return False
                
        except Exception as e:
            self.logger.error(f"銘柄登録エラー: {e}")
            return False
    
    def _register_symbols_to_kabu_api(self, symbols: List[str]) -> bool:
        """kabu APIへの銘柄登録実行"""
        try:
            token = self.auth_manager.get_current_token()
            if not token:
                self.logger.error("認証トークンが無効です")
                return False
            
            # 銘柄リスト構築
            symbol_list = []
            for symbol in symbols[:50]:  # 最大50銘柄
                symbol_list.append({
                    'Symbol': str(symbol),
                    'Exchange': 1  # 東証
                })
            
            obj = {'Symbols': symbol_list}
            json_data = json.dumps(obj).encode('utf8')

            base_url = self.config.get('base_url', 'http://localhost:18080/kabusapi')
            url = f"{base_url}/register"
            req = urllib.request.Request(url, json_data, method='PUT')
            req.add_header('Content-Type', 'application/json')
            req.add_header('X-API-KEY', token)
            
            with urllib.request.urlopen(req, timeout=self.config['authentication']['timeout_seconds']) as res:
                if res.status == 200:
                    content = json.loads(res.read())
                    self.logger.info(f"kabu API銘柄登録成功: {content}")
                    return True
                else:
                    self.logger.error(f"kabu API銘柄登録失敗: HTTP {res.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"kabu API銘柄登録エラー: {e}")
            return False
    
    def get_realtime_data_for_selected(self, symbol: str) -> pd.DataFrame:
        """リアルタイムデータ取得（適応的頻度）"""
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return pd.DataFrame()
            
            return self.realtime_client.get_realtime_data(symbol)
            
        except Exception as e:
            self.logger.error(f"リアルタイムデータ取得エラー {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        kabu STATIONから現在価格を取得する。
        
        Args:
            symbol: 銘柄コード（例: "8031"）
        
        Returns:
            float: 現在価格（円）。取得失敗時はNone。
        """
        try:
            df = self.get_realtime_data_for_selected(symbol)
            if not df.empty and 'current_price' in df.columns:
                price = float(df['current_price'].iloc[0])
                if price > 0:
                    return price
            self.logger.warning(f"[PRICE_FETCH] {symbol}: kabu API価格取得失敗")
            return None
        except Exception as e:
            self.logger.error(f"[PRICE_FETCH] {symbol}: 現在価格取得エラー ({e})")
            return None
    
    def execute_dynamic_orders(self, switch_data: Dict) -> Dict[str, Any]:
        """動的注文実行（段階的統合）"""
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return {
                        'success': False,
                        'error': 'システム初期化に失敗しました',
                        'timestamp': datetime.now().isoformat()
                    }
            
            # 注文実行
            result = self.order_executor.execute_dynamic_order(switch_data)
            
            return {
                'success': result.success,
                'order_id': result.order_id,
                'executed_price': result.executed_price,
                'error': result.error_message,
                'timestamp': result.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"動的注文実行エラー: {e}")
            return {
                'success': False,
                'error': f'システムエラー: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def monitor_position_status(self) -> Dict[str, Any]:
        """ポジション監視"""
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return {
                        'success': False,
                        'error': 'システム初期化に失敗しました'
                    }
            
            # 基本的な状態情報を返す
            return {
                'success': True,
                'registered_symbols_count': len(self.symbol_registry.get_all_registered_symbols()),
                'registered_symbols': self.symbol_registry.get_all_registered_symbols(),
                'authentication_status': 'authenticated' if self.auth_manager.is_token_valid() else 'expired',
                'integration_mode': self.order_executor.integration_mode,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"ポジション監視エラー: {e}")
            return {
                'success': False,
                'error': f'監視エラー: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状況取得"""
        return {
            'initialized': self.is_initialized,
            'authentication_valid': self.auth_manager.is_token_valid() if self.auth_manager else False,
            'registered_symbols_count': len(self.symbol_registry.get_all_registered_symbols()) if self.symbol_registry else 0,
            'integration_mode': self.order_executor.integration_mode if self.order_executor else 'unknown'
        }

# 統合インターフェース
class DSSMSKabuIntegrator:
    """DSSMS-kabu API統合インターフェース"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.kabu_manager = KabuIntegrationManager(config_path)
        self.logger = setup_logger(__name__)
    
    def initialize_integration(self) -> bool:
        """統合システム初期化"""
        return self.kabu_manager.initialize()
    
    def sync_screening_results_to_kabu(self, available_funds: float) -> bool:
        """DSSMS スクリーニング結果をkabu APIに同期"""
        try:
            # DSSMSスクリーニング結果を取得してkabu APIに登録
            symbols = []  # 実際にはDSSMSから取得
            return self.kabu_manager.register_screening_symbols(symbols)
        except Exception as e:
            self.logger.error(f"スクリーニング結果同期エラー: {e}")
            return False
    
    def execute_intelligent_switch(self, switch_recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """インテリジェント切替の実行"""
        try:
            # DSSMSインテリジェント切替の推奨をkabu API注文に変換
            order_data = {
                'symbol': switch_recommendation.get('to_symbol'),
                'side': '2',  # 買い
                'quantity': switch_recommendation.get('quantity', 100),
                'price': switch_recommendation.get('price', 0)
            }
            
            return self.kabu_manager.execute_dynamic_orders(order_data)
            
        except Exception as e:
            self.logger.error(f"インテリジェント切替実行エラー: {e}")
            return {
                'success': False,
                'error': f'切替実行エラー: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_integrated_status(self) -> Dict[str, Any]:
        """統合システム状況取得"""
        return self.kabu_manager.get_system_status()

if __name__ == "__main__":
    # テスト実行
    integrator = DSSMSKabuIntegrator()
    
    if integrator.initialize_integration():
        print("[OK] DSSMS-kabu API統合初期化成功")
        
        # システム状況確認
        status = integrator.get_integrated_status()
        print(f"システム状況: {status}")
    else:
        print("[ERROR] 初期化に失敗しました")

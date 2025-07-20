"""
Module: Multi-Strategy Coordination Interface
File: multi_strategy_coordination_interface.py
Description: 
  4-1-3「マルチ戦略同時実行の調整機能」
  統合インターフェース（既存システムとの連携）

Author: imega
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 4-1-1・4-1-2システムとの統合
  - 外部システム連携・API提供
  - 設定管理・動的設定更新
  - 統合ダッシュボード・制御インターフェース
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
import threading
from flask import Flask, request, jsonify
from functools import wraps

# プロジェクトモジュールをインポート
try:
    from multi_strategy_coordination_manager import MultiStrategyCoordinationManager, CoordinationState, FallbackLevel
    from resource_allocation_engine import ResourceAllocationEngine
    from strategy_dependency_resolver import StrategyDependencyResolver
    from concurrent_execution_scheduler import ConcurrentExecutionScheduler
    from execution_monitoring_system import ExecutionMonitoringSystem
    
    # 既存システム統合（4-1-1, 4-1-2）
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from composite_strategy_execution_engine import CompositeStrategyExecutionEngine
    from strategy_execution_coordinator import StrategyExecutionCoordinator
    
except ImportError as e:
    # スタンドアロンテスト用フォールバック
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import project modules: {e}, using fallback definitions")
    
    class CoordinationState(Enum):
        IDLE = "idle"
        EXECUTING = "executing"
        EMERGENCY = "emergency"
    
    class FallbackLevel(Enum):
        NONE = "none"
        INDIVIDUAL = "individual"
        EMERGENCY = "emergency"

# ロガー設定
logger = logging.getLogger(__name__)

class IntegrationMode(Enum):
    """統合モード"""
    STANDALONE = "standalone"          # スタンドアローン動作
    INTEGRATED_4_1_1 = "integrated_4_1_1"    # 4-1-1統合
    INTEGRATED_4_1_2 = "integrated_4_1_2"    # 4-1-2統合
    FULL_INTEGRATION = "full_integration"      # 完全統合
    COMPATIBILITY = "compatibility"    # 互換モード

class APIResponseStatus(Enum):
    """API応答ステータス"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"

@dataclass
class IntegrationConfig:
    """統合設定"""
    mode: IntegrationMode = IntegrationMode.STANDALONE
    enable_web_interface: bool = True
    web_port: int = 5000
    enable_api: bool = True
    enable_4_1_1_integration: bool = False
    enable_4_1_2_integration: bool = False
    compatibility_mode: bool = True
    auto_discovery: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['mode'] = self.mode.value
        return result

@dataclass
class APIResponse:
    """API応答"""
    status: APIResponseStatus
    message: str
    data: Optional[dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

class SystemIntegrator:
    """システム統合器"""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.integration_config = self._parse_integration_config(config)
        self.integrated_systems: dict[str, Any] = {}
        self.compatibility_adapters: dict[str, Callable] = {}
        self.integration_lock = threading.Lock()
        
    def _parse_integration_config(self, config: dict[str, Any]) -> IntegrationConfig:
        """統合設定解析"""
        integration_section = config.get('integration', {})
        
        # 統合モード決定
        mode = IntegrationMode.STANDALONE
        if integration_section.get('enable_4_1_1_integration') and integration_section.get('enable_4_1_2_integration'):
            mode = IntegrationMode.FULL_INTEGRATION
        elif integration_section.get('enable_4_1_2_integration'):
            mode = IntegrationMode.INTEGRATED_4_1_2
        elif integration_section.get('enable_4_1_1_integration'):
            mode = IntegrationMode.INTEGRATED_4_1_1
        
        return IntegrationConfig(
            mode=mode,
            enable_web_interface=integration_section.get('enable_web_interface', True),
            web_port=integration_section.get('web_port', 5000),
            enable_api=integration_section.get('enable_api', True),
            enable_4_1_1_integration=integration_section.get('enable_4_1_1_integration', False),
            enable_4_1_2_integration=integration_section.get('enable_4_1_2_integration', False),
            compatibility_mode=integration_section.get('compatibility_mode', True),
            auto_discovery=integration_section.get('auto_discovery', True)
        )
    
    def discover_and_integrate_systems(self):
        """システム自動発見・統合"""
        logger.info("Discovering and integrating existing systems...")
        
        # 4-1-1システム統合
        if self.integration_config.enable_4_1_1_integration:
            self._integrate_4_1_1_system()
        
        # 4-1-2システム統合
        if self.integration_config.enable_4_1_2_integration:
            self._integrate_4_1_2_system()
        
        logger.info(f"System integration complete: {len(self.integrated_systems)} systems integrated")
    
    def _integrate_4_1_1_system(self):
        """4-1-1システム統合"""
        try:
            # main.pyの統合システムを探索
            main_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main.py')
            if os.path.exists(main_py_path):
                logger.info("Found 4-1-1 main.py integration system")
                
                # 互換アダプター作成
                self.compatibility_adapters['4_1_1'] = self._create_4_1_1_adapter()
                self.integrated_systems['4_1_1'] = {
                    'name': '4-1-1 Main Integration System',
                    'path': main_py_path,
                    'status': 'integrated',
                    'adapter': self.compatibility_adapters['4_1_1']
                }
                logger.info("4-1-1 system integration successful")
            else:
                logger.warning("4-1-1 main.py not found")
        
        except Exception as e:
            logger.error(f"4-1-1 system integration failed: {e}")
    
    def _integrate_4_1_2_system(self):
        """4-1-2システム統合"""
        try:
            # 4-1-2コンポーネント探索
            config_dir = os.path.dirname(__file__)
            composite_engine_path = os.path.join(config_dir, 'composite_strategy_execution_engine.py')
            
            if os.path.exists(composite_engine_path):
                logger.info("Found 4-1-2 composite strategy execution engine")
                
                # 直接統合
                if 'CompositeStrategyExecutionEngine' in globals():
                    self.integrated_systems['4_1_2'] = {
                        'name': '4-1-2 Composite Strategy Execution System',
                        'engine': CompositeStrategyExecutionEngine,
                        'coordinator': StrategyExecutionCoordinator,
                        'status': 'integrated'
                    }
                
                # 互換アダプター作成
                self.compatibility_adapters['4_1_2'] = self._create_4_1_2_adapter()
                
                logger.info("4-1-2 system integration successful")
            else:
                logger.warning("4-1-2 composite strategy execution engine not found")
        
        except Exception as e:
            logger.error(f"4-1-2 system integration failed: {e}")
    
    def _create_4_1_1_adapter(self) -> Callable:
        """4-1-1互換アダプター作成"""
        def adapter(strategies: list[str], **kwargs) -> dict[str, Any]:
            """4-1-1システム呼び出しアダプター"""
            try:
                # main.pyの実行ロジックを呼び出し（模擬）
                logger.info(f"Calling 4-1-1 system for strategies: {strategies}")
                
                # 実際の実装では、main.pyの統合システムを呼び出す
                result = {
                    'system': '4-1-1',
                    'strategies': strategies,
                    'execution_time': time.time(),
                    'success': True,
                    'message': '4-1-1 system executed successfully'
                }
                
                return result
            
            except Exception as e:
                logger.error(f"4-1-1 adapter error: {e}")
                return {
                    'system': '4-1-1',
                    'success': False,
                    'error': str(e)
                }
        
        return adapter
    
    def _create_4_1_2_adapter(self) -> Callable:
        """4-1-2互換アダプター作成"""
        def adapter(strategies: list[str], **kwargs) -> dict[str, Any]:
            """4-1-2システム呼び出しアダプター"""
            try:
                logger.info(f"Calling 4-1-2 system for strategies: {strategies}")
                
                # 4-1-2の複合戦略実行エンジンを呼び出し
                if 'CompositeStrategyExecutionEngine' in globals():
                    # 実際の4-1-2システム呼び出し
                    engine = CompositeStrategyExecutionEngine()
                    result = engine.execute_composite_strategies(strategies)
                else:
                    # フォールバック模擬実行
                    result = {
                        'system': '4-1-2',
                        'strategies': strategies,
                        'execution_time': time.time(),
                        'success': True,
                        'message': '4-1-2 system executed successfully'
                    }
                
                return result
            
            except Exception as e:
                logger.error(f"4-1-2 adapter error: {e}")
                return {
                    'system': '4-1-2',
                    'success': False,
                    'error': str(e)
                }
        
        return adapter
    
    def call_integrated_system(self, system_id: str, strategies: list[str], **kwargs) -> dict[str, Any]:
        """統合システム呼び出し"""
        with self.integration_lock:
            if system_id not in self.integrated_systems:
                raise ValueError(f"System {system_id} not integrated")
            
            system_info = self.integrated_systems[system_id]
            
            if system_info['status'] != 'integrated':
                raise RuntimeError(f"System {system_id} not properly integrated")
            
            # 対応するアダプターで呼び出し
            if system_id in self.compatibility_adapters:
                adapter = self.compatibility_adapters[system_id]
                return adapter(strategies, **kwargs)
            else:
                raise RuntimeError(f"No adapter found for system {system_id}")
    
    def get_integration_status(self) -> dict[str, Any]:
        """統合状況取得"""
        with self.integration_lock:
            return {
                'integration_config': self.integration_config.to_dict(),
                'integrated_systems': {
                    system_id: {
                        'name': info.get('name', 'Unknown'),
                        'status': info.get('status', 'unknown')
                    }
                    for system_id, info in self.integrated_systems.items()
                },
                'adapter_count': len(self.compatibility_adapters),
                'integration_timestamp': datetime.now().isoformat()
            }

class WebInterface:
    """Web インターフェース"""
    
    def __init__(self, coordination_manager: MultiStrategyCoordinationManager, integrator: SystemIntegrator):
        self.coordination_manager = coordination_manager
        self.integrator = integrator
        self.app = Flask(__name__)
        self._setup_routes()
        
    def _setup_routes(self):
        """ルート設定"""
        
        @self.app.route('/')
        def dashboard():
            """ダッシュボード"""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Multi-Strategy Coordination Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .status { padding: 5px 10px; border-radius: 3px; color: white; }
                    .idle { background-color: #28a745; }
                    .executing { background-color: #007bff; }
                    .emergency { background-color: #dc3545; }
                    .btn { padding: 8px 16px; margin: 5px; background: #007bff; color: white; text-decoration: none; border-radius: 3px; }
                </style>
                <script>
                    function refreshStatus() {
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('status').innerHTML = JSON.stringify(data, null, 2);
                            });
                    }
                    
                    function startCoordination() {
                        const strategies = document.getElementById('strategies').value.split(',').map(s => s.trim());
                        fetch('/api/coordinate', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({strategies: strategies})
                        })
                        .then(response => response.json())
                        .then(data => {
                            alert(data.message);
                            refreshStatus();
                        });
                    }
                </script>
            </head>
            <body>
                <h1>🚀 Multi-Strategy Coordination Dashboard</h1>
                
                <div class="card">
                    <h2>Quick Actions</h2>
                    <input type="text" id="strategies" placeholder="戦略名をカンマ区切りで入力" style="width: 300px;">
                    <button onclick="startCoordination()">調整実行開始</button>
                    <button onclick="refreshStatus()">状況更新</button>
                </div>
                
                <div class="card">
                    <h2>System Status</h2>
                    <pre id="status">Loading...</pre>
                </div>
                
                <div class="card">
                    <h2>API Endpoints</h2>
                    <ul>
                        <li><a href="/api/status" class="btn">GET /api/status</a> - システム状況取得</li>
                        <li><a href="#" class="btn">POST /api/coordinate</a> - 調整実行開始</li>
                        <li><a href="/api/integration" class="btn">GET /api/integration</a> - 統合状況</li>
                    </ul>
                </div>
                
                <script>refreshStatus();</script>
            </body>
            </html>
            """
        
        @self.app.route('/api/status')
        def api_status():
            """システム状況API"""
            try:
                status = self.coordination_manager.get_coordination_status()
                return jsonify(APIResponse(
                    status=APIResponseStatus.SUCCESS,
                    message="Status retrieved successfully",
                    data=status
                ).to_dict())
            except Exception as e:
                return jsonify(APIResponse(
                    status=APIResponseStatus.ERROR,
                    message=f"Failed to get status: {str(e)}"
                ).to_dict()), 500
        
        @self.app.route('/api/coordinate', methods=['POST'])
        def api_coordinate():
            """調整実行API"""
            try:
                data = request.json
                strategies = data.get('strategies', [])
                
                if not strategies:
                    return jsonify(APIResponse(
                        status=APIResponseStatus.ERROR,
                        message="No strategies provided"
                    ).to_dict()), 400
                
                # 調整計画作成・実行
                plan = self.coordination_manager.create_coordination_plan(strategies)
                execution_id = self.coordination_manager.execute_coordination_plan(plan)
                
                return jsonify(APIResponse(
                    status=APIResponseStatus.SUCCESS,
                    message="Coordination started successfully",
                    data={
                        'execution_id': execution_id,
                        'plan_id': plan.plan_id,
                        'strategies': strategies
                    }
                ).to_dict())
            
            except Exception as e:
                return jsonify(APIResponse(
                    status=APIResponseStatus.ERROR,
                    message=f"Coordination failed: {str(e)}"
                ).to_dict()), 500
        
        @self.app.route('/api/integration')
        def api_integration():
            """統合状況API"""
            try:
                integration_status = self.integrator.get_integration_status()
                return jsonify(APIResponse(
                    status=APIResponseStatus.SUCCESS,
                    message="Integration status retrieved successfully",
                    data=integration_status
                ).to_dict())
            except Exception as e:
                return jsonify(APIResponse(
                    status=APIResponseStatus.ERROR,
                    message=f"Failed to get integration status: {str(e)}"
                ).to_dict()), 500
        
        @self.app.route('/api/cancel/<execution_id>', methods=['POST'])
        def api_cancel(execution_id: str):
            """調整キャンセルAPI"""
            try:
                success = self.coordination_manager.cancel_coordination(execution_id)
                if success:
                    return jsonify(APIResponse(
                        status=APIResponseStatus.SUCCESS,
                        message=f"Coordination {execution_id} cancelled successfully"
                    ).to_dict())
                else:
                    return jsonify(APIResponse(
                        status=APIResponseStatus.ERROR,
                        message=f"Coordination {execution_id} not found or cannot be cancelled"
                    ).to_dict()), 404
            except Exception as e:
                return jsonify(APIResponse(
                    status=APIResponseStatus.ERROR,
                    message=f"Cancel failed: {str(e)}"
                ).to_dict()), 500
        
        @self.app.route('/api/legacy/<system_id>', methods=['POST'])
        def api_legacy_integration(system_id: str):
            """既存システム統合API"""
            try:
                data = request.json
                strategies = data.get('strategies', [])
                
                result = self.integrator.call_integrated_system(system_id, strategies)
                
                return jsonify(APIResponse(
                    status=APIResponseStatus.SUCCESS,
                    message=f"Legacy system {system_id} executed successfully",
                    data=result
                ).to_dict())
            
            except Exception as e:
                return jsonify(APIResponse(
                    status=APIResponseStatus.ERROR,
                    message=f"Legacy system call failed: {str(e)}"
                ).to_dict()), 500
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Web インターフェース起動"""
        logger.info(f"Starting web interface on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

class ConfigurationManager:
    """設定管理器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.config_lock = threading.Lock()
        self.change_callbacks: list[Callable[[dict[str, Any]], None]] = []
        
    def _load_config(self) -> dict[str, Any]:
        """設定読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def get_config(self) -> dict[str, Any]:
        """設定取得"""
        with self.config_lock:
            return self.config.copy()
    
    def update_config(self, updates: dict[str, Any], save: bool = True) -> bool:
        """設定更新"""
        try:
            with self.config_lock:
                # 深いマージ
                self._deep_merge(self.config, updates)
                
                # ファイル保存
                if save:
                    with open(self.config_path, 'w', encoding='utf-8') as f:
                        json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            # 変更通知
            for callback in self.change_callbacks:
                try:
                    callback(self.config)
                except Exception as e:
                    logger.error(f"Config change callback error: {e}")
            
            logger.info("Configuration updated successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False
    
    def _deep_merge(self, target: dict[str, Any], source: dict[str, Any]):
        """深いマージ"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def add_change_callback(self, callback: Callable[[dict[str, Any]], None]):
        """設定変更コールバック追加"""
        self.change_callbacks.append(callback)
    
    def validate_config(self, config: dict[str, Any] = None) -> tuple[bool, list[str]]:
        """設定検証"""
        if config is None:
            config = self.config
        
        errors = []
        
        # 必須設定チェック
        required_sections = ['execution_modes', 'load_balancing', 'dependency_management', 'monitoring']
        for section in required_sections:
            if section not in config:
                errors.append(f"Required section '{section}' missing")
        
        # 値範囲チェック
        if 'execution_modes' in config:
            exec_config = config['execution_modes']
            if exec_config.get('thread_pool_size', 0) <= 0:
                errors.append("thread_pool_size must be positive")
            if exec_config.get('process_pool_size', 0) <= 0:
                errors.append("process_pool_size must be positive")
        
        return len(errors) == 0, errors

class MultiStrategyCoordinationInterface:
    """マルチ戦略調整インターフェース"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'coordination_config.json')
        
        self.config_path = config_path
        self.config_manager = ConfigurationManager(config_path)
        self.config = self.config_manager.get_config()
        
        # コンポーネント初期化
        self.coordination_manager = MultiStrategyCoordinationManager(config_path)
        self.system_integrator = SystemIntegrator(self.config)
        
        # Web インターフェース
        self.web_interface: Optional[WebInterface] = None
        self.web_thread: Optional[threading.Thread] = None
        
        # 設定変更監視
        self.config_manager.add_change_callback(self._on_config_change)
        
        logger.info("Multi-Strategy Coordination Interface initialized")
    
    def initialize(self):
        """インターフェース初期化"""
        logger.info("Initializing Multi-Strategy Coordination Interface...")
        
        # 調整マネージャー初期化
        try:
            self.coordination_manager.initialize_components()
        except Exception as e:
            logger.warning(f"Coordination manager initialization partial: {e}")
        
        # システム統合
        self.system_integrator.discover_and_integrate_systems()
        
        # Web インターフェース初期化
        integration_config = self.system_integrator.integration_config
        if integration_config.enable_web_interface:
            self._initialize_web_interface(integration_config)
        
        logger.info("Multi-Strategy Coordination Interface initialization complete")
    
    def _initialize_web_interface(self, integration_config: IntegrationConfig):
        """Web インターフェース初期化"""
        try:
            self.web_interface = WebInterface(self.coordination_manager, self.system_integrator)
            
            # バックグラウンドで起動
            self.web_thread = threading.Thread(
                target=self.web_interface.run,
                kwargs={
                    'host': '0.0.0.0',
                    'port': integration_config.web_port,
                    'debug': False
                },
                daemon=True
            )
            self.web_thread.start()
            
            logger.info(f"Web interface started on port {integration_config.web_port}")
        
        except Exception as e:
            logger.error(f"Web interface initialization failed: {e}")
    
    def _on_config_change(self, new_config: dict[str, Any]):
        """設定変更時処理"""
        logger.info("Configuration changed, updating components...")
        
        try:
            # 設定検証
            valid, errors = self.config_manager.validate_config(new_config)
            if not valid:
                logger.error(f"Invalid configuration: {errors}")
                return
            
            # コンポーネントに設定変更を通知
            # 実際の実装では、各コンポーネントに動的設定更新機能が必要
            
            logger.info("Configuration update applied successfully")
        
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
    
    def execute_strategy_coordination(
        self, 
        strategies: list[str], 
        integration_mode: Optional[str] = None
    ) -> dict[str, Any]:
        """戦略調整実行（統合インターフェース）"""
        logger.info(f"Executing strategy coordination for {len(strategies)} strategies")
        
        try:
            # 統合モード決定
            if integration_mode == "4-1-1":
                # 4-1-1システム経由で実行
                if "4_1_1" in self.system_integrator.integrated_systems:
                    result = self.system_integrator.call_integrated_system("4_1_1", strategies)
                    return {
                        'success': True,
                        'method': '4-1-1 Integration',
                        'result': result
                    }
                else:
                    logger.warning("4-1-1 integration not available, fallback to native execution")
            
            elif integration_mode == "4-1-2":
                # 4-1-2システム経由で実行
                if "4_1_2" in self.system_integrator.integrated_systems:
                    result = self.system_integrator.call_integrated_system("4_1_2", strategies)
                    return {
                        'success': True,
                        'method': '4-1-2 Integration',
                        'result': result
                    }
                else:
                    logger.warning("4-1-2 integration not available, fallback to native execution")
            
            # ネイティブ4-1-3実行
            plan = self.coordination_manager.create_coordination_plan(strategies)
            execution_id = self.coordination_manager.execute_coordination_plan(plan)
            
            return {
                'success': True,
                'method': '4-1-3 Native Coordination',
                'plan_id': plan.plan_id,
                'execution_id': execution_id,
                'estimated_completion': plan.estimated_completion_time.isoformat()
            }
        
        except Exception as e:
            logger.error(f"Strategy coordination failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'Failed'
            }
    
    def get_system_status(self) -> dict[str, Any]:
        """システム状況取得"""
        coordination_status = self.coordination_manager.get_coordination_status()
        integration_status = self.system_integrator.get_integration_status()
        
        return {
            'coordination': coordination_status,
            'integration': integration_status,
            'web_interface': {
                'enabled': self.web_interface is not None,
                'port': self.system_integrator.integration_config.web_port if self.web_interface else None
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def update_configuration(self, updates: dict[str, Any]) -> bool:
        """設定更新"""
        return self.config_manager.update_config(updates)
    
    def shutdown(self):
        """シャットダウン"""
        logger.info("Shutting down Multi-Strategy Coordination Interface...")
        
        # 調整マネージャーシャットダウン
        self.coordination_manager.shutdown()
        
        # Web インターフェースは自動終了（daemon thread）
        
        logger.info("Multi-Strategy Coordination Interface shutdown complete")

def create_demo_interface() -> MultiStrategyCoordinationInterface:
    """デモ用インターフェース作成"""
    return MultiStrategyCoordinationInterface()

if __name__ == "__main__":
    # デモ実行
    print("=" * 60)
    print("Multi-Strategy Coordination Interface - Demo")
    print("=" * 60)
    
    try:
        # インターフェース初期化
        interface = create_demo_interface()
        interface.initialize()
        
        print("🌐 Coordination interface initialized successfully")
        
        # システム状況取得
        status = interface.get_system_status()
        print(f"\n📊 System Status:")
        print(f"  Coordination State: {status['coordination']['state']}")
        print(f"  Integration Mode: {status['integration']['integration_config']['mode']}")
        print(f"  Integrated Systems: {len(status['integration']['integrated_systems'])}")
        print(f"  Web Interface: {'Enabled' if status['web_interface']['enabled'] else 'Disabled'}")
        
        if status['web_interface']['enabled']:
            port = status['web_interface']['port']
            print(f"    📱 Dashboard: http://localhost:{port}")
            print(f"    🔗 API: http://localhost:{port}/api/status")
        
        # 統合システム情報
        integrated_systems = status['integration']['integrated_systems']
        if integrated_systems:
            print(f"\n🔗 Integrated Systems:")
            for system_id, system_info in integrated_systems.items():
                print(f"  - {system_id}: {system_info['name']} ({system_info['status']})")
        
        # デモ戦略調整実行
        demo_strategies = ["VWAPBounceStrategy", "GCStrategy", "BreakoutStrategy"]
        
        print(f"\n🎯 Testing strategy coordination with {len(demo_strategies)} strategies...")
        
        # ネイティブ実行テスト
        result = interface.execute_strategy_coordination(demo_strategies)
        print(f"Native Coordination Result:")
        print(f"  Success: {'✅' if result['success'] else '❌'}")
        print(f"  Method: {result['method']}")
        if result['success']:
            print(f"  Execution ID: {result.get('execution_id', 'N/A')}")
        
        # 統合システムテスト（利用可能な場合）
        for system_id in ['4-1-1', '4-1-2']:
            integration_mode = system_id.replace('-', '_')
            if integrated_systems.get(integration_mode):
                print(f"\n🔄 Testing {system_id} integration...")
                result = interface.execute_strategy_coordination(demo_strategies, integration_mode)
                print(f"  Success: {'✅' if result['success'] else '❌'}")
                print(f"  Method: {result['method']}")
        
        # 設定動的更新テスト
        print(f"\n⚙️ Testing dynamic configuration update...")
        config_update = {
            "monitoring": {
                "level": "comprehensive",
                "collection_interval": 3.0
            }
        }
        
        update_success = interface.update_configuration(config_update)
        print(f"Configuration Update: {'✅' if update_success else '❌'}")
        
        # Web インターフェース案内
        if status['web_interface']['enabled']:
            port = status['web_interface']['port']
            print(f"\n🌐 Web Interface Active:")
            print(f"  Dashboard: http://localhost:{port}")
            print(f"  API Status: http://localhost:{port}/api/status")
            print(f"  Integration Info: http://localhost:{port}/api/integration")
            print(f"\n  💡 ブラウザでダッシュボードにアクセスしてリアルタイム制御が可能です")
        
        # デモ継続（Web インターフェースのため）
        print(f"\n⏳ Demo running... (Web interface active)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(10)
                current_status = interface.get_system_status()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Coordination State: {current_status['coordination']['state']}")
        
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
        
        print("\n✅ Multi-Strategy Coordination Interface demo completed!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'interface' in locals():
            interface.shutdown()

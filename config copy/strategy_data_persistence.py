"""
Module: Strategy Data Persistence
File: strategy_data_persistence.py
Description: 
  戦略特性データの永続化機能
  複数データソース（特性・パラメータ）の統合、バージョン管理、変更履歴の保持
  optimized_parameters.pyと連携し、最新パラメータを採用

Author: imega
Created: 2025-07-08
Modified: 2025-07-08

Dependencies:
  - json
  - os
  - pandas
  - datetime
  - typing
  - config.strategy_characteristics_manager
  - config.optimized_parameters
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import pandas as pd
import hashlib
from dataclasses import dataclass

# 内部モジュールのインポート
try:
    from .strategy_characteristics_manager import StrategyCharacteristicsManager
    from .optimized_parameters import OptimizedParameterManager
except ImportError:
    # 直接実行時の対応
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config.strategy_characteristics_manager import StrategyCharacteristicsManager
    from config.optimized_parameters import OptimizedParameterManager

# ロガーの設定
logger = logging.getLogger(__name__)

@dataclass
class DataVersion:
    """データバージョン情報"""
    version: str
    timestamp: str
    hash_value: str
    description: str = ""
    author: str = "system"

@dataclass
class ChangeRecord:
    """変更履歴レコード"""
    timestamp: str
    change_type: str  # "create", "update", "delete"
    field_name: str
    old_value: Any
    new_value: Any
    reason: str = ""
    author: str = "system"

class StrategyDataPersistence:
    """
    戦略特性データの永続化管理クラス
    
    機能：
    - データのバージョン管理
    - 変更履歴の保持
    - 複数データソースからの統合
    - CRUD操作の提供
    """
    
    def __init__(self, base_path: str = None):
        """
        永続化管理クラスの初期化
        
        Args:
            base_path: データ保存ベースパス（デフォルト: logs/strategy_persistence）
        """
        if base_path is None:
            base_path = os.path.join("logs", "strategy_persistence")
        
        self.base_path = base_path
        self.data_dir = os.path.join(base_path, "data")
        self.versions_dir = os.path.join(base_path, "versions")
        self.history_dir = os.path.join(base_path, "history")
        self.metadata_dir = os.path.join(base_path, "metadata")
        
        # ディレクトリ作成
        for directory in [self.data_dir, self.versions_dir, self.history_dir, self.metadata_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 関連マネージャーの初期化
        self.characteristics_manager = StrategyCharacteristicsManager()
        self.parameters_manager = OptimizedParameterManager()
        
        # メタデータファイルパス
        self.metadata_file = os.path.join(self.metadata_dir, "persistence_metadata.json")
        
        # メタデータ初期化
        self._initialize_metadata()
        
        logger.info(f"StrategyDataPersistence initialized with base_path: {base_path}")
    
    def _initialize_metadata(self):
        """メタデータファイルの初期化"""
        if not os.path.exists(self.metadata_file):
            metadata = {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0.0",
                "strategies": {},
                "data_sources": {
                    "characteristics": "strategy_characteristics_manager",
                    "parameters": "optimized_parameters"
                },
                "schema_version": "1.0"
            }
            self._save_json(self.metadata_file, metadata)
            logger.info("Metadata file initialized")
    
    def _generate_version_string(self) -> str:
        """バージョン文字列の生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v{timestamp}"
    
    def _calculate_hash(self, data: Any) -> str:
        """データのハッシュ値計算"""
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(json_str.encode('utf-8')).hexdigest()
    
    def _save_json(self, filepath: str, data: Any) -> bool:
        """JSONファイルの保存"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON file {filepath}: {e}")
            return False
    
    def _load_json(self, filepath: str) -> Optional[Dict[str, Any]]:
        """JSONファイルの読み込み"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to load JSON file {filepath}: {e}")
            return None
    
    def _record_change(self, strategy_name: str, change_type: str, 
                      field_name: str, old_value: Any, new_value: Any, 
                      reason: str = "", author: str = "system"):
        """変更履歴の記録"""
        change_record = ChangeRecord(
            timestamp=datetime.now().isoformat(),
            change_type=change_type,
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            author=author
        )
        
        history_file = os.path.join(self.history_dir, f"{strategy_name}_history.json")
        history_data = self._load_json(history_file) or {"changes": []}
        
        history_data["changes"].append({
            "timestamp": change_record.timestamp,
            "change_type": change_record.change_type,
            "field_name": change_record.field_name,
            "old_value": change_record.old_value,
            "new_value": change_record.new_value,
            "reason": change_record.reason,
            "author": change_record.author
        })
        
        # 履歴を最新100件に制限
        if len(history_data["changes"]) > 100:
            history_data["changes"] = history_data["changes"][-100:]
        
        self._save_json(history_file, history_data)
        logger.debug(f"Change recorded for {strategy_name}: {change_type} on {field_name}")
    
    def _create_version_backup(self, strategy_name: str, data: Dict[str, Any], 
                              description: str = "") -> DataVersion:
        """バージョンバックアップの作成"""
        version = self._generate_version_string()
        hash_value = self._calculate_hash(data)
        
        version_data = DataVersion(
            version=version,
            timestamp=datetime.now().isoformat(),
            hash_value=hash_value,
            description=description
        )
        
        # バージョンファイルの保存
        version_file = os.path.join(self.versions_dir, f"{strategy_name}_{version}.json")
        version_info = {
            "version_info": {
                "version": version_data.version,
                "timestamp": version_data.timestamp,
                "hash_value": version_data.hash_value,
                "description": version_data.description,
                "author": version_data.author
            },
            "data": data
        }
        
        self._save_json(version_file, version_info)
        logger.info(f"Version backup created: {strategy_name} {version}")
        
        return version_data
    
    def save_strategy_data(self, strategy_name: str, data: Dict[str, Any], 
                          description: str = "", author: str = "system") -> bool:
        """
        戦略データの保存
        
        Args:
            strategy_name: 戦略名
            data: 保存するデータ
            description: 変更の説明
            author: 作成者
            
        Returns:
            bool: 保存成功の可否
        """
        try:
            # 既存データの取得
            data_file = os.path.join(self.data_dir, f"{strategy_name}.json")
            old_data = self._load_json(data_file)
            
            # バージョンバックアップの作成
            if old_data:
                self._create_version_backup(strategy_name, old_data, 
                                          f"Backup before update: {description}")
            
            # 新しいデータの準備
            new_data = {
                "strategy_name": strategy_name,
                "last_updated": datetime.now().isoformat(),
                "author": author,
                "version": self._generate_version_string(),
                "hash_value": self._calculate_hash(data),
                "data": data
            }
            
            # データ保存
            if self._save_json(data_file, new_data):
                # 変更履歴の記録
                change_type = "create" if not old_data else "update"
                self._record_change(strategy_name, change_type, "full_data", 
                                  old_data, new_data, description, author)
                
                # メタデータの更新
                self._update_metadata(strategy_name, new_data)
                
                logger.info(f"Strategy data saved successfully: {strategy_name}")
                return True
            else:
                logger.error(f"Failed to save strategy data: {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving strategy data {strategy_name}: {e}")
            return False
    
    def load_strategy_data(self, strategy_name: str, version: str = None) -> Optional[Dict[str, Any]]:
        """
        戦略データの読み込み
        
        Args:
            strategy_name: 戦略名
            version: 指定バージョン（None=最新）
            
        Returns:
            Dict: 戦略データ、存在しない場合はNone
        """
        try:
            if version:
                # 指定バージョンの読み込み
                version_file = os.path.join(self.versions_dir, f"{strategy_name}_{version}.json")
                version_data = self._load_json(version_file)
                return version_data["data"] if version_data else None
            else:
                # 最新データの読み込み
                data_file = os.path.join(self.data_dir, f"{strategy_name}.json")
                strategy_data = self._load_json(data_file)
                return strategy_data["data"] if strategy_data else None
                
        except Exception as e:
            logger.error(f"Error loading strategy data {strategy_name}: {e}")
            return None
    
    def delete_strategy_data(self, strategy_name: str, reason: str = "", 
                           author: str = "system") -> bool:
        """
        戦略データの削除
        
        Args:
            strategy_name: 戦略名
            reason: 削除理由
            author: 削除者
            
        Returns:
            bool: 削除成功の可否
        """
        try:
            data_file = os.path.join(self.data_dir, f"{strategy_name}.json")
            
            if os.path.exists(data_file):
                # 削除前のバックアップ
                old_data = self._load_json(data_file)
                if old_data:
                    self._create_version_backup(strategy_name, old_data, 
                                              f"Backup before deletion: {reason}")
                    
                    # 変更履歴の記録
                    self._record_change(strategy_name, "delete", "full_data", 
                                      old_data, None, reason, author)
                
                # ファイル削除
                os.remove(data_file)
                
                # メタデータの更新
                self._remove_from_metadata(strategy_name)
                
                logger.info(f"Strategy data deleted: {strategy_name}")
                return True
            else:
                logger.warning(f"Strategy data not found for deletion: {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting strategy data {strategy_name}: {e}")
            return False
    
    def _update_metadata(self, strategy_name: str, data: Dict[str, Any]):
        """メタデータの更新"""
        metadata = self._load_json(self.metadata_file) or {}
        
        if "strategies" not in metadata:
            metadata["strategies"] = {}
        
        metadata["strategies"][strategy_name] = {
            "last_updated": data.get("last_updated"),
            "version": data.get("version"),
            "hash_value": data.get("hash_value"),
            "author": data.get("author")
        }
        
        metadata["last_updated"] = datetime.now().isoformat()
        
        self._save_json(self.metadata_file, metadata)
    
    def _remove_from_metadata(self, strategy_name: str):
        """メタデータからの戦略削除"""
        metadata = self._load_json(self.metadata_file) or {}
        
        if "strategies" in metadata and strategy_name in metadata["strategies"]:
            del metadata["strategies"][strategy_name]
            metadata["last_updated"] = datetime.now().isoformat()
            self._save_json(self.metadata_file, metadata)
    
    def list_strategies(self) -> List[str]:
        """保存されている戦略一覧の取得"""
        try:
            metadata = self._load_json(self.metadata_file) or {}
            return list(metadata.get("strategies", {}).keys())
        except Exception as e:
            logger.error(f"Error listing strategies: {e}")
            return []
    
    def get_strategy_versions(self, strategy_name: str) -> List[Dict[str, Any]]:
        """戦略のバージョン履歴取得"""
        try:
            versions = []
            for filename in os.listdir(self.versions_dir):
                if filename.startswith(f"{strategy_name}_v") and filename.endswith('.json'):
                    version_data = self._load_json(os.path.join(self.versions_dir, filename))
                    if version_data and "version_info" in version_data:
                        versions.append(version_data["version_info"])
            
            # タイムスタンプでソート（新しい順）
            versions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return versions
            
        except Exception as e:
            logger.error(f"Error getting strategy versions {strategy_name}: {e}")
            return []
    
    def get_change_history(self, strategy_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """変更履歴の取得"""
        try:
            history_file = os.path.join(self.history_dir, f"{strategy_name}_history.json")
            history_data = self._load_json(history_file)
            
            if history_data and "changes" in history_data:
                changes = history_data["changes"][-limit:]  # 最新のlimit件
                changes.reverse()  # 新しい順にソート
                return changes
            return []
            
        except Exception as e:
            logger.error(f"Error getting change history {strategy_name}: {e}")
            return []


class StrategyDataIntegrator:
    """
    複数データソースの統合管理クラス
    
    optimized_parameters.pyとstrategy_characteristics_manager.pyから
    データを統合し、最新の最適化済みパラメータを採用
    """
    
    def __init__(self, persistence_manager: StrategyDataPersistence = None):
        """
        統合管理クラスの初期化
        
        Args:
            persistence_manager: 永続化マネージャーのインスタンス
        """
        self.persistence = persistence_manager or StrategyDataPersistence()
        
        logger.info("StrategyDataIntegrator initialized")
    
    def integrate_strategy_data(self, strategy_name: str, ticker: str = None,
                               force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        戦略データの統合
        
        Args:
            strategy_name: 戦略名
            ticker: ティッカーシンボル
            force_refresh: 強制リフレッシュフラグ
            
        Returns:
            Dict: 統合された戦略データ
        """
        try:
            logger.info(f"Integrating strategy data for {strategy_name}")
            
            # 1. 特性データの取得
            characteristics_data = self._get_characteristics_data(strategy_name)
            
            # 2. 最適化パラメータデータの取得
            parameters_data = self._get_parameters_data(strategy_name, ticker)
            
            # 3. データの統合
            integrated_data = self._merge_data_sources(
                characteristics_data, parameters_data, strategy_name, ticker
            )
            
            # 4. 統合データの保存
            if integrated_data:
                description = f"Integrated data from characteristics and parameters"
                if ticker:
                    description += f" for {ticker}"
                
                success = self.persistence.save_strategy_data(
                    strategy_name, integrated_data, description, "integrator"
                )
                
                if success:
                    logger.info(f"Strategy data integration completed: {strategy_name}")
                    return integrated_data
                else:
                    logger.error(f"Failed to save integrated data: {strategy_name}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error integrating strategy data {strategy_name}: {e}")
            return None
    
    def _get_characteristics_data(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """特性データの取得"""
        try:
            # 戦略特性管理クラスから利用可能なメソッドを使用
            characteristics = {}
            
            # トレンド適性データ
            trend_data = {}
            for trend in ["uptrend", "downtrend", "sideways"]:
                trend_info = self.persistence.characteristics_manager.get_trend_suitability(strategy_name, trend)
                if trend_info:
                    trend_data[trend] = trend_info
            
            if trend_data:
                characteristics["trend_suitability"] = trend_data
            
            # ボラティリティ適性データ
            volatility_data = {}
            for vol_level in ["low", "medium", "high"]:
                vol_info = self.persistence.characteristics_manager.get_volatility_suitability(strategy_name, vol_level)
                if vol_info:
                    volatility_data[vol_level] = vol_info
            
            if volatility_data:
                characteristics["volatility_suitability"] = volatility_data
            
            # パラメータ履歴
            param_history = self.persistence.characteristics_manager.get_parameter_history(strategy_name)
            if param_history:
                characteristics["parameter_history"] = param_history
            
            # 最適パラメータ
            best_params = self.persistence.characteristics_manager.get_best_parameters(strategy_name)
            if best_params:
                characteristics["best_parameters"] = best_params
            
            return characteristics if characteristics else None
            
        except Exception as e:
            logger.error(f"Error getting characteristics data for {strategy_name}: {e}")
            return None
    
    def _get_parameters_data(self, strategy_name: str, ticker: str = None) -> Optional[Dict[str, Any]]:
        """最適化パラメータデータの取得"""
        try:
            # 実際に利用可能なメソッドを使用
            params = self.persistence.parameters_manager.get_best_config_by_metric(strategy_name, "sharpe_ratio")
            return params
        except Exception as e:
            logger.error(f"Error getting parameters data for {strategy_name}: {e}")
            return None
    
    def _merge_data_sources(self, characteristics_data: Optional[Dict[str, Any]], 
                          parameters_data: Optional[Dict[str, Any]], 
                          strategy_name: str, ticker: str = None) -> Optional[Dict[str, Any]]:
        """データソースのマージ"""
        try:
            merged_data = {
                "integration_metadata": {
                    "strategy_name": strategy_name,
                    "ticker": ticker,
                    "integration_timestamp": datetime.now().isoformat(),
                    "data_sources": {
                        "characteristics_available": characteristics_data is not None,
                        "parameters_available": parameters_data is not None
                    }
                }
            }
            
            # 特性データの統合
            if characteristics_data:
                merged_data["characteristics"] = characteristics_data
                logger.debug(f"Characteristics data merged for {strategy_name}")
            
            # パラメータデータの統合
            if parameters_data:
                merged_data["parameters"] = parameters_data
                logger.debug(f"Parameters data merged for {strategy_name}")
            
            # 統合データの検証
            if self._validate_merged_data(merged_data):
                return merged_data
            else:
                logger.error(f"Merged data validation failed for {strategy_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error merging data sources for {strategy_name}: {e}")
            return None
    
    def _validate_merged_data(self, merged_data: Dict[str, Any]) -> bool:
        """統合データの検証"""
        try:
            # 必須フィールドの確認
            required_fields = ["integration_metadata"]
            for field in required_fields:
                if field not in merged_data:
                    logger.error(f"Required field missing: {field}")
                    return False
            
            # メタデータの検証
            metadata = merged_data["integration_metadata"]
            if "strategy_name" not in metadata or not metadata["strategy_name"]:
                logger.error("Strategy name is required in metadata")
                return False
            
            # 少なくとも一つのデータソースが必要
            sources = metadata.get("data_sources", {})
            if not (sources.get("characteristics_available") or sources.get("parameters_available")):
                logger.error("At least one data source must be available")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating merged data: {e}")
            return False
    
    def get_latest_integrated_data(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """最新の統合データ取得"""
        return self.persistence.load_strategy_data(strategy_name)
    
    def refresh_strategy_integration(self, strategy_name: str, ticker: str = None) -> bool:
        """戦略統合データの強制リフレッシュ"""
        try:
            integrated_data = self.integrate_strategy_data(strategy_name, ticker, force_refresh=True)
            return integrated_data is not None
        except Exception as e:
            logger.error(f"Error refreshing strategy integration {strategy_name}: {e}")
            return False


# ユーティリティ関数
def create_persistence_manager(base_path: str = None) -> StrategyDataPersistence:
    """永続化マネージャーのファクトリ関数"""
    return StrategyDataPersistence(base_path)

def create_integrator(persistence_manager: StrategyDataPersistence = None) -> StrategyDataIntegrator:
    """統合マネージャーのファクトリ関数"""
    return StrategyDataIntegrator(persistence_manager)


if __name__ == "__main__":
    # 基本テスト
    persistence = create_persistence_manager()
    integrator = create_integrator(persistence)
    
    # テストデータ
    test_data = {
        "test_field": "test_value",
        "timestamp": datetime.now().isoformat()
    }
    
    # 保存テスト
    success = persistence.save_strategy_data("test_strategy", test_data, "Initial test data")
    print(f"Save test: {'SUCCESS' if success else 'FAILED'}")
    
    # 読み込みテスト
    loaded_data = persistence.load_strategy_data("test_strategy")
    print(f"Load test: {'SUCCESS' if loaded_data else 'FAILED'}")
    
    # 統合テスト
    integrated_data = integrator.integrate_strategy_data("test_strategy")
    print(f"Integration test: {'SUCCESS' if integrated_data else 'FAILED'}")

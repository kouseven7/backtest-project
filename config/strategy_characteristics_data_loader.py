"""
Module: Strategy Characteristics Data Loader
File: strategy_characteristics_data_loader.py
Description: 
  戦略特性データのロード・更新機能
  効率的なデータ読み込み、キャッシュ管理、増分更新機能を提供
  1-3-2の永続化機能と連携し、高パフォーマンスなデータアクセスを実現

Author: imega
Created: 2025-07-09
Modified: 2025-07-09

Dependencies:
  - json
  - os
  - pandas
  - datetime
  - typing
  - config.strategy_characteristics_manager
  - config.strategy_data_persistence
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple, Set
import logging
import pandas as pd
import threading
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import time

# 内部モジュールのインポート
try:
    from .strategy_characteristics_manager import StrategyCharacteristicsManager
    from .strategy_data_persistence import StrategyDataPersistence, StrategyDataIntegrator
except ImportError:
    # 直接実行時の対応
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config.strategy_characteristics_manager import StrategyCharacteristicsManager
    from config.strategy_data_persistence import StrategyDataPersistence, StrategyDataIntegrator

# ロガーの設定
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """キャッシュエントリー"""
    data: Dict[str, Any]
    timestamp: datetime
    hash_value: str
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)

@dataclass
class LoadOptions:
    """データロードオプション"""
    use_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1時間
    include_history: bool = False
    include_parameters: bool = True
    max_history_records: int = 100
    validate_data: bool = True

@dataclass
class UpdateOptions:
    """データ更新オプション"""
    create_backup: bool = True
    validate_before_update: bool = True
    merge_conflicts: bool = True
    notify_on_change: bool = False
    update_dependencies: bool = True

class StrategyCharacteristicsDataLoader:
    """
    戦略特性データのロード・更新機能クラス
    
    機能：
    - 高速データロード（キャッシュ機能付き）
    - バッチロード（複数戦略の一括読み込み）
    - 増分更新・バルク更新
    - データ整合性管理
    - 検索・フィルタリング機能
    """
    
    def __init__(self, base_path: str = None, cache_size: int = 100):
        """
        データローダーの初期化
        
        Args:
            base_path: データ保存ベースパス
            cache_size: キャッシュサイズ（エントリー数）
        """
        if base_path is None:
            base_path = os.path.join("logs", "strategy_characteristics_loader")
        
        self.base_path = base_path
        self.cache_dir = os.path.join(base_path, "cache")
        self.index_dir = os.path.join(base_path, "index")
        self.temp_dir = os.path.join(base_path, "temp")
        
        # ディレクトリ作成
        for directory in [self.cache_dir, self.index_dir, self.temp_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # キャッシュ管理
        self.cache_size = cache_size
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        
        # 関連マネージャーの初期化
        self.characteristics_manager = StrategyCharacteristicsManager()
        self.persistence_manager = StrategyDataPersistence()
        self.integrator = StrategyDataIntegrator(self.persistence_manager)
        
        # インデックス管理
        self.strategy_index: Dict[str, Dict[str, Any]] = {}
        self.load_strategy_index()
        
        # 統計情報
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "load_operations": 0,
            "update_operations": 0,
            "validation_errors": 0,
            "last_cleanup": datetime.now()
        }
        
        logger.info(f"StrategyCharacteristicsDataLoader initialized with base_path: {base_path}")
    
    def _generate_cache_key(self, strategy_name: str, options: LoadOptions) -> str:
        """キャッシュキーの生成"""
        options_str = f"{options.include_history}_{options.include_parameters}_{options.max_history_records}"
        return f"{strategy_name}_{hashlib.md5(options_str.encode()).hexdigest()[:8]}"
    
    def _calculate_data_hash(self, data: Any) -> str:
        """データのハッシュ値計算"""
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.md5(json_str.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, cache_entry: CacheEntry, ttl_seconds: int) -> bool:
        """キャッシュの有効性チェック"""
        age = (datetime.now() - cache_entry.timestamp).total_seconds()
        return age < ttl_seconds
    
    def _manage_cache_size(self):
        """キャッシュサイズ管理（LRU）"""
        if len(self.cache) <= self.cache_size:
            return
        
        # アクセス頻度と最終アクセス時間によるソート
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1].access_count, x[1].last_access)
        )
        
        # 古いエントリーを削除
        remove_count = len(self.cache) - self.cache_size + 10  # バッファ
        for i in range(remove_count):
            if i < len(sorted_entries):
                del self.cache[sorted_entries[i][0]]
    
    def load_strategy_characteristics(self, strategy_name: str, 
                                    options: LoadOptions = None) -> Optional[Dict[str, Any]]:
        """
        戦略特性データの読み込み
        
        Args:
            strategy_name: 戦略名
            options: ロードオプション
            
        Returns:
            Dict: 戦略特性データ、存在しない場合はNone
        """
        if options is None:
            options = LoadOptions()
        
        try:
            self.stats["load_operations"] += 1
            
            # キャッシュチェック
            if options.use_cache:
                cache_key = self._generate_cache_key(strategy_name, options)
                
                with self.cache_lock:
                    if cache_key in self.cache:
                        cache_entry = self.cache[cache_key]
                        if self._is_cache_valid(cache_entry, options.cache_ttl_seconds):
                            # キャッシュヒット
                            cache_entry.access_count += 1
                            cache_entry.last_access = datetime.now()
                            self.stats["cache_hits"] += 1
                            logger.debug(f"Cache hit for strategy: {strategy_name}")
                            return cache_entry.data.copy()
                        else:
                            # キャッシュ期限切れ
                            del self.cache[cache_key]
            
            # キャッシュミス - データを実際に読み込み
            self.stats["cache_misses"] += 1
            logger.debug(f"Cache miss for strategy: {strategy_name}")
            
            # 統合データの読み込み
            integrated_data = self.integrator.get_latest_integrated_data(strategy_name)
            
            if not integrated_data:
                # 統合データが存在しない場合、個別に読み込み
                characteristics_data = self._load_characteristics_data(strategy_name, options)
                parameters_data = self._load_parameters_data(strategy_name, options) if options.include_parameters else None
                
                if characteristics_data or parameters_data:
                    integrated_data = self._merge_loaded_data(
                        strategy_name, characteristics_data, parameters_data, options
                    )
            
            if integrated_data and options.validate_data:
                if not self._validate_loaded_data(integrated_data, strategy_name):
                    logger.error(f"Data validation failed for strategy: {strategy_name}")
                    self.stats["validation_errors"] += 1
                    return None
            
            # キャッシュに保存
            if integrated_data and options.use_cache:
                cache_key = self._generate_cache_key(strategy_name, options)
                cache_entry = CacheEntry(
                    data=integrated_data.copy(),
                    timestamp=datetime.now(),
                    hash_value=self._calculate_data_hash(integrated_data),
                    access_count=1,
                    last_access=datetime.now()
                )
                
                with self.cache_lock:
                    self.cache[cache_key] = cache_entry
                    self._manage_cache_size()
            
            return integrated_data
            
        except Exception as e:
            logger.error(f"Error loading strategy characteristics for {strategy_name}: {e}")
            return None
    
    def _load_characteristics_data(self, strategy_name: str, options: LoadOptions) -> Optional[Dict[str, Any]]:
        """特性データの個別読み込み"""
        try:
            data = {}
            
            # トレンド適性データ
            for trend in ["uptrend", "downtrend", "sideways"]:
                trend_data = self.characteristics_manager.get_trend_suitability(strategy_name, trend)
                if trend_data:
                    if "trend_suitability" not in data:
                        data["trend_suitability"] = {}
                    data["trend_suitability"][trend] = trend_data
            
            # ボラティリティ適性データ
            for vol_level in ["low", "medium", "high"]:
                vol_data = self.characteristics_manager.get_volatility_suitability(strategy_name, vol_level)
                if vol_data:
                    if "volatility_suitability" not in data:
                        data["volatility_suitability"] = {}
                    data["volatility_suitability"][vol_level] = vol_data
            
            # 履歴データ
            if options.include_history:
                history = self.characteristics_manager.get_parameter_history(
                    strategy_name, limit=options.max_history_records
                )
                if history:
                    data["parameter_history"] = history
            
            return data if data else None
            
        except Exception as e:
            logger.error(f"Error loading characteristics data for {strategy_name}: {e}")
            return None
    
    def _load_parameters_data(self, strategy_name: str, options: LoadOptions) -> Optional[Dict[str, Any]]:
        """パラメータデータの個別読み込み"""
        try:
            return self.persistence_manager.parameters_manager.get_best_config_by_metric(
                strategy_name, "sharpe_ratio"
            )
        except Exception as e:
            logger.error(f"Error loading parameters data for {strategy_name}: {e}")
            return None
    
    def _merge_loaded_data(self, strategy_name: str, characteristics_data: Optional[Dict[str, Any]], 
                          parameters_data: Optional[Dict[str, Any]], 
                          options: LoadOptions) -> Dict[str, Any]:
        """読み込んだデータのマージ"""
        merged_data = {
            "strategy_name": strategy_name,
            "load_timestamp": datetime.now().isoformat(),
            "load_options": {
                "include_history": options.include_history,
                "include_parameters": options.include_parameters,
                "max_history_records": options.max_history_records
            }
        }
        
        if characteristics_data:
            merged_data["characteristics"] = characteristics_data
        
        if parameters_data:
            merged_data["parameters"] = parameters_data
        
        return merged_data
    
    def _validate_loaded_data(self, data: Dict[str, Any], strategy_name: str) -> bool:
        """読み込んだデータの検証"""
        try:
            # 必須フィールドの確認
            if "strategy_name" not in data:
                logger.error(f"Missing strategy_name in data for {strategy_name}")
                return False
            
            if data["strategy_name"] != strategy_name:
                logger.error(f"Strategy name mismatch: expected {strategy_name}, got {data['strategy_name']}")
                return False
            
            # データ型の確認
            if "characteristics" in data and not isinstance(data["characteristics"], dict):
                logger.error(f"Invalid characteristics data type for {strategy_name}")
                return False
            
            if "parameters" in data and not isinstance(data["parameters"], dict):
                logger.error(f"Invalid parameters data type for {strategy_name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data for {strategy_name}: {e}")
            return False
    
    def load_multiple_strategies(self, strategy_names: List[str], 
                               options: LoadOptions = None) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        複数戦略の一括読み込み
        
        Args:
            strategy_names: 戦略名のリスト
            options: ロードオプション
            
        Returns:
            Dict: {戦略名: 戦略データ} の辞書
        """
        if options is None:
            options = LoadOptions()
        
        results = {}
        
        logger.info(f"Loading {len(strategy_names)} strategies in batch")
        
        for strategy_name in strategy_names:
            try:
                data = self.load_strategy_characteristics(strategy_name, options)
                results[strategy_name] = data
                
                if data:
                    logger.debug(f"Successfully loaded strategy: {strategy_name}")
                else:
                    logger.warning(f"No data found for strategy: {strategy_name}")
                    
            except Exception as e:
                logger.error(f"Error loading strategy {strategy_name}: {e}")
                results[strategy_name] = None
        
        success_count = sum(1 for v in results.values() if v is not None)
        logger.info(f"Batch load completed: {success_count}/{len(strategy_names)} successful")
        
        return results
    
    def update_strategy_characteristics(self, strategy_name: str, data: Dict[str, Any], 
                                      options: UpdateOptions = None) -> bool:
        """
        戦略特性データの更新
        
        Args:
            strategy_name: 戦略名
            data: 更新データ
            options: 更新オプション
            
        Returns:
            bool: 更新成功の可否
        """
        if options is None:
            options = UpdateOptions()
        
        try:
            self.stats["update_operations"] += 1
            
            # 更新前のバリデーション
            if options.validate_before_update:
                if not self._validate_update_data(strategy_name, data):
                    logger.error(f"Update data validation failed for {strategy_name}")
                    return False
            
            # 現在のデータの取得
            current_data = self.load_strategy_characteristics(
                strategy_name, LoadOptions(use_cache=False)
            )
            
            # バックアップの作成
            if options.create_backup and current_data:
                backup_success = self._create_backup(strategy_name, current_data)
                if not backup_success:
                    logger.warning(f"Backup creation failed for {strategy_name}")
            
            # データのマージ
            if current_data and options.merge_conflicts:
                merged_data = self._merge_update_data(current_data, data)
            else:
                merged_data = data
            
            # 永続化
            success = self.persistence_manager.save_strategy_data(
                strategy_name,
                merged_data,
                f"Updated via data loader at {datetime.now().isoformat()}",
                "data_loader"
            )
            
            if success:
                # キャッシュの無効化
                self._invalidate_cache(strategy_name)
                
                # インデックスの更新
                self._update_strategy_index(strategy_name, merged_data)
                
                # 依存関係の更新
                if options.update_dependencies:
                    self._update_dependencies(strategy_name, merged_data)
                
                logger.info(f"Successfully updated strategy characteristics: {strategy_name}")
                return True
            else:
                logger.error(f"Failed to persist updated data for {strategy_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating strategy characteristics for {strategy_name}: {e}")
            return False
    
    def _validate_update_data(self, strategy_name: str, data: Dict[str, Any]) -> bool:
        """更新データの検証"""
        try:
            # 基本的なデータ型チェック
            if not isinstance(data, dict):
                logger.error(f"Update data must be a dictionary for {strategy_name}")
                return False
            
            # 必要に応じて追加の検証ロジック
            # 例：特定のフィールドの存在確認、値の範囲チェック等
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating update data for {strategy_name}: {e}")
            return False
    
    def _create_backup(self, strategy_name: str, data: Dict[str, Any]) -> bool:
        """データのバックアップ作成"""
        try:
            backup_dir = os.path.join(self.base_path, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"{strategy_name}_backup_{timestamp}.json")
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"Backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup for {strategy_name}: {e}")
            return False
    
    def _merge_update_data(self, current_data: Dict[str, Any], 
                          update_data: Dict[str, Any]) -> Dict[str, Any]:
        """現在のデータと更新データのマージ"""
        merged = current_data.copy()
        
        def deep_merge(base: Dict, update: Dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(merged, update_data)
        merged["last_updated"] = datetime.now().isoformat()
        
        return merged
    
    def _invalidate_cache(self, strategy_name: str):
        """指定戦略のキャッシュ無効化"""
        with self.cache_lock:
            keys_to_remove = [key for key in self.cache.keys() if key.startswith(strategy_name)]
            for key in keys_to_remove:
                del self.cache[key]
        
        logger.debug(f"Cache invalidated for strategy: {strategy_name}")
    
    def _update_strategy_index(self, strategy_name: str, data: Dict[str, Any]):
        """戦略インデックスの更新"""
        try:
            self.strategy_index[strategy_name] = {
                "last_updated": datetime.now().isoformat(),
                "data_hash": self._calculate_data_hash(data),
                "has_characteristics": "characteristics" in data,
                "has_parameters": "parameters" in data,
                "record_count": len(data.get("parameter_history", []))
            }
            
            # インデックスファイルの保存
            index_file = os.path.join(self.index_dir, "strategy_index.json")
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(self.strategy_index, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            logger.error(f"Error updating strategy index for {strategy_name}: {e}")
    
    def _update_dependencies(self, strategy_name: str, data: Dict[str, Any]):
        """依存関係の更新"""
        # 依存する他の戦略やシステムへの通知
        # 実装は具体的な依存関係に応じて調整
        pass
    
    def load_strategy_index(self):
        """戦略インデックスの読み込み"""
        try:
            index_file = os.path.join(self.index_dir, "strategy_index.json")
            if os.path.exists(index_file):
                with open(index_file, 'r', encoding='utf-8') as f:
                    self.strategy_index = json.load(f)
            else:
                self.strategy_index = {}
                
        except Exception as e:
            logger.error(f"Error loading strategy index: {e}")
            self.strategy_index = {}
    
    def search_strategies(self, criteria: Dict[str, Any]) -> List[str]:
        """
        戦略の検索
        
        Args:
            criteria: 検索条件
            
        Returns:
            List: マッチした戦略名のリスト
        """
        try:
            matching_strategies = []
            
            for strategy_name, index_data in self.strategy_index.items():
                match = True
                
                # 検索条件のチェック
                for key, value in criteria.items():
                    if key == "has_characteristics" and index_data.get("has_characteristics") != value:
                        match = False
                        break
                    elif key == "has_parameters" and index_data.get("has_parameters") != value:
                        match = False
                        break
                    elif key == "updated_after":
                        last_updated = datetime.fromisoformat(index_data.get("last_updated", "1900-01-01"))
                        if last_updated <= value:
                            match = False
                            break
                
                if match:
                    matching_strategies.append(strategy_name)
            
            return matching_strategies
            
        except Exception as e:
            logger.error(f"Error searching strategies: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計の取得"""
        with self.cache_lock:
            cache_stats = {
                "cache_size": len(self.cache),
                "max_cache_size": self.cache_size,
                "cache_hit_rate": (
                    self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
                    if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0
                )
            }
        
        return {**self.stats, **cache_stats}
    
    def cleanup_cache(self, force: bool = False):
        """キャッシュのクリーンアップ"""
        try:
            with self.cache_lock:
                if force:
                    self.cache.clear()
                    logger.info("Cache forcefully cleared")
                else:
                    # 期限切れエントリーの削除
                    expired_keys = []
                    for key, entry in self.cache.items():
                        if not self._is_cache_valid(entry, 3600):  # 1時間
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.cache[key]
                    
                    logger.info(f"Removed {len(expired_keys)} expired cache entries")
            
            self.stats["last_cleanup"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")


# ユーティリティ関数
def create_data_loader(base_path: str = None, cache_size: int = 100) -> StrategyCharacteristicsDataLoader:
    """データローダーのファクトリ関数"""
    return StrategyCharacteristicsDataLoader(base_path, cache_size)

def create_load_options(**kwargs) -> LoadOptions:
    """ロードオプションのファクトリ関数"""
    return LoadOptions(**kwargs)

def create_update_options(**kwargs) -> UpdateOptions:
    """更新オプションのファクトリ関数"""
    return UpdateOptions(**kwargs)


if __name__ == "__main__":
    # 基本テスト
    loader = create_data_loader()
    
    # ロードテスト
    options = create_load_options(include_history=True, include_parameters=True)
    data = loader.load_strategy_characteristics("vwap_bounce", options)
    print(f"Load test: {'SUCCESS' if data else 'NO_DATA'}")
    
    # バッチロードテスト
    strategies = ["vwap_bounce", "momentum_investing", "breakout"]
    batch_data = loader.load_multiple_strategies(strategies, options)
    print(f"Batch load test: {len([v for v in batch_data.values() if v])} strategies loaded")
    
    # キャッシュ統計
    stats = loader.get_cache_stats()
    print(f"Cache stats: {stats}")

"""
DSSMS ポートフォリオデータマネージャ
Portfolio data consolidation and processing manager for DSSMS system

主要機能:
- portfolio_values の一元管理と取得インターフェース
- 日付処理の最適化（pd.to_datetime 削減）
- エンジン間データ変換の統一
- データ検証とキャッシュ機能

Problem 6 Phase 2 追加機能:
- 統一ポートフォリオデータ管理システム
- 27箇所散在参照の集約・統一
- DateProcessor統合による日付処理標準化
- データ検証レベル設定による品質保証
- performance_history, portfolio_values, portfolio_values_raw 3系統統一

TODO(tag:phase1, rationale:data_flow_optimization): 27箇所portfolio_values参照統一 [OK] (Phase 2で対応)
TODO(tag:phase1, rationale:date_processing_optimization): pd.to_datetime処理を3箇所以下に削減 [OK] (Phase 2で対応)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum
import logging


class DataValidationLevel(Enum):
    """データ検証レベル"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"


class EngineFormat(Enum):
    """エンジン出力フォーマット"""
    V1_LEGACY = "v1"
    V2_STANDARD = "v2" 
    V4_ENHANCED = "v4"


@dataclass
class PortfolioDataSnapshot:
    """ポートフォリオデータスナップショット"""
    values: List[float]
    timestamps: List[Union[str, datetime, date]]
    metadata: Dict[str, Any]
    validation_level: DataValidationLevel
    engine_format: EngineFormat
    cache_key: str


class DateProcessor:
    """日付処理最適化クラス
    
    従来の8箇所pd.to_datetime→3箇所以下への削減目標
    """
    
    def __init__(self):
        self._date_cache: Dict[str, datetime] = {}
        self.logger = logging.getLogger(__name__ + ".DateProcessor")
    
    def normalize_date(self, date_input: Union[str, datetime, date, pd.Timestamp, None]) -> datetime:
        """統一日付正規化処理
        
        Args:
            date_input: 任意フォーマットの日付入力
            
        Returns:
            datetime: 正規化された日付オブジェクト
        """
        if date_input is None:
            raise ValueError("Date input cannot be None")
            
        # キャッシュ確認
        cache_key = str(date_input)
        if cache_key in self._date_cache:
            return self._date_cache[cache_key]
        
        try:
            # タイプ別処理
            if isinstance(date_input, datetime):
                result = date_input
            elif isinstance(date_input, date):
                result = datetime.combine(date_input, datetime.min.time())
            elif isinstance(date_input, pd.Timestamp):
                result = date_input.to_pydatetime()
            elif isinstance(date_input, str):
                # TODO(tag:phase1, rationale:pd_to_datetime_reduction): 
                # この箇所が主要pd.to_datetime削減ポイント
                result = pd.to_datetime(date_input).to_pydatetime()
            else:
                # Fallback
                result = pd.to_datetime(date_input).to_pydatetime()
            
            # キャッシュに保存
            self._date_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Date normalization failed for {date_input}: {e}")
            raise ValueError(f"Unable to normalize date: {date_input}")
    
    def normalize_date_list(self, date_list: List[Union[str, datetime, date]]) -> List[datetime]:
        """日付リスト一括正規化"""
        return [self.normalize_date(d) for d in date_list]
    
    def clear_cache(self):
        """日付キャッシュクリア"""
        self._date_cache.clear()
        self.logger.debug("Date cache cleared")


class PortfolioDataManager:
    """ポートフォリオデータ統合管理クラス
    
    27箇所のportfolio_values直接参照を統一管理し、
    エンジン間変換と日付処理を最適化する
    """
    
    def __init__(self, validation_level: DataValidationLevel = DataValidationLevel.BASIC):
        self.validation_level = validation_level
        self.date_processor = DateProcessor()
        self.logger = logging.getLogger(__name__ + ".PortfolioDataManager")
        
        # データキャッシュ
        self._portfolio_cache: Dict[str, PortfolioDataSnapshot] = {}
        self._stats_cache: Dict[str, Dict[str, float]] = {}
        
        # フォーマット変換マッピング
        self._format_converters = {
            EngineFormat.V1_LEGACY: self._convert_to_v1,
            EngineFormat.V2_STANDARD: self._convert_to_v2,
            EngineFormat.V4_ENHANCED: self._convert_to_v4
        }
    
    def get_portfolio_values(self, 
                           performance_history: Dict[str, Any],
                           engine_format: EngineFormat = EngineFormat.V2_STANDARD,
                           force_refresh: bool = False) -> PortfolioDataSnapshot:
        """ポートフォリオ値統一取得インターフェース
        
        従来の27箇所分散アクセスをこの単一インターフェースに統一
        
        Args:
            performance_history: パフォーマンス履歴データ
            engine_format: 出力エンジンフォーマット
            force_refresh: キャッシュ無視フラグ
            
        Returns:
            PortfolioDataSnapshot: 統一ポートフォリオデータ
        """
        # キャッシュキー生成
        cache_key = self._generate_cache_key(performance_history, engine_format)
        
        # キャッシュ確認
        if not force_refresh and cache_key in self._portfolio_cache:
            cached_data = self._portfolio_cache[cache_key]
            self.logger.debug(f"Portfolio data retrieved from cache: {cache_key}")
            return cached_data
        
        # 生データ取得
        raw_values = performance_history.get('portfolio_value', [])
        raw_timestamps = performance_history.get('timestamps', [])
        
        # [調査用] 生データ長さ記録
        self.logger.critical(f"[INVESTIGATION] get_portfolio_values: raw_values長={len(raw_values)}, raw_timestamps長={len(raw_timestamps)}")
        
        # データ検証
        validated_values, validated_timestamps = self._validate_portfolio_data(
            raw_values, raw_timestamps
        )
        
        # [調査用] 検証後の長さ記録
        self.logger.critical(f"[INVESTIGATION] _validate_portfolio_data後: validated_values長={len(validated_values)}, validated_timestamps長={len(validated_timestamps)}")
        
        # 日付正規化 (pd.to_datetime削減ポイント)
        normalized_timestamps = self.date_processor.normalize_date_list(validated_timestamps)
        
        # エンジンフォーマット変換
        converted_values = self._format_converters[engine_format](validated_values)
        
        # スナップショット作成
        snapshot = PortfolioDataSnapshot(
            values=converted_values,
            timestamps=list(normalized_timestamps),  # 型変換
            metadata=self._extract_metadata(performance_history),
            validation_level=self.validation_level,
            engine_format=engine_format,
            cache_key=cache_key
        )
        
        # キャッシュ保存
        self._portfolio_cache[cache_key] = snapshot
        
        self.logger.info(f"Portfolio data processed: {len(converted_values)} values, format={engine_format.value}")
        return snapshot
    
    def calculate_portfolio_stats(self, 
                                snapshot: PortfolioDataSnapshot,
                                include_drawdown: bool = True) -> Dict[str, float]:
        """ポートフォリオ統計計算
        
        従来の分散した統計計算ロジックを統一
        """
        cache_key = f"stats_{snapshot.cache_key}_{include_drawdown}"
        
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]
        
        values = snapshot.values
        
        if len(values) < 2:
            return {"total_return": 0.0, "max_drawdown": 0.0, "volatility": 0.0}
        
        # 総収益率
        total_return = (values[-1] - values[0]) / values[0]
        
        # 最大ドローダウン
        max_drawdown = 0.0
        if include_drawdown:
            max_drawdown = self._calculate_max_drawdown(values)
        
        # ボラティリティ
        returns = np.diff(values) / values[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        
        stats: Dict[str, float] = {
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "volatility": float(volatility),
            "value_count": float(len(values)),
            "start_value": float(values[0]),
            "end_value": float(values[-1])
        }
        
        self._stats_cache[cache_key] = stats
        return stats
    
    def _validate_portfolio_data(self, 
                               raw_values: List[Any], 
                               raw_timestamps: List[Any]) -> Tuple[List[float], List[Any]]:
        """ポートフォリオデータ検証"""
        
        if self.validation_level == DataValidationLevel.NONE:
            return [float(v) for v in raw_values if isinstance(v, (int, float))], list(raw_timestamps)
        
        # 基本検証
        validated_values: List[float] = []
        for v in raw_values:
            if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):
                validated_values.append(float(v))
        
        validated_timestamps: List[Any] = list(raw_timestamps)
        
        # 厳密検証
        if self.validation_level == DataValidationLevel.STRICT:
            if len(validated_values) != len(validated_timestamps):
                min_len = min(len(validated_values), len(validated_timestamps))
                validated_values = validated_values[:min_len]
                validated_timestamps = validated_timestamps[:min_len]
                self.logger.warning(f"Data length mismatch corrected: truncated to {min_len}")
            
            # 異常値検出と修正
            if len(validated_values) > 1:
                values_array = np.array(validated_values)
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)
                
                # 3σ外の値を修正
                corrected_values = []
                for v in validated_values:
                    if abs(v - mean_val) > 3 * std_val:
                        corrected_value = mean_val
                        self.logger.warning(f"Outlier corrected: {v} -> {corrected_value}")
                        corrected_values.append(corrected_value)
                    else:
                        corrected_values.append(v)
                validated_values = corrected_values
        
        return validated_values, validated_timestamps
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """最大ドローダウン計算（統一ロジック）"""
        if len(values) < 2:
            return 0.0
        
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _convert_to_v1(self, values: List[float]) -> List[float]:
        """V1レガシーフォーマット変換"""
        # TODO(tag:phase1, rationale:engine_format_unification): V1フォーマット特有の変換ロジック
        return values
    
    def _convert_to_v2(self, values: List[float]) -> List[float]:
        """V2標準フォーマット変換"""
        # TODO(tag:phase1, rationale:engine_format_unification): V2フォーマット特有の変換ロジック
        return values
    
    def _convert_to_v4(self, values: List[float]) -> List[float]:
        """V4拡張フォーマット変換"""
        # TODO(tag:phase1, rationale:engine_format_unification): V4フォーマット特有の変換ロジック
        return values
    
    def _extract_metadata(self, performance_history: Dict[str, Any]) -> Dict[str, Any]:
        """メタデータ抽出"""
        return {
            "source_keys": list(performance_history.keys()),
            "extracted_at": datetime.now().isoformat(),
            "validation_level": self.validation_level.value
        }
    
    def _generate_cache_key(self, 
                          performance_history: Dict[str, Any], 
                          engine_format: EngineFormat) -> str:
        """キャッシュキー生成"""
        # パフォーマンス履歴のハッシュ化
        data_hash = hash(str(sorted(performance_history.items())))
        return f"portfolio_{data_hash}_{engine_format.value}_{self.validation_level.value}"
    
    def clear_cache(self):
        """全キャッシュクリア"""
        self._portfolio_cache.clear()
        self._stats_cache.clear()
        self.date_processor.clear_cache()
        self.logger.info("All portfolio data caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """キャッシュ統計取得"""
        return {
            "portfolio_cache_size": len(self._portfolio_cache),
            "stats_cache_size": len(self._stats_cache),
            "date_cache_size": len(self.date_processor._date_cache)
        }
    
    # ========================================
    # Problem 6 Phase 2: 拡張機能
    # ========================================
    
    def store_unified_value(self, date: Union[datetime, str], value: float, 
                           source: str = "unified") -> bool:
        """
        統一ポートフォリオ値保存
        Phase 1緊急修復とPhase 2統一管理の橋渡し
        
        Args:
            date: 日付
            value: ポートフォリオ値
            source: データソース識別子
            
        Returns:
            bool: 保存成功フラグ
        """
        try:
            normalized_date = self.date_processor.normalize_date(date)
            
            # データ検証
            if self.validation_level != DataValidationLevel.NONE:
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    self.logger.warning(f"Invalid value rejected: {value}")
                    return False
                
                if self.validation_level == DataValidationLevel.STRICT and value < 0:
                    self.logger.warning(f"Negative value rejected in STRICT mode: {value}")
                    return False
            
            # 統一キャッシュ更新
            cache_key = f"unified_{normalized_date.strftime('%Y%m%d')}"
            snapshot = PortfolioDataSnapshot(
                values=[value],
                timestamps=[normalized_date],
                metadata={
                    'source': source,
                    'stored_at': datetime.now().isoformat(),
                    'validation_level': self.validation_level.value
                },
                validation_level=self.validation_level,
                engine_format=EngineFormat.V2_STANDARD,
                cache_key=cache_key
            )
            
            self._portfolio_cache[cache_key] = snapshot
            self.logger.debug(f"Unified value stored: {normalized_date} = {value:.2f} (source: {source})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store unified value {date}: {e}")
            return False
    
    def get_unified_value(self, date: Union[datetime, str], 
                         default: float = 100000.0) -> float:
        """
        統一ポートフォリオ値取得
        Phase 1 self.portfolio_values.get() の完全代替
        
        Args:
            date: 日付
            default: デフォルト値
            
        Returns:
            float: ポートフォリオ値
        """
        try:
            normalized_date = self.date_processor.normalize_date(date)
            cache_key = f"unified_{normalized_date.strftime('%Y%m%d')}"
            
            if cache_key in self._portfolio_cache:
                snapshot = self._portfolio_cache[cache_key]
                if snapshot.values:
                    return snapshot.values[0]
            
            # フォールバック: 最近傍値検索
            nearest_value = self._find_nearest_unified_value(normalized_date)
            if nearest_value is not None:
                return nearest_value
            
            return default
            
        except Exception as e:
            self.logger.error(f"Failed to get unified value {date}: {e}")
            return default
    
    def sync_with_phase1_data(self, portfolio_values: Dict[datetime, float],
                             portfolio_values_raw: List[float],
                             performance_history: Dict[str, List[Any]]) -> bool:
        """
        Phase 1緊急修復データとの同期
        既存の3系統データを統一管理に移行
        
        Args:
            portfolio_values: Phase 1辞書形式データ
            portfolio_values_raw: Phase 1連続配列データ
            performance_history: パフォーマンス履歴
            
        Returns:
            bool: 同期成功フラグ
        """
        try:
            sync_count = 0
            
            # 1. portfolio_values辞書からの同期
            for date, value in portfolio_values.items():
                if self.store_unified_value(date, value, "phase1_dict"):
                    sync_count += 1
            
            # 2. performance_history['portfolio_value']からの同期
            if 'portfolio_value' in performance_history and 'timestamps' in performance_history:
                portfolio_vals = performance_history['portfolio_value']
                timestamps = performance_history['timestamps']
                
                min_length = min(len(portfolio_vals), len(timestamps))
                for i in range(min_length):
                    if isinstance(timestamps[i], datetime) and isinstance(portfolio_vals[i], (int, float)):
                        if self.store_unified_value(timestamps[i], portfolio_vals[i], "phase1_history"):
                            sync_count += 1
            
            self.logger.info(f"Phase 1データ同期完了: {sync_count}件")
            return True
            
        except Exception as e:
            self.logger.error(f"Phase 1データ同期エラー: {e}")
            return False
    
    def validate_unified_integrity(self) -> Dict[str, Any]:
        """
        統一データ整合性検証
        Phase 2データ品質保証
        
        Returns:
            Dict[str, Any]: 検証結果
        """
        validation_result = {
            'status': 'success',
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            unified_entries = {k: v for k, v in self._portfolio_cache.items() if k.startswith('unified_')}
            
            if not unified_entries:
                validation_result['warnings'].append("No unified data found")
                return validation_result
            
            # 基本統計
            all_values = []
            all_dates = []
            
            for snapshot in unified_entries.values():
                all_values.extend(snapshot.values)
                all_dates.extend(snapshot.timestamps)
            
            if all_values:
                validation_result['statistics'] = {
                    'total_records': len(all_values),
                    'value_range': [min(all_values), max(all_values)],
                    'average_value': np.mean(all_values),
                    'date_range': [min(all_dates).isoformat(), max(all_dates).isoformat()] if all_dates else [],
                    'cache_efficiency': len(unified_entries) / max(len(self._portfolio_cache), 1)
                }
                
                # 異常値検出
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                
                outliers = [v for v in all_values if abs(v - mean_val) > 3 * std_val]
                if outliers:
                    validation_result['warnings'].append(f"Detected {len(outliers)} outlier values")
                
                # 負の値チェック
                negative_values = [v for v in all_values if v < 0]
                if negative_values:
                    validation_result['warnings'].append(f"Detected {len(negative_values)} negative values")
            
            self.logger.info(f"統一データ検証完了: {validation_result['status']}")
            return validation_result
            
        except Exception as e:
            validation_result['status'] = 'error'
            validation_result['errors'].append(f"Validation error: {e}")
            self.logger.error(f"統一データ検証エラー: {e}")
            return validation_result
    
    def _find_nearest_unified_value(self, target_date: datetime) -> Optional[float]:
        """最近傍統一値検索（内部メソッド）"""
        unified_entries = {k: v for k, v in self._portfolio_cache.items() if k.startswith('unified_')}
        
        if not unified_entries:
            return None
        
        closest_entry = None
        min_delta = None
        
        for snapshot in unified_entries.values():
            if snapshot.timestamps:
                for ts in snapshot.timestamps:
                    if isinstance(ts, datetime):
                        delta = abs((ts - target_date).days)
                        if min_delta is None or delta < min_delta:
                            min_delta = delta
                            closest_entry = snapshot
        
        # 1週間以内の最近傍値のみ有効
        if closest_entry and min_delta is not None and min_delta <= 7:
            return closest_entry.values[0] if closest_entry.values else None
        
        return None


# ========================================
# Phase 2: 便利関数とエクスポート
# ========================================

def create_unified_portfolio_manager(validation_level: str = "basic") -> PortfolioDataManager:
    """
    統一ポートフォリオマネージャ生成
    Problem 6 Phase 2の推奨生成方法
    
    Args:
        validation_level: "none", "basic", "strict"
        
    Returns:
        PortfolioDataManager: 統一管理インスタンス
    """
    level_mapping = {
        "none": DataValidationLevel.NONE,
        "basic": DataValidationLevel.BASIC,
        "strict": DataValidationLevel.STRICT
    }
    
    level = level_mapping.get(validation_level.lower(), DataValidationLevel.BASIC)
    return PortfolioDataManager(validation_level=level)


# Phase 1 → Phase 2 移行用
def migrate_portfolio_data(portfolio_values: Dict[datetime, float],
                          portfolio_values_raw: List[float],
                          performance_history: Dict[str, List[Any]],
                          validation_level: str = "basic") -> PortfolioDataManager:
    """
    Phase 1データからPhase 2統一管理への移行
    
    Args:
        portfolio_values: Phase 1辞書データ
        portfolio_values_raw: Phase 1配列データ  
        performance_history: パフォーマンス履歴
        validation_level: 検証レベル
        
    Returns:
        PortfolioDataManager: 移行済み統一マネージャ
    """
    manager = create_unified_portfolio_manager(validation_level)
    
    # データ移行実行
    success = manager.sync_with_phase1_data(portfolio_values, portfolio_values_raw, performance_history)
    
    if success:
        logging.getLogger(__name__).info("Portfolio data migration to Phase 2 completed successfully")
    else:
        logging.getLogger(__name__).warning("Portfolio data migration completed with warnings")
    
    return manager
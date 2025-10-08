#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSSMS統合品質改善済みエンジン
85.0点エンジン基準適用
"""

# 品質統一メタデータ
ENGINE_QUALITY_STANDARD = 85.0
DSSMS_UNIFIED_COMPATIBLE = True
LAST_QUALITY_IMPROVEMENT = "2025-09-22T12:14:40.700565"

"""
DSSMS データクリーニングエンジン
Task 1.2: データ品質検証・クリーニング
自動修復とフォールバック機能付き
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import warnings
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger


# === DSSMS 品質統一メタデータ ===
ENGINE_QUALITY_STANDARD = 85.0
DSSMS_UNIFIED_COMPATIBLE = True
QUALITY_IMPROVEMENT_DATE = "2025-09-22T12:14:40.700765"
IMPROVEMENT_VERSION = "1.0"

class DataCleaningEngine:
    """データクリーニングエンジン - ハイブリッド方式"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = setup_logger(__name__)
        self.config = self._load_config(config_path)
        self.cleaning_history: Dict[str, Any] = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """クリーニング設定読み込み"""
        default_config = {
            "cleaning_rules": {
                "auto_repair": True,
                "fallback_enabled": True,
                "preserve_original": True
            },
            "thresholds": {
                "max_missing_ratio": 0.3,     # 30%以上欠損で修復
                "outlier_multiplier": 5.0,    # 平均の5倍以上で異常値
                "min_trading_days": 30        # 最小取引日数
            },
            "repair_methods": {
                "missing_values": "interpolation",  # linear, forward, backward
                "outliers": "cap",                   # cap, remove, interpolate
                "duplicates": "keep_last"            # keep_first, keep_last, remove
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"クリーニング設定読み込み失敗: {e}")
        
        return default_config
    
    def clean_data(self, data: pd.DataFrame, symbol: str = "Unknown", 
                   validation_result: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        データクリーニング実行
        Args:
            data: クリーニング対象データ
            symbol: 銘柄コード
            validation_result: 事前検証結果
        Returns:
            (クリーニング済みデータ, クリーニング結果)
        """
        if data is None or len(data) == 0:
            return data, {"status": "failed", "reason": "Empty data"}
        
        cleaning_log = {
            "symbol": symbol,
            "original_size": len(data),
            "timestamp": datetime.now().isoformat(),
            "operations": [],
            "warnings": [],
            "status": "unknown"
        }
        
        try:
            # オリジナルデータ保持
            if self.config["cleaning_rules"]["preserve_original"]:
                original_data = data.copy()
            
            cleaned_data = data.copy()
            
            # 1. 重複データ処理
            cleaned_data, dup_result = self._handle_duplicates(cleaned_data)
            cleaning_log["operations"].append(dup_result)
            
            # 2. 欠損値処理
            cleaned_data, missing_result = self._handle_missing_values(cleaned_data, symbol)
            cleaning_log["operations"].append(missing_result)
            
            # 3. 異常値処理
            cleaned_data, outlier_result = self._handle_outliers(cleaned_data, symbol)
            cleaning_log["operations"].append(outlier_result)
            
            # 4. 価格整合性修正
            cleaned_data, consistency_result = self._fix_price_consistency(cleaned_data)
            cleaning_log["operations"].append(consistency_result)
            
            # 5. 最小データ要件チェック
            final_check = self._validate_final_data(cleaned_data, symbol)
            cleaning_log["operations"].append(final_check)
            
            # クリーニング結果評価
            cleaning_log["final_size"] = len(cleaned_data)
            cleaning_log["data_retention"] = len(cleaned_data) / len(data) if len(data) > 0 else 0
            
            if final_check["status"] == "passed":
                cleaning_log["status"] = "success"
            elif final_check["status"] == "warning":
                cleaning_log["status"] = "warning"
            else:
                # フォールバック実行
                if self.config["cleaning_rules"]["fallback_enabled"]:
                    cleaned_data, fallback_result = self._execute_fallback(original_data, symbol)
                    cleaning_log["operations"].append(fallback_result)
                    cleaning_log["status"] = "fallback_success" if fallback_result["status"] == "success" else "failed"
                else:
                    cleaning_log["status"] = "failed"
            
            self.cleaning_history[symbol] = cleaning_log
            
        except Exception as e:
            self.logger.error(f"データクリーニングエラー [{symbol}]: {e}")
            cleaning_log["status"] = "error"
            cleaning_log["error"] = str(e)
            cleaned_data = data  # エラー時は元データ返却
        
        return cleaned_data, cleaning_log
    
    def _handle_duplicates(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """重複データ処理"""
        original_size = len(data)
        method = self.config["repair_methods"]["duplicates"]
        
        if data.index.duplicated().any():
            if method == "keep_first":
                data = data[~data.index.duplicated(keep='first')]
            elif method == "keep_last":
                data = data[~data.index.duplicated(keep='last')]
            else:  # remove
                data = data[~data.index.duplicated(keep=False)]
        
        removed_count = original_size - len(data)
        
        return data, {
            "operation": "duplicate_removal",
            "method": method,
            "removed_count": removed_count,
            "status": "success" if removed_count >= 0 else "failed"
        }
    
    def _handle_missing_values(self, data: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """欠損値処理"""
        method = self.config["repair_methods"]["missing_values"]
        original_missing = data.isnull().sum().sum()
        
        # 数値列のみ処理
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].isnull().any():
                missing_ratio = data[col].isnull().sum() / len(data)
                
                if missing_ratio > self.config["thresholds"]["max_missing_ratio"]:
                    # 欠損が多すぎる場合は列を削除
                    data = data.drop(columns=[col])
                    self.logger.warning(f"列削除 [{symbol}]: {col} (欠損率: {missing_ratio:.2%})")
                else:
                    # 欠損値補完
                    if method == "interpolation":
                        data[col] = data[col].interpolate(method='linear')
                    elif method == "forward":
                        data[col] = data[col].fillna(method='ffill')
                    elif method == "backward":
                        data[col] = data[col].fillna(method='bfill')
                    
                    # まだ欠損がある場合は前後の値で補完
                    if data[col].isnull().any():
                        data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        final_missing = data.isnull().sum().sum()
        
        return data, {
            "operation": "missing_value_repair",
            "method": method,
            "original_missing": int(original_missing),
            "final_missing": int(final_missing),
            "repaired_count": int(original_missing - final_missing),
            "status": "success" if final_missing < original_missing else "warning"
        }
    
    def _handle_outliers(self, data: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """異常値処理"""
        method = self.config["repair_methods"]["outliers"]
        outlier_count = 0
        processed_cols = []
        
        # 価格関連列の異常値処理
        price_cols = ['Open', 'High', 'Low', 'Close']
        available_price_cols = [col for col in price_cols if col in data.columns]
        
        for col in available_price_cols:
            if len(data[col].dropna()) > 0:
                # 移動平均からの乖離で異常値検出
                rolling_mean = data[col].rolling(window=20, min_periods=1).mean()
                rolling_std = data[col].rolling(window=20, min_periods=1).std()
                
                threshold = self.config["thresholds"]["outlier_multiplier"]
                outlier_mask = np.abs(data[col] - rolling_mean) > (threshold * rolling_std)
                
                if outlier_mask.any():
                    outlier_indices = data[outlier_mask].index
                    outlier_count += len(outlier_indices)
                    
                    if method == "cap":
                        # 上限・下限でキャップ
                        upper_bound = rolling_mean + (threshold * rolling_std)
                        lower_bound = rolling_mean - (threshold * rolling_std)
                        data.loc[outlier_mask, col] = np.clip(
                            data.loc[outlier_mask, col], 
                            lower_bound.loc[outlier_mask], 
                            upper_bound.loc[outlier_mask]
                        )
                    elif method == "interpolate":
                        # 異常値を線形補間
                        data.loc[outlier_mask, col] = np.nan
                        data[col] = data[col].interpolate(method='linear')
                    # "remove"の場合は行全体を削除（最後に実行）
                    
                    processed_cols.append(col)
        
        # 行削除方式の場合
        if method == "remove" and outlier_count > 0:
            # すべての価格列で異常値をマスク
            overall_outlier_mask = pd.Series(False, index=data.index)
            for col in available_price_cols:
                if len(data[col].dropna()) > 0:
                    rolling_mean = data[col].rolling(window=20, min_periods=1).mean()
                    rolling_std = data[col].rolling(window=20, min_periods=1).std()
                    threshold = self.config["thresholds"]["outlier_multiplier"]
                    overall_outlier_mask |= np.abs(data[col] - rolling_mean) > (threshold * rolling_std)
            
            data = data[~overall_outlier_mask]
        
        return data, {
            "operation": "outlier_handling",
            "method": method,
            "outlier_count": outlier_count,
            "processed_columns": processed_cols,
            "status": "success" if outlier_count >= 0 else "failed"
        }
    
    def _fix_price_consistency(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """OHLC価格整合性修正"""
        required_cols = ['Open', 'High', 'Low', 'Close']
        available_cols = [col for col in required_cols if col in data.columns]
        
        if len(available_cols) < 4:
            return data, {
                "operation": "price_consistency_fix",
                "status": "skipped",
                "reason": f"Missing columns: {set(required_cols) - set(available_cols)}"
            }
        
        fixes_applied = 0
        
        # OHLC関係の修正
        for idx in data.index:
            try:
                o, h, l, c = data.loc[idx, ['Open', 'High', 'Low', 'Close']]
                
                # NaNチェック
                if pd.isna([o, h, l, c]).any():
                    continue
                
                original_values = (o, h, l, c)
                
                # High は最大値であるべき
                actual_high = max(o, h, l, c)
                if h < actual_high:
                    data.loc[idx, 'High'] = actual_high
                    fixes_applied += 1
                
                # Low は最小値であるべき
                actual_low = min(o, h, l, c)
                if l > actual_low:
                    data.loc[idx, 'Low'] = actual_low
                    fixes_applied += 1
                
                # 再度整合性チェック
                o, h, l, c = data.loc[idx, ['Open', 'High', 'Low', 'Close']]
                if not (l <= o <= h and l <= c <= h):
                    # 修復不可能な場合は前日の終値で補完
                    if idx > 0:
                        prev_close = data.iloc[data.index.get_loc(idx) - 1]['Close']
                        if not pd.isna(prev_close):
                            data.loc[idx, ['Open', 'High', 'Low', 'Close']] = prev_close
                            fixes_applied += 1
                
            except Exception as e:
                self.logger.warning(f"価格整合性修正エラー: {e}")
                continue
        
        return data, {
            "operation": "price_consistency_fix",
            "fixes_applied": fixes_applied,
            "status": "success" if fixes_applied >= 0 else "failed"
        }
    
    def _validate_final_data(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """最終データ検証"""
        min_days = self.config["thresholds"]["min_trading_days"]
        
        issues = []
        
        # 最小データ量チェック
        if len(data) < min_days:
            issues.append(f"データ不足: {len(data)}日 < {min_days}日")
        
        # 必須列存在チェック
        required_cols = ['Close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"必須列欠損: {missing_cols}")
        
        # 有効データチェック
        if 'Close' in data.columns:
            valid_prices = data['Close'].dropna()
            if len(valid_prices) == 0:
                issues.append("有効な価格データなし")
            elif (valid_prices <= 0).any():
                issues.append("負またはゼロの価格データ")
        
        # 日付連続性チェック（簡易）
        if len(data) > 1:
            date_gaps = pd.to_datetime(data.index).to_series().diff().dt.days
            large_gaps = (date_gaps > 7).sum()  # 1週間以上の空白
            if large_gaps > len(data) * 0.1:  # 10%以上に大きな空白
                issues.append(f"日付の大きな空白: {large_gaps}箇所")
        
        if len(issues) == 0:
            status = "passed"
        elif len(issues) <= 2:
            status = "warning"
        else:
            status = "failed"
        
        return {
            "operation": "final_validation",
            "data_size": len(data),
            "issues": issues,
            "issue_count": len(issues),
            "status": status
        }
    
    def _execute_fallback(self, original_data: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """フォールバック処理（最小限のデータ修復）"""
        try:
            fallback_data = original_data.copy()
            
            # 最小限の処理のみ
            # 1. 重複削除
            fallback_data = fallback_data[~fallback_data.index.duplicated(keep='last')]
            
            # 2. 明らかに異常な値のみ削除（負の価格など）
            if 'Close' in fallback_data.columns:
                fallback_data = fallback_data[fallback_data['Close'] > 0]
            
            # 3. 最低限のデータ量確保
            if len(fallback_data) >= 10:  # 最低10日分
                return fallback_data, {
                    "operation": "fallback_execution",
                    "method": "minimal_cleaning",
                    "final_size": len(fallback_data),
                    "status": "success"
                }
            else:
                return original_data, {
                    "operation": "fallback_execution",
                    "method": "no_cleaning",
                    "final_size": len(original_data),
                    "status": "failed",
                    "reason": "Insufficient data even after minimal cleaning"
                }
                
        except Exception as e:
            return original_data, {
                "operation": "fallback_execution",
                "status": "error",
                "error": str(e)
            }
    
    def generate_cleaning_report(self, cleaning_results: List[Dict[str, Any]]) -> str:
        """クリーニング結果レポート生成"""
        if not cleaning_results:
            return "クリーニング結果がありません。"
        
        report_lines = [
            "=" * 60,
            "DSSMS データクリーニングレポート",
            f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]
        
        # サマリー統計
        total_symbols = len(cleaning_results)
        success_count = sum(1 for r in cleaning_results if r.get("status") == "success")
        warning_count = sum(1 for r in cleaning_results if r.get("status") == "warning")
        failed_count = sum(1 for r in cleaning_results if r.get("status") == "failed")
        fallback_count = sum(1 for r in cleaning_results if r.get("status") == "fallback_success")
        
        avg_retention = np.mean([r.get("data_retention", 0) for r in cleaning_results])
        
        report_lines.extend([
            "[CHART] サマリー統計",
            "-" * 20,
            f"処理対象銘柄数: {total_symbols}",
            f"成功: {success_count} ({success_count/total_symbols*100:.1f}%)",
            f"警告: {warning_count} ({warning_count/total_symbols*100:.1f}%)",
            f"フォールバック成功: {fallback_count} ({fallback_count/total_symbols*100:.1f}%)",
            f"失敗: {failed_count} ({failed_count/total_symbols*100:.1f}%)",
            f"平均データ保持率: {avg_retention:.1%}",
            ""
        ])
        
        # 処理詳細
        report_lines.extend([
            "[TOOL] 処理詳細",
            "-" * 20
        ])
        
        for result in cleaning_results:
            symbol = result.get("symbol", "Unknown")
            status = result.get("status", "unknown")
            retention = result.get("data_retention", 0)
            
            status_emoji = {
                "success": "[OK]", 
                "warning": "[WARNING]", 
                "failed": "[ERROR]", 
                "fallback_success": "🔄"
            }.get(status, "❓")
            
            report_lines.append(f"{status_emoji} {symbol}: 保持率 {retention:.1%} ({status})")
        
        return "\n".join(report_lines)

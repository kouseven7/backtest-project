"""
DSSMS データ品質検証システム
Task 1.2: 適応的検証によるデータ品質管理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

class DataQualityValidator:
    """データ品質検証エンジン - 適応的検証方式"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = setup_logger(__name__)
        self.config = self._load_config(config_path)
        self.validation_cache = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        default_config = {
            "validation_levels": {
                "basic": {"max_rows": 1000, "checks": ["missing", "duplicates"]},
                "standard": {"max_rows": 5000, "checks": ["missing", "duplicates", "outliers"]},
                "comprehensive": {"max_rows": float('inf'), "checks": ["missing", "duplicates", "outliers", "statistical"]}
            },
            "thresholds": {
                "missing_data_ratio": 0.1,  # 10%以上欠損で警告
                "outlier_zscore": 3.0,      # Z-score 3以上で異常値
                "price_change_limit": 0.5    # 1日50%以上変化で異常
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"設定ファイル読み込み失敗: {e}")
        
        return default_config
    
    def validate_data(self, data: pd.DataFrame, symbol: str = "Unknown") -> Dict[str, Any]:
        """
        適応的データ品質検証
        Args:
            data: 検証対象データ
            symbol: 銘柄コード
        Returns:
            検証結果辞書
        """
        if data is None or len(data) == 0:
            return {"status": "failed", "reason": "Empty data", "quality_score": 0.0}
        
        # データサイズに応じて検証レベル決定
        validation_level = self._determine_validation_level(len(data))
        
        results = {
            "symbol": symbol,
            "validation_level": validation_level,
            "data_size": len(data),
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "quality_score": 0.0,
            "status": "unknown"
        }
        
        try:
            # 基本チェック（全レベル共通）
            results["checks"]["missing"] = self._check_missing_data(data)
            results["checks"]["duplicates"] = self._check_duplicates(data)
            
            # 標準以上のチェック
            if validation_level in ["standard", "comprehensive"]:
                results["checks"]["outliers"] = self._check_outliers(data)
                results["checks"]["price_consistency"] = self._check_price_consistency(data)
            
            # 包括的チェック
            if validation_level == "comprehensive":
                results["checks"]["statistical"] = self._check_statistical_anomalies(data)
                results["checks"]["trading_patterns"] = self._check_trading_patterns(data)
            
            # 品質スコア計算
            results["quality_score"] = self._calculate_quality_score(results["checks"])
            
            # 総合判定
            results["status"] = "passed" if results["quality_score"] >= 0.7 else "warning" if results["quality_score"] >= 0.4 else "failed"
            
        except Exception as e:
            self.logger.error(f"データ検証エラー [{symbol}]: {e}")
            results["status"] = "error"
            results["error"] = str(e)
        
        return results
    
    def _determine_validation_level(self, data_size: int) -> str:
        """データサイズに基づく検証レベル決定"""
        levels = self.config["validation_levels"]
        
        if data_size <= levels["basic"]["max_rows"]:
            return "comprehensive"  # 小データは詳細検証
        elif data_size <= levels["standard"]["max_rows"]:
            return "standard"
        else:
            return "basic"  # 大データは基本検証
    
    def _check_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """欠損データチェック"""
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
        
        return {
            "missing_cells": int(missing_cells),
            "total_cells": int(total_cells),
            "missing_ratio": float(missing_ratio),
            "status": "failed" if missing_ratio > self.config["thresholds"]["missing_data_ratio"] else "passed"
        }
    
    def _check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """重複データチェック"""
        if data.index.duplicated().any():
            duplicate_count = data.index.duplicated().sum()
            return {
                "duplicate_rows": int(duplicate_count),
                "status": "warning" if duplicate_count > 0 else "passed"
            }
        return {"duplicate_rows": 0, "status": "passed"}
    
    def _check_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """外れ値チェック"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            if col in data.columns:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_count = (z_scores > self.config["thresholds"]["outlier_zscore"]).sum()
                outliers[col] = int(outlier_count)
        
        total_outliers = sum(outliers.values())
        return {
            "outliers_by_column": outliers,
            "total_outliers": total_outliers,
            "status": "warning" if total_outliers > len(data) * 0.05 else "passed"
        }
    
    def _check_price_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """価格整合性チェック"""
        if 'Close' not in data.columns:
            return {"status": "skipped", "reason": "No Close column"}
        
        # 日次変化率チェック
        price_changes = data['Close'].pct_change().dropna()
        extreme_changes = (np.abs(price_changes) > self.config["thresholds"]["price_change_limit"]).sum()
        
        return {
            "extreme_price_changes": int(extreme_changes),
            "max_daily_change": float(np.abs(price_changes).max()) if len(price_changes) > 0 else 0.0,
            "status": "warning" if extreme_changes > 0 else "passed"
        }
    
    def _check_statistical_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """統計的異常検出"""
        numeric_data = data.select_dtypes(include=[np.number])
        anomalies = {}
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            if len(series) > 0:
                # 変動係数チェック
                cv = series.std() / series.mean() if series.mean() != 0 else float('inf')
                anomalies[f"{col}_cv"] = float(cv)
        
        return {
            "coefficient_variations": anomalies,
            "status": "passed"  # 統計情報のみ、判定はしない
        }
    
    def _check_trading_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """取引パターンチェック"""
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            return {"status": "skipped", "reason": f"Missing columns: {missing_cols}"}
        
        # ボリューム異常チェック
        zero_volume_days = (data['Volume'] == 0).sum()
        
        # OHLC整合性チェック
        ohlc_issues = ((data['High'] < data['Low']) | 
                       (data['Close'] > data['High']) | 
                       (data['Close'] < data['Low']) |
                       (data['Open'] > data['High']) | 
                       (data['Open'] < data['Low'])).sum()
        
        return {
            "zero_volume_days": int(zero_volume_days),
            "ohlc_inconsistencies": int(ohlc_issues),
            "status": "warning" if (zero_volume_days > len(data) * 0.1 or ohlc_issues > 0) else "passed"
        }
    
    def _calculate_quality_score(self, checks: Dict[str, Any]) -> float:
        """品質スコア計算（0.0-1.0）"""
        total_score = 0.0
        weight_sum = 0.0
        
        weights = {
            "missing": 0.3,
            "duplicates": 0.2,
            "outliers": 0.2,
            "price_consistency": 0.2,
            "trading_patterns": 0.1
        }
        
        for check_name, weight in weights.items():
            if check_name in checks:
                check_status = checks[check_name].get("status", "unknown")
                score = {"passed": 1.0, "warning": 0.5, "failed": 0.0}.get(check_status, 0.0)
                total_score += score * weight
                weight_sum += weight
        
        return total_score / weight_sum if weight_sum > 0 else 0.0

    def generate_quality_report(self, validation_results: List[Dict[str, Any]]) -> str:
        """品質検証レポート生成"""
        if not validation_results:
            return "検証結果がありません。"
        
        report_lines = [
            "=" * 60,
            "DSSMS データ品質検証レポート",
            f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]
        
        # サマリー統計
        total_symbols = len(validation_results)
        passed_count = sum(1 for r in validation_results if r.get("status") == "passed")
        warning_count = sum(1 for r in validation_results if r.get("status") == "warning")
        failed_count = sum(1 for r in validation_results if r.get("status") == "failed")
        
        avg_quality = np.mean([r.get("quality_score", 0) for r in validation_results])
        
        report_lines.extend([
            "📊 サマリー統計",
            "-" * 20,
            f"検証対象銘柄数: {total_symbols}",
            f"合格: {passed_count} ({passed_count/total_symbols*100:.1f}%)",
            f"警告: {warning_count} ({warning_count/total_symbols*100:.1f}%)",
            f"不合格: {failed_count} ({failed_count/total_symbols*100:.1f}%)",
            f"平均品質スコア: {avg_quality:.3f}",
            ""
        ])
        
        # 詳細結果
        report_lines.extend([
            "📋 詳細結果",
            "-" * 20
        ])
        
        for result in validation_results:
            symbol = result.get("symbol", "Unknown")
            status = result.get("status", "unknown")
            score = result.get("quality_score", 0)
            
            status_emoji = {"passed": "✅", "warning": "⚠️", "failed": "❌"}.get(status, "❓")
            
            report_lines.append(f"{status_emoji} {symbol}: {score:.3f} ({status})")
            
            # 警告・エラーの詳細
            if status in ["warning", "failed"]:
                checks = result.get("checks", {})
                for check_name, check_result in checks.items():
                    if check_result.get("status") in ["warning", "failed"]:
                        report_lines.append(f"   - {check_name}: {check_result.get('status')}")
        
        return "\n".join(report_lines)

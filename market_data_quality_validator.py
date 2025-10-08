#!/usr/bin/env python3
"""
TODO #14 Phase 4: MarketDataQualityValidator

リアルマーケットデータの品質検証・補完システム
Phase 3で取得した実データに対する品質保証機能を提供

主要機能:
1. データ完整性チェック（必須列、行数、日付連続性）
2. 異常値検出（価格・出来高の異常パターン）
3. 欠損データ補完（前方補完、線形補間）
4. データ品質レポート生成
5. バックテスト基本理念遵守確認
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings

class QualityLevel(Enum):
    """データ品質レベル"""
    EXCELLENT = "excellent"     # 99%以上の品質
    GOOD = "good"              # 95-99%の品質
    ACCEPTABLE = "acceptable"   # 85-95%の品質
    POOR = "poor"              # 70-85%の品質
    UNUSABLE = "unusable"      # 70%未満の品質

class DataIssueType(Enum):
    """データ問題タイプ"""
    MISSING_COLUMNS = "missing_columns"
    MISSING_DATA = "missing_data"
    OUTLIER_PRICE = "outlier_price"
    OUTLIER_VOLUME = "outlier_volume"
    DATE_GAPS = "date_gaps"
    NEGATIVE_VALUES = "negative_values"
    ZERO_VOLUME = "zero_volume"
    INSUFFICIENT_DATA = "insufficient_data"

@dataclass
class DataQualityIssue:
    """データ品質問題の詳細"""
    issue_type: DataIssueType
    severity: str  # "critical", "warning", "info"
    description: str
    affected_rows: List[int]
    suggested_action: str
    auto_fixable: bool = False

@dataclass
class DataQualityReport:
    """データ品質レポート"""
    data_type: str  # "index_data", "dow_data", etc.
    total_rows: int
    quality_level: QualityLevel
    quality_score: float  # 0-100
    issues: List[DataQualityIssue]
    fixed_issues: List[DataQualityIssue]
    recommendations: List[str]
    validation_timestamp: datetime
    backtest_compliance: bool  # バックテスト基本理念遵守チェック

class MarketDataQualityValidator:
    """
    マーケットデータ品質検証システム
    
    TODO #14 Phase 4: リアルデータ品質保証システム
    Phase 3で取得された実データの品質検証・補完を実行
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 auto_fix_enabled: bool = True,
                 min_data_points: int = 5,
                 outlier_threshold: float = 3.0):
        """
        初期化
        
        Parameters:
            logger: ロガーインスタンス
            auto_fix_enabled: 自動修正有効フラグ
            min_data_points: 最小データポイント数
            outlier_threshold: 異常値検出閾値（標準偏差の倍数）
        """
        self.logger = logger or logging.getLogger(__name__)
        self.auto_fix_enabled = auto_fix_enabled
        self.min_data_points = min_data_points
        self.outlier_threshold = outlier_threshold
        
        # 必須列定義
        self.required_columns = {
            'basic': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'index_data': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'dow_data': ['Open', 'High', 'Low', 'Close', 'Volume']
        }
        
        self.logger.info("MarketDataQualityValidator initialized")
    
    def validate_data_quality(self, 
                            data: pd.DataFrame, 
                            data_type: str = "unknown") -> DataQualityReport:
        """
        データ品質の包括的検証
        
        Parameters:
            data: 検証対象データ
            data_type: データタイプ（"index_data", "dow_data", etc.）
            
        Returns:
            DataQualityReport: 品質検証レポート
        """
        self.logger.info(f"🔍 データ品質検証開始: {data_type} ({len(data)} rows)")
        
        issues = []
        fixed_issues = []
        recommendations = []
        
        # 1. 基本構造チェック
        structure_issues = self._check_data_structure(data, data_type)
        issues.extend(structure_issues)
        
        # 2. 欠損データチェック
        missing_issues = self._check_missing_data(data)
        issues.extend(missing_issues)
        
        # 3. 異常値チェック
        outlier_issues = self._check_outliers(data)
        issues.extend(outlier_issues)
        
        # 4. 日付連続性チェック（インデックスが日付の場合）
        date_issues = self._check_date_continuity(data)
        issues.extend(date_issues)
        
        # 5. バックテスト基本理念遵守チェック
        backtest_compliance = self._check_backtest_compliance(data)
        
        # 6. 自動修正（有効な場合）
        if self.auto_fix_enabled:
            data_fixed, auto_fixed_issues = self._auto_fix_issues(data, issues)
            fixed_issues.extend(auto_fixed_issues)
        else:
            data_fixed = data.copy()
        
        # 7. 品質スコア計算
        quality_score = self._calculate_quality_score(data, issues, fixed_issues)
        quality_level = self._determine_quality_level(quality_score)
        
        # 8. 推奨事項生成
        recommendations = self._generate_recommendations(issues, quality_level)
        
        # レポート作成
        report = DataQualityReport(
            data_type=data_type,
            total_rows=len(data),
            quality_level=quality_level,
            quality_score=quality_score,
            issues=issues,
            fixed_issues=fixed_issues,
            recommendations=recommendations,
            validation_timestamp=datetime.now(),
            backtest_compliance=backtest_compliance
        )
        
        self.logger.info(f"✅ データ品質検証完了: {data_type} - {quality_level.value} ({quality_score:.1f}%)")
        
        return report
    
    def _check_data_structure(self, data: pd.DataFrame, data_type: str) -> List[DataQualityIssue]:
        """データ構造の基本チェック"""
        issues = []
        
        # 必須列チェック
        required_cols = self.required_columns.get(data_type, self.required_columns['basic'])
        missing_columns = [col for col in required_cols if col not in data.columns]
        
        if missing_columns:
            issues.append(DataQualityIssue(
                issue_type=DataIssueType.MISSING_COLUMNS,
                severity="critical",
                description=f"必須列が不足: {missing_columns}",
                affected_rows=[],
                suggested_action=f"必須列を追加: {missing_columns}",
                auto_fixable=False
            ))
        
        # 最小データ数チェック
        if len(data) < self.min_data_points:
            issues.append(DataQualityIssue(
                issue_type=DataIssueType.INSUFFICIENT_DATA,
                severity="critical",
                description=f"データ数不足: {len(data)} < {self.min_data_points}",
                affected_rows=list(range(len(data))),
                suggested_action="より多くのデータを取得",
                auto_fixable=False
            ))
        
        return issues
    
    def _check_missing_data(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """欠損データチェック"""
        issues = []
        
        for column in data.columns:
            missing_count = data[column].isna().sum()
            if missing_count > 0:
                missing_rows = data[data[column].isna()].index.tolist()
                missing_percentage = (missing_count / len(data)) * 100
                
                severity = "critical" if missing_percentage > 20 else "warning" if missing_percentage > 5 else "info"
                
                issues.append(DataQualityIssue(
                    issue_type=DataIssueType.MISSING_DATA,
                    severity=severity,
                    description=f"{column}列に欠損データ: {missing_count}個 ({missing_percentage:.1f}%)",
                    affected_rows=missing_rows,
                    suggested_action="前方補完または線形補間による修正",
                    auto_fixable=True
                ))
        
        return issues
    
    def _check_outliers(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """異常値チェック"""
        issues = []
        
        # 価格データの異常値チェック
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns:
                outlier_indices = self._detect_outliers(data[col])
                if len(outlier_indices) > 0:
                    issues.append(DataQualityIssue(
                        issue_type=DataIssueType.OUTLIER_PRICE,
                        severity="warning",
                        description=f"{col}列に異常値: {len(outlier_indices)}個",
                        affected_rows=outlier_indices,
                        suggested_action="異常値の検証または除外",
                        auto_fixable=False
                    ))
        
        # 出来高の異常値チェック
        if 'Volume' in data.columns:
            # 負の出来高チェック
            negative_volume = data[data['Volume'] < 0].index.tolist()
            if len(negative_volume) > 0:
                issues.append(DataQualityIssue(
                    issue_type=DataIssueType.NEGATIVE_VALUES,
                    severity="critical",
                    description=f"負の出来高: {len(negative_volume)}個",
                    affected_rows=negative_volume,
                    suggested_action="出来高を0または前値で置換",
                    auto_fixable=True
                ))
            
            # ゼロ出来高チェック
            zero_volume = data[data['Volume'] == 0].index.tolist()
            if len(zero_volume) > 0:
                issues.append(DataQualityIssue(
                    issue_type=DataIssueType.ZERO_VOLUME,
                    severity="info",
                    description=f"ゼロ出来高: {len(zero_volume)}個",
                    affected_rows=zero_volume,
                    suggested_action="取引休日等の確認",
                    auto_fixable=False
                ))
        
        return issues
    
    def _detect_outliers(self, series: pd.Series) -> List[int]:
        """統計的異常値検出"""
        if len(series.dropna()) < 5:  # データが少なすぎる場合はスキップ
            return []
        
        # Z-score方法による異常値検出
        z_scores = np.abs((series - series.mean()) / series.std())
        outlier_mask = z_scores > self.outlier_threshold
        
        return series[outlier_mask].index.tolist()
    
    def _check_date_continuity(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """日付連続性チェック"""
        issues = []
        
        # インデックスが日付型の場合のみチェック
        if isinstance(data.index, pd.DatetimeIndex):
            # 営業日ベースでの日付欠損チェック
            date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')  # 営業日
            missing_dates = date_range.difference(data.index)
            
            if len(missing_dates) > 0:
                # 土日を除いた実際の欠損営業日をカウント
                weekdays_missing = [d for d in missing_dates if d.weekday() < 5]
                
                if len(weekdays_missing) > 0:
                    issues.append(DataQualityIssue(
                        issue_type=DataIssueType.DATE_GAPS,
                        severity="warning",
                        description=f"営業日の欠損: {len(weekdays_missing)}日",
                        affected_rows=[],
                        suggested_action="市場休日の確認または追加データ取得",
                        auto_fixable=False
                    ))
        
        return issues
    
    def _check_backtest_compliance(self, data: pd.DataFrame) -> bool:
        """バックテスト基本理念遵守チェック"""
        # 基本的な価格データ存在チェック
        essential_columns = ['Close']  # 最低限Closeは必要
        has_essential_data = all(col in data.columns for col in essential_columns)
        
        # データが空でないことをチェック
        if has_essential_data and 'Close' in data.columns:
            has_sufficient_data = len(data) > 0 and not data['Close'].dropna().empty
        else:
            has_sufficient_data = False
        
        compliance = has_essential_data and has_sufficient_data
        
        if compliance:
            self.logger.info("✅ バックテスト基本理念遵守: 実データでのシグナル生成・取引実行可能")
        else:
            self.logger.warning("❌ バックテスト基本理念違反: 実データ不足でシグナル生成不可")
        
        return compliance
    
    def _auto_fix_issues(self, data: pd.DataFrame, issues: List[DataQualityIssue]) -> Tuple[pd.DataFrame, List[DataQualityIssue]]:
        """自動修正処理"""
        data_fixed = data.copy()
        fixed_issues = []
        
        for issue in issues:
            if issue.auto_fixable:
                if issue.issue_type == DataIssueType.MISSING_DATA:
                    # 前方補完による欠損値修正
                    column_with_missing = None
                    for col in data.columns:
                        if data[col].isna().sum() > 0:
                            column_with_missing = col
                            break
                    
                    if column_with_missing:
                        original_missing = data_fixed[column_with_missing].isna().sum()
                        data_fixed[column_with_missing] = data_fixed[column_with_missing].fillna(method='ffill')
                        # 前方補完できない場合は後方補完
                        data_fixed[column_with_missing] = data_fixed[column_with_missing].fillna(method='bfill')
                        
                        remaining_missing = data_fixed[column_with_missing].isna().sum()
                        if remaining_missing < original_missing:
                            fixed_issues.append(issue)
                            self.logger.info(f"🔧 自動修正: {column_with_missing}の欠損値を補完 ({original_missing} → {remaining_missing})")
                
                elif issue.issue_type == DataIssueType.NEGATIVE_VALUES:
                    # 負の出来高を0で置換
                    if 'Volume' in data_fixed.columns:
                        negative_count = (data_fixed['Volume'] < 0).sum()
                        data_fixed.loc[data_fixed['Volume'] < 0, 'Volume'] = 0
                        if negative_count > 0:
                            fixed_issues.append(issue)
                            self.logger.info(f"🔧 自動修正: 負の出来高を0で置換 ({negative_count}個)")
        
        return data_fixed, fixed_issues
    
    def _calculate_quality_score(self, data: pd.DataFrame, issues: List[DataQualityIssue], fixed_issues: List[DataQualityIssue]) -> float:
        """品質スコア計算（0-100）"""
        total_penalty = 0
        
        for issue in issues:
            if issue not in fixed_issues:  # 修正されていない問題のみペナルティ
                if issue.severity == "critical":
                    total_penalty += 30
                elif issue.severity == "warning":
                    total_penalty += 15
                elif issue.severity == "info":
                    total_penalty += 5
        
        # データ充足度ボーナス
        completeness_bonus = 0
        if len(data) >= self.min_data_points:
            completeness_ratio = min(len(data) / (self.min_data_points * 2), 1.0)
            completeness_bonus = completeness_ratio * 10
        
        score = max(0, 100 - total_penalty + completeness_bonus)
        return min(100, score)
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """品質レベル決定"""
        if score >= 99:
            return QualityLevel.EXCELLENT
        elif score >= 95:
            return QualityLevel.GOOD
        elif score >= 85:
            return QualityLevel.ACCEPTABLE
        elif score >= 70:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNUSABLE
    
    def _generate_recommendations(self, issues: List[DataQualityIssue], quality_level: QualityLevel) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        if quality_level == QualityLevel.UNUSABLE:
            recommendations.append("❌ データ品質が使用不可レベルです。データソースを確認してください")
        elif quality_level == QualityLevel.POOR:
            recommendations.append("⚠️ データ品質が低いです。重要な問題を修正してください")
        
        critical_issues = [i for i in issues if i.severity == "critical"]
        if critical_issues:
            recommendations.append(f"🚨 重要な問題が{len(critical_issues)}個あります。即座に対処してください")
        
        warning_issues = [i for i in issues if i.severity == "warning"]
        if warning_issues:
            recommendations.append(f"⚠️ 警告レベルの問題が{len(warning_issues)}個あります。確認を推奨します")
        
        if quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD]:
            recommendations.append("✅ データ品質は良好です。バックテスト実行に適しています")
        
        return recommendations
    
    def generate_quality_report_text(self, report: DataQualityReport) -> str:
        """品質レポートのテキスト形式生成"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"📊 MarketDataQualityValidator レポート - {report.data_type}")
        lines.append("=" * 80)
        lines.append(f"🕐 検証時刻: {report.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"📈 データ行数: {report.total_rows}")
        lines.append(f"🎯 品質レベル: {report.quality_level.value.upper()}")
        lines.append(f"📊 品質スコア: {report.quality_score:.1f}%")
        lines.append(f"🔍 バックテスト基本理念遵守: {'✅ YES' if report.backtest_compliance else '❌ NO'}")
        lines.append("")
        
        if report.issues:
            lines.append("🚨 検出された問題:")
            for i, issue in enumerate(report.issues, 1):
                severity_icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
                icon = severity_icon.get(issue.severity, "❓")
                lines.append(f"  {i}. {icon} {issue.description}")
                lines.append(f"     影響行数: {len(issue.affected_rows)}")
                lines.append(f"     推奨対応: {issue.suggested_action}")
                if issue.auto_fixable:
                    lines.append("     ✅ 自動修正可能")
                lines.append("")
        else:
            lines.append("✅ 問題は検出されませんでした")
            lines.append("")
        
        if report.fixed_issues:
            lines.append("🔧 自動修正された問題:")
            for i, issue in enumerate(report.fixed_issues, 1):
                lines.append(f"  {i}. ✅ {issue.description}")
            lines.append("")
        
        if report.recommendations:
            lines.append("💡 推奨事項:")
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)

# TODO #14 Phase 4統合: RealMarketDataFetcher統合関数
def validate_fetched_data_quality(data_type: str, 
                                data: pd.DataFrame,
                                auto_fix: bool = True) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Phase 3で取得されたデータの品質検証・修正
    
    Parameters:
        data_type: データタイプ ("index_data", "dow_data", etc.)
        data: 検証対象データ
        auto_fix: 自動修正フラグ
        
    Returns:
        Tuple[検証・修正後データ, 品質レポート]
    """
    validator = MarketDataQualityValidator(auto_fix_enabled=auto_fix)
    
    # 品質検証実行
    quality_report = validator.validate_data_quality(data, data_type)
    
    # 自動修正された場合は修正後データを返す
    if auto_fix and quality_report.fixed_issues:
        # 修正処理を再実行してデータを取得
        fixed_data, _ = validator._auto_fix_issues(data, quality_report.issues)
        return fixed_data, quality_report
    
    return data, quality_report
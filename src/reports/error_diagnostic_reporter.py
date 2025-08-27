"""
エラー診断レポート生成システム
DSSMS Phase 3 Task 3.1: レポート生成システム改良

ログファイル解析、根本原因特定、自動修復提案機能を提供します。
"""

import logging
import pandas as pd
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
import traceback

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger

logger = setup_logger(__name__)

@dataclass
class ErrorDiagnostic:
    """エラー診断結果のデータクラス"""
    error_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    frequency: int
    first_occurrence: datetime
    last_occurrence: datetime
    affected_components: List[str]
    root_cause: str
    suggested_fixes: List[str]
    related_errors: List[str]
    confidence_score: float  # 0.0-1.0

@dataclass
class SystemHealthReport:
    """システム健康度レポートのデータクラス"""
    overall_health_score: float  # 0.0-100.0
    critical_issues_count: int
    high_priority_issues_count: int
    error_trend: str  # IMPROVING, STABLE, DETERIORATING
    system_availability: float  # 0.0-100.0
    performance_score: float  # 0.0-100.0
    recommendations: List[str]

class ErrorDiagnosticReporter:
    """エラー診断レポート生成クラス"""
    
    def __init__(self, log_directory: str = "logs", config_path: Optional[str] = None):
        """
        初期化
        
        Parameters:
            log_directory (str): ログディレクトリパス
            config_path (str): 設定ファイルパス（オプション）
        """
        self.log_directory = Path(log_directory)
        self.config_path = config_path
        self.error_patterns = self._load_error_patterns()
        self.severity_keywords = self._load_severity_keywords()
        self.component_mappings = self._load_component_mappings()
        
    def _load_error_patterns(self) -> Dict[str, str]:
        """エラーパターンマッピングを読み込み"""
        return {
            r"ImportError.*IntelligentSwitchManager": "IMPORT_ERROR_INTELLIGENT_SWITCH",
            r"ModuleNotFoundError.*dssms": "MODULE_NOT_FOUND_DSSMS",
            r"AttributeError.*switch": "ATTRIBUTE_ERROR_SWITCH",
            r"ValueError.*portfolio": "VALUE_ERROR_PORTFOLIO", 
            r"KeyError.*strategy": "KEY_ERROR_STRATEGY",
            r"FileNotFoundError": "FILE_NOT_FOUND",
            r"PermissionError": "PERMISSION_ERROR",
            r"ConnectionError": "CONNECTION_ERROR",
            r"TimeoutError": "TIMEOUT_ERROR",
            r"Memory.*Error": "MEMORY_ERROR",
            r"Switching success rate: 0%": "ZERO_SWITCHING_SUCCESS",
            r"Portfolio value.*0\.01": "PORTFOLIO_VALUE_COLLAPSE",
            r"Total return.*-100%": "TOTAL_RETURN_FAILURE",
            r"Max drawdown.*100%": "MAX_DRAWDOWN_FAILURE"
        }
        
    def _load_severity_keywords(self) -> Dict[str, str]:
        """重要度キーワードマッピングを読み込み"""
        return {
            "CRITICAL": ["CRITICAL", "FATAL", "EMERGENCY", "Portfolio value.*0\.01", "Total return.*-100%"],
            "HIGH": ["ERROR", "EXCEPTION", "FAILED", "TIMEOUT", "CONNECTION"],
            "MEDIUM": ["WARNING", "WARN", "DEPRECATED", "RETRY"],
            "LOW": ["INFO", "DEBUG", "NOTICE"]
        }
        
    def _load_component_mappings(self) -> Dict[str, List[str]]:
        """コンポーネントマッピングを読み込み"""
        return {
            "DSSMS_CORE": ["dssms", "switch_coordinator", "intelligent_switch"],
            "PORTFOLIO": ["portfolio", "position", "balance"],
            "STRATEGY": ["strategy", "signal", "backtest"],
            "DATA": ["data_fetcher", "data_processor", "yfinance"],
            "RISK": ["risk_management", "drawdown", "volatility"],
            "REPORTING": ["report", "excel", "output"],
            "CONFIGURATION": ["config", "parameter", "setting"]
        }
        
    def analyze_logs(self, hours_back: int = 24) -> Tuple[List[ErrorDiagnostic], SystemHealthReport]:
        """
        ログファイルを解析してエラー診断を実行
        
        Parameters:
            hours_back (int): 分析対象時間（時間）
            
        Returns:
            Tuple[List[ErrorDiagnostic], SystemHealthReport]: 診断結果とシステム健康度
        """
        logger.info(f"エラー診断開始: 過去{hours_back}時間のログを分析")
        
        try:
            # ログファイルを収集
            log_entries = self._collect_log_entries(hours_back)
            logger.info(f"収集したログエントリ数: {len(log_entries)}")
            
            # エラーパターンを検出
            error_detections = self._detect_error_patterns(log_entries)
            logger.info(f"検出したエラーパターン数: {len(error_detections)}")
            
            # 診断結果を生成
            diagnostics = self._generate_diagnostics(error_detections)
            
            # システム健康度を評価
            health_report = self._evaluate_system_health(diagnostics, log_entries)
            
            logger.info(f"エラー診断完了: {len(diagnostics)}個の診断結果を生成")
            return diagnostics, health_report
            
        except Exception as e:
            logger.error(f"エラー診断中にエラーが発生: {e}")
            logger.error(traceback.format_exc())
            return [], self._create_emergency_health_report()
            
    def _collect_log_entries(self, hours_back: int) -> List[Dict[str, Any]]:
        """ログエントリを収集"""
        entries = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            # ログファイルを検索
            for log_file in self.log_directory.glob("*.log"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            entry = self._parse_log_line(line, log_file.name, line_num)
                            if entry and entry.get('timestamp', datetime.min) >= cutoff_time:
                                entries.append(entry)
                except Exception as e:
                    logger.warning(f"ログファイル {log_file} の読み込みエラー: {e}")
                    
        except Exception as e:
            logger.error(f"ログ収集エラー: {e}")
            
        return sorted(entries, key=lambda x: x.get('timestamp', datetime.min))
        
    def _parse_log_line(self, line: str, filename: str, line_num: int) -> Optional[Dict[str, Any]]:
        """ログ行を解析"""
        try:
            # タイムスタンプを抽出
            timestamp_patterns = [
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
                r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})',
                r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})'
            ]
            
            timestamp = None
            for pattern in timestamp_patterns:
                match = re.search(pattern, line)
                if match:
                    try:
                        timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                        break
                    except ValueError:
                        try:
                            timestamp = datetime.strptime(match.group(1), '%m/%d/%Y %H:%M:%S')
                            break
                        except ValueError:
                            try:
                                timestamp = datetime.strptime(match.group(1), '%Y-%m-%dT%H:%M:%S')
                                break
                            except ValueError:
                                continue
                                
            # ログレベルを抽出
            level_match = re.search(r'\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b', line, re.IGNORECASE)
            level = level_match.group(1).upper() if level_match else "INFO"
            
            return {
                'timestamp': timestamp or datetime.now(),
                'level': level,
                'message': line.strip(),
                'filename': filename,
                'line_number': line_num
            }
            
        except Exception as e:
            logger.debug(f"ログ行解析エラー (行 {line_num}): {e}")
            return None
            
    def _detect_error_patterns(self, log_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """エラーパターンを検出"""
        detections = []
        
        for entry in log_entries:
            message = entry['message']
            
            # 各エラーパターンをチェック
            for pattern, error_type in self.error_patterns.items():
                if re.search(pattern, message, re.IGNORECASE):
                    detection = {
                        'error_type': error_type,
                        'entry': entry,
                        'pattern': pattern,
                        'severity': self._determine_severity(message),
                        'components': self._identify_components(message)
                    }
                    detections.append(detection)
                    
        return detections
        
    def _determine_severity(self, message: str) -> str:
        """メッセージから重要度を判定"""
        for severity, keywords in self.severity_keywords.items():
            for keyword in keywords:
                if re.search(keyword, message, re.IGNORECASE):
                    return severity
        return "LOW"
        
    def _identify_components(self, message: str) -> List[str]:
        """影響を受けるコンポーネントを特定"""
        components = []
        for component, keywords in self.component_mappings.items():
            for keyword in keywords:
                if re.search(keyword, message, re.IGNORECASE):
                    components.append(component)
        return list(set(components))
        
    def _generate_diagnostics(self, detections: List[Dict[str, Any]]) -> List[ErrorDiagnostic]:
        """検出結果から診断を生成"""
        grouped_errors = defaultdict(list)
        
        # エラータイプでグループ化
        for detection in detections:
            grouped_errors[detection['error_type']].append(detection)
            
        diagnostics = []
        for error_type, error_list in grouped_errors.items():
            # 最初と最後の発生時刻
            timestamps = [d['entry']['timestamp'] for d in error_list]
            first_occurrence = min(timestamps)
            last_occurrence = max(timestamps)
            
            # 影響コンポーネント
            all_components = []
            for error in error_list:
                all_components.extend(error['components'])
            affected_components = list(set(all_components))
            
            # 重要度（最も高い重要度を採用）
            severities = [d['severity'] for d in error_list]
            severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            severity = min(severities, key=lambda x: severity_order.index(x))
            
            # 根本原因と修復提案を生成
            root_cause, suggested_fixes = self._analyze_root_cause(error_type, error_list)
            
            # 関連エラーを特定
            related_errors = self._find_related_errors(error_type, grouped_errors)
            
            # 信頼度スコアを計算
            confidence_score = self._calculate_confidence_score(error_list)
            
            diagnostic = ErrorDiagnostic(
                error_type=error_type,
                severity=severity,
                frequency=len(error_list),
                first_occurrence=first_occurrence,
                last_occurrence=last_occurrence,
                affected_components=affected_components,
                root_cause=root_cause,
                suggested_fixes=suggested_fixes,
                related_errors=related_errors,
                confidence_score=confidence_score
            )
            
            diagnostics.append(diagnostic)
            
        return sorted(diagnostics, key=lambda x: (
            ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].index(x.severity),
            -x.frequency
        ))
        
    def _analyze_root_cause(self, error_type: str, error_list: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """根本原因分析と修復提案"""
        root_cause_map = {
            "IMPORT_ERROR_INTELLIGENT_SWITCH": (
                "IntelligentSwitchManagerモジュールの依存関係不備",
                [
                    "src/dssms/intelligent_switch_manager.pyファイルの存在確認",
                    "MockSwitchManagerの実装確認",
                    "緊急フォールバック機構の実装",
                    "依存関係の再インストール"
                ]
            ),
            "ZERO_SWITCHING_SUCCESS": (
                "戦略切り替えメカニズムの完全な機能停止",
                [
                    "スイッチングロジックの緊急診断実行",
                    "ポートフォリオ計算エンジンの修復",
                    "戦略選択アルゴリズムの検証",
                    "緊急モックシステムの有効化"
                ]
            ),
            "PORTFOLIO_VALUE_COLLAPSE": (
                "ポートフォリオ価値計算の致命的エラー",
                [
                    "ポジション計算ロジックの緊急修復",
                    "資金管理システムの再初期化",
                    "取引履歴の整合性チェック",
                    "緊急バックアップからの復元"
                ]
            ),
            "TOTAL_RETURN_FAILURE": (
                "リターン計算アルゴリズムの完全な故障",
                [
                    "収益率計算式の検証と修正",
                    "基準価格データの整合性確認",
                    "累積リターン計算の再実装",
                    "パフォーマンス指標の再校正"
                ]
            )
        }
        
        if error_type in root_cause_map:
            return root_cause_map[error_type]
        else:
            return f"{error_type}の詳細な原因分析が必要", ["専門的な調査を実施してください"]
            
    def _find_related_errors(self, error_type: str, grouped_errors: Dict[str, List]) -> List[str]:
        """関連エラーを特定"""
        related_patterns = {
            "IMPORT_ERROR_INTELLIGENT_SWITCH": ["MODULE_NOT_FOUND_DSSMS", "ATTRIBUTE_ERROR_SWITCH"],
            "ZERO_SWITCHING_SUCCESS": ["PORTFOLIO_VALUE_COLLAPSE", "TOTAL_RETURN_FAILURE"],
            "PORTFOLIO_VALUE_COLLAPSE": ["VALUE_ERROR_PORTFOLIO", "ZERO_SWITCHING_SUCCESS"],
            "TOTAL_RETURN_FAILURE": ["MAX_DRAWDOWN_FAILURE", "PORTFOLIO_VALUE_COLLAPSE"]
        }
        
        if error_type in related_patterns:
            return [et for et in related_patterns[error_type] if et in grouped_errors]
        return []
        
    def _calculate_confidence_score(self, error_list: List[Dict[str, Any]]) -> float:
        """信頼度スコアを計算"""
        base_score = 0.5
        
        # 頻度による調整
        frequency_bonus = min(0.3, len(error_list) * 0.05)
        
        # 重要度による調整
        high_severity_count = sum(1 for e in error_list if e['severity'] in ['CRITICAL', 'HIGH'])
        severity_bonus = min(0.2, high_severity_count * 0.1)
        
        return min(1.0, base_score + frequency_bonus + severity_bonus)
        
    def _evaluate_system_health(self, diagnostics: List[ErrorDiagnostic], 
                              log_entries: List[Dict[str, Any]]) -> SystemHealthReport:
        """システム健康度を評価"""
        critical_count = sum(1 for d in diagnostics if d.severity == 'CRITICAL')
        high_count = sum(1 for d in diagnostics if d.severity == 'HIGH')
        
        # 基本健康度計算
        base_health = 100.0
        health_penalty = critical_count * 30 + high_count * 10
        overall_health = max(0.0, base_health - health_penalty)
        
        # エラートレンド分析
        if len(log_entries) > 0:
            error_entries = [e for e in log_entries if e['level'] in ['ERROR', 'CRITICAL', 'FATAL']]
            if len(error_entries) > 10:
                trend = "DETERIORATING"
            elif len(error_entries) > 5:
                trend = "STABLE"
            else:
                trend = "IMPROVING"
        else:
            trend = "STABLE"
            
        # 可用性とパフォーマンススコア
        availability = max(0.0, 100.0 - critical_count * 25)
        performance = max(0.0, 100.0 - critical_count * 20 - high_count * 5)
        
        # 推奨事項
        recommendations = []
        if critical_count > 0:
            recommendations.append("緊急: 致命的エラーの即座な対応が必要")
        if high_count > 2:
            recommendations.append("高優先度エラーの系統的な修復を推奨")
        if overall_health < 50:
            recommendations.append("システム全体の包括的な診断と修復が必要")
            
        return SystemHealthReport(
            overall_health_score=overall_health,
            critical_issues_count=critical_count,
            high_priority_issues_count=high_count,
            error_trend=trend,
            system_availability=availability,
            performance_score=performance,
            recommendations=recommendations
        )
        
    def _create_emergency_health_report(self) -> SystemHealthReport:
        """緊急時の健康度レポートを作成"""
        return SystemHealthReport(
            overall_health_score=0.0,
            critical_issues_count=1,
            high_priority_issues_count=0,
            error_trend="CRITICAL",
            system_availability=0.0,
            performance_score=0.0,
            recommendations=["緊急: 診断システム自体にエラーが発生"]
        )
        
    def generate_html_report(self, diagnostics: List[ErrorDiagnostic], 
                           health_report: SystemHealthReport, output_path: str) -> str:
        """HTML形式のエラー診断レポートを生成"""
        html_template = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DSSMS エラー診断レポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .health-score {{ font-size: 24px; font-weight: bold; }}
        .critical {{ color: #dc3545; }}
        .high {{ color: #fd7e14; }}
        .medium {{ color: #ffc107; }}
        .low {{ color: #28a745; }}
        .error-card {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .recommendations {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>DSSMS エラー診断レポート</h1>
        <p>生成日時: {timestamp}</p>
        <div class="health-score {health_class}">
            システム健康度: {health_score:.1f}/100.0
        </div>
    </div>
    
    <h2>システム概要</h2>
    <table>
        <tr><th>項目</th><th>値</th></tr>
        <tr><td>致命的問題数</td><td class="critical">{critical_count}</td></tr>
        <tr><td>高優先度問題数</td><td class="high">{high_count}</td></tr>
        <tr><td>エラー傾向</td><td>{error_trend}</td></tr>
        <tr><td>システム可用性</td><td>{availability:.1f}%</td></tr>
        <tr><td>パフォーマンススコア</td><td>{performance:.1f}%</td></tr>
    </table>
    
    <h2>検出されたエラー ({error_count}件)</h2>
    {error_details}
    
    <div class="recommendations">
        <h2>推奨事項</h2>
        <ul>
        {recommendations}
        </ul>
    </div>
</body>
</html>
        """
        
        # HTMLコンテンツを構築
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        health_class = "critical" if health_report.overall_health_score < 30 else "high" if health_report.overall_health_score < 60 else "medium" if health_report.overall_health_score < 80 else "low"
        
        error_details = ""
        for diag in diagnostics:
            severity_class = diag.severity.lower()
            error_details += f"""
            <div class="error-card">
                <h3 class="{severity_class}">{diag.error_type} ({diag.severity})</h3>
                <p><strong>発生頻度:</strong> {diag.frequency}回</p>
                <p><strong>初回発生:</strong> {diag.first_occurrence.strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>最終発生:</strong> {diag.last_occurrence.strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>影響コンポーネント:</strong> {', '.join(diag.affected_components)}</p>
                <p><strong>根本原因:</strong> {diag.root_cause}</p>
                <p><strong>修復提案:</strong></p>
                <ul>
                    {''.join(f'<li>{fix}</li>' for fix in diag.suggested_fixes)}
                </ul>
                <p><strong>信頼度:</strong> {diag.confidence_score:.2f}</p>
            </div>
            """
            
        recommendations_list = "".join(f'<li>{rec}</li>' for rec in health_report.recommendations)
        
        html_content = html_template.format(
            timestamp=timestamp,
            health_score=health_report.overall_health_score,
            health_class=health_class,
            critical_count=health_report.critical_issues_count,
            high_count=health_report.high_priority_issues_count,
            error_trend=health_report.error_trend,
            availability=health_report.system_availability,
            performance=health_report.performance_score,
            error_count=len(diagnostics),
            error_details=error_details,
            recommendations=recommendations_list
        )
        
        # ファイルに出力
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"HTMLエラー診断レポートを出力: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"HTMLレポート出力エラー: {e}")
            return ""

if __name__ == "__main__":
    # テスト実行
    reporter = ErrorDiagnosticReporter()
    diagnostics, health_report = reporter.analyze_logs(hours_back=24)
    
    print(f"エラー診断結果: {len(diagnostics)}件")
    print(f"システム健康度: {health_report.overall_health_score:.1f}/100.0")
    
    if diagnostics:
        print("\n=== 主要なエラー ===")
        for diag in diagnostics[:5]:  # 上位5件
            print(f"- {diag.error_type} ({diag.severity}): {diag.frequency}回")
            
    # HTMLレポート生成
    html_path = "output/error_diagnostic_report.html"
    reporter.generate_html_report(diagnostics, health_report, html_path)

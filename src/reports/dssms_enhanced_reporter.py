"""
DSSMS 強化レポート生成システム
DSSMS Phase 3 Task 3.1: レポート生成システム改良

DSSMS固有の分析機能、戦略パフォーマンス比較、
ポートフォリオ診断、切り替え成功率分析を提供します。
"""

import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys
import traceback
from dataclasses import dataclass, asdict
import re

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger
from src.reports.error_diagnostic_reporter import ErrorDiagnostic, SystemHealthReport
from src.reports.report_integration_manager import ReportGenerationRequest, ReportGenerationResult

logger = setup_logger(__name__)

@dataclass
class DSSMSPerformanceMetrics:
    """DSSMS パフォーマンス指標"""
    switching_success_rate: float
    portfolio_value: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    strategy_distribution: Dict[str, float]
    switching_frequency: int
    last_switch_timestamp: Optional[datetime]
    active_strategies: List[str]
    failed_switches: int

@dataclass
class StrategyAnalysis:
    """戦略分析結果"""
    strategy_name: str
    active_periods: int
    total_return: float
    win_rate: float
    avg_holding_period: float
    risk_adjusted_return: float
    performance_ranking: int
    issues_detected: List[str]
    recommendations: List[str]

@dataclass
class PortfolioDiagnostic:
    """ポートフォリオ診断結果"""
    current_value: float
    initial_value: float
    value_change_percent: float
    position_count: int
    cash_ratio: float
    risk_exposure: float
    diversification_score: float
    anomalies_detected: List[str]
    health_status: str

class DSSMSEnhancedReporter:
    """DSSMS強化レポート生成クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Parameters:
            config_path (str): 設定ファイルパス（オプション）
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # ログディレクトリの設定
        self.log_directory = Path("logs")
        self.output_directory = Path("output")
        self.output_directory.mkdir(exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        default_config = {
            "analysis_window_hours": 24,
            "min_switching_threshold": 5.0,  # 最小切り替え成功率（%）
            "portfolio_health_thresholds": {
                "critical": 10.0,
                "warning": 50.0,
                "good": 80.0
            },
            "strategy_performance_weights": {
                "return": 0.4,
                "risk_adjusted": 0.3,
                "consistency": 0.3
            },
            "dssms_components": [
                "switch_coordinator",
                "intelligent_switch_manager", 
                "portfolio_calculator",
                "strategy_selector",
                "risk_manager"
            ]
        }
        
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"DSSMS設定ファイルを読み込み: {self.config_path}")
            except Exception as e:
                logger.warning(f"DSSMS設定ファイル読み込みエラー: {e}")
                
        return default_config
        
    def analyze_dssms_performance(self, hours_back: int = 24) -> DSSMSPerformanceMetrics:
        """
        DSSMSパフォーマンスを分析
        
        Parameters:
            hours_back (int): 分析対象時間（時間）
            
        Returns:
            DSSMSPerformanceMetrics: パフォーマンス指標
        """
        logger.info(f"DSSMSパフォーマンス分析開始: 過去{hours_back}時間")
        
        try:
            # ログエントリを収集
            log_entries = self._collect_dssms_logs(hours_back)
            
            # 切り替え成功率を分析
            switching_metrics = self._analyze_switching_performance(log_entries)
            
            # ポートフォリオ指標を抽出
            portfolio_metrics = self._extract_portfolio_metrics(log_entries)
            
            # 戦略分布を分析
            strategy_distribution = self._analyze_strategy_distribution(log_entries)
            
            # アクティブ戦略を特定
            active_strategies = self._identify_active_strategies(log_entries)
            
            return DSSMSPerformanceMetrics(
                switching_success_rate=switching_metrics.get('success_rate', 0.0),
                portfolio_value=portfolio_metrics.get('current_value', 0.01),
                total_return=portfolio_metrics.get('total_return', -100.0),
                max_drawdown=portfolio_metrics.get('max_drawdown', 100.0),
                sharpe_ratio=portfolio_metrics.get('sharpe_ratio', -999.0),
                strategy_distribution=strategy_distribution,
                switching_frequency=switching_metrics.get('frequency', 0),
                last_switch_timestamp=switching_metrics.get('last_switch'),
                active_strategies=active_strategies,
                failed_switches=switching_metrics.get('failed_switches', 0)
            )
            
        except Exception as e:
            logger.error(f"DSSMSパフォーマンス分析エラー: {e}")
            logger.error(traceback.format_exc())
            return self._create_emergency_performance_metrics()
            
    def _collect_dssms_logs(self, hours_back: int) -> List[Dict[str, Any]]:
        """DSSMS関連ログを収集"""
        all_entries = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            for log_file in self.log_directory.glob("*.log"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            # DSSMS関連のログのみフィルタ
                            if any(component in line.lower() for component in ['dssms', 'switch', 'portfolio', 'strategy']):
                                entry = self._parse_log_line(line, log_file.name, line_num)
                                if entry and entry.get('timestamp', datetime.min) >= cutoff_time:
                                    all_entries.append(entry)
                except Exception as e:
                    logger.warning(f"DSSMSログファイル読み込みエラー {log_file}: {e}")
                    
        except Exception as e:
            logger.error(f"DSSMSログ収集エラー: {e}")
            
        return sorted(all_entries, key=lambda x: x.get('timestamp', datetime.min))
        
    def _parse_log_line(self, line: str, filename: str, line_num: int) -> Optional[Dict[str, Any]]:
        """ログ行を解析（DSSMS固有の情報を抽出）"""
        try:
            # 基本的なタイムスタンプ・レベル抽出
            timestamp_patterns = [
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
                r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})'
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
                            continue
                            
            level_match = re.search(r'\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b', line, re.IGNORECASE)
            level = level_match.group(1).upper() if level_match else "INFO"
            
            # DSSMS固有の情報を抽出
            dssms_info = self._extract_dssms_info(line)
            
            entry = {
                'timestamp': timestamp or datetime.now(),
                'level': level,
                'message': line.strip(),
                'filename': filename,
                'line_number': line_num
            }
            
            # DSSMS情報をマージ
            entry.update(dssms_info)
            
            return entry
            
        except Exception as e:
            logger.debug(f"DSSMSログ行解析エラー (行 {line_num}): {e}")
            return None
            
    def _extract_dssms_info(self, line: str) -> Dict[str, Any]:
        """ログ行からDSSMS固有の情報を抽出"""
        info = {}
        
        # 切り替え成功率
        switch_match = re.search(r'switching success rate:\s*([0-9.]+)%', line, re.IGNORECASE)
        if switch_match:
            info['switching_success_rate'] = float(switch_match.group(1))
            
        # ポートフォリオ価値
        portfolio_match = re.search(r'portfolio value[:\s]*([0-9.]+)', line, re.IGNORECASE)
        if portfolio_match:
            info['portfolio_value'] = float(portfolio_match.group(1))
            
        # トータルリターン
        return_match = re.search(r'total return[:\s]*(-?[0-9.]+)%?', line, re.IGNORECASE)
        if return_match:
            info['total_return'] = float(return_match.group(1))
            
        # 最大ドローダウン
        drawdown_match = re.search(r'max drawdown[:\s]*([0-9.]+)%?', line, re.IGNORECASE)
        if drawdown_match:
            info['max_drawdown'] = float(drawdown_match.group(1))
            
        # 戦略名
        strategy_match = re.search(r'strategy[:\s]*([A-Za-z_]+)', line, re.IGNORECASE)
        if strategy_match:
            info['strategy_name'] = strategy_match.group(1)
            
        # 切り替えイベント
        if any(keyword in line.lower() for keyword in ['switch', 'switching', 'switched']):
            info['is_switch_event'] = True
            
        # エラーイベント
        if any(keyword in line.upper() for keyword in ['ERROR', 'FAILED', 'EXCEPTION']):
            info['is_error_event'] = True
            
        return info
        
    def _analyze_switching_performance(self, log_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """切り替えパフォーマンスを分析"""
        switch_events = [e for e in log_entries if e.get('is_switch_event', False)]
        error_events = [e for e in log_entries if e.get('is_error_event', False)]
        
        # 成功率を計算
        success_rates = [e.get('switching_success_rate') for e in log_entries if e.get('switching_success_rate') is not None]
        current_success_rate = success_rates[-1] if success_rates else 0.0
        
        # 最後の切り替え時刻
        last_switch = switch_events[-1]['timestamp'] if switch_events else None
        
        # 失敗した切り替え数を推定
        failed_switches = len([e for e in error_events if 'switch' in e.get('message', '').lower()])
        
        return {
            'success_rate': current_success_rate,
            'frequency': len(switch_events),
            'last_switch': last_switch,
            'failed_switches': failed_switches,
            'error_rate': len(error_events) / max(len(log_entries), 1) * 100.0
        }
        
    def _extract_portfolio_metrics(self, log_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ポートフォリオ指標を抽出"""
        portfolio_values = [e.get('portfolio_value') for e in log_entries if e.get('portfolio_value') is not None]
        total_returns = [e.get('total_return') for e in log_entries if e.get('total_return') is not None]
        max_drawdowns = [e.get('max_drawdown') for e in log_entries if e.get('max_drawdown') is not None]
        
        return {
            'current_value': portfolio_values[-1] if portfolio_values else 0.01,
            'total_return': total_returns[-1] if total_returns else -100.0,
            'max_drawdown': max_drawdowns[-1] if max_drawdowns else 100.0,
            'sharpe_ratio': self._calculate_sharpe_ratio(total_returns),
            'value_history': portfolio_values
        }
        
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """シャープレシオを計算"""
        if len(returns) < 2:
            return -999.0
            
        try:
            returns_array = np.array(returns)
            if np.std(returns_array) == 0:
                return 0.0
            return np.mean(returns_array) / np.std(returns_array)
        except Exception:
            return -999.0
            
    def _analyze_strategy_distribution(self, log_entries: List[Dict[str, Any]]) -> Dict[str, float]:
        """戦略分布を分析"""
        strategy_events = [e for e in log_entries if e.get('strategy_name')]
        strategy_counts = {}
        
        for event in strategy_events:
            strategy = event['strategy_name']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
        total_events = sum(strategy_counts.values())
        if total_events == 0:
            return {}
            
        return {strategy: count / total_events * 100.0 for strategy, count in strategy_counts.items()}
        
    def _identify_active_strategies(self, log_entries: List[Dict[str, Any]]) -> List[str]:
        """アクティブ戦略を特定"""
        recent_entries = log_entries[-50:]  # 最近50エントリ
        strategies = set()
        
        for entry in recent_entries:
            if entry.get('strategy_name'):
                strategies.add(entry['strategy_name'])
                
        return list(strategies)
        
    def _create_emergency_performance_metrics(self) -> DSSMSPerformanceMetrics:
        """緊急時のパフォーマンス指標を作成"""
        return DSSMSPerformanceMetrics(
            switching_success_rate=0.0,
            portfolio_value=0.01,
            total_return=-100.0,
            max_drawdown=100.0,
            sharpe_ratio=-999.0,
            strategy_distribution={},
            switching_frequency=0,
            last_switch_timestamp=None,
            active_strategies=[],
            failed_switches=999
        )
        
    def analyze_strategy_performance(self, hours_back: int = 24) -> List[StrategyAnalysis]:
        """
        戦略別パフォーマンスを分析
        
        Parameters:
            hours_back (int): 分析対象時間（時間）
            
        Returns:
            List[StrategyAnalysis]: 戦略分析結果
        """
        logger.info("戦略別パフォーマンス分析開始")
        
        try:
            log_entries = self._collect_dssms_logs(hours_back)
            
            # 戦略別にログをグループ化
            strategy_groups = {}
            for entry in log_entries:
                strategy = entry.get('strategy_name')
                if strategy:
                    if strategy not in strategy_groups:
                        strategy_groups[strategy] = []
                    strategy_groups[strategy].append(entry)
                    
            analyses = []
            for strategy_name, strategy_logs in strategy_groups.items():
                analysis = self._analyze_single_strategy(strategy_name, strategy_logs)
                analyses.append(analysis)
                
            # パフォーマンスランキングを設定
            analyses.sort(key=lambda x: x.risk_adjusted_return, reverse=True)
            for rank, analysis in enumerate(analyses, 1):
                analysis.performance_ranking = rank
                
            return analyses
            
        except Exception as e:
            logger.error(f"戦略パフォーマンス分析エラー: {e}")
            return []
            
    def _analyze_single_strategy(self, strategy_name: str, strategy_logs: List[Dict[str, Any]]) -> StrategyAnalysis:
        """単一戦略を分析"""
        # 基本統計
        active_periods = len(strategy_logs)
        
        # リターン計算
        returns = [log.get('total_return', 0) for log in strategy_logs if log.get('total_return') is not None]
        total_return = returns[-1] if returns else 0.0
        
        # 勝率計算（リターンが正の割合）
        positive_returns = [r for r in returns if r > 0]
        win_rate = len(positive_returns) / max(len(returns), 1) * 100.0
        
        # 平均保有期間（ログエントリ間隔から推定）
        if len(strategy_logs) > 1:
            time_diffs = []
            for i in range(1, len(strategy_logs)):
                diff = (strategy_logs[i]['timestamp'] - strategy_logs[i-1]['timestamp']).total_seconds() / 3600.0
                time_diffs.append(diff)
            avg_holding_period = np.mean(time_diffs) if time_diffs else 0.0
        else:
            avg_holding_period = 0.0
            
        # リスク調整リターン
        risk_adjusted_return = self._calculate_risk_adjusted_return(returns)
        
        # 問題検出
        issues = self._detect_strategy_issues(strategy_name, strategy_logs)
        
        # 推奨事項
        recommendations = self._generate_strategy_recommendations(strategy_name, issues, total_return, win_rate)
        
        return StrategyAnalysis(
            strategy_name=strategy_name,
            active_periods=active_periods,
            total_return=total_return,
            win_rate=win_rate,
            avg_holding_period=avg_holding_period,
            risk_adjusted_return=risk_adjusted_return,
            performance_ranking=0,  # 後で設定
            issues_detected=issues,
            recommendations=recommendations
        )
        
    def _calculate_risk_adjusted_return(self, returns: List[float]) -> float:
        """リスク調整リターンを計算"""
        if len(returns) < 2:
            return 0.0
            
        try:
            returns_array = np.array(returns)
            volatility = np.std(returns_array)
            if volatility == 0:
                return np.mean(returns_array)
            return np.mean(returns_array) / volatility
        except Exception:
            return 0.0
            
    def _detect_strategy_issues(self, strategy_name: str, strategy_logs: List[Dict[str, Any]]) -> List[str]:
        """戦略の問題を検出"""
        issues = []
        
        # エラーイベントの確認
        error_count = len([log for log in strategy_logs if log.get('is_error_event', False)])
        if error_count > 0:
            issues.append(f"エラーイベント {error_count} 件検出")
            
        # パフォーマンス問題
        returns = [log.get('total_return', 0) for log in strategy_logs if log.get('total_return') is not None]
        if returns and returns[-1] < -50.0:
            issues.append("重大な損失が発生")
            
        # 活動レベル
        if len(strategy_logs) < 5:
            issues.append("活動レベルが低い")
            
        return issues
        
    def _generate_strategy_recommendations(self, strategy_name: str, issues: List[str], 
                                         total_return: float, win_rate: float) -> List[str]:
        """戦略推奨事項を生成"""
        recommendations = []
        
        if total_return < -20.0:
            recommendations.append("パラメータの再最適化を検討")
            
        if win_rate < 30.0:
            recommendations.append("エントリー条件の見直しが必要")
            
        if "エラーイベント" in str(issues):
            recommendations.append("エラー処理の改善が必要")
            
        if "活動レベルが低い" in issues:
            recommendations.append("戦略の有効性を検証")
            
        if not recommendations:
            recommendations.append("現在のパフォーマンスを維持")
            
        return recommendations
        
    def diagnose_portfolio_health(self, hours_back: int = 24) -> PortfolioDiagnostic:
        """
        ポートフォリオ健康度診断
        
        Parameters:
            hours_back (int): 分析対象時間（時間）
            
        Returns:
            PortfolioDiagnostic: 診断結果
        """
        logger.info("ポートフォリオ健康度診断開始")
        
        try:
            log_entries = self._collect_dssms_logs(hours_back)
            
            # ポートフォリオ指標を抽出
            portfolio_metrics = self._extract_portfolio_metrics(log_entries)
            
            current_value = portfolio_metrics.get('current_value', 0.01)
            initial_value = 1000000.0  # 仮定：初期資金100万円
            
            value_change_percent = (current_value - initial_value) / initial_value * 100.0
            
            # 健康度判定
            health_status = self._determine_portfolio_health(current_value, value_change_percent)
            
            # 異常検出
            anomalies = self._detect_portfolio_anomalies(portfolio_metrics, log_entries)
            
            return PortfolioDiagnostic(
                current_value=current_value,
                initial_value=initial_value,
                value_change_percent=value_change_percent,
                position_count=self._estimate_position_count(log_entries),
                cash_ratio=self._estimate_cash_ratio(current_value),
                risk_exposure=self._calculate_risk_exposure(portfolio_metrics),
                diversification_score=self._calculate_diversification_score(log_entries),
                anomalies_detected=anomalies,
                health_status=health_status
            )
            
        except Exception as e:
            logger.error(f"ポートフォリオ診断エラー: {e}")
            return self._create_emergency_portfolio_diagnostic()
            
    def _determine_portfolio_health(self, current_value: float, value_change_percent: float) -> str:
        """ポートフォリオ健康度を判定"""
        thresholds = self.config.get('portfolio_health_thresholds', {})
        
        if current_value < 10000 or value_change_percent < -90.0:
            return "CRITICAL"
        elif value_change_percent < -50.0:
            return "WARNING"
        elif value_change_percent > thresholds.get('good', 10.0):
            return "EXCELLENT"
        elif value_change_percent > 0:
            return "GOOD"
        else:
            return "FAIR"
            
    def _detect_portfolio_anomalies(self, portfolio_metrics: Dict[str, Any], 
                                   log_entries: List[Dict[str, Any]]) -> List[str]:
        """ポートフォリオ異常を検出"""
        anomalies = []
        
        # 異常な価値下落
        if portfolio_metrics.get('current_value', 0) < 100:
            anomalies.append("ポートフォリオ価値が異常に低い")
            
        # 最大ドローダウン異常
        if portfolio_metrics.get('max_drawdown', 0) > 95.0:
            anomalies.append("最大ドローダウンが危険レベル")
            
        # 切り替え失敗の頻発
        failed_switches = len([e for e in log_entries if e.get('is_error_event') and 'switch' in e.get('message', '')])
        if failed_switches > 10:
            anomalies.append("戦略切り替えの失敗が頻発")
            
        return anomalies
        
    def _estimate_position_count(self, log_entries: List[Dict[str, Any]]) -> int:
        """ポジション数を推定"""
        # ログから戦略の多様性を推定
        unique_strategies = set(e.get('strategy_name') for e in log_entries if e.get('strategy_name'))
        return len(unique_strategies)
        
    def _estimate_cash_ratio(self, current_value: float) -> float:
        """キャッシュ比率を推定"""
        # 現在の価値から推定（簡易計算）
        if current_value < 1000:
            return 100.0  # 全てキャッシュ状態
        else:
            return 10.0  # 通常運用時の推定値
            
    def _calculate_risk_exposure(self, portfolio_metrics: Dict[str, Any]) -> float:
        """リスクエクスポージャーを計算"""
        max_drawdown = portfolio_metrics.get('max_drawdown', 0)
        return min(100.0, max_drawdown)  # ドローダウンをリスク指標として使用
        
    def _calculate_diversification_score(self, log_entries: List[Dict[str, Any]]) -> float:
        """多様化スコアを計算"""
        strategy_distribution = self._analyze_strategy_distribution(log_entries)
        if not strategy_distribution:
            return 0.0
            
        # 戦略数と分布の均等性から多様化スコアを計算
        strategy_count = len(strategy_distribution)
        if strategy_count == 1:
            return 20.0
        elif strategy_count == 2:
            return 50.0
        elif strategy_count >= 3:
            return 80.0
        else:
            return 0.0
            
    def _create_emergency_portfolio_diagnostic(self) -> PortfolioDiagnostic:
        """緊急時のポートフォリオ診断を作成"""
        return PortfolioDiagnostic(
            current_value=0.01,
            initial_value=1000000.0,
            value_change_percent=-99.999,
            position_count=0,
            cash_ratio=100.0,
            risk_exposure=100.0,
            diversification_score=0.0,
            anomalies_detected=["システム全体に重大な問題が発生"],
            health_status="CRITICAL"
        )
        
    def generate_dssms_html_report(self, output_path: str, hours_back: int = 24) -> str:
        """
        DSSMS専用HTMLレポートを生成
        
        Parameters:
            output_path (str): 出力パス
            hours_back (int): 分析対象時間（時間）
            
        Returns:
            str: 生成されたファイルパス
        """
        logger.info(f"DSSMS専用HTMLレポート生成開始: {output_path}")
        
        try:
            # 各種分析を実行
            performance_metrics = self.analyze_dssms_performance(hours_back)
            strategy_analyses = self.analyze_strategy_performance(hours_back)
            portfolio_diagnostic = self.diagnose_portfolio_health(hours_back)
            
            # HTMLコンテンツを構築
            html_content = self._build_dssms_html_content(
                performance_metrics, strategy_analyses, portfolio_diagnostic
            )
            
            # ファイルに出力
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"DSSMS HTMLレポートを保存: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"DSSMS HTMLレポート生成エラー: {e}")
            logger.error(traceback.format_exc())
            return ""
            
    def _build_dssms_html_content(self, performance_metrics: DSSMSPerformanceMetrics,
                                strategy_analyses: List[StrategyAnalysis],
                                portfolio_diagnostic: PortfolioDiagnostic) -> str:
        """DSSMSのHTMLコンテンツを構築"""
        
        html_template = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DSSMS 詳細分析レポート</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }}
        .header {{ background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); color: white; padding: 40px; text-align: center; }}
        .critical {{ background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }}
        .warning {{ background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%); }}
        .good {{ background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%); }}
        
        .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; padding: 30px; }}
        .metric-card {{ background: #f8f9fa; border-radius: 10px; padding: 25px; text-align: center; transition: transform 0.2s; }}
        .metric-card:hover {{ transform: translateY(-5px); box-shadow: 0 5px 20px rgba(0,0,0,0.1); }}
        
        .big-number {{ font-size: 48px; font-weight: bold; margin: 15px 0; }}
        .section {{ margin: 30px; padding: 25px; background: #f8f9fa; border-radius: 10px; }}
        
        .strategy-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }}
        .strategy-card {{ background: white; border-radius: 8px; padding: 20px; border-left: 5px solid #007bff; }}
        
        .portfolio-status {{ padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .status-critical {{ background: #ffebee; border-left: 5px solid #f44336; }}
        .status-warning {{ background: #fff3e0; border-left: 5px solid #ff9800; }}
        .status-good {{ background: #e8f5e8; border-left: 5px solid #4caf50; }}
        
        .anomaly-list {{ background: #ffcdd2; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .recommendation-list {{ background: #c8e6c9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; font-weight: bold; }}
        
        .timestamp {{ color: #666; font-size: 0.9em; }}
        .highlight {{ background: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header {portfolio_health_class}">
            <h1>🔄 DSSMS 詳細分析レポート</h1>
            <p class="timestamp">生成日時: {timestamp}</p>
            <div class="big-number">ポートフォリオ健康度: {portfolio_health}</div>
        </div>

        <!-- パフォーマンス ダッシュボード -->
        <div class="dashboard">
            <div class="metric-card">
                <h3>切り替え成功率</h3>
                <div class="big-number {switching_class}">{switching_success_rate:.1f}%</div>
                <p>失敗回数: {failed_switches}</p>
            </div>
            <div class="metric-card">
                <h3>ポートフォリオ価値</h3>
                <div class="big-number {portfolio_class}">¥{portfolio_value:,.0f}</div>
                <p>変化率: {value_change:.2f}%</p>
            </div>
            <div class="metric-card">
                <h3>総リターン</h3>
                <div class="big-number {return_class}">{total_return:.2f}%</div>
                <p>シャープ比: {sharpe_ratio:.2f}</p>
            </div>
            <div class="metric-card">
                <h3>最大ドローダウン</h3>
                <div class="big-number {drawdown_class}">{max_drawdown:.1f}%</div>
                <p>リスクエクスポージャー: {risk_exposure:.1f}%</p>
            </div>
        </div>

        <!-- 戦略分析セクション -->
        <div class="section">
            <h2>[CHART] 戦略別パフォーマンス分析</h2>
            {strategy_section}
        </div>

        <!-- ポートフォリオ診断セクション -->
        <div class="section">
            <h2>💼 ポートフォリオ詳細診断</h2>
            {portfolio_section}
        </div>

        <!-- システム状態セクション -->
        <div class="section">
            <h2>⚙️ システム状態</h2>
            {system_section}
        </div>
    </div>
</body>
</html>
        """
        
        # データの準備
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        portfolio_health_class = self._get_portfolio_health_class(portfolio_diagnostic.health_status)
        
        # クラス判定
        switching_class = self._get_performance_class(performance_metrics.switching_success_rate, [10, 30, 60])
        portfolio_class = self._get_value_class(performance_metrics.portfolio_value)
        return_class = self._get_performance_class(performance_metrics.total_return, [-50, -10, 5])
        drawdown_class = self._get_inverse_performance_class(performance_metrics.max_drawdown, [20, 50, 80])
        
        # セクション構築
        strategy_section = self._build_strategy_section_html(strategy_analyses)
        portfolio_section = self._build_portfolio_section_html(portfolio_diagnostic)
        system_section = self._build_system_section_html(performance_metrics)
        
        return html_template.format(
            timestamp=timestamp,
            portfolio_health_class=portfolio_health_class,
            portfolio_health=portfolio_diagnostic.health_status,
            switching_success_rate=performance_metrics.switching_success_rate,
            switching_class=switching_class,
            failed_switches=performance_metrics.failed_switches,
            portfolio_value=performance_metrics.portfolio_value,
            portfolio_class=portfolio_class,
            value_change=portfolio_diagnostic.value_change_percent,
            total_return=performance_metrics.total_return,
            return_class=return_class,
            sharpe_ratio=performance_metrics.sharpe_ratio,
            max_drawdown=performance_metrics.max_drawdown,
            drawdown_class=drawdown_class,
            risk_exposure=portfolio_diagnostic.risk_exposure,
            strategy_section=strategy_section,
            portfolio_section=portfolio_section,
            system_section=system_section
        )
        
    def _get_portfolio_health_class(self, health_status: str) -> str:
        """ポートフォリオ健康度からクラスを取得"""
        mapping = {
            "CRITICAL": "critical",
            "WARNING": "warning",
            "FAIR": "warning",
            "GOOD": "good",
            "EXCELLENT": "good"
        }
        return mapping.get(health_status, "warning")
        
    def _get_performance_class(self, value: float, thresholds: List[float]) -> str:
        """パフォーマンス値からクラスを取得"""
        if value >= thresholds[2]:
            return "low"  # 良い（緑）
        elif value >= thresholds[1]:
            return "medium"  # 普通（黄）
        elif value >= thresholds[0]:
            return "high"  # 悪い（オレンジ）
        else:
            return "critical"  # 深刻（赤）
            
    def _get_inverse_performance_class(self, value: float, thresholds: List[float]) -> str:
        """逆方向パフォーマンス値からクラスを取得（ドローダウンなど、低い方が良い指標）"""
        if value <= thresholds[0]:
            return "low"  # 良い
        elif value <= thresholds[1]:
            return "medium"  # 普通
        elif value <= thresholds[2]:
            return "high"  # 悪い
        else:
            return "critical"  # 深刻
            
    def _get_value_class(self, value: float) -> str:
        """ポートフォリオ価値からクラスを取得"""
        if value >= 500000:
            return "low"  # 良い
        elif value >= 100000:
            return "medium"  # 普通
        elif value >= 10000:
            return "high"  # 悪い
        else:
            return "critical"  # 深刻
            
    def _build_strategy_section_html(self, strategy_analyses: List[StrategyAnalysis]) -> str:
        """戦略セクションのHTMLを構築"""
        if not strategy_analyses:
            return "<p>戦略データが利用できません。</p>"
            
        cards = ""
        for strategy in strategy_analyses:
            performance_class = self._get_performance_class(strategy.total_return, [-20, 0, 10])
            issues_html = "<br>".join(strategy.issues_detected) if strategy.issues_detected else "問題なし"
            recommendations_html = "<br>".join(strategy.recommendations)
            
            cards += f"""
            <div class="strategy-card">
                <h4>[TARGET] {strategy.strategy_name} (ランク #{strategy.performance_ranking})</h4>
                <div class="big-number {performance_class}">{strategy.total_return:.2f}%</div>
                <table>
                    <tr><td>勝率</td><td>{strategy.win_rate:.1f}%</td></tr>
                    <tr><td>活動期間</td><td>{strategy.active_periods}</td></tr>
                    <tr><td>リスク調整リターン</td><td>{strategy.risk_adjusted_return:.2f}</td></tr>
                    <tr><td>平均保有期間</td><td>{strategy.avg_holding_period:.2f}時間</td></tr>
                </table>
                <div class="anomaly-list">
                    <strong>検出された問題:</strong><br>{issues_html}
                </div>
                <div class="recommendation-list">
                    <strong>推奨事項:</strong><br>{recommendations_html}
                </div>
            </div>
            """
            
        return f'<div class="strategy-grid">{cards}</div>'
        
    def _build_portfolio_section_html(self, portfolio_diagnostic: PortfolioDiagnostic) -> str:
        """ポートフォリオセクションのHTMLを構築"""
        status_class = f"status-{self._get_portfolio_health_class(portfolio_diagnostic.health_status)}"
        
        anomalies_html = "<br>".join(portfolio_diagnostic.anomalies_detected) if portfolio_diagnostic.anomalies_detected else "異常は検出されませんでした"
        
        return f"""
        <div class="portfolio-status {status_class}">
            <h3>[UP] 現在のポートフォリオ状態: {portfolio_diagnostic.health_status}</h3>
            
            <table>
                <tr><td>現在価値</td><td>¥{portfolio_diagnostic.current_value:,.0f}</td></tr>
                <tr><td>初期価値</td><td>¥{portfolio_diagnostic.initial_value:,.0f}</td></tr>
                <tr><td>価値変化率</td><td>{portfolio_diagnostic.value_change_percent:.2f}%</td></tr>
                <tr><td>ポジション数</td><td>{portfolio_diagnostic.position_count}</td></tr>
                <tr><td>キャッシュ比率</td><td>{portfolio_diagnostic.cash_ratio:.1f}%</td></tr>
                <tr><td>リスクエクスポージャー</td><td>{portfolio_diagnostic.risk_exposure:.1f}%</td></tr>
                <tr><td>多様化スコア</td><td>{portfolio_diagnostic.diversification_score:.1f}/100</td></tr>
            </table>
            
            <div class="highlight">
                <strong>[WARNING] 検出された異常:</strong><br>{anomalies_html}
            </div>
        </div>
        """
        
    def _build_system_section_html(self, performance_metrics: DSSMSPerformanceMetrics) -> str:
        """システムセクションのHTMLを構築"""
        last_switch = performance_metrics.last_switch_timestamp.strftime("%Y-%m-%d %H:%M:%S") if performance_metrics.last_switch_timestamp else "不明"
        
        active_strategies_html = ", ".join(performance_metrics.active_strategies) if performance_metrics.active_strategies else "なし"
        
        strategy_dist_html = ""
        for strategy, percentage in performance_metrics.strategy_distribution.items():
            strategy_dist_html += f"<tr><td>{strategy}</td><td>{percentage:.1f}%</td></tr>"
            
        return f"""
        <table>
            <tr><td>切り替え頻度</td><td>{performance_metrics.switching_frequency}回</td></tr>
            <tr><td>最後の切り替え</td><td>{last_switch}</td></tr>
            <tr><td>アクティブ戦略</td><td>{active_strategies_html}</td></tr>
            <tr><td>失敗した切り替え</td><td>{performance_metrics.failed_switches}回</td></tr>
        </table>
        
        <h4>[CHART] 戦略分布</h4>
        <table>
            <thead><tr><th>戦略</th><th>使用率</th></tr></thead>
            <tbody>{strategy_dist_html}</tbody>
        </table>
        """

if __name__ == "__main__":
    # テスト実行
    reporter = DSSMSEnhancedReporter()
    
    print("🔄 DSSMS強化レポーター テスト実行")
    
    # パフォーマンス分析
    performance = reporter.analyze_dssms_performance(hours_back=24)
    print(f"切り替え成功率: {performance.switching_success_rate:.1f}%")
    print(f"ポートフォリオ価値: ¥{performance.portfolio_value:,.0f}")
    
    # 戦略分析
    strategies = reporter.analyze_strategy_performance(hours_back=24)
    print(f"分析された戦略数: {len(strategies)}")
    
    # ポートフォリオ診断
    portfolio = reporter.diagnose_portfolio_health(hours_back=24)
    print(f"ポートフォリオ健康度: {portfolio.health_status}")
    
    # HTMLレポート生成
    html_path = "output/dssms_enhanced_report.html"
    reporter.generate_dssms_html_report(html_path, hours_back=24)
    print(f"DSSMSレポートを生成: {html_path}")

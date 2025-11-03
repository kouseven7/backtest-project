"""
マルチ戦略システムポジション管理バグ調査スクリプト

問題:
- 全取引のstrategy列がティッカーシンボル（8306.T）になっている
- holding_period_daysが異常に短い（1-10日）
- 各戦略単独では正常なのにマルチ戦略で不具合

主な機能:
- CSVデータの詳細分析とstrategy列の値検証
- holding_period分析による異常に短い保有期間の検出
- Entry/Exit日付検証による日付計算の整合性確認
- performance_metrics.json検証による実行サマリーの矛盾検出
- ログ解析による実行フローの追跡
- 修正推奨箇所の特定とバグ修正が必要なファイル・メソッドのリスト化

統合コンポーネント:
- IntegratedExecutionManager: データ渡しの検証
- StrategyExecutionManager: データフローの追跡
- ComprehensiveReporter: CSV出力時のデータマッピング検証

セーフティ機能/注意事項:
- 最新のレポートディレクトリを自動検出
- ファイルが見つからない場合のエラーハンドリング
- 調査結果をJSON形式で保存
- 推測を避け、実際のデータに基づいて分析

Author: Backtest Project Team
Created: 2025-10-30
Last Modified: 2025-10-30
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトパス設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logger_config import setup_logger


class MultiStrategyPositionBugInvestigator:
    """マルチ戦略システムポジション管理バグ調査クラス"""
    
    def __init__(self, output_dir: str = None):
        """
        初期化
        
        Args:
            output_dir: 調査対象のoutputディレクトリパス
                       Noneの場合は最新のレポートを自動検出
        """
        self.logger = setup_logger(
            "MultiStrategyBugInvestigator",
            log_file="logs/multistrategy_bug_investigation.log"
        )
        
        if output_dir is None:
            # 最新のレポートディレクトリを自動検出
            reports_dir = project_root / "output" / "comprehensive_reports"
            if reports_dir.exists():
                subdirs = [d for d in reports_dir.iterdir() if d.is_dir()]
                if subdirs:
                    self.output_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
                    self.logger.info(f"Auto-detected output directory: {self.output_dir}")
                else:
                    raise FileNotFoundError("No report directories found")
            else:
                raise FileNotFoundError(f"Reports directory not found: {reports_dir}")
        else:
            self.output_dir = Path(output_dir)
        
        # 調査結果保存先
        self.investigation_results = {
            'timestamp': datetime.now().isoformat(),
            'output_dir': str(self.output_dir),
            'findings': [],
            'recommendations': []
        }
    
    def run_full_investigation(self) -> Dict[str, Any]:
        """
        フル調査実行
        
        Returns:
            調査結果辞書
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Multi-Strategy Position Bug Investigation")
        self.logger.info("=" * 80)
        
        try:
            # 1. CSVデータ読み込み・検証
            self.logger.info("[STEP 1/6] Loading and validating CSV data...")
            self._investigate_csv_data()
            
            # 2. strategy列の値分析
            self.logger.info("[STEP 2/6] Analyzing strategy column values...")
            self._analyze_strategy_column()
            
            # 3. holding_period分析
            self.logger.info("[STEP 3/6] Analyzing holding periods...")
            self._analyze_holding_periods()
            
            # 4. Entry/Exit日付の検証
            self.logger.info("[STEP 4/6] Validating entry/exit dates...")
            self._validate_entry_exit_dates()
            
            # 5. performance_metrics.json検証
            self.logger.info("[STEP 5/6] Validating performance metrics...")
            self._validate_performance_metrics()
            
            # 6. 実行フロー追跡
            self.logger.info("[STEP 6/6] Tracing execution flow...")
            self._trace_execution_flow()
            
            # 7. 推奨修正箇所の特定
            self.logger.info("[FINAL STEP] Identifying fix recommendations...")
            self._identify_fix_recommendations()
            
            self.logger.info("=" * 80)
            self.logger.info("Investigation completed successfully")
            self.logger.info("=" * 80)
            
            return self.investigation_results
            
        except Exception as e:
            self.logger.error(f"Investigation failed: {e}")
            self.investigation_results['error'] = str(e)
            return self.investigation_results
    
    def _investigate_csv_data(self):
        """CSVデータ読み込み・基本検証"""
        trades_csv = self.output_dir / f"{self.output_dir.name.split('_')[0]}_trades.csv"
        
        if not trades_csv.exists():
            finding = f"CRITICAL: trades.csv not found at {trades_csv}"
            self.logger.error(finding)
            self.investigation_results['findings'].append(finding)
            return
        
        # CSVロード
        df = pd.read_csv(trades_csv)
        
        finding = {
            'step': 'csv_data_validation',
            'total_rows': len(df),
            'columns': list(df.columns),
            'data_types': {k: str(v) for k, v in df.dtypes.to_dict().items()},
            'sample_rows': df.head(3).to_dict('records')
        }
        
        self.investigation_results['findings'].append(finding)
        self.logger.info(f"Loaded {len(df)} rows from trades.csv")
        
        # strategy列の値を確認
        if 'strategy' in df.columns:
            unique_strategies = df['strategy'].unique()
            self.logger.info(f"Unique strategy values: {unique_strategies}")
            
            # BUG検出: strategy列がティッカー名になっている
            if all('.' in str(s) for s in unique_strategies):  # 日本株ティッカー形式
                self.investigation_results['findings'].append({
                    'bug_detected': 'STRATEGY_COLUMN_CORRUPTED',
                    'description': 'strategy列の値が全てティッカーシンボルになっている',
                    'expected': ['VWAPBreakoutStrategy', 'MomentumInvestingStrategy', '...'],
                    'actual': list(unique_strategies),
                    'severity': 'CRITICAL'
                })
        
        self.trades_df = df
    
    def _analyze_strategy_column(self):
        """strategy列の詳細分析"""
        if not hasattr(self, 'trades_df'):
            return
        
        df = self.trades_df
        
        if 'strategy' not in df.columns:
            self.investigation_results['findings'].append({
                'bug_detected': 'MISSING_STRATEGY_COLUMN',
                'severity': 'CRITICAL'
            })
            return
        
        # 値の分布
        strategy_counts = df['strategy'].value_counts().to_dict()
        
        finding = {
            'step': 'strategy_column_analysis',
            'value_distribution': strategy_counts,
            'unique_count': len(df['strategy'].unique()),
            'null_count': int(df['strategy'].isnull().sum())
        }
        
        # ティッカーシンボルパターン検出
        sample_strategy = df['strategy'].iloc[0]
        if '.' in str(sample_strategy) and str(sample_strategy).endswith('.T'):
            finding['bug_confirmed'] = 'strategy列がティッカーシンボルで上書きされている'
            finding['likely_cause'] = 'IntegratedExecutionManager._execute_single_strategy()でtickerをstrategy_nameとして渡している可能性'
        
        self.investigation_results['findings'].append(finding)
    
    def _analyze_holding_periods(self):
        """holding_period分析"""
        if not hasattr(self, 'trades_df'):
            return
        
        df = self.trades_df
        
        if 'holding_period_days' not in df.columns:
            return
        
        stats = {
            'step': 'holding_period_analysis',
            'mean': float(df['holding_period_days'].mean()),
            'median': float(df['holding_period_days'].median()),
            'min': int(df['holding_period_days'].min()),
            'max': int(df['holding_period_days'].max()),
            'distribution': {int(k): int(v) for k, v in df['holding_period_days'].value_counts().to_dict().items()}
        }
        
        # 異常検出: 平均が3日未満
        if stats['mean'] < 3:
            stats['anomaly'] = '異常に短い保有期間（平均<3日）'
            stats['expected_range'] = '数日から数週間（5-30日）'
        
        self.investigation_results['findings'].append(stats)
    
    def _validate_entry_exit_dates(self):
        """Entry/Exit日付の検証"""
        if not hasattr(self, 'trades_df'):
            return
        
        df = self.trades_df
        
        # entry_date, exit_date列の存在確認
        if 'entry_date' not in df.columns or 'exit_date' not in df.columns:
            return
        
        # 日付変換
        df['entry_date_dt'] = pd.to_datetime(df['entry_date'])
        df['exit_date_dt'] = pd.to_datetime(df['exit_date'])
        
        # holding_period再計算
        df['holding_period_calculated'] = (df['exit_date_dt'] - df['entry_date_dt']).dt.days
        
        # holding_period_days列との比較
        if 'holding_period_days' in df.columns:
            mismatch = (df['holding_period_days'] != df['holding_period_calculated']).sum()
            
            finding = {
                'step': 'entry_exit_validation',
                'mismatch_count': int(mismatch),
                'sample_mismatches': df[df['holding_period_days'] != df['holding_period_calculated']].head(3).to_dict('records') if mismatch > 0 else []
            }
            
            self.investigation_results['findings'].append(finding)
    
    def _validate_performance_metrics(self):
        """performance_metrics.json検証"""
        metrics_json = self.output_dir / f"{self.output_dir.name.split('_')[0]}_performance_metrics.json"
        
        if not metrics_json.exists():
            return
        
        import json
        with open(metrics_json, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        finding = {
            'step': 'performance_metrics_validation',
            'basic_metrics': metrics.get('basic_metrics', {}),
            'execution_summary': metrics.get('execution_summary', {})
        }
        
        # 異常検出: total_executions=0 なのに successful_strategies>0
        exec_summary = metrics.get('execution_summary', {})
        if exec_summary.get('total_executions', 0) == 0 and exec_summary.get('successful_strategies', 0) > 0:
            finding['anomaly'] = 'total_executions=0 だが successful_strategies>0（矛盾）'
        
        self.investigation_results['findings'].append(finding)
    
    def _trace_execution_flow(self):
        """実行フロー追跡（ログ解析）"""
        # integrated_execution.logを解析
        log_file = project_root / "logs" / "integrated_execution.log"
        
        if not log_file.exists():
            self.investigation_results['findings'].append({
                'step': 'execution_flow_trace',
                'status': 'log_file_not_found',
                'log_path': str(log_file)
            })
            return
        
        # ログから戦略実行の流れを抽出
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_lines = f.readlines()
        
        # "Executing single strategy" パターンを検索
        execution_patterns = []
        for line in log_lines:
            if "Executing single strategy:" in line:
                execution_patterns.append(line.strip())
        
        finding = {
            'step': 'execution_flow_trace',
            'execution_count': len(execution_patterns),
            'sample_executions': execution_patterns[:5] if execution_patterns else []
        }
        
        self.investigation_results['findings'].append(finding)
    
    def _identify_fix_recommendations(self):
        """修正推奨箇所の特定"""
        recommendations = []
        
        # 1. IntegratedExecutionManager._execute_single_strategy()
        recommendations.append({
            'file': 'main_system/execution_control/integrated_execution_manager.py',
            'method': '_execute_single_strategy()',
            'issue': 'tickerをstrategy_nameとして渡している可能性',
            'fix': 'execute_strategy()の引数を修正: strategy_name を正しく渡す',
            'priority': 'CRITICAL'
        })
        
        # 2. StrategyExecutionManager.execute_strategy()
        recommendations.append({
            'file': 'main_system/execution_control/strategy_execution_manager.py',
            'method': 'execute_strategy()',
            'issue': '戦略名がティッカーシンボルで上書きされている',
            'fix': 'バックテスト結果の strategy 列の値を確認・修正',
            'priority': 'CRITICAL'
        })
        
        # 3. ComprehensiveReporter（CSV出力）
        recommendations.append({
            'file': 'main_system/reporting/comprehensive_reporter.py',
            'method': 'generate_comprehensive_report()',
            'issue': 'trades.csvのstrategy列マッピングが誤っている可能性',
            'fix': 'strategy列のデータソースを確認',
            'priority': 'HIGH'
        })
        
        self.investigation_results['recommendations'] = recommendations
        
        self.logger.info("=" * 80)
        self.logger.info("FIX RECOMMENDATIONS:")
        self.logger.info("=" * 80)
        for i, rec in enumerate(recommendations, 1):
            self.logger.info(f"{i}. [{rec['priority']}] {rec['file']}")
            self.logger.info(f"   Issue: {rec['issue']}")
            self.logger.info(f"   Fix: {rec['fix']}")
    
    def print_summary(self):
        """調査結果サマリー出力"""
        print("\n" + "=" * 80)
        print("Multi-Strategy Position Bug Investigation Summary")
        print("=" * 80)
        
        print(f"\nOutput Directory: {self.output_dir}")
        print(f"Timestamp: {self.investigation_results['timestamp']}")
        
        print("\n[FINDINGS]")
        for i, finding in enumerate(self.investigation_results['findings'], 1):
            if isinstance(finding, dict):
                print(f"{i}. {finding.get('step', 'N/A')}")
                if 'bug_detected' in finding:
                    print(f"   BUG: {finding['bug_detected']}")
                if 'anomaly' in finding:
                    print(f"   ANOMALY: {finding['anomaly']}")
            else:
                print(f"{i}. {finding}")
        
        print("\n[RECOMMENDATIONS]")
        for i, rec in enumerate(self.investigation_results['recommendations'], 1):
            print(f"{i}. [{rec['priority']}] {rec['file']} - {rec['method']}")
            print(f"   Fix: {rec['fix']}")
        
        print("\n" + "=" * 80)


def main():
    """メインエントリーポイント"""
    print("Multi-Strategy Position Bug Investigation Tool")
    print("=" * 80)
    
    try:
        # 調査実行
        investigator = MultiStrategyPositionBugInvestigator()
        results = investigator.run_full_investigation()
        
        # サマリー出力
        investigator.print_summary()
        
        # 結果をJSONで保存
        import json
        output_file = project_root / "logs" / "multistrategy_bug_investigation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[INFO] Investigation results saved to: {output_file}")
        
        return investigator
        
    except Exception as e:
        print(f"\n[ERROR] Investigation failed: {e}")
        raise


if __name__ == '__main__':
    investigator = main()

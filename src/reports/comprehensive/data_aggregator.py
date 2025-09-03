"""
データ集約モジュール

DSSMS改善プロジェクト Phase 3 Task 3.3
包括的レポートシステムのデータ集約コンポーネント

機能:
- 階層化データ集約（サマリー/詳細/包括的レベル）
- 既存レポートシステムとの統合
- DSSMS、戦略、パフォーマンス、リスクデータの統合
- キャッシュ機能付きデータ処理
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from functools import lru_cache

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger


class DataAggregator:
    """データ集約クラス"""
    
    def __init__(self):
        """初期化"""
        self.logger = setup_logger(__name__)
        self.logger.info("=== DataAggregator 初期化開始 ===")
        
        # キャッシュ設定
        self.cache_enabled = True
        self.cache_duration = timedelta(hours=24)
        self.data_cache = {}
        
        # データソースパス設定
        self.data_sources = {
            'dssms_results': project_root / "output",
            'strategy_results': project_root / "output",
            'existing_reports': project_root / "src" / "reports",
            'backtest_data': project_root / "data",
            'config_data': project_root / "config"
        }
        
        self.logger.info("DataAggregator 初期化完了")
    
    def aggregate_data(
        self,
        data_sources: Optional[List[str]] = None,
        date_range: Optional[Dict[str, datetime]] = None,
        strategies: Optional[List[str]] = None,
        level: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        データ集約メイン処理
        
        Args:
            data_sources: データソースリスト
            date_range: 対象期間
            strategies: 対象戦略リスト
            level: 詳細レベル（summary/detailed/comprehensive）
            
        Returns:
            集約されたデータ
        """
        try:
            self.logger.info(f"データ集約開始 - レベル: {level}")
            
            # 集約データ初期化
            aggregated_data = {
                'metadata': {
                    'aggregation_timestamp': datetime.now(),
                    'level': level,
                    'data_sources': data_sources or [],
                    'date_range': date_range,
                    'strategies': strategies or []
                },
                'dssms_data': {},
                'strategy_data': {},
                'performance_data': {},
                'risk_data': {},
                'existing_reports_data': {},
                'summary_statistics': {}
            }
            
            # DSSMSデータ集約
            self.logger.info("DSSMSデータ集約開始")
            aggregated_data['dssms_data'] = self._aggregate_dssms_data(
                date_range=date_range,
                level=level
            )
            
            # 戦略データ集約
            self.logger.info("戦略データ集約開始")
            aggregated_data['strategy_data'] = self._aggregate_strategy_data(
                strategies=strategies,
                date_range=date_range,
                level=level
            )
            
            # パフォーマンスデータ集約
            self.logger.info("パフォーマンスデータ集約開始")
            aggregated_data['performance_data'] = self._aggregate_performance_data(
                strategies=strategies,
                date_range=date_range,
                level=level
            )
            
            # リスクデータ集約
            self.logger.info("リスクデータ集約開始")
            aggregated_data['risk_data'] = self._aggregate_risk_data(
                strategies=strategies,
                date_range=date_range,
                level=level
            )
            
            # 既存レポートデータ統合
            self.logger.info("既存レポートデータ統合開始")
            aggregated_data['existing_reports_data'] = self._integrate_existing_reports(
                level=level
            )
            
            # サマリー統計計算
            self.logger.info("サマリー統計計算開始")
            aggregated_data['summary_statistics'] = self._calculate_summary_statistics(
                aggregated_data,
                level=level
            )
            
            self.logger.info("データ集約完了")
            return aggregated_data
            
        except Exception as e:
            self.logger.error(f"データ集約エラー: {e}")
            return {}
    
    def _aggregate_dssms_data(
        self,
        date_range: Optional[Dict[str, datetime]] = None,
        level: str = "comprehensive"
    ) -> Dict[str, Any]:
        """DSSMSデータ集約"""
        try:
            dssms_data = {}
            
            # DSSMSシミュレーション結果ファイル検索
            output_dir = self.data_sources['dssms_results']
            dssms_files = []
            
            # CSVファイル検索
            for pattern in ['*dssms*.csv', '*simulation*.csv', '*balanced*.csv']:
                dssms_files.extend(output_dir.glob(pattern))
            
            self.logger.info(f"DSSMS関連ファイル検出数: {len(dssms_files)}")
            
            for file_path in dssms_files:
                try:
                    # ファイル読み込み
                    df = pd.read_csv(file_path)
                    
                    # 日付範囲フィルタリング
                    if date_range and 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        if 'start' in date_range:
                            df = df[df['Date'] >= date_range['start']]
                        if 'end' in date_range:
                            df = df[df['Date'] <= date_range['end']]
                    
                    # レベル別データ処理
                    if level == "summary":
                        # サマリーレベル：主要統計のみ
                        file_summary = {
                            'total_records': len(df),
                            'date_range': {
                                'start': df['Date'].min() if 'Date' in df.columns else None,
                                'end': df['Date'].max() if 'Date' in df.columns else None
                            },
                            'columns': list(df.columns),
                            'numeric_columns_summary': {}
                        }
                        
                        # 数値列の基本統計
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            file_summary['numeric_columns_summary'][col] = {
                                'mean': float(df[col].mean()),
                                'std': float(df[col].std()),
                                'min': float(df[col].min()),
                                'max': float(df[col].max())
                            }
                        
                        dssms_data[file_path.stem] = file_summary
                        
                    elif level == "detailed":
                        # 詳細レベル：集約データ＋サンプル
                        dssms_data[file_path.stem] = {
                            'full_data': df.head(1000),  # 最初の1000行
                            'summary': df.describe(),
                            'columns_info': {
                                'total_columns': len(df.columns),
                                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                                'datetime_columns': len(df.select_dtypes(include=['datetime']).columns),
                                'object_columns': len(df.select_dtypes(include=['object']).columns)
                            }
                        }
                        
                    else:  # comprehensive
                        # 包括的レベル：全データ
                        dssms_data[file_path.stem] = {
                            'full_data': df,
                            'summary': df.describe(),
                            'info': {
                                'shape': df.shape,
                                'columns': list(df.columns),
                                'dtypes': df.dtypes.to_dict(),
                                'memory_usage': df.memory_usage(deep=True).sum(),
                                'null_counts': df.isnull().sum().to_dict()
                            }
                        }
                    
                except Exception as e:
                    self.logger.warning(f"DSSMSファイル処理エラー {file_path}: {e}")
                    continue
            
            return dssms_data
            
        except Exception as e:
            self.logger.error(f"DSSMSデータ集約エラー: {e}")
            return {}
    
    def _aggregate_strategy_data(
        self,
        strategies: Optional[List[str]] = None,
        date_range: Optional[Dict[str, datetime]] = None,
        level: str = "comprehensive"
    ) -> Dict[str, Any]:
        """戦略データ集約"""
        try:
            strategy_data = {}
            
            # 戦略ディレクトリ検索
            strategies_dir = project_root / "strategies"
            if not strategies_dir.exists():
                self.logger.warning("戦略ディレクトリが見つかりません")
                return {}
            
            # 戦略ファイル検索
            strategy_files = list(strategies_dir.glob("*.py"))
            self.logger.info(f"戦略ファイル検出数: {len(strategy_files)}")
            
            for strategy_file in strategy_files:
                strategy_name = strategy_file.stem
                
                # 特定戦略のみ処理する場合
                if strategies and strategy_name not in strategies:
                    continue
                
                try:
                    # 戦略情報取得
                    strategy_info = self._extract_strategy_info(strategy_file, level)
                    if strategy_info:
                        strategy_data[strategy_name] = strategy_info
                        
                except Exception as e:
                    self.logger.warning(f"戦略ファイル処理エラー {strategy_file}: {e}")
                    continue
            
            return strategy_data
            
        except Exception as e:
            self.logger.error(f"戦略データ集約エラー: {e}")
            return {}
    
    def _extract_strategy_info(self, strategy_file: Path, level: str) -> Dict[str, Any]:
        """戦略ファイルから情報抽出"""
        try:
            with open(strategy_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            strategy_info = {
                'file_path': str(strategy_file),
                'file_size': strategy_file.stat().st_size,
                'last_modified': datetime.fromtimestamp(strategy_file.stat().st_mtime)
            }
            
            if level == "summary":
                # サマリーレベル：基本情報のみ
                strategy_info.update({
                    'lines_count': len(content.splitlines()),
                    'has_backtest_method': 'def backtest(' in content,
                    'has_class_definition': 'class ' in content
                })
                
            elif level == "detailed":
                # 詳細レベル：クラス・メソッド情報
                lines = content.splitlines()
                strategy_info.update({
                    'lines_count': len(lines),
                    'class_definitions': [line.strip() for line in lines if line.strip().startswith('class ')],
                    'method_definitions': [line.strip() for line in lines if line.strip().startswith('def ')],
                    'imports': [line.strip() for line in lines if line.strip().startswith('import ') or line.strip().startswith('from ')],
                    'docstring': self._extract_docstring(content)
                })
                
            else:  # comprehensive
                # 包括的レベル：全情報
                strategy_info.update({
                    'full_content': content,
                    'lines_count': len(content.splitlines()),
                    'detailed_analysis': self._analyze_strategy_code(content)
                })
            
            return strategy_info
            
        except Exception as e:
            self.logger.error(f"戦略情報抽出エラー {strategy_file}: {e}")
            return {}
    
    def _extract_docstring(self, content: str) -> Optional[str]:
        """ドキュメント文字列抽出"""
        try:
            lines = content.splitlines()
            in_docstring = False
            docstring_lines = []
            quote_type = None
            
            for line in lines:
                stripped = line.strip()
                
                if not in_docstring:
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        quote_type = stripped[:3]
                        in_docstring = True
                        docstring_lines.append(stripped[3:])
                        if stripped.endswith(quote_type) and len(stripped) > 6:
                            break
                else:
                    if stripped.endswith(quote_type):
                        docstring_lines.append(stripped[:-3])
                        break
                    else:
                        docstring_lines.append(line)
            
            return '\n'.join(docstring_lines) if docstring_lines else None
            
        except Exception:
            return None
    
    def _analyze_strategy_code(self, content: str) -> Dict[str, Any]:
        """戦略コード詳細分析"""
        try:
            analysis = {
                'complexity_metrics': {},
                'dependencies': [],
                'key_methods': [],
                'parameters': []
            }
            
            lines = content.splitlines()
            
            # 複雑度メトリクス
            analysis['complexity_metrics'] = {
                'total_lines': len(lines),
                'code_lines': len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
                'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
                'function_count': len([line for line in lines if line.strip().startswith('def ')]),
                'class_count': len([line for line in lines if line.strip().startswith('class ')])
            }
            
            # 依存関係
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    analysis['dependencies'].append(stripped)
            
            # 主要メソッド
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('def '):
                    method_name = stripped.split('(')[0].replace('def ', '').strip()
                    analysis['key_methods'].append(method_name)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"戦略コード分析エラー: {e}")
            return {}
    
    def _aggregate_performance_data(
        self,
        strategies: Optional[List[str]] = None,
        date_range: Optional[Dict[str, datetime]] = None,
        level: str = "comprehensive"
    ) -> Dict[str, Any]:
        """パフォーマンスデータ集約"""
        try:
            performance_data = {}
            
            # パフォーマンス関連ファイル検索
            output_dir = self.data_sources['strategy_results']
            performance_files = []
            
            for pattern in ['*performance*.csv', '*results*.csv', '*analysis*.csv']:
                performance_files.extend(output_dir.glob(pattern))
            
            self.logger.info(f"パフォーマンスファイル検出数: {len(performance_files)}")
            
            for file_path in performance_files:
                try:
                    df = pd.read_csv(file_path)
                    
                    # レベル別処理
                    if level == "summary":
                        performance_data[file_path.stem] = {
                            'record_count': len(df),
                            'columns': list(df.columns)
                        }
                    else:
                        performance_data[file_path.stem] = {
                            'data': df if level == "comprehensive" else df.head(100),
                            'summary': df.describe() if not df.empty else {}
                        }
                        
                except Exception as e:
                    self.logger.warning(f"パフォーマンスファイル処理エラー {file_path}: {e}")
                    continue
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"パフォーマンスデータ集約エラー: {e}")
            return {}
    
    def _aggregate_risk_data(
        self,
        strategies: Optional[List[str]] = None,
        date_range: Optional[Dict[str, datetime]] = None,
        level: str = "comprehensive"
    ) -> Dict[str, Any]:
        """リスクデータ集約"""
        try:
            risk_data = {}
            
            # リスク設定ファイル読み込み
            risk_config_path = project_root / "config" / "risk_management.py"
            if risk_config_path.exists():
                with open(risk_config_path, 'r', encoding='utf-8') as f:
                    risk_content = f.read()
                
                risk_data['risk_configuration'] = {
                    'file_exists': True,
                    'content_length': len(risk_content),
                    'configuration': risk_content if level == "comprehensive" else "配置済み"
                }
            
            # リスク関連ファイル検索
            config_dir = self.data_sources['config_data']
            risk_files = list(config_dir.glob("*risk*.py")) + list(config_dir.glob("*risk*.json"))
            
            for risk_file in risk_files:
                try:
                    if risk_file.suffix == '.json':
                        with open(risk_file, 'r', encoding='utf-8') as f:
                            risk_data[risk_file.stem] = json.load(f)
                    else:
                        with open(risk_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        risk_data[risk_file.stem] = {
                            'content': content if level == "comprehensive" else f"ファイルサイズ: {len(content)} 文字"
                        }
                        
                except Exception as e:
                    self.logger.warning(f"リスクファイル処理エラー {risk_file}: {e}")
                    continue
            
            return risk_data
            
        except Exception as e:
            self.logger.error(f"リスクデータ集約エラー: {e}")
            return {}
    
    def _integrate_existing_reports(self, level: str = "comprehensive") -> Dict[str, Any]:
        """既存レポートシステム統合"""
        try:
            reports_data = {}
            
            # 既存レポートファイル検索
            reports_dir = self.data_sources['existing_reports']
            report_files = list(reports_dir.glob("*.py"))
            
            self.logger.info(f"既存レポートファイル検出数: {len(report_files)}")
            
            for report_file in report_files:
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if level == "summary":
                        reports_data[report_file.stem] = {
                            'file_size': len(content),
                            'has_class': 'class ' in content,
                            'has_main': '__main__' in content
                        }
                    elif level == "detailed":
                        reports_data[report_file.stem] = {
                            'file_info': {
                                'size': len(content),
                                'lines': len(content.splitlines()),
                                'last_modified': datetime.fromtimestamp(report_file.stat().st_mtime)
                            },
                            'code_structure': {
                                'classes': [line.strip() for line in content.splitlines() if line.strip().startswith('class ')],
                                'functions': [line.strip() for line in content.splitlines() if line.strip().startswith('def ')]
                            }
                        }
                    else:  # comprehensive
                        reports_data[report_file.stem] = {
                            'full_content': content,
                            'file_path': str(report_file),
                            'metadata': {
                                'size': len(content),
                                'lines': len(content.splitlines()),
                                'last_modified': datetime.fromtimestamp(report_file.stat().st_mtime)
                            }
                        }
                        
                except Exception as e:
                    self.logger.warning(f"レポートファイル処理エラー {report_file}: {e}")
                    continue
            
            return reports_data
            
        except Exception as e:
            self.logger.error(f"既存レポート統合エラー: {e}")
            return {}
    
    def _calculate_summary_statistics(
        self,
        aggregated_data: Dict[str, Any],
        level: str = "comprehensive"
    ) -> Dict[str, Any]:
        """サマリー統計計算"""
        try:
            summary_stats = {
                'data_overview': {
                    'total_dssms_files': len(aggregated_data.get('dssms_data', {})),
                    'total_strategies': len(aggregated_data.get('strategy_data', {})),
                    'total_performance_files': len(aggregated_data.get('performance_data', {})),
                    'total_risk_configs': len(aggregated_data.get('risk_data', {})),
                    'total_existing_reports': len(aggregated_data.get('existing_reports_data', {}))
                },
                'generation_info': {
                    'timestamp': datetime.now(),
                    'level': level,
                    'processing_time': 0  # 実際の処理時間は呼び出し元で設定
                }
            }
            
            # DSSMSデータ統計
            if aggregated_data.get('dssms_data'):
                dssms_stats = {
                    'total_records': 0,
                    'file_sizes': []
                }
                
                for file_name, file_data in aggregated_data['dssms_data'].items():
                    if isinstance(file_data, dict):
                        if 'total_records' in file_data:
                            dssms_stats['total_records'] += file_data['total_records']
                        if 'full_data' in file_data and hasattr(file_data['full_data'], '__len__'):
                            dssms_stats['total_records'] += len(file_data['full_data'])
                
                summary_stats['dssms_statistics'] = dssms_stats
            
            # 戦略統計
            if aggregated_data.get('strategy_data'):
                strategy_stats = {
                    'total_strategies': len(aggregated_data['strategy_data']),
                    'strategy_types': [],
                    'total_lines_of_code': 0
                }
                
                for strategy_name, strategy_info in aggregated_data['strategy_data'].items():
                    if isinstance(strategy_info, dict) and 'lines_count' in strategy_info:
                        strategy_stats['total_lines_of_code'] += strategy_info['lines_count']
                
                summary_stats['strategy_statistics'] = strategy_stats
            
            return summary_stats
            
        except Exception as e:
            self.logger.error(f"サマリー統計計算エラー: {e}")
            return {}
    
    def aggregate_comparison_data(
        self,
        comparison_items: List[Dict[str, Any]],
        comparison_type: str = "strategies",
        level: str = "detailed"
    ) -> Dict[str, Any]:
        """比較用データ集約"""
        try:
            self.logger.info(f"比較データ集約開始: {comparison_type}")
            
            comparison_data = {
                'comparison_type': comparison_type,
                'comparison_items': comparison_items,
                'level': level,
                'timestamp': datetime.now(),
                'data': {}
            }
            
            if comparison_type == "strategies":
                # 戦略比較データ
                for item in comparison_items:
                    strategy_name = item.get('name') or item.get('strategy')
                    if strategy_name:
                        strategy_data = self._aggregate_strategy_data(
                            strategies=[strategy_name],
                            level=level
                        )
                        comparison_data['data'][strategy_name] = strategy_data.get(strategy_name, {})
            
            elif comparison_type == "periods":
                # 期間比較データ
                for item in comparison_items:
                    period_name = item.get('name')
                    date_range = item.get('date_range')
                    if period_name and date_range:
                        period_data = self.aggregate_data(
                            date_range=date_range,
                            level=level
                        )
                        comparison_data['data'][period_name] = period_data
            
            elif comparison_type == "configurations":
                # 設定比較データ
                for item in comparison_items:
                    config_name = item.get('name')
                    config_params = item.get('config')
                    if config_name:
                        # 設定固有のデータ集約ロジック
                        comparison_data['data'][config_name] = {
                            'config': config_params,
                            'metadata': {
                                'item': item,
                                'timestamp': datetime.now()
                            }
                        }
            
            self.logger.info(f"比較データ集約完了: {len(comparison_data['data'])} 項目")
            return comparison_data
            
        except Exception as e:
            self.logger.error(f"比較データ集約エラー: {e}")
            return {}


if __name__ == "__main__":
    # デモ実行
    aggregator = DataAggregator()
    
    # サンプルデータ集約
    result = aggregator.aggregate_data(level="summary")
    print(f"データ集約結果: {len(result)} カテゴリ")
    
    for category, data in result.items():
        if isinstance(data, dict):
            print(f"  {category}: {len(data)} 項目")
        else:
            print(f"  {category}: {type(data)}")

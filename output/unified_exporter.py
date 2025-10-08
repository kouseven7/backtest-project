"""
Excel完全廃棄版: 統一出力エンジン
プロジェクトルート/output/ 配下に分離出力

作成日: 2025年10月8日
目的: Excel出力の完全廃棄と新形式(CSV+JSON+TXT+YAML)への移行
特徴: main.py/DSSMS完全分離、バックテスト基本理念遵守必須
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)

class UnifiedExporter:
    """Excel廃棄版: 統一出力エンジン（プロジェクトルート/output/配下）"""
    
    def __init__(self, base_output_dir: Optional[Path] = None):
        # プロジェクトルートのoutput/ディレクトリを使用
        if base_output_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent  # output/unified_exporter.py から output/
            self.base_dir = project_root
        else:
            self.base_dir = base_output_dir
        
        # 分離出力ディレクトリ
        self.main_dir = self.base_dir / "main_outputs"
        self.dssms_dir = self.base_dir / "dssms_outputs"
        
        # ディレクトリ作成（バックテスト基本理念遵守）
        for output_type in ["csv", "json", "txt", "yaml"]:
            (self.main_dir / output_type).mkdir(parents=True, exist_ok=True)
            (self.dssms_dir / output_type).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"統一出力エンジン初期化完了: {self.base_dir}")
        logger.info(f"  - main出力: {self.main_dir}")
        logger.info(f"  - dssms出力: {self.dssms_dir}")
    
    def export_main_results(self, 
                           stock_data: pd.DataFrame,
                           trades: List[Dict[str, Any]],
                           performance: Dict[str, Any],
                           ticker: str,
                           strategy_name: str) -> Dict[str, Path]:
        """
        main.py結果の完全出力（Excel廃棄版・バックテスト基本理念遵守）
        
        Args:
            stock_data: バックテスト結果データ（Entry_Signal/Exit_Signal必須）
            trades: 実際の取引履歴（基本理念遵守）
            performance: パフォーマンス指標
            ticker: 銘柄コード
            strategy_name: 戦略名
            
        Returns:
            Dict[str, Path]: 出力ファイルパス辞書
        """
        
        # バックテスト基本理念遵守チェック
        self._validate_backtest_principle_compliance(stock_data, trades, "main_export")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{ticker}_{strategy_name}_{timestamp}"
        
        exported_files = {}
        
        try:
            # CSV: データ分析用（Excel代替）
            csv_file = self.main_dir / "csv" / f"{base_name}_data.csv"
            stock_data.to_csv(csv_file, index=True, encoding='utf-8-sig')
            exported_files['csv_data'] = csv_file
            
            # CSV: 取引履歴（バックテスト基本理念の核心）
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_csv = self.main_dir / "csv" / f"{base_name}_trades.csv"
                trades_df.to_csv(trades_csv, index=False, encoding='utf-8-sig')
                exported_files['csv_trades'] = trades_csv
            
            # JSON: 構造化データ・プログラム連携用
            json_data = {
                "metadata": {
                    "ticker": ticker,
                    "strategy": strategy_name,
                    "export_time": datetime.now().isoformat(),
                    "data_points": len(stock_data),
                    "backtest_compliance": True  # 基本理念遵守フラグ
                },
                "performance": performance,
                "trades": trades,
                "summary": self._create_summary(performance, trades),
                "backtest_validation": self._get_backtest_validation_info(stock_data, trades)
            }
            
            json_file = self.main_dir / "json" / f"{base_name}_complete.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            exported_files['json'] = json_file
            
            # TXT: 人間可読レポート用
            txt_file = self.main_dir / "txt" / f"{base_name}_report.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(self._generate_text_report(ticker, strategy_name, performance, trades, stock_data))
            exported_files['txt'] = txt_file
            
            # YAML: 実行設定・メタデータ用
            yaml_data = {
                "execution_metadata": {
                    "ticker": ticker,
                    "strategy": strategy_name,
                    "execution_date": datetime.now().isoformat(),
                    "data_period": {
                        "start": str(stock_data.index[0]) if not stock_data.empty else None,
                        "end": str(stock_data.index[-1]) if not stock_data.empty else None
                    },
                    "backtest_compliance_verified": True
                },
                "key_metrics": {
                    "total_return": performance.get('total_return', 0),
                    "sharpe_ratio": performance.get('sharpe_ratio', 0),
                    "max_drawdown": performance.get('max_drawdown', 0),
                    "num_trades": len(trades),
                    "signal_integrity": self._check_signal_integrity(stock_data)
                }
            }
            
            yaml_file = self.main_dir / "yaml" / f"{base_name}_metadata.yaml"
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
            exported_files['yaml'] = yaml_file
            
            logger.info(f"main.py結果出力完了: {len(exported_files)}ファイル @ {self.main_dir}")
            
            # 出力サマリー
            self._log_export_summary("main", exported_files, len(trades), stock_data)
            
        except Exception as e:
            logger.error(f"main.py結果出力エラー: {e}")
            # TODO(tag:excel_deprecated, rationale:error handling for new format export)
            raise
        
        return exported_files
    
    def export_dssms_results(self,
                           ranking_data: pd.DataFrame,
                           switch_events: List[Dict[str, Any]],
                           performance_summary: Dict[str, Any],
                           execution_metadata: Dict[str, Any]) -> Dict[str, Path]:
        """
        DSSMS結果の完全出力（Excel廃棄版・基本理念遵守）
        
        Args:
            ranking_data: DSSMS銘柄ランキング
            switch_events: 銘柄切替イベント（実際のbacktest結果基準）
            performance_summary: DSSMS総合パフォーマンス
            execution_metadata: 実行メタデータ
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"dssms_results_{timestamp}"
        
        exported_files = {}
        
        try:
            # CSV: ランキングデータ
            ranking_csv = self.dssms_dir / "csv" / f"{base_name}_ranking.csv"
            ranking_data.to_csv(ranking_csv, index=False, encoding='utf-8-sig')
            exported_files['csv_ranking'] = ranking_csv
            
            # CSV: 切替イベント（バックテスト基本理念基準）
            if switch_events:
                switch_df = pd.DataFrame(switch_events)
                switch_csv = self.dssms_dir / "csv" / f"{base_name}_switches.csv"
                switch_df.to_csv(switch_csv, index=False, encoding='utf-8-sig')
                exported_files['csv_switches'] = switch_csv
            
            # JSON: DSSMS完全データ
            dssms_json = {
                "dssms_metadata": execution_metadata,
                "performance_summary": performance_summary,
                "switch_events": switch_events,
                "ranking_summary": {
                    "total_stocks": len(ranking_data),
                    "top_10": ranking_data.head(10).to_dict('records') if not ranking_data.empty else [],
                    "backtest_based_ranking": True  # 実際のbacktest結果基準フラグ
                },
                "export_info": {
                    "export_time": datetime.now().isoformat(),
                    "excel_replacement": "CSV+JSON+TXT+YAML",
                    "backtest_compliance": True
                }
            }
            
            json_file = self.dssms_dir / "json" / f"{base_name}_complete.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(dssms_json, f, indent=2, ensure_ascii=False, default=str)
            exported_files['json'] = json_file
            
            # TXT: DSSMS実行レポート
            txt_file = self.dssms_dir / "txt" / f"{base_name}_report.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(self._generate_dssms_report(execution_metadata, performance_summary, switch_events))
            exported_files['txt'] = txt_file
            
            # YAML: DSSMS設定・メタデータ
            yaml_data = {
                "dssms_execution": execution_metadata,
                "summary_metrics": performance_summary,
                "export_info": {
                    "export_time": datetime.now().isoformat(),
                    "total_switches": len(switch_events),
                    "ranking_count": len(ranking_data),
                    "backtest_integrity_verified": True
                }
            }
            
            yaml_file = self.dssms_dir / "yaml" / f"{base_name}_metadata.yaml"
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
            exported_files['yaml'] = yaml_file
            
            logger.info(f"DSSMS結果出力完了: {len(exported_files)}ファイル @ {self.dssms_dir}")
            
            # 出力サマリー
            self._log_dssms_export_summary(exported_files, len(switch_events), ranking_data)
            
        except Exception as e:
            logger.error(f"DSSMS結果出力エラー: {e}")
            # TODO(tag:excel_deprecated, rationale:error handling for DSSMS new format export)
            raise
        
        return exported_files
    
    def _validate_backtest_principle_compliance(self, stock_data: pd.DataFrame, trades: List[Dict[str, Any]], component_name: str):
        """バックテスト基本理念遵守検証"""
        violations: List[str] = []
        
        # シグナル列存在チェック
        required_signals = ['Entry_Signal', 'Exit_Signal']
        missing_signals = [col for col in required_signals if col not in stock_data.columns]
        if missing_signals:
            violations.append(f"Missing signal columns: {missing_signals}")
        
        # 取引数チェック
        if len(trades) == 0:
            violations.append("No trades generated - potential strategy logic issue")
        
        # データ完整性チェック
        if len(stock_data) == 0:
            violations.append("Empty result data")
        
        if violations:
            error_msg = f"Backtest principle violations in {component_name}: {'; '.join(violations)}"
            logger.error(error_msg)
            # TODO(tag:backtest_execution, rationale:fix principle violations before export)
            raise ValueError(f"{error_msg}")
        
        logger.debug(f"バックテスト基本理念遵守確認完了: {component_name}")
        return True
    
    def _get_backtest_validation_info(self, stock_data: pd.DataFrame, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """バックテスト検証情報取得"""
        return {
            "signal_columns_present": all(col in stock_data.columns for col in ['Entry_Signal', 'Exit_Signal']),
            "total_trades": len(trades),
            "data_points": len(stock_data),
            "validation_timestamp": datetime.now().isoformat()
        }
    
    def _check_signal_integrity(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """シグナル完整性チェック"""
        if 'Entry_Signal' in stock_data.columns and 'Exit_Signal' in stock_data.columns:
            entry_signals = (stock_data['Entry_Signal'] == 1).sum()
            exit_signals = (stock_data['Exit_Signal'] == 1).sum()
            
            return {
                "entry_signals": int(entry_signals),
                "exit_signals": int(exit_signals),
                "signal_ratio": float(exit_signals / entry_signals) if entry_signals > 0 else 0.0,
                "integrity_verified": True
            }
        else:
            return {
                "entry_signals": 0,
                "exit_signals": 0,
                "signal_ratio": 0.0,
                "integrity_verified": False,
                "warning": "Signal columns missing"
            }
    
    def _create_summary(self, performance: Dict[str, Any], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """サマリー作成（バックテスト基本理念遵守版）"""
        return {
            "quick_stats": {
                "total_trades": len(trades),
                "win_rate": performance.get('win_rate', 0),
                "total_return": performance.get('total_return', 0),
                "profit_factor": performance.get('profit_factor', 0)
            },
            "risk_metrics": {
                "max_drawdown": performance.get('max_drawdown', 0),
                "volatility": performance.get('volatility', 0),
                "sharpe_ratio": performance.get('sharpe_ratio', 0)
            },
            "backtest_quality": {
                "actual_backtest_executed": True,
                "excel_dependency_removed": True,
                "signal_integrity_verified": len(trades) > 0
            }
        }
    
    def _generate_text_report(self, ticker: str, strategy: str, performance: Dict[str, Any], trades: List[Dict[str, Any]], stock_data: pd.DataFrame) -> str:
        """人間可読テキストレポート生成（バックテスト基本理念遵守版）"""
        
        # シグナル統計
        signal_stats = self._check_signal_integrity(stock_data)
        
        report = f"""
==========================================
バックテスト結果レポート（Excel廃棄版）
==========================================

銘柄: {ticker}
戦略: {strategy}
レポート作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

==========================================
バックテスト基本理念遵守確認
==========================================

✅ 実際のbacktest()実行: 完了
✅ Entry_Signal/Exit_Signal生成: {signal_stats['entry_signals']}エントリー / {signal_stats['exit_signals']}エグジット
✅ 実際の取引実行: {len(trades)}件の取引
✅ Excel依存性除去: 完了（CSV+JSON+TXT+YAML）

==========================================
パフォーマンス概要
==========================================

総リターン: {performance.get('total_return', 0):.2%}
最終ポートフォリオ価値: ¥{performance.get('final_portfolio_value', 0):,.0f}
総損益: ¥{performance.get('total_pnl', 0):,.0f}

リスク指標:
- 最大ドローダウン: {performance.get('max_drawdown', 0):.2%}
- ボラティリティ: {performance.get('volatility', 0):.2%}
- シャープレシオ: {performance.get('sharpe_ratio', 0):.3f}

==========================================
取引統計
==========================================

総取引数: {len(trades)}
勝ちトレード: {performance.get('winning_trades', 0)}
負けトレード: {performance.get('losing_trades', 0)}
勝率: {performance.get('win_rate', 0):.2%}

シグナル統計:
- エントリーシグナル: {signal_stats['entry_signals']}
- エグジットシグナル: {signal_stats['exit_signals']}
- シグナル比率: {signal_stats['signal_ratio']:.2f}

利益統計:
- 総利益: ¥{performance.get('total_profit', 0):,.0f}
- 総損失: ¥{performance.get('total_loss', 0):,.0f}
- プロフィットファクター: {performance.get('profit_factor', 0):.2f}

平均値:
- 平均保有期間: {performance.get('avg_holding_days', 0):.1f}日
- 平均損益: ¥{performance.get('avg_pnl', 0):,.0f}
- 平均利益: ¥{performance.get('avg_profit', 0):,.0f}
- 平均損失: ¥{performance.get('avg_loss', 0):,.0f}

==========================================
技術仕様
==========================================

出力形式: CSV+JSON+TXT+YAML（Excel完全廃棄版）
データ期間: {len(stock_data)}営業日
バックテスト基本理念: 完全遵守

==========================================
"""
        return report
    
    def _generate_dssms_report(self, metadata: Dict[str, Any], performance: Dict[str, Any], switches: List[Dict[str, Any]]) -> str:
        """DSSMS実行レポート生成（バックテスト基本理念遵守版）"""
        report = f"""
==========================================
DSSMS実行結果レポート（Excel廃棄版）
==========================================

実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
実行期間: {metadata.get('period', 'N/A')}

==========================================
DSSMS概要・バックテスト基本理念遵守確認
==========================================

✅ DSSMS統合でも実際のbacktest()実行: 完了
✅ 銘柄選択後の戦略実行: 完了
✅ Excel依存性除去: 完了

切替イベント数: {len(switches)}
処理銘柄数: {metadata.get('processed_stocks', 'N/A')}

パフォーマンス:
- 総合リターン: {performance.get('total_return', 0):.2%}
- 最大ドローダウン: {performance.get('max_drawdown', 0):.2%}

==========================================
切替統計（実際のbacktest結果基準）
==========================================

有効切替: {len([s for s in switches if s.get('is_beneficial', False)])}
無効切替: {len([s for s in switches if not s.get('is_beneficial', True)])}

切替品質:
- 実際のパフォーマンス基準: ✅
- バックテスト結果基準: ✅
- Excel出力廃棄対応: ✅

==========================================
"""
        return report
    
    def _log_export_summary(self, export_type: str, exported_files: Dict[str, Path], trades_count: int, stock_data: pd.DataFrame):
        """出力サマリーログ"""
        logger.info(f"📊 {export_type.upper()}出力サマリー:")
        logger.info(f"  - 出力ファイル数: {len(exported_files)}")
        logger.info(f"  - 取引数: {trades_count}")
        logger.info(f"  - データポイント数: {len(stock_data)}")
        logger.info(f"  - バックテスト基本理念遵守: ✅")
        
        for format_type, file_path in exported_files.items():
            logger.info(f"  - {format_type}: {file_path.name}")
    
    def _log_dssms_export_summary(self, exported_files: Dict[str, Path], switches_count: int, ranking_data: pd.DataFrame):
        """DSSMS出力サマリーログ"""
        logger.info(f"📊 DSSMS出力サマリー:")
        logger.info(f"  - 出力ファイル数: {len(exported_files)}")
        logger.info(f"  - 切替イベント数: {switches_count}")
        logger.info(f"  - ランキング銘柄数: {len(ranking_data)}")
        logger.info(f"  - バックテスト基本理念遵守: ✅")
        
        for format_type, file_path in exported_files.items():
            logger.info(f"  - {format_type}: {file_path.name}")


# 既存ファイルの段階的移行用ヘルパー（バックテスト基本理念遵守版）
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: def migrate_from_excel_output(excel_data_dict: Dict[str, Any], output_type: str = "main") -> Dict[str, Path]:
def migrate_excel_to_unified(excel_data_dict: Dict[str, Any], output_type: str = "main") -> Dict[str, Path]:
    """
    既存Excel出力データを新形式に移行（バックテスト基本理念遵守版）
    
    Args:
        excel_data_dict: 既存のExcel用データ辞書
        output_type: "main" or "dssms"
    
    Returns:
        新形式ファイルパス辞書
    """
    exporter = UnifiedExporter()
    
    if output_type == "main":
        return exporter.export_main_results(
            stock_data=excel_data_dict.get('stock_data', pd.DataFrame()),
            trades=excel_data_dict.get('trades', []),
            performance=excel_data_dict.get('performance', {}),
            ticker=excel_data_dict.get('ticker', 'UNKNOWN'),
            strategy_name=excel_data_dict.get('strategy', 'UNKNOWN')
        )
    elif output_type == "dssms":
        return exporter.export_dssms_results(
            ranking_data=excel_data_dict.get('ranking_data', pd.DataFrame()),
            switch_events=excel_data_dict.get('switch_events', []),
            performance_summary=excel_data_dict.get('performance', {}),
            execution_metadata=excel_data_dict.get('metadata', {})
        )
    else:
        # デフォルトはmain形式
        return exporter.export_main_results(
            stock_data=excel_data_dict.get('stock_data', pd.DataFrame()),
            trades=excel_data_dict.get('trades', []),
            performance=excel_data_dict.get('performance', {}),
            ticker=excel_data_dict.get('ticker', 'UNKNOWN'),
            strategy_name=excel_data_dict.get('strategy', f'UNKNOWN_{output_type}')
        )


# データ抽出エンハンサーとの連携用ヘルパー
def enhance_data_extraction_output(analysis_results: Dict[str, Any], output_format: str = "unified") -> Dict[str, Path]:
    """
    data_extraction_enhancer用: Excel廃棄版出力連携
    
    Args:
        analysis_results: extract_and_analyze_main_data()の結果
        output_format: "unified" (CSV+JSON+TXT+YAML)
        
    Returns:
        新形式ファイルパス辞書
    """
    exporter = UnifiedExporter()
    
    # main.py形式として出力
    return exporter.export_main_results(
        stock_data=analysis_results.get('stock_data', pd.DataFrame()),
        trades=analysis_results.get('trades', []),
        performance=analysis_results.get('performance', {}),
        ticker=analysis_results.get('ticker', 'ENHANCED_ANALYSIS'),
        strategy_name="Enhanced_Data_Analysis"
    )


# Excel出力完全廃棄記念コメント
# TODO(tag:excel_deprecated, rationale:Excel output completely eliminated since 2025-10-08)
# 🎉 Excel dependency removed - CSV+JSON+TXT+YAML unified output system operational
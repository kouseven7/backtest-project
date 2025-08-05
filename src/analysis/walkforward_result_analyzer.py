"""
ウォークフォワードテスト結果の分析・レポート生成

実行結果を集計・分析し、Excel形式でレポートを生成します。
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.logger_config import setup_logger

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plotting_available = True
except ImportError:
    plotting_available = False
    print("matplotlib/seaborn not available - グラフ機能は無効化されます")

class WalkforwardResultAnalyzer:
    """ウォークフォワードテスト結果の分析クラス"""
    
    def __init__(self, results: Optional[List[Dict[str, Any]]] = None):
        """
        初期化
        
        Args:
            results: ウォークフォワードテスト結果のリスト
        """
        self.logger = setup_logger(__name__)
        self.results = results or []
        self.df = None
        
        if self.results:
            self.df = pd.DataFrame(self.results)
            self.logger.info(f"結果データを読み込みました: {len(self.results)}件")
    
    def load_results(self, results: List[Dict[str, Any]]):
        """結果データを読み込み"""
        self.results = results
        self.df = pd.DataFrame(results)
        self.logger.info(f"結果データを読み込みました: {len(results)}件")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """サマリーレポートを生成"""
        
        if self.df is None or len(self.df) == 0:
            return {"error": "分析対象の結果データがありません"}
        
        try:
            report = {}
            
            # 基本統計
            report["basic_stats"] = self._generate_basic_stats()
            
            # 戦略別分析
            report["strategy_analysis"] = self._analyze_by_strategy()
            
            # 市場状況別分析
            report["market_condition_analysis"] = self._analyze_by_market_condition()
            
            # シンボル別分析
            report["symbol_analysis"] = self._analyze_by_symbol()
            
            # 期間別分析
            report["period_analysis"] = self._analyze_by_period()
            
            # リスク分析
            report["risk_analysis"] = self._analyze_risk_metrics()
            
            # 相関分析
            report["correlation_analysis"] = self._analyze_correlations()
            
            self.logger.info("サマリーレポート生成完了")
            return report
            
        except Exception as e:
            self.logger.error(f"サマリーレポート生成エラー: {e}")
            return {"error": str(e)}
    
    def _generate_basic_stats(self) -> Dict[str, Any]:
        """基本統計情報を生成"""
        
        stats = {
            "total_results": len(self.df),
            "unique_symbols": len(self.df['symbol'].unique()) if 'symbol' in self.df.columns else 0,
            "unique_strategies": len(self.df['strategy'].unique()) if 'strategy' in self.df.columns else 0,
            "unique_periods": len(self.df['period_name'].unique()) if 'period_name' in self.df.columns else 0,
            "date_range": {
                "earliest_start": self.df['period_start'].min() if 'period_start' in self.df.columns else None,
                "latest_end": self.df['period_end'].max() if 'period_end' in self.df.columns else None
            }
        }
        
        # リターンが存在する場合の統計
        if 'total_return' in self.df.columns:
            returns = self.df['total_return'].dropna()
            if len(returns) > 0:
                stats["return_stats"] = {
                    "mean_return": round(returns.mean(), 4),
                    "median_return": round(returns.median(), 4),
                    "std_return": round(returns.std(), 4),
                    "min_return": round(returns.min(), 4),
                    "max_return": round(returns.max(), 4),
                    "positive_rate": round((returns > 0).mean(), 4),
                    "total_observations": len(returns)
                }
        
        return stats
    
    def _analyze_by_strategy(self) -> Dict[str, Any]:
        """戦略別分析"""
        
        if 'strategy' not in self.df.columns:
            return {"error": "戦略列が見つかりません"}
        
        analysis = {}
        
        for strategy in self.df['strategy'].unique():
            strategy_data = self.df[self.df['strategy'] == strategy]
            
            strategy_stats = {
                "total_tests": len(strategy_data),
                "success_rate": 0,
                "avg_return": 0,
                "volatility": 0,
                "max_drawdown_avg": 0,
                "sharpe_ratio_avg": 0
            }
            
            # リターンがある場合
            if 'total_return' in strategy_data.columns:
                returns = strategy_data['total_return'].dropna()
                if len(returns) > 0:
                    strategy_stats["success_rate"] = round((returns > 0).mean(), 4)
                    strategy_stats["avg_return"] = round(returns.mean(), 4)
                    strategy_stats["volatility"] = round(returns.std(), 4)
            
            # その他の指標
            for metric in ['max_drawdown', 'sharpe_ratio']:
                if metric in strategy_data.columns:
                    values = strategy_data[metric].dropna()
                    if len(values) > 0:
                        strategy_stats[f"{metric}_avg"] = round(values.mean(), 4)
            
            analysis[strategy] = strategy_stats
        
        return analysis
    
    def _analyze_by_market_condition(self) -> Dict[str, Any]:
        """市場状況別分析"""
        
        if 'market_condition' not in self.df.columns:
            return {"error": "市場状況列が見つかりません"}
        
        analysis = {}
        
        for condition in self.df['market_condition'].unique():
            condition_data = self.df[self.df['market_condition'] == condition]
            
            condition_stats = {
                "total_tests": len(condition_data),
                "success_rate": 0,
                "avg_return": 0,
                "best_strategy": None,
                "worst_strategy": None
            }
            
            if 'total_return' in condition_data.columns:
                returns = condition_data['total_return'].dropna()
                if len(returns) > 0:
                    condition_stats["success_rate"] = round((returns > 0).mean(), 4)
                    condition_stats["avg_return"] = round(returns.mean(), 4)
                    
                    # 戦略別パフォーマンス
                    if 'strategy' in condition_data.columns:
                        strategy_returns = condition_data.groupby('strategy')['total_return'].mean()
                        if len(strategy_returns) > 0:
                            condition_stats["best_strategy"] = strategy_returns.idxmax()
                            condition_stats["worst_strategy"] = strategy_returns.idxmin()
            
            analysis[condition] = condition_stats
        
        return analysis
    
    def _analyze_by_symbol(self) -> Dict[str, Any]:
        """シンボル別分析"""
        
        if 'symbol' not in self.df.columns:
            return {"error": "シンボル列が見つかりません"}
        
        # 上位/下位パフォーマーのみ報告（全シンボルは多すぎる可能性）
        symbol_performance = {}
        
        if 'total_return' in self.df.columns:
            symbol_returns = self.df.groupby('symbol')['total_return'].agg(['mean', 'count']).round(4)
            symbol_returns = symbol_returns[symbol_returns['count'] >= 3]  # 最低3回のテスト
            
            if len(symbol_returns) > 0:
                # 上位5位
                top_performers = symbol_returns.nlargest(5, 'mean')
                symbol_performance["top_performers"] = top_performers.to_dict('index')
                
                # 下位5位
                bottom_performers = symbol_returns.nsmallest(5, 'mean')
                symbol_performance["bottom_performers"] = bottom_performers.to_dict('index')
        
        symbol_performance["total_symbols_tested"] = len(self.df['symbol'].unique())
        
        return symbol_performance
    
    def _analyze_by_period(self) -> Dict[str, Any]:
        """期間別分析"""
        
        if 'period_name' not in self.df.columns:
            return {"error": "期間名列が見つかりません"}
        
        analysis = {}
        
        for period in self.df['period_name'].unique():
            period_data = self.df[self.df['period_name'] == period]
            
            period_stats = {
                "total_tests": len(period_data),
                "success_rate": 0,
                "avg_return": 0
            }
            
            if 'total_return' in period_data.columns:
                returns = period_data['total_return'].dropna()
                if len(returns) > 0:
                    period_stats["success_rate"] = round((returns > 0).mean(), 4)
                    period_stats["avg_return"] = round(returns.mean(), 4)
            
            analysis[period] = period_stats
        
        return analysis
    
    def _analyze_risk_metrics(self) -> Dict[str, Any]:
        """リスク指標分析"""
        
        risk_analysis = {}
        
        # ボラティリティ分析
        if 'volatility' in self.df.columns:
            vol_data = self.df['volatility'].dropna()
            if len(vol_data) > 0:
                risk_analysis["volatility_stats"] = {
                    "mean": round(vol_data.mean(), 4),
                    "median": round(vol_data.median(), 4),
                    "std": round(vol_data.std(), 4),
                    "high_vol_threshold": round(vol_data.quantile(0.75), 4)
                }
        
        # ドローダウン分析
        if 'max_drawdown' in self.df.columns:
            dd_data = self.df['max_drawdown'].dropna()
            if len(dd_data) > 0:
                risk_analysis["drawdown_stats"] = {
                    "mean": round(dd_data.mean(), 4),
                    "median": round(dd_data.median(), 4),
                    "worst": round(dd_data.min(), 4),
                    "acceptable_threshold": round(dd_data.quantile(0.25), 4)
                }
        
        # シャープレシオ分析
        if 'sharpe_ratio' in self.df.columns:
            sharpe_data = self.df['sharpe_ratio'].dropna()
            if len(sharpe_data) > 0:
                risk_analysis["sharpe_stats"] = {
                    "mean": round(sharpe_data.mean(), 4),
                    "median": round(sharpe_data.median(), 4),
                    "excellent_threshold": round(sharpe_data.quantile(0.75), 4),
                    "good_rate": round((sharpe_data > 1.0).mean(), 4)
                }
        
        return risk_analysis
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """相関分析"""
        
        correlation_analysis = {}
        
        # 数値列を抽出
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) >= 2:
            try:
                correlation_matrix = self.df[numeric_columns].corr().round(4)
                
                # 重要な相関関係を抽出
                important_correlations = {}
                
                if 'total_return' in correlation_matrix.columns:
                    return_correlations = correlation_matrix['total_return'].drop('total_return')
                    # 強い相関（絶対値0.3以上）を抽出
                    strong_correlations = return_correlations[abs(return_correlations) >= 0.3]
                    if len(strong_correlations) > 0:
                        important_correlations["return_correlations"] = strong_correlations.to_dict()
                
                correlation_analysis["important_correlations"] = important_correlations
                correlation_analysis["correlation_matrix_shape"] = correlation_matrix.shape
                
            except Exception as e:
                correlation_analysis["error"] = f"相関分析エラー: {e}"
        
        return correlation_analysis
    
    def export_to_excel(self, output_path: str) -> bool:
        """結果をExcelファイルに出力"""
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                
                # 生データ
                if self.df is not None:
                    self.df.to_excel(writer, sheet_name='Raw_Data', index=False)
                
                # サマリーレポート
                summary = self.generate_summary_report()
                
                # 基本統計
                if 'basic_stats' in summary:
                    basic_df = pd.json_normalize(summary['basic_stats']).T
                    basic_df.to_excel(writer, sheet_name='Basic_Stats')
                
                # 戦略別分析
                if 'strategy_analysis' in summary:
                    strategy_df = pd.DataFrame(summary['strategy_analysis']).T
                    strategy_df.to_excel(writer, sheet_name='Strategy_Analysis')
                
                # 市場状況別分析
                if 'market_condition_analysis' in summary:
                    market_df = pd.DataFrame(summary['market_condition_analysis']).T
                    market_df.to_excel(writer, sheet_name='Market_Analysis')
                
                # 期間別分析
                if 'period_analysis' in summary:
                    period_df = pd.DataFrame(summary['period_analysis']).T
                    period_df.to_excel(writer, sheet_name='Period_Analysis')
            
            self.logger.info(f"Excelファイル出力完了: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Excel出力エラー: {e}")
            return False
    
    def generate_performance_charts(self, output_dir: str) -> bool:
        """パフォーマンスチャートを生成"""
        
        if not plotting_available:
            self.logger.warning("matplotlib/seaborn が利用できません - チャート生成をスキップ")
            return False
        
        if self.df is None or len(self.df) == 0:
            self.logger.warning("チャート生成用のデータがありません")
            return False
        
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 戦略別リターン分布
            if 'strategy' in self.df.columns and 'total_return' in self.df.columns:
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=self.df, x='strategy', y='total_return')
                plt.title('Strategy Performance Distribution')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'strategy_performance_distribution.png')
                plt.close()
            
            # 市場状況別パフォーマンス
            if 'market_condition' in self.df.columns and 'total_return' in self.df.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=self.df, x='market_condition', y='total_return')
                plt.title('Performance by Market Condition')
                plt.tight_layout()
                plt.savefig(output_dir / 'market_condition_performance.png')
                plt.close()
            
            # リスク・リターン散布図
            if all(col in self.df.columns for col in ['total_return', 'volatility', 'strategy']):
                plt.figure(figsize=(10, 8))
                for strategy in self.df['strategy'].unique():
                    strategy_data = self.df[self.df['strategy'] == strategy]
                    plt.scatter(strategy_data['volatility'], strategy_data['total_return'], 
                              label=strategy, alpha=0.7)
                plt.xlabel('Volatility')
                plt.ylabel('Total Return')
                plt.title('Risk-Return Profile by Strategy')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'risk_return_scatter.png')
                plt.close()
            
            self.logger.info(f"パフォーマンスチャート生成完了: {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"チャート生成エラー: {e}")
            return False
    
    def get_best_configurations(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """最高パフォーマンスの設定を取得"""
        
        if self.df is None or len(self.df) == 0:
            return []
        
        if 'total_return' not in self.df.columns:
            return []
        
        # リターンでソート
        sorted_results = self.df.sort_values('total_return', ascending=False)
        
        # 上位N件を取得
        top_results = sorted_results.head(top_n)
        
        best_configs = []
        for _, row in top_results.iterrows():
            config = {
                "rank": len(best_configs) + 1,
                "symbol": row.get('symbol', 'N/A'),
                "strategy": row.get('strategy', 'N/A'),
                "period": row.get('period_name', 'N/A'),
                "market_condition": row.get('market_condition', 'N/A'),
                "total_return": row.get('total_return', 0),
                "volatility": row.get('volatility', 0),
                "sharpe_ratio": row.get('sharpe_ratio', 0),
                "max_drawdown": row.get('max_drawdown', 0)
            }
            best_configs.append(config)
        
        return best_configs

if __name__ == "__main__":
    # テスト用のダミーデータ
    dummy_results = [
        {
            "symbol": "AAPL", "strategy": "VWAPBreakout", "period_name": "2020_covid",
            "market_condition": "downtrend", "total_return": 5.2, "volatility": 2.1,
            "sharpe_ratio": 2.48, "max_drawdown": -3.1
        },
        {
            "symbol": "MSFT", "strategy": "Momentum", "period_name": "2021_recovery", 
            "market_condition": "uptrend", "total_return": 8.7, "volatility": 1.8,
            "sharpe_ratio": 4.83, "max_drawdown": -1.2
        }
    ]
    
    analyzer = WalkforwardResultAnalyzer(dummy_results)
    summary = analyzer.generate_summary_report()
    
    print("=== ウォークフォワード結果分析テスト ===")
    print(f"基本統計: {summary.get('basic_stats', {})}")
    print(f"戦略分析: {summary.get('strategy_analysis', {})}")
    
    best_configs = analyzer.get_best_configurations(top_n=2)
    print(f"\n最高パフォーマンス設定:")
    for config in best_configs:
        print(f"  {config['rank']}位: {config['strategy']} - {config['total_return']}%")

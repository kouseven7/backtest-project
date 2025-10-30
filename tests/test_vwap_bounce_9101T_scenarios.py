"""
VWAP_Bounce戦略 3シナリオ比較テスト（9101.T 2023-2024データ）

3つのパラメータセットを比較:
1. デフォルト（range-bound）: 元の設計意図
2. 条件緩和（range-bound）: レンジ相場で実用的なエントリー頻度
3. トレンドフィルターOFF: 全トレンドでVWAPバウンス検出

主な機能:
- 複数パラメータセットの自動テスト実行
- シナリオ別結果の詳細ログ出力
- 3シナリオの比較レポート生成
- エントリー頻度・勝率・損益の横比較

統合コンポーネント:
- tests.test_vwap_bounce_8306T: ベーステストランナー
- strategies.VWAP_Bounce: テスト対象戦略

セーフティ機能/注意事項:
- copilot-instructions.md準拠（モックデータ禁止）
- strategies/VWAP_Bounce.pyは修正しない
- パラメータはテストスクリプト内で上書き
- 実データ（yfinance）のみ使用

Author: Backtest Project Team
Created: 2025-10-30
Last Modified: 2025-10-30
"""
import sys
from pathlib import Path
from typing import Dict, Any, List

# プロジェクトルート設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 元のテストランナーをインポート
from tests.test_vwap_bounce_8306T import VWAPBounceTestRunner

# 戦略インポート
from strategies.VWAP_Bounce import VWAPBounceStrategy

# ロガー設定
from config.logger_config import setup_logger

logger = setup_logger(__name__)


class VWAPBounceScenarioTestRunner(VWAPBounceTestRunner):
    """VWAP_Bounce戦略 3シナリオ比較テストランナー"""
    
    def __init__(self):
        """初期化（9101.T用）"""
        super().__init__(
            ticker="9101.T",
            index_ticker="^N225"
        )
        
        # シナリオ結果格納用
        self.scenario_results: List[Dict[str, Any]] = []
    
    def run_scenario(self, scenario_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        1シナリオ実行
        
        Parameters:
            scenario_name: シナリオ名
            params: カスタムパラメータ（Noneの場合デフォルト使用）
            
        Returns:
            Dict[str, Any]: シナリオ結果
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"シナリオ開始: {scenario_name}")
        self.logger.info("=" * 80)
        
        # パラメータログ出力
        if params:
            self.logger.info("カスタムパラメータ:")
            for key, value in params.items():
                self.logger.info(f"  {key}: {value}")
        else:
            self.logger.info("デフォルトパラメータ使用")
        
        try:
            # 戦略初期化（パラメータ上書き）
            self.strategy = VWAPBounceStrategy(
                data=self.stock_data.copy(),
                params=params,  # カスタムパラメータまたはNone
                price_column="Adj Close",
                volume_column="Volume"
            )
            
            # 戦略初期化
            self.strategy.initialize_strategy()
            self.logger.info("戦略初期化完了")
            
            # バックテスト実行
            self.logger.info("バックテスト実行中...")
            self.result_data = self.strategy.backtest()
            
            if self.result_data is None:
                self.logger.error("バックテスト結果がNullです")
                return {}
            
            self.logger.info("バックテスト実行完了")
            
            # 検証実行
            signals = self.validate_signals()
            vwap_logic = self.validate_vwap_logic()
            trend = self.validate_trend_detection()
            trend_dist = self.analyze_trend_distribution()
            performance = self.calculate_performance()
            
            # シナリオ結果構築
            result = {
                "scenario_name": scenario_name,
                "params": params if params else "default",
                "entry_count": signals.get('entry_count', 0),
                "exit_count": signals.get('exit_count', 0),
                "total_trades": performance.get('total_trades', 0),
                "total_pnl": performance.get('total_pnl', 0),
                "total_pnl_pct": performance.get('total_pnl_pct', 0.0),
                "win_rate": performance.get('win_rate', 0.0),
                "avg_hold_days": performance.get('avg_hold_days', 0.0),
                "vwap_candidates": trend_dist.get('vwap_candidates', 0),
                "trend_counts": trend_dist.get('trend_counts', {}),
                "trades": performance.get('trades', [])
            }
            
            self.logger.info("\n--- シナリオ結果サマリー ---")
            self.logger.info(f"エントリー回数: {result['entry_count']}")
            self.logger.info(f"総取引回数: {result['total_trades']}")
            self.logger.info(f"総損益: {result['total_pnl']:,.0f}円 ({result['total_pnl_pct']:.2f}%)")
            self.logger.info(f"勝率: {result['win_rate']:.2f}%")
            self.logger.info(f"平均保有期間: {result['avg_hold_days']:.1f}日")
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"シナリオ完了: {scenario_name}")
            self.logger.info("=" * 80 + "\n")
            
            return result
            
        except Exception as e:
            self.logger.error(f"シナリオ実行中にエラー発生: {e}", exc_info=True)
            return {}
    
    def run_all_scenarios(self) -> bool:
        """
        3シナリオすべて実行
        
        Returns:
            bool: 実行成功=True
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("VWAP_Bounce戦略 3シナリオ比較テスト開始")
        self.logger.info("銘柄: 9101.T（日本郵船）")
        self.logger.info("期間: 2023/01/01 ~ 2024/12/31（2年間）")
        self.logger.info("=" * 80 + "\n")
        
        try:
            # Phase 1: データ取得（共通）
            if not self.fetch_data(start_date="2023-01-01", end_date="2024-12-31"):
                self.logger.error("データ取得失敗")
                return False
            
            # Phase 2: 3シナリオ定義
            scenarios = [
                {
                    "name": "シナリオ1: デフォルト（range-bound）",
                    "params": None  # デフォルト使用
                },
                {
                    "name": "シナリオ2: 条件緩和（range-bound）",
                    "params": {
                        "vwap_lower_threshold": 0.98,  # -2%に緩和
                        "vwap_upper_threshold": 1.02,  # 変更なし
                        "volume_increase_threshold": 1.1,  # 1.1倍に緩和
                        "bullish_candle_min_pct": 0.003,  # 0.3%に緩和
                        "allowed_trends": ["range-bound"]  # range-boundのみ
                    }
                },
                {
                    "name": "シナリオ3: トレンドフィルターOFF",
                    "params": {
                        "vwap_lower_threshold": 0.98,  # -2%に緩和
                        "vwap_upper_threshold": 1.02,  # 変更なし
                        "volume_increase_threshold": 1.1,  # 1.1倍に緩和
                        "bullish_candle_min_pct": 0.003,  # 0.3%に緩和
                        "trend_filter_enabled": False  # トレンドフィルターOFF
                    }
                }
            ]
            
            # Phase 3: 各シナリオ実行
            for scenario in scenarios:
                result = self.run_scenario(
                    scenario_name=scenario["name"],
                    params=scenario["params"]
                )
                
                if result:
                    self.scenario_results.append(result)
            
            # Phase 4: 比較レポート生成
            self.generate_comparison_report()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("VWAP_Bounce戦略 3シナリオ比較テスト完了")
            self.logger.info("=" * 80 + "\n")
            
            return True
            
        except Exception as e:
            self.logger.error(f"テスト実行中にエラー発生: {e}", exc_info=True)
            return False
    
    def generate_comparison_report(self):
        """3シナリオ比較レポート生成"""
        if not self.scenario_results:
            self.logger.warning("シナリオ結果が存在しません")
            return
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("3シナリオ比較レポート")
        self.logger.info("=" * 80 + "\n")
        
        # テーブルヘッダー
        self.logger.info(f"{'シナリオ':<30} {'エントリー':>10} {'取引数':>8} {'総損益':>12} {'損益率':>10} {'勝率':>8} {'保有日数':>10}")
        self.logger.info("-" * 95)
        
        # 各シナリオ結果
        for result in self.scenario_results:
            scenario_name = result['scenario_name'].replace("シナリオ", "S")
            entry_count = result['entry_count']
            total_trades = result['total_trades']
            total_pnl = result['total_pnl']
            total_pnl_pct = result['total_pnl_pct']
            win_rate = result['win_rate']
            avg_hold_days = result['avg_hold_days']
            
            self.logger.info(
                f"{scenario_name:<30} "
                f"{entry_count:>10} "
                f"{total_trades:>8} "
                f"{total_pnl:>12,.0f}円 "
                f"{total_pnl_pct:>8.2f}% "
                f"{win_rate:>7.1f}% "
                f"{avg_hold_days:>9.1f}日"
            )
        
        self.logger.info("\n" + "-" * 80)
        
        # 詳細分析
        self.logger.info("\n--- 詳細分析 ---\n")
        
        for i, result in enumerate(self.scenario_results, 1):
            self.logger.info(f"{i}. {result['scenario_name']}")
            
            # トレンド分布
            trend_counts = result.get('trend_counts', {})
            if trend_counts:
                total_days = sum(trend_counts.values())
                self.logger.info("  トレンド分布:")
                for trend, count in trend_counts.items():
                    pct = (count / total_days * 100) if total_days > 0 else 0
                    self.logger.info(f"    {trend}: {count}日 ({pct:.1f}%)")
            
            # VWAP条件該当日
            vwap_candidates = result.get('vwap_candidates', 0)
            self.logger.info(f"  VWAP条件該当日: {vwap_candidates}日")
            
            # 取引詳細
            trades = result.get('trades', [])
            if trades:
                self.logger.info(f"  取引詳細（最大5件）:")
                for j, trade in enumerate(trades[:5], 1):
                    self.logger.info(
                        f"    {j}. "
                        f"エントリー: {trade['entry_date']} {trade['entry_price']:.2f}円, "
                        f"イグジット: {trade['exit_date']} {trade['exit_price']:.2f}円, "
                        f"損益: {trade['pnl_yen']:,.0f}円 ({trade['pnl_pct']:.2f}%), "
                        f"保有: {trade['hold_days']}日"
                    )
            else:
                self.logger.info("  取引なし")
            
            self.logger.info("")
        
        # 推奨事項
        self.logger.info("\n" + "=" * 80)
        self.logger.info("推奨事項")
        self.logger.info("=" * 80 + "\n")
        
        # 最もエントリーが多いシナリオ
        max_entries = max(r['entry_count'] for r in self.scenario_results)
        best_entry_scenario = [r for r in self.scenario_results if r['entry_count'] == max_entries][0]
        
        # 最も損益が高いシナリオ
        max_pnl = max(r['total_pnl'] for r in self.scenario_results)
        best_pnl_scenario = [r for r in self.scenario_results if r['total_pnl'] == max_pnl][0]
        
        self.logger.info(f"エントリー頻度が最も高い: {best_entry_scenario['scenario_name']}")
        self.logger.info(f"  エントリー回数: {best_entry_scenario['entry_count']}回")
        self.logger.info(f"  月間平均: {best_entry_scenario['entry_count'] / 24:.1f}回/月")
        
        self.logger.info(f"\n総損益が最も高い: {best_pnl_scenario['scenario_name']}")
        self.logger.info(f"  総損益: {best_pnl_scenario['total_pnl']:,.0f}円 ({best_pnl_scenario['total_pnl_pct']:.2f}%)")
        self.logger.info(f"  勝率: {best_pnl_scenario['win_rate']:.2f}%")
        
        # 実用性評価
        self.logger.info("\n--- 実用性評価 ---")
        for result in self.scenario_results:
            entry_count = result['entry_count']
            win_rate = result['win_rate']
            total_pnl_pct = result['total_pnl_pct']
            
            # 月間エントリー頻度（2年=24ヶ月）
            monthly_entries = entry_count / 24
            
            # 実用性スコア
            if entry_count == 0:
                score = "不可（エントリーなし）"
            elif monthly_entries < 0.5:
                score = "低（エントリー頻度不足）"
            elif win_rate < 30:
                score = "低（勝率不足）"
            elif total_pnl_pct < 0:
                score = "低（損失）"
            elif monthly_entries >= 1 and win_rate >= 40 and total_pnl_pct > 0:
                score = "高（実用的）"
            else:
                score = "中（要改善）"
            
            self.logger.info(f"{result['scenario_name']}: {score}")
        
        self.logger.info("\n" + "=" * 80 + "\n")


def main():
    """メイン実行関数"""
    # テストランナー作成
    test_runner = VWAPBounceScenarioTestRunner()
    
    # 3シナリオ実行
    success = test_runner.run_all_scenarios()
    
    # 終了コード
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

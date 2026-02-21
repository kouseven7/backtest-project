"""
エグジット戦略単体検証スクリプト（Phase 4拡張版）

単一戦略・単一銘柄で、エグジットのみを差し替えて比較検証。

主な機能:
- GCエントリー固定
- 複数エグジット戦略の比較
- CSV出力（PF, Sharpe, 取引数等）
- 日次バックテスト対応
- Phase 4拡張: Sharpe Ratio詳細計算、Max Drawdown分析、累積PL曲線可視化

統合コンポーネント:
- GCStrategyWithExit
- BaseExitStrategy派生クラス群
- data_fetcher経由でデータ取得
- matplotlib経由で可視化

セーフティ機能/注意事項:
- 銘柄・期間固定（再現性担保）
- エントリーロジック変更禁止
- ルックアヘッドバイアス防止（copilot-instructions.md準拠）
- Sharpe Ratio: 年率換算（250営業日基準）
- Max Drawdown: 累積PL基準で計算

Author: Backtest Project Team
Created: 2026-01-22
Last Modified: 2026-01-22
"""
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド指定

# プロジェクトルートをパス追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.gc_strategy_with_exit import GCStrategyWithExit
from strategies.exit_strategies.trailing_stop_exit import TrailingStopExit
from strategies.exit_strategies.take_profit_exit import TakeProfitExit
from strategies.exit_strategies.fixed_stop_loss_exit import FixedStopLossExit
from strategies.exit_strategies.composite_exit import CompositeExit
from strategies.exit_strategies.trend_following_exit import TrendFollowingExit
from data_fetcher import get_parameters_and_data


def plot_cumulative_pl_curves(all_results: dict, output_dir: Path, ticker: str):
    """
    累積PL曲線可視化（Phase 4実装）
    
    Args:
        all_results: {戦略名: {'cumulative_pl': [累積PL配列], 'metrics': {...}}}
        output_dir: 出力ディレクトリ
        ticker: 銘柄コード
    
    Note:
        - 全戦略の累積PLを1つのグラフに重ねて表示
        - PNG形式で保存
        - 上位5戦略のみプロット（視認性確保）
    """
    print(f"\n[VISUALIZATION] 累積PL曲線作成中...")
    
    # 有効な戦略のみフィルタ（累積PLデータあり）
    valid_strategies = {
        name: data for name, data in all_results.items()
        if len(data['cumulative_pl']) > 0
    }
    
    if len(valid_strategies) == 0:
        print(f"[WARNING] 累積PLデータが存在しないためスキップ")
        return
    
    # PF上位5戦略を選択
    sorted_strategies = sorted(
        valid_strategies.items(),
        key=lambda x: x[1]['metrics']['profit_factor'],
        reverse=True
    )[:5]
    
    # プロット作成
    plt.figure(figsize=(14, 8))
    
    for strategy_name, data in sorted_strategies:
        cumulative_pl = data['cumulative_pl']
        pf = data['metrics']['profit_factor']
        win_rate = data['metrics']['win_rate']
        
        # トレード番号（x軸）
        trade_numbers = list(range(1, len(cumulative_pl) + 1))
        
        # プロット
        plt.plot(
            trade_numbers,
            cumulative_pl,
            label=f"{strategy_name} (PF={pf:.2f}, WR={win_rate:.1%})",
            linewidth=2
        )
    
    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Cumulative Profit/Loss (JPY)', fontsize=12)
    plt.title(f'Cumulative P/L Curves - Top 5 Exit Strategies ({ticker})', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"cumulative_pl_curves_{ticker.replace('.', '_')}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] 累積PL曲線保存: {output_path}")


def calculate_performance_metrics(results_df: pd.DataFrame, strategy_name: str = None) -> dict:
    """
    パフォーマンス指標計算（Phase 4拡張版）
    
    Args:
        results_df: バックテスト結果DataFrame
        strategy_name: 戦略名（可視化用）
    
    Returns:
        パフォーマンス指標辞書
    
    Note:
        - BaseStrategy.backtest()の戻り値は'Entry_Signal'/'Exit_Signal'列を含むDataFrame
        - Phase 4拡張: 年率換算Sharpe Ratio、詳細Max Drawdown、累積PL曲線データ
    """
    if len(results_df) == 0:
        return {
            'total_trades': 0,
            'entry_signals': 0,
            'exit_signals': 0,
            'total_return': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'sharpe_ratio_annualized': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'cumulative_pl': []
        }
    
    # エントリー/エグジット数を計算
    entry_signals = len(results_df[results_df.get('Entry_Signal', 0) == 1])
    exit_signals = len(results_df[results_df.get('Exit_Signal', 0) == -1])
    
    # Profit_Loss列が存在する場合のみ計算
    if 'Profit_Loss' in results_df.columns:
        # 基本統計
        total_trades = len(results_df)
        total_return = results_df['Profit_Loss'].sum()
        
        # プロフィットファクター
        gross_profit = results_df[results_df['Profit_Loss'] > 0]['Profit_Loss'].sum()
        gross_loss = abs(results_df[results_df['Profit_Loss'] < 0]['Profit_Loss'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # 勝率
        win_trades = len(results_df[results_df['Profit_Loss'] > 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0.0
        
        # Phase 4拡張: シャープレシオ（年率換算）
        # 前提: 取引ごとのリターンを年率換算（250営業日基準）
        returns = results_df['Profit_Loss']
        if returns.std() > 0 and total_trades > 1:
            # トレード単位のシャープレシオ
            sharpe_ratio = returns.mean() / returns.std()
            # 年率換算: sqrt(250 / 平均保有日数)で調整
            # 簡易版: 取引数ベースで年率換算 sqrt(250 / (取引期間日数 / 取引数))
            trading_days = len(results_df)
            trades_per_year = 250 / (trading_days / total_trades) if trading_days > 0 else 1
            sharpe_ratio_annualized = sharpe_ratio * np.sqrt(trades_per_year)
        else:
            sharpe_ratio = 0.0
            sharpe_ratio_annualized = 0.0
        
        # Phase 4拡張: 最大ドローダウン（累積PL基準）
        cumulative_returns = returns.cumsum()
        running_max = cumulative_returns.cummax()
        drawdown = cumulative_returns - running_max
        max_drawdown = drawdown.min()
        
        # 最大ドローダウン率（初期資金を100万円と仮定）
        initial_capital = 1_000_000
        max_drawdown_pct = (max_drawdown / initial_capital) * 100 if initial_capital > 0 else 0.0
        
        # 累積PL曲線データ（可視化用）
        cumulative_pl = cumulative_returns.tolist()
    else:
        # Profit_Loss列がない場合はN/A
        total_trades = exit_signals  # エグジット数=取引数
        total_return = 0.0
        profit_factor = 0.0
        win_rate = 0.0
        sharpe_ratio = 0.0
        sharpe_ratio_annualized = 0.0
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
        cumulative_pl = []
    
    return {
        'total_trades': total_trades,
        'entry_signals': entry_signals,
        'exit_signals': exit_signals,
        'total_return': total_return,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'sharpe_ratio_annualized': sharpe_ratio_annualized,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'cumulative_pl': cumulative_pl
    }


def validate_exit_strategies(
    ticker: str = "7203.T",
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    warmup_days: int = 150
):
    """
    エグジット戦略比較検証
    
    Args:
        ticker: 検証銘柄（デフォルト: トヨタ自動車7203.T）
        start_date: 開始日
        end_date: 終了日
        warmup_days: ウォームアップ期間
    """
    print("\n" + "=" * 80)
    print("エグジット戦略単体検証")
    print("=" * 80)
    print(f"銘柄: {ticker}")
    print(f"期間: {start_date} ~ {end_date}")
    print(f"ウォームアップ: {warmup_days}日")
    print("=" * 80 + "\n")
    
    # データ取得
    print("[STEP 1/3] データ取得中...")
    try:
        _, _, _, stock_data, index_data = get_parameters_and_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            warmup_days=warmup_days
        )
        print(f"[OK] データ取得完了: {len(stock_data)} rows")
    except Exception as e:
        print(f"[ERROR] データ取得エラー: {e}")
        return
    
    # エグジット戦略リスト（Phase 1-3: 全戦略 + 組み合わせ）
    exit_strategies = [
        # Phase 1: TrailingStop
        TrailingStopExit(trailing_stop_pct=0.03),  # 3%
        TrailingStopExit(trailing_stop_pct=0.05),  # 5%
        TrailingStopExit(trailing_stop_pct=0.08),  # 8%
        
        # Phase 2: TakeProfit
        TakeProfitExit(take_profit_pct=0.10),      # 10%
        TakeProfitExit(take_profit_pct=0.15),      # 15%
        TakeProfitExit(take_profit_pct=0.20),      # 20%
        
        # Phase 2: FixedStopLoss
        FixedStopLossExit(stop_loss_pct=0.02),     # 2%
        FixedStopLossExit(stop_loss_pct=0.03),     # 3%
        FixedStopLossExit(stop_loss_pct=0.05),     # 5%
        
        # Phase 2: 組み合わせ戦略
        CompositeExit(
            strategies=[
                TrailingStopExit(trailing_stop_pct=0.05),
                TakeProfitExit(take_profit_pct=0.15)
            ],
            name="TrailingStop(5%) + TakeProfit(15%)"
        ),
        CompositeExit(
            strategies=[
                TrailingStopExit(trailing_stop_pct=0.05),
                FixedStopLossExit(stop_loss_pct=0.03)
            ],
            name="TrailingStop(5%) + StopLoss(3%)"
        ),
        
        # Phase 3: TrendFollowing
        TrendFollowingExit(min_hold_days=3, max_hold_days=60, confidence_threshold=0.5),
        TrendFollowingExit(min_hold_days=5, max_hold_days=90, confidence_threshold=0.6),
        TrendFollowingExit(min_hold_days=1, max_hold_days=30, confidence_threshold=0.4),
    ]
    
    results = []
    all_results = {}  # Phase 4: 可視化用データ保存
    
    print(f"\n[STEP 2/3] バックテスト実行中...")
    for exit_strategy in exit_strategies:
        print(f"\n{'=' * 80}")
        print(f"[TEST] {exit_strategy}")
        print(f"{'=' * 80}")
        
        try:
            # 戦略実行
            strategy = GCStrategyWithExit(
                data=stock_data.copy(),
                exit_strategy=exit_strategy,
                ticker=ticker
            )
            
            # バックテスト実行（全期間）
            backtest_results = strategy.backtest()
            
            # Phase 4拡張: パフォーマンス指標計算
            metrics = calculate_performance_metrics(backtest_results, str(exit_strategy))
            
            # 結果出力
            print(f"\n[RESULTS]")
            print(f"  エントリー数: {metrics['entry_signals']}")
            print(f"  エグジット数: {metrics['exit_signals']}")
            print(f"  取引数: {metrics['total_trades']}")
            
            if metrics['total_return'] > 0 or 'Profit_Loss' in backtest_results.columns:
                print(f"  総損益: {metrics['total_return']:.2f}円")
                print(f"  プロフィットファクター: {metrics['profit_factor']:.2f}")
                print(f"  勝率: {metrics['win_rate']:.1%}")
                print(f"  シャープレシオ: {metrics['sharpe_ratio']:.2f}")
                print(f"  シャープレシオ(年率): {metrics['sharpe_ratio_annualized']:.2f}")
                print(f"  最大ドローダウン: {metrics['max_drawdown']:.2f}円 ({metrics['max_drawdown_pct']:.2f}%)")
            else:
                print(f"  [NOTE] Profit_Loss列が存在しないため、詳細指標は計算されません")
            
            # Phase 4: 可視化用データ保存
            all_results[exit_strategy.name] = {
                'cumulative_pl': metrics['cumulative_pl'],
                'metrics': metrics
            }
            
            # 結果集計（汎用パラメータ取得）
            strategy_params = {}
            if hasattr(exit_strategy, 'trailing_stop_pct'):
                strategy_params['trailing_stop_pct'] = exit_strategy.trailing_stop_pct
            if hasattr(exit_strategy, 'take_profit_pct'):
                strategy_params['take_profit_pct'] = exit_strategy.take_profit_pct
            if hasattr(exit_strategy, 'stop_loss_pct'):
                strategy_params['stop_loss_pct'] = exit_strategy.stop_loss_pct
            if hasattr(exit_strategy, 'min_hold_days'):
                strategy_params['min_hold_days'] = exit_strategy.min_hold_days
            if hasattr(exit_strategy, 'max_hold_days'):
                strategy_params['max_hold_days'] = exit_strategy.max_hold_days
            if hasattr(exit_strategy, 'confidence_threshold'):
                strategy_params['confidence_threshold'] = exit_strategy.confidence_threshold
            
            results.append({
                'exit_strategy': exit_strategy.name,
                **strategy_params,
                'entry_signals': metrics['entry_signals'],
                'exit_signals': metrics['exit_signals'],
                'total_trades': metrics['total_trades'],
                'total_return': metrics['total_return'],
                'profit_factor': metrics['profit_factor'],
                'win_rate': metrics['win_rate'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sharpe_ratio_annualized': metrics['sharpe_ratio_annualized'],
                'max_drawdown': metrics['max_drawdown'],
                'max_drawdown_pct': metrics['max_drawdown_pct']
            })
        
        except Exception as e:
            print(f"[ERROR] バックテストエラー: {e}")
            import traceback
            traceback.print_exc()
    
    # 結果出力
    print(f"\n[STEP 3/4] 結果出力中...")
    results_df = pd.DataFrame(results)
    
    # 出力ディレクトリ作成
    output_dir = project_root / "output" / "exit_strategy_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSVファイル出力
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"exit_strategy_comparison_{ticker.replace('.', '_')}_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n[OK] 結果出力完了: {csv_path}")
    
    # Phase 4: 累積PL曲線可視化
    print(f"\n[STEP 4/4] 累積PL曲線可視化中...")
    plot_cumulative_pl_curves(all_results, output_dir, ticker)
    
    # サマリー出力
    print(f"\n{'=' * 80}")
    print("検証結果サマリー（Phase 4拡張版）")
    print(f"{'=' * 80}")
    print(results_df.to_string(index=False))
    print(f"{'=' * 80}\n")
    
    # 最適戦略判定（PF基準）
    if len(results_df) > 0 and results_df['total_trades'].sum() > 0:
        if results_df['profit_factor'].max() > 0:
            best_strategy = results_df.loc[results_df['profit_factor'].idxmax()]
            print(f"[BEST] プロフィットファクター最高:")
            print(f"  戦略: {best_strategy['exit_strategy']}")
            
            # パラメータ表示（存在する場合のみ）
            if 'trailing_stop_pct' in best_strategy and best_strategy['trailing_stop_pct'] > 0:
                print(f"  トレーリングストップ: {best_strategy['trailing_stop_pct']:.1%}")
            if 'take_profit_pct' in best_strategy and best_strategy['take_profit_pct'] > 0:
                print(f"  利確率: {best_strategy['take_profit_pct']:.1%}")
            if 'stop_loss_pct' in best_strategy and best_strategy['stop_loss_pct'] > 0:
                print(f"  損切率: {best_strategy['stop_loss_pct']:.1%}")
            if 'min_hold_days' in best_strategy and best_strategy['min_hold_days'] > 0:
                print(f"  最低保有期間: {best_strategy['min_hold_days']}日")
            if 'max_hold_days' in best_strategy and best_strategy['max_hold_days'] > 0:
                print(f"  最大保有期間: {best_strategy['max_hold_days']}日")
            if 'confidence_threshold' in best_strategy and best_strategy['confidence_threshold'] > 0:
                print(f"  信頼度閾値: {best_strategy['confidence_threshold']:.2f}")
            
            print(f"  PF: {best_strategy['profit_factor']:.2f}")
            print(f"  総損益: {best_strategy['total_return']:.2f}円")
            print(f"  Sharpe Ratio (年率): {best_strategy['sharpe_ratio_annualized']:.2f}")
            print(f"  Max Drawdown: {best_strategy['max_drawdown']:.2f}円 ({best_strategy['max_drawdown_pct']:.2f}%)")
    
    return results_df


if __name__ == "__main__":
    # デフォルト設定で検証実行
    validate_exit_strategies(
        ticker="7203.T",
        start_date="2023-01-01",
        end_date="2024-12-31",
        warmup_days=150
    )

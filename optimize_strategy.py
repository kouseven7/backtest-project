"""
戦略のパラメータ最適化を実行するスクリプト
"""
import argparse
import logging
import os
import sys
from datetime import datetime

# 戦略クラスのインポート
from strategies.Breakout import BreakoutStrategy
from strategies.contrarian_strategy import ContrarianStrategy
from strategies.gc_strategy_signal import GCStrategy
from strategies.Momentum_Investing import MomentumInvestingStrategy  # 追加インポート
from strategies.Opening_Gap import OpeningGapStrategy  # 追加インポート
from strategies.VWAP_Bounce import VWAPBounceStrategy  # 追加インポート
from strategies.VWAP_Breakout import VWAPBreakoutStrategy  # VWAPブレイクアウト戦略を追加

# 最適化モジュールのインポート
from optimization.optimize_breakout_strategy import optimize_breakout_strategy
from optimization.optimize_gc_strategy import optimize_gc_strategy
from optimization.optimize_contrarian_strategy import optimize_contrarian_strategy  # 追加
from optimization.optimize_momentum_strategy import optimize_momentum_strategy  # 追加インポート
from optimization.optimize_opening_gap_strategy import optimize_opening_gap_strategy  # 追加インポート
from optimization.optimize_vwap_bounce_strategy import optimize_vwap_bounce_strategy  # 追加インポート
from optimization.optimize_vwap_breakout_strategy import optimize_vwap_breakout_strategy  # ブレイクアウト戦略最適化を追加
from optimization.configs.breakout_optimization import PARAM_GRID

# 半自動システム関連のインポート
from config.optimized_parameters import OptimizedParameterManager
from optimization.overfitting_detector import OverfittingDetector
from validation.parameter_validator import ParameterValidator
from tools.parameter_reviewer import ParameterReviewer

from data_fetcher import get_parameters_and_data
from data_processor import preprocess_data
from indicators.indicator_calculator import compute_indicators
from config.logger_config import setup_logger

import json

# ロガーの設定
logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\optimization.log")

def main():
    parser = argparse.ArgumentParser(description='戦略パラメータの最適化')
    parser.add_argument('--strategy', type=str, default='breakout',
                        choices=['breakout', 'vwap_breakout', 'momentum', 'gc', 'contrarian', 'opening_gap', 'vwap_bounce'],
                        help='最適化する戦略')
    parser.add_argument('--parallel', action='store_true',
                        help='並列処理を使用する')
    parser.add_argument('--save-results', action='store_true',
                        help='最適化結果をJSONファイルに保存する')
    parser.add_argument('--validate', action='store_true',
                        help='パラメータの妥当性検証とオーバーフィッティング検出を実行する')
    parser.add_argument('--auto-approve', action='store_true',
                        help='低リスクの場合に自動承認する')
    parser.add_argument('--reviewer-id', type=str, default='auto_optimizer',
                        help='レビュアーID（自動承認時に使用）')
    args = parser.parse_args()
    
    # データ取得
    logger.info("株価データを取得中...")
    ticker, start_date, end_date, stock_data, index_data = get_parameters_and_data()
    
    # データ前処理
    stock_data = preprocess_data(stock_data)
    stock_data = compute_indicators(stock_data)
    
    # 選択した戦略の最適化を実行
    strategy_map = {
        'breakout': optimize_breakout_strategy,
        'contrarian': optimize_contrarian_strategy,
        'gc': optimize_gc_strategy,
        'momentum': optimize_momentum_strategy,
        'opening_gap': optimize_opening_gap_strategy,
        'vwap_bounce': optimize_vwap_bounce_strategy,
        'vwap_breakout': optimize_vwap_breakout_strategy,  # VWAPブレイクアウト戦略の最適化関数を追加
    }
    
    if args.strategy in strategy_map:
        logger.info(f"{args.strategy}戦略の最適化を開始...")
        optimize_func = strategy_map[args.strategy]
        
        # VWAPブレイクアウト戦略は市場インデックスデータが必要
        if args.strategy == 'vwap_breakout':
            results = optimize_func(stock_data, index_data, use_parallel=args.parallel)
        else:
            results = optimize_func(stock_data, use_parallel=args.parallel)
        
        # 最適化結果の処理
        if not results.empty:
            best_params = results.iloc[0].to_dict()
            best_score = best_params.pop("score", None)
            logger.info(f"最適化完了: 最良スコア = {best_score}")
            logger.info(f"最適パラメータ: {best_params}")
            
            # 半自動システムの実行
            if args.save_results or args.validate or args.auto_approve:
                strategy_class_name = get_strategy_class_name(args.strategy)
                process_optimization_results(
                    strategy_class_name, 
                    results, 
                    ticker, 
                    start_date, 
                    end_date,
                    args
                )
        else:
            logger.warning("最適化に失敗しました。結果が空です。")
    else:
        logger.error(f"未実装の戦略: {args.strategy}")
        logger.info(f"実装済み戦略: {list(strategy_map.keys())}")


def get_strategy_class_name(strategy_key: str) -> str:
    """戦略キーからクラス名を取得"""
    strategy_class_map = {
        'breakout': 'BreakoutStrategy',
        'contrarian': 'ContrarianStrategy', 
        'gc': 'GCStrategy',
        'momentum': 'MomentumInvestingStrategy',
        'opening_gap': 'OpeningGapStrategy',
        'vwap_bounce': 'VWAPBounceStrategy',
        'vwap_breakout': 'VWAPBreakoutStrategy'
    }
    return strategy_class_map.get(strategy_key, strategy_key)


def process_optimization_results(strategy_name: str, results, ticker: str, 
                                start_date: str, end_date: str, args) -> None:
    """最適化結果の半自動処理"""
    logger.info("半自動システムによる結果処理を開始...")
    
    # 最適化結果をJSONに保存
    if args.save_results:
        logger.info("最適化結果をJSONファイルに保存中...")
        save_optimization_results(strategy_name, results, ticker, start_date, end_date)
    
    # パラメータ検証とオーバーフィッティング検出
    if args.validate or args.auto_approve:
        logger.info("パラメータ検証とオーバーフィッティング検出を実行中...")
        validate_and_review_results(strategy_name, results, args)


def save_optimization_results(strategy_name: str, results, ticker: str,
                             start_date: str, end_date: str) -> None:
    """最適化結果をJSONファイルに保存"""
    try:
        param_manager = OptimizedParameterManager()
        
        # 結果の上位5つを保存
        top_results = results.head(5)
        
        for idx, row in top_results.iterrows():
            # パラメータとスコアを分離
            param_dict = row.to_dict()
            score = param_dict.pop("score", 0)
              # パフォーマンス指標（仮の値、実際の実装では計算が必要）
            performance_metrics = {
                'sharpe_ratio': score,  # スコアをシャープレシオとして使用
                'total_return': 0.1,   # 実際の実装では計算が必要
                'max_drawdown': -0.05, # 実際の実装では計算が必要
                'win_rate': 0.6,       # 実際の実装では計算が必要
                'profit_factor': 1.5,  # 実際の実装では計算が必要
                'volatility': 0.15     # 実際の実装では計算が必要
            }
            param_manager.save_optimized_params(
                strategy_name=strategy_name,
                ticker=ticker,
                params=param_dict,
                metrics=performance_metrics
            )
        
        logger.info(f"上位{len(top_results)}件の最適化結果を保存しました")
        
    except Exception as e:
        logger.error(f"最適化結果の保存中にエラーが発生: {e}")


def validate_and_review_results(strategy_name: str, results, args) -> None:
    """パラメータ検証とレビュー処理"""
    try:
        param_manager = OptimizedParameterManager()
        overfitting_detector = OverfittingDetector()
        param_validator = ParameterValidator()
        
        # 最上位の結果を検証
        best_result = results.iloc[0].to_dict()
        score = best_result.pop("score", 0)
        
        # パフォーマンスデータ（実際の実装では詳細計算が必要）
        performance_data = {
            'sharpe_ratio': score,
            'total_return': 0.1,
            'max_drawdown': -0.05,
            'win_rate': 0.6,
            'volatility': 0.15
        }        # オーバーフィッティング検出（DataFrameで渡す）
        overfitting_result = overfitting_detector.detect_overfitting(results)
        logger.info(f"オーバーフィッティング検出: {overfitting_result['overfitting_risk']}")
        
        # パラメータ妥当性検証（自動戦略判別を使用）
        validation_result = param_validator.validate_auto(best_result, strategy_hint=args.strategy)
        logger.info(f"パラメータ妥当性: {'合格' if validation_result['valid'] else '不合格'}")
          # 自動承認の判定
        if args.auto_approve:
            overall_risk = calculate_overall_risk(overfitting_result, validation_result)
            
            if overall_risk == 'low':
                # 自動承認
                logger.info("✅ 低リスクのため自動承認されました")
                
                # 自動承認の記録（param_idは自動生成）
                import uuid
                param_id = str(uuid.uuid4())[:8]  # 8文字のランダムID                save_auto_approval_record(strategy_name, param_id, args.reviewer_id,
                                        overfitting_result, validation_result)
            else:
                logger.info(f"⚠️ リスクレベル: {overall_risk} - 手動レビューが必要です")
                logger.info("parameter_reviewer.pyを使用して手動レビューを実行してください")
        
    except Exception as e:
        logger.error(f"検証プロセス中にエラーが発生: {e}")
        import traceback
        logger.error(f"スタックトレース: {traceback.format_exc()}")


def calculate_overall_risk(overfitting_result: dict, validation_result: dict) -> str:
    """総合リスクレベルの計算"""
    risk_levels = ['low', 'medium', 'high']
      # オーバーフィッティングリスク
    overfitting_risk = overfitting_result.get('overfitting_risk', 'medium')
      # 妥当性検証リスク
    validation_risk = 'low' if validation_result.get('valid', False) else 'high'
    if validation_result.get('warnings', []):
        validation_risk = 'medium' if validation_risk == 'low' else validation_risk
    
    # より高いリスクレベルを採用
    overall_risk_index = max(
        risk_levels.index(overfitting_risk),
        risk_levels.index(validation_risk)
    )
    
    return risk_levels[overall_risk_index]


def save_auto_approval_record(strategy_name: str, param_id: str, reviewer_id: str,
                             overfitting_result: dict, validation_result: dict) -> None:
    """自動承認記録を保存"""
    try:
        review_data = {
            'parameter_id': param_id,
            'strategy_name': strategy_name,
            'decision': 'approved',
            'reviewer_id': reviewer_id,
            'review_date': datetime.now().isoformat(),            'notes': f"自動承認 - オーバーフィッティングリスク: {overfitting_result['overfitting_risk']}, "
                    f"パラメータ妥当性: {'合格' if validation_result['valid'] else '不合格'}",
            'confidence_level': 4,  # 自動承認の信頼度
            'risk_acceptance': 'low',
            'auto_approved': True
        }
        
        # レビュー履歴の保存
        history_dir = os.path.join(os.path.dirname(__file__), 'config', 'review_history')
        os.makedirs(history_dir, exist_ok=True)
        
        history_file = os.path.join(history_dir, f"{strategy_name}_reviews.json")
        
        # 既存履歴の読み込み
        reviews = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    reviews = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                reviews = []
        
        # 新しいレビューを追加
        reviews.append(review_data)
        
        # 保存
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)
        
        logger.info(f"自動承認記録を保存しました: {param_id}")
        
    except Exception as e:
        logger.error(f"自動承認記録の保存中にエラーが発生: {e}")

if __name__ == "__main__":
    main()
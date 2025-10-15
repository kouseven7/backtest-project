"""
最小重み配分システムのテスト実行スクリプト

このスクリプトは最小重み配分システム（Minimum Weight System）の機能をテストします。
主な目的:
- 戦略の信号生成が正しく行われているか検証
- 同日のエントリー/エグジット処理が意図通りに動作するか確認
- バックテスト基本理念に従った処理が行われているか確認
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# プロジェクトルートを追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MinWeightSystemTest")

# 必要なインポート
from config.logger_config import setup_logger
from strategies.VWAP_Breakout import VWAPBreakoutStrategy
from signal_processing import check_same_day_entry_exit
from config.optimized_parameters import OptimizedParameterManager

def load_test_data():
    """
    テスト用の株価データをロードします
    テスト用のダミーデータを生成するか、既存のデータを使用します
    """
    try:
        # まずyfinanceからのデータ取得を試みる
        import yfinance as yf
        logger.info("yfinanceからサンプルデータを取得します")
        ticker = "AAPL"
        data = yf.download(ticker, start="2023-01-01", end="2023-12-31")
        if len(data) > 100:  # データ十分ある場合
            logger.info(f"yfinanceから{len(data)}件のデータを取得しました")
            return data
        else:
            logger.warning("十分なデータが取得できませんでした。ダミーデータを生成します。")
            raise Exception("十分なデータがありません")
    except Exception as e:
        logger.warning(f"データ取得エラー: {e}. ダミーデータを生成します。")
        
        # ダミーデータ生成（200営業日分）
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")
        np.random.seed(42)  # 再現性のため
        
        base_price = 100.0
        price_data = []
        
        # ランダムウォーク生成
        for i in range(len(dates)):
            if i == 0:
                price_data.append(base_price)
            else:
                change = np.random.normal(0, 1.0) * 1.0  # 1%の標準偏差
                price_data.append(price_data[-1] * (1 + change/100))
        
        # データフレーム作成
        data = pd.DataFrame(index=dates)
        data['Open'] = price_data
        data['High'] = [price * (1 + np.random.uniform(0, 0.02)) for price in price_data]
        data['Low'] = [price * (1 - np.random.uniform(0, 0.02)) for price in price_data]
        data['Close'] = [price * (1 + np.random.normal(0, 0.005)) for price in price_data]
        data['Adj Close'] = data['Close']
        data['Volume'] = [np.random.randint(100000, 1000000) for _ in range(len(dates))]
        
        logger.info(f"{len(data)}件のダミーデータを生成しました")
        return data

def run_single_strategy_test(strategy_class, data, params=None):
    """
    単一戦略のバックテストを実行し、Entry/Exit信号を検証
    """
    if params is None:
        params = {}  # デフォルトパラメータ
    
    try:
        # VWAPBreakoutStrategyには市場インデックスデータが必要なので、ダミーを作成
        # データと同じインデックスを持つダミーの市場データを生成
        index_data = pd.DataFrame(index=data.index)
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if col in data.columns:
                index_data[col] = data[col] * 0.9  # 適当な値を設定
        
        # 戦略インスタンス作成
        strategy = strategy_class(
            data=data.copy(),
            index_data=index_data,  # VWAPBreakoutStrategyに必要な引数を追加
            params=params,
            price_column="Adj Close"
        )
        
        # バックテスト実行
        logger.info(f"{strategy_class.__name__}のバックテストを開始します")
        result_df = strategy.backtest()
        
        # 結果分析
        entry_count = (result_df['Entry_Signal'] == 1).sum()
        exit_count = (result_df['Exit_Signal'] == 1).sum() + (result_df['Exit_Signal'] == -1).sum()
        
        logger.info(f"戦略実行結果: エントリー{entry_count}件, エグジット{exit_count}件")
        
        # 同日エントリー/エグジットチェック
        same_day_results = check_same_day_entry_exit(result_df)
        if same_day_results['has_same_day_signals']:
            logger.info(f"同日エントリー/エグジット検出: {same_day_results['same_day_count']}件")
            logger.info(f"最初の例: {same_day_results.get('first_example', 'なし')}")
        else:
            logger.info("同日エントリー/エグジットなし")
        
        return result_df
        
    except Exception as e:
        logger.error(f"戦略実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_signals(result_df):
    """
    信号分析: Entry/Exitの時間差、連続性などを分析
    """
    if result_df is None or len(result_df) == 0:
        logger.warning("分析対象データがありません")
        return
    
    # 連続エントリー/エグジット検出
    entry_indices = result_df[result_df['Entry_Signal'] == 1].index
    exit_indices = result_df[(result_df['Exit_Signal'] == 1) | (result_df['Exit_Signal'] == -1)].index
    
    # エントリー連続
    consecutive_entries = []
    for i in range(1, len(entry_indices)):
        days_between = (entry_indices[i] - entry_indices[i-1]).days
        if days_between <= 2:  # 2営業日以内に連続エントリー
            consecutive_entries.append((entry_indices[i-1], entry_indices[i]))
    
    if consecutive_entries:
        logger.info(f"連続エントリー検出: {len(consecutive_entries)}件")
        for first, second in consecutive_entries[:3]:  # 最初の3件のみ表示
            logger.info(f"  - {first.date()} と {second.date()}")
    else:
        logger.info("連続エントリーなし")
    
    # エントリーからエグジットまでの保有期間分析
    holding_periods = []
    for entry_date in entry_indices:
        # このエントリー以降の最初のエグジットを探す
        future_exits = exit_indices[exit_indices > entry_date]
        if len(future_exits) > 0:
            exit_date = future_exits[0]
            holding_days = (exit_date - entry_date).days
            holding_periods.append(holding_days)
    
    if holding_periods:
        avg_holding = sum(holding_periods) / len(holding_periods)
        min_holding = min(holding_periods)
        max_holding = max(holding_periods)
        logger.info(f"保有期間分析: 平均{avg_holding:.1f}日, 最短{min_holding}日, 最長{max_holding}日")
    else:
        logger.info("保有期間データなし")
    
    # 同日エントリー/エグジット再確認（詳細分析）
    same_day = []
    for date in result_df.index:
        if result_df.loc[date, 'Entry_Signal'] == 1 and result_df.loc[date, 'Exit_Signal'] != 0:
            same_day.append(date)
    
    if same_day:
        logger.info(f"同日エントリー/エグジット（詳細分析）: {len(same_day)}件")
        for date in same_day[:3]:  # 最初の3件のみ表示
            entry_val = result_df.loc[date, 'Entry_Signal']
            exit_val = result_df.loc[date, 'Exit_Signal']
            logger.info(f"  - {date.date()}: Entry={entry_val}, Exit={exit_val}")
    
    # シグナル存在確認（バックテスト基本理念遵守の確認）
    if 'Entry_Signal' not in result_df.columns:
        logger.warning("Entry_Signal列が存在しません - バックテスト基本理念違反")
    if 'Exit_Signal' not in result_df.columns:
        logger.warning("Exit_Signal列が存在しません - バックテスト基本理念違反")
    
    # ゼロ件検査
    if (result_df['Entry_Signal'] == 1).sum() == 0:
        logger.warning("エントリーシグナルがゼロ件です")
    if ((result_df['Exit_Signal'] == 1) | (result_df['Exit_Signal'] == -1)).sum() == 0:
        logger.warning("エグジットシグナルがゼロ件です")

def main():
    """メイン実行関数"""
    print("=== 最小重み配分システムテスト開始 ===")
    
    try:
        # 1. テストデータ取得
        print("1. テストデータ取得中...")
        test_data = load_test_data()
        print(f"  - {len(test_data)}件のデータを取得")
        
        # 2. パラメータマネージャー初期化
        param_manager = OptimizedParameterManager()
        print("2. パラメータマネージャー初期化完了")
        
        # 3. VWAPブレイクアウト戦略のテスト
        print("3. VWAPブレイクアウト戦略テスト実行中...")
        vwap_params = {'vwap_period': 20, 'atr_period': 14, 'atr_multiplier': 1.5}
        vwap_result = run_single_strategy_test(VWAPBreakoutStrategy, test_data, vwap_params)
        
        # 4. 信号分析
        print("4. 信号分析実行中...")
        analyze_signals(vwap_result)
        
        # 5. 検証結果出力
        print("\n=== テスト結果サマリー ===")
        
        if vwap_result is not None:
            entry_count = (vwap_result['Entry_Signal'] == 1).sum()
            exit_count = (vwap_result['Exit_Signal'] != 0).sum()
            balance = entry_count - exit_count
            
            print(f"総エントリー数: {entry_count}")
            print(f"総エグジット数: {exit_count}")
            print(f"差分: {balance}")
            
            # 同日エントリー/エグジット最終確認
            same_day_results = check_same_day_entry_exit(vwap_result)
            if same_day_results['has_same_day_signals']:
                print(f"同日エントリー/エグジット検出: {same_day_results['same_day_count']}件")
                print("同日エントリー/エグジットは検出のみを行い、シグナルは修正していません。")
                print("これはバックテスト基本理念に従い、シグナルを戦略通りに維持するためです。")
            else:
                print("同日エントリー/エグジットなし")
            
            # 結果検証
            print("\nバックテスト基本理念検証:")
            print(f"Entry_Signal列存在: {'はい' if 'Entry_Signal' in vwap_result.columns else 'いいえ'}")
            print(f"Exit_Signal列存在: {'はい' if 'Exit_Signal' in vwap_result.columns else 'いいえ'}")
            print(f"実際にバックテスト実行: {'はい' if entry_count > 0 else 'いいえ'}")
        else:
            print("戦略実行結果がありません")
        
        print("\n=== テスト完了 ===")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
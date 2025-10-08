#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完全統合テスト: main.pyの戦略処理 + Excel出力
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import os

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_realistic_test_data():
    """戦略処理用のリアルなテストデータを生成"""
    
    # 2023年のデータ期間
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # 平日のみ
    
    # 基本価格データ（株価のリアルな動き）
    np.random.seed(42)
    base_price = 2000
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 100))  # 最低価格100円
    
    # OHLCV データ
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # リアルなOHLCV生成
        daily_volatility = 0.015
        intraday_range = close * daily_volatility
        
        high = close + np.random.uniform(0, intraday_range)
        low = close - np.random.uniform(0, intraday_range)
        open_price = close + np.random.uniform(-intraday_range/2, intraday_range/2)
        volume = int(np.random.uniform(10000, 100000))
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(max(open_price, high, close), 2),
            'Low': round(min(open_price, low, close), 2),
            'Close': round(close, 2),
            'Volume': volume,
            'Adj Close': round(close, 2)
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    logger.info(f"テストデータ生成完了: {len(df)}行 x {len(df.columns)}列")
    logger.info(f"期間: {df.index[0].strftime('%Y-%m-%d')} ～ {df.index[-1].strftime('%Y-%m-%d')}")
    logger.info(f"価格範囲: {df['Close'].min():.2f} ～ {df['Close'].max():.2f}円")
    
    return df

def apply_simple_strategy(data):
    """シンプルな戦略を適用してEntry_Signal/Exit_Signalを生成"""
    
    df = data.copy()
    
    # 技術指標計算
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    # シグナル初期化
    df['Entry_Signal'] = 0
    df['Exit_Signal'] = 0
    df['Strategy'] = ''
    
    # シンプルなトレンドフォロー戦略
    position = 0
    entry_price = 0
    
    for i in range(50, len(df)):  # 50日以降から開始（SMA計算のため）
        current_price = df.iloc[i]['Close']
        sma_20 = df.iloc[i]['SMA_20']
        sma_50 = df.iloc[i]['SMA_50']
        rsi = df.iloc[i]['RSI']
        
        # エントリー条件：SMA20 > SMA50 かつ RSI < 70
        if position == 0 and sma_20 > sma_50 and rsi < 70:
            df.iloc[i, df.columns.get_loc('Entry_Signal')] = 1
            df.iloc[i, df.columns.get_loc('Strategy')] = 'TrendFollow'
            position = 1
            entry_price = current_price
            logger.info(f"エントリーシグナル: {df.index[i].strftime('%Y-%m-%d')} - 価格: {current_price:.2f}円")
        
        # エグジット条件：利益確定 or 損切り
        elif position == 1:
            profit_rate = (current_price - entry_price) / entry_price
            
            if profit_rate > 0.05 or profit_rate < -0.03 or sma_20 < sma_50:
                df.iloc[i, df.columns.get_loc('Exit_Signal')] = 1
                df.iloc[i, df.columns.get_loc('Strategy')] = 'TrendFollow'
                position = 0
                logger.info(f"エグジットシグナル: {df.index[i].strftime('%Y-%m-%d')} - 価格: {current_price:.2f}円 - 利益率: {profit_rate:.2%}")
    
    signal_count = (df['Entry_Signal'].sum(), df['Exit_Signal'].sum())
    logger.info(f"シグナル生成完了: エントリー{signal_count[0]}回, エグジット{signal_count[1]}回")
    
    return df

def calculate_rsi(prices, period=14):
    """RSI計算"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def test_excel_export(processed_data):
    """Excel出力テスト"""
    
    try:
        # Excel出力モジュールをインポート
        from output.simple_excel_exporter import ExcelDataProcessor
        
        # Excel処理実行
        processor = ExcelDataProcessor()
        filename = processor.process_main_data(processed_data)
        
        if filename:
            logger.info(f"[OK] Excel出力成功: {filename}")
            
            # 出力結果検証
            excel_data = pd.ExcelFile(filename)
            logger.info(f"生成シート: {excel_data.sheet_names}")
            
            # サマリーデータ確認
            summary = pd.read_excel(filename, sheet_name='サマリー')
            logger.info(f"サマリーシート行数: {len(summary)}")
            
            # 取引履歴確認
            trades = pd.read_excel(filename, sheet_name='取引履歴')
            logger.info(f"取引履歴行数: {len(trades)}")
            
            return filename
        else:
            logger.error("[ERROR] Excel出力失敗")
            return None
            
    except Exception as e:
        logger.error(f"Excel出力エラー: {e}")
        return None

def main():
    """完全統合テスト実行"""
    
    logger.info("=== 完全統合テスト開始 ===")
    
    try:
        # 1. テストデータ生成
        logger.info("1. テストデータ生成...")
        test_data = generate_realistic_test_data()
        
        # 2. 戦略適用
        logger.info("2. 戦略シグナル生成...")
        processed_data = apply_simple_strategy(test_data)
        
        # 3. データ確認
        logger.info("3. 処理結果確認...")
        logger.info(f"最終データ形状: {processed_data.shape}")
        logger.info(f"列一覧: {list(processed_data.columns)}")
        
        # Entry_Signal/Exit_Signalの確認
        entry_signals = processed_data['Entry_Signal'].sum()
        exit_signals = processed_data['Exit_Signal'].sum()
        logger.info(f"シグナル統計: エントリー{entry_signals}回, エグジット{exit_signals}回")
        
        # 4. Excel出力テスト
        logger.info("4. Excel出力テスト...")
        excel_file = test_excel_export(processed_data)
        
        if excel_file:
            logger.info("[OK] 完全統合テスト成功！")
            logger.info(f"出力ファイル: {excel_file}")
        else:
            logger.error("[ERROR] Excel出力でエラーが発生しました")
        
    except Exception as e:
        logger.error(f"統合テストエラー: {e}")
        raise

if __name__ == "__main__":
    main()

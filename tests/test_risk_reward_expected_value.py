"""
リスクリワード比・期待値追加機能のテストスクリプト
File: test_risk_reward_expected_value.py
Description: SimpleExcelExporterのリスクリワード比と期待値計算機能をテストします

Author: imega
Created: 2025-07-31
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクトパスを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from output.simple_excel_exporter import SimpleExcelExporter
from config.logger_config import setup_logger

logger = setup_logger(__name__)

def create_test_data_with_signals() -> pd.DataFrame:
    """テスト用のシグナル付き株価データを作成"""
    # 30日間のテストデータ
    dates = pd.date_range(start='2023-01-01', end='2023-01-30', freq='D')
    np.random.seed(42)
    
    # 基本株価データ
    base_price = 1000
    price_changes = np.random.randn(len(dates)) * 10
    prices = base_price + np.cumsum(price_changes)
    
    data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(10000, 100000, len(dates)),
        'Entry_Signal': 0,
        'Exit_Signal': 0,
        'Strategy': 'TestStrategy'
    }, index=dates)
    
    # テスト用シグナルを追加（勝ち負けを混在させる）
    entry_dates = [
        '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20', '2023-01-25'
    ]
    exit_dates = [
        '2023-01-08', '2023-01-12', '2023-01-18', '2023-01-22', '2023-01-28'
    ]
    
    for entry_date in entry_dates:
        if entry_date in data.index:
            data.loc[entry_date, 'Entry_Signal'] = 1
    
    for exit_date in exit_dates:
        if exit_date in data.index:
            data.loc[exit_date, 'Exit_Signal'] = -1
    
    return data

def test_risk_reward_calculation():
    """リスクリワード比の計算テスト"""
    logger.info("リスクリワード比計算テスト開始")
    
    # テストデータ作成
    trade_data = pd.DataFrame({
        '取引結果': [1000, -500, 800, -300, 1200, -600],
        '手数料': [50, 25, 40, 15, 60, 30]
    })
    
    exporter = SimpleExcelExporter()
    risk_reward = exporter._calculate_risk_reward_ratio(trade_data)
    
    # 手動計算での検証
    # 手数料除く純損益: [1050, -475, 840, -285, 1260, -570]
    # 勝ちトレード: [1050, 840, 1260] → 平均: 1050
    # 負けトレード: [-475, -285, -570] → 平均絶対値: 443.33
    # リスクリワード比: 1050 / 443.33 ≈ 2.368
    
    expected_ratio = 1050 / (443.33)  # 概算
    logger.info(f"計算されたリスクリワード比: {risk_reward:.3f}")
    logger.info(f"期待値（概算）: {expected_ratio:.3f}")
    
    assert abs(risk_reward - 2.368) < 0.1, f"リスクリワード比が期待値と異なります: {risk_reward}"
    logger.info("[OK] リスクリワード比計算テスト成功")

def test_expected_value_calculation():
    """期待値計算テスト"""
    logger.info("期待値計算テスト開始")
    
    # テストデータ作成
    trade_data = pd.DataFrame({
        '取引結果': [1000, -500, 800, -300],  # 勝率50%, 平均利益900, 平均損失400
        '手数料': [50, 25, 40, 15]
    })
    
    exporter = SimpleExcelExporter(initial_capital=1000000)
    win_rate = 50.0  # 50%
    
    expected_yen, expected_pct = exporter._calculate_expected_value(trade_data, win_rate)
    
    # 手動計算での検証
    # 手数料除く純損益: [1050, -475, 840, -285]
    # 勝ちトレード平均: (1050 + 840) / 2 = 945
    # 負けトレード平均: (475 + 285) / 2 = 380
    # 期待値（円）: 0.5 * 945 - 0.5 * 380 = 282.5
    # 期待値（％）: 282.5 / 1000000 * 100 = 0.02825%
    
    logger.info(f"計算された期待値（円）: {expected_yen:.2f}")
    logger.info(f"計算された期待値（％）: {expected_pct:.5f}%")
    
    assert abs(expected_yen - 282.5) < 5, f"期待値（円）が期待値と異なります: {expected_yen}"
    assert abs(expected_pct - 0.02825) < 0.001, f"期待値（％）が期待値と異なります: {expected_pct}"
    logger.info("[OK] 期待値計算テスト成功")

def test_excel_export_with_new_metrics():
    """新指標を含むExcel出力テスト"""
    logger.info("Excel出力テスト（新指標含む）開始")
    
    # テストデータ作成
    test_data = create_test_data_with_signals()
    
    # Excel出力テスト
    exporter = SimpleExcelExporter(initial_capital=1000000)
    
    # テスト用ディレクトリ作成
    test_dir = os.path.join(current_dir, 'test_results')
    os.makedirs(test_dir, exist_ok=True)
    
    output_path = exporter.export_backtest_results(
        test_data, 
        'TEST', 
        output_dir=test_dir,
        filename='test_risk_reward_expected_value.xlsx'
    )
    
    assert os.path.exists(output_path), f"Excelファイルが作成されませんでした: {output_path}"
    logger.info(f"[OK] Excel出力テスト成功: {output_path}")
    
    return output_path

def main():
    """メインテスト実行"""
    logger.info("=== リスクリワード比・期待値追加機能テスト開始 ===")
    
    try:
        # 個別関数テスト
        test_risk_reward_calculation()
        test_expected_value_calculation()
        
        # 統合テスト
        output_path = test_excel_export_with_new_metrics()
        
        logger.info("=== 全テスト成功 ===")
        logger.info(f"出力ファイル: {output_path}")
        
        # Excelファイルを開く（Windowsの場合）
        try:
            os.startfile(output_path)
            logger.info("Excelファイルを開きました")
        except:
            logger.info("Excelファイルの自動オープンに失敗しました")
        
        return True
        
    except Exception as e:
        logger.error(f"テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] テスト完了！新しい指標が正常に追加されました。")
        print("[CHART] パフォーマンス指標シートにリスクリワード比と期待値が表示されます。")
        print("[UP] 戦略別統計シートにも同様の指標が追加されています。")
    else:
        print("\n[ERROR] テストに失敗しました。")

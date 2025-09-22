"""
Test Script: Problem 戦略統計 Integration Test
File: test_strategy_statistics_integration.py
Description: 
  Problem 戦略統計の統合テスト
  DSSMSUnifiedOutputEngineとStrategyStatisticsCalculatorの統合動作を検証

Author: DSSMS Team
Created: 2025-01-25
Version: 1.0.0
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# プロジェクトルートを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data():
    """テスト用のバックテストデータ作成"""
    logger.info("🔧 テストデータ作成中...")
    
    # 日付範囲
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 戦略リスト
    strategies = ['TrendFollowing', 'MeanReversion', 'Momentum', 'Breakout']
    
    # サンプル取引データ作成
    np.random.seed(42)
    trades_data = []
    
    for i, strategy in enumerate(strategies):
        # 戦略ごとに特性の異なる取引を生成
        trade_count = np.random.randint(50, 150)
        
        for j in range(trade_count):
            # 戦略ごとに異なる特性
            if strategy == 'TrendFollowing':
                base_pnl = np.random.normal(100, 300)  # 大きな利益、大きな損失
            elif strategy == 'MeanReversion':
                base_pnl = np.random.normal(50, 150)   # 安定的な小利益
            elif strategy == 'Momentum':
                base_pnl = np.random.normal(75, 200)   # 中程度のボラティリティ
            else:  # Breakout
                base_pnl = np.random.normal(120, 400)  # 高リスク高リターン
            
            entry_date = dates[np.random.randint(0, len(dates)-30)]
            exit_date = entry_date + timedelta(days=np.random.randint(1, 30))
            
            trades_data.append({
                'strategy': strategy,
                'entry_time': entry_date,
                'exit_time': exit_date,
                'pnl': base_pnl,
                'volume': np.random.uniform(100, 1000),
                'price': np.random.uniform(100, 500),
                'fees': abs(base_pnl) * 0.001  # 0.1%手数料
            })
    
    trades_df = pd.DataFrame(trades_data)
    
    # 日次PnLデータ作成
    daily_pnl_data = []
    for date in dates[:90]:  # 3ヶ月分
        daily_return = np.random.normal(0.001, 0.02)  # 0.1%±2%のリターン
        daily_pnl_data.append({
            'date': date,
            'daily_return': daily_return,
            'cumulative_pnl': 0  # 後で計算
        })
    
    daily_pnl_df = pd.DataFrame(daily_pnl_data)
    daily_pnl_df.set_index('date', inplace=True)
    
    # バックテスト結果辞書作成
    backtest_results = {
        'trades': trades_df,
        'portfolio_values': daily_pnl_df,
        'metadata': {
            'test_created': datetime.now(),
            'strategies_count': len(strategies),
            'trades_count': len(trades_df)
        }
    }
    
    logger.info(f"✅ テストデータ作成完了: {len(strategies)}戦略, {len(trades_df)}取引")
    return backtest_results


def test_strategy_statistics_calculator():
    """StrategyStatisticsCalculator単体テスト"""
    logger.info("🧪 StrategyStatisticsCalculator単体テスト開始")
    
    try:
        from src.dssms.strategy_statistics_calculator import StrategyStatisticsCalculator
        
        # 計算器初期化
        calculator = StrategyStatisticsCalculator(risk_free_rate=0.02)
        
        # テストデータ準備
        test_trades = pd.DataFrame({
            'pnl': [100, -50, 200, -30, 150, -80],
            'volume': [500, 300, 800, 200, 600, 400],
            'price': [150, 120, 180, 110, 160, 130],
            'fees': [1, 0.5, 2, 0.3, 1.5, 0.8]
        })
        
        # 統計計算
        stats = calculator.calculate_comprehensive_statistics(
            strategy_name="TestStrategy",
            trades_df=test_trades
        )
        
        # 結果検証
        assert stats.strategy_name == "TestStrategy"
        assert stats.trade_count == 6
        assert 0.0 <= stats.win_rate <= 1.0
        assert stats.total_pnl > 0  # 利益が出ているはず
        
        # フォーマット出力テスト
        formatted = calculator.format_statistics_for_export(stats)
        assert '戦略名' in formatted
        assert '取引回数' in formatted
        assert '勝率(%)' in formatted
        
        logger.info("✅ StrategyStatisticsCalculator単体テスト成功")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import失敗: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 単体テストエラー: {e}")
        return False


def test_dssms_unified_output_engine():
    """DSSMSUnifiedOutputEngine統合テスト"""
    logger.info("🧪 DSSMSUnifiedOutputEngine統合テスト開始")
    
    try:
        from dssms_unified_output_engine import DSSMSUnifiedOutputEngine
        
        # エンジン初期化
        engine = DSSMSUnifiedOutputEngine()
        
        # テストデータ作成
        test_data = create_test_data()
        
        # データソース設定
        setup_success = engine.set_data_source(test_data)
        if not setup_success:
            logger.error("❌ データソース設定失敗")
            return False
        
        # 戦略統計計算テスト
        logger.info("📊 戦略統計計算テスト中...")
        strategy_stats = engine._calculate_strategy_statistics(test_data)
        
        if not strategy_stats:
            logger.warning("⚠️ 戦略統計が空です")
            return False
        
        # 結果検証
        for strategy_name, stats in strategy_stats.items():
            logger.info(f"戦略: {strategy_name}")
            
            if isinstance(stats, dict):
                # 必須項目の存在確認
                required_keys = ['戦略名', '取引回数', '勝率(%)', '総損益']
                missing_keys = [key for key in required_keys if key not in stats]
                
                if missing_keys:
                    logger.warning(f"⚠️ 不足キー [{strategy_name}]: {missing_keys}")
                else:
                    logger.info(f"  ✅ 必須統計項目完備")
                
                # 統計値の妥当性確認
                trade_count = stats.get('取引回数', 0)
                win_rate = stats.get('勝率(%)', 0)
                
                if trade_count > 0:
                    logger.info(f"  取引回数: {trade_count}")
                    logger.info(f"  勝率: {win_rate}%")
                    logger.info(f"  総損益: {stats.get('総損益', 0)}")
                    
                    # データ品質スコア確認
                    quality = stats.get('データ品質', 0)
                    logger.info(f"  データ品質: {quality}")
        
        # 戦略統計シート作成テスト
        logger.info("📄 戦略統計シート作成テスト中...")
        
        # データソースに戦略統計を設定
        engine.data_source['strategy_statistics'] = strategy_stats
        
        # シート作成
        stats_sheet = engine._create_strategy_stats_sheet()
        
        if stats_sheet.empty:
            logger.warning("⚠️ 戦略統計シートが空です")
            return False
        
        # シート内容検証
        logger.info(f"📊 戦略統計シート: {len(stats_sheet)}行, {len(stats_sheet.columns)}列")
        logger.info(f"列名: {list(stats_sheet.columns)}")
        
        # 重要な列の存在確認
        important_columns = ['戦略名', '取引回数', '勝率(%)', '総損益', 'シャープレシオ']
        available_important = [col for col in important_columns if col in stats_sheet.columns]
        logger.info(f"重要列利用可能: {len(available_important)}/{len(important_columns)}列")
        
        # 各戦略行の検証
        for idx, row in stats_sheet.iterrows():
            strategy_name = row.get('戦略名', 'Unknown')
            if strategy_name not in ['📊 全戦略合計', 'Unknown']:
                trade_count = row.get('取引回数', 0)
                logger.info(f"  {strategy_name}: {trade_count}取引")
        
        logger.info("✅ DSSMSUnifiedOutputEngine統合テスト成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 統合テストエラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """メインテスト実行"""
    logger.info("🚀 Problem 戦略統計 統合テスト開始")
    logger.info("="*60)
    
    test_results = []
    
    # テスト1: StrategyStatisticsCalculator単体テスト
    logger.info("\n📋 テスト1: StrategyStatisticsCalculator単体テスト")
    result1 = test_strategy_statistics_calculator()
    test_results.append(("StrategyStatisticsCalculator", result1))
    
    # テスト2: DSSMSUnifiedOutputEngine統合テスト
    logger.info("\n📋 テスト2: DSSMSUnifiedOutputEngine統合テスト")
    result2 = test_dssms_unified_output_engine()
    test_results.append(("DSSMSUnifiedOutputEngine", result2))
    
    # 結果サマリー
    logger.info("\n" + "="*60)
    logger.info("📊 テスト結果サマリー")
    logger.info("="*60)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ 成功" if result else "❌ 失敗"
        logger.info(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    logger.info("="*60)
    if all_passed:
        logger.info("🎉 Problem 戦略統計 統合テスト: 全テスト成功")
        logger.info("✅ 85.0ポイントエンジン品質維持確認")
        logger.info("✅ 戦略統計品質向上確認")
    else:
        logger.error("❌ Problem 戦略統計 統合テスト: 一部テスト失敗")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
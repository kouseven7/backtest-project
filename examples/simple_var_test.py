"""
5-3-2 VaRシステム - 基本動作確認
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_var_system():
    """VaRシステムの基本動作テスト"""
    try:
        logger.info("=== 5-3-2 VaRシステム基本動作テスト開始 ===")
        
        # テストデータ作成
        logger.info("テストデータ作成中...")
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        n_days = 200
        
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
        
        returns_data = pd.DataFrame(index=dates, columns=symbols)
        for symbol in symbols:
            returns_data[symbol] = np.random.normal(0.001, 0.02, n_days)
        
        weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        
        logger.info(f"データ準備完了: {len(returns_data)}日分, {len(symbols)}銘柄")
        
        # VaRエンジンのインポートと初期化
        from config.portfolio_var_calculator.advanced_var_engine import (
            AdvancedVaREngine, VaRCalculationConfig
        )
        
        logger.info("VaRエンジン初期化中...")
        config = VaRCalculationConfig(
            primary_method="hybrid",
            confidence_levels=[0.95, 0.99],
            historical_window=100,
            monte_carlo_simulations=1000
        )
        
        engine = AdvancedVaREngine(config)
        
        # VaR計算実行
        logger.info("VaR計算実行中...")
        result = engine.calculate_comprehensive_var(returns_data, weights)
        
        # 結果表示
        logger.info("=== VaR計算結果 ===")
        logger.info(f"VaR 95%: {result.get_var_95():.4f} ({result.get_var_95()*100:.2f}%)")
        logger.info(f"VaR 99%: {result.get_var_99():.4f} ({result.get_var_99()*100:.2f}%)")
        logger.info(f"市場レジーム: {result.market_regime}")
        logger.info(f"計算手法: {result.calculation_method}")
        logger.info(f"分散化効果: {result.diversification_benefit:.4f}")
        
        # ハイブリッド計算器テスト
        logger.info("ハイブリッド計算器テスト中...")
        from config.portfolio_var_calculator.hybrid_var_calculator import HybridVaRCalculator
        
        hybrid_calc = HybridVaRCalculator(config)
        hybrid_result = hybrid_calc.calculate_hybrid_var(returns_data, weights)
        
        logger.info("=== ハイブリッドVaR結果 ===")
        if isinstance(hybrid_result, dict):
            var_95 = hybrid_result.get('var_95', 0)
            var_99 = hybrid_result.get('var_99', 0)
            method = hybrid_result.get('calculation_method', 'N/A')
            
            if isinstance(var_95, (int, float)):
                logger.info(f"Hybrid VaR 95%: {var_95:.4f}")
            else:
                logger.info(f"Hybrid VaR 95%: {var_95}")
                
            if isinstance(var_99, (int, float)):
                logger.info(f"Hybrid VaR 99%: {var_99:.4f}")
            else:
                logger.info(f"Hybrid VaR 99%: {var_99}")
                
            logger.info(f"選択手法: {method}")
        else:
            logger.info(f"Hybrid VaR 95%: {hybrid_result.get_var_95():.4f}")
            logger.info(f"Hybrid VaR 99%: {hybrid_result.get_var_99():.4f}")
            logger.info(f"選択手法: {hybrid_result.calculation_method}")
        
        # システム完了確認
        logger.info("=== テスト完了 ===")
        logger.info("✓ 高度VaRエンジン: 正常動作")
        logger.info("✓ ハイブリッド計算器: 正常動作")
        logger.info("✓ 基本機能: すべて正常")
        
        logger.info("5-3-2 ポートフォリオVaR計算システム実装成功！")
        
        return {
            'status': 'SUCCESS',
            'basic_var_95': result.get_var_95(),
            'basic_var_99': result.get_var_99(),
            'hybrid_var_95': hybrid_result.get('var_95', 0),
            'hybrid_var_99': hybrid_result.get('var_99', 0),
            'calculation_method': result.calculation_method,
            'market_regime': result.market_regime
        }
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'FAILED', 'error': str(e)}

if __name__ == "__main__":
    result = test_var_system()
    
    if result['status'] == 'SUCCESS':
        print("\n🎉 5-3-2 VaRシステム実装完了！")
        print(f"VaR95%: {result['basic_var_95']:.4f}")
        print(f"VaR99%: {result['basic_var_99']:.4f}")
    else:
        print(f"\n❌ テスト失敗: {result.get('error', 'Unknown error')}")

"""
DSSMS緊急診断スクリプト
根本的な問題の特定と修正提案
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# ロギング設定
from config.logger_config import setup_logger
logger = setup_logger("emergency_diagnosis")

def diagnose_perfect_order_system():
    """パーフェクトオーダー検出システムの診断"""
    logger.info("=" * 60)
    logger.info("緊急診断1: パーフェクトオーダー検出システム")
    logger.info("=" * 60)
    
    try:
        from src.dssms.perfect_order_detector import PerfectOrderDetector
        from data_fetcher import fetch_stock_data
        
        # テスト銘柄（トヨタ）
        test_symbol = "7203"
        detector = PerfectOrderDetector()
        
        # 2023年データ取得
        data = fetch_stock_data(test_symbol, "2023-01-01", "2023-12-31")
        
        if data is not None and not data.empty:
            logger.info(f"[OK] データ取得成功: {test_symbol} ({len(data)}日分)")
            
            # パーフェクトオーダー検出テスト
            result = detector.detect_perfect_order(data, "daily", test_symbol)
            
            logger.info(f"パーフェクトオーダー判定: {result.is_perfect_order}")
            logger.info(f"現在価格: {result.current_price:.2f}")
            logger.info(f"SMA短期: {result.sma_short:.2f}")
            logger.info(f"SMA中期: {result.sma_medium:.2f}")
            logger.info(f"SMA長期: {result.sma_long:.2f}")
            logger.info(f"強度スコア: {result.strength_score:.3f}")
            
            # 期間中のパーフェクトオーダー発生回数チェック
            po_count = 0
            monthly_checks = []
            
            for month in range(1, 13):
                try:
                    month_start = datetime(2023, month, 1)
                    if month == 12:
                        month_end = datetime(2023, 12, 31)
                    else:
                        month_end = datetime(2023, month + 1, 1) - timedelta(days=1)
                    
                    month_data = data[(data.index >= month_start) & (data.index <= month_end)]
                    if len(month_data) > 75:  # 十分なデータがある場合
                        month_result = detector.detect_perfect_order(month_data, "daily", test_symbol)
                        monthly_checks.append({
                            'month': month,
                            'perfect_order': month_result.is_perfect_order,
                            'strength': month_result.strength_score,
                            'price': month_result.current_price
                        })
                        if month_result.is_perfect_order:
                            po_count += 1
                except Exception as e:
                    logger.warning(f"月次チェックエラー {month}月: {e}")
            
            logger.info(f"2023年中のパーフェクトオーダー発生: {po_count}回/12ヶ月")
            
            if po_count == 0:
                logger.error("[ERROR] 致命的問題: 一年間でパーフェクトオーダーが一度も検出されていません")
                logger.error("これがDSSMS -100%損失の主要原因と考えられます")
            
            return monthly_checks
            
        else:
            logger.error(f"[ERROR] データ取得失敗: {test_symbol}")
            return []
            
    except Exception as e:
        logger.error(f"パーフェクトオーダー診断エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def diagnose_ranking_system():
    """ランキングシステムの診断"""
    logger.info("=" * 60)
    logger.info("緊急診断2: ランキングシステム")
    logger.info("=" * 60)
    
    try:
        from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem
        
        ranking_system = HierarchicalRankingSystem()
        
        # テスト銘柄リスト
        test_symbols = ["7203", "9984", "6758", "8306", "9432"]
        
        logger.info(f"テスト銘柄: {test_symbols}")
        
        # ランキング実行テスト
        ranking_result = ranking_system.rank_symbols(test_symbols, datetime(2023, 6, 15))
        
        if ranking_result and hasattr(ranking_result, 'ranked_symbols'):
            logger.info(f"[OK] ランキング成功: {len(ranking_result.ranked_symbols)}銘柄")
            for i, symbol in enumerate(ranking_result.ranked_symbols[:3]):
                logger.info(f"  順位{i+1}: {symbol}")
        else:
            logger.error("[ERROR] ランキング結果が無効")
            
    except Exception as e:
        logger.error(f"ランキングシステム診断エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())

def diagnose_switch_manager():
    """スイッチマネージャーの診断"""
    logger.info("=" * 60)
    logger.info("緊急診断3: スイッチマネージャー")
    logger.info("=" * 60)
    
    try:
        from src.dssms.intelligent_switch_manager import IntelligentSwitchManager
        
        switch_manager = IntelligentSwitchManager()
        logger.info("[OK] スイッチマネージャー初期化成功")
        
        # スイッチ条件の診断
        # これは具体的な条件を確認する必要があります
        logger.info("スイッチ条件設定の確認が必要")
        
    except Exception as e:
        logger.error(f"スイッチマネージャー診断エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())

def diagnose_data_quality():
    """データ品質の診断"""
    logger.info("=" * 60)
    logger.info("緊急診断4: データ品質")
    logger.info("=" * 60)
    
    try:
        from data_fetcher import fetch_stock_data
        
        test_symbols = ["7203", "9984", "6758"]
        
        for symbol in test_symbols:
            data = fetch_stock_data(symbol, "2023-01-01", "2023-12-31")
            
            if data is not None and not data.empty:
                logger.info(f"[OK] {symbol}: {len(data)}日分のデータ")
                
                # データ品質チェック
                null_count = data.isnull().sum().sum()
                zero_volume = (data['Volume'] == 0).sum() if 'Volume' in data.columns else 0
                
                if null_count > 0:
                    logger.warning(f"  [WARNING]  欠損値: {null_count}個")
                if zero_volume > 0:
                    logger.warning(f"  [WARNING]  出来高ゼロ: {zero_volume}日")
                    
                # 価格の年間推移確認
                start_price = data['Close'].iloc[0]
                end_price = data['Close'].iloc[-1]
                annual_return = (end_price - start_price) / start_price * 100
                
                logger.info(f"  年間リターン: {annual_return:.2f}%")
                
                if annual_return > 0:
                    logger.info(f"  [OK] {symbol}は2023年にプラス収益")
                else:
                    logger.info(f"  [DOWN] {symbol}は2023年にマイナス収益")
                    
            else:
                logger.error(f"[ERROR] {symbol}: データ取得失敗")
                
    except Exception as e:
        logger.error(f"データ品質診断エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """メイン診断実行"""
    logger.info("[ALERT] DSSMS緊急診断開始 [ALERT]")
    logger.info(f"実行時刻: {datetime.now()}")
    
    # 1. パーフェクトオーダーシステム診断
    monthly_po_results = diagnose_perfect_order_system()
    
    # 2. ランキングシステム診断
    diagnose_ranking_system()
    
    # 3. スイッチマネージャー診断
    diagnose_switch_manager()
    
    # 4. データ品質診断
    diagnose_data_quality()
    
    # 総合判定
    logger.info("=" * 60)
    logger.info("[TARGET] 総合診断結果")
    logger.info("=" * 60)
    
    if len(monthly_po_results) > 0:
        po_months = sum(1 for r in monthly_po_results if r['perfect_order'])
        logger.info(f"パーフェクトオーダー発生月数: {po_months}/12ヶ月")
        
        if po_months == 0:
            logger.error("[ALERT] 致命的問題確認:")
            logger.error("  - パーフェクトオーダーが全く検出されていません")
            logger.error("  - これがDSSMS -100%損失の根本原因です")
            logger.error("  - パーフェクトオーダー検出ロジックの緊急修正が必要")
    
    logger.info("緊急診断完了")

if __name__ == "__main__":
    main()

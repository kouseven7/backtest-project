"""
日次処理詳細調査 - _get_optimal_symbol()メソッド内部動作分析

この調査ファイルは、_get_optimal_symbol()メソッドの内部動作を詳細にログ出力し、
symbol=None が返される原因を特定することを目的とします。

調査方針:
1. _get_optimal_symbol()の各ステップにデバッグログを追加
2. DSS Core V3の実際の動作状況を確認
3. フォールバック処理の動作確認
4. 実際の戻り値とログの対応関係確認

Author: Investigation Team
Created: 2026-01-03
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# ログ設定
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_get_optimal_symbol_detailed():
    """
    _get_optimal_symbol()メソッドの詳細分析
    """
    logger.info("=== _get_optimal_symbol()詳細分析開始 ===")
    
    try:
        # DSSMSIntegratedBacktester実体作成
        from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester
        
        # 設定
        config = {
            'initial_capital': 1000000,
            'target_symbols': ['9101', '9104', '9107', '9202', '4502'],
            'dssms': {'enabled': True, 'method': 'dss_core_v3'}
        }
        
        logger.info("DSSMSIntegratedBacktester初期化開始")
        backtester = DSSMSIntegratedBacktester(config)
        
        # テスト日付: 2025-01-15
        target_date = datetime(2025, 1, 15)
        
        # --- CHECK POINT 1: コンポーネント状態確認 ---
        logger.info("=== CHECK POINT 1: コンポーネント状態確認 ===")
        
        # DSS Core V3利用可能性
        try:
            from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
            logger.info("✓ DSS Core V3 import成功")
            dss_available_check = True
        except ImportError as e:
            logger.error(f"✗ DSS Core V3 import失敗: {e}")
            dss_available_check = False
        
        # backtester.dss_coreの状態
        logger.info(f"backtester.dss_core: {backtester.dss_core}")
        logger.info(f"dss_available global: {backtester.dss_available if hasattr(backtester, 'dss_available') else 'Not defined'}")
        
        # nikkei225_screenerの状態
        logger.info(f"backtester.nikkei225_screener: {backtester.nikkei225_screener}")
        
        # advanced_ranking_engineの状態
        logger.info(f"backtester.advanced_ranking_engine: {backtester.advanced_ranking_engine}")
        
        # --- CHECK POINT 2: _get_optimal_symbol()詳細実行 ---
        logger.info("=== CHECK POINT 2: _get_optimal_symbol()詳細実行 ===")
        
        # 元のメソッドを一時的にラップして詳細ログを追加
        original_method = backtester._get_optimal_symbol
        
        def wrapped_get_optimal_symbol(target_date, target_symbols=None):
            logger.info(f"🔍 _get_optimal_symbol()呼び出し開始: date={target_date}, symbols={target_symbols}")
            
            try:
                # Step 1: ensure_components()
                logger.info("📋 Step 1: ensure_components()実行")
                backtester.ensure_components()
                logger.info("✓ ensure_components()完了")
                
                # Step 2: ensure_advanced_ranking()
                logger.info("📋 Step 2: ensure_advanced_ranking()実行")  
                backtester.ensure_advanced_ranking()
                logger.info("✓ ensure_advanced_ranking()完了")
                
                # Step 3: DSS Core V3チェック
                logger.info("📋 Step 3: DSS Core V3チェック")
                logger.info(f"   backtester.dss_core: {backtester.dss_core}")
                
                # dss_availableの確認方法を調べる
                try:
                    import src.dssms.dssms_integrated_main
                    dss_available = getattr(src.dssms.dssms_integrated_main, 'dss_available', None)
                    logger.info(f"   dss_available global: {dss_available}")
                except:
                    logger.info("   dss_available global: 取得失敗")
                
                if backtester.dss_core and dss_available_check:
                    logger.info("✓ DSS Core V3条件満たす - DSS実行予定")
                    
                    # DSS Core V3実行
                    logger.info("🚀 DSS Core V3 run_daily_selection()実行")
                    try:
                        dss_result = backtester.dss_core.run_daily_selection(target_date)
                        logger.info(f"   DSS結果: {dss_result}")
                        
                        selected_symbol = dss_result.get('selected_symbol')
                        logger.info(f"   選択銘柄: {selected_symbol}")
                        
                        if selected_symbol:
                            logger.info(f"🎯 DSS Core V3成功 - 銘柄選択: {selected_symbol}")
                            return selected_symbol
                        else:
                            logger.warning("⚠️ DSS Core V3からselected_symbol=None")
                    except Exception as e:
                        logger.error(f"❌ DSS Core V3実行エラー: {e}")
                        logger.error(f"   エラー詳細: {str(e)}")
                        import traceback
                        logger.error(f"   トレースバック: {traceback.format_exc()}")
                else:
                    logger.warning(f"❌ DSS Core V3条件不足 - フォールバックへ (dss_core: {backtester.dss_core}, dss_available: {dss_available_check})")
                
                # Step 4: フォールバック処理
                logger.info("📋 Step 4: フォールバック処理")
                if backtester.nikkei225_screener:
                    logger.info("✓ Nikkei225Screener利用可能")
                    
                    try:
                        logger.info("🔍 フィルター済み銘柄取得開始")
                        available_funds = backtester.portfolio_value * 0.8
                        logger.info(f"   利用可能資金: {available_funds:,.0f}円")
                        
                        filtered_symbols = backtester.nikkei225_screener.get_filtered_symbols(available_funds)
                        logger.info(f"   フィルター済み銘柄数: {len(filtered_symbols)}")
                        logger.info(f"   フィルター済み銘柄(先頭10件): {filtered_symbols[:10] if len(filtered_symbols) > 10 else filtered_symbols}")
                        
                        if filtered_symbols:
                            logger.info("🚀 AdvancedRanking選択開始")
                            
                            # フォールバック機能の確認
                            try:
                                from src.config.system_modes import get_fallback_policy, ComponentType
                                fallback_available = True
                                logger.info("✓ フォールバック機能利用可能")
                            except ImportError:
                                fallback_available = False
                                logger.info("❌ フォールバック機能利用不可")
                            
                            if fallback_available:
                                logger.info("🔄 フォールバック経由でadvanced_ranking_selection実行")
                                fallback_policy = get_fallback_policy()
                                selected = fallback_policy.handle_component_failure(
                                    component_type=ComponentType.DSSMS_CORE,
                                    component_name="DSSMSIntegratedBacktester._get_optimal_symbol",
                                    error=RuntimeError("DSS Core V3 unavailable"),
                                    fallback_func=lambda: backtester._advanced_ranking_selection(filtered_symbols, target_date),
                                    context={
                                        "target_date": target_date.isoformat(),
                                        "available_symbols": len(filtered_symbols),
                                        "portfolio_value": backtester.portfolio_value
                                    }
                                )
                            else:
                                logger.info("🔄 直接advanced_ranking_selection実行")
                                selected = backtester._advanced_ranking_selection(filtered_symbols, target_date)
                            
                            logger.info(f"🎯 AdvancedRanking選択結果: {selected}")
                            return selected
                        
                        else:
                            logger.error("❌ フィルター済み銘柄が0件")
                    
                    except Exception as e:
                        logger.error(f"❌ フォールバック処理エラー: {e}")
                        import traceback
                        logger.error(f"   トレースバック: {traceback.format_exc()}")
                else:
                    logger.error("❌ Nikkei225Screener利用不可")
                
                # 最終的にNoneを返す場合
                logger.error("💀 全ての選択方法が失敗 - None返却")
                return None
                
            except Exception as e:
                logger.error(f"💥 _get_optimal_symbol()で予期しないエラー: {e}")
                import traceback
                logger.error(f"   トレースバック: {traceback.format_exc()}")
                return None
        
        # ラップしたメソッドを実行
        result = wrapped_get_optimal_symbol(target_date)
        logger.info(f"🏁 最終結果: {result}")
        
        # --- CHECK POINT 3: 結果分析 ---
        logger.info("=== CHECK POINT 3: 結果分析 ===")
        
        if result is None:
            logger.error("❌ 分析結果: symbol=None が確認されました")
            logger.error("   原因調査が必要です")
        else:
            logger.info(f"✓ 分析結果: 銘柄選択成功 - {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"💥 詳細分析で予期しないエラー: {e}")
        import traceback
        logger.error(f"トレースバック: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    print("=== 日次処理詳細調査: _get_optimal_symbol()内部動作分析 ===")
    result = analyze_get_optimal_symbol_detailed()
    print(f"調査結果: {result}")
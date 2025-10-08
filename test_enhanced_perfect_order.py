"""
修正版Perfect Order検出テスト
MultiIndex対応とパーフェクトオーダー検出率改善
"""
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from config.error_handling import fetch_stock_data

logger = setup_logger("perfect_order_test")

def normalize_data_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    MultiIndex列を正規化してClose列アクセスを可能にする
    """
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex の場合、最初のレベル（Price）を使用
        data_normalized = data.copy()
        data_normalized.columns = [col[0] for col in data.columns]
        logger.debug("MultiIndex columns normalized")
        return data_normalized
    return data

def detect_enhanced_perfect_order(data: pd.DataFrame, symbol: str) -> dict:
    """
    改良版パーフェクトオーダー検出
    
    検出パターン:
    1. 厳密なPerfect Order: 価格 > SMA5 > SMA25 > SMA75
    2. 準Perfect Order: 価格 > SMA5 > SMA25 (SMA75無視)
    3. 上昇トレンド: 価格 > SMA5 かつ SMA5が上向き
    """
    try:
        # データ正規化
        data_norm = normalize_data_columns(data)
        
        if 'Close' not in data_norm.columns:
            logger.error(f"Close column not found. Available: {list(data_norm.columns)}")
            return {"error": "Close column not found"}
        
        close_prices = data_norm['Close'].dropna()
        if len(close_prices) < 75:
            return {"error": "Insufficient data (< 75 days)"}
        
        # SMA計算
        sma5 = close_prices.rolling(window=5).mean()
        sma25 = close_prices.rolling(window=25).mean()
        sma75 = close_prices.rolling(window=75).mean()
        
        # 最新値取得
        current_price = close_prices.iloc[-1]
        sma5_current = sma5.iloc[-1]
        sma25_current = sma25.iloc[-1]
        sma75_current = sma75.iloc[-1]
        
        # 過去の傾向確認
        sma5_prev = sma5.iloc[-5] if len(sma5) >= 5 else sma5_current
        
        # 各種判定
        strict_perfect = current_price > sma5_current > sma25_current > sma75_current
        semi_perfect = current_price > sma5_current > sma25_current
        uptrend = current_price > sma5_current and sma5_current > sma5_prev
        
        # 月別統計計算
        monthly_stats = []
        for month in range(1, 13):
            try:
                month_start = datetime(2023, month, 1)
                if month == 12:
                    month_end = datetime(2023, 12, 31)
                else:
                    month_end = datetime(2023, month + 1, 1)
                
                month_mask = (data_norm.index >= month_start) & (data_norm.index < month_end)
                month_data = data_norm[month_mask]
                
                if len(month_data) < 25:
                    continue
                
                month_close = month_data['Close'].dropna()
                month_sma5 = month_close.rolling(window=5).mean()
                month_sma25 = month_close.rolling(window=25).mean()
                
                # 月末判定
                if len(month_sma25.dropna()) > 0:
                    month_price = month_close.iloc[-1]
                    month_sma5_val = month_sma5.iloc[-1]
                    month_sma25_val = month_sma25.iloc[-1]
                    
                    month_semi_perfect = month_price > month_sma5_val > month_sma25_val
                    
                    monthly_stats.append({
                        "month": month,
                        "price": month_price,
                        "sma5": month_sma5_val,
                        "sma25": month_sma25_val,
                        "semi_perfect": month_semi_perfect
                    })
                    
            except Exception as e:
                logger.warning(f"Month {month} analysis failed: {e}")
        
        # 検出回数計算
        semi_perfect_months = sum(1 for stat in monthly_stats if stat['semi_perfect'])
        
        result = {
            "symbol": symbol,
            "current_price": float(current_price),
            "sma5": float(sma5_current),
            "sma25": float(sma25_current),
            "sma75": float(sma75_current),
            "strict_perfect_order": strict_perfect,
            "semi_perfect_order": semi_perfect,
            "uptrend": uptrend,
            "monthly_stats": monthly_stats,
            "semi_perfect_months": semi_perfect_months,
            "total_months": len(monthly_stats)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced perfect order detection: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def main():
    """メイン実行"""
    logger.info("[SEARCH] 改良版パーフェクトオーダー検出テスト開始")
    
    # データ取得
    data = fetch_stock_data("7203", "2023-01-01", "2023-12-31")
    
    # 改良版検出実行
    result = detect_enhanced_perfect_order(data, "7203")
    
    if "error" in result:
        logger.error(f"[ERROR] 検出エラー: {result['error']}")
        return
    
    # 結果表示
    logger.info("=" * 60)
    logger.info("改良版パーフェクトオーダー検出結果")
    logger.info("=" * 60)
    
    logger.info(f"銘柄: {result['symbol']}")
    logger.info(f"現在価格: {result['current_price']:.2f}")
    logger.info(f"SMA5: {result['sma5']:.2f}")
    logger.info(f"SMA25: {result['sma25']:.2f}")
    logger.info(f"SMA75: {result['sma75']:.2f}")
    
    logger.info(f"\n判定結果:")
    logger.info(f"  厳密Perfect Order: {result['strict_perfect_order']}")
    logger.info(f"  準Perfect Order: {result['semi_perfect_order']}")
    logger.info(f"  上昇トレンド: {result['uptrend']}")
    
    logger.info(f"\n2023年月別統計:")
    logger.info(f"  準Perfect Order発生月数: {result['semi_perfect_months']}/{result['total_months']}")
    
    # 月別詳細
    for stat in result['monthly_stats']:
        status = "[OK]" if stat['semi_perfect'] else "[ERROR]"
        logger.info(f"  {stat['month']:2d}月: {status} 価格={stat['price']:.0f}, SMA5={stat['sma5']:.0f}, SMA25={stat['sma25']:.0f}")
    
    # 判定
    if result['semi_perfect_months'] > 0:
        logger.info(f"\n[OK] 準Perfect Order検出成功: {result['semi_perfect_months']}回")
        logger.info("これで売買シグナル生成が可能になります")
    else:
        logger.error("[ERROR] 依然として検出されていません。戦略の見直しが必要です")

if __name__ == "__main__":
    main()

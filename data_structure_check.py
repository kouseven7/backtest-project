"""
データ構造確認スクリプト
取得したデータの構造を確認し、パーフェクトオーダー検出の問題を特定
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

logger = setup_logger("data_structure_check")

def check_data_structure():
    """データ構造の確認"""
    logger.info("=" * 60)
    logger.info("データ構造確認")
    logger.info("=" * 60)
    
    # テストデータ取得
    try:
        data = fetch_stock_data("7203", "2023-01-01", "2023-12-31")
        
        logger.info(f"データ形状: {data.shape}")
        logger.info(f"列名: {list(data.columns)}")
        logger.info(f"インデックス: {data.index.name}")
        logger.info(f"データ型:")
        for col in data.columns:
            logger.info(f"  {col}: {data[col].dtype}")
        
        logger.info(f"\n最初の5行:")
        logger.info(f"{data.head()}")
        
        logger.info(f"\n最後の5行:")
        logger.info(f"{data.tail()}")
        
        # MultiIndex確認
        if isinstance(data.columns, pd.MultiIndex):
            logger.info("MultiIndex列を検出!")
            logger.info(f"レベル数: {data.columns.nlevels}")
            for level in range(data.columns.nlevels):
                logger.info(f"レベル{level}: {data.columns.get_level_values(level).unique()}")
            
            # 列を平坦化
            logger.info("\n列の平坦化テスト:")
            if data.columns.nlevels == 2:
                data.columns = [col[0] if col[1] == '7203.T' else f"{col[0]}_{col[1]}" 
                               for col in data.columns]
                logger.info(f"平坦化後の列名: {list(data.columns)}")
        
        # Close列の存在確認
        close_candidates = [col for col in data.columns if 'Close' in str(col)]
        logger.info(f"Close列候補: {close_candidates}")
        
        if close_candidates:
            close_col = close_candidates[0]
            logger.info(f"Close列使用: {close_col}")
            
            # 基本統計
            close_data = data[close_col].dropna()
            logger.info(f"Close価格統計:")
            logger.info(f"  期間: {len(close_data)}日")
            logger.info(f"  開始価格: {close_data.iloc[0]:.2f}")
            logger.info(f"  終了価格: {close_data.iloc[-1]:.2f}")
            logger.info(f"  最高価格: {close_data.max():.2f}")
            logger.info(f"  最低価格: {close_data.min():.2f}")
            logger.info(f"  年間リターン: {((close_data.iloc[-1] / close_data.iloc[0]) - 1) * 100:.2f}%")
            
            # SMA計算テスト
            logger.info(f"\nSMA計算テスト:")
            sma5 = close_data.rolling(window=5).mean().iloc[-1]
            sma25 = close_data.rolling(window=25).mean().iloc[-1]
            sma75 = close_data.rolling(window=75).mean().iloc[-1]
            
            logger.info(f"SMA5: {sma5:.2f}")
            logger.info(f"SMA25: {sma25:.2f}")
            logger.info(f"SMA75: {sma75:.2f}")
            logger.info(f"現在価格: {close_data.iloc[-1]:.2f}")
            
            # パーフェクトオーダー判定
            is_perfect = close_data.iloc[-1] > sma5 > sma25 > sma75
            logger.info(f"パーフェクトオーダー: {is_perfect}")
            
            if not is_perfect:
                logger.info("パーフェクトオーダーではない理由:")
                logger.info(f"  現在価格 > SMA5: {close_data.iloc[-1]} > {sma5:.2f} = {close_data.iloc[-1] > sma5}")
                logger.info(f"  SMA5 > SMA25: {sma5:.2f} > {sma25:.2f} = {sma5 > sma25}")
                logger.info(f"  SMA25 > SMA75: {sma25:.2f} > {sma75:.2f} = {sma25 > sma75}")
        
        return True
        
    except Exception as e:
        logger.error(f"データ構造確認エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_simple_perfect_order():
    """簡易パーフェクトオーダー検出テスト"""
    logger.info("=" * 60)
    logger.info("簡易パーフェクトオーダー検出テスト")
    logger.info("=" * 60)
    
    try:
        data = fetch_stock_data("7203", "2023-01-01", "2023-12-31")
        
        # 列構造を正規化
        if isinstance(data.columns, pd.MultiIndex):
            # MultiIndex の場合、最初のレベルを使用
            data.columns = [col[0] for col in data.columns]
        
        logger.info(f"正規化後の列名: {list(data.columns)}")
        
        if 'Close' in data.columns:
            close_prices = data['Close'].dropna()
            
            # 2023年の月ごとのパーフェクトオーダー確認
            perfect_months = 0
            
            for month in range(1, 13):
                try:
                    month_start = datetime(2023, month, 1)
                    if month == 12:
                        month_end = datetime(2023, 12, 31)
                    else:
                        month_end = datetime(2023, month + 1, 1)
                    
                    month_data = data[(data.index >= month_start) & (data.index < month_end)]
                    if len(month_data) < 75:
                        continue
                    
                    month_close = month_data['Close'].dropna()
                    if len(month_close) < 75:
                        continue
                    
                    # SMA計算
                    sma5 = month_close.rolling(window=5).mean().iloc[-1]
                    sma25 = month_close.rolling(window=25).mean().iloc[-1]
                    sma75 = month_close.rolling(window=75).mean().iloc[-1]
                    current = month_close.iloc[-1]
                    
                    is_perfect = current > sma5 > sma25 > sma75
                    
                    logger.info(f"{month}月末: 価格={current:.2f}, SMA5={sma5:.2f}, SMA25={sma25:.2f}, SMA75={sma75:.2f}, Perfect={is_perfect}")
                    
                    if is_perfect:
                        perfect_months += 1
                        
                except Exception as e:
                    logger.warning(f"{month}月のチェックでエラー: {e}")
            
            logger.info(f"\n2023年のパーフェクトオーダー月数: {perfect_months}/12")
            
            if perfect_months == 0:
                logger.error("❌ 2023年中にパーフェクトオーダーが一度も発生していません")
                logger.error("これは以下の可能性があります:")
                logger.error("1. 2023年は日本株が下落基調だった")
                logger.error("2. パーフェクトオーダーの定義が厳しすぎる")
                logger.error("3. SMA期間設定が適切でない")
            else:
                logger.info(f"✅ パーフェクトオーダーが{perfect_months}回発生")
        
        return True
        
    except Exception as e:
        logger.error(f"簡易テストエラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """メイン実行"""
    logger.info("🔍 データ構造・パーフェクトオーダー確認開始")
    
    # データ構造確認
    check_data_structure()
    
    # 簡易パーフェクトオーダーテスト
    test_simple_perfect_order()
    
    logger.info("確認完了")

if __name__ == "__main__":
    main()

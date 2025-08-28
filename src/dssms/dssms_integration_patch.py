
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf

def update_symbol_ranking_with_real_data(symbols: List[str], date: datetime) -> Dict[str, float]:
    """修正版: 実データベースのシンボルランキング更新"""
    scores = {}
    
    for symbol in symbols:
        try:
            # データ取得期間を短縮（パフォーマンス向上）
            end_date = date
            start_date = date - timedelta(days=5)  # 5日間のデータ
            
            # yfinanceでデータ取得
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty and len(data) >= 2:
                # テクニカル分析によるスコア計算
                closes = data['Close'].values
                volumes = data['Volume'].values
                
                # 1. 価格モメンタム (30%)
                price_change = (closes[-1] - closes[0]) / closes[0]
                momentum_score = max(0, min(1, (price_change + 0.1) / 0.2))  # -10%～+10%を0-1に正規化
                
                # 2. ボラティリティ (20%) - 低ボラティリティを好む
                volatility = np.std(closes) / np.mean(closes)
                volatility_score = max(0, min(1, 1 - volatility))
                
                # 3. 出来高 (20%)
                avg_volume = np.mean(volumes)
                volume_score = min(1, avg_volume / 1000000)  # 100万株基準
                
                # 4. トレンド (30%)
                if len(closes) >= 3:
                    trend = (closes[-1] - closes[-3]) / closes[-3]
                    trend_score = max(0, min(1, (trend + 0.05) / 0.1))
                else:
                    trend_score = 0.5
                
                # 総合スコア計算
                total_score = (
                    momentum_score * 0.3 +
                    volatility_score * 0.2 +
                    volume_score * 0.2 +
                    trend_score * 0.3
                )
                
                scores[symbol] = float(total_score)
                
            else:
                # データが不十分な場合はランダムスコア（低め）
                scores[symbol] = np.random.uniform(0.1, 0.4)
                
        except Exception as e:
            # エラー時は低スコア
            scores[symbol] = np.random.uniform(0.05, 0.2)
            print(f"スコア計算エラー {symbol}: {e}")
    
    return scores

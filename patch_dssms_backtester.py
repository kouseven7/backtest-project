"""
DSSMSバックテスター修正パッチ
適用方法: import patch_dssms_backtester; patch_dssms_backtester.apply_patch()
"""

def apply_patch():
    """DSSMSバックテスターに修正を適用"""
    import sys
    import os
    
    # パッチ1: yfinanceデータ取得の改善
    def improved_data_fetch(symbol, start_date, end_date):
        import yfinance as yf
        from datetime import datetime, timedelta
        
        try:
            # 日本株の場合は.Tサフィックスを追加
            ticker_symbol = symbol if symbol.endswith('.T') else f"{symbol}.T"
            
            # データ取得期間を調整（開始日を少し前に設定）
            adjusted_start = start_date - timedelta(days=7)
            
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(start=adjusted_start, end=end_date, period="1d")
            
            if data.empty:
                print(f"Warning: No data for {ticker_symbol}")
                return None
            
            # タイムゾーン設定
            if data.index.tz is None:
                data.index = data.index.tz_localize('Asia/Tokyo')
            
            # 必要な期間にフィルタ
            data = data[data.index >= start_date]
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    # パッチ2: パフォーマンス計算の修正
    def improved_portfolio_value_extraction(portfolio_history):
        """ポートフォリオ履歴から値を適切に抽出"""
        values = []
        for record in portfolio_history:
            if isinstance(record, dict):
                if 'portfolio_value' in record:
                    values.append(record['portfolio_value'])
                elif 'value' in record:
                    values.append(record['value'])
            elif isinstance(record, (int, float)):
                values.append(record)
        return values
    
    print("DSSMSバックテスター修正パッチ適用完了")
    
    return {
        "improved_data_fetch": improved_data_fetch,
        "improved_portfolio_value_extraction": improved_portfolio_value_extraction
    }

if __name__ == "__main__":
    apply_patch()

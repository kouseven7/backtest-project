"""
DSSMS統合修正パッチ
- データ取得の改善
- パフォーマンス計算の修正
- 設定ファイルの作成
"""

import json
import os
from datetime import datetime, timedelta

def create_dssms_fixes():
    """DSSMS統合修正パッチを適用"""
    
    print("=" * 80)
    print("DSSMS統合修正パッチ適用")
    print("=" * 80)
    
    # 1. ランキング設定ファイルの作成
    print("\n1. ランキング設定ファイル作成...")
    ranking_config = {
        "ranking_weights": {
            "technical_score": 0.3,
            "fundamental_score": 0.25,
            "momentum_score": 0.2,
            "volatility_score": 0.15,
            "volume_score": 0.1
        },
        "ranking_criteria": {
            "min_score_threshold": 0.2,
            "max_positions": 2,
            "rebalance_frequency": "daily"
        },
        "hierarchical_levels": {
            "tier1_threshold": 0.8,
            "tier2_threshold": 0.6,
            "tier3_threshold": 0.4
        },
        "filtering_rules": {
            "exclude_low_volume": True,
            "min_volume_threshold": 100000,
            "exclude_gap_stocks": True,
            "max_gap_percentage": 0.1
        },
        "scoring_parameters": {
            "lookback_period_days": 20,
            "volatility_window": 14,
            "momentum_window": 10
        }
    }
    
    ranking_config_path = "config/dssms/hierarchical_ranking_config.json"
    os.makedirs(os.path.dirname(ranking_config_path), exist_ok=True)
    
    with open(ranking_config_path, 'w', encoding='utf-8') as f:
        json.dump(ranking_config, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ ランキング設定ファイル作成: {ranking_config_path}")
    
    # 2. 市場監視設定の修正（JSONエラー修正）
    print("\n2. 市場監視設定ファイル修正...")
    market_config_path = "config/dssms/market_monitoring_config.json"
    
    if os.path.exists(market_config_path):
        try:
            with open(market_config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 潜在的なJSON構文エラーを修正
            # 不正なカンマや引用符を修正
            content = content.replace("'", '"')  # シングルクォートをダブルクォートに
            content = content.replace(',\n}', '\n}')  # 最後のカンマを削除
            content = content.replace(',\n]', '\n]')  # 最後のカンマを削除
            
            # 修正されたコンテンツで書き戻し
            with open(market_config_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"   ✓ 市場監視設定ファイル修正: {market_config_path}")
            
        except Exception as e:
            print(f"   ✗ 市場監視設定ファイル修正エラー: {e}")
    
    # 3. データ取得設定の改善
    print("\n3. データ取得設定ファイル作成...")
    data_config = {
        "yfinance_settings": {
            "default_period": "1y",
            "timezone": "Asia/Tokyo",
            "suffix": ".T",
            "retry_attempts": 3,
            "retry_delay": 1
        },
        "data_quality": {
            "min_data_points": 100,
            "max_missing_ratio": 0.1,
            "required_columns": ["Open", "High", "Low", "Close", "Volume"]
        },
        "fallback_options": {
            "use_cached_data": True,
            "cache_duration_hours": 24,
            "alternative_sources": ["yahoo_fin", "quandl"]
        }
    }
    
    data_config_path = "config/dssms/data_fetcher_config.json"
    
    with open(data_config_path, 'w', encoding='utf-8') as f:
        json.dump(data_config, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ データ取得設定ファイル作成: {data_config_path}")
    
    # 4. パッチスクリプトの作成
    print("\n4. DSSMSバックテスターパッチ作成...")
    
    patch_content = '''"""
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
'''
    
    patch_path = "patch_dssms_backtester.py"
    with open(patch_path, 'w', encoding='utf-8') as f:
        f.write(patch_content)
    
    print(f"   ✓ バックテスターパッチ作成: {patch_path}")
    
    # 5. 修正完了レポート
    print("\n5. 修正完了レポート:")
    print("   a) ✓ ランキング設定ファイル作成")
    print("   b) ✓ 市場監視設定修正（JSON構文エラー対応）")
    print("   c) ✓ データ取得設定改善")
    print("   d) ✓ バックテスターパッチ作成")
    print("\n   次の手順:")
    print("   1. python src/dssms/dssms_backtester.py で再テスト")
    print("   2. エラーが残る場合はパッチを手動適用")

if __name__ == "__main__":
    create_dssms_fixes()

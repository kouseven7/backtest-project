"""
DSSMS Phase 1 Simple Demo
日経225スクリーナーとパーフェクトオーダー検出のデモ
"""

import sys
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.dssms.nikkei225_screener import Nikkei225Screener
from src.dssms.dssms_data_manager import DSSMSDataManager
from src.dssms.fundamental_analyzer import FundamentalAnalyzer
from src.dssms.perfect_order_detector import PerfectOrderDetector

def main():
    print("=" * 60)
    print("[ROCKET] DSSMS Phase 1 Simple Demo")
    print("=" * 60)
    
    # 1. 銘柄スクリーニング
    print("\n1. 銘柄スクリーニング...")
    screener = Nikkei225Screener()
    available_funds = 1_000_000  # 100万円
    
    selected_symbols = screener.get_filtered_symbols(available_funds)
    print(f"選定銘柄数: {len(selected_symbols)}")
    print(f"選定銘柄: {', '.join(selected_symbols)}")
    
    if not selected_symbols:
        print("選定された銘柄がありません。")
        return
    
    # 2. データ取得
    print("\n2. マルチタイムフレームデータ取得...")
    data_manager = DSSMSDataManager()
    
    # 最大3銘柄でテスト
    test_symbols = selected_symbols[:3]
    batch_data = data_manager.batch_get_multi_timeframe_data(test_symbols)
    
    print(f"データ取得完了: {len(batch_data)} 銘柄")
    
    # 3. 業績分析
    print("\n3. 業績分析...")
    analyzer = FundamentalAnalyzer()
    
    fundamental_results = analyzer.batch_analyze_fundamentals(test_symbols)
    
    print("業績分析結果:")
    for symbol, result in fundamental_results.items():
        print(f"  {symbol}: スコア {result['fundamental_score']:.3f}")
        print(f"    営業利益黒字: {result['operating_profit_positive']}")
        print(f"    連続増益: {result['consecutive_growth']}")
        print(f"    コンセンサス予想超え: {result['consensus_beat']}")
    
    # 4. パーフェクトオーダー検出
    print("\n4. パーフェクトオーダー検出...")
    detector = PerfectOrderDetector()
    
    def data_provider(symbol):
        return batch_data.get(symbol, {})
    
    available_symbols = [s for s in test_symbols if s in batch_data]
    perfect_order_results = detector.batch_analyze_symbols(available_symbols, data_provider)
    
    print("パーフェクトオーダー分析結果:")
    for result in perfect_order_results:
        print(f"  {result.symbol}:")
        print(f"    優先度レベル: {result.priority_level}")
        print(f"    総合スコア: {result.composite_score:.3f}")
        print(f"    日足パーフェクトオーダー: {result.daily_result.is_perfect_order}")
        print(f"    週足パーフェクトオーダー: {result.weekly_result.is_perfect_order}")
        print(f"    月足パーフェクトオーダー: {result.monthly_result.is_perfect_order}")
        print(f"    現在価格: {result.daily_result.current_price:.2f}")
        print(f"    日足SMA(5): {result.daily_result.sma_short:.2f}")
        print(f"    日足SMA(25): {result.daily_result.sma_medium:.2f}")
        print(f"    日足SMA(75): {result.daily_result.sma_long:.2f}")
    
    # 5. 統合ランキング
    print("\n5. 統合ランキング...")
    ranking = []
    
    for symbol in available_symbols:
        fundamental = fundamental_results.get(symbol, {})
        perfect_order = next((r for r in perfect_order_results if r.symbol == symbol), None)
        
        # 複合スコア計算（業績40% + パーフェクトオーダー60%）
        fundamental_score = fundamental.get('fundamental_score', 0.0)
        perfect_order_score = perfect_order.composite_score if perfect_order else 0.0
        
        composite_score = fundamental_score * 0.4 + perfect_order_score * 0.6
        
        ranking.append({
            'symbol': symbol,
            'composite_score': composite_score,
            'fundamental_score': fundamental_score,
            'perfect_order_score': perfect_order_score,
            'priority_level': perfect_order.priority_level if perfect_order else 3
        })
    
    # スコア順でソート
    ranking.sort(key=lambda x: (x['priority_level'], -x['composite_score']))
    
    print("統合ランキング:")
    print("順位 | 銘柄 | 総合スコア | 業績スコア | パーフェクトオーダー | 優先度")
    print("-" * 70)
    
    for i, item in enumerate(ranking, 1):
        print(f"{i:2d}位 | {item['symbol']:4s} | "
              f"{item['composite_score']:8.3f} | "
              f"{item['fundamental_score']:8.3f} | "
              f"{item['perfect_order_score']:15.3f} | "
              f"{item['priority_level']:6d}")
    
    # 推奨銘柄
    if ranking:
        top_symbol = ranking[0]
        print(f"\n[TARGET] 推奨銘柄: {top_symbol['symbol']}")
        print(f"   総合スコア: {top_symbol['composite_score']:.3f}")
        print(f"   優先度レベル: {top_symbol['priority_level']}")
    
    print("\n" + "=" * 60)
    print("[OK] DSSMS Phase 1 Demo 完了")
    print("=" * 60)

if __name__ == "__main__":
    main()

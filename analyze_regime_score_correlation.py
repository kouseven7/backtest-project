"""
レジーム別スコア相関分析スクリプト

目的: 市場レジームとGC Strategyスコアの関係を詳細に分析
"""
import re
import sys
from pathlib import Path
from collections import defaultdict

def find_latest_backtest_folder():
    """最新のバックテストフォルダを自動検出"""
    backtest_dir = Path("output/dssms_integration")
    
    if not backtest_dir.exists():
        print(f"エラー: {backtest_dir} が見つかりません")
        return None
    
    # backtest_YYYYMMDD_HHMMSS形式またはdssms_YYYYMMDD_HHMMSS形式のフォルダを検索
    backtest_folders = sorted(backtest_dir.glob("backtest_*"), reverse=True)
    dssms_folders = sorted(backtest_dir.glob("dssms_*"), reverse=True)
    
    all_folders = backtest_folders + dssms_folders
    all_folders.sort(reverse=True, key=lambda p: p.name)
    
    if not all_folders:
        print(f"エラー: {backtest_dir} にバックテストフォルダが見つかりません")
        return None
    
    return all_folders[0]

# コマンドライン引数または自動検出
if len(sys.argv) > 1:
    LOG_DIR = Path(sys.argv[1])
else:
    LOG_DIR = find_latest_backtest_folder()
    if LOG_DIR is None:
        sys.exit(1)
    print(f"最新のバックテストフォルダを使用: {LOG_DIR}")
    print()

# ログファイルパス
ALL_LOG = LOG_DIR / f"dssms_all_detailed_{LOG_DIR.name.replace('backtest_', '')}.log"
SCORE_LOG = LOG_DIR / f"strategy_selection_{LOG_DIR.name.replace('backtest_', '')}.log"

if not ALL_LOG.exists():
    print(f"エラー: {ALL_LOG} が見つかりません")
    sys.exit(1)
if not SCORE_LOG.exists():
    print(f"エラー: {SCORE_LOG} が見つかりません")
    sys.exit(1)

def extract_market_regimes():
    """dssms_all_detailed_*.logからレジーム情報を抽出"""
    regimes = []
    
    with open(ALL_LOG, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_ticker = None
    for i, line in enumerate(lines):
        # Starting comprehensive market analysis for 9843
        ticker_match = re.search(r'market analysis for (\d+)', line)
        if ticker_match:
            current_ticker = ticker_match.group(1)
        
        # [MARKET_ANALYSIS] Market analysis completed - Regime: sideways, Confidence: 1.00
        if 'Market analysis completed - Regime:' in line and current_ticker:
            regime_match = re.search(r'Regime: (\w+)', line)
            if regime_match:
                regime = regime_match.group(1)
                regimes.append({
                    'ticker': current_ticker,
                    'regime': regime,
                    'line_num': i+1
                })
                # リセット（次の銘柄用）
                current_ticker = None
    
    return regimes

def extract_gc_scores():
    """strategy_selection_*.logからGC Strategyスコアを抽出"""
    scores = []
    
    with open(SCORE_LOG, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_ticker = None
    for i, line in enumerate(lines):
        # Strategy scores calculated for 9843:
        ticker_match = re.search(r'scores calculated for (\d+)', line)
        if ticker_match:
            current_ticker = ticker_match.group(1)
        
        # [SCORE_DETAIL]   - GCStrategy: 0.3450
        if 'GCStrategy:' in line and current_ticker:
            score_match = re.search(r'GCStrategy: ([\d\.]+)', line)
            if score_match:
                score = score_match.group(1)
                scores.append({
                    'ticker': current_ticker,
                    'score': score,
                    'line_num': i+1
                })
    
    return scores

def analyze_correlation(regimes, scores):
    """レジームとスコアの相関を分析"""
    # 銘柄ごとに最初のレジームとスコアをマッピング
    ticker_data = defaultdict(lambda: {'regime': None, 'score': None})
    
    for r in regimes:
        if ticker_data[r['ticker']]['regime'] is None:
            ticker_data[r['ticker']]['regime'] = r['regime']
    
    for s in scores:
        if ticker_data[s['ticker']]['score'] is None:
            ticker_data[s['ticker']]['score'] = s['score']
    
    # レジーム別にスコアをグループ化
    regime_scores = defaultdict(list)
    for ticker, data in ticker_data.items():
        if data['regime'] and data['score']:
            regime_scores[data['regime']].append(float(data['score']))
    
    return regime_scores, ticker_data

def print_results(regime_scores, ticker_data):
    """結果を表示"""
    print("=" * 80)
    print("市場レジーム別 GC Strategy スコア分析")
    print("=" * 80)
    print()
    
    # レジーム別統計
    print("【レジーム別スコア統計】")
    print(f"{'Regime':<20} {'Count':<8} {'AvgScore':<10} {'MinScore':<10} {'MaxScore':<10}")
    print("-" * 70)
    
    for regime in sorted(regime_scores.keys()):
        scores = regime_scores[regime]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        
        print(f"{regime:<20} {len(scores):<8} {avg_score:<10.4f} {min_score:<10.4f} {max_score:<10.4f}")
    
    print()
    print("【ユニークスコア一覧】")
    all_scores = [score for scores in regime_scores.values() for score in scores]
    unique_scores = sorted(set(all_scores))
    for score in unique_scores:
        print(f"  GCStrategy: {score:.4f}")
    
    print()
    print(f"【ユニークスコア数】: {len(unique_scores)} 種類")
    
    print()
    print("【銘柄別詳細（サンプル20件）】")
    print(f"{'Ticker':<10} {'Regime':<20} {'GCScore':<10}")
    print("-" * 50)
    
    count = 0
    for ticker in sorted(ticker_data.keys()):
        data = ticker_data[ticker]
        if data['regime'] and data['score']:
            print(f"{ticker:<10} {data['regime']:<20} {data['score']:<10}")
            count += 1
            if count >= 20:
                break
    
    print()
    print("=" * 80)
    print("【判定】")
    
    if len(unique_scores) >= 3:
        print("✅ 完全成功: 3種類以上のスコアが確認されました")
    elif len(unique_scores) == 2:
        print("⚠️ 部分的成功: 2種類のスコアのみ")
        print("   → テスト期間にdowntrendが存在しない可能性があります")
    else:
        print("❌ 失敗: スコアが固定値です")
    
    print()
    
    # レジーム別に異なるスコアか確認
    regime_has_different_scores = False
    regime_avg_scores = {}
    for regime, scores in regime_scores.items():
        avg = sum(scores) / len(scores)
        regime_avg_scores[regime] = avg
    
    if len(set(regime_avg_scores.values())) > 1:
        print("✅ レジーム適応成功: レジームごとにスコアが異なります")
        regime_has_different_scores = True
    else:
        print("❌ レジーム適応失敗: すべてのレジームで同じスコアです")
    
    print("=" * 80)

if __name__ == "__main__":
    print("市場レジーム別スコア分析を開始...")
    print()
    
    regimes = extract_market_regimes()
    print(f"抽出されたレジーム情報: {len(regimes)} 件")
    
    scores = extract_gc_scores()
    print(f"抽出されたGCスコア: {len(scores)} 件")
    print()
    
    regime_scores, ticker_data = analyze_correlation(regimes, scores)
    
    print_results(regime_scores, ticker_data)

"""
ドローダウン計算の違い分析

現象:
- CSV実測（equity_curve_recorder）: 2.64% (28,051円)
- ログ表示（ComprehensivePerformanceAnalyzer）: 2.36% (23,635円)
- 差分: 0.28% (4,416円)

原因:
1. ComprehensivePerformanceAnalyzer._calculate_max_drawdown()
   - 累積PnLベースの計算
   - drawdown = (累積PnL - 累積PnL最大値) / 初期資本1,000,000円
   - 取引時点のPnLのみを使用

2. equity_curve_recorder.reconstruct_daily_snapshots()
   - Portfolio Valueベースの計算
   - drawdown = (Portfolio Value - Peak Value) / Peak Value
   - 日次でのPortfolio全体を評価（cash + position）

具体例:
【ComprehensivePerformanceAnalyzer】
- 累積PnL最大: 仮に60,000円
- 累積PnL最小: 仮に36,365円（60,000 - 23,635）
- ドローダウン: 23,635 / 1,000,000 = 2.36%

【equity_curve_recorder】
- Peak Value: 1,062,542.97円（2024-07-04）
- Portfolio Value: 1,034,491.24円（2024-08-15）
- ドローダウン: (1,062,542.97 - 1,034,491.24) / 1,062,542.97 = 2.64%

結論:
- 両者は異なる計算方法を使用している
- ComprehensivePerformanceAnalyzer: 取引ベースの累積PnL
- equity_curve_recorder: 日次ベースのPortfolio Value
- どちらも正しいが、参照している値が異なる

推奨:
- 日次のPortfolio Value推移を正確に把握する場合はequity_curve_recorderの2.64%が適切
- 取引PnLのみを評価する場合はComprehensivePerformanceAnalyzerの2.36%が適切
- 一般的には日次Portfolio Valueベース（2.64%）が標準的

修正不要:
- cash_balance変動は正常（17種類確認）
- ドローダウン計算方法の違いによる差分
- 43.60% → 2.64%の大幅改善は達成済み
"""

if __name__ == "__main__":
    print(__doc__)

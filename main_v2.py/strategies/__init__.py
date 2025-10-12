"""
main_v2 戦略管理モジュール

Phase 1 対応:
- VWAPBreakoutStrategy単体実装

Phase 2 対応予定 (main.py実証済み7戦略):
- VWAPBreakoutStrategy (Phase1で実装済み)
- MomentumInvestingStrategy  
- BreakoutStrategy
- VWAPBounceStrategy
- OpeningGapStrategy
- ContrarianStrategy
- GCStrategy

再利用予定モジュール:
- strategies.VWAP_Breakout.VWAPBreakoutStrategy (高優先度)
- strategies.Momentum_Investing.MomentumInvestingStrategy (高優先度)
- strategies.Breakout.BreakoutStrategy (高優先度)
- strategies.VWAP_Bounce.VWAPBounceStrategy (高優先度)
- strategies.Opening_Gap.OpeningGapStrategy (高優先度)
- strategies.contrarian_strategy.ContrarianStrategy (高優先度)
- strategies.gc_strategy_signal.GCStrategy (高優先度)
"""

# TODO: Phase 1実装予定
# 1. VWAPBreakoutStrategy統合テスト
# 2. Entry_Signal/Exit_Signal生成確認
# 3. backtest()メソッド動作確認

class StrategyManager:
    """main_v2.py専用戦略管理クラス"""
    
    def __init__(self):
        self.phase = "Phase 1"
        self.available_strategies = ["VWAPBreakoutStrategy"]
        self.active_strategies = []
        
    def get_strategy_list(self):
        """利用可能戦略リスト取得"""
        return self.available_strategies
        
    def add_strategy(self, strategy_name: str):
        """戦略追加"""
        if strategy_name in self.available_strategies:
            self.active_strategies.append(strategy_name)
            return True
        return False
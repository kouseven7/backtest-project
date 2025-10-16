"""
Module: Risk Management
File: risk_management.py
Description: 
  トレードにおけるリスク管理を行うためのモジュールです。
  最大ドローダウン、1回の取引あたりの損失制限、ポジション管理などを実装しています。

Author: imega
Created: 2023-04-01
Modified: 2025-04-14

Dependencies:
  - None
"""

class RiskManagement:
    def __init__(self, total_assets: float, max_drawdown: float = 0.10, max_loss_per_trade: float = 0.03):
        """
        リスク管理モジュールの初期化。

        Parameters:
            total_assets (float): 総資産額
            max_drawdown (float): 最大許容ドローダウン（デフォルトは10%）
            max_loss_per_trade (float): 1回のトレードで許容される最大損失（デフォルトは3%）
        """
        self.total_assets = total_assets
        self.max_drawdown = max_drawdown
        self.max_loss_per_trade = max_loss_per_trade
        self.current_drawdown = 0.0
        # 短期取引戦略に適した値に調整
        self.per_ticker_limit = 5  # 同一銘柄に対する制限を拡大
        self.daily_losses = 0
        self.max_daily_losses = 3  # 同日での最大連敗数
        self.active_trades = {}  # 各戦略のポジションサイズを管理
        self.ticker_positions = {}  # 銘柄ごとのポジション数を管理
        self.max_total_positions = 100  # 全体で持てるポジションの合計を100までに設定（短期取引向け）

    def check_position_size(self, strategy_name: str, ticker: str = None) -> bool:
        """
        各戦略ごとのポジションサイズが制限を超えないように制限。
        全戦略の合計ポジション数も制限する。
        銘柄指定がある場合は、同一銘柄への集中も制限する。

        Parameters:
            strategy_name (str): 戦略名
            ticker (str, optional): 銘柄コード

        Returns:
            bool: ポジションを持てる場合は True、それ以外は False
        """
        # 各戦略の許容ポジション数を増加（3→10）し、短期取引を活発化
        per_strategy_limit = 10
        
        # 基本チェック：全体の制限と戦略ごとの制限
        if (self.active_trades.get(strategy_name, 0) >= per_strategy_limit or 
                self.get_total_positions() >= self.max_total_positions):
            return False
            
        # 銘柄指定がある場合、同一銘柄への集中も制限
        if ticker is not None and hasattr(self, 'ticker_positions'):
            current_ticker_positions = self.ticker_positions.get(ticker, 0)
            # 同一銘柄への集中を制限（デフォルト2）
            if current_ticker_positions >= self.per_ticker_limit:
                return False
                
        return True
                
    def update_position(self, strategy_name: str, position_size: int, ticker: str = None):
        """
        ポジションサイズを更新。
        銘柄指定がある場合は、銘柄ごとのポジション数も更新。

        Parameters:
            strategy_name (str): 戦略名
            position_size (int): 追加するポジションサイズ
            ticker (str, optional): 銘柄コード
        """
        # 戦略ごとのポジション数更新
        self.active_trades[strategy_name] = self.active_trades.get(strategy_name, 0) + position_size
        
        # 銘柄ごとのポジション数も更新
        if ticker is not None:
            if not hasattr(self, 'ticker_positions'):
                self.ticker_positions = {}
            self.ticker_positions[ticker] = self.ticker_positions.get(ticker, 0) + position_size

    def get_total_positions(self) -> int:
        """
        現在の全体のポジション数を取得。

        Returns:
            int: 全体のポジション数
        """
        return sum(self.active_trades.values())

    def check_drawdown(self, current_assets: float) -> bool:
        """
        ドローダウンが最大許容値を超えていないか確認。

        Parameters:
            current_assets (float): 現在の総資産額

        Returns:
            bool: ドローダウンが許容範囲内の場合は True、それ以外は False
        """
        self.current_drawdown = (self.total_assets - current_assets) / self.total_assets
        return self.current_drawdown <= self.max_drawdown

    def check_loss_per_trade(self, entry_price: float, current_price: float) -> bool:
        """
        1回のトレードでの損失が許容範囲内か確認。

        Parameters:
            entry_price (float): エントリー価格
            current_price (float): 現在の価格

        Returns:
            bool: 損失が許容範囲内の場合は True、それ以外は False
        """
        loss = (entry_price - current_price) / entry_price
        return loss <= self.max_loss_per_trade

    def check_daily_losses(self) -> bool:
        """
        同日での連敗数が許容範囲内か確認。

        Returns:
            bool: 連敗数が許容範囲内の場合は True、それ以外は False
        """
        return self.daily_losses < self.max_daily_losses

    def reset_daily_losses(self):
        """
        1日の連敗数をリセット。
        """
        self.daily_losses = 0

    def stop_trading(self):
        """
        システム全体を停止し、通知を行う。
        """
        print("リスク管理システム: 最大ドローダウンを超えたため、取引を停止します。")
        # 必要に応じて通知機能を追加（例: メール、Slack通知など）
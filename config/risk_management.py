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
        self.daily_losses = 0
        self.max_daily_losses = 3  # 同日での最大連敗数
        self.active_trades = {}  # 各戦略のポジションサイズを管理
        self.max_total_positions = 3  # 全体で持てるポジションの合計を3までに制限

        # 戦略の優先順位
        self.strategy_priority = [
            "VWAP Breakout.py",
            "Momentum Investing.py",
            "Opening Gap.py",
            "VWAP Bounce.py",
            "contrarian strategy.py",
            "Breakout.py",
            "gc_strategy_signal.py"
        ]

    def check_position_size(self, strategy_name: str) -> bool:
        """
        各戦略ごとのポジションサイズが1単元を超えないように制限。

        Parameters:
            strategy_name (str): 戦略名

        Returns:
            bool: ポジションを持てる場合は True、それ以外は False
        """
        if self.active_trades.get(strategy_name, 0) >= 1:
            return False
        if self.get_total_positions() >= self.max_total_positions:
            return False
        return True

    def update_position(self, strategy_name: str, position_size: int):
        """
        ポジションサイズを更新。

        Parameters:
            strategy_name (str): 戦略名
            position_size (int): 追加するポジションサイズ
        """
        self.active_trades[strategy_name] = self.active_trades.get(strategy_name, 0) + position_size

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
        if self.current_drawdown > self.max_drawdown:
            return False
        return True

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
        if self.daily_losses >= self.max_daily_losses:
            return False
        return True

    def reset_daily_losses(self):
        """
        1日の連敗数をリセット。
        """
        self.daily_losses = 0

    def prioritize_strategies(self, signals: dict) -> list:
        """
        戦略の優先順位に基づいてシグナルをソート。

        Parameters:
            signals (dict): 戦略名をキー、シグナル値をバリューとする辞書

        Returns:
            list: 優先順位に基づいてソートされた戦略名のリスト
        """
        return sorted(signals.keys(), key=lambda x: self.strategy_priority.index(x))

    def stop_trading(self):
        """
        システム全体を停止し、通知を行う。
        """
        print("リスク管理システム: 最大ドローダウンを超えたため、取引を停止します。")
        # 必要に応じて通知機能を追加（例: メール、Slack通知など）
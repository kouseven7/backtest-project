# ファイル: strategies/gc_strategy_signal.py
import pandas as pd
import logging
from strategies.base_strategy import BaseStrategy

class GCStrategy(BaseStrategy):
    """
    GC戦略（ゴールデンクロス戦略）の実装クラス。
    短期移動平均と長期移動平均のゴールデンクロス／デッドクロスを基にエントリー／イグジットシグナルを生成し、
    Excelから取得した戦略パラメータ（例: 利益確定％、損切割合％、短期・長期移動平均期間）を反映させます。
    """
    def __init__(self, data: pd.DataFrame, params: dict, price_column: str = "Adj Close"):
        """
        Parameters:
            data (pd.DataFrame): 株価データ
            params (dict): 戦略パラメータ（例: {"短期移動平均": 5, "長期移動平均": 25, ...}）
            price_column (str): インジケーター計算に使用する価格カラム（デフォルトは "Adj Close"）
        """
        super().__init__(data, params)
        self.price_column = price_column
        
        # 指定された価格カラムが存在するか確認、なければ 'Close' を代用
        if self.price_column not in self.data.columns:
            self.logger.warning(
                f"指定された価格カラム '{self.price_column}' が見つかりません。'Close' カラムを代用します。"
            )
            self.price_column = "Close"
        
        # 戦略パラメータの読み込み
        self.short_window = int(self.params.get("短期移動平均", 5))
        self.long_window = int(self.params.get("長期移動平均", 25))
        self.profit_take = float(self.params.get("利益確定％", 5))
        self.stop_loss = float(self.params.get("損切割合％", -3))
        
        self.logger.info(
            "GCStrategy initialized with short_window=%d, long_window=%d, profit_take=%.2f, stop_loss=%.2f",
            self.short_window, self.long_window, self.profit_take, self.stop_loss
        )
        
        # 移動平均の計算（指定した価格カラムを使用）
        self.data[f"SMA_{self.short_window}"] = self.data[self.price_column].rolling(window=self.short_window).mean()
        self.data[f"SMA_{self.long_window}"] = self.data[self.price_column].rolling(window=self.long_window).mean()

    def generate_entry_signal(self, idx: int) -> int:
        """
        指定されたインデックス位置でのエントリーシグナルを生成する。
        短期移動平均が長期移動平均を上回った場合、1を返す。
        """
        short_sma = self.data[f"SMA_{self.short_window}"].iloc[idx]
        long_sma = self.data[f"SMA_{self.long_window}"].iloc[idx]
        if pd.isna(short_sma) or pd.isna(long_sma):
            return 0
        return 1 if short_sma > long_sma else 0

    def generate_exit_signal(self, idx: int) -> int:
        """
        指定されたインデックス位置でのイグジットシグナルを生成する。
        短期移動平均が長期移動平均を下回った場合、-1を返す。
        """
        short_sma = self.data[f"SMA_{self.short_window}"].iloc[idx]
        long_sma = self.data[f"SMA_{self.long_window}"].iloc[idx]
        if pd.isna(short_sma) or pd.isna(long_sma):
            return 0
        return -1 if short_sma < long_sma else 0

    def generate_signals(self) -> pd.DataFrame:
        """
        全データに対してエントリーおよびイグジットシグナルを生成し、DataFrameにシグナルカラムを追加する。
        """
        entry_signals = []
        exit_signals = []
        for i in range(len(self.data)):
            entry_signals.append(self.generate_entry_signal(i))
            exit_signals.append(self.generate_exit_signal(i))
        self.data["Entry_Signal"] = entry_signals
        self.data["Exit_Signal"] = exit_signals
        return self.data

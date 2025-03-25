# ファイル: strategies/base_strategy.py
import logging

class BaseStrategy:
    """
    BaseStrategyは、全戦略に共通する基本処理（パラメータ初期化、エントリー／イグジット判定、ログ出力など）を実装する基底クラスです。
    各戦略は、このクラスを継承して固有のシグナル生成ロジックを実装してください。
    """
    def __init__(self, data, params=None):
        """
        Parameters:
            data (pd.DataFrame): 戦略適用対象の株価データなどの時系列データ
            params (dict, optional): 戦略固有のパラメータ（例: 利益確定%、損切割合%など）
        """
        self.data = data
        self.params = params if params is not None else {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.initialize_strategy()

    def initialize_strategy(self):
        """
        戦略初期化処理。
        各戦略で固有の初期化処理がある場合は、このメソッドをオーバーライドしてください。
        """
        self.logger.info("Strategy initialized with parameters: %s", self.params)

    def generate_entry_signal(self, index):
        """
        指定されたインデックスにおけるエントリーシグナルの判定を行います。
        このメソッドは各戦略でオーバーライドして実装してください。

        Parameters:
            index: シグナル判定対象のデータのインデックス

        Returns:
            signal: エントリーシグナル（例: 1 = エントリー、0 = エントリーなし）
        """
        raise NotImplementedError("Subclasses must implement generate_entry_signal()")

    def generate_exit_signal(self, index):
        """
        指定されたインデックスにおけるイグジットシグナルの判定を行います。
        このメソッドは各戦略でオーバーライドして実装してください。

        Parameters:
            index: シグナル判定対象のデータのインデックス

        Returns:
            signal: イグジットシグナル（例: -1 = イグジット、0 = イグジットなし）
        """
        raise NotImplementedError("Subclasses must implement generate_exit_signal()")

    def log_trade(self, message):
        """
        トレードやシグナルに関する情報をログ出力する共通メソッドです。

        Parameters:
            message (str): ログに記録するメッセージ
        """
        self.logger.info(message)

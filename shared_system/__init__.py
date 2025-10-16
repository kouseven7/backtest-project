"""
shared_system - main.py・DSSMS共有モジュール群

このパッケージはmain.pyとDSSMSシステムで共有可能なユーティリティとデータ処理機能を提供します。

構成:
- common_utils: 共通ユーティリティ（最適化、ファイル操作、監視）
- data_processing: データ処理（データプロセッサ、構造ハンドリング）
- indicators: 共通指標計算（既存のindicators/を維持）
"""

__version__ = "1.0.0"
__author__ = "backtest-project integration team"

# 主要モジュールのインポート
from . import common_utils
from . import data_processing

__all__ = [
    'common_utils',
    'data_processing'
]
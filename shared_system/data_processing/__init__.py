"""
data_processing - データ処理モジュール

このモジュールは以下の機能を提供します:
- data_processor: データプロセッサ
"""

try:
    from .data_processor import *
except ImportError:
    pass

__all__ = [
    'data_processor'
]
"""
data_acquisition - データ取得・キャッシュ・エラーハンドリングモジュール

このモジュールは以下の機能を提供します:
- cache_manager: データキャッシュ管理
- error_handling: 拡張エラーハンドリング
- lazy_import_manager: 遅延インポート管理
- data_feed_integration: データフィード統合
"""

try:
    from .cache_manager import *
except ImportError:
    pass

try:
    from .error_handling import *
except ImportError:
    pass

try:
    from .lazy_import_manager import *
except ImportError:
    pass

try:
    from .data_feed_integration import *
except ImportError:
    pass

__all__ = [
    'cache_manager',
    'error_handling',
    'lazy_import_manager', 
    'data_feed_integration'
]
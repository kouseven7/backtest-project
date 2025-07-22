"""
最小限のmain.py テスト
問題の箇所を特定するためのシンプルなテスト
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime

# プロジェクトのルートディレクトリを sys.path に追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

print("1. 基本インポート完了")

try:
    from config.logger_config import setup_logger
    print("2. Logger設定インポート完了")
except Exception as e:
    print(f"Logger設定インポートエラー: {e}")
    sys.exit(1)

try:
    logger = setup_logger(__name__, log_file=r"C:\Users\imega\Documents\my_backtest_project\logs\test.log")
    print("3. Logger初期化完了")
except Exception as e:
    print(f"Logger初期化エラー: {e}")
    sys.exit(1)

try:
    from data_fetcher import get_parameters_and_data
    print("4. data_fetcher インポート完了")
except Exception as e:
    print(f"data_fetcherインポートエラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("5. 最小限テスト完了")

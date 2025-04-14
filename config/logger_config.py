"""
Module: Logger Configuration
File: logger_config.py
Description: 
  ログ設定を行うためのモジュールです。標準出力やファイル出力のロガーを構築します。

Author: imega
Created: 2023-04-01
Modified: 2025-04-14

Dependencies:
  - logging
  - sys
  - os
"""

import logging
import sys
import os

def setup_logger(name: str, level=logging.INFO, log_file: str = None) -> logging.Logger:
    """
    指定した名前とログレベルでロガーを設定して返す。
    - 標準出力（sys.stdout）にログを出力。
    - log_fileが指定されていれば、ファイルにもログを出力する。
    - 既にハンドラーが設定されている場合は重複して追加しない。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')

    # 標準出力用ハンドラーを追加（重複防止）
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # log_fileが指定されている場合はFileHandlerを追加（重複防止）
    if log_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        # ログディレクトリが存在しない場合は作成
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


"""
backtest_runner.py
ストップロス最適化用バックテスト実行ユーティリティ

役割:
  - gc_strategy_signal.py の stop_loss 値を書き換える
  - run_dssms_with_detailed_logs.py をサブプロセスで実行する
  - 最新の出力ディレクトリを特定して結果を返す
"""

import os
import re
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ========================================================
# 定数（プロジェクト構造に合わせて変更）
# ========================================================

# プロジェクトルートを自動検出（このスクリプトの3階層上）
# stoploss_optimizer/utils/backtest_runner.py → プロジェクトルート
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]  # my_backtest_project/

GC_STRATEGY_FILE = PROJECT_ROOT / "strategies" / "gc_strategy_signal.py"
BACKTEST_SCRIPT = PROJECT_ROOT / "run_dssms_with_detailed_logs.py"
OUTPUT_BASE_DIR = PROJECT_ROOT / "output" / "dssms_integration"

# Pythonインタープリタ（仮想環境を使う場合は変更不要 - subprocess が継承）
PYTHON_CMD = "python"


# ========================================================
# ストップロス閾値の書き換え
# ========================================================

def set_stop_loss_threshold(threshold: float) -> bool:
    """
    gc_strategy_signal.py の default_params["stop_loss"] を書き換える。

    既存の run_gc_optimization_phase2_fixed.py と同じ方式（正規表現）を踏襲。

    Args:
        threshold: 例) -0.05 → 0.05 として書き込む（マイナス不要）

    Returns:
        True: 成功, False: 失敗
    """
    # 内部表現は正の値（例: -5% → 0.05）
    value = abs(threshold)

    if not GC_STRATEGY_FILE.exists():
        logger.error(f"GCStrategyファイルが見つかりません: {GC_STRATEGY_FILE}")
        return False

    with open(GC_STRATEGY_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # バックアップ（上書き前）
    backup_path = GC_STRATEGY_FILE.with_suffix(".py.bak")
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(content)

    # 正規表現で default_params 内の stop_loss のみ置換
    # 対象: "stop_loss": 0.03 や "stop_loss": 0.05 など
    pattern = r'("stop_loss":\s*)[\d.]+'
    replacement = rf'\g<1>{value:.2f}'
    new_content, count = re.subn(pattern, replacement, content)

    if count == 0:
        logger.error("stop_loss の書き換え箇所が見つかりませんでした")
        return False

    with open(GC_STRATEGY_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)

    logger.info(f"stop_loss を {value:.2f} ({value*100:.0f}%) に設定 ({count}箇所書き換え)")
    return True


def get_current_stop_loss() -> float:
    """現在の stop_loss 値を取得する（確認用）"""
    if not GC_STRATEGY_FILE.exists():
        return -1.0

    with open(GC_STRATEGY_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r'"stop_loss":\s*([\d.]+)', content)
    if match:
        return float(match.group(1))
    return -1.0


# ========================================================
# バックテスト実行
# ========================================================

def run_backtest(start_date: str, end_date: str, threshold: float,
                 timeout_minutes: int = 30) -> dict:
    """
    指定期間・閾値でバックテストを実行し、結果を返す。

    Args:
        start_date: "2022-01-01" 形式
        end_date:   "2022-12-31" 形式
        threshold:  -0.05 など（マイナス値）
        timeout_minutes: タイムアウト（デフォルト30分）

    Returns:
        dict: {
            'success': bool,
            'output_dir': Path or None,
            'threshold': float,
            'start_date': str,
            'end_date': str,
            'duration_seconds': float,
            'error': str or None
        }
    """
    result = {
        'success': False,
        'output_dir': None,
        'threshold': threshold,
        'start_date': start_date,
        'end_date': end_date,
        'duration_seconds': 0.0,
        'error': None
    }

    # 1. ストップロス値を設定
    if not set_stop_loss_threshold(threshold):
        result['error'] = "stop_loss 書き換え失敗"
        return result

    # 書き換え後の値を確認
    current = get_current_stop_loss()
    logger.info(f"バックテスト開始: {start_date}〜{end_date}, stop_loss={current:.2f}")

    # 2. バックテスト実行前の出力ディレクトリ一覧を取得
    before_dirs = set(_get_output_dirs())

    # 3. サブプロセスで実行
    cmd = [
        PYTHON_CMD,
        str(BACKTEST_SCRIPT),
        "--start-date", start_date,
        "--end-date", end_date
    ]

    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_minutes * 60,
            cwd=str(PROJECT_ROOT)
        )
        elapsed = time.time() - start_time
        result['duration_seconds'] = elapsed

        if proc.returncode != 0:
            result['error'] = f"終了コード {proc.returncode}: {proc.stderr[-500:]}"
            logger.error(f"バックテスト失敗: {result['error']}")
            return result

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        result['duration_seconds'] = elapsed
        result['error'] = f"タイムアウト ({timeout_minutes}分)"
        logger.error(result['error'])
        return result

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"バックテスト実行エラー: {e}")
        return result

    # 4. 新しく生成された出力ディレクトリを特定
    after_dirs = set(_get_output_dirs())
    new_dirs = after_dirs - before_dirs

    if not new_dirs:
        result['error'] = "出力ディレクトリが生成されませんでした"
        logger.error(result['error'])
        return result

    # 最新のディレクトリを採用（複数の場合は最新）
    output_dir = max(new_dirs, key=lambda p: p.stat().st_mtime)
    result['output_dir'] = output_dir
    result['success'] = True

    logger.info(f"完了: {elapsed:.0f}秒, 出力={output_dir.name}")
    return result


# ========================================================
# 出力ディレクトリの管理
# ========================================================

def _get_output_dirs() -> list[Path]:
    """dssms_YYYYMMDD_HHMMSS 形式のディレクトリ一覧を返す"""
    if not OUTPUT_BASE_DIR.exists():
        return []
    return [
        p for p in OUTPUT_BASE_DIR.iterdir()
        if p.is_dir() and p.name.startswith("dssms_")
    ]


def get_latest_output_dir() -> Path | None:
    """最新の出力ディレクトリを返す（確認用）"""
    dirs = _get_output_dirs()
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)

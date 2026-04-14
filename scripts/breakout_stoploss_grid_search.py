import sys
import re
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
from glob import glob

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

BREAKOUT_PY = project_root / "strategies" / "Breakout.py"
MAIN_PY = project_root / "src" / "dssms" / "dssms_integrated_main.py"
PYTHON = project_root / ".venv-3" / "Scripts" / "python.exe"
OUTPUT_DIR = project_root / "output"
REPORT_DIR = OUTPUT_DIR / "dssms_integration"

STOP_LOSS_CANDIDATES = [0.02, 0.03, 0.05, 0.07, 0.08, 0.10]

FIXED_PARAMS = {
    "take_profit": 0.05,
    "trailing_stop": 1.0,
    "volume_threshold": 1.2,
    "breakout_buffer": 0.01,
}

TEST_PERIODS = [
    ("2018-01-01", "2018-12-31", "2018悪年"),
    ("2020-01-01", "2020-12-31", "2020良年"),
    ("2024-01-01", "2024-12-31", "2024悪年"),
]

def set_breakout_params(stop_loss):
    content = BREAKOUT_PY.read_text(encoding="utf-8")
    for key, val in {**FIXED_PARAMS, "stop_loss": stop_loss}.items():
        pattern = rf'("{key}":\s*)[0-9.]+'
        content = re.sub(pattern, rf"\g<1>{val}", content)
    BREAKOUT_PY.write_text(content, encoding="utf-8")

def verify_params():
    content = BREAKOUT_PY.read_text(encoding="utf-8")
    found = {}
    for key in ["take_profit", "trailing_stop", "stop_loss"]:
        m = re.search(rf'"{key}":\s*([0-9.]+)', content)
        if m:
            found[key] = float(m.group(1))
    return found

def run_backtest(start_date, end_date):
    cmd = [
        str(PYTHON), str(MAIN_PY),
        "--start-date", start_date,
        "--end-date", end_date,
        "--force-strategy", "BreakoutStrategy",
    ]
    print(f"    サブプロセス起動中: {start_date} - {end_date}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(project_root),
    )
    # リアルタイムで進捗を表示
    while True:
        line = process.stdout.readline()
        if not line:
            break
        try:
            decoded = line.decode("cp932", errors="replace").rstrip()
        except Exception:
            decoded = line.decode("utf-8", errors="replace").rstrip()
        # 重要な行だけ表示（ノイズを減らす）
        if any(kw in decoded for kw in ["ERROR", "WARNING", "完了", "取引", "スクリーニング", "screening", "PF", "純利益"]):
            print(f"    [{decoded}]")
    process.wait(timeout=7200)  # 2時間タイムアウト
    if process.returncode != 0:
        raise RuntimeError(f"バックテスト失敗 returncode={process.returncode}")

def get_latest_report(start_date):
    pattern = str(REPORT_DIR / "dssms_*" / "comprehensive_report.txt")
    reports = sorted(glob(pattern), reverse=True)
    if not reports:
        raise FileNotFoundError(f"レポートが見つかりません: {pattern}")
    content = Path(reports[0]).read_text(encoding="utf-8")
    if start_date not in content:
        raise ValueError(f"期間不一致: {start_date} が {reports[0]} にありません")
    return content

def parse_report(content):
    def ex(pat, cast=float):
        m = re.search(pat, content)
        return cast(m.group(1).replace(",", "").replace("¥", "")) if m else 0
    return {
        "trades": ex(r"総取引数:\s*(\d+)", int),
        "winrate": ex(r"勝率:\s*([\d.]+)%"),
        "pf":      ex(r"プロフィットファクター:\s*([\d.]+)"),
        "pnl":     ex(r"純利益:\s*[¥\\]?([-\d,]+)"),
    }

def main():
    print("BreakoutStrategy stop_loss グリッドサーチ開始")
    original = BREAKOUT_PY.read_text(encoding="utf-8")
    results = []

    try:
        for sl in STOP_LOSS_CANDIDATES:
            sl_label = f"{sl*100:.0f}%"
            set_breakout_params(sl)
            print(f"\nstop_loss={sl_label} 設定確認: {verify_params()}")

            for start, end, year_label in TEST_PERIODS:
                print(f"  {year_label} 実行中...")
                try:
                    run_backtest(start, end)
                    content = get_latest_report(start)
                    m = parse_report(content)
                    print(f"  完了: 取引数:{m['trades']} 勝率:{m['winrate']}% PF:{m['pf']} 純損益:{m['pnl']:+,.0f}円")
                    results.append({"stop_loss": sl_label, "year": year_label, **m})
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({"stop_loss": sl_label, "year": year_label,
                                    "trades": 0, "winrate": 0, "pf": 0, "pnl": 0})
    finally:
        BREAKOUT_PY.write_text(original, encoding="utf-8")
        print("\nBreakout.py を元に復元しました")

    if not results:
        print("結果なし")
        return

    df = pd.DataFrame(results)
    out = OUTPUT_DIR / f"breakout_stoploss_grid_result_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"CSV保存: {out}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
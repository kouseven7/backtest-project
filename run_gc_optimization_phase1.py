"""
GC戦略 Phase 1最適化スクリプト

目的: take_profit/trailing_stop_pctの最適組み合わせを18パターンで探索
期間: 2024-01~05, 2025-01~05（5ヶ月×2期間）
評価指標: 総損益+勝率

Author: Backtest Project Team
Created: 2026-01-16
"""

import os
import re
import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd

# パラメータ設定
PARAM_PATTERNS = [
    # ID, take_profit, trailing_stop, stop_loss
    (1, 0.05, 0.03, 0.03),  # 保守的
    (2, 0.05, 0.05, 0.03),  # 中程度trailing
    (3, 0.05, 0.07, 0.03),  # 広めtrailing
    (4, 0.10, 0.03, 0.03),  # 高めtake_profit
    (5, 0.10, 0.05, 0.03),  # バランス型
    (6, 0.10, 0.07, 0.03),  # 広めtrailing+高めtake_profit
    (7, 0.15, 0.03, 0.03),  # 最高take_profit
    (8, 0.15, 0.05, 0.03),  # 高めtake_profit+中程度trailing
    (9, 0.15, 0.07, 0.03),  # 最大拡大型
]

TEST_PERIODS = [
    ("2024Q1", "2024-01-01", "2024-05-31"),  # 2024年前半
    ("2025Q1", "2025-01-01", "2025-05-31"),  # 2025年前半
]

# パス設定
GC_STRATEGY_FILE = "strategies/gc_strategy_signal.py"
BACKUP_FILE = "strategies/gc_strategy_signal_backup_phase1_20260116.py"
OUTPUT_BASE = "output/gc_optimization"
LOG_FILE = f"{OUTPUT_BASE}/execution_log.txt"


def log_message(message: str, to_console: bool = True, to_file: bool = True):
    """ログメッセージを出力"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    
    if to_console:
        print(log_line)
    
    if to_file:
        os.makedirs(OUTPUT_BASE, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")


def create_backup():
    """gc_strategy_signal.pyのバックアップ作成"""
    log_message("=" * 80)
    log_message("Phase 1最適化スクリプト開始")
    log_message("=" * 80)
    
    if not os.path.exists(GC_STRATEGY_FILE):
        log_message(f"ERROR: {GC_STRATEGY_FILE} が見つかりません")
        return False
    
    shutil.copy(GC_STRATEGY_FILE, BACKUP_FILE)
    log_message(f"バックアップ作成完了: {BACKUP_FILE}")
    return True


def update_gc_params(pattern_id: int, take_profit: float, trailing_stop: float, stop_loss: float):
    """gc_strategy_signal.pyのdefault_paramsを書き換え"""
    log_message(f"パターン{pattern_id}: take_profit={take_profit}, trailing_stop={trailing_stop}, stop_loss={stop_loss}")
    
    with open(GC_STRATEGY_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # default_paramsセクションを正規表現で特定
    pattern = r'default_params = \{[^}]+\}'
    
    new_params = f'''default_params = {{
        "short_window": 5,
        "long_window": 25,
        "take_profit": {take_profit},
        "stop_loss": {stop_loss},
        "trailing_stop_pct": {trailing_stop},
        "max_hold_days": 300,
        "exit_on_death_cross": True,
        "trend_filter_enabled": False,
        "allowed_trends": ["uptrend"]
    }}'''
    
    content_new = re.sub(pattern, new_params, content, flags=re.DOTALL)
    
    with open(GC_STRATEGY_FILE, 'w', encoding='utf-8') as f:
        f.write(content_new)
    
    log_message(f"パラメータ更新完了: {GC_STRATEGY_FILE}")


def run_dssms_backtest(start_date: str, end_date: str) -> tuple:
    """DSSMS統合バックテスト実行（Windows CP932対応）"""
    log_message(f"DSSMS実行開始: {start_date} ~ {end_date}")
    
    cmd = [
        'python', '-m', 'src.dssms.dssms_integrated_main',
        '--start-date', start_date,
        '--end-date', end_date
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='cp932',  # Windows環境対応
            errors='replace',   # デコードエラー時の代替文字
            timeout=600  # 10分タイムアウト
        )
        
        log_message(f"DSSMS実行完了: exit_code={result.returncode}")
        
        if result.returncode != 0:
            log_message(f"WARNING: DSSMS実行がエラーで終了しました")
            log_message(f"STDERR: {result.stderr[:500]}")  # 最初の500文字のみ
        
        return result.returncode, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        log_message("ERROR: DSSMS実行がタイムアウトしました（10分）")
        return -1, "", "Timeout"
    except Exception as e:
        log_message(f"ERROR: DSSMS実行中に例外が発生しました: {e}")
        return -1, "", str(e)


def find_latest_dssms_output() -> str:
    """最新のDSSMS出力ディレクトリを取得"""
    dssms_base = "output/dssms_integration"
    
    if not os.path.exists(dssms_base):
        log_message(f"ERROR: {dssms_base} が見つかりません")
        return None
    
    # dssms_YYYYMMDD_HHMMSSパターンのディレクトリを検索
    dirs = [d for d in os.listdir(dssms_base) if d.startswith("dssms_")]
    
    if not dirs:
        log_message(f"ERROR: {dssms_base} 内にDSSMS出力ディレクトリが見つかりません")
        return None
    
    # 最新のディレクトリを取得
    latest_dir = sorted(dirs, reverse=True)[0]
    full_path = os.path.join(dssms_base, latest_dir)
    
    log_message(f"最新DSSMS出力: {full_path}")
    return full_path


def analyze_transactions(csv_path: str) -> dict:
    """all_transactions.csvから統計を計算"""
    if not os.path.exists(csv_path):
        log_message(f"ERROR: {csv_path} が見つかりません")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            log_message("WARNING: 取引データが0件です")
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'avg_return': 0,
                'win_rate': 0,
                'max_profit': 0,
                'max_loss': 0
            }
        
        total_trades = len(df)
        total_pnl = df['pnl'].sum()
        avg_return = df['return_pct'].mean() * 100
        wins = len(df[df['pnl'] > 0])
        win_rate = (wins / total_trades) * 100
        max_profit = df['pnl'].max()
        max_loss = df['pnl'].min()
        
        result = {
            'total_trades': total_trades,
            'total_pnl': round(total_pnl, 2),
            'avg_return': round(avg_return, 2),
            'win_rate': round(win_rate, 1),
            'max_profit': round(max_profit, 2),
            'max_loss': round(max_loss, 2)
        }
        
        log_message(f"集計結果: 取引数={total_trades}, 総損益={total_pnl:.0f}円, 勝率={win_rate:.1f}%")
        return result
    
    except Exception as e:
        log_message(f"ERROR: CSV集計中に例外が発生しました: {e}")
        return None


def save_pattern_results(pattern_id: int, period_name: str, dssms_output_dir: str, 
                         params: dict, stats: dict):
    """パターン結果を保存"""
    pattern_dir = f"{OUTPUT_BASE}/pattern_{pattern_id:02d}_{period_name}"
    os.makedirs(pattern_dir, exist_ok=True)
    
    # all_transactions.csvをコピー
    src_csv = os.path.join(dssms_output_dir, "all_transactions.csv")
    dst_csv = os.path.join(pattern_dir, "all_transactions.csv")
    
    if os.path.exists(src_csv):
        shutil.copy(src_csv, dst_csv)
        log_message(f"取引CSVコピー: {dst_csv}")
    
    # execution_results.jsonをコピー（存在する場合）
    src_json = os.path.join(dssms_output_dir, "execution_results.json")
    dst_json = os.path.join(pattern_dir, "execution_results.json")
    
    if os.path.exists(src_json):
        shutil.copy(src_json, dst_json)
    
    # pattern_info.jsonを作成
    pattern_info = {
        'pattern_id': pattern_id,
        'period': period_name,
        'parameters': params,
        'statistics': stats,
        'timestamp': datetime.now().isoformat()
    }
    
    info_path = os.path.join(pattern_dir, "pattern_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(pattern_info, f, indent=2, ensure_ascii=False)
    
    log_message(f"パターン情報保存: {info_path}")


def generate_summary(all_results: list):
    """全パターンの比較表とbest_params.jsonを生成"""
    log_message("=" * 80)
    log_message("結果分析開始")
    log_message("=" * 80)
    
    # CSVサマリー作成
    summary_path = f"{OUTPUT_BASE}/optimization_summary.csv"
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    log_message(f"比較表生成完了: {summary_path}")
    
    # 最適パラメータ特定（総損益+勝率の複合評価）
    best_score = -float('inf')
    best_pattern = None
    
    for result in all_results:
        # 評価式: score = total_pnl + (win_rate * 1000)
        # 理由: 総損益重視だが、勝率50%と60%の差（+10,000円相当）も考慮
        score = result['total_pnl'] + (result['win_rate'] * 1000)
        
        if score > best_score:
            best_score = score
            best_pattern = result
    
    # best_params.json保存
    if best_pattern:
        best_params = {
            'best_pattern_id': best_pattern['pattern_id'],
            'best_period': best_pattern['period'],
            'parameters': {
                'take_profit': best_pattern['take_profit'],
                'trailing_stop': best_pattern['trailing_stop'],
                'stop_loss': best_pattern['stop_loss']
            },
            'statistics': {
                'total_trades': best_pattern['total_trades'],
                'total_pnl': best_pattern['total_pnl'],
                'win_rate': best_pattern['win_rate'],
                'avg_return': best_pattern['avg_return'],
                'max_profit': best_pattern['max_profit'],
                'max_loss': best_pattern['max_loss']
            },
            'score': best_score,
            'timestamp': datetime.now().isoformat()
        }
        
        best_path = f"{OUTPUT_BASE}/best_params.json"
        with open(best_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        
        log_message("=" * 80)
        log_message("最適パラメータ特定完了")
        log_message(f"パターンID: {best_pattern['pattern_id']} ({best_pattern['period']})")
        log_message(f"take_profit: {best_pattern['take_profit']}, trailing_stop: {best_pattern['trailing_stop']}")
        log_message(f"総損益: {best_pattern['total_pnl']:.0f}円, 勝率: {best_pattern['win_rate']:.1f}%")
        log_message(f"評価スコア: {best_score:.0f}")
        log_message("=" * 80)


def restore_backup():
    """gc_strategy_signal.pyを復元"""
    if os.path.exists(BACKUP_FILE):
        shutil.copy(BACKUP_FILE, GC_STRATEGY_FILE)
        log_message(f"原状復元完了: {GC_STRATEGY_FILE}")
        return True
    else:
        log_message(f"WARNING: バックアップファイルが見つかりません: {BACKUP_FILE}")
        return False


def main():
    """メイン実行"""
    start_time = datetime.now()
    all_results = []
    
    # 1. バックアップ作成
    if not create_backup():
        log_message("ERROR: バックアップ作成に失敗しました。処理を中断します。")
        return
    
    # 2. 18パターン実行ループ
    total_patterns = len(PARAM_PATTERNS) * len(TEST_PERIODS)
    current_pattern = 0
    
    for pattern_id, take_profit, trailing_stop, stop_loss in PARAM_PATTERNS:
        for period_name, start_date, end_date in TEST_PERIODS:
            current_pattern += 1
            
            log_message("=" * 80)
            log_message(f"パターン {current_pattern}/{total_patterns}: ID={pattern_id}, 期間={period_name}")
            log_message("=" * 80)
            
            # 2-1. パラメータ書き換え
            update_gc_params(pattern_id, take_profit, trailing_stop, stop_loss)
            
            # 2-2. DSSMS実行
            exit_code, stdout, stderr = run_dssms_backtest(start_date, end_date)
            
            if exit_code != 0:
                log_message(f"WARNING: パターン{pattern_id}({period_name})はエラーで終了しました。スキップします。")
                continue
            
            # 2-3. 結果取得
            dssms_output_dir = find_latest_dssms_output()
            
            if not dssms_output_dir:
                log_message(f"ERROR: パターン{pattern_id}({period_name})のDSSMS出力が見つかりません。スキップします。")
                continue
            
            csv_path = os.path.join(dssms_output_dir, "all_transactions.csv")
            stats = analyze_transactions(csv_path)
            
            if not stats:
                log_message(f"ERROR: パターン{pattern_id}({period_name})の集計に失敗しました。スキップします。")
                continue
            
            # 2-4. 結果保存
            params = {
                'take_profit': take_profit,
                'trailing_stop': trailing_stop,
                'stop_loss': stop_loss
            }
            
            save_pattern_results(pattern_id, period_name, dssms_output_dir, params, stats)
            
            # 2-5. 結果リストに追加
            result_record = {
                'pattern_id': pattern_id,
                'period': period_name,
                'take_profit': take_profit,
                'trailing_stop': trailing_stop,
                'stop_loss': stop_loss,
                **stats
            }
            all_results.append(result_record)
            
            log_message(f"パターン{pattern_id}({period_name})完了")
    
    # 3. 結果分析
    if all_results:
        generate_summary(all_results)
    else:
        log_message("ERROR: 有効な結果が1つもありません")
    
    # 4. 原状復元
    restore_backup()
    
    # 5. 実行時間レポート
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    
    log_message("=" * 80)
    log_message("Phase 1最適化スクリプト完了")
    log_message(f"実行時間: {elapsed_time}")
    log_message(f"成功パターン数: {len(all_results)}/{total_patterns}")
    log_message("=" * 80)


if __name__ == "__main__":
    main()

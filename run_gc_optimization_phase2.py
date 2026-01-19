"""
GC戦略 Phase 2最適化スクリプト - stop_loss最適化 + 動的トレーリングストップ実装

Phase 2-1: stop_loss最適化（6テスト）
  - stop_loss: 2%, 3%, 5%（3段階）
  - Phase 1最適パラメータ（take_profit=15%, trailing_stop=5%）を固定
  - テスト期間: 2024-01~05, 2025-01~05

Phase 2-2: 動的トレーリングストップ実装（18テスト）
  - 実装方式: 段階的（案A）
  - 閾値1（+5%時点）: 4%, 5%, 6%（3パターン）
  - 閾値2（+10%時点）: 7%, 8%, 10%（3パターン）
  - 組み合わせ: 3 × 3 = 9パターン × 2期間 = 18テスト

主な機能:
- update_gc_params_phase2_1(): stop_loss最適化用パラメータ書き換え
- update_gc_params_phase2_2(): 動的トレーリングストップ実装用パラメータ書き換え
- run_dssms_backtest(): DSSMS統合バックテスト実行（CP932対応、10分タイムアウト）
- analyze_transactions(): all_transactions.csv統計計算
- find_best_params(): 最適パラメータ特定（score = total_pnl + win_rate×1000）
- save_summary(): 全パターン結果をCSV+JSON形式で保存

統合コンポーネント:
- gc_strategy_signal.py: パラメータ書き換え対象（default_params、動的トレーリングストップ実装）
- dssms_integrated_main.py: DSSMS統合バックテスト実行
- all_transactions.csv: バックテスト結果（統計計算元データ）

セーフティ機能/注意事項:
- バックアップ自動作成: gc_strategy_signal_backup_phase2_YYYYMMDD.py
- 原状復元機能: 全テスト完了後、gc_strategy_signal.pyを復元
- エラー処理: 各パターン実行時のエラーを記録、次パターンへ継続
- 10分タイムアウト: 長時間実行を防止
- Windows CP932対応: 日本語出力を正しく処理

Author: Backtest Project Team
Created: 2026-01-17
Last Modified: 2026-01-17
"""

import subprocess
import shutil
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import re
import time

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent

# 出力先ディレクトリ
OUTPUT_DIR = PROJECT_ROOT / "output" / "gc_optimization_phase2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# gc_strategy_signal.pyのパス
GC_STRATEGY_FILE = PROJECT_ROOT / "strategies" / "gc_strategy_signal.py"
GC_STRATEGY_BACKUP = PROJECT_ROOT / "strategies" / f"gc_strategy_signal_backup_phase2_{datetime.now().strftime('%Y%m%d')}.py"

# Phase 2-1: stop_loss最適化パターン（6テスト）
# Phase 1最適パラメータ（take_profit=15%, trailing_stop=5%）を固定
PHASE2_1_PATTERNS = [
    # stop_loss 2%
    {"id": "10A", "period": "2024Q1", "start": "2024-01-01", "end": "2024-05-31", 
     "take_profit": 0.15, "trailing_stop": 0.05, "stop_loss": 0.02, "dynamic": False},
    {"id": "10B", "period": "2025Q1", "start": "2025-01-01", "end": "2025-05-31", 
     "take_profit": 0.15, "trailing_stop": 0.05, "stop_loss": 0.02, "dynamic": False},
    # stop_loss 3%（Phase 1最優秀、ベースライン）
    {"id": "11A", "period": "2024Q1", "start": "2024-01-01", "end": "2024-05-31", 
     "take_profit": 0.15, "trailing_stop": 0.05, "stop_loss": 0.03, "dynamic": False},
    {"id": "11B", "period": "2025Q1", "start": "2025-01-01", "end": "2025-05-31", 
     "take_profit": 0.15, "trailing_stop": 0.05, "stop_loss": 0.03, "dynamic": False},
    # stop_loss 5%
    {"id": "12A", "period": "2024Q1", "start": "2024-01-01", "end": "2024-05-31", 
     "take_profit": 0.15, "trailing_stop": 0.05, "stop_loss": 0.05, "dynamic": False},
    {"id": "12B", "period": "2025Q1", "start": "2025-01-01", "end": "2025-05-31", 
     "take_profit": 0.15, "trailing_stop": 0.05, "stop_loss": 0.05, "dynamic": False},
]

# Phase 2-2: 動的トレーリングストップ実装パターン（18テスト）
# 閾値1（+5%時点）: 4%, 5%, 6%、閾値2（+10%時点）: 7%, 8%, 10%
PHASE2_2_PATTERNS = []
pattern_id_base = 13  # パターンID開始番号
for threshold1 in [0.04, 0.05, 0.06]:  # 閾値1（+5%時点の拡大率）
    for threshold2 in [0.07, 0.08, 0.10]:  # 閾値2（+10%時点の拡大率）
        # 2024年期間
        PHASE2_2_PATTERNS.append({
            "id": f"{pattern_id_base}A",
            "period": "2024Q1",
            "start": "2024-01-01",
            "end": "2024-05-31",
            "take_profit": 0.15,
            "trailing_stop": 0.05,  # 初期値（動的トレーリングストップで上書き）
            "stop_loss": 0.03,  # Phase 1最優秀
            "dynamic": True,
            "threshold1": threshold1,
            "threshold2": threshold2,
        })
        # 2025年期間
        PHASE2_2_PATTERNS.append({
            "id": f"{pattern_id_base}B",
            "period": "2025Q1",
            "start": "2025-01-01",
            "end": "2025-05-31",
            "take_profit": 0.15,
            "trailing_stop": 0.05,
            "stop_loss": 0.03,
            "dynamic": True,
            "threshold1": threshold1,
            "threshold2": threshold2,
        })
        pattern_id_base += 1

# 全パターン統合
ALL_PATTERNS = PHASE2_1_PATTERNS + PHASE2_2_PATTERNS


def backup_gc_strategy():
    """gc_strategy_signal.pyをバックアップ"""
    print(f"バックアップ作成: {GC_STRATEGY_BACKUP.name}")
    shutil.copy(GC_STRATEGY_FILE, GC_STRATEGY_BACKUP)


def restore_gc_strategy():
    """gc_strategy_signal.pyを復元"""
    print(f"原状復元: {GC_STRATEGY_FILE.name}")
    shutil.copy(GC_STRATEGY_BACKUP, GC_STRATEGY_FILE)


def update_gc_params_phase2_1(take_profit: float, trailing_stop: float, stop_loss: float):
    """
    Phase 2-1: stop_loss最適化用パラメータ書き換え
    gc_strategy_signal.pyのdefault_paramsセクションを書き換え（動的トレーリングストップなし）
    """
    with open(GC_STRATEGY_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    # default_paramsセクションを正規表現で検索・置換
    pattern = r'default_params\s*=\s*\{[^}]*\}'
    new_params = f'''default_params = {{
        "take_profit": {take_profit},
        "trailing_stop_pct": {trailing_stop},
        "stop_loss": {stop_loss},
        "max_hold_days": 300,
    }}'''
    
    content = re.sub(pattern, new_params, content, flags=re.DOTALL)
    
    # 動的トレーリングストップ実装部分を削除（Phase 2-1では不要）
    # _get_dynamic_trailing_stop()メソッドが存在する場合は削除
    dynamic_method_pattern = r'\n\s+def _get_dynamic_trailing_stop\(self[^)]*\)[^:]*:.*?(?=\n\s+def |\nclass |\Z)'
    content = re.sub(dynamic_method_pattern, '', content, flags=re.DOTALL)
    
    # generate_exit_signal()内の動的トレーリングストップ呼び出しを削除
    # trailing_stop_pct = self._get_dynamic_trailing_stop(...) を削除
    dynamic_call_pattern = r'\n\s+trailing_stop_pct\s*=\s*self\._get_dynamic_trailing_stop\([^)]*\)[^\n]*'
    content = re.sub(dynamic_call_pattern, '', content)
    
    with open(GC_STRATEGY_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Phase 2-1パラメータ更新: take_profit={take_profit}, trailing_stop={trailing_stop}, stop_loss={stop_loss}")


def update_gc_params_phase2_2(take_profit: float, trailing_stop: float, stop_loss: float, 
                               threshold1: float, threshold2: float):
    """
    Phase 2-2: 動的トレーリングストップ実装用パラメータ書き換え
    gc_strategy_signal.pyのdefault_paramsセクションを書き換え + 動的トレーリングストップ実装
    
    Args:
        take_profit: 利確閾値
        trailing_stop: トレーリングストップ初期値
        stop_loss: ストップロス閾値
        threshold1: +5%時点のtrailing_stop拡大率
        threshold2: +10%時点のtrailing_stop拡大率
    """
    with open(GC_STRATEGY_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    # default_paramsセクションを正規表現で検索・置換
    pattern = r'default_params\s*=\s*\{[^}]*\}'
    new_params = f'''default_params = {{
        "take_profit": {take_profit},
        "trailing_stop_pct": {trailing_stop},
        "stop_loss": {stop_loss},
        "max_hold_days": 300,
        "dynamic_trailing_threshold1": {threshold1},
        "dynamic_trailing_threshold2": {threshold2},
    }}'''
    
    content = re.sub(pattern, new_params, content, flags=re.DOTALL)
    
    # 動的トレーリングストップメソッド追加（存在しない場合）
    dynamic_method = f'''
    def _get_dynamic_trailing_stop(self, entry_price: float, current_price: float) -> float:
        """
        利益率に応じてtrailing_stop_pctを段階的に拡大
        
        Args:
            entry_price: エントリー価格
            current_price: 現在価格
        
        Returns:
            動的調整後のtrailing_stop_pct
        """
        profit_pct = (current_price - entry_price) / entry_price
        
        threshold1 = self.params.get("dynamic_trailing_threshold1", {threshold1})
        threshold2 = self.params.get("dynamic_trailing_threshold2", {threshold2})
        
        if profit_pct >= 0.10:  # +10%以上
            return threshold2
        elif profit_pct >= 0.05:  # +5%以上
            return threshold1
        else:
            return self.params.get("trailing_stop_pct", {trailing_stop})  # 初期値
'''
    
    # 既存の動的トレーリングストップメソッドを削除
    dynamic_method_pattern = r'\n\s+def _get_dynamic_trailing_stop\(self[^)]*\)[^:]*:.*?(?=\n\s+def |\nclass |\Z)'
    content = re.sub(dynamic_method_pattern, '', content, flags=re.DOTALL)
    
    # generate_exit_signal()メソッドの直前に動的トレーリングストップメソッドを追加
    generate_exit_signal_pattern = r'(\n\s+def generate_exit_signal\(self[^)]*\)[^:]*:)'
    content = re.sub(generate_exit_signal_pattern, dynamic_method + r'\1', content)
    
    # generate_exit_signal()内のtrailing_stop_pct取得部分を動的呼び出しに変更
    # trailing_stop_pct = self.params.get("trailing_stop_pct", 0.03) を置換
    trailing_stop_get_pattern = r'trailing_stop_pct\s*=\s*self\.params\.get\("trailing_stop_pct"[^)]*\)'
    trailing_stop_dynamic_call = 'trailing_stop_pct = self._get_dynamic_trailing_stop(entry_price, current_price)'
    
    # generate_exit_signal()内の最初のtrailing_stop_pct取得のみを置換（複数ある場合は最初のみ）
    # まず、generate_exit_signal()メソッド全体を抽出
    generate_exit_signal_full_pattern = r'(def generate_exit_signal\(self[^)]*\)[^:]*:.*?)(?=\n\s+def |\nclass |\Z)'
    match = re.search(generate_exit_signal_full_pattern, content, flags=re.DOTALL)
    
    if match:
        generate_exit_signal_content = match.group(1)
        # generate_exit_signal()内のtrailing_stop_pct取得を動的呼び出しに変更
        # ただし、entry_priceとcurrent_priceが取得可能な位置に配置
        # 通常、entry_price = position["entry_price"]、current_price = data['Adj Close'].iloc[idx]の後
        # この部分を探して、その直後にtrailing_stop_pct動的呼び出しを配置
        
        # entry_priceとcurrent_priceの取得部分を探す
        entry_price_pattern = r'entry_price\s*=\s*position\["entry_price"\]'
        current_price_pattern = r'current_price\s*=\s*data\[\'Adj Close\'\]\.iloc\[idx\]'
        
        # 既存のtrailing_stop_pct取得を削除
        generate_exit_signal_content = re.sub(trailing_stop_get_pattern, '', generate_exit_signal_content)
        
        # current_price取得の直後にtrailing_stop_pct動的呼び出しを追加
        current_price_insertion_pattern = r'(current_price\s*=\s*data\[\'Adj Close\'\]\.iloc\[idx\][^\n]*\n)'
        generate_exit_signal_content = re.sub(
            current_price_insertion_pattern,
            r'\1        ' + trailing_stop_dynamic_call + '\n',
            generate_exit_signal_content
        )
        
        # 元のcontentに反映
        content = re.sub(generate_exit_signal_full_pattern, generate_exit_signal_content, content, flags=re.DOTALL)
    
    with open(GC_STRATEGY_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Phase 2-2パラメータ更新: take_profit={take_profit}, trailing_stop={trailing_stop}, "
          f"stop_loss={stop_loss}, threshold1={threshold1}, threshold2={threshold2}")


def run_dssms_backtest(start_date: str, end_date: str) -> tuple:
    """
    DSSMS統合バックテスト実行（Windows CP932対応）
    
    Args:
        start_date: 開始日（YYYY-MM-DD）
        end_date: 終了日（YYYY-MM-DD）
    
    Returns:
        (returncode, stdout, stderr)
    """
    cmd = [
        "python", "-m", "src.dssms.dssms_integrated_main",
        "--start-date", start_date,
        "--end-date", end_date
    ]
    
    print(f"実行: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="cp932",  # Windows日本語環境対応
        errors="replace",
        timeout=600  # 10分タイムアウト
    )
    
    return result.returncode, result.stdout, result.stderr


def analyze_transactions(csv_path: str) -> dict:
    """
    all_transactions.csvから統計を計算
    
    Returns:
        統計情報辞書
    """
    try:
        df = pd.read_csv(csv_path)
        
        total_trades = len(df)
        total_pnl = df["pnl"].sum()
        avg_return = df["return_pct"].mean() * 100
        wins = len(df[df["pnl"] > 0])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        max_profit = df["pnl"].max()
        max_loss = df["pnl"].min()
        
        return {
            "total_trades": total_trades,
            "total_pnl": round(total_pnl, 2),
            "avg_return": round(avg_return, 2),
            "win_rate": round(win_rate, 1),
            "max_profit": round(max_profit, 2),
            "max_loss": round(max_loss, 2),
        }
    except Exception as e:
        print(f"統計計算エラー: {e}")
        return {
            "total_trades": 0,
            "total_pnl": 0,
            "avg_return": 0,
            "win_rate": 0,
            "max_profit": 0,
            "max_loss": 0,
        }


def find_best_params(results: list) -> dict:
    """
    最適パラメータを特定
    
    評価スコア = 総損益 + 勝率×1000
    
    Returns:
        最適パターン情報
    """
    best_pattern = None
    best_score = float("-inf")
    
    for result in results:
        if result["status"] != "success":
            continue
        
        stats = result["stats"]
        score = stats["total_pnl"] + stats["win_rate"] * 1000
        
        if score > best_score:
            best_score = score
            best_pattern = {
                "best_pattern_id": result["pattern_id"],
                "best_period": result["period"],
                "parameters": {
                    "take_profit": result["params"]["take_profit"],
                    "trailing_stop": result["params"]["trailing_stop"],
                    "stop_loss": result["params"]["stop_loss"],
                    "dynamic": result["params"]["dynamic"],
                },
                "statistics": stats,
                "score": round(score, 2),
                "timestamp": datetime.now().isoformat(),
            }
            
            # 動的トレーリングストップの場合は閾値を追加
            if result["params"]["dynamic"]:
                best_pattern["parameters"]["threshold1"] = result["params"]["threshold1"]
                best_pattern["parameters"]["threshold2"] = result["params"]["threshold2"]
    
    return best_pattern


def save_summary(results: list):
    """全パターン結果をCSV+JSON形式で保存"""
    # CSV形式
    csv_data = []
    for result in results:
        row = {
            "pattern_id": result["pattern_id"],
            "period": result["period"],
            "take_profit": result["params"]["take_profit"],
            "trailing_stop": result["params"]["trailing_stop"],
            "stop_loss": result["params"]["stop_loss"],
            "dynamic": result["params"]["dynamic"],
            "status": result["status"],
        }
        
        # 動的トレーリングストップの場合は閾値を追加
        if result["params"]["dynamic"]:
            row["threshold1"] = result["params"]["threshold1"]
            row["threshold2"] = result["params"]["threshold2"]
        else:
            row["threshold1"] = None
            row["threshold2"] = None
        
        if result["status"] == "success":
            stats = result["stats"]
            row.update({
                "total_trades": stats["total_trades"],
                "total_pnl": stats["total_pnl"],
                "avg_return": stats["avg_return"],
                "win_rate": stats["win_rate"],
                "max_profit": stats["max_profit"],
                "max_loss": stats["max_loss"],
            })
        else:
            row.update({
                "total_trades": 0,
                "total_pnl": 0,
                "avg_return": 0,
                "win_rate": 0,
                "max_profit": 0,
                "max_loss": 0,
            })
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_path = OUTPUT_DIR / "optimization_summary_phase2.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"比較表保存: {csv_path}")
    
    # JSON形式（詳細情報）
    json_path = OUTPUT_DIR / "all_results_phase2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"詳細情報保存: {json_path}")


def main():
    print("=" * 80)
    print("GC戦略 Phase 2最適化スクリプト開始")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # ログファイル作成
    log_file = OUTPUT_DIR / "execution_log_phase2.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"GC戦略 Phase 2最適化スクリプト実行ログ\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
    
    # バックアップ作成
    backup_gc_strategy()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"バックアップ作成: {GC_STRATEGY_BACKUP.name}\n\n")
    
    # Phase 2-1 + Phase 2-2実行
    results = []
    
    print(f"全{len(ALL_PATTERNS)}パターン実行開始")
    print()
    
    for i, pattern in enumerate(ALL_PATTERNS, 1):
        pattern_id = pattern["id"]
        period = pattern["period"]
        start_date = pattern["start"]
        end_date = pattern["end"]
        take_profit = pattern["take_profit"]
        trailing_stop = pattern["trailing_stop"]
        stop_loss = pattern["stop_loss"]
        dynamic = pattern["dynamic"]
        
        print(f"[{i}/{len(ALL_PATTERNS)}] パターン{pattern_id}-{period} 実行中...")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{i}/{len(ALL_PATTERNS)}] パターン{pattern_id}-{period} 実行中...\n")
            f.write(f"  期間: {start_date} ~ {end_date}\n")
            f.write(f"  take_profit: {take_profit}, trailing_stop: {trailing_stop}, stop_loss: {stop_loss}\n")
            if dynamic:
                f.write(f"  動的トレーリングストップ: threshold1={pattern['threshold1']}, threshold2={pattern['threshold2']}\n")
            else:
                f.write(f"  動的トレーリングストップ: なし\n")
        
        try:
            # パラメータ書き換え
            if dynamic:
                update_gc_params_phase2_2(
                    take_profit, trailing_stop, stop_loss,
                    pattern["threshold1"], pattern["threshold2"]
                )
            else:
                update_gc_params_phase2_1(take_profit, trailing_stop, stop_loss)
            
            # DSSMS実行
            start_time = time.time()
            returncode, stdout, stderr = run_dssms_backtest(start_date, end_date)
            elapsed = time.time() - start_time
            
            if returncode != 0:
                print(f"  エラー: 実行失敗（returncode={returncode}）")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"  エラー: 実行失敗（returncode={returncode}、{elapsed:.1f}秒）\n")
                    f.write(f"  stderr: {stderr[:500]}\n\n")
                
                results.append({
                    "pattern_id": pattern_id,
                    "period": period,
                    "params": pattern,
                    "status": "error",
                    "error": stderr[:500],
                })
                continue
            
            # 最新のall_transactions.csvを探す
            dssms_output_dir = PROJECT_ROOT / "output" / "dssms_integration"
            latest_dir = max(dssms_output_dir.glob("dssms_*"), key=lambda p: p.stat().st_mtime)
            transactions_csv = latest_dir / "all_transactions.csv"
            
            if not transactions_csv.exists():
                print(f"  エラー: all_transactions.csv が見つかりません")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"  エラー: all_transactions.csv が見つかりません（{elapsed:.1f}秒）\n\n")
                
                results.append({
                    "pattern_id": pattern_id,
                    "period": period,
                    "params": pattern,
                    "status": "error",
                    "error": "all_transactions.csv not found",
                })
                continue
            
            # 統計計算
            stats = analyze_transactions(str(transactions_csv))
            
            print(f"  成功: 取引数={stats['total_trades']}件, 総損益={stats['total_pnl']:.0f}円, "
                  f"勝率={stats['win_rate']:.1f}% ({elapsed:.1f}秒)")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"  成功: 取引数={stats['total_trades']}件, 総損益={stats['total_pnl']:.0f}円, "
                        f"勝率={stats['win_rate']:.1f}% ({elapsed:.1f}秒)\n\n")
            
            # 結果保存
            pattern_output_dir = OUTPUT_DIR / f"pattern_{pattern_id}_{period}"
            pattern_output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(transactions_csv, pattern_output_dir / "all_transactions.csv")
            
            # パラメータ情報をJSONで保存
            pattern_info = {
                "pattern_id": pattern_id,
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "parameters": pattern,
                "statistics": stats,
                "elapsed_time": round(elapsed, 1),
            }
            with open(pattern_output_dir / "pattern_info.json", "w", encoding="utf-8") as f:
                json.dump(pattern_info, f, indent=2, ensure_ascii=False)
            
            results.append({
                "pattern_id": pattern_id,
                "period": period,
                "params": pattern,
                "status": "success",
                "stats": stats,
                "elapsed_time": round(elapsed, 1),
            })
        
        except subprocess.TimeoutExpired:
            print(f"  タイムアウト: 10分経過")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"  タイムアウト: 10分経過\n\n")
            
            results.append({
                "pattern_id": pattern_id,
                "period": period,
                "params": pattern,
                "status": "timeout",
                "error": "10分タイムアウト",
            })
        
        except Exception as e:
            print(f"  エラー: {str(e)}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"  エラー: {str(e)}\n\n")
            
            results.append({
                "pattern_id": pattern_id,
                "period": period,
                "params": pattern,
                "status": "error",
                "error": str(e),
            })
        
        print()
    
    # 結果分析
    print("=" * 80)
    print("全パターン実行完了、結果分析中...")
    print("=" * 80)
    print()
    
    # 比較表・詳細情報保存
    save_summary(results)
    
    # 最適パラメータ特定
    best_params = find_best_params(results)
    if best_params:
        best_params_file = OUTPUT_DIR / "best_params_phase2.json"
        with open(best_params_file, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        print(f"最適パラメータ保存: {best_params_file}")
        print()
        print("最適パラメータ（Phase 2）:")
        print(f"  パターンID: {best_params['best_pattern_id']}")
        print(f"  期間: {best_params['best_period']}")
        print(f"  take_profit: {best_params['parameters']['take_profit']}")
        print(f"  trailing_stop: {best_params['parameters']['trailing_stop']}")
        print(f"  stop_loss: {best_params['parameters']['stop_loss']}")
        print(f"  動的トレーリングストップ: {best_params['parameters']['dynamic']}")
        if best_params['parameters']['dynamic']:
            print(f"  threshold1: {best_params['parameters']['threshold1']}")
            print(f"  threshold2: {best_params['parameters']['threshold2']}")
        print(f"  総損益: {best_params['statistics']['total_pnl']:.0f}円")
        print(f"  勝率: {best_params['statistics']['win_rate']:.1f}%")
        print(f"  評価スコア: {best_params['score']:.0f}")
    else:
        print("最適パラメータが見つかりませんでした（全パターン失敗）")
    
    # 原状復元
    print()
    restore_gc_strategy()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n原状復元完了\n")
        f.write(f"実行終了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print()
    print("=" * 80)
    print("Phase 2最適化スクリプト完了")
    print(f"成功パターン: {len([r for r in results if r['status'] == 'success'])}/{len(ALL_PATTERNS)}")
    print(f"出力先: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

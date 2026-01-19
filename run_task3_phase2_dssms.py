"""
Task 3 Phase 2: トレーリングストップ最適化検証（7-12月DSSMS統合テスト）

目的: trailing_stop_pct/take_profitを3%, 5%, 10%, 15%, 20%でテストし、総損益ベースで最適値を特定

実行方法:
1. 各パターンでgc_strategy_signal.pyのデフォルトパラメータを書き換え
2. DSSMS統合バックテストを実行（python -m src.dssms.dssms_integrated_main --start-date 2025-07-01 --end-date 2025-12-31）
3. 結果をoutput/task3_phase2_results/配下に保存

テスト期間: 2025-07-01 ~ 2025-12-31（6ヶ月）
推定所要時間: 約50-75分（5パターン×10-15分）

Author: GitHub Copilot
Created: 2026-01-16
"""
import os
import sys
import pandas as pd
import shutil
import subprocess
from datetime import datetime

# プロジェクトルートを追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# テストパラメータ定義（5パターン）
test_patterns = [
    {"pattern": "A", "trailing_stop_pct": 0.03, "take_profit": 0.03, "description": "保守的（3%)"},
    {"pattern": "B", "trailing_stop_pct": 0.05, "take_profit": 0.05, "description": "現在値ベースライン（5%）"},
    {"pattern": "C", "trailing_stop_pct": 0.10, "take_profit": 0.10, "description": "中程度拡大（10%）"},
    {"pattern": "D", "trailing_stop_pct": 0.15, "take_profit": 0.15, "description": "大幅拡大（15%）"},
    {"pattern": "E", "trailing_stop_pct": 0.20, "take_profit": 0.20, "description": "最大拡大（20%）"},
]

# 共通パラメータ（固定）
common_params = {
    "stop_loss": 0.03,  # 3%固定
    "max_hold_days": 300,  # 実質無効化（yfinanceのデータ制約を割り切り）
    "short_window": 5,
    "long_window": 25,
    "exit_on_death_cross": True,
    "trend_filter_enabled": False,
}

# DSSMS統合テスト期間（7-12月、6ヶ月）
test_start = "2025-07-01"
test_end = "2025-12-31"

# 出力ディレクトリ
output_base_dir = os.path.join(project_root, "output", "task3_phase2_results")
os.makedirs(output_base_dir, exist_ok=True)

# バックアップファイル
gc_strategy_file = os.path.join(project_root, "strategies", "gc_strategy_signal.py")
backup_file = os.path.join(project_root, "strategies", "gc_strategy_signal_backup_task3_phase2_20260116.py")


def backup_gc_strategy():
    """GC戦略ファイルをバックアップ"""
    if not os.path.exists(backup_file):
        shutil.copy2(gc_strategy_file, backup_file)
        print(f"[INFO] GC戦略バックアップ作成: {backup_file}")
    else:
        print(f"[INFO] GC戦略バックアップ既存: {backup_file}")


def update_gc_strategy_params(trailing_stop_pct, take_profit):
    """gc_strategy_signal.pyのデフォルトパラメータを更新"""
    with open(gc_strategy_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # デフォルトパラメータのセクションを特定（Line 54-67付近）
    # "default_params = {" から "}" までを書き換え
    import re
    
    # 正規表現でdefault_paramsセクションを特定
    pattern = r'(default_params = \{)(.*?)(\})'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError("gc_strategy_signal.pyのdefault_paramsセクションが見つかりません")
    
    # 新しいパラメータを生成
    new_params = f'''default_params = {{
            "short_window": {common_params["short_window"]},       # 短期移動平均期間（デフォルト: 5）
            "long_window": {common_params["long_window"]},       # 長期移動平均期間（デフォルト: 25）
            "take_profit": {take_profit},     # [Task 3 Phase 2] 利益確定レベル（テスト中）
            "stop_loss": {common_params["stop_loss"]},       # [Task 3 Phase 2] ストップロスレベル（3%固定）
            "trailing_stop_pct": {trailing_stop_pct},  # [Task 3 Phase 2] トレーリングストップ（テスト中）
            "max_hold_days": {common_params["max_hold_days"]},     # [Task 3 Phase 2] 最大保有期間（300日=実質無効化）
            "exit_on_death_cross": {common_params["exit_on_death_cross"]},  # デッドクロスでイグジット（デフォルト: True）
            "trend_filter_enabled": {common_params["trend_filter_enabled"]},  # トレンドフィルター（デフォルト: False）
            "allowed_trends": ["uptrend"]  # 許可するトレンド（デフォルト: 上昇トレンドのみ）
        }}'''
    
    # 書き換え
    content_new = content[:match.start()] + new_params + content[match.end():]
    
    with open(gc_strategy_file, "w", encoding="utf-8") as f:
        f.write(content_new)
    
    print(f"[INFO] gc_strategy_signal.py更新: trailing_stop_pct={trailing_stop_pct}, take_profit={take_profit}")


def run_dssms_backtest(pattern):
    """DSSMS統合バックテストを実行"""
    print(f"\n{'='*80}")
    print(f"パターン{pattern['pattern']}: {pattern['description']}")
    print(f"trailing_stop_pct={pattern['trailing_stop_pct']*100}%, take_profit={pattern['take_profit']*100}%")
    print(f"{'='*80}\n")
    
    # DSSMS統合コマンド実行
    cmd = [
        sys.executable, "-m", "src.dssms.dssms_integrated_main",
        "--start-date", test_start,
        "--end-date", test_end
    ]
    
    print(f"[INFO] コマンド実行: {' '.join(cmd)}")
    print(f"[INFO] 推定所要時間: 約10-15分（6ヶ月）\n")
    
    # UnicodeDecodeError対策: CP932エンコーディング+errors='ignore'
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='cp932', errors='ignore')
    
    if result.returncode != 0:
        print(f"[エラー] バックテスト実行失敗（パターン{pattern['pattern']}）:")
        print(result.stderr)
        return None
    
    print(f"[INFO] バックテスト完了（パターン{pattern['pattern']}）")
    return result.stdout


def extract_latest_dssms_output():
    """最新のDSSMS出力ディレクトリを特定"""
    dssms_output_base = os.path.join(project_root, "output", "dssms_integration")
    
    if not os.path.exists(dssms_output_base):
        raise FileNotFoundError(f"DSSMS出力ディレクトリが見つかりません: {dssms_output_base}")
    
    # dssms_YYYYMMDDHHMMSSディレクトリを取得（最新）
    dssms_dirs = [d for d in os.listdir(dssms_output_base) if d.startswith("dssms_")]
    if not dssms_dirs:
        raise FileNotFoundError(f"DSSMS出力ディレクトリが空です: {dssms_output_base}")
    
    dssms_dirs.sort(reverse=True)
    latest_dir = os.path.join(dssms_output_base, dssms_dirs[0])
    
    print(f"[INFO] 最新DSSMS出力: {latest_dir}")
    return latest_dir


def copy_dssms_results(pattern, latest_dssms_dir):
    """DSSMS出力結果をTask 3用ディレクトリにコピー"""
    pattern_dir = os.path.join(output_base_dir, f"pattern_{pattern['pattern']}")
    os.makedirs(pattern_dir, exist_ok=True)
    
    # コピー対象ファイル
    files_to_copy = [
        "all_transactions.csv",
        "dssms_comprehensive_report.json",
        "dssms_switch_history.csv",
        "execution_results.json"
    ]
    
    for filename in files_to_copy:
        src = os.path.join(latest_dssms_dir, filename)
        dst = os.path.join(pattern_dir, filename)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"[INFO] コピー完了: {filename} -> pattern_{pattern['pattern']}/")
        else:
            print(f"[警告] ファイル不在: {filename}")
    
    # パラメータ情報をJSONで保存
    import json
    param_info = {
        "pattern": pattern["pattern"],
        "description": pattern["description"],
        "trailing_stop_pct": pattern["trailing_stop_pct"],
        "take_profit": pattern["take_profit"],
        "common_params": common_params,
        "test_period": {"start": test_start, "end": test_end},
        "executed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(pattern_dir, "parameters.json"), "w", encoding="utf-8") as f:
        json.dump(param_info, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] パラメータ情報保存: pattern_{pattern['pattern']}/parameters.json")


def analyze_pattern_results(pattern):
    """パターン別結果を分析"""
    pattern_dir = os.path.join(output_base_dir, f"pattern_{pattern['pattern']}")
    transactions_file = os.path.join(pattern_dir, "all_transactions.csv")
    
    if not os.path.exists(transactions_file):
        print(f"[警告] all_transactions.csv不在（パターン{pattern['pattern']}）")
        return {
            "pattern": pattern["pattern"],
            "trailing_stop_pct": pattern["trailing_stop_pct"],
            "take_profit": pattern["take_profit"],
            "total_trades": 0,
            "total_pnl": 0,
            "avg_return_pct": 0,
            "win_rate": 0
        }
    
    df = pd.read_csv(transactions_file)
    
    if len(df) == 0:
        print(f"[警告] 取引なし（パターン{pattern['pattern']}）")
        return {
            "pattern": pattern["pattern"],
            "trailing_stop_pct": pattern["trailing_stop_pct"],
            "take_profit": pattern["take_profit"],
            "total_trades": 0,
            "total_pnl": 0,
            "avg_return_pct": 0,
            "win_rate": 0
        }
    
    # 保有中ポジションを除外（exit_date = NaT）
    df = df[df['exit_date'].notna()].copy()
    
    total_trades = len(df)
    total_pnl = df['pnl'].sum()
    avg_return_pct = df['return_pct'].mean() * 100
    win_rate = (df['pnl'] > 0).sum() / total_trades if total_trades > 0 else 0
    
    return {
        "pattern": pattern["pattern"],
        "trailing_stop_pct": pattern["trailing_stop_pct"],
        "take_profit": pattern["take_profit"],
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "avg_return_pct": avg_return_pct,
        "win_rate": win_rate
    }


def restore_gc_strategy():
    """GC戦略をバックアップから復元"""
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, gc_strategy_file)
        print(f"\n[INFO] GC戦略を復元しました: {gc_strategy_file}")
    else:
        print(f"\n[警告] バックアップファイルが見つかりません: {backup_file}")


def main():
    """メイン実行"""
    print("\n" + "="*80)
    print("Task 3 Phase 2: トレーリングストップ最適化検証（7-12月DSSMS統合テスト）")
    print("="*80 + "\n")
    
    # バックアップ作成
    backup_gc_strategy()
    
    # 結果格納用リスト
    all_results = []
    
    try:
        for pattern in test_patterns:
            print(f"\n{'='*80}")
            print(f"パターン{pattern['pattern']}: {pattern['description']}開始")
            print(f"{'='*80}")
            
            # GC戦略パラメータ更新
            update_gc_strategy_params(pattern["trailing_stop_pct"], pattern["take_profit"])
            
            # DSSMS統合バックテスト実行
            output = run_dssms_backtest(pattern)
            
            if output is None:
                print(f"[警告] パターン{pattern['pattern']}スキップ（実行失敗）")
                continue
            
            # 最新DSSMS出力ディレクトリを取得
            latest_dssms_dir = extract_latest_dssms_output()
            
            # 結果をTask 3用ディレクトリにコピー
            copy_dssms_results(pattern, latest_dssms_dir)
            
            # 結果分析
            result = analyze_pattern_results(pattern)
            all_results.append(result)
            
            print(f"\n[パターン{pattern['pattern']}結果]")
            print(f"  取引数: {result['total_trades']}件")
            print(f"  総損益: {result['total_pnl']:,.0f}円")
            print(f"  平均リターン: {result['avg_return_pct']:.2f}%")
            print(f"  勝率: {result['win_rate']*100:.1f}%")
    
    except KeyboardInterrupt:
        print("\n[INFO] ユーザーによる中断")
    
    except Exception as e:
        print(f"\n[エラー] 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # GC戦略を復元
        restore_gc_strategy()
    
    # 結果比較表を作成
    if all_results:
        print("\n" + "="*80)
        print("Phase 2結果サマリー（7-12月・DSSMS統合）")
        print("="*80 + "\n")
        
        df_results = pd.DataFrame(all_results)
        print(df_results.to_string(index=False))
        
        # CSV保存
        results_csv = os.path.join(output_base_dir, "phase2_summary.csv")
        df_results.to_csv(results_csv, index=False, encoding='utf-8-sig')
        print(f"\n[OUTPUT] {results_csv} 生成完了")
        
        # 最適パターン特定（総損益最大）
        if len(df_results) > 0 and df_results['total_pnl'].max() > 0:
            best_pattern = df_results.loc[df_results['total_pnl'].idxmax()]
            print("\n" + "="*80)
            print("最適パターン（総損益ベース）")
            print("="*80)
            print(f"パターン{best_pattern['pattern']}: trailing_stop_pct={best_pattern['trailing_stop_pct']*100}%, take_profit={best_pattern['take_profit']*100}%")
            print(f"総損益: {best_pattern['total_pnl']:,.0f}円")
            print(f"取引数: {best_pattern['total_trades']}件")
            print(f"平均リターン: {best_pattern['avg_return_pct']:.2f}%")
            print(f"勝率: {best_pattern['win_rate']*100:.1f}%")
        else:
            print("\n[警告] 全パターンで取引なしまたは損失")
    
    print("\n" + "="*80)
    print("Task 3 Phase 2実行完了")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

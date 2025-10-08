#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 1.1: 切替数カウント機構の検証
DSSMSの銘柄切替数が117回→2回に激減する問題の根本原因特定
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import json
import warnings
import traceback

warnings.filterwarnings('ignore')

def diagnose_switch_count_mechanism():
    """切替カウント機構の詳細診断"""
    print("[SEARCH] Task 1.1: 切替数カウント機構の検証開始")
    print("=" * 60)
    
    try:
        # プロジェクトルートをパスに追加
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        sys.path.append(str(project_root / "src"))
        
        # 1. 現在のDSSMSBacktesterの状態確認
        from src.dssms.dssms_backtester import DSSMSBacktester
        
        print("[OK] DSSMSBacktesterのインポート成功")
        
        # 2. バックテスター初期化とswitch_history確認
        backtester = DSSMSBacktester()
        
        print(f"[CHART] 初期化後のswitch_history長: {len(backtester.switch_history)}")
        print(f"[CHART] 初期化後のperformance_history長: {len(backtester.performance_history)}")
        
        # 3. 設定確認
        print(f"\n[TOOL] 切替判定設定確認:")
        print(f"  - min_holding_period_hours: {backtester.min_holding_period_hours}")
        print(f"  - switch_cost_rate: {backtester.switch_cost_rate}")
        
        # 4. 決定論的設定の影響確認
        if hasattr(backtester, 'deterministic_config'):
            config = backtester.deterministic_config
            print(f"📝 決定論的モード設定:")
            for key, value in config.items():
                print(f"  - {key}: {value}")
        
        # 5. 簡単なシミュレーション実行（短期間テスト）
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)  # 10日間のテスト
        symbols = ['7203', '9984', '6758', '8306', '9432']
        
        print(f"\n[TEST] テストシミュレーション実行: {start_date.date()} - {end_date.date()}")
        
        result = backtester.simulate_dynamic_selection(
            start_date=start_date,
            end_date=end_date, 
            symbol_universe=symbols
        )
        
        print(f"[OK] シミュレーション完了")
        print(f"[CHART] 実行後のswitch_history長: {len(backtester.switch_history)}")
        
        # 6. switch_history詳細分析
        if backtester.switch_history:
            print("\n[LIST] 切替履歴詳細（最初の10件）:")
            for i, switch in enumerate(backtester.switch_history[:10]):
                print(f"  {i+1}: {switch.from_symbol} -> {switch.to_symbol} "
                      f"({switch.timestamp.strftime('%Y-%m-%d %H:%M')})")
        else:
            print("[WARNING] switch_historyが空です")
        
        # 7. 切替決定メカニズムの確認
        print(f"\n[TARGET] 切替決定メカニズム:")
        
        # _evaluate_switch_decisionメソッドの存在確認
        if hasattr(backtester, '_evaluate_switch_decision'):
            print("  [OK] _evaluate_switch_decision メソッド存在")
        else:
            print("  [ERROR] _evaluate_switch_decision メソッド未実装")
        
        # intelligent_switch_managerの存在確認
        if hasattr(backtester, 'intelligent_switch_manager'):
            print(f"  [OK] intelligent_switch_manager 存在: {type(backtester.intelligent_switch_manager)}")
        else:
            print("  [ERROR] intelligent_switch_manager 未設定")
        
        # hierarchical_ranking_systemの存在確認
        if hasattr(backtester, 'ranking_system'):
            print(f"  [OK] ranking_system 存在: {type(backtester.ranking_system)}")
        else:
            print("  [ERROR] ranking_system 未設定")
        
        return {
            'success': True,
            'switch_count': len(backtester.switch_history),
            'backtester': backtester,
            'config': {
                'min_holding_period_hours': backtester.min_holding_period_hours,
                'switch_cost_rate': backtester.switch_cost_rate,
                'deterministic_config': getattr(backtester, 'deterministic_config', {})
            }
        }
        
    except Exception as e:
        print(f"[ERROR] 診断エラー: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def diagnose_data_persistence():
    """データ永続化・キャッシュ問題調査"""
    print(f"\n[SEARCH] Task 1.2: データ永続化・キャッシュ問題調査")
    print("-" * 40)
    
    try:
        # 1. backtest_results/dssms_results/ の状態確認
        results_dir = Path("backtest_results/dssms_results/")
        files = []
        if results_dir.exists():
            files = list(results_dir.glob("*"))
            print(f"📁 dssms_results内ファイル数: {len(files)}")
            for file in files[:10]:  # 最初の10件
                print(f"  - {file.name} ({file.stat().st_size} bytes)")
        else:
            print("📁 dssms_resultsディレクトリが存在しません")
        
        # 2. DSSMS関連設定ファイル調査
        config_files = []
        search_patterns = ["*dssms*.json", "*dssms*.config", "*dssms*.cache"]
        for pattern in search_patterns:
            config_files.extend(Path(".").rglob(pattern))
        
        print(f"\n📄 DSSMS関連設定ファイル: {len(config_files)}件")
        for file in config_files[:10]:
            print(f"  - {file}")
        
        # 3. __pycache__確認
        pycache_dirs = list(Path(".").rglob("__pycache__"))
        print(f"\n🗂️ __pycache__ディレクトリ数: {len(pycache_dirs)}")
        
        # 4. logs/dssms/ ディレクトリ確認
        dssms_logs_dir = Path("logs/dssms/")
        dssms_logs = []
        if dssms_logs_dir.exists():
            dssms_logs = list(dssms_logs_dir.glob("*"))
            print(f"📜 DSSMSログファイル数: {len(dssms_logs)}")
            for log_file in dssms_logs:
                print(f"  - {log_file.name}")
        
        return {
            'success': True,
            'results_files_count': len(files),
            'config_files_count': len(config_files),
            'pycache_dirs': len(pycache_dirs),
            'dssms_logs_count': len(dssms_logs)
        }
        
    except Exception as e:
        print(f"[ERROR] データ永続化調査エラー: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def diagnose_engine_integration():
    """統一エンジン統合状況の確認"""
    print(f"\n[SEARCH] Task 1.3: 統一エンジン統合状況確認")
    print("-" * 40)
    
    engine_files = [
        'dssms_unified_output_engine.py',
        'dssms_unified_output_engine_fixed.py',
        'dssms_unified_output_engine_fixed_v3.py',
        'dssms_unified_output_engine_fixed_v4.py'
    ]
    
    engine_status = {}
    
    for engine_file in engine_files:
        engine_path = Path(engine_file)
        if engine_path.exists():
            print(f"[OK] {engine_file} 存在 ({engine_path.stat().st_size} bytes)")
            engine_status[engine_file] = {'exists': True, 'size': engine_path.stat().st_size}
        else:
            print(f"[ERROR] {engine_file} 不存在")
            engine_status[engine_file] = {'exists': False}
    
    # DSSMSExcelExporterV2の確認
    try:
        sys.path.append(str(Path.cwd()))
        from output.dssms_excel_exporter_v2 import DSSMSExcelExporterV2
        print(f"[OK] DSSMSExcelExporterV2 インポート成功")
        engine_status['DSSMSExcelExporterV2'] = {'available': True}
    except Exception as e:
        print(f"[ERROR] DSSMSExcelExporterV2 インポートエラー: {e}")
        engine_status['DSSMSExcelExporterV2'] = {'available': False, 'error': str(e)}
    
    return {
        'success': True,
        'engine_status': engine_status
    }

def main():
    """診断メイン実行"""
    print("[ROCKET] DSSMS切替数激減問題 根本原因診断開始")
    print("=" * 80)
    
    results = {}
    
    # Task 1.1実行
    results['switch_count'] = diagnose_switch_count_mechanism()
    
    # Task 1.2実行
    results['data_persistence'] = diagnose_data_persistence()
    
    # Task 1.3実行
    results['engine_integration'] = diagnose_engine_integration()
    
    # 結果サマリー
    print(f"\n[CHART] 診断結果サマリー")
    print("=" * 40)
    
    if results['switch_count']['success']:
        switch_count = results['switch_count']['switch_count']
        print(f"[OK] 切替機構診断: 成功 (検出された切替数: {switch_count})")
        
        if switch_count < 10:
            print("  [WARNING] 切替数が異常に少ない - 切替判定ロジック調査が必要")
        else:
            print("  [OK] 切替数は正常範囲")
    else:
        print(f"[ERROR] 切替機構診断: 失敗")
    
    if results['data_persistence']['success']:
        print(f"[OK] 永続化診断: 成功")
        print(f"  - 結果ファイル数: {results['data_persistence']['results_files_count']}")
        print(f"  - 設定ファイル数: {results['data_persistence']['config_files_count']}")
    else:
        print(f"[ERROR] 永続化診断: 失敗")
    
    if results['engine_integration']['success']:
        print(f"[OK] エンジン統合診断: 成功")
    else:
        print(f"[ERROR] エンジン統合診断: 失敗")
    
    # 次のアクション提案
    print(f"\n[TARGET] 推奨される次のアクション:")
    if results['switch_count']['success']:
        switch_count = results['switch_count']['switch_count']
        if switch_count < 10:
            print("  1. _evaluate_switch_decision関数の詳細デバッグ")
            print("  2. intelligent_switch_managerの設定確認")
            print("  3. 決定論的モードの影響調査")
        else:
            print("  1. 切替数は正常、出力エンジン側の問題調査")
            print("  2. DSSMSExcelExporterV2の切替数取得ロジック確認")
    
    print("  4. python src\\dssms\\dssms_backtester.py の実行テスト")
    
    # 結果をJSONで保存
    results_file = Path("diagnose_switch_count_results.json")
    try:
        # 診断結果をシリアライズ可能な形式に変換
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: v for k, v in value.items() 
                    if k != 'backtester'  # backtesteオブジェクトは除外
                }
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 診断結果を保存: {results_file}")
    except Exception as e:
        print(f"[WARNING] 結果保存エラー: {e}")
    
    return results

if __name__ == "__main__":
    main()

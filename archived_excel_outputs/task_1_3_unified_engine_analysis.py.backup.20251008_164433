#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 1.3: 統一エンジン影響度分析
各エンジンファイルのswitch_history処理ロジック比較とデータフロー追跡
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import re

def analyze_unified_engine_impact():
    """統一エンジンの影響度分析"""
    print("🔍 Task 1.3: 統一エンジン影響度分析")
    print("=" * 60)
    
    engines_to_analyze = [
        'dssms_unified_output_engine.py',
        'dssms_unified_output_engine_fixed.py',
        'dssms_unified_output_engine_fixed_v3.py',
        'dssms_unified_output_engine_fixed_v4.py'
    ]
    
    analysis_results = {}
    
    for engine_name in engines_to_analyze:
        if Path(engine_name).exists():
            print(f"\n📋 {engine_name} の分析:")
            result = analyze_single_engine(engine_name)
            analysis_results[engine_name] = result
        else:
            print(f"❌ {engine_name} が見つかりません")
            analysis_results[engine_name] = {'exists': False}
    
    return analysis_results

def analyze_single_engine(engine_path):
    """個別エンジンファイルの詳細分析"""
    try:
        with open(engine_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        analysis = {
            'exists': True,
            'file_size': len(content),
            'switch_history_usage': [],
            'data_conversion_methods': [],
            'excel_output_methods': [],
            'date_fixing_methods': [],
            'holding_period_calculation': []
        }
        
        # switch_historyの使用箇所を検索
        switch_matches = re.findall(r'.*switch_history.*', content, re.IGNORECASE)
        analysis['switch_history_usage'] = switch_matches[:5]  # 最初の5件
        
        # データ変換メソッドを検索
        conversion_matches = re.findall(r'def.*convert.*\(.*\):', content)
        analysis['data_conversion_methods'] = conversion_matches
        
        # Excel出力メソッドを検索
        excel_matches = re.findall(r'def.*excel.*\(.*\):', content)
        analysis['excel_output_methods'] = excel_matches
        
        # 日付修正メソッドを検索
        date_matches = re.findall(r'def.*date.*\(.*\):', content)
        analysis['date_fixing_methods'] = date_matches
        
        # 保有期間計算を検索
        holding_matches = re.findall(r'.*holding.*period.*', content, re.IGNORECASE)
        analysis['holding_period_calculation'] = holding_matches[:3]  # 最初の3件
        
        # 詳細情報表示
        print(f"  📊 ファイルサイズ: {analysis['file_size']} 文字")
        print(f"  🔄 switch_history使用箇所: {len(analysis['switch_history_usage'])}件")
        print(f"  📈 データ変換メソッド: {len(analysis['data_conversion_methods'])}件")
        print(f"  📊 Excel出力メソッド: {len(analysis['excel_output_methods'])}件")
        print(f"  📅 日付修正メソッド: {len(analysis['date_fixing_methods'])}件")
        print(f"  ⏱️ 保有期間計算: {len(analysis['holding_period_calculation'])}件")
        
        # 重要メソッドの詳細表示
        if analysis['data_conversion_methods']:
            print(f"    変換メソッド: {analysis['data_conversion_methods']}")
        if analysis['date_fixing_methods']:
            print(f"    日付修正: {analysis['date_fixing_methods']}")
        if analysis['holding_period_calculation']:
            print(f"    保有期間計算: {analysis['holding_period_calculation'][:2]}")
            
        return analysis
        
    except Exception as e:
        print(f"    ❌ 分析エラー: {e}")
        return {'exists': True, 'error': str(e)}

def trace_data_flow():
    """バックテスター→エンジン間のデータフロー追跡"""
    print(f"\n🔄 データフロー追跡分析")
    print("-" * 40)
    
    try:
        # 1. DSSMSBacktesterのswitch_history生成確認
        print("📋 1. DSSMSBacktester.switch_history 生成過程:")
        with open('src/dssms/dssms_backtester.py', 'r', encoding='utf-8') as f:
            backtester_content = f.read()
        
        # switch_historyへの追加箇所を検索
        append_matches = re.findall(r'.*switch_history\.append.*', backtester_content)
        print(f"  - switch_history.append() 呼び出し: {len(append_matches)}件")
        for match in append_matches[:3]:
            print(f"    → {match.strip()}")
        
        # 2. 各エンジンでのswitch_history取得方法確認
        print(f"\n📋 2. 統一エンジンでのswitch_history取得:")
        engines = ['dssms_unified_output_engine.py', 'dssms_unified_output_engine_fixed.py']
        
        for engine in engines:
            if Path(engine).exists():
                with open(engine, 'r', encoding='utf-8') as f:
                    engine_content = f.read()
                
                # switch_historyの取得方法を検索
                get_matches = re.findall(r'.*backtester\.switch_history.*', engine_content)
                print(f"  {engine}:")
                print(f"    - switch_history取得: {len(get_matches)}件")
                for match in get_matches[:2]:
                    print(f"      → {match.strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ データフロー追跡エラー: {e}")
        return False

def identify_switching_logic_changes():
    """切替判定ロジックの変更点特定"""
    print(f"\n🔧 切替判定ロジック変更点特定")
    print("-" * 40)
    
    try:
        # DSSMSBacktester の _evaluate_switch_decision 詳細分析
        with open('src/dssms/dssms_backtester.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # _evaluate_switch_decision メソッドの完全抽出
        method_start = content.find('def _evaluate_switch_decision')
        if method_start != -1:
            # 次のメソッドまたはクラス終了を見つける
            remaining_content = content[method_start:]
            lines = remaining_content.split('\n')
            
            method_lines = []
            indent_level = None
            
            for i, line in enumerate(lines):
                if i == 0:  # def行
                    method_lines.append(line)
                    continue
                
                # インデントレベルを初回設定
                if indent_level is None and line.strip():
                    indent_level = len(line) - len(line.lstrip())
                
                # メソッド終了判定
                if line.strip() and (len(line) - len(line.lstrip())) <= indent_level - 4:
                    if line.strip().startswith('def ') or line.strip().startswith('class '):
                        break
                
                method_lines.append(line)
                if len(method_lines) > 100:  # 安全装置
                    break
            
            print(f"📋 _evaluate_switch_decision メソッド詳細:")
            print("=" * 50)
            for i, line in enumerate(method_lines[:30]):  # 最初の30行
                print(f"{i+1:2d}: {line}")
            
            # 重要な条件分岐を特定
            critical_conditions = []
            for line in method_lines:
                if any(keyword in line.lower() for keyword in ['if ', 'return false', 'return true', 'min_holding']):
                    critical_conditions.append(line.strip())
            
            print(f"\n🎯 重要な判定条件:")
            for condition in critical_conditions[:10]:
                print(f"  → {condition}")
                
        else:
            print("❌ _evaluate_switch_decision メソッドが見つかりません")
        
        return True
        
    except Exception as e:
        print(f"❌ 切替ロジック分析エラー: {e}")
        return False

def check_current_engine_usage():
    """現在使用されているエンジンの確認"""
    print(f"\n📋 現在使用エンジンの確認")
    print("-" * 40)
    
    try:
        # dssms_backtester.pyでのインポート文確認
        with open('src/dssms/dssms_backtester.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # インポート関連の行を抽出
        import_lines = []
        for line in content.split('\n'):
            if 'import' in line.lower() and any(keyword in line.lower() for keyword in ['excel', 'exporter', 'output', 'unified']):
                import_lines.append(line.strip())
        
        print("📊 検出されたインポート文:")
        for line in import_lines:
            print(f"  → {line}")
        
        # DSSMSExcelExporterV2の使用箇所確認
        v2_usage = re.findall(r'.*DSSMSExcelExporterV2.*', content)
        print(f"\n📊 DSSMSExcelExporterV2使用箇所: {len(v2_usage)}件")
        for usage in v2_usage[:3]:
            print(f"  → {usage.strip()}")
        
        # output実行箇所の確認
        output_calls = re.findall(r'.*\.export.*\(.*\)', content)
        print(f"\n📊 export呼び出し箇所: {len(output_calls)}件")
        for call in output_calls[:3]:
            print(f"  → {call.strip()}")
            
        return True
        
    except Exception as e:
        print(f"❌ エンジン使用確認エラー: {e}")
        return False

def main():
    """Task 1.3 メイン実行"""
    print("🚀 Task 1.3: 統一エンジン影響度分析 開始")
    print("=" * 80)
    
    # 1. 統一エンジン分析
    engine_results = analyze_unified_engine_impact()
    
    # 2. データフロー追跡
    flow_success = trace_data_flow()
    
    # 3. 切替ロジック変更点特定
    logic_success = identify_switching_logic_changes()
    
    # 4. 現在のエンジン使用状況確認
    engine_usage_success = check_current_engine_usage()
    
    # 結果サマリー
    print(f"\n📊 Task 1.3 実行結果サマリー")
    print("=" * 50)
    
    existing_engines = [name for name, result in engine_results.items() if result.get('exists', False)]
    print(f"✅ 分析完了エンジン数: {len(existing_engines)}")
    print(f"✅ データフロー追跡: {'成功' if flow_success else '失敗'}")
    print(f"✅ 切替ロジック分析: {'成功' if logic_success else '失敗'}")
    print(f"✅ エンジン使用確認: {'成功' if engine_usage_success else '失敗'}")
    
    # 重要な発見事項を特定
    print(f"\n🎯 重要な発見事項:")
    for engine_name, result in engine_results.items():
        if result.get('exists') and not result.get('error'):
            switch_count = len(result.get('switch_history_usage', []))
            if switch_count > 0:
                print(f"  📋 {engine_name}: switch_history処理 {switch_count}件")
    
    # 次のタスクへの推奨事項
    print(f"\n🎯 次のアクション（Task 1.2完了→Task 2へ）:")
    print("1. Task 1.2: データ永続化・キャッシュ問題調査の完了")
    print("2. Task 1.3の結果をroadmap2.mdに記録")
    print("3. Task 2.1: 日付処理ロジック検証の実行")
    
    # 結果を保存
    output_file = f"task_1_3_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'engine_analysis': engine_results,
            'data_flow_success': flow_success,
            'logic_analysis_success': logic_success,
            'engine_usage_success': engine_usage_success
        }, f, ensure_ascii=False, indent=2)
    
    print(f"💾 結果保存: {output_file}")
    
    return {
        'engine_analysis': engine_results,
        'data_flow_success': flow_success,
        'logic_analysis_success': logic_success,
        'engine_usage_success': engine_usage_success
    }

if __name__ == "__main__":
    main()

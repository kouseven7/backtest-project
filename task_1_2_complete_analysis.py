#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 1.2: データ永続化・キャッシュ問題調査（完了版）
隠れファイル、設定ファイル、キャッシュの詳細分析
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import os
import glob

def complete_cache_and_persistence_analysis():
    """データ永続化・キャッシュ問題の完全分析"""
    print("🔍 Task 1.2: データ永続化・キャッシュ問題調査（完了版）")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'hidden_files': {},
        'cache_directories': {},
        'dssms_config_files': {},
        'temp_files': {}
    }
    
    # 1. 隠れファイル・ディレクトリの詳細調査
    analyze_hidden_files(results)
    
    # 2. __pycache__ディレクトリの調査
    analyze_cache_directories(results)
    
    # 3. DSSMS設定ファイルの詳細調査
    analyze_dssms_config_files(results)
    
    # 4. 一時ファイル・ロックファイルの調査
    analyze_temp_files(results)
    
    # 5. メモリキャッシュ・状態保存の調査
    analyze_memory_state(results)
    
    return results

def analyze_hidden_files(results):
    """隠れファイル・ディレクトリの調査"""
    print("\n📁 1. 隠れファイル・ディレクトリ調査")
    print("-" * 40)
    
    try:
        # backtest_results/dssms_results の隠れファイル
        dssms_results_path = Path("backtest_results/dssms_results")
        if dssms_results_path.exists():
            print(f"📋 {dssms_results_path} 内の全ファイル:")
            all_files = list(dssms_results_path.glob("*"))
            hidden_files = []
            
            for file_path in all_files:
                file_stat = file_path.stat()
                is_hidden = file_path.name.startswith('.')
                if is_hidden or file_path.name.startswith('~'):
                    hidden_files.append({
                        'name': file_path.name,
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })
                
                print(f"  {'🔒' if is_hidden else '📄'} {file_path.name} ({file_stat.st_size} bytes)")
            
            results['hidden_files']['dssms_results'] = hidden_files
        else:
            print("❌ dssms_results ディレクトリが見つかりません")
            results['hidden_files']['dssms_results'] = []
        
        # プロジェクトルートの隠れファイル
        root_hidden = []
        for item in Path(".").glob(".*"):
            if item.is_file():
                root_hidden.append(item.name)
                print(f"  🔒 ルート隠れファイル: {item.name}")
        
        results['hidden_files']['project_root'] = root_hidden
        
    except Exception as e:
        print(f"❌ 隠れファイル調査エラー: {e}")
        results['hidden_files']['error'] = str(e)

def analyze_cache_directories(results):
    """__pycache__ディレクトリの調査"""
    print("\n🗂️ 2. __pycache__ディレクトリ調査")
    print("-" * 40)
    
    try:
        # __pycache__ディレクトリの検索
        cache_dirs = list(Path(".").rglob("__pycache__"))
        print(f"📊 検出された__pycache__ディレクトリ数: {len(cache_dirs)}")
        
        cache_info = []
        dssms_cache_count = 0
        
        for cache_dir in cache_dirs[:20]:  # 最初の20件
            cache_files = list(cache_dir.glob("*.pyc"))
            cache_size = sum(f.stat().st_size for f in cache_files)
            
            is_dssms_related = 'dssms' in str(cache_dir).lower()
            if is_dssms_related:
                dssms_cache_count += 1
            
            cache_info.append({
                'path': str(cache_dir),
                'file_count': len(cache_files),
                'total_size': cache_size,
                'is_dssms_related': is_dssms_related
            })
            
            print(f"  {'🎯' if is_dssms_related else '📂'} {cache_dir}: {len(cache_files)}ファイル ({cache_size}bytes)")
        
        results['cache_directories'] = {
            'total_count': len(cache_dirs),
            'dssms_related_count': dssms_cache_count,
            'details': cache_info
        }
        
        if len(cache_dirs) > 1000:
            print("⚠️ 異常に多数のキャッシュディレクトリが存在")
            print("  推奨: キャッシュクリア実行")
        
    except Exception as e:
        print(f"❌ キャッシュディレクトリ調査エラー: {e}")
        results['cache_directories']['error'] = str(e)

def analyze_dssms_config_files(results):
    """DSSMS設定ファイルの詳細調査"""
    print("\n📋 3. DSSMS設定ファイル詳細調査")
    print("-" * 40)
    
    try:
        # DSSMS関連設定ファイルの検索
        config_patterns = [
            "**/dssms*.json",
            "**/dssms*.config",
            "**/dssms*.cache",
            "**/*dssms*.py"
        ]
        
        config_files = {}
        total_config_files = 0
        
        for pattern in config_patterns:
            files = list(Path(".").glob(pattern))
            config_files[pattern] = []
            
            for file_path in files:
                if file_path.is_file():
                    file_stat = file_path.stat()
                    config_files[pattern].append({
                        'path': str(file_path),
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })
                    total_config_files += 1
                    print(f"  📄 {file_path} ({file_stat.st_size} bytes)")
        
        results['dssms_config_files'] = {
            'total_count': total_config_files,
            'by_pattern': config_files
        }
        
        # 設定ファイルの内容チェック（重要ファイルのみ）
        important_configs = [
            "config/dssms/intelligent_switch_config.json",
            "config/dssms/market_monitoring_config.json"
        ]
        
        print(f"\n📋 重要設定ファイルの内容チェック:")
        for config_path in important_configs:
            if Path(config_path).exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()[:200]  # 最初の200文字
                    print(f"  ✅ {config_path}: 正常 ({len(content)}文字)")
                except Exception as e:
                    print(f"  ❌ {config_path}: エラー - {e}")
            else:
                print(f"  ⚠️ {config_path}: ファイルなし")
        
    except Exception as e:
        print(f"❌ DSSMS設定ファイル調査エラー: {e}")
        results['dssms_config_files']['error'] = str(e)

def analyze_temp_files(results):
    """一時ファイル・ロックファイルの調査"""
    print("\n🔒 4. 一時ファイル・ロックファイル調査")
    print("-" * 40)
    
    try:
        temp_patterns = [
            "**/*.tmp",
            "**/*.lock",
            "**/*.~*",
            "**/~$*"
        ]
        
        temp_files = {}
        total_temp_files = 0
        
        for pattern in temp_patterns:
            files = list(Path(".").glob(pattern))
            temp_files[pattern] = []
            
            for file_path in files:
                if file_path.is_file():
                    file_stat = file_path.stat()
                    temp_files[pattern].append({
                        'path': str(file_path),
                        'size': file_stat.st_size,
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })
                    total_temp_files += 1
                    print(f"  🔒 {file_path} ({file_stat.st_size} bytes)")
        
        results['temp_files'] = {
            'total_count': total_temp_files,
            'by_pattern': temp_files
        }
        
        if total_temp_files == 0:
            print("  ✅ 一時ファイル・ロックファイルなし")
        
    except Exception as e:
        print(f"❌ 一時ファイル調査エラー: {e}")
        results['temp_files']['error'] = str(e)

def analyze_memory_state(results):
    """メモリキャッシュ・状態保存の調査"""
    print("\n🧠 5. メモリキャッシュ・状態保存調査")
    print("-" * 40)
    
    try:
        # DSSMSBacktester のソースコードから状態保存機構を調査
        backtester_path = Path("src/dssms/dssms_backtester.py")
        if backtester_path.exists():
            with open(backtester_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 状態保存関連のキーワード検索
            state_keywords = ['cache', 'save', 'load', 'pickle', 'json.dump', 'persist']
            state_findings = []
            
            for keyword in state_keywords:
                matches = [line.strip() for line in content.split('\n') if keyword.lower() in line.lower()]
                if matches:
                    state_findings.extend(matches[:3])  # 最初の3件
                    print(f"  📋 '{keyword}' 関連: {len(matches)}件")
            
            results['memory_state'] = {
                'state_keywords_found': len(state_findings),
                'sample_findings': state_findings[:10]
            }
            
            # switch_historyの初期化と永続化確認
            switch_init = [line.strip() for line in content.split('\n') if 'switch_history' in line and '=' in line]
            print(f"  📊 switch_history初期化: {len(switch_init)}件")
            for init_line in switch_init[:3]:
                print(f"    → {init_line}")
        
    except Exception as e:
        print(f"❌ メモリ状態調査エラー: {e}")
        results['memory_state']['error'] = str(e)

def main():
    """Task 1.2完了版 メイン実行"""
    print("🚀 Task 1.2: データ永続化・キャッシュ問題調査（完了版）")
    print("=" * 80)
    
    # 完全分析実行
    analysis_results = complete_cache_and_persistence_analysis()
    
    # 結果サマリー
    print(f"\n📊 Task 1.2 完全分析結果サマリー")
    print("=" * 50)
    
    hidden_count = len(analysis_results.get('hidden_files', {}).get('dssms_results', []))
    cache_count = analysis_results.get('cache_directories', {}).get('total_count', 0)
    config_count = analysis_results.get('dssms_config_files', {}).get('total_count', 0)
    temp_count = analysis_results.get('temp_files', {}).get('total_count', 0)
    
    print(f"✅ 隠れファイル: {hidden_count}件")
    print(f"✅ キャッシュディレクトリ: {cache_count}件")
    print(f"✅ DSSMS設定ファイル: {config_count}件")
    print(f"✅ 一時ファイル: {temp_count}件")
    
    # 重要な発見事項
    print(f"\n🎯 重要な発見事項:")
    if hidden_count > 0:
        print(f"  ⚠️ dssms_results内に隠れファイル/ロックファイルあり")
    if cache_count > 1000:
        print(f"  ⚠️ 過剰なキャッシュディレクトリ（推奨：クリア実行）")
    if temp_count > 0:
        print(f"  ⚠️ 一時ファイル/ロックファイルが残存")
    
    # 結果保存
    output_file = f"task_1_2_complete_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 完全分析結果保存: {output_file}")
    
    # 次のアクション
    print(f"\n🎯 Task 1.2 完了 → 次のアクション:")
    print("1. roadmap2.mdにTask 1.2, 1.3結果を記録")
    print("2. Task 2.1: 日付処理ロジック検証の実行")
    print("3. 必要に応じてキャッシュクリア推奨")
    
    return analysis_results

if __name__ == "__main__":
    main()

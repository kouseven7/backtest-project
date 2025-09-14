#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task 6.2: エンジンファイル関係性の完全整理
全エンジンファイルの関係性・責任範囲・使用状況の包括調査
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

class EngineFilesRelationshipAnalyzer:
    def __init__(self):
        self.project_root = Path('.')
        self.analysis_results = {}
        self.engine_files = []
        self.dssms_backtester_path = 'src/dssms/dssms_backtester.py'
        
    def analyze_complete_engine_relationships(self):
        """全エンジンファイル関係性の完全整理"""
        print("🔍 Task 6.2: エンジンファイル関係性の完全整理")
        print("=" * 80)
        
        # 1. 全エンジンファイルの発見・分類
        self._discover_all_engine_files()
        
        # 2. Task 4.2調査対象エンジンとの関係性特定
        self._analyze_task42_engine_relationship()
        
        # 3. 実際の使用状況・呼び出し関係の追跡
        self._trace_actual_engine_usage()
        
        # 4. エンジン間の継承・依存関係分析
        self._analyze_engine_dependencies()
        
        # 5. Excel出力責任エンジンの最終特定
        self._identify_excel_output_responsible_engine()
        
        # 6. 混乱状況の整理・問題特定
        self._identify_confusion_problems()
        
        return self.analysis_results
    
    def _discover_all_engine_files(self):
        """1. 全エンジンファイルの発見・分類"""
        print("\n🔍 1. 全エンジンファイルの発見・分類")
        print("-" * 60)
        
        # エンジン関連ファイルのパターン
        engine_patterns = [
            '*engine*.py',
            '*output*.py',
            '*unified*.py',
            '*dssms*.py'
        ]
        
        discovered_files = {
            'engine_files': [],
            'output_files': [],
            'unified_files': [],
            'dssms_files': []
        }
        
        try:
            # パターンベースでファイル発見
            for pattern in engine_patterns:
                for file_path in self.project_root.rglob(pattern):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        relative_path = str(file_path.relative_to(self.project_root))
                        
                        if 'engine' in file_path.name.lower():
                            discovered_files['engine_files'].append(relative_path)
                        if 'output' in file_path.name.lower():
                            discovered_files['output_files'].append(relative_path)
                        if 'unified' in file_path.name.lower():
                            discovered_files['unified_files'].append(relative_path)
                        if 'dssms' in file_path.name.lower():
                            discovered_files['dssms_files'].append(relative_path)
            
            # 特に重要なエンジンファイルの詳細分析
            important_engines = [
                'dssms_unified_output_engine.py',
                'dssms_unified_output_engine_fixed.py',
                'dssms_unified_output_engine_fixed_v3.py',
                'dssms_unified_output_engine_fixed_v4.py',
                'src/dssms/unified_output_engine.py'
            ]
            
            engine_analysis = {}
            for engine_file in important_engines:
                if os.path.exists(engine_file):
                    file_path = Path(engine_file)
                    stat_info = file_path.stat()
                    
                    engine_analysis[engine_file] = {
                        'exists': True,
                        'size': stat_info.st_size,
                        'modified': datetime.fromtimestamp(stat_info.st_mtime),
                        'location': 'root' if not '/' in engine_file else 'src_directory'
                    }
                else:
                    engine_analysis[engine_file] = {'exists': False}
            
            self.analysis_results['discovered_files'] = discovered_files
            self.analysis_results['important_engines'] = engine_analysis
            
            print(f"📊 発見されたファイル:")
            print(f"   エンジンファイル: {len(discovered_files['engine_files'])}個")
            print(f"   出力ファイル: {len(discovered_files['output_files'])}個")
            print(f"   統一ファイル: {len(discovered_files['unified_files'])}個")
            
            print(f"\n📋 重要エンジンファイルの状況:")
            for engine, info in engine_analysis.items():
                status = "✅ 存在" if info['exists'] else "❌ 不存在"
                if info['exists']:
                    print(f"   {status} {engine}: {info['size']:,} bytes, {info['location']}")
                else:
                    print(f"   {status} {engine}")
            
        except Exception as e:
            print(f"❌ ファイル発見エラー: {e}")
            self.analysis_results['discovered_files'] = {'error': str(e)}
    
    def _analyze_task42_engine_relationship(self):
        """2. Task 4.2調査対象エンジンとの関係性特定"""
        print("\n🔍 2. Task 4.2調査対象エンジンとの関係性特定")
        print("-" * 60)
        
        try:
            task42_results_file = 'task_4_2_results_20250912_115837.json'
            
            if os.path.exists(task42_results_file):
                with open(task42_results_file, 'r', encoding='utf-8') as f:
                    task42_data = json.load(f)
                
                # Task 4.2で調査されたエンジンファイル
                task42_engines = list(task42_data.get('detailed_implementations', {}).keys())
                
                # 現在存在するファイルとの照合
                relationship_analysis = {}
                for engine_name in task42_engines:
                    relationship_analysis[engine_name] = {
                        'in_task42': True,
                        'exists_now': os.path.exists(engine_name),
                        'current_location': None,
                        'relationship_type': 'unknown'
                    }
                    
                    if relationship_analysis[engine_name]['exists_now']:
                        # ファイルの現在の状況確認
                        file_path = Path(engine_name)
                        stat_info = file_path.stat()
                        
                        relationship_analysis[engine_name].update({
                            'current_size': stat_info.st_size,
                            'last_modified': datetime.fromtimestamp(stat_info.st_mtime),
                            'current_location': 'root_directory'
                        })
                        
                        # Task 4.2時点と現在の関係性判定
                        task42_size = task42_data['detailed_implementations'][engine_name].get('file_size', 0)
                        current_size = stat_info.st_size
                        
                        if task42_size == current_size:
                            relationship_analysis[engine_name]['relationship_type'] = 'identical'
                        elif abs(task42_size - current_size) < 1000:  # 1KB以内
                            relationship_analysis[engine_name]['relationship_type'] = 'minor_changes'
                        else:
                            relationship_analysis[engine_name]['relationship_type'] = 'significant_changes'
                
                # 現在使用中エンジンの特定（dssms_backtesterから）
                current_engine = self._identify_currently_used_engine()
                
                self.analysis_results['task42_relationship'] = {
                    'task42_engines': task42_engines,
                    'relationship_analysis': relationship_analysis,
                    'current_engine': current_engine,
                    'task42_highest_score': {
                        'engine': 'dssms_unified_output_engine.py',
                        'score': 85.0,
                        'same_as_current': current_engine == 'dssms_unified_output_engine.py'
                    }
                }
                
                print(f"📊 Task 4.2調査対象エンジン: {len(task42_engines)}個")
                for engine, analysis in relationship_analysis.items():
                    exists = "✅" if analysis['exists_now'] else "❌"
                    relation = analysis['relationship_type']
                    print(f"   {exists} {engine}: {relation}")
                
                print(f"🎯 最高スコアエンジン: dssms_unified_output_engine.py (85.0点)")
                print(f"🔄 現在使用中: {current_engine}")
                print(f"✅ 同一性: {'同じ' if current_engine == 'dssms_unified_output_engine.py' else '異なる'}")
                
            else:
                print("❌ Task 4.2結果ファイルが見つかりません")
                self.analysis_results['task42_relationship'] = {'error': 'task42_results_not_found'}
                
        except Exception as e:
            print(f"❌ Task 4.2関係性分析エラー: {e}")
            self.analysis_results['task42_relationship'] = {'error': str(e)}
    
    def _identify_currently_used_engine(self):
        """現在使用中のエンジンの特定"""
        try:
            if os.path.exists(self.dssms_backtester_path):
                with open(self.dssms_backtester_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # インポート文からエンジンを特定
                import_patterns = [
                    r'from\s+(\S*unified_output_engine\S*)\s+import',
                    r'import\s+(\S*unified_output_engine\S*)',
                    r'(\w*unified_output_engine\w*)',
                ]
                
                for pattern in import_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        return matches[0]
                
                # 直接的なファイル参照を検索
                if 'dssms_unified_output_engine' in content:
                    return 'dssms_unified_output_engine.py'
                
                return 'unknown'
            else:
                return 'backtester_not_found'
        except Exception as e:
            return f'error: {e}'
    
    def _trace_actual_engine_usage(self):
        """3. 実際の使用状況・呼び出し関係の追跡"""
        print("\n🔍 3. 実際の使用状況・呼び出し関係の追跡")
        print("-" * 60)
        
        usage_analysis = {
            'backtester_imports': [],
            'engine_calls': [],
            'actual_execution_flow': {}
        }
        
        try:
            # DSSMSバックテスターの詳細分析
            if os.path.exists(self.dssms_backtester_path):
                with open(self.dssms_backtester_path, 'r', encoding='utf-8') as f:
                    backtester_content = f.read()
                
                # インポート文の解析
                import_lines = [line.strip() for line in backtester_content.split('\n') if 'import' in line and 'engine' in line.lower()]
                usage_analysis['backtester_imports'] = import_lines
                
                # エンジン呼び出しの検索
                engine_call_patterns = [
                    r'(\w*Engine\w*)\(',
                    r'(\w*engine\w*)\.',
                    r'(\w*output\w*)\.',
                ]
                
                for pattern in engine_call_patterns:
                    matches = re.findall(pattern, backtester_content, re.IGNORECASE)
                    usage_analysis['engine_calls'].extend(matches)
                
                # 実際の実行フローの推定
                if 'unified_output_engine' in backtester_content:
                    usage_analysis['actual_execution_flow']['primary_engine'] = 'unified_output_engine'
                elif 'dssms_unified_output_engine' in backtester_content:
                    usage_analysis['actual_execution_flow']['primary_engine'] = 'dssms_unified_output_engine'
                else:
                    usage_analysis['actual_execution_flow']['primary_engine'] = 'unknown'
                
                print(f"📋 バックテスターからのインポート:")
                for import_line in import_lines:
                    print(f"   {import_line}")
                
                print(f"🔄 エンジン呼び出し検出: {len(set(usage_analysis['engine_calls']))}種類")
                for call in set(usage_analysis['engine_calls']):
                    print(f"   {call}")
                
                print(f"🎯 主要エンジン: {usage_analysis['actual_execution_flow']['primary_engine']}")
                
            else:
                print("❌ dssms_backtester.py が見つかりません")
                usage_analysis['error'] = 'backtester_not_found'
            
            self.analysis_results['usage_analysis'] = usage_analysis
            
        except Exception as e:
            print(f"❌ 使用状況追跡エラー: {e}")
            self.analysis_results['usage_analysis'] = {'error': str(e)}
    
    def _analyze_engine_dependencies(self):
        """4. エンジン間の継承・依存関係分析"""
        print("\n🔍 4. エンジン間の継承・依存関係分析")
        print("-" * 60)
        
        dependencies = {
            'inheritance_chain': {},
            'import_dependencies': {},
            'version_evolution': {}
        }
        
        try:
            # 重要エンジンファイルの依存関係分析
            important_engines = [
                'dssms_unified_output_engine.py',
                'dssms_unified_output_engine_fixed.py',
                'dssms_unified_output_engine_fixed_v3.py',
                'dssms_unified_output_engine_fixed_v4.py'
            ]
            
            for engine_file in important_engines:
                if os.path.exists(engine_file):
                    with open(engine_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # インポート依存関係の分析
                    import_lines = [line.strip() for line in content.split('\n') if line.strip().startswith('import') or line.strip().startswith('from')]
                    dependencies['import_dependencies'][engine_file] = import_lines
                    
                    # クラス継承の分析
                    class_def_pattern = r'class\s+(\w+)\s*\([^)]*\):'
                    class_matches = re.findall(class_def_pattern, content)
                    dependencies['inheritance_chain'][engine_file] = class_matches
                    
                    # バージョン進化の推定
                    if 'v4' in engine_file:
                        dependencies['version_evolution'][engine_file] = 'latest_version'
                    elif 'v3' in engine_file:
                        dependencies['version_evolution'][engine_file] = 'intermediate_version'
                    elif 'fixed' in engine_file:
                        dependencies['version_evolution'][engine_file] = 'bug_fix_version'
                    else:
                        dependencies['version_evolution'][engine_file] = 'original_version'
            
            # 依存関係の可視化
            print(f"📊 エンジン依存関係:")
            for engine, imports in dependencies['import_dependencies'].items():
                print(f"   {engine}:")
                relevant_imports = [imp for imp in imports if 'engine' in imp.lower() or 'output' in imp.lower()]
                for imp in relevant_imports[:5]:  # 最初の5個のみ表示
                    print(f"     - {imp}")
            
            print(f"\n🔄 バージョン進化:")
            for engine, version_type in dependencies['version_evolution'].items():
                print(f"   {engine}: {version_type}")
            
            self.analysis_results['dependencies'] = dependencies
            
        except Exception as e:
            print(f"❌ 依存関係分析エラー: {e}")
            self.analysis_results['dependencies'] = {'error': str(e)}
    
    def _identify_excel_output_responsible_engine(self):
        """5. Excel出力責任エンジンの最終特定"""
        print("\n🔍 5. Excel出力責任エンジンの最終特定")
        print("-" * 60)
        
        excel_responsibility = {
            'responsible_engine': 'unknown',
            'excel_generation_methods': {},
            'output_flow': {}
        }
        
        try:
            # 各エンジンでのExcel出力機能の分析
            engine_files = [
                'dssms_unified_output_engine.py',
                'dssms_unified_output_engine_fixed.py',
                'dssms_unified_output_engine_fixed_v3.py',
                'dssms_unified_output_engine_fixed_v4.py'
            ]
            
            for engine_file in engine_files:
                if os.path.exists(engine_file):
                    with open(engine_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Excel出力関連メソッドの検索
                    excel_methods = []
                    excel_patterns = [
                        r'def\s+(\w*excel\w*)\(',
                        r'def\s+(\w*export\w*)\(',
                        r'def\s+(\w*save\w*)\(',
                        r'(\w*\.xlsx\w*)',
                        r'(openpyxl|xlsxwriter)',
                    ]
                    
                    for pattern in excel_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        excel_methods.extend(matches)
                    
                    excel_responsibility['excel_generation_methods'][engine_file] = {
                        'excel_methods': list(set(excel_methods)),
                        'has_excel_capability': len(excel_methods) > 0,
                        'openpyxl_usage': 'openpyxl' in content,
                        'xlsx_references': content.count('.xlsx')
                    }
            
            # 最も責任を持つエンジンの特定
            max_excel_capability = 0
            responsible_engine = 'unknown'
            
            for engine, capabilities in excel_responsibility['excel_generation_methods'].items():
                capability_score = (
                    len(capabilities['excel_methods']) * 2 +
                    capabilities['xlsx_references'] +
                    (10 if capabilities['openpyxl_usage'] else 0)
                )
                
                if capability_score > max_excel_capability:
                    max_excel_capability = capability_score
                    responsible_engine = engine
            
            excel_responsibility['responsible_engine'] = responsible_engine
            excel_responsibility['confidence_score'] = max_excel_capability
            
            print(f"📊 Excel出力機能分析:")
            for engine, capabilities in excel_responsibility['excel_generation_methods'].items():
                has_excel = "✅" if capabilities['has_excel_capability'] else "❌"
                print(f"   {has_excel} {engine}: {len(capabilities['excel_methods'])}メソッド, {capabilities['xlsx_references']}個xlsx参照")
            
            print(f"\n🎯 Excel出力責任エンジン: {responsible_engine}")
            print(f"📊 信頼度スコア: {max_excel_capability}")
            
            self.analysis_results['excel_responsibility'] = excel_responsibility
            
        except Exception as e:
            print(f"❌ Excel責任分析エラー: {e}")
            self.analysis_results['excel_responsibility'] = {'error': str(e)}
    
    def _identify_confusion_problems(self):
        """6. 混乱状況の整理・問題特定"""
        print("\n🔍 6. 混乱状況の整理・問題特定")
        print("-" * 60)
        
        confusion_problems = {
            'identified_problems': [],
            'confusion_sources': {},
            'resolution_priorities': {}
        }
        
        try:
            # 問題1: 複数の類似エンジンファイルの存在
            engine_files = self.analysis_results.get('important_engines', {})
            existing_engines = [name for name, info in engine_files.items() if info.get('exists', False)]
            
            if len(existing_engines) > 1:
                confusion_problems['identified_problems'].append({
                    'problem': 'multiple_similar_engines',
                    'description': f'{len(existing_engines)}個の類似エンジンファイルが並存',
                    'files': existing_engines,
                    'severity': 'high'
                })
            
            # 問題2: Task 4.2の85.0点エンジンと現在使用中エンジンの不一致
            task42_relation = self.analysis_results.get('task42_relationship', {})
            if not task42_relation.get('task42_highest_score', {}).get('same_as_current', False):
                confusion_problems['identified_problems'].append({
                    'problem': 'task42_current_engine_mismatch',
                    'description': 'Task 4.2の最高スコアエンジンと現在使用中エンジンが異なる',
                    'task42_best': 'dssms_unified_output_engine.py',
                    'current': task42_relation.get('current_engine', 'unknown'),
                    'severity': 'critical'
                })
            
            # 問題3: Excel出力責任の曖昧性
            excel_resp = self.analysis_results.get('excel_responsibility', {})
            if excel_resp.get('confidence_score', 0) < 10:
                confusion_problems['identified_problems'].append({
                    'problem': 'unclear_excel_responsibility',
                    'description': 'Excel出力責任エンジンが不明確',
                    'responsible_engine': excel_resp.get('responsible_engine', 'unknown'),
                    'confidence': excel_resp.get('confidence_score', 0),
                    'severity': 'medium'
                })
            
            # 問題4: エンジンファイルの場所の混乱
            root_engines = [name for name, info in engine_files.items() 
                          if info.get('exists', False) and info.get('location') == 'root']
            src_engines = [name for name, info in engine_files.items() 
                         if info.get('exists', False) and info.get('location') == 'src_directory']
            
            if len(root_engines) > 0 and len(src_engines) > 0:
                confusion_problems['identified_problems'].append({
                    'problem': 'mixed_engine_locations',
                    'description': 'エンジンファイルがルートディレクトリとsrcディレクトリに分散',
                    'root_engines': root_engines,
                    'src_engines': src_engines,
                    'severity': 'medium'
                })
            
            # 混乱の根本原因分析
            confusion_problems['confusion_sources'] = {
                'version_proliferation': len(existing_engines) > 2,
                'lack_of_cleanup': 'fixed' in str(existing_engines) and 'v4' in str(existing_engines),
                'unclear_primary_engine': task42_relation.get('current_engine') == 'unknown',
                'inconsistent_task_results': True  # Task 6.1とTask 4.2の矛盾から
            }
            
            # 解決優先度の設定
            for problem in confusion_problems['identified_problems']:
                if problem['severity'] == 'critical':
                    confusion_problems['resolution_priorities'][problem['problem']] = 1
                elif problem['severity'] == 'high':
                    confusion_problems['resolution_priorities'][problem['problem']] = 2
                else:
                    confusion_problems['resolution_priorities'][problem['problem']] = 3
            
            print(f"🚨 特定された問題: {len(confusion_problems['identified_problems'])}件")
            for i, problem in enumerate(confusion_problems['identified_problems'], 1):
                severity_icon = "🔴" if problem['severity'] == 'critical' else "🟡" if problem['severity'] == 'high' else "🟢"
                print(f"   {i}. {severity_icon} {problem['description']}")
            
            print(f"\n🔍 混乱の根本原因:")
            for source, exists in confusion_problems['confusion_sources'].items():
                status = "✅" if exists else "❌"
                print(f"   {status} {source}")
            
            self.analysis_results['confusion_problems'] = confusion_problems
            
        except Exception as e:
            print(f"❌ 混乱問題特定エラー: {e}")
            self.analysis_results['confusion_problems'] = {'error': str(e)}
    
    def generate_engine_relationship_summary(self):
        """エンジン関係性サマリーの生成"""
        summary = {
            'critical_findings': [],
            'engine_status': {},
            'recommendations': []
        }
        
        try:
            # 重要な発見事項
            task42_relation = self.analysis_results.get('task42_relationship', {})
            if task42_relation.get('task42_highest_score', {}).get('same_as_current', False):
                summary['critical_findings'].append(
                    "✅ Task 4.2の最高スコアエンジン(85.0点)と現在使用中エンジンが同一"
                )
            else:
                summary['critical_findings'].append(
                    "🚨 Task 4.2の最高スコアエンジン(85.0点)と現在使用中エンジンが異なる"
                )
            
            # エンジンステータス
            important_engines = self.analysis_results.get('important_engines', {})
            for engine, info in important_engines.items():
                if info.get('exists', False):
                    summary['engine_status'][engine] = {
                        'status': 'active',
                        'size': info.get('size', 0),
                        'location': info.get('location', 'unknown')
                    }
                else:
                    summary['engine_status'][engine] = {'status': 'missing'}
            
            # 推奨事項
            confusion_problems = self.analysis_results.get('confusion_problems', {})
            critical_problems = [p for p in confusion_problems.get('identified_problems', []) 
                               if p['severity'] == 'critical']
            
            if critical_problems:
                summary['recommendations'].append(
                    "1. Critical問題の即座解決（エンジン不一致問題）"
                )
            
            existing_engines = [name for name, status in summary['engine_status'].items() 
                              if status['status'] == 'active']
            if len(existing_engines) > 2:
                summary['recommendations'].append(
                    f"2. 不要エンジンファイルの整理（{len(existing_engines)}個→1-2個）"
                )
            
            summary['recommendations'].append(
                "3. Excel出力責任エンジンの明確化"
            )
            summary['recommendations'].append(
                "4. エンジンファイル配置の標準化"
            )
            
        except Exception as e:
            summary['error'] = str(e)
        
        return summary
    
    def save_results(self):
        """結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"task_6_2_engine_relationship_analysis_{timestamp}.json"
        
        # サマリーも含めて保存
        complete_results = {
            'analysis_results': self.analysis_results,
            'summary': self.generate_engine_relationship_summary()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n✅ エンジン関係性分析結果保存: {output_file}")
        return output_file

def main():
    """メイン実行"""
    print("🔍 Task 6.2: エンジンファイル関係性の完全整理")
    print("=" * 80)
    
    analyzer = EngineFilesRelationshipAnalyzer()
    
    # エンジン関係性分析実行
    analyzer.analyze_complete_engine_relationships()
    
    # サマリー生成
    summary = analyzer.generate_engine_relationship_summary()
    
    # 結果保存
    output_file = analyzer.save_results()
    
    print("\n" + "=" * 80)
    print("📋 Task 6.2完了サマリー")
    print("=" * 80)
    
    print("🔍 重要な発見:")
    for finding in summary.get('critical_findings', []):
        print(f"   {finding}")
    
    print(f"\n📊 エンジンステータス:")
    for engine, status in summary.get('engine_status', {}).items():
        status_icon = "✅" if status['status'] == 'active' else "❌"
        print(f"   {status_icon} {engine}: {status['status']}")
    
    print(f"\n💡 推奨事項:")
    for recommendation in summary.get('recommendations', []):
        print(f"   {recommendation}")
    
    return output_file, summary

if __name__ == "__main__":
    main()
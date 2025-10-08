#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
コンテキスト内エンジンファイルのExcel出力影響調査
各エンジンファイルがDSSMSのExcel出力に与える実際の影響度を分析
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

class ContextEngineExcelImpactAnalyzer:
    def __init__(self):
        self.context_engines = [
            'dssms_unified_output_engine.py',
            'dssms_unified_output_engine_fixed.py', 
            'dssms_unified_output_engine_fixed_v4.py',
            'dssms_unified_output_engine_fixed_v3.py',
            'dssms_excel_fix_phase3.py',
            'dssms_excel_fix_phase2.py',
            'dssms_enhanced_excel_exporter.py'
        ]
        self.analysis_results = {}
        
    def analyze_engine_excel_impact(self):
        """各エンジンのExcel出力への実際の影響度を分析"""
        print("[SEARCH] コンテキスト内エンジンのExcel出力影響調査開始")
        print("=" * 60)
        
        for engine_file in self.context_engines:
            if os.path.exists(engine_file):
                print(f"\n📄 分析中: {engine_file}")
                impact_data = self._analyze_single_engine(engine_file)
                self.analysis_results[engine_file] = impact_data
            else:
                print(f"\n[ERROR] ファイル未発見: {engine_file}")
                self.analysis_results[engine_file] = {"status": "not_found"}
                
        return self.analysis_results
    
    def _analyze_single_engine(self, engine_file):
        """単一エンジンファイルの詳細分析"""
        try:
            with open(engine_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            analysis = {
                "file_info": {
                    "size_bytes": len(content),
                    "line_count": len(content.split('\n')),
                    "exists": True
                },
                "excel_impact": self._analyze_excel_impact(content),
                "current_usage": self._check_current_usage(engine_file),
                "implementation_status": self._check_implementation_status(content),
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: "excel_output_methods": self._find_excel_methods(content),
                "potential_conflicts": self._check_conflicts(content)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": str(e), "status": "analysis_failed"}
    
    def _analyze_excel_impact(self, content):
        """Excel出力への影響度分析"""
        excel_keywords = [
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: 'excel', 'xlsx', 'openpyxl', 'ExcelWriter',
            'to_excel', 'create_sheet', 'workbook'
        ]
        
        impact_score = 0
        found_keywords = []
        
        for keyword in excel_keywords:
            count = content.lower().count(keyword.lower())
            if count > 0:
                impact_score += count
                found_keywords.append(f"{keyword}: {count}")
                
        return {
            "impact_score": impact_score,
            "found_keywords": found_keywords,
            "has_excel_functionality": impact_score > 0
        }
    
    def _check_current_usage(self, engine_file):
        """現在の使用状況確認"""
        # main.pyやdssms_backtester.pyでの使用確認
        usage_files = [
            'main.py',
            'src/dssms/dssms_backtester.py',
            'src/dssms/dssms_backtester_v2.py'
        ]
        
        usage_status = {}
        
        for usage_file in usage_files:
            if os.path.exists(usage_file):
                try:
                    with open(usage_file, 'r', encoding='utf-8') as f:
                        usage_content = f.read()
                    
                    # ファイル名からモジュール名を抽出
                    module_name = Path(engine_file).stem
                    is_imported = module_name in usage_content
                    
                    usage_status[usage_file] = {
                        "is_imported": is_imported,
                        "import_count": usage_content.count(module_name)
                    }
                except:
                    usage_status[usage_file] = {"error": "読み込み失敗"}
            else:
                usage_status[usage_file] = {"status": "file_not_found"}
                
        return usage_status
    
    def _check_implementation_status(self, content):
        """実装状況確認"""
        if len(content.strip()) == 0:
            return {
                "status": "empty_file",
                "severity": "critical",
                "description": "完全空ファイル - 実装なし"
            }
        
        # 重要メソッドの実装確認
        critical_methods = [
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: 'generate_excel', 'create_excel', '_generate_excel_output',
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: 'to_excel', 'export_excel', 'excel_output'
        ]
        
        implemented_methods = []
        for method in critical_methods:
            if method in content:
                implemented_methods.append(method)
        
        if len(implemented_methods) == 0:
            return {
                "status": "no_excel_methods",
                "severity": "high", 
                "description": "Excel出力メソッドが実装されていない"
            }
        elif len(implemented_methods) < 2:
            return {
                "status": "minimal_implementation",
                "severity": "medium",
                "implemented_methods": implemented_methods
            }
        else:
            return {
                "status": "well_implemented", 
                "severity": "low",
                "implemented_methods": implemented_methods
            }
    
    def _find_excel_methods(self, content):
        """Excel関連メソッドの特定"""
        lines = content.split('\n')
        excel_methods = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if (line_stripped.startswith('def ') and 
                ('excel' in line_stripped.lower() or 
                 'workbook' in line_stripped.lower() or
                 'sheet' in line_stripped.lower())):
                excel_methods.append({
                    "line_number": i + 1,
                    "method_name": line_stripped,
                    "context": lines[max(0, i-1):min(len(lines), i+3)]
                })
        
        return excel_methods
    
    def _check_conflicts(self, content):
        """他エンジンとの競合確認"""
        conflict_indicators = [
            'DSSMSUnifiedOutputEngine',
            'DSSMSExcelExporter', 
            'EnhancedDSSMSExcelExporter',
            'class.*Engine',
            'class.*Exporter'
        ]
        
        conflicts = []
        for indicator in conflict_indicators:
            if indicator in content:
                conflicts.append(indicator)
        
        return conflicts
    
    def generate_impact_report(self):
        """影響度レポート生成"""
        if not self.analysis_results:
            self.analyze_engine_excel_impact()
        
        print("\n" + "="*80)
        print("[CHART] Excel出力影響度分析レポート")
        print("="*80)
        
        # 影響度による分類
        high_impact = []
        medium_impact = []
        low_impact = []
        no_impact = []
        
        for engine_file, data in self.analysis_results.items():
            if isinstance(data, dict) and 'excel_impact' in data:
                impact_score = data['excel_impact']['impact_score']
                has_excel = data['excel_impact']['has_excel_functionality']
                
                if impact_score > 20 and has_excel:
                    high_impact.append((engine_file, impact_score))
                elif impact_score > 5 and has_excel:
                    medium_impact.append((engine_file, impact_score))
                elif has_excel:
                    low_impact.append((engine_file, impact_score))
                else:
                    no_impact.append(engine_file)
        
        print(f"\n🔴 高影響ファイル ({len(high_impact)}件):")
        for file, score in high_impact:
            usage = self._get_usage_summary(file)
            print(f"  - {file} (影響度: {score}) {usage}")
            
        print(f"\n🟡 中影響ファイル ({len(medium_impact)}件):")
        for file, score in medium_impact:
            usage = self._get_usage_summary(file)
            print(f"  - {file} (影響度: {score}) {usage}")
            
        print(f"\n🟢 低影響ファイル ({len(low_impact)}件):")
        for file, score in low_impact:
            usage = self._get_usage_summary(file)
            print(f"  - {file} (影響度: {score}) {usage}")
            
        print(f"\n⚪ 無影響ファイル ({len(no_impact)}件):")
        for file in no_impact:
            print(f"  - {file}")
        
        # 競合問題の特定
        self._analyze_conflicts()
        
        # 推奨アクション
        self._generate_recommendations()
    
    def _get_usage_summary(self, engine_file):
        """使用状況サマリー"""
        if engine_file not in self.analysis_results:
            return "[未分析]"
            
        data = self.analysis_results[engine_file]
        if 'current_usage' not in data:
            return "[使用状況不明]"
            
        usage = data['current_usage']
        active_usage = []
        
        for file, info in usage.items():
            if isinstance(info, dict) and info.get('is_imported', False):
                active_usage.append(file)
        
        if active_usage:
            return f"[使用中: {', '.join(active_usage)}]"
        else:
            return "[未使用]"
    
    def _analyze_conflicts(self):
        """競合分析"""
        print(f"\n[WARNING]  競合問題分析:")
        
        conflict_count = 0
        for engine_file, data in self.analysis_results.items():
            if isinstance(data, dict) and 'potential_conflicts' in data:
                conflicts = data['potential_conflicts']
                if conflicts:
                    conflict_count += 1
                    print(f"  🔸 {engine_file}: {', '.join(conflicts)}")
        
        if conflict_count == 0:
            print("  [OK] 重大な競合は検出されませんでした")
        else:
            print(f"  [ERROR] {conflict_count}個のファイルで潜在的競合を検出")
    
    def _generate_recommendations(self):
        """推奨アクション生成"""
        print(f"\n[IDEA] 推奨アクション:")
        
        # 使用中で高影響のファイルを特定
        active_high_impact = []
        unused_files = []
        
        for engine_file, data in self.analysis_results.items():
            if isinstance(data, dict):
                usage = data.get('current_usage', {})
                impact = data.get('excel_impact', {}).get('impact_score', 0)
                
                is_used = any(info.get('is_imported', False) 
                             for info in usage.values() 
                             if isinstance(info, dict))
                
                if is_used and impact > 10:
                    active_high_impact.append(engine_file)
                elif not is_used:
                    unused_files.append(engine_file)
        
        if active_high_impact:
            print("  1️⃣ 優先確認対象（使用中 & 高影響）:")
            for file in active_high_impact:
                print(f"     - {file}")
        
        if unused_files:
            print("  2️⃣ 削除候補（未使用ファイル）:")
            for file in unused_files:
                print(f"     - {file}")
        
        print("  3️⃣ 統一化推奨: 複数のExcelエンジンが存在する場合は統一を検討")
    
    def save_results(self, output_file=None):
        """結果保存"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"context_engine_excel_impact_analysis_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n[OK] 分析結果保存: {output_file}")
        return output_file

def main():
    """メイン実行"""
    print("[ROCKET] コンテキスト内エンジンファイルのExcel出力影響調査")
    print("=" * 60)
    
    analyzer = ContextEngineExcelImpactAnalyzer()
    
    # 分析実行
    analyzer.analyze_engine_excel_impact()
    
    # レポート生成
    analyzer.generate_impact_report()
    
    # 結果保存
    output_file = analyzer.save_results()
    
    print(f"\n[OK] 調査完了: {output_file}")
    
    return output_file

if __name__ == "__main__":
    main()
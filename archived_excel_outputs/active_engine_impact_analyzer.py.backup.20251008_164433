#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重要発見: 現在使用中エンジンのExcel出力問題調査
dssms_unified_output_engine.py が src/dssms/dssms_backtester.py で使用されていることが判明
この影響をDSSMSのExcel出力に与える深刻度を調査
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

class ActiveEngineImpactAnalyzer:
    def __init__(self):
        self.active_engine = 'dssms_unified_output_engine.py'
        self.backtester_file = 'src/dssms/dssms_backtester.py'
        self.results = {}
        
    def analyze_active_engine_impact(self):
        """現在使用中エンジンの詳細影響分析"""
        print("🚨 重要発見: 現在使用中エンジンのExcel出力問題調査")
        print("=" * 80)
        print(f"使用中エンジン: {self.active_engine}")
        print(f"使用元ファイル: {self.backtester_file}")
        print("=" * 80)
        
        # 1. 使用箇所の詳細分析
        self._analyze_usage_context()
        
        # 2. Task 4.2結果との照合
        self._compare_with_task42_results()
        
        # 3. 実際のExcel出力への影響評価
        self._evaluate_excel_output_impact()
        
        # 4. Problem 15として新問題定義
        self._define_as_new_problem()
        
        return self.results
    
    def _analyze_usage_context(self):
        """使用箇所の詳細分析"""
        print("\n🔍 1. 使用箇所の詳細分析")
        print("-" * 40)
        
        try:
            with open(self.backtester_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # インポート箇所の特定
            lines = content.split('\n')
            import_lines = []
            usage_lines = []
            
            for i, line in enumerate(lines):
                if 'dssms_unified_output_engine' in line.lower():
                    import_lines.append({
                        'line_number': i + 1,
                        'content': line.strip(),
                        'context': lines[max(0, i-2):min(len(lines), i+3)]
                    })
                
                if 'DSSMSUnifiedOutputEngine' in line:
                    usage_lines.append({
                        'line_number': i + 1,
                        'content': line.strip(),
                        'context': lines[max(0, i-2):min(len(lines), i+3)]
                    })
            
            self.results['usage_analysis'] = {
                'import_locations': import_lines,
                'usage_locations': usage_lines,
                'total_references': len(import_lines) + len(usage_lines)
            }
            
            print(f"📍 インポート箇所: {len(import_lines)}件")
            for imp in import_lines:
                print(f"   Line {imp['line_number']}: {imp['content']}")
            
            print(f"📍 使用箇所: {len(usage_lines)}件")
            for usage in usage_lines:
                print(f"   Line {usage['line_number']}: {usage['content']}")
                
        except Exception as e:
            print(f"❌ 分析エラー: {e}")
            self.results['usage_analysis'] = {'error': str(e)}
    
    def _compare_with_task42_results(self):
        """Task 4.2結果との照合"""
        print("\n🔍 2. Task 4.2結果との照合")
        print("-" * 40)
        
        # Task 4.2の結果ファイルから品質スコアを確認
        task42_file = 'task_4_2_results_20250912_115837.json'
        
        if os.path.exists(task42_file):
            try:
                with open(task42_file, 'r', encoding='utf-8') as f:
                    task42_data = json.load(f)
                
                # v1エンジンの品質スコア確認
                v1_score = task42_data.get('engine_quality_scores', {}).get('v1', 0)
                print(f"📊 現在使用中エンジン(v1)の品質スコア: {v1_score}点")
                
                if v1_score == 85.0:
                    print("✅ 品質スコアは最高評価（85.0点）")
                    quality_assessment = "excellent"
                elif v1_score >= 50.0:
                    print("🟡 品質スコアは中程度")
                    quality_assessment = "medium"
                else:
                    print("🔴 品質スコアは低評価")
                    quality_assessment = "poor"
                
                self.results['quality_assessment'] = {
                    'score': v1_score,
                    'assessment': quality_assessment,
                    'source': 'task_4_2_results'
                }
                
            except Exception as e:
                print(f"❌ Task 4.2結果読み込みエラー: {e}")
                self.results['quality_assessment'] = {'error': str(e)}
        else:
            print("⚠️ Task 4.2結果ファイルが見つかりません")
            self.results['quality_assessment'] = {'status': 'not_found'}
    
    def _evaluate_excel_output_impact(self):
        """実際のExcel出力への影響評価"""
        print("\n🔍 3. 実際のExcel出力への影響評価")
        print("-" * 40)
        
        # 最新のExcel出力ファイルを確認
        excel_pattern = "backtest_results/dssms_results/*.xlsx"
        excel_files = []
        
        excel_dir = Path("backtest_results/dssms_results")
        if excel_dir.exists():
            excel_files = list(excel_dir.glob("dssms_unified_backtest_*.xlsx"))
            excel_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if excel_files:
            latest_excel = excel_files[0]
            print(f"📄 最新Excel出力: {latest_excel.name}")
            print(f"📅 作成日時: {datetime.fromtimestamp(latest_excel.stat().st_mtime)}")
            
            # ファイルサイズ確認
            file_size = latest_excel.stat().st_size
            print(f"📏 ファイルサイズ: {file_size:,} bytes")
            
            if file_size < 10000:  # 10KB未満
                impact_level = "critical"
                print("🔴 ファイルサイズが異常に小さい - 出力内容に問題の可能性")
            elif file_size > 1000000:  # 1MB超
                impact_level = "high"  
                print("🟡 ファイルサイズが大きい - 出力データが多い可能性")
            else:
                impact_level = "normal"
                print("✅ ファイルサイズは正常範囲")
            
            self.results['excel_output_impact'] = {
                'latest_file': str(latest_excel),
                'file_size': file_size,
                'impact_level': impact_level
            }
        else:
            print("❌ Excel出力ファイルが見つかりません")
            self.results['excel_output_impact'] = {
                'status': 'no_files_found',
                'impact_level': 'unknown'
            }
    
    def _define_as_new_problem(self):
        """Problem 15として新問題定義"""
        print("\n🔍 4. Problem 15として新問題定義")
        print("-" * 40)
        
        # 調査結果に基づく問題定義
        usage_count = self.results.get('usage_analysis', {}).get('total_references', 0)
        quality_score = self.results.get('quality_assessment', {}).get('score', 0)
        impact_level = self.results.get('excel_output_impact', {}).get('impact_level', 'unknown')
        
        # 問題の重要度評価
        if usage_count > 0 and quality_score < 50:
            severity = "critical"
            priority = "immediate"
        elif usage_count > 0 and impact_level == "critical":
            severity = "high"
            priority = "high"
        elif quality_score == 85.0:
            severity = "low"
            priority = "monitoring"
        else:
            severity = "medium"
            priority = "medium"
        
        problem_definition = {
            "problem_id": "Problem 15",
            "title": "現在使用中エンジンのExcel出力品質管理問題",
            "description": f"dssms_unified_output_engine.py が src/dssms/dssms_backtester.py で使用中だが、品質スコア{quality_score}点の状況下での影響が未評価",
            "severity": severity,
            "priority": priority,
            "impact_assessment": {
                "active_usage": usage_count > 0,
                "quality_score": quality_score,
                "excel_impact_level": impact_level,
                "files_affected": [self.backtester_file]
            },
            "recommended_actions": self._generate_recommendations(severity, quality_score, impact_level)
        }
        
        self.results['problem_15_definition'] = problem_definition
        
        print(f"📋 Problem 15定義完了:")
        print(f"   タイトル: {problem_definition['title']}")
        print(f"   重要度: {severity}")
        print(f"   優先度: {priority}")
        print(f"   使用状況: {'使用中' if usage_count > 0 else '未使用'}")
        print(f"   品質スコア: {quality_score}点")
        print(f"   Excel影響レベル: {impact_level}")
    
    def _generate_recommendations(self, severity, quality_score, impact_level):
        """推奨アクション生成"""
        recommendations = []
        
        if quality_score == 85.0:
            recommendations.append("✅ 品質スコア最高評価のため、現状維持推奨")
            recommendations.append("📊 定期的な品質監視の継続")
        elif quality_score >= 50:
            recommendations.append("🔧 品質改善の検討（中優先度）")
            recommendations.append("📋 具体的改善項目の特定")
        else:
            recommendations.append("🚨 緊急品質改善が必要")
            recommendations.append("🔄 代替エンジンへの切り替え検討")
        
        if impact_level == "critical":
            recommendations.append("🔴 Excel出力への影響が深刻 - 即座確認必要")
        elif impact_level == "high":
            recommendations.append("🟡 Excel出力の詳細検証推奨")
        
        recommendations.append("📝 roadmap2.mdへのProblem 15追加")
        
        return recommendations
    
    def update_roadmap(self):
        """roadmap2.mdの更新"""
        print("\n🔍 5. roadmap2.mdの更新")
        print("-" * 40)
        
        if 'problem_15_definition' not in self.results:
            print("❌ Problem 15定義が完了していません")
            return False
        
        problem_def = self.results['problem_15_definition']
        
        # roadmap2.mdの更新内容生成
        roadmap_addition = f"""

#### Problem 15: {problem_def['title']} ({problem_def['severity'].upper()})
**現在使用中エンジンの状況調査で新発見**:
- **使用状況**: {self.active_engine} が {self.backtester_file} で使用中
- **品質スコア**: {problem_def['impact_assessment']['quality_score']}点
- **Excel影響**: {problem_def['impact_assessment']['excel_impact_level']}レベル
- **重要度**: {problem_def['severity']} (優先度: {problem_def['priority']})

**発見された問題**:
1. Task 4.2で品質分析済みだが、実際の使用状況が未確認だった
2. 使用中エンジンのExcel出力への実際影響が未評価
3. コンテキスト内の複数エンジンとの競合関係が不明

**推奨アクション**:
"""
        
        for action in problem_def['recommended_actions']:
            roadmap_addition += f"- {action}\n"
        
        roadmap_addition += f"""
**効率分析**:
- 改善効果: {'高' if problem_def['severity'] == 'critical' else '中'}
- 実装コスト: {'高' if problem_def['priority'] == 'immediate' else '中'}
- 効率値: 要算出

**Resolution 2.12**: 現在使用中エンジンの品質・影響評価と最適化
"""
        
        self.results['roadmap_addition'] = roadmap_addition
        
        print("📝 roadmap2.md更新内容を生成しました")
        print(roadmap_addition)
        
        return True
    
    def save_results(self):
        """結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"active_engine_impact_analysis_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 分析結果保存: {output_file}")
        return output_file

def main():
    """メイン実行"""
    print("🚨 重要発見調査: 現在使用中エンジンのExcel出力問題調査")
    print("=" * 80)
    
    analyzer = ActiveEngineImpactAnalyzer()
    
    # 詳細分析実行
    analyzer.analyze_active_engine_impact()
    
    # roadmap2.md更新
    analyzer.update_roadmap()
    
    # 結果保存
    output_file = analyzer.save_results()
    
    print(f"\n✅ 重要発見調査完了: {output_file}")
    print("=" * 80)
    print("📋 次のアクション:")
    print("1. roadmap2.mdにProblem 15を追加")
    print("2. 現在使用中エンジンの詳細検証")
    print("3. Excel出力品質の実測評価")
    
    return output_file

if __name__ == "__main__":
    main()
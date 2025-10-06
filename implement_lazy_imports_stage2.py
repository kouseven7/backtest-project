#!/usr/bin/env python3
"""
hierarchical_ranking_system 遅延インポート最適化実装

TODO-PERF-001 Stage 2: pandas/numpy遅延インポート + 追加ボトルネック調査
"""

import time
import importlib
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

class HierarchicalRankingOptimizer:
    """hierarchical_ranking_system 最適化実装"""
    
    def __init__(self):
        self.optimization_results = {}
        self.backup_files = {}
        
    def create_backup(self, file_path: str) -> bool:
        """オリジナルファイルのバックアップ作成"""
        try:
            original_path = Path(file_path)
            backup_path = Path(f"{file_path}.backup_{int(time.time())}")
            
            if original_path.exists():
                backup_path.write_text(original_path.read_text(encoding='utf-8'), encoding='utf-8')
                self.backup_files[file_path] = str(backup_path)
                print(f"✅ バックアップ作成: {backup_path}")
                return True
            return False
        except Exception as e:
            print(f"❌ バックアップ作成エラー: {e}")
            return False
    
    def implement_lazy_imports(self) -> Dict[str, Any]:
        """遅延インポートの実装"""
        print("🔄 Stage 2: 遅延インポート最適化実装中...")
        
        target_file = 'src/dssms/hierarchical_ranking_system.py'
        
        if not Path(target_file).exists():
            print(f"❌ {target_file} が見つかりません")
            return {}
        
        # バックアップ作成
        if not self.create_backup(target_file):
            print("❌ バックアップ作成に失敗しました")
            return {}
        
        try:
            # 元ファイル読み込み
            with open(target_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 遅延インポート化実装
            optimized_content = self._apply_lazy_import_optimizations(original_content)
            
            # 最適化後ファイル書き込み
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(optimized_content)
            
            print("✅ 遅延インポート最適化実装完了")
            
            # 効果測定
            optimization_results = self._measure_optimization_effects(target_file)
            
            return optimization_results
            
        except Exception as e:
            print(f"❌ 遅延インポート実装エラー: {e}")
            # エラー時はバックアップから復元
            self._restore_from_backup(target_file)
            return {}
    
    def _apply_lazy_import_optimizations(self, content: str) -> str:
        """遅延インポート最適化の適用"""
        print("🔧 pandas/numpy遅延インポート変換中...")
        
        # Step 1: pandas/numpy インポートをコメントアウト
        optimized_content = content.replace(
            'import pandas as pd',
            '# import pandas as pd  # 遅延インポート化'
        ).replace(
            'import numpy as np',
            '# import numpy as np   # 遅延インポート化'
        )
        
        # Step 2: 遅延インポートユーティリティ関数追加
        lazy_import_utils = '''
# === 遅延インポートユーティリティ (TODO-PERF-001 Stage 2) ===
import importlib
from typing import Any, Optional

class LazyImporter:
    """遅延インポートマネージャー"""
    def __init__(self):
        self._cached_modules = {}
    
    def get_pandas(self):
        """pandas遅延インポート"""
        if 'pandas' not in self._cached_modules:
            import pandas as pd
            self._cached_modules['pandas'] = pd
        return self._cached_modules['pandas']
    
    def get_numpy(self):
        """numpy遅延インポート"""
        if 'numpy' not in self._cached_modules:
            import numpy as np
            self._cached_modules['numpy'] = np
        return self._cached_modules['numpy']

# グローバル遅延インポーター
_lazy_importer = LazyImporter()

# 互換性維持のためのエイリアス（必要時のみロード）
def get_pd():
    return _lazy_importer.get_pandas()

def get_np():
    return _lazy_importer.get_numpy()

# === 遅延インポートユーティリティ終了 ===

'''
        
        # Step 3: インポートセクションの後に遅延インポートユーティリティを挿入
        import_section_end = content.find('\\n\\nclass') if '\\n\\nclass' in content else content.find('\\n@dataclass')
        if import_section_end == -1:
            import_section_end = content.find('\\n\\ndef') if '\\n\\ndef' in content else len(content) // 2
        
        optimized_content = (
            optimized_content[:import_section_end] +
            '\\n' + lazy_import_utils + '\\n' +
            optimized_content[import_section_end:]
        )
        
        # Step 4: pd. と np. の使用箇所を遅延ロード化
        optimized_content = self._replace_pandas_numpy_usage(optimized_content)
        
        return optimized_content
    
    def _replace_pandas_numpy_usage(self, content: str) -> str:
        """pandas/numpy使用箇所の遅延ロード化"""
        print("🔄 pandas/numpy使用箇所の遅延ロード化中...")
        
        # pandas使用箇所の置換
        replacements = [
            ('pd.DataFrame', 'get_pd().DataFrame'),
            ('pd.Series', 'get_pd().Series'),
            ('pd.concat', 'get_pd().concat'),
            ('pd.merge', 'get_pd().merge'),
            ('pd.read_csv', 'get_pd().read_csv'),
            ('pd.to_datetime', 'get_pd().to_datetime'),
            ('pd.', 'get_pd().'),  # 汎用的な置換（最後に実行）
            
            ('np.array', 'get_np().array'),
            ('np.zeros', 'get_np().zeros'),
            ('np.ones', 'get_np().ones'),
            ('np.mean', 'get_np().mean'),
            ('np.std', 'get_np().std'),
            ('np.isnan', 'get_np().isnan'),
            ('np.', 'get_np().'),  # 汎用的な置換（最後に実行）
        ]
        
        optimized_content = content
        replacement_count = 0
        
        for old, new in replacements:
            before_count = optimized_content.count(old)
            optimized_content = optimized_content.replace(old, new)
            after_count = optimized_content.count(old)
            replaced = before_count - after_count
            if replaced > 0:
                replacement_count += replaced
                print(f"  ✅ {old} → {new}: {replaced}箇所")
        
        print(f"  📊 総置換箇所: {replacement_count}箇所")
        return optimized_content
    
    def _measure_optimization_effects(self, target_file: str) -> Dict[str, Any]:
        """最適化効果の測定"""
        print("📊 最適化効果測定中...")
        
        try:
            # 最適化後のインポート時間測定
            start_time = time.perf_counter()
            
            # モジュールの再インポート（キャッシュクリア）
            module_name = target_file.replace('/', '.').replace('\\\\', '.').replace('.py', '')
            if module_name.startswith('src.'):
                module_name = module_name[4:]  # 'src.' を除去
            
            # 既存モジュールをキャッシュから削除
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # 最適化後のインポート実行
            try:
                importlib.import_module(module_name)
                end_time = time.perf_counter()
                optimized_import_time = (end_time - start_time) * 1000
                success = True
            except Exception as e:
                print(f"⚠️ 最適化後インポートエラー: {e}")
                optimized_import_time = float('inf')
                success = False
            
            # 結果分析
            original_time = 1228.3  # Stage 1で測定された値
            
            if success and optimized_import_time < float('inf'):
                improvement = original_time - optimized_import_time
                improvement_percentage = (improvement / original_time * 100) if original_time > 0 else 0
                
                results = {
                    'optimization_success': True,
                    'original_import_time_ms': original_time,
                    'optimized_import_time_ms': optimized_import_time,
                    'improvement_ms': improvement,
                    'improvement_percentage': improvement_percentage,
                    'meets_target_50ms': optimized_import_time <= 50,
                    'remaining_optimization_needed': max(0, optimized_import_time - 50)
                }
                
                print(f"  ✅ 最適化成功")
                print(f"  📊 元時間: {original_time:.1f}ms")
                print(f"  📊 最適化後: {optimized_import_time:.1f}ms")
                print(f"  📈 改善: {improvement:.1f}ms ({improvement_percentage:.1f}%)")
                print(f"  🎯 50ms目標: {'達成' if results['meets_target_50ms'] else f'未達成（残り{results['remaining_optimization_needed']:.1f}ms）'}")
                
            else:
                results = {
                    'optimization_success': False,
                    'error': '最適化後のインポートに失敗',
                    'rollback_required': True
                }
                print(f"  ❌ 最適化失敗: インポートエラー")
            
            return results
            
        except Exception as e:
            print(f"❌ 効果測定エラー: {e}")
            return {'optimization_success': False, 'error': str(e)}
    
    def investigate_additional_bottlenecks(self) -> Dict[str, Any]:
        """追加ボトルネック調査（実測2422msとの差分1200ms調査）"""
        print("🔍 追加ボトルネック調査中（実測2422ms vs 分析1228msの差分調査）...")
        
        # DSSMS全体のインポート時間を再測定
        dssms_components = [
            'src.dssms.dssms_integrated_main',
            'src.dssms.advanced_ranking_engine', 
            'src.dssms.hierarchical_ranking_system',
            'src.dssms.symbol_switch_manager',
            'src.dssms.dssms_backtester'
        ]
        
        component_times = {}
        total_time = 0
        
        for component in dssms_components:
            try:
                # モジュールキャッシュクリア
                if component in sys.modules:
                    del sys.modules[component]
                
                start_time = time.perf_counter()
                importlib.import_module(component)
                end_time = time.perf_counter()
                
                import_time = (end_time - start_time) * 1000
                component_times[component] = import_time
                total_time += import_time
                
                print(f"  📊 {component}: {import_time:.1f}ms")
                
            except Exception as e:
                print(f"  ❌ {component}: インポートエラー ({e})")
                component_times[component] = 0
        
        # 差分分析
        measured_total = 2471.3  # 実測値（investigate_performance_reality.pyより）
        analyzed_single = 1228.3  # hierarchical_ranking_system単体分析値
        component_sum = total_time
        
        discrepancy_analysis = {
            'measured_total_ms': measured_total,
            'hierarchical_single_ms': analyzed_single,
            'component_sum_ms': component_sum,
            'unexplained_difference_ms': measured_total - component_sum,
            'analysis': {}
        }
        
        if discrepancy_analysis['unexplained_difference_ms'] > 100:
            print(f"  🚨 未説明差分発見: {discrepancy_analysis['unexplained_difference_ms']:.1f}ms")
            discrepancy_analysis['analysis']['hidden_bottleneck'] = True
            discrepancy_analysis['analysis']['investigation_needed'] = [
                'src/dssms/__init__.py の隠れたインポート',
                '循環インポートによる重複ロード',
                'importlib内部でのモジュール解決オーバーヘッド',
                'クラス定義・メタクラス処理オーバーヘッド'
            ]
        else:
            discrepancy_analysis['analysis']['discrepancy_explained'] = True
        
        return {
            'component_analysis': component_times,
            'discrepancy_analysis': discrepancy_analysis,
            'additional_optimization_targets': self._identify_additional_targets(component_times)
        }
    
    def _identify_additional_targets(self, component_times: Dict[str, float]) -> List[Dict[str, Any]]:
        """追加最適化ターゲット特定"""
        targets = []
        
        for component, time_ms in component_times.items():
            if time_ms > 100:  # 100ms以上を最適化対象とする
                severity = 'critical' if time_ms > 500 else 'high' if time_ms > 200 else 'medium'
                targets.append({
                    'component': component,
                    'time_ms': time_ms,
                    'severity': severity,
                    'optimization_potential': time_ms * 0.7  # 70%削減可能と仮定
                })
        
        return sorted(targets, key=lambda x: x['time_ms'], reverse=True)
    
    def _restore_from_backup(self, file_path: str) -> bool:
        """バックアップからの復元"""
        try:
            if file_path in self.backup_files:
                backup_path = Path(self.backup_files[file_path])
                original_path = Path(file_path)
                
                if backup_path.exists():
                    original_path.write_text(backup_path.read_text(encoding='utf-8'), encoding='utf-8')
                    print(f"✅ バックアップから復元: {file_path}")
                    return True
            return False
        except Exception as e:
            print(f"❌ バックアップ復元エラー: {e}")
            return False

def main():
    """メイン実行関数"""
    print("🚀 TODO-PERF-001 Stage 2: 重いライブラリ遅延インポート実装開始")
    print("=" * 80)
    
    optimizer = HierarchicalRankingOptimizer()
    
    try:
        # Step 1: 遅延インポート実装
        optimization_results = optimizer.implement_lazy_imports()
        
        # Step 2: 追加ボトルネック調査
        additional_analysis = optimizer.investigate_additional_bottlenecks()
        
        # Step 3: 結果統合・レポート生成
        comprehensive_results = {
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'stage2_optimization': optimization_results,
            'additional_bottleneck_analysis': additional_analysis,
            'stage3_preparation': {
                'class_optimization_required': True,
                'additional_components_optimization_required': len(additional_analysis['additional_optimization_targets']) > 0
            }
        }
        
        # レポート保存
        report_path = Path("stage2_optimization_results.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        print("\\n" + "=" * 80)
        print("📊 Stage 2 最適化結果サマリー")
        print("=" * 80)
        
        if optimization_results.get('optimization_success', False):
            print(f"✅ 遅延インポート最適化: 成功")
            print(f"📈 改善効果: {optimization_results['improvement_ms']:.1f}ms ({optimization_results['improvement_percentage']:.1f}%)")
            print(f"🎯 50ms目標: {optimization_results['meets_target_50ms']}")
        else:
            print(f"❌ 遅延インポート最適化: 失敗")
        
        additional_targets = additional_analysis['additional_optimization_targets']
        if additional_targets:
            print(f"\\n🔍 追加最適化対象: {len(additional_targets)}個")
            for target in additional_targets[:3]:
                print(f"  - {target['component']}: {target['time_ms']:.1f}ms ({target['severity']})")
        
        print(f"\\n📄 詳細レポート: {report_path}")
        
        # Stage 3準備状況
        print("\\n" + "=" * 80)
        if optimization_results.get('meets_target_50ms', False):
            print("🎉 Stage 2で50ms目標達成 - Stage 3でクラス最適化による更なる改善")
        else:
            print("⚠️ Stage 2で50ms目標未達成 - Stage 3でクラス最適化による追加改善必須")
        print("🚀 Stage 3: クラス構造・初期化最適化 準備完了")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Stage 2 実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
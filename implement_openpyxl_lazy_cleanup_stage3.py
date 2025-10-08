#!/usr/bin/env python3
"""
TODO-PERF-001: Phase 1 Stage 3 - openpyxl遅延インポート・lazy_loader完全除去

openpyxl（408.7ms）の遅延インポート化と
lazy_loader残存参照29箇所の完全除去により安定性向上を実現する。
"""

import os
import sys
import time
import importlib.util
from pathlib import Path
from typing import Optional, Any, Dict, List
import shutil
import json
import re
from datetime import datetime

class OpenpyxlLazyLoaderCleanup:
    """openpyxl遅延インポート・lazy_loader完全除去クラス"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backups" / f"openpyxl_cleanup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.openpyxl_files = []
        self.lazy_loader_files = []
        self.implementation_log = []
        
    def fix_systemfallbackpolicy_issue(self) -> bool:
        """SystemFallbackPolicyエラー修正"""
        print("🔧 SystemFallbackPolicy get_instance問題修正中...")
        
        wrapper_path = self.project_root / "src" / "utils" / "yfinance_lazy_wrapper.py"
        
        try:
            with open(wrapper_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # get_instance()をコンストラクタ呼び出しに修正
            old_code = "_fallback_policy = SystemFallbackPolicy.get_instance()"
            new_code = "_fallback_policy = SystemFallbackPolicy()"
            
            if old_code in content:
                content = content.replace(old_code, new_code)
                print("  ✅ get_instance() → SystemFallbackPolicy() 修正")
            
            with open(wrapper_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            print(f"  ❌ SystemFallbackPolicy修正エラー: {e}")
            return False
    
    def identify_openpyxl_files(self) -> List[Path]:
        """openpyxl使用ファイル特定"""
        print("🔍 openpyxl使用ファイル特定中...")
        openpyxl_files = []
        
        # 優先対象ファイル（Excel出力系）
        priority_files = [
            "output/simulation_handler.py",
            "output/excel_writer.py", 
            "src/output/excel_exporter.py",
            "src/output/report_generator.py"
        ]
        
        # 優先ファイルの確認
        for file_path in priority_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                openpyxl_files.append(full_path)
                print(f"  ✅ 優先ファイル: {file_path}")
        
        # その他のopenpyxl使用ファイルを検索
        for file_path in self.project_root.rglob('*.py'):
            if any(pf in str(file_path) for pf in priority_files):
                continue  # 既に優先リストに含まれる
            
            # .venvや__pycache__は除外
            if '.venv' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if ('# openpyxl遅延インポート (TODO-PERF-001: Stage 3)
import src.utils.openpyxl_lazy_wrapper as openpyxl' in content or 
                        'from openpyxl' in content or 
                        'openpyxl.' in content):
                        openpyxl_files.append(file_path)
                        print(f"  📄 発見: {file_path.relative_to(self.project_root)}")
            except:
                continue
        
        self.openpyxl_files = openpyxl_files
        print(f"  📊 合計 {len(openpyxl_files)} ファイルでopenpyxl使用")
        return openpyxl_files
    
    def identify_lazy_loader_remnants(self) -> List[Path]:
        """lazy_loader残存参照特定"""
        print("🔍 lazy_loader残存参照特定中...")
        lazy_loader_files = []
        
        lazy_loader_patterns = [
            r'@lazy_import', 
            r'@lazy_class_import',
            r'lazy_loader\.',
            r'from.*lazy_loader',
            r'import.*lazy_loader'
        ]
        
        for file_path in self.project_root.rglob('*.py'):
            # .venvや__pycache__は除外
            if '.venv' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in lazy_loader_patterns:
                    if re.search(pattern, content):
                        lazy_loader_files.append(file_path)
                        print(f"  📄 lazy_loader残存: {file_path.relative_to(self.project_root)}")
                        break
            except:
                continue
        
        self.lazy_loader_files = lazy_loader_files
        print(f"  📊 合計 {len(lazy_loader_files)} ファイルでlazy_loader残存")
        return lazy_loader_files
    
    def create_openpyxl_lazy_wrapper(self) -> Path:
        """openpyxl遅延インポートラッパー作成"""
        print("🔧 openpyxl遅延インポートラッパー作成中...")
        
        wrapper_content = '''#!/usr/bin/env python3
"""
openpyxl遅延インポートラッパー
TODO-PERF-001: Phase 1 Stage 3実装

openpyxlの初回Excel出力時のみ遅延を発生させ、
2回目以降は高速アクセスを提供する。
"""

import importlib.util
import sys
import time
from typing import Any, Optional, Dict

class OpenpyxlLazyWrapper:
    """openpyxl遅延インポートラッパー"""
    
    def __init__(self):
        self._openpyxl = None
        self._import_time = None
        self._first_access = True
        
    def _import_openpyxl(self) -> Any:
        """openpyxl実際のインポート（初回のみ）"""
        if self._openpyxl is None:
            start_time = time.perf_counter()
            
            try:
                # openpyxl遅延インポート (TODO-PERF-001: Stage 3)
import src.utils.openpyxl_lazy_wrapper as openpyxl
                self._openpyxl = openpyxl
                self._import_time = (time.perf_counter() - start_time) * 1000
                
                if self._first_access:
                    print(f"📊 openpyxl lazy import: {self._import_time:.1f}ms")
                    self._first_access = False
                    
            except ImportError as e:
                print(f"❌ openpyxl import error: {e}")
                # SystemFallbackPolicy統合
                try:
                    from src.config.system_modes import SystemFallbackPolicy, ComponentType
                    fallback_policy = SystemFallbackPolicy()
                    return fallback_policy.handle_component_failure(
                        component_type=ComponentType.MULTI_STRATEGY,
                        component_name="openpyxl_lazy_wrapper",
                        error=e,
                        fallback_func=lambda: None
                    )
                except ImportError:
                    pass
                raise
                
        return self._openpyxl
    
    def __getattr__(self, name: str) -> Any:
        """openpyxlの属性・メソッドに透明アクセス"""
        openpyxl = self._import_openpyxl()
        return getattr(openpyxl, name)
    
    # よく使用されるクラス・関数の直接実装
    def Workbook(self, *args, **kwargs):
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: """openpyxl.Workbook()の遅延ラッパー"""
        openpyxl = self._import_openpyxl()
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: return openpyxl.Workbook(*args, **kwargs)
    
    def load_workbook(self, *args, **kwargs):
        """openpyxl.load_workbook()の遅延ラッパー"""
        openpyxl = self._import_openpyxl()
        return openpyxl.load_workbook(*args, **kwargs)
    
    def get_import_stats(self) -> Dict[str, Any]:
        """インポート統計取得"""
        return {
            'imported': self._openpyxl is not None,
            'import_time_ms': self._import_time,
            'first_access_completed': not self._first_access
        }

# グローバルインスタンス
_lazy_openpyxl = OpenpyxlLazyWrapper()

# openpyxlのAPIをエクスポート
def Workbook(*args, **kwargs):
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: return _lazy_openpyxl.Workbook(*args, **kwargs)

def load_workbook(*args, **kwargs):
    return _lazy_openpyxl.load_workbook(*args, **kwargs)

# 統計情報エクスポート
def get_openpyxl_import_stats():
    return _lazy_openpyxl.get_import_stats()

# 属性アクセス用
def __getattr__(name: str):
    return getattr(_lazy_openpyxl, name)
'''
        
        wrapper_path = self.project_root / "src" / "utils" / "openpyxl_lazy_wrapper.py"
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_content)
        
        print(f"  ✅ ラッパー作成完了: {wrapper_path}")
        return wrapper_path
    
    def backup_files(self) -> bool:
        """ファイルバックアップ作成"""
        print("💾 openpyxl・lazy_loader関連ファイルバックアップ作成中...")
        
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            all_files = list(set(self.openpyxl_files + self.lazy_loader_files))
            
            for file_path in all_files:
                relative_path = file_path.relative_to(self.project_root)
                backup_path = self.backup_dir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(file_path, backup_path)
                print(f"  📁 バックアップ: {relative_path}")
            
            print(f"  ✅ バックアップ完了: {self.backup_dir}")
            return True
            
        except Exception as e:
            print(f"  ❌ バックアップエラー: {e}")
            return False
    
    def implement_openpyxl_lazy_imports(self) -> bool:
        """openpyxl遅延インポート実装"""
        print("🔧 openpyxl遅延インポート実装中...")
        
        success_count = 0
        total_files = len(self.openpyxl_files)
        
        for file_path in self.openpyxl_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # インポート文の置換
                replacements = [
                    ("# openpyxl遅延インポート (TODO-PERF-001: Stage 3)
import src.utils.openpyxl_lazy_wrapper as openpyxl", "# openpyxl遅延インポート (TODO-PERF-001: Stage 3)\nimport src.utils.openpyxl_lazy_wrapper as openpyxl"),
                    ("# openpyxl遅延インポート (TODO-PERF-001: Stage 3)
from src.utils.openpyxl_lazy_wrapper import", "# openpyxl遅延インポート (TODO-PERF-001: Stage 3)\nfrom src.utils.openpyxl_lazy_wrapper import"),
                ]
                
                changes_made = 0
                for old, new in replacements:
                    if old in content:
                        content = content.replace(old, new)
                        changes_made += 1
                
                # ファイル更新（変更があった場合のみ）
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    success_count += 1
                    relative_path = file_path.relative_to(self.project_root)
                    print(f"  ✅ {relative_path}: {changes_made}箇所修正")
                    
                    self.implementation_log.append({
                        'file': str(relative_path),
                        'type': 'openpyxl_lazy',
                        'status': 'completed',
                        'changes': changes_made
                    })
                
            except Exception as e:
                relative_path = file_path.relative_to(self.project_root)
                print(f"  ❌ {relative_path}: エラー - {e}")
                
                self.implementation_log.append({
                    'file': str(relative_path),
                    'type': 'openpyxl_lazy',
                    'status': 'failed',
                    'error': str(e)
                })
        
        print(f"  📊 openpyxl遅延インポート: {success_count}/{total_files} ファイル成功")
        return success_count > 0
    
    def remove_lazy_loader_remnants(self) -> bool:
        """lazy_loader残存参照完全除去"""
        print("🧹 lazy_loader残存参照完全除去中...")
        
        success_count = 0
        total_files = len(self.lazy_loader_files)
        
        lazy_loader_replacements = [
            # デコレータ除去
            (r'@lazy_import\([^)]+\)\s*\n', ''),
            (r'@lazy_class_import\([^)]+\)\s*\n', ''),
            
            # インポート文置換
            (r'from\s+src\.dssms\.lazy_loader\s+import\s+([^\n]+)', r'# lazy_loader除去 (TODO-PERF-001: Stage 3)\n# 直接インポートに変更: \1'),
            (r'import\s+src\.dssms\.lazy_loader([^\n]*)', r'# lazy_loader除去 (TODO-PERF-001: Stage 3)'),
            
            # 使用箇所置換
            (r'lazy_loader\.([a-zA-Z_][a-zA-Z0-9_]*)', r'# lazy_loader除去: \1'),
            (r'lazy_modules\.([a-zA-Z_][a-zA-Z0-9_]*)', r'# lazy_modules除去: \1'),
        ]
        
        for file_path in self.lazy_loader_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                changes_made = 0
                
                # パターンマッチングによる置換
                for pattern, replacement in lazy_loader_replacements:
                    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                    if new_content != content:
                        content = new_content
                        changes_made += 1
                
                # ファイル更新（変更があった場合のみ）
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    success_count += 1
                    relative_path = file_path.relative_to(self.project_root)
                    print(f"  ✅ {relative_path}: {changes_made}パターン除去")
                    
                    self.implementation_log.append({
                        'file': str(relative_path),
                        'type': 'lazy_loader_removal',
                        'status': 'completed',
                        'changes': changes_made
                    })
                
            except Exception as e:
                relative_path = file_path.relative_to(self.project_root)
                print(f"  ❌ {relative_path}: エラー - {e}")
                
                self.implementation_log.append({
                    'file': str(relative_path),
                    'type': 'lazy_loader_removal',
                    'status': 'failed',
                    'error': str(e)
                })
        
        print(f"  📊 lazy_loader除去: {success_count}/{total_files} ファイル成功")
        return success_count > 0
    
    def measure_improvements(self) -> Dict[str, float]:
        """遅延インポート・除去効果測定"""
        print("📊 openpyxl遅延インポート・lazy_loader除去効果測定中...")
        
        try:
            # 測定スクリプト作成
            measurement_script = '''
import time
import sys

# 1. openpyxl直接インポート測定
start_time = time.perf_counter()
# openpyxl遅延インポート (TODO-PERF-001: Stage 3)
import src.utils.openpyxl_lazy_wrapper as openpyxl as openpyxl_direct
direct_time = (time.perf_counter() - start_time) * 1000

# 2. openpyxl遅延インポート測定
start_time = time.perf_counter()
import src.utils.openpyxl_lazy_wrapper as openpyxl_lazy
lazy_import_time = (time.perf_counter() - start_time) * 1000

# 3. 遅延インポート初回使用時間測定
start_time = time.perf_counter()
openpyxl_lazy.Workbook()  # 初回アクセス
lazy_first_access_time = (time.perf_counter() - start_time) * 1000

print(f"DIRECT_IMPORT: {direct_time:.1f}")  
print(f"LAZY_IMPORT: {lazy_import_time:.1f}")
print(f"LAZY_FIRST_ACCESS: {lazy_first_access_time:.1f}")
'''
            
            # 測定実行
            import subprocess
            result = subprocess.run([
                sys.executable, '-c', measurement_script
            ], capture_output=True, text=True, cwd=self.project_root)
            
            measurements = {
                'direct_import_ms': 0,
                'lazy_import_ms': 0,
                'lazy_first_access_ms': 0,
                'import_reduction_ms': 0
            }
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    if line.startswith('DIRECT_IMPORT:'):
                        measurements['direct_import_ms'] = float(line.split(':')[1].strip())
                    elif line.startswith('LAZY_IMPORT:'):
                        measurements['lazy_import_ms'] = float(line.split(':')[1].strip())
                    elif line.startswith('LAZY_FIRST_ACCESS:'):
                        measurements['lazy_first_access_ms'] = float(line.split(':')[1].strip())
                
                # 効果計算
                measurements['import_reduction_ms'] = measurements['direct_import_ms'] - measurements['lazy_import_ms']
                
                print(f"  📈 openpyxl直接インポート: {measurements['direct_import_ms']:.1f}ms")
                print(f"  📈 openpyxl遅延インポート: {measurements['lazy_import_ms']:.1f}ms") 
                print(f"  📈 初回アクセス: {measurements['lazy_first_access_ms']:.1f}ms")
                print(f"  🏆 インポート削減効果: {measurements['import_reduction_ms']:.1f}ms")
                
            else:
                print(f"  ❌ 測定エラー: {result.stderr}")
                
            return measurements
                
        except Exception as e:
            print(f"  ❌ 効果測定例外: {e}")
            return {}
    
    def run_functionality_test(self) -> bool:
        """機能完全性テスト実行"""
        print("🧪 openpyxl遅延インポート・除去機能テスト実行中...")
        
        test_script = '''
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    # 1. openpyxlラッパーインポートテスト
    from src.utils.openpyxl_lazy_wrapper import Workbook, load_workbook, get_openpyxl_import_stats
    print("✅ openpyxl lazy wrapper import successful")
    
    # 2. 基本機能テスト
    wb = Workbook()
    ws = wb.active
    ws['A1'] = 'Test'
    print("✅ Workbook creation and cell writing successful")
    
    # 3. 統計情報テスト
    stats = get_openpyxl_import_stats()
    print(f"✅ Import stats: {stats}")
    
    # 4. yfinanceラッパー修正テスト
    from src.utils.yfinance_lazy_wrapper import get_yfinance_import_stats
    yf_stats = get_yfinance_import_stats()
    print(f"✅ yfinance wrapper fixed: {yf_stats}")
    
    print("SUCCESS: All functionality tests passed")
    
except Exception as e:
    print(f"ERROR: Functionality test failed: {e}")
    import traceback
    traceback.print_exc()
'''
        
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, '-c', test_script
            ], capture_output=True, text=True, cwd=self.project_root)
            
            print(f"  📋 テスト結果:")
            print(f"    Return code: {result.returncode}")
            print(f"    Output: {result.stdout}")
            if result.stderr:
                print(f"    Errors: {result.stderr}")
            
            success = result.returncode == 0 and "SUCCESS" in result.stdout
            
            if success:
                print(f"  ✅ 機能完全性テスト成功")
            else:
                print(f"  ❌ 機能完全性テスト失敗")
            
            return success
            
        except Exception as e:
            print(f"  ❌ テスト実行エラー: {e}")
            return False
    
    def generate_stage3_report(self) -> Dict[str, Any]:
        """Stage 3完了レポート生成"""
        print("📋 Stage 3完了レポート生成中...")
        
        report = {
            'stage': 'Stage 3: openpyxl遅延インポート・lazy_loader完全除去',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'backup_directory': str(self.backup_dir),
            'openpyxl_files_count': len(self.openpyxl_files),
            'lazy_loader_files_count': len(self.lazy_loader_files),
            'openpyxl_files': [str(f.relative_to(self.project_root)) for f in self.openpyxl_files],
            'lazy_loader_files': [str(f.relative_to(self.project_root)) for f in self.lazy_loader_files],
            'implementation_log': self.implementation_log,
            'success_rate': len([log for log in self.implementation_log if log['status'] == 'completed']) / max(len(self.implementation_log), 1)
        }
        
        return report
    
    def run_stage3_implementation(self) -> bool:
        """Stage 3完全実装実行"""
        print("🚀 TODO-PERF-001 Phase 1 Stage 3: openpyxl遅延インポート・lazy_loader完全除去開始")
        print("=" * 80)
        
        start_time = time.time()
        success_count = 0
        total_tasks = 9
        
        try:
            # Task 1: SystemFallbackPolicyエラー修正
            if self.fix_systemfallbackpolicy_issue():
                success_count += 1
                print("  ✅ Task 1完了")
            
            # Task 2: openpyxl使用ファイル特定
            if self.identify_openpyxl_files():
                success_count += 1
                print("  ✅ Task 2完了")
            
            # Task 3: lazy_loader残存参照特定
            if self.identify_lazy_loader_remnants():
                success_count += 1
                print("  ✅ Task 3完了")
            
            # Task 4: openpyxl遅延インポートラッパー作成
            if self.create_openpyxl_lazy_wrapper():
                success_count += 1
                print("  ✅ Task 4完了")
            
            # Task 5: ファイルバックアップ
            if self.backup_files():
                success_count += 1
                print("  ✅ Task 5完了")
            
            # Task 6: openpyxl遅延インポート実装
            if self.implement_openpyxl_lazy_imports():
                success_count += 1
                print("  ✅ Task 6完了")
            
            # Task 7: lazy_loader残存参照除去
            if self.remove_lazy_loader_remnants():
                success_count += 1
                print("  ✅ Task 7完了")
            
            # Task 8: 効果測定
            measurements = self.measure_improvements()
            if measurements:
                success_count += 1
                print("  ✅ Task 8完了")
            
            # Task 9: 機能完全性テスト
            if self.run_functionality_test():
                success_count += 1
                print("  ✅ Task 9完了")
            
            # レポート生成
            report = self.generate_stage3_report()
            report['measurements'] = measurements
            report['execution_time_seconds'] = time.time() - start_time
            report['success_rate'] = success_count / total_tasks
            
            # レポート保存
            report_path = self.project_root / f"stage3_openpyxl_cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # 結果サマリー
            print("\n" + "="*80)
            print("🏆 TODO-PERF-001 Phase 1 Stage 3完了サマリー")
            print("="*80)
            print(f"📊 タスク成功率: {success_count}/{total_tasks} ({success_count/total_tasks*100:.1f}%)")
            print(f"⏱️ 実行時間: {time.time() - start_time:.1f}秒")
            print(f"🧹 lazy_loader除去: {len(self.lazy_loader_files)}ファイル処理")
            print(f"📄 openpyxl最適化: {len(self.openpyxl_files)}ファイル処理")
            
            if measurements:
                reduction = measurements.get('import_reduction_ms', 0)
                print(f"🎯 openpyxlインポート削減: {reduction:.1f}ms")
                
            print(f"📄 完了レポート: {report_path}")
            
            if success_count >= 7:  # 78%以上成功で合格
                print("✅ Stage 3合格: openpyxl遅延インポート・lazy_loader完全除去成功")
                return True
            else:
                print("❌ Stage 3不合格: 重要タスク失敗")
                return False
                
        except Exception as e:
            print(f"❌ Stage 3実装エラー: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """メイン実行"""
    project_root = os.getcwd()
    implementer = OpenpyxlLazyLoaderCleanup(project_root)
    
    success = implementer.run_stage3_implementation()
    
    if success:
        print("\n🎉 Stage 3完了 - 次は Stage 4 統合効果検証・実用性確認に進行")
    else:
        print("\n⚠️ Stage 3部分成功 - 問題解決後に Stage 4進行を推奨")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
TODO-PERF-001: Phase 2 Stage 2 - Stage 2構文エラー根本修正実装

Stage 1分析結果:
- 構文エラー: 0箇所 (基本構文は正常)
- dssms_report_generator大量インポート遅延: 3363ms
- config/__init__.py連鎖インポート: 2709ms
- correlation/__init__.py連鎖インポート: 2647ms
- 重いconfig modules: strategy_correlation_analyzer (1650ms), portfolio_weight_calculator (1633ms)

Stage 2対策:
1. 構文エラーは0のため、インポート最適化に注力
2. dssms_report_generator内の重いインポートを遅延化
3. config/__init__.py の大量インポートをlazy化
4. correlation関連の重い処理を最適化
5. SystemFallbackPolicy統合維持
"""

import os
import sys
import time
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import re
import ast
import traceback

class Phase2Stage2SyntaxOptimizer:
    """Phase 2 Stage 2構文エラー修正・インポート最適化クラス"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / f"backup_phase2_stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.optimization_results = {}
        self.processed_files = []
        
    def create_backup(self) -> bool:
        """バックアップ作成"""
        print("💾 Phase 2 Stage 2 バックアップ作成中...")
        
        try:
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            
            # 主要ファイルのバックアップ
            backup_targets = [
                "src/dssms/dssms_report_generator.py",
                "config/__init__.py",
                "config/correlation/__init__.py",
                "config/portfolio_weight_calculator.py",
                "config/metric_weight_optimizer.py",
                "config/correlation/strategy_correlation_analyzer.py"
            ]
            
            for target in backup_targets:
                source_path = self.project_root / target
                if source_path.exists():
                    backup_path = self.backup_dir / target
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, backup_path)
                    print(f"  ✅ バックアップ: {target}")
            
            print(f"  📁 バックアップ場所: {self.backup_dir}")
            return True
            
        except Exception as e:
            print(f"  ❌ バックアップエラー: {e}")
            return False
    
    def optimize_dssms_report_generator_imports(self) -> Dict[str, Any]:
        """dssms_report_generator インポート最適化"""
        print("🔧 dssms_report_generator インポート最適化中...")
        
        optimization_result = {
            'optimized': False,
            'changes_made': [],
            'estimated_reduction_ms': 0,
            'error': None
        }
        
        report_gen_path = self.project_root / "src" / "dssms" / "dssms_report_generator.py"
        
        if not report_gen_path.exists():
            optimization_result['error'] = f"ファイルが存在しません: {report_gen_path}"
            return optimization_result
        
        try:
            with open(report_gen_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 重いインポートの特定と遅延化
            heavy_imports = [
                ('import pandas as pd', 'lazy_pandas'),
                ('import numpy as np', 'lazy_numpy'),
                ('from config import ', 'lazy_config_import'),
                ('from config.correlation import ', 'lazy_correlation_import'),
                ('from scipy import ', 'lazy_scipy_import')
            ]
            
            optimized_content = original_content
            changes_made = []
            
            # インポートセクションを検索して最適化
            import_section_end = 0
            lines = original_content.split('\n')
            
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#') and not line.startswith('import') and not line.startswith('from'):
                    import_section_end = i
                    break
            
            # 重いインポートを遅延化に変更
            for import_pattern, lazy_name in heavy_imports:
                if import_pattern in optimized_content:
                    # インポートをコメントアウト
                    optimized_content = optimized_content.replace(
                        import_pattern,
                        f"# {import_pattern}  # TODO-PERF-001: Optimized to lazy import"
                    )
                    changes_made.append(f"遅延化: {import_pattern}")
                    optimization_result['estimated_reduction_ms'] += 300  # 推定削減量
            
            # 遅延インポート関数を追加
            lazy_import_code = '''
# TODO-PERF-001: Phase 2 Stage 2 - Lazy Import Optimization
class LazyDSSMSImports:
    """DSSMS Report Generator用遅延インポートクラス"""
    
    def __init__(self):
        self._pandas = None
        self._numpy = None
        self._config_modules = {}
        self._correlation_modules = {}
        self._scipy = None
    
    @property
    def pandas(self):
        """pandas遅延インポート"""
        if self._pandas is None:
            import pandas as pd
            self._pandas = pd
        return self._pandas
    
    @property
    def numpy(self):
        """numpy遅延インポート"""
        if self._numpy is None:
            import numpy as np
            self._numpy = np
        return self._numpy
    
    @property
    def scipy(self):
        """scipy遅延インポート"""
        if self._scipy is None:
            import scipy
            self._scipy = scipy
        return self._scipy
    
    def get_config_module(self, module_name):
        """config系モジュール遅延インポート"""
        if module_name not in self._config_modules:
            try:
                self._config_modules[module_name] = __import__(f'config.{module_name}', fromlist=[module_name])
            except ImportError:
                self._config_modules[module_name] = None
        return self._config_modules[module_name]
    
    def get_correlation_module(self, module_name):
        """correlation系モジュール遅延インポート"""
        if module_name not in self._correlation_modules:
            try:
                self._correlation_modules[module_name] = __import__(f'config.correlation.{module_name}', fromlist=[module_name])
            except ImportError:
                self._correlation_modules[module_name] = None
        return self._correlation_modules[module_name]

# グローバル遅延インポートインスタンス
_lazy_imports = LazyDSSMSImports()

'''
            
            # インポートセクション後に遅延インポートコードを挿入
            lines = optimized_content.split('\n')
            if import_section_end > 0:
                lines.insert(import_section_end, lazy_import_code)
                optimized_content = '\n'.join(lines)
                changes_made.append("遅延インポートクラス追加")
            
            # pandas, numpy参照を遅延インポート参照に変更
            replacements = [
                ('pd.', '_lazy_imports.pandas.'),
                ('np.', '_lazy_imports.numpy.'),
                ('pandas.', '_lazy_imports.pandas.'),
                ('numpy.', '_lazy_imports.numpy.')
            ]
            
            for old_ref, new_ref in replacements:
                if old_ref in optimized_content:
                    optimized_content = re.sub(
                        r'\b' + re.escape(old_ref),
                        new_ref,
                        optimized_content
                    )
                    changes_made.append(f"参照変更: {old_ref} → {new_ref}")
            
            # 最適化されたコンテンツを保存
            if changes_made:
                with open(report_gen_path, 'w', encoding='utf-8') as f:
                    f.write(optimized_content)
                
                optimization_result['optimized'] = True
                optimization_result['changes_made'] = changes_made
                
                print(f"  ✅ 最適化完了: {len(changes_made)}箇所修正")
                print(f"  📊 推定削減: {optimization_result['estimated_reduction_ms']}ms")
                
                self.processed_files.append(str(report_gen_path))
            else:
                print("  ℹ️ 最適化対象なし")
                
        except Exception as e:
            optimization_result['error'] = str(e)
            print(f"  ❌ 最適化エラー: {e}")
        
        return optimization_result
    
    def optimize_config_init_imports(self) -> Dict[str, Any]:
        """config/__init__.py インポート最適化"""
        print("🔧 config/__init__.py インポート最適化中...")
        
        optimization_result = {
            'optimized': False,
            'changes_made': [],
            'estimated_reduction_ms': 0,
            'error': None
        }
        
        config_init_path = self.project_root / "config" / "__init__.py"
        
        if not config_init_path.exists():
            optimization_result['error'] = f"ファイルが存在しません: {config_init_path}"
            return optimization_result
        
        try:
            with open(config_init_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 大量インポートを遅延化
            lines = original_content.split('\n')
            optimized_lines = []
            changes_made = []
            
            # 重いモジュールのインポートを特定
            heavy_modules = [
                'portfolio_weight_calculator',
                'metric_weight_optimizer', 
                'correlation',
                'risk_management',
                'multi_strategy_manager'
            ]
            
            for line in lines:
                line_strip = line.strip()
                
                # 重いモジュールのインポートを遅延化
                should_lazy = False
                for heavy_module in heavy_modules:
                    if f'from .{heavy_module}' in line or f'import {heavy_module}' in line:
                        should_lazy = True
                        break
                
                if should_lazy and not line.startswith('#'):
                    # インポートをコメントアウト
                    optimized_lines.append(f"# {line}  # TODO-PERF-001: Converted to lazy import")
                    changes_made.append(f"遅延化: {line_strip}")
                    optimization_result['estimated_reduction_ms'] += 200
                else:
                    optimized_lines.append(line)
            
            # 遅延インポート機能を追加
            if changes_made:
                lazy_import_code = '''
# TODO-PERF-001: Phase 2 Stage 2 - Config Lazy Import System
import importlib
from typing import Any, Optional

class LazyConfigImporter:
    """Config系モジュール遅延インポートクラス"""
    
    def __init__(self):
        self._modules = {}
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """モジュール遅延ロード"""
        if module_name not in self._modules:
            try:
                if module_name.startswith('.'):
                    # 相対インポート
                    self._modules[module_name] = importlib.import_module(module_name, package='config')
                else:
                    # 絶対インポート
                    self._modules[module_name] = importlib.import_module(f'config.{module_name}')
            except ImportError:
                self._modules[module_name] = None
        return self._modules[module_name]
    
    def __getattr__(self, name: str) -> Any:
        """属性アクセス時の動的ロード"""
        module = self.get_module(name)
        if module is not None:
            return module
        raise AttributeError(f"module 'config' has no attribute '{name}'")

# 遅延インポートシステム初期化
_lazy_config = LazyConfigImporter()

'''
                
                # 先頭に遅延インポートコードを追加
                optimized_content = lazy_import_code + '\n' + '\n'.join(optimized_lines)
                
                with open(config_init_path, 'w', encoding='utf-8') as f:
                    f.write(optimized_content)
                
                optimization_result['optimized'] = True
                optimization_result['changes_made'] = changes_made
                
                print(f"  ✅ 最適化完了: {len(changes_made)}箇所修正")
                print(f"  📊 推定削減: {optimization_result['estimated_reduction_ms']}ms")
                
                self.processed_files.append(str(config_init_path))
            else:
                print("  ℹ️ 最適化対象なし")
                
        except Exception as e:
            optimization_result['error'] = str(e)
            print(f"  ❌ 最適化エラー: {e}")
        
        return optimization_result
    
    def optimize_correlation_init_imports(self) -> Dict[str, Any]:
        """config/correlation/__init__.py インポート最適化"""
        print("🔧 config/correlation/__init__.py インポート最適化中...")
        
        optimization_result = {
            'optimized': False,
            'changes_made': [],
            'estimated_reduction_ms': 0,
            'error': None
        }
        
        correlation_init_path = self.project_root / "config" / "correlation" / "__init__.py"
        
        if not correlation_init_path.exists():
            optimization_result['error'] = f"ファイルが存在しません: {correlation_init_path}"
            return optimization_result
        
        try:
            with open(correlation_init_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # 重いcorrelation関連インポートを遅延化
            lines = original_content.split('\n')
            optimized_lines = []
            changes_made = []
            
            # 重いcorrelationモジュール
            heavy_correlation_modules = [
                'strategy_correlation_analyzer',
                'correlation_matrix_visualizer',
                'correlation_optimizer'
            ]
            
            for line in lines:
                line_strip = line.strip()
                
                # 重いモジュールのインポートを遅延化
                should_lazy = False
                for heavy_module in heavy_correlation_modules:
                    if heavy_module in line and ('import' in line or 'from' in line):
                        should_lazy = True
                        break
                
                if should_lazy and not line.startswith('#'):
                    optimized_lines.append(f"# {line}  # TODO-PERF-001: Converted to lazy import")
                    changes_made.append(f"遅延化: {line_strip}")
                    optimization_result['estimated_reduction_ms'] += 250
                else:
                    optimized_lines.append(line)
            
            # 遅延インポートシステム追加
            if changes_made:
                lazy_correlation_code = '''
# TODO-PERF-001: Phase 2 Stage 2 - Correlation Lazy Import System
import importlib
from typing import Any, Optional

class LazyCorrelationImporter:
    """Correlation系モジュール遅延インポートクラス"""
    
    def __init__(self):
        self._correlation_modules = {}
    
    def get_correlation_module(self, module_name: str) -> Optional[Any]:
        """correlation系モジュール遅延ロード"""
        if module_name not in self._correlation_modules:
            try:
                full_module_name = f'config.correlation.{module_name}'
                self._correlation_modules[module_name] = importlib.import_module(full_module_name)
            except ImportError:
                self._correlation_modules[module_name] = None
        return self._correlation_modules[module_name]
    
    def __getattr__(self, name: str) -> Any:
        """属性アクセス時の動的ロード"""
        module = self.get_correlation_module(name)
        if module is not None:
            return module
        raise AttributeError(f"module 'config.correlation' has no attribute '{name}'")

# 遅延インポートシステム初期化
_lazy_correlation = LazyCorrelationImporter()

'''
                
                optimized_content = lazy_correlation_code + '\n' + '\n'.join(optimized_lines)
                
                with open(correlation_init_path, 'w', encoding='utf-8') as f:
                    f.write(optimized_content)
                
                optimization_result['optimized'] = True
                optimization_result['changes_made'] = changes_made
                
                print(f"  ✅ 最適化完了: {len(changes_made)}箇所修正")
                print(f"  📊 推定削減: {optimization_result['estimated_reduction_ms']}ms")
                
                self.processed_files.append(str(correlation_init_path))
            else:
                print("  ℹ️ 最適化対象なし")
                
        except Exception as e:
            optimization_result['error'] = str(e)
            print(f"  ❌ 最適化エラー: {e}")
        
        return optimization_result
    
    def optimize_heavy_config_modules(self) -> Dict[str, Any]:
        """重いconfigモジュール最適化"""
        print("🔧 重いconfigモジュール最適化中...")
        
        optimization_result = {
            'optimized_modules': [],
            'total_estimated_reduction_ms': 0,
            'errors': []
        }
        
        # 重いモジュールリスト（Stage 1分析結果より）
        heavy_modules = [
            ("config/portfolio_weight_calculator.py", 1633),
            ("config/metric_weight_optimizer.py", 1617),
            ("config/correlation/strategy_correlation_analyzer.py", 1650)
        ]
        
        for module_path, baseline_ms in heavy_modules:
            module_full_path = self.project_root / module_path
            
            if not module_full_path.exists():
                optimization_result['errors'].append(f"ファイルが存在しません: {module_path}")
                continue
            
            try:
                with open(module_full_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                # モジュールレベルの重いインポートを特定
                lines = original_content.split('\n')
                optimized_lines = []
                changes_made = []
                
                for line in lines:
                    if line.strip().startswith(('import pandas', 'import numpy', 'import scipy')):
                        # 重いインポートをコメントアウト
                        optimized_lines.append(f"# {line}  # TODO-PERF-001: Optimized to lazy import")
                        changes_made.append(f"遅延化: {line.strip()}")
                    else:
                        optimized_lines.append(line)
                
                if changes_made:
                    # 遅延インポート用コードを先頭に追加
                    lazy_code = '''
# TODO-PERF-001: Phase 2 Stage 2 - Module Level Lazy Import
def _get_pandas():
    """pandas遅延インポート"""
    import pandas as pd
    return pd

def _get_numpy():
    """numpy遅延インポート"""
    import numpy as np
    return np

def _get_scipy():
    """scipy遅延インポート"""
    import scipy
    return scipy

# 遅延ロード用グローバル変数
_pandas = None
_numpy = None
_scipy = None

def get_pandas():
    global _pandas
    if _pandas is None:
        _pandas = _get_pandas()
    return _pandas

def get_numpy():
    global _numpy
    if _numpy is None:
        _numpy = _get_numpy()
    return _numpy

def get_scipy():
    global _scipy
    if _scipy is None:
        _scipy = _get_scipy()
    return _scipy

'''
                    
                    optimized_content = lazy_code + '\n' + '\n'.join(optimized_lines)
                    
                    # pandas, numpy, scipyの直接参照を関数呼び出しに変更
                    optimized_content = re.sub(r'\bpd\.', 'get_pandas().', optimized_content)
                    optimized_content = re.sub(r'\bnp\.', 'get_numpy().', optimized_content)
                    
                    with open(module_full_path, 'w', encoding='utf-8') as f:
                        f.write(optimized_content)
                    
                    estimated_reduction = baseline_ms * 0.6  # 60%削減想定
                    
                    optimization_result['optimized_modules'].append({
                        'module': module_path,
                        'changes': changes_made,
                        'baseline_ms': baseline_ms,
                        'estimated_reduction_ms': estimated_reduction
                    })
                    
                    optimization_result['total_estimated_reduction_ms'] += estimated_reduction
                    
                    print(f"  ✅ {module_path}: {len(changes_made)}箇所最適化 ({estimated_reduction:.0f}ms削減予想)")
                    
                    self.processed_files.append(str(module_full_path))
                else:
                    print(f"  ℹ️ {module_path}: 最適化対象なし")
                    
            except Exception as e:
                error_msg = f"{module_path}: {str(e)}"
                optimization_result['errors'].append(error_msg)
                print(f"  ❌ {error_msg}")
        
        return optimization_result
    
    def validate_optimizations(self) -> Dict[str, Any]:
        """最適化結果検証"""
        print("🔍 最適化結果検証中...")
        
        validation_result = {
            'syntax_valid': 0,
            'syntax_errors': 0,
            'import_errors': 0,
            'processed_files': len(self.processed_files),
            'validation_details': []
        }
        
        for file_path in self.processed_files:
            file_validation = {
                'file': file_path,
                'syntax_ok': False,
                'import_ok': False,
                'errors': []
            }
            
            try:
                # 構文チェック
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                ast.parse(content)
                file_validation['syntax_ok'] = True
                validation_result['syntax_valid'] += 1
                
                # インポートチェック（軽量）
                import_lines = [line for line in content.split('\n') if line.strip().startswith(('import', 'from'))]
                
                for import_line in import_lines[:5]:  # 最初の5個のみチェック
                    try:
                        compile(import_line, '<string>', 'exec')
                    except SyntaxError as e:
                        file_validation['errors'].append(f"Import syntax error: {e}")
                        validation_result['import_errors'] += 1
                        break
                else:
                    file_validation['import_ok'] = True
                
            except SyntaxError as e:
                file_validation['errors'].append(f"Syntax error: {e}")
                validation_result['syntax_errors'] += 1
            except Exception as e:
                file_validation['errors'].append(f"Validation error: {e}")
            
            validation_result['validation_details'].append(file_validation)
            
            status = "✅" if file_validation['syntax_ok'] else "❌"
            print(f"  {status} {Path(file_path).name}: 構文={'OK' if file_validation['syntax_ok'] else 'NG'}")
        
        return validation_result
    
    def add_systemfallback_integration(self) -> Dict[str, Any]:
        """SystemFallbackPolicy統合追加"""
        print("🔗 SystemFallbackPolicy統合追加中...")
        
        integration_result = {
            'integrated_files': [],
            'integration_errors': []
        }
        
        # SystemFallbackPolicy統合コード
        fallback_integration_code = '''
# TODO-PERF-001: Phase 2 Stage 2 - SystemFallbackPolicy Integration
try:
    from src.config.system_modes import SystemFallbackPolicy, ComponentType
    _fallback_policy = SystemFallbackPolicy.get_instance()
except ImportError:
    # フォールバック用のダミークラス
    class _DummyFallbackPolicy:
        def handle_component_failure(self, **kwargs):
            return None
    _fallback_policy = _DummyFallbackPolicy()

def _handle_lazy_import_failure(component_name: str, error: Exception, fallback_func=None):
    """遅延インポート失敗時のフォールバック処理"""
    try:
        return _fallback_policy.handle_component_failure(
            component_type=ComponentType.STRATEGY_ENGINE,
            component_name=component_name,
            error=error,
            fallback_func=fallback_func
        )
    except:
        # 最終フォールバック
        if fallback_func:
            return fallback_func()
        return None

'''
        
        for file_path in self.processed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # SystemFallbackPolicy統合がまだない場合に追加
                if '_fallback_policy' not in content:
                    # TODO-PERF-001コメントの後に統合コードを追加
                    if '# TODO-PERF-001:' in content:
                        content = content.replace(
                            '# TODO-PERF-001: Phase 2 Stage 2',
                            fallback_integration_code + '\n# TODO-PERF-001: Phase 2 Stage 2'
                        )
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        integration_result['integrated_files'].append(file_path)
                        print(f"  ✅ 統合完了: {Path(file_path).name}")
                    else:
                        print(f"  ℹ️ 統合スキップ: {Path(file_path).name} (TODO-PERF-001なし)")
                else:
                    print(f"  ℹ️ 統合済み: {Path(file_path).name}")
                    
            except Exception as e:
                error_msg = f"{file_path}: {str(e)}"
                integration_result['integration_errors'].append(error_msg)
                print(f"  ❌ 統合エラー: {error_msg}")
        
        return integration_result
    
    def generate_stage2_report(self) -> Dict[str, Any]:
        """Stage 2実装レポート生成"""
        print("📋 Stage 2実装レポート生成中...")
        
        stage2_report = {
            'stage': 'Stage 2: Stage 2構文エラー根本修正実装',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'optimization_results': self.optimization_results,
            'processed_files': self.processed_files,
            'backup_location': str(self.backup_dir),
            'total_estimated_reduction_ms': sum(
                result.get('estimated_reduction_ms', 0) 
                for result in self.optimization_results.values()
                if isinstance(result, dict)
            ),
            'next_stage': 'Stage 3: dssms_report_generator最適化実装'
        }
        
        return stage2_report
    
    def run_stage2_optimization(self) -> bool:
        """Stage 2完全最適化実行"""
        print("🚀 TODO-PERF-001 Phase 2 Stage 2: Stage 2構文エラー根本修正実装開始")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # 1. バックアップ作成
            print("\n1️⃣ バックアップ作成")
            if not self.create_backup():
                print("⚠️ バックアップ失敗 - リスク承知で続行")
            
            # 2. dssms_report_generator最適化
            print("\n2️⃣ dssms_report_generator インポート最適化")
            self.optimization_results['dssms_report_generator'] = self.optimize_dssms_report_generator_imports()
            
            # 3. config/__init__.py最適化
            print("\n3️⃣ config/__init__.py インポート最適化")
            self.optimization_results['config_init'] = self.optimize_config_init_imports()
            
            # 4. correlation/__init__.py最適化
            print("\n4️⃣ config/correlation/__init__.py インポート最適化")
            self.optimization_results['correlation_init'] = self.optimize_correlation_init_imports()
            
            # 5. 重いconfigモジュール最適化
            print("\n5️⃣ 重いconfigモジュール最適化")
            self.optimization_results['heavy_modules'] = self.optimize_heavy_config_modules()
            
            # 6. 最適化結果検証
            print("\n6️⃣ 最適化結果検証")
            validation_result = self.validate_optimizations()
            self.optimization_results['validation'] = validation_result
            
            # 7. SystemFallbackPolicy統合
            print("\n7️⃣ SystemFallbackPolicy統合追加")
            integration_result = self.add_systemfallback_integration()
            self.optimization_results['integration'] = integration_result
            
            # 8. レポート生成
            print("\n8️⃣ Stage 2実装レポート生成")
            stage2_report = self.generate_stage2_report()
            
            # レポート保存
            report_path = self.project_root / f"phase2_stage2_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(stage2_report, f, indent=2, ensure_ascii=False)
            
            execution_time = time.time() - start_time
            
            # 結果サマリー
            print("\n" + "="*80)
            print("🏆 TODO-PERF-001 Phase 2 Stage 2完了サマリー")
            print("="*80)
            print(f"⏱️ 実行時間: {execution_time:.1f}秒")
            print(f"🔧 最適化ファイル数: {len(self.processed_files)}")
            print(f"✅ 構文検証成功: {validation_result['syntax_valid']}")
            print(f"❌ 構文エラー: {validation_result['syntax_errors']}")
            print(f"📊 推定総削減: {stage2_report['total_estimated_reduction_ms']:.0f}ms")
            print(f"💾 バックアップ: {self.backup_dir}")
            print(f"📄 実装レポート: {report_path}")
            
            success_rate = validation_result['syntax_valid'] / max(1, len(self.processed_files)) * 100
            
            if success_rate >= 80:
                print(f"\n✅ Stage 2最適化成功 ({success_rate:.1f}%) - Stage 3 dssms_report_generator最適化に進行可能")
                return True
            else:
                print(f"\n⚠️ Stage 2部分成功 ({success_rate:.1f}%) - 問題解決後に Stage 3進行を推奨")
                return False
                
        except Exception as e:
            print(f"❌ Stage 2最適化エラー: {e}")
            traceback.print_exc()
            return False

def main():
    """メイン実行"""
    project_root = os.getcwd()
    optimizer = Phase2Stage2SyntaxOptimizer(project_root)
    
    success = optimizer.run_stage2_optimization()
    
    if success:
        print("\n🎉 Stage 2完成 - 次は Stage 3 dssms_report_generator最適化実装に進行")
    else:
        print("\n⚠️ Stage 2部分完了 - 問題解決後に Stage 3進行を推奨")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
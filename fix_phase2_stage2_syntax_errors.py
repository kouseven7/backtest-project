#!/usr/bin/env python3
"""
TODO-PERF-001: Phase 2 Stage 2 Syntax Error Fix - 構文エラー緊急修正

Stage 2最適化で発生した構文エラーを修正
"""

import os
import sys
from pathlib import Path

def fix_config_init_syntax():
    """config/__init__.py 構文エラー修正"""
    print("🔧 config/__init__.py 構文エラー修正中...")
    
    config_init = Path("config/__init__.py")
    if not config_init.exists():
        print("  ❌ ファイルが存在しません")
        return False
    
    try:
        with open(config_init, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 構文エラーの原因となるコードを修正
        fixed_content = content.replace(
            'try:\n    from src.config.system_modes import SystemFallbackPolicy, ComponentType\n    _fallback_policy = SystemFallbackPolicy.get_instance()',
            'try:\n    from src.config.system_modes import SystemFallbackPolicy, ComponentType\n    _fallback_policy = SystemFallbackPolicy()'
        )
        
        # ComponentType参照エラー修正
        fixed_content = fixed_content.replace(
            'component_type=ComponentType.STRATEGY_ENGINE,',
            'component_type="STRATEGY_ENGINE",'
        )
        
        with open(config_init, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print("  ✅ config/__init__.py 修正完了")
        return True
        
    except Exception as e:
        print(f"  ❌ 修正エラー: {e}")
        return False

def fix_correlation_init_syntax():
    """config/correlation/__init__.py 構文エラー修正"""
    print("🔧 config/correlation/__init__.py 構文エラー修正中...")
    
    correlation_init = Path("config/correlation/__init__.py")
    if not correlation_init.exists():
        print("  ❌ ファイルが存在しません")
        return False
    
    try:
        with open(correlation_init, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 同様の構文エラー修正
        fixed_content = content.replace(
            'try:\n    from src.config.system_modes import SystemFallbackPolicy, ComponentType\n    _fallback_policy = SystemFallbackPolicy.get_instance()',
            'try:\n    from src.config.system_modes import SystemFallbackPolicy, ComponentType\n    _fallback_policy = SystemFallbackPolicy()'
        )
        
        fixed_content = fixed_content.replace(
            'component_type=ComponentType.STRATEGY_ENGINE,',
            'component_type="STRATEGY_ENGINE",'
        )
        
        with open(correlation_init, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print("  ✅ config/correlation/__init__.py 修正完了")
        return True
        
    except Exception as e:
        print(f"  ❌ 修正エラー: {e}")
        return False

def validate_syntax():
    """構文検証"""
    print("🔍 構文検証中...")
    
    import ast
    
    files_to_check = [
        "config/__init__.py",
        "config/correlation/__init__.py"
    ]
    
    all_valid = True
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                ast.parse(content)
                print(f"  ✅ {file_path}: 構文OK")
                
            except SyntaxError as e:
                print(f"  ❌ {file_path}: 構文エラー - {e}")
                all_valid = False
            except Exception as e:
                print(f"  ⚠️ {file_path}: 検証エラー - {e}")
                all_valid = False
        else:
            print(f"  ❌ {file_path}: ファイル未存在")
            all_valid = False
    
    return all_valid

def main():
    """メイン実行"""
    print("🚀 TODO-PERF-001 Phase 2 Stage 2 構文エラー緊急修正開始")
    print("="*60)
    
    success = True
    
    # 1. config/__init__.py修正
    if not fix_config_init_syntax():
        success = False
    
    # 2. config/correlation/__init__.py修正
    if not fix_correlation_init_syntax():
        success = False
    
    # 3. 構文検証
    print("\n🔍 修正結果検証")
    if validate_syntax():
        print("✅ 全ファイル構文OK")
    else:
        print("⚠️ 一部ファイルに構文問題あり")
        success = False
    
    if success:
        print("\n🎉 構文エラー修正完了 - Stage 3進行可能")
    else:
        print("\n⚠️ 構文エラー修正に問題あり")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
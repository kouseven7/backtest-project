"""
Module: Multi-Strategy Coordination Package Requirements
File: install_coordination_requirements.py
Description: 
  4-1-3「マルチ戦略同時実行の調整機能」
  パッケージ依存関係インストールスクリプト

Author: imega  
Created: 2025-07-20
Modified: 2025-07-20

Functions:
  - 必要パッケージ依存関係確認・インストール
  - Python環境互換性チェック
  - システムリソースライブラリ設定
"""

import os
import sys
import subprocess
import importlib.util
from typing import List, Tuple, Dict, Any
import json

# 必須パッケージ定義
COORDINATION_REQUIREMENTS = [
    # ネットワーク・依存関係管理
    'networkx>=3.0',
    
    # システムリソース監視
    'psutil>=5.9.0',
    
    # Web インターフェース
    'flask>=2.3.0',
    'flask-cors>=4.0.0',
    
    # データ処理・分析
    'pandas>=2.0.0',
    'numpy>=1.24.0',
    
    # 日時処理
    'python-dateutil>=2.8.0',
    
    # 設定・データ管理
    'pyyaml>=6.0',
    
    # 非同期処理（Python 3.8以降で最適化）
    'aiofiles>=23.0.0',
    'aiohttp>=3.8.0',
    
    # ロギング・監視
    'structlog>=23.0.0',
    'colorama>=0.4.0'  # Windows コンソール色付け
]

# オプショナルパッケージ（高度機能用）
OPTIONAL_REQUIREMENTS = [
    # 機械学習・統計分析（異常検知用）
    'scipy>=1.10.0',
    'scikit-learn>=1.3.0',
    
    # 可視化
    'plotly>=5.0.0',
    'dash>=2.14.0',  # インタラクティブダッシュボード
    
    # 高性能データ処理
    'numba>=0.57.0',
    
    # データベース連携
    'sqlalchemy>=2.0.0',
    'redis>=4.5.0'
]

def check_python_version() -> Tuple[bool, str]:
    """Python バージョン確認"""
    version_info = sys.version_info
    current_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    # 最小要件: Python 3.8
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 8):
        return False, f"Python 3.8+ required, found {current_version}"
    
    # 推奨: Python 3.9+
    if version_info.major == 3 and version_info.minor < 9:
        return True, f"Python {current_version} is supported but 3.9+ is recommended"
    
    return True, f"Python {current_version} is fully supported"

def check_package_installed(package_spec: str) -> Tuple[bool, str]:
    """パッケージインストール状況確認"""
    try:
        # パッケージ名とバージョン仕様を分離
        if '>=' in package_spec:
            package_name = package_spec.split('>=')[0]
            min_version = package_spec.split('>=')[1]
        else:
            package_name = package_spec
            min_version = None
        
        # インポート試行
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, f"{package_name} not installed"
        
        # バージョン確認（可能な場合）
        if min_version:
            try:
                module = importlib.import_module(package_name)
                if hasattr(module, '__version__'):
                    current_version = module.__version__
                    return True, f"{package_name} {current_version} installed"
                else:
                    return True, f"{package_name} installed (version unknown)"
            except ImportError:
                return False, f"{package_name} import failed"
        
        return True, f"{package_name} installed"
        
    except Exception as e:
        return False, f"Check failed for {package_spec}: {e}"

def install_packages(packages: List[str], description: str = "") -> Dict[str, Any]:
    """パッケージインストール実行"""
    print(f"\n📦 Installing {description}packages...")
    
    results = {
        'total': len(packages),
        'successful': 0,
        'failed': 0,
        'details': []
    }
    
    for package in packages:
        try:
            print(f"  Installing {package}...")
            
            # pip install 実行
            cmd = [sys.executable, '-m', 'pip', 'install', package]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=300  # 5分タイムアウト
            )
            
            if result.returncode == 0:
                print(f"    [OK] {package} installed successfully")
                results['successful'] += 1
                results['details'].append({
                    'package': package,
                    'status': 'success',
                    'message': 'Installed successfully'
                })
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                print(f"    [ERROR] {package} installation failed: {error_msg}")
                results['failed'] += 1
                results['details'].append({
                    'package': package,
                    'status': 'failed',
                    'message': error_msg
                })
                
        except subprocess.TimeoutExpired:
            print(f"    [ERROR] {package} installation timeout")
            results['failed'] += 1
            results['details'].append({
                'package': package,
                'status': 'failed',
                'message': 'Installation timeout'
            })
            
        except Exception as e:
            print(f"    [ERROR] {package} installation error: {e}")
            results['failed'] += 1
            results['details'].append({
                'package': package,
                'status': 'failed',
                'message': str(e)
            })
    
    return results

def check_system_capabilities() -> Dict[str, Any]:
    """システム機能確認"""
    capabilities = {
        'system_monitoring': False,
        'web_interface': False,
        'async_support': False,
        'network_analysis': False,
        'multiprocessing': True  # Python標準ライブラリ
    }
    
    # システム監視機能
    try:
        import psutil
        capabilities['system_monitoring'] = True
    except ImportError:
        pass
    
    # Web インターフェース
    try:
        import flask
        capabilities['web_interface'] = True
    except ImportError:
        pass
    
    # 非同期処理サポート
    try:
        import asyncio
        import aiofiles
        capabilities['async_support'] = True
    except ImportError:
        pass
    
    # ネットワーク分析
    try:
        import networkx
        capabilities['network_analysis'] = True
    except ImportError:
        pass
    
    return capabilities

def create_requirements_txt():
    """requirements.txt ファイル生成"""
    try:
        requirements_content = "\n# Multi-Strategy Coordination System Requirements\n"
        requirements_content += "# Generated automatically by install_coordination_requirements.py\n\n"
        
        requirements_content += "# Core Requirements\n"
        for package in COORDINATION_REQUIREMENTS:
            requirements_content += f"{package}\n"
        
        requirements_content += "\n# Optional Requirements (Advanced Features)\n"
        for package in OPTIONAL_REQUIREMENTS:
            requirements_content += f"# {package}\n"
        
        with open('coordination_requirements.txt', 'w', encoding='utf-8') as f:
            f.write(requirements_content)
        
        print("[OK] coordination_requirements.txt created")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create requirements.txt: {e}")
        return False

def main():
    """メイン実行"""
    print("[ROCKET] Multi-Strategy Coordination System - Package Installation")
    print("=" * 70)
    
    # Python バージョン確認
    print("🐍 Python Version Check")
    version_ok, version_msg = check_python_version()
    print(f"   {version_msg}")
    
    if not version_ok:
        print("[ERROR] Incompatible Python version. Please upgrade to Python 3.8+")
        sys.exit(1)
    
    print("[OK] Python version is compatible\n")
    
    # 既存パッケージ確認
    print("[LIST] Checking existing packages...")
    installed_packages = []
    missing_packages = []
    
    for package in COORDINATION_REQUIREMENTS:
        is_installed, msg = check_package_installed(package)
        if is_installed:
            print(f"  [OK] {msg}")
            installed_packages.append(package)
        else:
            print(f"  [ERROR] {msg}")
            missing_packages.append(package)
    
    print(f"\nPackage Summary: {len(installed_packages)} installed, {len(missing_packages)} missing")
    
    # 必要なパッケージをインストール
    if missing_packages:
        print(f"\n[TOOL] Installing {len(missing_packages)} missing packages...")
        
        install_results = install_packages(missing_packages, "required ")
        
        print(f"\nInstallation Results:")
        print(f"  Successful: {install_results['successful']}")
        print(f"  Failed: {install_results['failed']}")
        
        if install_results['failed'] > 0:
            print(f"\nFailed installations:")
            for detail in install_results['details']:
                if detail['status'] == 'failed':
                    print(f"  [ERROR] {detail['package']}: {detail['message']}")
        
    else:
        print("[OK] All required packages are already installed!")
    
    # オプションパッケージ確認
    print(f"\n[TARGET] Optional Packages Check:")
    optional_installed = 0
    for package in OPTIONAL_REQUIREMENTS:
        is_installed, msg = check_package_installed(package)
        if is_installed:
            print(f"  [OK] {msg}")
            optional_installed += 1
        else:
            print(f"  ➖ {msg}")
    
    print(f"Optional packages: {optional_installed}/{len(OPTIONAL_REQUIREMENTS)} installed")
    
    # システム機能確認
    print(f"\n[SEARCH] System Capabilities Check:")
    capabilities = check_system_capabilities()
    
    capability_descriptions = {
        'system_monitoring': 'System Resource Monitoring (psutil)',
        'web_interface': 'Web Dashboard Interface (Flask)',
        'async_support': 'Asynchronous Processing Support', 
        'network_analysis': 'Dependency Graph Analysis (NetworkX)',
        'multiprocessing': 'Parallel Execution Support'
    }
    
    enabled_capabilities = 0
    for cap, enabled in capabilities.items():
        desc = capability_descriptions.get(cap, cap)
        if enabled:
            print(f"  [OK] {desc}")
            enabled_capabilities += 1
        else:
            print(f"  [ERROR] {desc}")
    
    print(f"System capabilities: {enabled_capabilities}/{len(capabilities)} available")
    
    # requirements.txt 生成
    print(f"\n📄 Generating requirements file...")
    create_requirements_txt()
    
    # 最終サマリー
    print(f"\n" + "=" * 70)
    print("[CHART] Installation Summary")
    print("=" * 70)
    
    all_required_installed = len(missing_packages) == 0
    core_capabilities = all(['system_monitoring', 'web_interface', 'network_analysis']) <= capabilities.keys()
    
    if all_required_installed and enabled_capabilities >= 4:
        print("[SUCCESS] INSTALLATION SUCCESSFUL!")
        print("   All required packages are installed and system capabilities are available.")
        print("   The Multi-Strategy Coordination System is ready to use.")
        
        print(f"\n[TARGET] Next Steps:")
        print(f"   1. Run: python demo_multi_strategy_coordination.py")
        print(f"   2. Access web dashboard at: http://localhost:5000")
        print(f"   3. Review system logs and configuration files")
        
    elif all_required_installed:
        print("[WARNING] PARTIAL SUCCESS")
        print("   Required packages installed, but some capabilities may be limited.")
        print("   Consider installing optional packages for full functionality.")
        
    else:
        print("[ERROR] INSTALLATION ISSUES")
        print("   Some required packages could not be installed.")
        print("   Please resolve the installation errors before proceeding.")
        
        if missing_packages:
            print(f"\n[TOOL] Manual installation commands:")
            print(f"   pip install {' '.join(missing_packages)}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()

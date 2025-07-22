#!/usr/bin/env python3
"""
Quick test script for 4-1-3 Multi-Strategy Coordination System
"""
import sys
import json

def test_basic_imports():
    print("🔧 Testing basic imports...")
    try:
        # Standard library imports
        import json, sys, time, logging, threading
        from datetime import datetime, timedelta
        print("✅ Standard library imports: OK")
        
        # Data processing
        try:
            import pandas, numpy
            print("✅ pandas, numpy: OK")
        except ImportError:
            print("❌ pandas, numpy: Not available")
        
        # System monitoring
        try:
            import psutil
            print("✅ psutil: OK")
        except ImportError:
            print("❌ psutil: Not available")
        
        # Optional packages
        missing_packages = []
        optional_packages = ['networkx', 'flask', 'aiofiles']
        
        for package in optional_packages:
            try:
                __import__(package)
                print(f"✅ {package}: OK")
            except ImportError:
                print(f"❌ {package}: Not available")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
            print("Run: pip install networkx flask aiofiles flask-cors pyyaml structlog")
        
        return len(missing_packages) == 0
        
    except Exception as e:
        print(f"❌ Basic import test failed: {e}")
        return False

def test_coordination_files():
    print("\n🗂️ Testing coordination files...")
    required_files = [
        'coordination_config.json',
        'resource_allocation_engine.py',
        'strategy_dependency_resolver.py', 
        'concurrent_execution_scheduler.py',
        'execution_monitoring_system.py',
        'multi_strategy_coordination_manager.py',
        'multi_strategy_coordination_interface.py'
    ]
    
    missing_files = []
    for file in required_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 100:  # Basic size check
                    print(f"✅ {file}: OK ({len(content)} chars)")
                else:
                    print(f"❌ {file}: Too small or empty")
                    missing_files.append(file)
        except FileNotFoundError:
            print(f"❌ {file}: Not found")
            missing_files.append(file)
        except Exception as e:
            print(f"❌ {file}: Error - {e}")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_python_syntax():
    print("\n🐍 Testing Python syntax...")
    python_files = [
        'resource_allocation_engine.py',
        'strategy_dependency_resolver.py',
        'concurrent_execution_scheduler.py',
        'execution_monitoring_system.py',
        'multi_strategy_coordination_manager.py',
        'multi_strategy_coordination_interface.py'
    ]
    
    syntax_errors = []
    for file in python_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, file, 'exec')
            print(f"✅ {file}: Syntax OK")
        except SyntaxError as e:
            print(f"❌ {file}: Syntax error - {e}")
            syntax_errors.append(file)
        except FileNotFoundError:
            print(f"❌ {file}: Not found")
            syntax_errors.append(file)
        except Exception as e:
            print(f"❌ {file}: Error - {e}")
            syntax_errors.append(file)
    
    return len(syntax_errors) == 0

def main():
    print("🚀 4-1-3 Multi-Strategy Coordination System - Quick Test")
    print("=" * 60)
    
    # Test sequence
    imports_ok = test_basic_imports()
    files_ok = test_coordination_files()
    syntax_ok = test_python_syntax()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("-" * 30)
    print(f"Basic Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"File Presence: {'✅ PASS' if files_ok else '❌ FAIL'}")  
    print(f"Python Syntax: {'✅ PASS' if syntax_ok else '❌ FAIL'}")
    
    overall_success = imports_ok and files_ok and syntax_ok
    
    if overall_success:
        print("\n🎉 QUICK TEST PASSED!")
        print("System appears to be properly implemented.")
        print("Next: Run full integration test")
    else:
        print("\n⚠️ ISSUES DETECTED")
        print("Please resolve the errors above before proceeding.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

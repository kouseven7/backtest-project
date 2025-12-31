#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe Syntax Check - PowerShell Unicode Crash Avoidance
"""

import py_compile
import sys

try:
    py_compile.compile('src/dssms/dssms_integrated_main.py', doraise=True)
    print("Syntax check passed: No errors found")
except py_compile.PyCompileError as e:
    print(f"Syntax error found: {e}")
    # Extract line number from error message
    error_str = str(e)
    if 'line' in error_str:
        print(f"Error details: {error_str}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
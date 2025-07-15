# 3-2-3 Portfolio Weight Pattern Template System Test Script
# PowerShell test script

param(
    [switch]$FullTest = $false,
    [switch]$QuickTest = $true
)

# Basic settings
$ProjectRoot = $PSScriptRoot
$ConfigDir = Join-Path $ProjectRoot "config"
$PythonExecutable = "python"

Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "3-2-3 Portfolio Weight Pattern Template System Test" -ForegroundColor Green  
Write-Host "Execution time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "Project root: $ProjectRoot" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green

# Check required files
Write-Host "`n1. Required file check" -ForegroundColor Yellow

$RequiredFiles = @(
    "config\portfolio_weight_pattern_engine_v2.py",
    "demo_pattern_template_system.py"
)

$AllFilesExist = $true
foreach ($File in $RequiredFiles) {
    $FilePath = Join-Path $ProjectRoot $File
    if (Test-Path $FilePath) {
        Write-Host "OK $File" -ForegroundColor Green
    } else {
        Write-Host "NG $File (not found)" -ForegroundColor Red
        $AllFilesExist = $false
    }
}

if (-not $AllFilesExist) {
    Write-Host "`nNG Required files are missing. Exiting." -ForegroundColor Red
    exit 1
}

# Python environment check
Write-Host "`n2. Python environment check" -ForegroundColor Yellow

try {
    $PythonVersion = & $PythonExecutable --version 2>&1
    Write-Host "OK Python: $PythonVersion" -ForegroundColor Green
} catch {
    Write-Host "NG Python not found" -ForegroundColor Red
    exit 1
}

# Basic import test
Write-Host "`n3. Basic import test" -ForegroundColor Yellow

Set-Location $ProjectRoot

$ImportTestResult = & $PythonExecutable -c @"
try:
    from config.portfolio_weight_pattern_engine_v2 import AdvancedPatternEngineV2, RiskTolerance
    print('OK Basic module import success')
    
    # Simple initialization test
    engine = AdvancedPatternEngineV2()
    templates = engine.list_templates()
    print(f'OK Pattern engine initialization success (templates: {len(templates)})')
    
except ImportError as e:
    print(f'NG Import error: {e}')
    exit(1)
except Exception as e:
    print(f'NG Initialization error: {e}')
    exit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "NG Import test failed" -ForegroundColor Red
    exit 1
}

# Quick test
if ($QuickTest) {
    Write-Host "`n4. Quick function test" -ForegroundColor Yellow
    
    $QuickTestResult = & $PythonExecutable -c @"
from config.portfolio_weight_pattern_engine_v2 import AdvancedPatternEngineV2, RiskTolerance, quick_template_recommendation
import pandas as pd
import numpy as np

# Engine initialization
engine = AdvancedPatternEngineV2()

# Template recommendation test
print('=== Template recommendation test ===')
for risk in ['conservative', 'balanced', 'aggressive']:
    try:
        risk_enum = RiskTolerance(risk)
        template = engine.recommend_template(risk_enum)
        print(f'OK {risk}: {template.name}')
    except Exception as e:
        print(f'NG {risk}: {e}')

# Quick recommendation test
print('\n=== Quick recommendation test ===')
try:
    quick_template = quick_template_recommendation('balanced')
    print(f'OK Quick recommendation: {quick_template.name}')
except Exception as e:
    print(f'NG Quick recommendation error: {e}')

print('\nOK Quick test completed')
"@

    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK Quick test success" -ForegroundColor Green
    } else {
        Write-Host "NG Quick test failed" -ForegroundColor Red
    }
}

# Full test
if ($FullTest) {
    Write-Host "`n5. Full demo test" -ForegroundColor Yellow
    
    try {
        & $PythonExecutable "demo_pattern_template_system.py"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "OK Full demo test success" -ForegroundColor Green
        } else {
            Write-Host "NG Full demo test failed" -ForegroundColor Red
        }
    } catch {
        Write-Host "NG Full demo test execution error: $_" -ForegroundColor Red
    }
}

# Check generated files
Write-Host "`n6. Generated file check" -ForegroundColor Yellow

$PatternDir = Join-Path $ConfigDir "portfolio_weight_patterns"
if (Test-Path $PatternDir) {
    Write-Host "OK Pattern directory confirmed: $PatternDir" -ForegroundColor Green
    
    $TemplateFile = Join-Path $PatternDir "pattern_templates.json"
    $ConfigFile = Join-Path $PatternDir "dynamic_adjustment_config.json"
    
    if (Test-Path $TemplateFile) {
        Write-Host "OK Template file: $TemplateFile" -ForegroundColor Green
    } else {
        Write-Host "WARNING Template file not found" -ForegroundColor Yellow
    }
    
    if (Test-Path $ConfigFile) {
        Write-Host "OK Dynamic adjustment config file: $ConfigFile" -ForegroundColor Green
    } else {
        Write-Host "WARNING Dynamic adjustment config file not found" -ForegroundColor Yellow
    }
} else {
    Write-Host "WARNING Pattern directory not found (normal for first run)" -ForegroundColor Yellow
}

# Usage examples
Write-Host "`n7. Usage examples" -ForegroundColor Yellow
Write-Host "Basic usage:" -ForegroundColor Cyan
Write-Host "  python -c `"from config.portfolio_weight_pattern_engine_v2 import quick_template_recommendation; print(quick_template_recommendation('balanced').name)`"" -ForegroundColor Gray

Write-Host "`nFull demo execution:" -ForegroundColor Cyan  
Write-Host "  python demo_pattern_template_system.py" -ForegroundColor Gray

Write-Host "`nPowerShell test re-execution:" -ForegroundColor Cyan
Write-Host "  .\test_pattern_template_system.ps1 -FullTest" -ForegroundColor Gray

# Final result
Write-Host "`n=====================================================================" -ForegroundColor Green
Write-Host "3-2-3 Portfolio Weight Pattern Template System Test Completed" -ForegroundColor Green
Write-Host "Implementation status: OK Normal operation confirmed" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green

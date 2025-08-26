# DSSMS Task 1.4 PowerShell Execution Script
# Bank Switch Mechanism Recovery Comprehensive Execution Environment

param(
    [Parameter(HelpMessage="Execution mode: demo, test, comprehensive, quick")]
    [ValidateSet("demo", "test", "comprehensive", "quick")]
    [string]$Mode = "demo",
    
    [Parameter(HelpMessage="Verbose log output")]
    [switch]$VerboseOutput,
    
    [Parameter(HelpMessage="Auto open result files")]
    [switch]$OpenResults,
    
    [Parameter(HelpMessage="Clean environment before execution")]
    [switch]$CleanStart
)

# Script information
$ScriptName = "DSSMS Task 1.4 - Bank Switch Mechanism Recovery"
$ScriptVersion = "1.0.0"

# Execution start time
$StartTime = Get-Date

Write-Host "🚀 $ScriptName" -ForegroundColor Cyan
Write-Host "Version: $ScriptVersion" -ForegroundColor Green
Write-Host "Start time: $($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Yellow
Write-Host "Execution mode: $Mode" -ForegroundColor Magenta
Write-Host "=" * 60 -ForegroundColor White

# Project root confirmation
$ProjectRoot = Get-Location
Write-Host "📁 Project root: $ProjectRoot" -ForegroundColor Blue

# Output directory confirmation/creation
$OutputDir = Join-Path $ProjectRoot "output\task_14_demo"
if (-not (Test-Path $OutputDir)) {
    New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null
    Write-Host "📂 Output directory created: $OutputDir" -ForegroundColor Green
}

# Clean start processing
if ($CleanStart) {
    Write-Host "🧹 Environment cleanup executing..." -ForegroundColor Yellow
    
    # Delete old log files
    $LogFiles = Get-ChildItem -Path $ProjectRoot -Filter "*.log" -Recurse | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-1) }
    if ($LogFiles) {
        $LogFiles | Remove-Item -Force
        Write-Host "   - Old log files $($LogFiles.Count) deleted" -ForegroundColor Yellow
    }
    
    # Delete old result files
    $OldResults = Get-ChildItem -Path $OutputDir -Filter "*demo*" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-1) }
    if ($OldResults) {
        $OldResults | Remove-Item -Force
        Write-Host "   - Old result files $($OldResults.Count) deleted" -ForegroundColor Yellow
    }
    
    Write-Host "✅ Cleanup completed" -ForegroundColor Green
}

# Python environment confirmation
Write-Host "🐍 Python environment checking..." -ForegroundColor Blue
try {
    $PythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python: $PythonVersion" -ForegroundColor Green
    } else {
        Write-Host "❌ Python execution failed" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Python environment not found" -ForegroundColor Red
    exit 1
}

# Execution mode processing
switch ($Mode) {
    "demo" {
        Write-Host "🎬 Demonstration execution" -ForegroundColor Cyan
        Write-Host "-" * 40 -ForegroundColor White
        
        # Demo script execution
        Write-Host "🚀 demo_dssms_task_1_4_en.py executing..." -ForegroundColor Blue
        if ($VerboseOutput) {
            python demo_dssms_task_1_4_en.py
        } else {
            python demo_dssms_task_1_4_en.py 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir "demo_execution.log")
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Demo execution completed" -ForegroundColor Green
        } else {
            Write-Host "❌ Demo execution failed" -ForegroundColor Red
        }
    }
    
    "test" {
        Write-Host "🧪 Test suite execution" -ForegroundColor Cyan
        Write-Host "-" * 40 -ForegroundColor White
        
        # Test file execution
        Write-Host "🚀 test_dssms_task_1_4_comprehensive.py executing..." -ForegroundColor Blue
        if ($VerboseOutput) {
            python test_dssms_task_1_4_comprehensive.py
        } else {
            python test_dssms_task_1_4_comprehensive.py 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir "test_execution.log")
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Test execution completed" -ForegroundColor Green
        } else {
            Write-Host "❌ Test execution failed" -ForegroundColor Red
        }
    }
    
    "comprehensive" {
        Write-Host "🔬 Comprehensive execution (Demo + Test)" -ForegroundColor Cyan
        Write-Host "-" * 40 -ForegroundColor White
        
        # 1. Demo execution
        Write-Host "1️⃣ Demonstration execution..." -ForegroundColor Blue
        python demo_dssms_task_1_4_en.py 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir "comprehensive_demo.log")
        $DemoResult = $LASTEXITCODE
        
        if ($DemoResult -eq 0) {
            Write-Host "✅ Demo completed" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Demo encountered issues" -ForegroundColor Yellow
        }
        
        Write-Host ""
        
        # 2. Test execution
        Write-Host "2️⃣ Test suite execution..." -ForegroundColor Blue
        python test_dssms_task_1_4_comprehensive.py 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir "comprehensive_test.log")
        $TestResult = $LASTEXITCODE
        
        if ($TestResult -eq 0) {
            Write-Host "✅ Test completed" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Test encountered issues" -ForegroundColor Yellow
        }
        
        # Overall result
        Write-Host ""
        Write-Host "📊 Comprehensive execution result:" -ForegroundColor Cyan
        Write-Host "   - Demo execution: $(if ($DemoResult -eq 0) { '✅ Success' } else { '❌ Failed' })" -ForegroundColor White
        Write-Host "   - Test execution: $(if ($TestResult -eq 0) { '✅ Success' } else { '❌ Failed' })" -ForegroundColor White
        
        if ($DemoResult -eq 0 -and $TestResult -eq 0) {
            Write-Host "🎉 Task 1.4: Complete success" -ForegroundColor Green
        } elseif ($DemoResult -eq 0 -or $TestResult -eq 0) {
            Write-Host "⚠️ Task 1.4: Partial success" -ForegroundColor Yellow
        } else {
            Write-Host "❌ Task 1.4: Needs correction" -ForegroundColor Red
        }
    }
    
    "quick" {
        Write-Host "⚡ Quick validation execution" -ForegroundColor Cyan
        Write-Host "-" * 40 -ForegroundColor White
        
        # Minimal validation
        Write-Host "🔍 Basic component validation..." -ForegroundColor Blue
        
        $ValidationScript = @"
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent))

print("🔍 DSSMS Task 1.4 Quick validation")
print("-" * 40)

# Component import validation
components = {
    "MockDSSMSSwitchCoordinatorV2": "src.dssms.mock_switch_coordinator_v2",
    "SwitchDiagnostics": "src.dssms.switch_diagnostics", 
    "MockDSSMSBacktesterV2Updated": "src.dssms.mock_backtester_v2_updated"
}

success_count = 0
for name, module in components.items():
    try:
        __import__(module)
        print(f"✅ {name}: Import success")
        success_count += 1
    except ImportError as e:
        print(f"❌ {name}: Import failed - {e}")
    except Exception as e:
        print(f"⚠️ {name}: Error - {e}")

print(f"\n📊 Validation result: {success_count}/{len(components)} components available")

if success_count >= 2:
    print("🎉 Task 1.4: Basic functions available")
    sys.exit(0)
elif success_count >= 1:
    print("⚠️ Task 1.4: Partial functions available")
    sys.exit(1)
else:
    print("❌ Task 1.4: Critical issues found")
    sys.exit(2)
"@
        
        $TempScript = Join-Path $env:TEMP "task_14_quick_validation.py"
        $ValidationScript | Out-File -FilePath $TempScript -Encoding UTF8
        
        python $TempScript
        $ValidationResult = $LASTEXITCODE
        
        Remove-Item $TempScript -Force
        
        if ($ValidationResult -eq 0) {
            Write-Host "✅ Quick validation completed: Functions normal" -ForegroundColor Green
        } elseif ($ValidationResult -eq 1) {
            Write-Host "⚠️ Quick validation completed: Partial issues" -ForegroundColor Yellow
        } else {
            Write-Host "❌ Quick validation completed: Critical issues" -ForegroundColor Red
        }
    }
}

# Execution time calculation
$EndTime = Get-Date
$Duration = $EndTime - $StartTime

Write-Host ""
Write-Host "⏱️ Execution time: $($Duration.ToString('mm\:ss\.fff'))" -ForegroundColor Blue
Write-Host "🕐 End time: $($EndTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Yellow

# Result file list
Write-Host ""
Write-Host "📁 Generated files:" -ForegroundColor Cyan
$ResultFiles = Get-ChildItem -Path $OutputDir -File | Sort-Object LastWriteTime -Descending | Select-Object -First 10

if ($ResultFiles) {
    foreach ($file in $ResultFiles) {
        $SizeKB = [math]::Round($file.Length / 1KB, 1)
        Write-Host "   - $($file.Name) ($SizeKB KB)" -ForegroundColor White
    }
} else {
    Write-Host "   - No files found" -ForegroundColor Yellow
}

# Auto open result files
if ($OpenResults -and $ResultFiles) {
    Write-Host ""
    Write-Host "📂 Opening result files..." -ForegroundColor Blue
    
    # Open latest report file
    $LatestReport = $ResultFiles | Where-Object { $_.Name -like "*report*" -or $_.Name -like "*summary*" } | Select-Object -First 1
    if ($LatestReport) {
        Start-Process "notepad.exe" -ArgumentList $LatestReport.FullName
        Write-Host "✅ Report file opened: $($LatestReport.Name)" -ForegroundColor Green
    }
    
    # Open output directory
    Start-Process "explorer.exe" -ArgumentList $OutputDir
    Write-Host "✅ Output directory opened" -ForegroundColor Green
}

Write-Host ""
Write-Host "🔚 DSSMS Task 1.4 execution completed" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor White

# Simple usage examples display
if ($Mode -eq "demo") {
    Write-Host ""
    Write-Host "💡 Other execution modes:" -ForegroundColor Green
    Write-Host "   .\run_dssms_task_1_4.ps1 -Mode test          # Test only execution" -ForegroundColor White
    Write-Host "   .\run_dssms_task_1_4.ps1 -Mode comprehensive # Demo+Test execution" -ForegroundColor White
    Write-Host "   .\run_dssms_task_1_4.ps1 -Mode quick         # Quick validation" -ForegroundColor White
    Write-Host "   .\run_dssms_task_1_4.ps1 -OpenResults        # Auto open result files" -ForegroundColor White
    Write-Host "   .\run_dssms_task_1_4.ps1 -CleanStart         # Execute after environment cleanup" -ForegroundColor White
}

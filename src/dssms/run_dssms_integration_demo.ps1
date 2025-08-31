# DSSMS Phase 2 Task 2.1: 統合システムデモ実行スクリプト
# PowerShell script for DSSMS Integration Demo execution

param(
    [string]$Mode = "basic",
    [string]$Symbols = "",
    [string]$TestLevel = "all",
    [switch]$Help,
    [switch]$Verbose,
    [switch]$SkipEnvCheck
)

# ヘルプ表示
if ($Help) {
    Write-Host @"
DSSMS Integration Demo Execution Script

Usage:
    .\run_dssms_integration_demo.ps1 [OPTIONS]

Options:
    -Mode <mode>        Demo mode: basic, comprehensive, test, benchmark, all (default: basic)
    -Symbols <symbols>  Comma-separated symbol list for comprehensive mode (e.g., "7203,6758,8306")
    -TestLevel <level>  Test level: unit, integration, stress, performance, all (default: all)
    -Verbose           Enable verbose output
    -SkipEnvCheck      Skip environment validation
    -Help              Show this help message

Examples:
    .\run_dssms_integration_demo.ps1 -Mode basic
    .\run_dssms_integration_demo.ps1 -Mode comprehensive -Symbols "7203,6758,8306"
    .\run_dssms_integration_demo.ps1 -Mode test -TestLevel integration
    .\run_dssms_integration_demo.ps1 -Mode all -Verbose

"@
    exit 0
}

# スクリプト設定
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

# ロギング関数
function Write-Log {
    param($Message, $Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Level] $Timestamp $Message"
    
    switch ($Level) {
        "ERROR" { Write-Host $LogMessage -ForegroundColor Red }
        "WARN"  { Write-Host $LogMessage -ForegroundColor Yellow }
        "INFO"  { Write-Host $LogMessage -ForegroundColor Green }
        "DEBUG" { if ($Verbose) { Write-Host $LogMessage -ForegroundColor Gray } }
        default { Write-Host $LogMessage }
    }
}

# 環境確認関数
function Test-Environment {
    Write-Log "Checking environment prerequisites..." "INFO"
    
    # Python確認
    try {
        $PythonVersion = python --version 2>&1
        Write-Log "Python version: $PythonVersion" "DEBUG"
    } catch {
        Write-Log "Python not found. Please install Python 3.8 or later." "ERROR"
        return $false
    }
    
    # プロジェクトディレクトリ確認
    if (-not (Test-Path $ProjectRoot)) {
        Write-Log "Project root directory not found: $ProjectRoot" "ERROR"
        return $false
    }
    
    # 必要ファイル確認
    $RequiredFiles = @(
        "src\dssms\dssms_integration_demo.py",
        "src\dssms\dssms_strategy_integration_manager.py",
        "src\dssms\dssms_integration_config.json"
    )
    
    foreach ($File in $RequiredFiles) {
        $FilePath = Join-Path $ProjectRoot $File
        if (-not (Test-Path $FilePath)) {
            Write-Log "Required file not found: $File" "ERROR"
            return $false
        }
    }
    
    Write-Log "Environment check passed" "INFO"
    return $true
}

# ディレクトリ準備関数
function Initialize-Directories {
    Write-Log "Initializing directories..." "DEBUG"
    
    $Directories = @(
        "logs",
        "output",
        "output\integration_demos",
        "output\integration_results"
    )
    
    foreach ($Dir in $Directories) {
        $DirPath = Join-Path $ProjectRoot $Dir
        if (-not (Test-Path $DirPath)) {
            New-Item -ItemType Directory -Path $DirPath -Force | Out-Null
            Write-Log "Created directory: $Dir" "DEBUG"
        }
    }
}

# Pythonパス設定関数
function Set-PythonPath {
    $env:PYTHONPATH = $ProjectRoot
    Write-Log "PYTHONPATH set to: $ProjectRoot" "DEBUG"
}

# デモ実行関数
function Start-Demo {
    param($DemoMode, $DemoSymbols, $DemoTestLevel)
    
    Write-Log "Starting DSSMS Integration Demo..." "INFO"
    Write-Log "Mode: $DemoMode" "INFO"
    
    # 作業ディレクトリ変更
    Push-Location $ProjectRoot
    
    try {
        # Pythonコマンド構築
        $PythonScript = "src\dssms\dssms_integration_demo.py"
        $PythonArgs = @("--mode", $DemoMode)
        
        if ($DemoSymbols -and $DemoMode -eq "comprehensive") {
            $PythonArgs += @("--symbols", $DemoSymbols)
        }
        
        if ($DemoTestLevel -and $DemoMode -eq "test") {
            $PythonArgs += @("--test-level", $DemoTestLevel)
        }
        
        # 実行ログ
        $CommandLine = "python $PythonScript " + ($PythonArgs -join " ")
        Write-Log "Executing: $CommandLine" "DEBUG"
        
        # デモ実行
        $StartTime = Get-Date
        & python $PythonScript @PythonArgs
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Demo completed successfully in $($Duration.TotalSeconds.ToString("F2")) seconds" "INFO"
        } else {
            Write-Log "Demo completed with errors (exit code: $LASTEXITCODE)" "WARN"
        }
        
    } catch {
        Write-Log "Demo execution failed: $_" "ERROR"
        throw
    } finally {
        Pop-Location
    }
}

# メイン実行部分
function Main {
    try {
        Write-Host @"
================================================================================
DSSMS PHASE 2 TASK 2.1: INTEGRATION SYSTEM DEMO EXECUTION
================================================================================
"@ -ForegroundColor Cyan
        
        Write-Log "DSSMS Integration Demo execution started" "INFO"
        Write-Log "Script directory: $ScriptDir" "DEBUG"
        Write-Log "Project root: $ProjectRoot" "DEBUG"
        
        # 環境確認
        if (-not $SkipEnvCheck) {
            if (-not (Test-Environment)) {
                Write-Log "Environment check failed. Use -SkipEnvCheck to bypass." "ERROR"
                exit 1
            }
        } else {
            Write-Log "Environment check skipped" "WARN"
        }
        
        # ディレクトリ初期化
        Initialize-Directories
        
        # Pythonパス設定
        Set-PythonPath
        
        # パラメータ検証
        $ValidModes = @("basic", "comprehensive", "test", "benchmark", "all")
        if ($Mode -notin $ValidModes) {
            Write-Log "Invalid mode: $Mode. Valid modes: $($ValidModes -join ', ')" "ERROR"
            exit 1
        }
        
        $ValidTestLevels = @("unit", "integration", "stress", "performance", "all")
        if ($TestLevel -notin $ValidTestLevels) {
            Write-Log "Invalid test level: $TestLevel. Valid levels: $($ValidTestLevels -join ', ')" "ERROR"
            exit 1
        }
        
        # デモ実行
        Start-Demo -DemoMode $Mode -DemoSymbols $Symbols -DemoTestLevel $TestLevel
        
        Write-Log "DSSMS Integration Demo execution completed" "INFO"
        
        # 結果ファイル確認
        $OutputDir = Join-Path $ProjectRoot "output\integration_demos"
        if (Test-Path $OutputDir) {
            $OutputFiles = Get-ChildItem $OutputDir -Filter "*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 3
            if ($OutputFiles) {
                Write-Log "Recent output files:" "INFO"
                foreach ($File in $OutputFiles) {
                    Write-Log "  $($File.Name) ($($File.LastWriteTime))" "INFO"
                }
            }
        }
        
        Write-Host @"

================================================================================
DEMO EXECUTION COMPLETED
================================================================================
Check the output directory for detailed results:
$OutputDir

Logs are available in:
$(Join-Path $ProjectRoot "logs")
================================================================================
"@ -ForegroundColor Green
        
    } catch {
        Write-Log "Execution failed: $_" "ERROR"
        Write-Log "Stack trace: $($_.ScriptStackTrace)" "DEBUG"
        exit 1
    }
}

# スクリプト実行開始
Main

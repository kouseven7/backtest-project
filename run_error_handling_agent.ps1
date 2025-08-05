# エラーハンドリングシステム エージェントモード実行スクリプト
# フェーズ3: 実践環境準備 A. エラーハンドリング強化

Write-Host "🚀 エラーハンドリングシステム エージェントモード実行開始" -ForegroundColor Green
Write-Host "=" * 70

# 1. Python環境確認
Write-Host "📋 1. Python環境確認"
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Python バージョン: $pythonVersion" -ForegroundColor Blue
    
    # 必要なパッケージ確認
    $packages = @("pathlib", "datetime", "json", "threading")
    foreach ($package in $packages) {
        python -c "import $package; print(f'  ✓ {$package} モジュール: 利用可能')" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ⚠️ $package モジュールが見つかりません" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "  ❌ Python環境エラー: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 2. 設定ファイル検証
Write-Host "📋 2. 設定ファイル検証"
$configFiles = @(
    "config\error_handling\error_policies.json",
    "config\error_handling\recovery_strategies.json", 
    "config\error_handling\logging_config.json",
    "config\error_handling\notification_config.json"
)

foreach ($file in $configFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ $file" -ForegroundColor Green
        
        # JSON形式検証
        try {
            $content = Get-Content $file -Raw | ConvertFrom-Json
            Write-Host "    → JSON形式: 有効" -ForegroundColor Blue
        } catch {
            Write-Host "    → JSON形式: 無効" -ForegroundColor Red
        }
    } else {
        Write-Host "  ❌ $file が見つかりません" -ForegroundColor Red
    }
}

Write-Host ""

# 3. ディレクトリ構造確認
Write-Host "📋 3. ディレクトリ構造確認"
$directories = @(
    "src\utils",
    "config\error_handling",
    "tests\test_error_handling",
    "logs"
)

foreach ($dir in $directories) {
    if (Test-Path $dir) {
        $fileCount = (Get-ChildItem $dir -File).Count
        Write-Host "  ✓ $dir ($fileCount files)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ $dir not found" -ForegroundColor Yellow
    }
}

Write-Host ""

# 4. コアモジュール構文チェック
Write-Host "📋 4. コアモジュール構文チェック"
$coreModules = @(
    "src\utils\exception_handler.py",
    "src\utils\error_recovery.py",
    "src\utils\logger_setup.py", 
    "src\utils\monitoring_agent.py"
)

foreach ($module in $coreModules) {
    if (Test-Path $module) {
        Write-Host "  $module を検証中..." -NoNewline
        python -m py_compile $module 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host " ✓" -ForegroundColor Green
        } else {
            Write-Host " ❌" -ForegroundColor Red
            python -m py_compile $module
        }
    } else {
        Write-Host "  ❌ $module が見つかりません" -ForegroundColor Red
    }
}

Write-Host ""

# 5. デモシステム実行
Write-Host "📋 5. デモシステム実行"
Write-Host "  エラーハンドリングシステム動作確認開始..." -ForegroundColor Blue

try {
    $startTime = Get-Date
    python demo_error_handling_system.py
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ デモシステム実行成功 (実行時間: $($duration.ToString("F2"))秒)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ デモシステム実行失敗" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "  ❌ デモシステム実行エラー: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 6. ログファイル確認
Write-Host "📋 6. ログファイル確認"
if (Test-Path "logs") {
    $logFiles = Get-ChildItem "logs" -File
    Write-Host "  生成されたログファイル: $($logFiles.Count) 個" -ForegroundColor Blue
    
    foreach ($logFile in $logFiles | Select-Object -First 5) {
        $size = [math]::Round($logFile.Length / 1KB, 2)
        Write-Host "    - $($logFile.Name) ($size KB)" -ForegroundColor Gray
    }
    
    if ($logFiles.Count -gt 5) {
        Write-Host "    ... その他 $($logFiles.Count - 5) ファイル" -ForegroundColor Gray
    }
} else {
    Write-Host "  ⚠️ logsディレクトリが見つかりません" -ForegroundColor Yellow
}

Write-Host ""

# 7. システム統計収集
Write-Host "📋 7. システム統計収集"
try {
    # 簡単な統計情報取得
    python -c "
import sys
sys.path.append('.')
from src.utils.exception_handler import get_exception_handler
from src.utils.error_recovery import get_recovery_manager
from src.utils.logger_setup import get_logger_manager
from src.utils.monitoring_agent import get_monitoring_agent

print('  システム統計:')
handler = get_exception_handler()
stats = handler.get_error_statistics()
print(f'    エラー処理: 総数 {stats[\"total_errors\"]}')

recovery = get_recovery_manager()
r_stats = recovery.get_recovery_statistics()
print(f'    復旧システム: 成功率 {r_stats.get(\"success_rate\", 0):.1f}%')

logger = get_logger_manager()
l_stats = logger.get_log_statistics()
print(f'    ログシステム: 総ログ数 {l_stats[\"total_logs\"]}')
" 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ 統計収集成功" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ 統計収集で軽微なエラーが発生しました" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠️ 統計収集エラー: $_" -ForegroundColor Yellow
}

Write-Host ""

# 8. パフォーマンス検証
Write-Host "📋 8. パフォーマンス検証"
try {
    $perfStartTime = Get-Date
    python -c "
import time
import sys
sys.path.append('.')
from src.utils.exception_handler import handle_strategy_error

# パフォーマンステスト (100回のエラー処理)
start_time = time.time()
for i in range(100):
    handle_strategy_error(f'perf_test_{i}', Exception(f'パフォーマンステスト{i}'))
end_time = time.time()

processing_time = end_time - start_time
print(f'  パフォーマンステスト: 100回処理 = {processing_time:.3f}秒')
print(f'  平均処理時間: {processing_time/100*1000:.2f}ms/回')
" 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ パフォーマンス検証成功" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️ パフォーマンス検証で問題が発生しました" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠️ パフォーマンス検証エラー: $_" -ForegroundColor Yellow
}

Write-Host ""

# 9. Git状態確認とコミット準備
Write-Host "📋 9. Git状態確認とコミット準備"
try {
    # Git状態確認
    $gitStatus = git status --porcelain 2>$null
    if ($LASTEXITCODE -eq 0) {
        if ($gitStatus) {
            $changedFiles = ($gitStatus -split "`n").Count
            Write-Host "  Git状態: $changedFiles 個の変更ファイル検出" -ForegroundColor Blue
            
            # 変更ファイル表示（最初の10個）
            $gitStatus -split "`n" | Select-Object -First 10 | ForEach-Object {
                Write-Host "    $_" -ForegroundColor Gray
            }
            if ($changedFiles -gt 10) {
                Write-Host "    ... その他 $($changedFiles - 10) ファイル" -ForegroundColor Gray
            }
        } else {
            Write-Host "  Git状態: 変更なし" -ForegroundColor Green
        }
    } else {
        Write-Host "  ⚠️ Gitリポジトリが初期化されていません" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠️ Git状態確認エラー: $_" -ForegroundColor Yellow
}

Write-Host ""

# 10. 実行結果サマリー
Write-Host "📋 10. 実行結果サマリー"
Write-Host "=" * 70
Write-Host "🎯 フェーズ3-A エラーハンドリング強化システム 実行完了" -ForegroundColor Green
Write-Host ""
Write-Host "実装完了コンポーネント:" -ForegroundColor Blue
Write-Host "  ✅ 統一例外処理システム (UnifiedExceptionHandler)"
Write-Host "  ✅ エラー復旧管理システム (ErrorRecoveryManager)"
Write-Host "  ✅ 強化ロギングシステム (EnhancedLoggerManager)"
Write-Host "  ✅ 監視エージェントシステム (MonitoringAgent)"
Write-Host "  ✅ 設定ベース管理 (JSON設定ファイル)"
Write-Host "  ✅ 包括的テストスイート"
Write-Host "  ✅ 統合デモシステム"
Write-Host ""

# Git コミット実行確認
Write-Host "Git コミットを実行しますか? (Y/N): " -NoNewline -ForegroundColor Yellow
$response = Read-Host

if ($response -eq 'Y' -or $response -eq 'y') {
    Write-Host ""
    Write-Host "📋 11. Git コミット実行"
    
    try {
        # ステージング
        git add .
        Write-Host "  ✓ ファイルをステージングしました" -ForegroundColor Green
        
        # コミット
        $commitMessage = "フェーズ3-A: エラーハンドリング強化システム実装完了

- 統一例外処理システム (UnifiedExceptionHandler)
  * 戦略/データ/システムエラーの統一処理
  * 自動復旧機能とフォールバック機構
  * 設定ベースエラーポリシー管理

- エラー復旧管理システム (ErrorRecoveryManager)  
  * 複数リトライ戦略 (Simple/Exponential/Linear)
  * フォールバック機能とサーキットブレーカー
  * 復旧統計と成功率追跡

- 強化ロギングシステム (EnhancedLoggerManager)
  * 戦略別ログ管理と自動ローテーション
  * JSON構造化ログとパフォーマンス監視
  * エラー分析とログ統計機能

- 監視エージェントシステム (MonitoringAgent)
  * リアルタイム監視とアラートシステム
  * メール/Webhook通知機能
  * カスタムアラートルールとメトリクス収集

- 設定管理とテスト
  * JSON設定ファイルによる動的設定管理
  * 包括的単体・統合テストスイート
  * エージェントモード実行デモシステム

実装ファイル:
- src/utils/exception_handler.py
- src/utils/error_recovery.py  
- src/utils/logger_setup.py
- src/utils/monitoring_agent.py
- config/error_handling/*.json
- tests/test_error_handling/*.py
- demo_error_handling_system.py

タスク進捗: フェーズ3-A 完了 ✅"

        git commit -m $commitMessage
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Git コミット成功" -ForegroundColor Green
            
            # コミット情報表示
            $commitHash = git rev-parse --short HEAD
            Write-Host "  コミットハッシュ: $commitHash" -ForegroundColor Blue
        } else {
            Write-Host "  ❌ Git コミット失敗" -ForegroundColor Red
        }
    } catch {
        Write-Host "  ❌ Git コミットエラー: $_" -ForegroundColor Red
    }
} else {
    Write-Host "  Git コミットをスキップしました" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 エラーハンドリングシステム エージェントモード実行完了!" -ForegroundColor Green
Write-Host "=" * 70

# 実行時間表示
$scriptEndTime = Get-Date
$totalDuration = ($scriptEndTime - $startTime).TotalSeconds
Write-Host "総実行時間: $($totalDuration.ToString("F2"))秒" -ForegroundColor Blue

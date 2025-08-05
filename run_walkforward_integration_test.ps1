# ウォークフォワードシステム統合テスト実行スクリプト
# pytest を使用してシステム全体のテストを実行

Write-Host "=== ウォークフォワードシステム統合テスト ===" -ForegroundColor Green

try {
    # Python環境の確認
    Write-Host "Python環境を確認中..." -ForegroundColor Yellow
    python --version
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Pythonが見つかりません。" -ForegroundColor Red
        exit 1
    }
    
    # pytest の確認とインストール
    Write-Host "pytest の確認中..." -ForegroundColor Yellow
    python -c "import pytest; print(f'pytest {pytest.__version__}')" 2>$null
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pytest がインストールされていません。インストール中..." -ForegroundColor Yellow
        python -m pip install pytest
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ pytest のインストールに失敗しました。" -ForegroundColor Red
            exit 1
        }
    }
    
    # 必要なディレクトリの確認
    Write-Host "テスト環境を準備中..." -ForegroundColor Yellow
    if (!(Test-Path "tests")) {
        Write-Host "❌ tests ディレクトリが見つかりません。" -ForegroundColor Red
        exit 1
    }
    
    if (!(Test-Path "src\analysis")) {
        Write-Host "❌ src\analysis ディレクトリが見つかりません。" -ForegroundColor Red
        exit 1
    }
    
    # 統合テストの実行
    Write-Host "ウォークフォワードシステム統合テストを実行中..." -ForegroundColor Yellow
    Write-Host "テストファイル: tests\test_walkforward_integration.py" -ForegroundColor Cyan
    
    python -m pytest tests\test_walkforward_integration.py -v --tb=short
    
    $testResult = $LASTEXITCODE
    
    if ($testResult -eq 0) {
        Write-Host ""
        Write-Host "✅ 統合テスト完了！全てのテストが成功しました。" -ForegroundColor Green
        Write-Host ""
        Write-Host "🎉 ウォークフォワードシステム統合テスト成功 🎉" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "確認されたシステム機能:" -ForegroundColor Yellow
        Write-Host "  ✓ シナリオ管理システム" -ForegroundColor White
        Write-Host "  ✓ ウォークフォワード実行エンジン" -ForegroundColor White
        Write-Host "  ✓ 結果分析・レポートシステム" -ForegroundColor White
        Write-Host "  ✓ Excel出力機能" -ForegroundColor White
        Write-Host "  ✓ エラーハンドリング" -ForegroundColor White
        Write-Host ""
        Write-Host "次のステップ:" -ForegroundColor Yellow
        Write-Host "  1. run_walkforward_demo.ps1 でデモ実行" -ForegroundColor White
        Write-Host "  2. 実際のデータでテスト実行" -ForegroundColor White
        
    } else {
        Write-Host ""
        Write-Host "❌ 統合テストでエラーが発生しました。" -ForegroundColor Red
        Write-Host "上記のテスト結果を確認してエラー原因を調べてください。" -ForegroundColor Yellow
        exit 1
    }
    
} catch {
    Write-Host ""
    Write-Host "❌ 予期しないエラーが発生しました: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "統合テストスクリプト完了" -ForegroundColor Green

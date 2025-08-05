# ウォークフォワードテストシステム デモ実行スクリプト
# Phase 2: パフォーマンス検証システムの動作確認

Write-Host "=== ウォークフォワードテストシステム デモ実行 ===" -ForegroundColor Green

try {
    # Python環境の確認
    Write-Host "Python環境を確認中..." -ForegroundColor Yellow
    python --version
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Pythonが見つかりません。Pythonをインストールしてください。" -ForegroundColor Red
        exit 1
    }
    
    # 必要なディレクトリの作成
    Write-Host "出力ディレクトリを準備中..." -ForegroundColor Yellow
    if (!(Test-Path "output")) {
        New-Item -ItemType Directory -Path "output" | Out-Null
    }
    if (!(Test-Path "output\walkforward_demo_results")) {
        New-Item -ItemType Directory -Path "output\walkforward_demo_results" | Out-Null
    }
    
    # デモ実行
    Write-Host "ウォークフォワードテストシステムデモを実行中..." -ForegroundColor Yellow
    python run_walkforward_demo.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ デモ実行完了！" -ForegroundColor Green
        Write-Host ""
        Write-Host "🎉 Phase 2: パフォーマンス検証システム実装完了 🎉" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "生成されたファイル:" -ForegroundColor Yellow
        if (Test-Path "output\walkforward_demo_results") {
            Get-ChildItem "output\walkforward_demo_results" | ForEach-Object {
                Write-Host "  - $($_.Name)" -ForegroundColor White
            }
        }
        Write-Host ""
        Write-Host "次のステップ:" -ForegroundColor Yellow
        Write-Host "  1. output\walkforward_demo_results\ のExcelファイルで詳細結果を確認" -ForegroundColor White
        Write-Host "  2. logs\ でログファイルを確認" -ForegroundColor White
        Write-Host "  3. 実際のデータでテスト実行する場合は設定を調整" -ForegroundColor White
        
    } else {
        Write-Host ""
        Write-Host "❌ デモ実行中にエラーが発生しました。" -ForegroundColor Red
        Write-Host "ログファイルを確認してエラー原因を調べてください。" -ForegroundColor Yellow
        exit 1
    }
    
} catch {
    Write-Host ""
    Write-Host "❌ 予期しないエラーが発生しました: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "デモ実行スクリプト完了" -ForegroundColor Green

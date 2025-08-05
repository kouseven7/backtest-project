# テスト実行スクリプト
# PowerShell用

Write-Host "バックテストシステム テスト実行" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Pythonの存在確認
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python が見つかりません" -ForegroundColor Red
    exit 1
}

# pytest の実行
Write-Host "pytest実行中..." -ForegroundColor Yellow
pytest tests/ -v

Write-Host "テスト完了" -ForegroundColor Green

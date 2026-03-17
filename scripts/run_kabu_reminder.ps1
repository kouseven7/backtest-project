# UTF-8対応ラッパースクリプト
# このスクリプトから呼び出すことで、日本語が正しく表示されます

# コンソールのコードページをUTF-8に設定
chcp 65001 | Out-Null
$PSDefaultParameterValues['*:Encoding'] = 'utf8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# メイン送信スクリプトを実行
& "$PSScriptRoot\send_kabu_login_reminder_new.ps1"

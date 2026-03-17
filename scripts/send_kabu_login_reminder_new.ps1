# UTF-8 BOM莉倥″縺ｧ菫晏ｭ倥☆繧九％縺ｨ
# 繧ｳ繝ｳ繧ｽ繝ｼ繝ｫ縺ｮ繧ｳ繝ｼ繝峨・繝ｼ繧ｸ繧旦TF-8縺ｫ險ｭ螳・
chcp 65001 | Out-Null

$PSDefaultParameterValues['*:Encoding'] = 'utf8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[System.Console]::InputEncoding = [System.Text.Encoding]::UTF8

$fromAddress  = "simasima.mk@gmail.com"
$toAddress    = "simasima.mk@gmail.com"
$appPassword  = "lbif pqbw jqqa uoeu"

$smtpServer = "smtp.gmail.com"
$smtpPort   = 587
$subject    = "[DSSMS] kabuステーション ログインしてください (6:30)"
$body       = @"
おはようございます。

DSSMSからの自動通知です。

kabuステーションの強制ログアウト（6:15）が完了しました。
取引開始前に、kabuステーションへの再ログインをお願いします。

【手順】
1. kabuステーションを開く
2. パスワードを入力
3. メール二段階認証を完了

【本日の予定】
- 前場スクリーニング: 09:30
- 後場スクリーニング: 12:30

ログインが完了していない場合、APIはyfinanceにフォールバックして動作します。
ペーパートレード中は実取引への影響はありません。

DSSMS自動通知
"@

$smtp = New-Object Net.Mail.SmtpClient($smtpServer, $smtpPort)
$smtp.EnableSsl = $true
$smtp.Credentials = New-Object Net.NetworkCredential($fromAddress, $appPassword)

$message = New-Object Net.Mail.MailMessage($fromAddress, $toAddress, $subject, $body)
$message.IsBodyHtml = $false
$message.BodyEncoding = [System.Text.Encoding]::UTF8
$message.SubjectEncoding = [System.Text.Encoding]::UTF8

try {
    $smtp.Send($message)
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    
    # ログファイルに記録（UTF-8で保存）
    $logDir = "$PSScriptRoot\..\logs"
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    }
    $logFile = "$logDir\kabu_reminder.log"
    $logMessage = "[OK] メール送信成功: $timestamp"
    Add-Content -Path $logFile -Value $logMessage -Encoding UTF8
    
    # コンソールには英語で出力（文字化け回避）
    Write-Host "[OK] Email sent successfully: $timestamp"
    exit 0
} catch {
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $errorMsg = $_.Exception.Message
    
    # ログファイルに記録
    $logMessage = "[ERROR] メール送信失敗: $timestamp - $errorMsg"
    Add-Content -Path $logFile -Value $logMessage -Encoding UTF8
    
    # コンソールには英語で出力
    Write-Host "[ERROR] Email sending failed: $errorMsg"
    exit 1
}

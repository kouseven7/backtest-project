"""
EmailNotifier - DSSMS発注失敗メール通知システム

発注失敗時（_execute_with_retry()で3回リトライ全失敗）にメール通知を送信する。
Pythonの標準ライブラリ（smtplib、email.mime）のみを使用。

主な機能:
- kabu_connection_config.jsonからメール設定を読み込み
- 発注失敗時に管理者へメール送信
- enabled=false時は送信スキップ（デフォルト）
- メール送信失敗時もシステム停止させない

統合コンポーネント:
- DSSMSScheduler: _execute_with_retry()で全リトライ失敗後に呼び出し

セーフティ機能/注意事項:
- メール送信失敗は握りつぶす（try-exceptで保護）
- sender_email/recipient_email未設定時はWARNINGログのみ
- 本番環境ではSMTP認証情報を環境変数から取得推奨

Author: Backtest Project Team
Created: 2026-03-12
Last Modified: 2026-03-12
"""

import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from config.logger_config import setup_logger


class EmailNotifier:
    """
    DSSMS発注失敗時のメール通知クラス。
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: kabu_connection_config.jsonのパス
        """
        self.logger = setup_logger("EmailNotifier")
        self.config_path = config_path

        # 設定読み込み
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            email_config = config.get("email_notification", {})
        except Exception as e:
            self.logger.error(f"[EMAIL_CONFIG_ERROR] 設定読み込み失敗: {e}")
            email_config = {}

        self.enabled = email_config.get("enabled", False)
        self.smtp_server = email_config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = email_config.get("smtp_port", 587)
        self.sender_email = email_config.get("sender_email", "")
        self.sender_password = email_config.get("sender_password", "")
        self.recipient_email = email_config.get("recipient_email", "")

        if not self.enabled:
            self.logger.info("[EMAIL_NOTIFIER] メール通知は無効化されています (enabled=false)")

    def send_order_failed(
        self,
        symbol: str,
        order_type: str,
        order_params: dict,
        retry_count: int
    ):
        """
        発注失敗時のメール通知を送信する。

        Args:
            symbol: 銘柄コード（例："8031"）
            order_type: "BUY" or "SELL"
            order_params: 発注パラメータの辞書
            retry_count: リトライ回数（通常3）
        """
        # enabled=falseならスキップ
        if not self.enabled:
            self.logger.debug("[EMAIL_SKIP] enabled=false のためメール送信をスキップ")
            return

        # メールアドレス未設定確認
        if not self.sender_email or not self.recipient_email:
            self.logger.warning(
                "[EMAIL_NOT_CONFIGURED] sender_emailまたはrecipient_emailが未設定のため、"
                "メール送信をスキップします"
            )
            return

        try:
            # メール本文作成
            body = f"""DSSMSで発注失敗が発生しました。

銘柄コード: {symbol}
注文種別: {order_type}
リトライ回数: {retry_count}回
発生時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

発注パラメータ:
{json.dumps(order_params, ensure_ascii=False, indent=2)}

ペーパートレードのため実際の損害はありませんが、確認してください。
"""

            # メールメッセージ作成
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"[DSSMS] 発注失敗アラート: {symbol} {order_type}"
            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # SMTP接続・送信
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()  # TLS暗号化
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            self.logger.info(
                f"[EMAIL_SENT] 発注失敗メール送信成功: {symbol} {order_type} → {self.recipient_email}"
            )

        except smtplib.SMTPAuthenticationError as e:
            self.logger.error(f"[EMAIL_FAILED] SMTP認証失敗: {e}")
        except smtplib.SMTPException as e:
            self.logger.error(f"[EMAIL_FAILED] SMTP送信エラー: {e}")
        except Exception as e:
            self.logger.error(f"[EMAIL_FAILED] メール送信失敗: {type(e).__name__}: {e}")

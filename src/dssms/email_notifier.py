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
from datetime import datetime, date
from typing import Any, Dict, List, cast
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
                loaded_config: Any = json.load(f)

            config = cast(Dict[str, Any], loaded_config) if isinstance(loaded_config, dict) else {}
            raw_email_config = config.get("email_notification", {})
            email_config = cast(Dict[str, Any], raw_email_config) if isinstance(raw_email_config, dict) else {}
        except Exception as e:
            self.logger.error(f"[EMAIL_CONFIG_ERROR] 設定読み込み失敗: {e}")
            email_config = {}

        self.enabled: bool = bool(email_config.get("enabled", False))
        self.smtp_server: str = str(email_config.get("smtp_server", "smtp.gmail.com"))
        self.smtp_port: int = int(email_config.get("smtp_port", 587))
        self.sender_email: str = str(email_config.get("sender_email", ""))
        self.sender_password: str = str(email_config.get("sender_password", ""))
        self.recipient_email: str = str(email_config.get("recipient_email", ""))

        if not self.enabled:
            self.logger.info("[EMAIL_NOTIFIER] メール通知は無効化されています (enabled=false)")

    def send_order_failed(
        self,
        symbol: str,
        order_type: str,
        order_params: Dict[str, Any],
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

    def send_morning_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        朝の実行サマリーをメール送信する。

        Args:
            summary_data: 朝サマリー情報

        Returns:
            bool: 送信成功時True、未送信または失敗時False
        """
        if not self.enabled:
            self.logger.debug("[EMAIL_SKIP] enabled=false のため朝サマリー送信をスキップ")
            return False

        if not self.sender_email or not self.recipient_email:
            self.logger.warning(
                "[EMAIL_NOT_CONFIGURED] sender_emailまたはrecipient_emailが未設定のため、"
                "朝サマリー送信をスキップします"
            )
            return False

        try:
            today = str(date.today())
            execution_time = summary_data.get("execution_time", "-")
            status = summary_data.get("status", "正常")
            error_message = summary_data.get("error_message", "")
            cash_balance = float(summary_data.get("cash_balance", 0) or 0)
            unrealized_pnl = summary_data.get("unrealized_pnl")
            total_assets = float(summary_data.get("total_assets", 0) or 0)
            daily_pnl = float(summary_data.get("daily_pnl", 0) or 0)
            positions: List[Dict[str, Any]] = summary_data.get("positions") or []
            screened_symbols: List[str] = summary_data.get("screened_symbols") or []
            hours_since_last_run = float(summary_data.get("hours_since_last_run", 0.0) or 0.0)

            if unrealized_pnl is None:
                unrealized_pnl_text = "取得失敗"
            else:
                unrealized_pnl_text = f"{float(unrealized_pnl):+,.0f}円"

            if positions:
                position_lines: List[str] = []
                for position in positions:
                    symbol = position.get("symbol", "-")
                    price = float(position.get("price", 0) or 0)
                    shares = int(position.get("shares", 0) or 0)
                    position_unrealized = float(position.get("unrealized_pnl", 0) or 0)
                    position_lines.append(
                        f"* {symbol}：{price:,.0f}円 × {shares}株（含み{position_unrealized:+,.0f}円）"
                    )
                positions_text = "\n".join(position_lines)
            else:
                positions_text = "なし"

            screened_symbols_text = ", ".join(screened_symbols) if screened_symbols else "なし"
            anomaly_line = f"前回実行からの経過時間：{hours_since_last_run:.1f}時間"
            if hours_since_last_run >= 24:
                anomaly_line += " ⚠ 前日スキップの可能性あり"

            error_line = f"エラー内容：{error_message}\n" if error_message else ""

            body = (
                f"■ 実行結果\n"
                f"実行時刻：{execution_time}\n"
                f"ステータス：{status}\n"
                f"{error_line}"
                f"\n"
                f"■ 資産状況\n"
                f"現金残高：{cash_balance:,.0f}円\n"
                f"含み損益：{unrealized_pnl_text}\n"
                f"総資産：{total_assets:,.0f}円\n"
                f"当日損益：{daily_pnl:+,.0f}円\n"
                f"\n"
                f"■ 保有ポジション（{len(positions)}/3銘柄）\n"
                f"{positions_text}\n"
                f"\n"
                f"■ 本日のスクリーニング結果\n"
                f"選択銘柄：{screened_symbols_text}\n"
                f"\n"
                f"■ 異常検知\n"
                f"{anomaly_line}"
            )

            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"【DSSMS】{today} 朝の実行サマリー"
            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            self.logger.info(f"[EMAIL_SENT] 朝サマリー送信成功 → {self.recipient_email}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            self.logger.error(f"[EMAIL_FAILED] 朝サマリーSMTP認証失敗: {e}")
        except smtplib.SMTPException as e:
            self.logger.error(f"[EMAIL_FAILED] 朝サマリーSMTP送信エラー: {e}")
        except Exception as e:
            self.logger.error(f"[EMAIL_FAILED] 朝サマリー送信失敗: {type(e).__name__}: {e}")

        return False

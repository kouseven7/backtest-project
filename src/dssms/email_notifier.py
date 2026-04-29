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

    def send_order_executed(
        self,
        order_type: str,
        code: str,
        price: float,
        shares: int,
        profit_loss: float = None,
    ) -> bool:
        """
        発注成功時のメール通知を送信する。

        Args:
            order_type: "BUY" or "SELL"
            code: 銘柄コード（例: "8031"）
            price: 約定価格
            shares: 約定株数
            profit_loss: 損益（SELL時のみ任意）
        """
        if not self.enabled:
            self.logger.debug("[EMAIL_SKIP] enabled=false のためメール送信をスキップ")
            return False

        if not self.sender_email or not self.recipient_email:
            self.logger.warning(
                "[EMAIL_NOT_CONFIGURED] sender_emailまたはrecipient_emailが未設定のため、"
                "メール送信をスキップします"
            )
            return False

        try:
            pnl_line = ""
            if profit_loss is not None:
                pnl_line = f"損益: {profit_loss:+,.0f}円\n"

            body = f"""DSSMSで注文が約定しました。

銘柄コード: {code}
注文種別: {order_type}
約定価格: {price:,.2f}円
株数: {shares}株
{pnl_line}発生時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"[DSSMS] 注文約定: {code} {order_type}"
            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            self.logger.info(
                f"[EMAIL_SENT] 約定通知メール送信成功: {code} {order_type} -> {self.recipient_email}"
            )
            return True

        except smtplib.SMTPAuthenticationError as e:
            self.logger.error(f"[EMAIL_FAILED] SMTP認証失敗: {e}")
            return False
        except smtplib.SMTPException as e:
            self.logger.error(f"[EMAIL_FAILED] SMTP送信失敗: {e}")
            return False
        except Exception as e:
            self.logger.error(f"[EMAIL_FAILED] 予期しないエラー: {e}")
            return False

    def send_dd_alert(self, current_dd_pct: float, equity: float, threshold: float,
                      close_result: dict = None) -> bool:
        """
        DD閾値超過時の緊急通知メール

        Args:
            current_dd_pct: 現在のDD（%）
            equity: 現在の総資産（円）
            threshold: 閾値（%）
        """
        if not self.enabled:
            self.logger.debug("[EMAIL_SKIP] enabled=false のためメール送信をスキップ")
            return False

        if not self.sender_email or not self.recipient_email:
            self.logger.warning(
                "[EMAIL_NOT_CONFIGURED] sender_emailまたはrecipient_emailが未設定のため、"
                "メール送信をスキップします"
            )
            return False

        try:
            occurred_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            subject = f"[DSSMS緊急] DDアラート: {current_dd_pct:.1f}% に到達"
            body = f"""DSSMSでDD閾値超過を検知しました。

現在のDD: {current_dd_pct:.2f}%
現在の総資産: {equity:,.0f}円
設定閾値: {threshold:.1f}%

スケジューラーを停止しました。
発生時刻: {occurred_at}
"""

            # 強制決済結果ブロック（close_result がある場合のみ追記）
            if close_result and close_result.get('closed_count', 0) > 0:
                lines = [
                    "",
                    "【強制決済結果】",
                    f"決済件数: {close_result['closed_count']}件",
                    f"合計損益: {close_result['total_pnl']:+,.0f}円",
                    "銘柄別:",
                ]
                for d in close_result.get('details', []):
                    if 'error' in d:
                        lines.append(f"  {d['symbol']}: 決済失敗 ({d['error']})")
                    else:
                        lines.append(
                            f"  {d['symbol']}: {d['pnl']:+,.0f}円"
                            f" (exit: {d['exit_price']:.0f}円 × {d['shares']}株)"
                        )
                # 既存の本文に追記
                body += "\n".join(lines)

            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, msg.as_string())

            self.logger.info(f"[EMAIL_SENT] DDアラート送信成功 → {self.recipient_email}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            self.logger.error(f"[EMAIL_FAILED] SMTP認証失敗: {e}")
            return False
        except smtplib.SMTPException as e:
            self.logger.error(f"[EMAIL_FAILED] SMTP送信失敗: {e}")
            return False
        except Exception as e:
            self.logger.error(f"[EMAIL_FAILED] 予期しないエラー: {e}")
            return False

    def send_morning_summary(self, summary_data: dict) -> bool:
        """朝の実行サマリーメール送信"""
        if not self.enabled:
            self.logger.debug("[EMAIL_SKIP] enabled=false のためメール送信をスキップ")
            return False

        try:
            from datetime import date
            import json

            today = date.today().isoformat()
            execution_time = summary_data.get('execution_time', '--:--')
            status = summary_data.get('status', '不明')
            error_message = summary_data.get('error_message', '')
            cash_balance = summary_data.get('cash_balance', 0)
            unrealized_pnl = summary_data.get('unrealized_pnl', None)
            total_assets = summary_data.get('total_assets', 0)
            daily_pnl = summary_data.get('daily_pnl', 0)
            positions = summary_data.get('positions', [])
            screened_symbols = summary_data.get('screened_symbols', [])
            hours_since_last_run = summary_data.get('hours_since_last_run', 0.0)
            current_dd_pct = summary_data.get('current_dd_pct', None)

            # 含み損益の表示
            if unrealized_pnl is None:
                unrealized_str = "取得失敗"
            else:
                unrealized_str = f"{unrealized_pnl:+,.0f}円"

            # ポジション表示
            if positions:
                pos_lines = "\n".join([
                    f"* {p.get('symbol', '?')}：{p.get('price', 0):,.0f}円 × {p.get('shares', 0)}株"
                    f"（含み{p.get('unrealized_pnl', 0):+,.0f}円）"
                    for p in positions
                ])
            else:
                pos_lines = "なし"

            # スクリーニング結果
            screened_str = "、".join(screened_symbols) if screened_symbols else "なし"

            # 異常検知
            skip_warning = ""
            if hours_since_last_run >= 24:
                skip_warning = " ⚠ 前日スキップの可能性あり"

            # エラー行
            error_line = f"エラー内容：{error_message}\n" if status == '異常' and error_message else ""
            dd_line = f"{current_dd_pct:.2f}%" if current_dd_pct is not None else "-"

            body = f"""■ 実行結果
実行時刻：{execution_time}
ステータス：{status}
{error_line}
■ 資産状況
現金残高：{cash_balance:,.0f}円
含み損益：{unrealized_str}
総資産：{total_assets:,.0f}円
当日損益：{daily_pnl:+,.0f}円
DD（開始残高比）：{dd_line}

■ 保有ポジション（{len(positions)}/3銘柄）
{pos_lines}

■ 本日のスクリーニング結果
選択銘柄：{screened_str}

■ 異常検知
前回実行からの経過時間：{hours_since_last_run:.1f}時間{skip_warning}
"""

            subject = f"【DSSMS】{today} 朝の実行サマリー"

            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, msg.as_string())

            self.logger.info(f"[EMAIL_SENT] 朝サマリー送信成功 → {self.recipient_email}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            self.logger.error(f"[EMAIL_FAILED] SMTP認証失敗: {e}")
            return False
        except smtplib.SMTPException as e:
            self.logger.error(f"[EMAIL_FAILED] SMTP送信失敗: {e}")
            return False
        except Exception as e:
            self.logger.error(f"[EMAIL_FAILED] 予期しないエラー: {e}")
            return False

    def send_afternoon_summary(self, summary_data: dict) -> bool:
        """朝の実行サマリーメール送信"""
        if not self.enabled:
            self.logger.debug("[EMAIL_SKIP] enabled=false のためメール送信をスキップ")
            return False

        try:
            from datetime import date
            import json

            today = date.today().isoformat()
            execution_time = summary_data.get('execution_time', '--:--')
            status = summary_data.get('status', '不明')
            error_message = summary_data.get('error_message', '')
            cash_balance = summary_data.get('cash_balance', 0)
            unrealized_pnl = summary_data.get('unrealized_pnl', None)
            total_assets = summary_data.get('total_assets', 0)
            daily_pnl = summary_data.get('daily_pnl', 0)
            positions = summary_data.get('positions', [])
            screened_symbols = summary_data.get('screened_symbols', [])
            hours_since_last_run = summary_data.get('hours_since_last_run', 0.0)
            current_dd_pct = summary_data.get('current_dd_pct', None)

            # 含み損益の表示
            if unrealized_pnl is None:
                unrealized_str = "取得失敗"
            else:
                unrealized_str = f"{unrealized_pnl:+,.0f}円"

            # ポジション表示
            if positions:
                pos_lines = "\n".join([
                    f"* {p.get('symbol', '?')}：{p.get('price', 0):,.0f}円 × {p.get('shares', 0)}株"
                    f"（含み{p.get('unrealized_pnl', 0):+,.0f}円）"
                    for p in positions
                ])
            else:
                pos_lines = "なし"

            # スクリーニング結果
            screened_str = "、".join(screened_symbols) if screened_symbols else "なし"

            # 異常検知
            skip_warning = ""
            if hours_since_last_run >= 24:
                skip_warning = " ⚠ 前日スキップの可能性あり"

            # エラー行
            error_line = f"エラー内容：{error_message}\n" if status == '異常' and error_message else ""
            dd_line = f"{current_dd_pct:.2f}%" if current_dd_pct is not None else "-"

            body = f"""■ 実行結果
実行時刻：{execution_time}
ステータス：{status}
{error_line}
■ 資産状況
現金残高：{cash_balance:,.0f}円
含み損益：{unrealized_str}
総資産：{total_assets:,.0f}円
当日損益：{daily_pnl:+,.0f}円
DD（開始残高比）：{dd_line}

■ 保有ポジション（{len(positions)}/3銘柄）
{pos_lines}

■ 本日のスクリーニング結果
選択銘柄：{screened_str}

■ 異常検知
前回実行からの経過時間：{hours_since_last_run:.1f}時間{skip_warning}
"""

            subject = f"【DSSMS】{today} 後場の実行サマリー"

            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, msg.as_string())

            self.logger.info(f"[EMAIL_SENT] 午後サマリー送信成功 → {self.recipient_email}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            self.logger.error(f"[EMAIL_FAILED] SMTP認証失敗: {e}")
            return False
        except smtplib.SMTPException as e:
            self.logger.error(f"[EMAIL_FAILED] SMTP送信失敗: {e}")
            return False
        except Exception as e:
            self.logger.error(f"[EMAIL_FAILED] 予期しないエラー: {e}")
            return False

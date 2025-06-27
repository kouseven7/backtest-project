"""
トレード結果の詳細な分析とレポート生成を行うモジュール
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import os
import json
import tempfile
import webbrowser


class TradeAnalyzer:
    """トレード結果の分析を行うクラス"""
    
    def __init__(self, trade_results: Dict, strategy_name: str, parameters: Dict = None):
        """
        初期化
        
        Parameters:
            trade_results (Dict): バックテスト結果
            strategy_name (str): 戦略名
            parameters (Dict, optional): 戦略パラメータ
        """
        self.trade_results = trade_results
        self.strategy_name = strategy_name
        self.parameters = parameters or {}
        self.logger = logging.getLogger(__name__)
        
        # 取引履歴のDataFrame
        self.trades = trade_results.get("取引履歴", pd.DataFrame())
        # 損益推移のDataFrame
        self.pnl_summary = trade_results.get("損益推移", pd.DataFrame())
        # 取引統計の辞書
        self.stats = trade_results.get("取引統計", {})
        # 月次パフォーマンスのDataFrame
        self.monthly = trade_results.get("月次パフォーマンス", pd.DataFrame())
        
    def analyze_all(self, output_dir: str) -> Dict[str, Any]:
        """
        全ての分析を実行し、結果を出力
        
        Parameters:
            output_dir (str): 出力ディレクトリ
            
        Returns:
            Dict[str, Any]: 分析結果サマリー
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {}
        
        # 基本統計の分析
        results["basic_stats"] = self._analyze_basic_stats()
        
        # トレード特性の分析
        if not self.trades.empty:
            results["trade_characteristics"] = self._analyze_trade_characteristics()
        
        # 時系列分析
        if not self.pnl_summary.empty:
            results["time_series"] = self._analyze_time_series()
        
        # 結果をJSONとして保存
        results_file = os.path.join(output_dir, f"{self.strategy_name}_analysis_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            # NaNやDatetime型をJSON化できないので変換
            json.dump(self._convert_for_json(results), f, ensure_ascii=False, indent=2)
        
        # HTMLレポート生成
        html_file = self._generate_html_report(output_dir, results, timestamp)
        
        return results
    
    def _analyze_basic_stats(self) -> Dict[str, Any]:
        """基本統計の分析"""
        result = {}
        
        # 取引統計があれば使用
        if self.stats:
            result.update(self.stats)
        
        # 取引履歴からの統計
        if not self.trades.empty:
            # 勝率
            win_trades = self.trades[self.trades["損益率"] > 0]
            loss_trades = self.trades[self.trades["損益率"] <= 0]
            
            result["トレード総数"] = len(self.trades)
            result["勝ちトレード数"] = len(win_trades)
            result["負けトレード数"] = len(loss_trades)
            result["勝率"] = len(win_trades) / len(self.trades) if len(self.trades) > 0 else 0
            
            # 平均損益
            result["平均リターン"] = self.trades["損益率"].mean() if len(self.trades) > 0 else 0
            result["平均勝ちリターン"] = win_trades["損益率"].mean() if len(win_trades) > 0 else 0
            result["平均負けリターン"] = loss_trades["損益率"].mean() if len(loss_trades) > 0 else 0
            result["リターン標準偏差"] = self.trades["損益率"].std() if len(self.trades) > 0 else 0
            
            # プロフィットファクター
            total_wins = win_trades["損益率"].sum() if len(win_trades) > 0 else 0
            total_losses = abs(loss_trades["損益率"].sum()) if len(loss_trades) > 0 else 0
            result["プロフィットファクター"] = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # 保有期間
            if "保有期間" in self.trades.columns:
                result["平均保有期間"] = self.trades["保有期間"].mean()
                result["最長保有期間"] = self.trades["保有期間"].max()
                result["最短保有期間"] = self.trades["保有期間"].min()
                
        # 損益推移からの統計
        if not self.pnl_summary.empty and "累積損益" in self.pnl_summary.columns:
            # 最大ドローダウン
            max_dd_pct = self._calculate_max_drawdown()
            result["最大ドローダウン"] = max_dd_pct
            
            # 累積リターン
            if len(self.pnl_summary) > 0:
                result["累積リターン"] = self.pnl_summary["累積損益"].iloc[-1]
            
        return result
    
    def _analyze_trade_characteristics(self) -> Dict[str, Any]:
        """トレードの特性分析"""
        result = {}
        
        if self.trades.empty:
            return result
            
        # 曜日別分析（取引日の曜日情報がある場合）
        if isinstance(self.trades.index, pd.DatetimeIndex):
            # 曜日別のリターン
            weekday_performance = self.trades.groupby(self.trades.index.dayofweek)["損益率"].agg(["mean", "count"])
            weekday_map = {0: "月曜", 1: "火曜", 2: "水曜", 3: "木曜", 4: "金曜", 5: "土曜", 6: "日曜"}
            weekday_performance.index = weekday_performance.index.map(weekday_map)
            result["曜日別パフォーマンス"] = weekday_performance.to_dict()
        
        # ポジションサイズと損益の相関（ポジションサイズがある場合）
        if "ポジションサイズ" in self.trades.columns:
            correlation = self.trades[["ポジションサイズ", "損益率"]].corr().iloc[0, 1]
            result["ポジションサイズと損益の相関"] = correlation
        
        # 連続勝ち負け分析
        if "損益率" in self.trades.columns:
            # 連勝/連敗の計算
            streak_data = self._calculate_streaks()
            result["最大連勝"] = streak_data["最大連勝"]
            result["最大連敗"] = streak_data["最大連敗"]
            result["平均連勝"] = streak_data["平均連勝"]
            result["平均連敗"] = streak_data["平均連敗"]
        
        return result
    
    def _analyze_time_series(self) -> Dict[str, Any]:
        """時系列分析"""
        result = {}
        
        if self.pnl_summary.empty:
            return result
            
        # 月次リターンがある場合
        if not self.monthly.empty and "月間リターン" in self.monthly.columns:
            # 月次統計
            monthly_returns = self.monthly["月間リターン"]
            result["月次平均リターン"] = monthly_returns.mean()
            result["月次リターン標準偏差"] = monthly_returns.std()
            result["勝ち月数"] = sum(monthly_returns > 0)
            result["負け月数"] = sum(monthly_returns <= 0)
            result["月次勝率"] = sum(monthly_returns > 0) / len(monthly_returns) if len(monthly_returns) > 0 else 0
        
        # 日次リターンがある場合
        if "日次損益" in self.pnl_summary.columns:
            daily_returns = self.pnl_summary["日次損益"]
            
            # 基本統計
            result["日次平均リターン"] = daily_returns.mean()
            result["日次リターン標準偏差"] = daily_returns.std()
            result["勝ち日数"] = sum(daily_returns > 0)
            result["負け日数"] = sum(daily_returns <= 0)
            result["日次勝率"] = sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0
            
            # カルマー比率の計算
            max_dd_pct = self._calculate_max_drawdown()
            if max_dd_pct != 0:
                annualized_return = daily_returns.mean() * 252
                result["カルマー比率"] = annualized_return / abs(max_dd_pct)
        
        return result
    
    def _calculate_max_drawdown(self) -> float:
        """最大ドローダウンを計算"""
        if self.pnl_summary.empty or "累積損益" not in self.pnl_summary.columns:
            return 0.0
            
        equity = self.pnl_summary["累積損益"]
        max_dd = 0.0
        
        try:
            peak = equity[0]
            for value in equity:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
        except Exception as e:
            self.logger.error(f"最大ドローダウン計算エラー: {str(e)}")
            return 0.0
            
        return max_dd * 100  # パーセント表示に変換
    
    def _calculate_streaks(self) -> Dict[str, float]:
        """連勝/連敗を計算"""
        result = {
            "最大連勝": 0,
            "最大連敗": 0,
            "平均連勝": 0,
            "平均連敗": 0
        }
        
        if self.trades.empty or "損益率" not in self.trades.columns:
            return result
            
        try:
            # 勝ちトレード = 1, 負けトレード = -1 に変換
            wins_losses = (self.trades["損益率"] > 0).astype(int).replace(0, -1)
            
            # 連勝/連敗の計算
            current_streak = 0
            current_type = 0
            win_streaks = []
            loss_streaks = []
            
            for result in wins_losses:
                if result == current_type or current_type == 0:
                    current_type = result
                    current_streak += 1
                else:
                    if current_type == 1:
                        win_streaks.append(current_streak)
                    else:
                        loss_streaks.append(current_streak)
                    current_streak = 1
                    current_type = result
            
            # 最後のストリークを追加
            if current_streak > 0:
                if current_type == 1:
                    win_streaks.append(current_streak)
                else:
                    loss_streaks.append(current_streak)
            
            # 結果の計算
            if win_streaks:
                result["最大連勝"] = max(win_streaks)
                result["平均連勝"] = sum(win_streaks) / len(win_streaks)
            
            if loss_streaks:
                result["最大連敗"] = max(loss_streaks)
                result["平均連敗"] = sum(loss_streaks) / len(loss_streaks)
                
        except Exception as e:
            self.logger.error(f"連勝/連敗計算エラー: {str(e)}")
            
        return result
    
    def _convert_for_json(self, obj):
        """JSON変換のためのオブジェクト変換"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        else:
            return obj
            
    def _generate_html_report(self, output_dir: str, results: Dict, timestamp: str) -> str:
        """HTML形式のレポートを生成"""
        html_file = os.path.join(output_dir, f"{self.strategy_name}_report_{timestamp}.html")
        
        # レポートの生成
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{self.strategy_name} トレード分析レポート</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    h1, h2, h3 {{
                        color: #2c3e50;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    tr:nth-child(even) {{
                        background-color: #f9f9f9;
                    }}
                    .stats-container {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    .stat-box {{
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 15px;
                        min-width: 200px;
                        flex: 1;
                    }}
                    .stat-box h3 {{
                        margin-top: 0;
                        border-bottom: 1px solid #eee;
                        padding-bottom: 8px;
                    }}
                    .good {{
                        color: green;
                    }}
                    .bad {{
                        color: red;
                    }}
                    .neutral {{
                        color: orange;
                    }}
                    .parameter-table {{
                        margin-top: 20px;
                        max-width: 600px;
                    }}
                </style>
            </head>
            <body>
                <h1>{self.strategy_name} トレード分析レポート</h1>
                <p>作成日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>
                
                <h2>戦略パラメータ</h2>
                <table class="parameter-table">
                    <tr>
                        <th>パラメータ</th>
                        <th>値</th>
                    </tr>
            """)
            
            # パラメータの出力
            for key, value in self.parameters.items():
                f.write(f"<tr><td>{key}</td><td>{value}</td></tr>\n")
                
            f.write("</table>\n")
            
            # 基本統計情報
            basic_stats = results.get("basic_stats", {})
            if basic_stats:
                f.write("""
                <h2>基本パフォーマンス指標</h2>
                <div class="stats-container">
                """)
                
                # 主要指標ボックス
                f.write("""
                <div class="stat-box">
                    <h3>主要指標</h3>
                    <table>
                """)
                
                # 累積リターン
                cum_return = basic_stats.get("累積リターン", 0)
                return_class = "good" if cum_return > 0 else "bad"
                f.write(f"<tr><td>累積リターン</td><td class='{return_class}'>{cum_return:.2f}%</td></tr>\n")
                
                # 勝率
                win_rate = basic_stats.get("勝率", 0) * 100
                win_rate_class = "good" if win_rate > 55 else ("neutral" if win_rate > 45 else "bad")
                f.write(f"<tr><td>勝率</td><td class='{win_rate_class}'>{win_rate:.1f}%</td></tr>\n")
                
                # プロフィットファクター
                pf = basic_stats.get("プロフィットファクター", 0)
                pf_class = "good" if pf > 1.5 else ("neutral" if pf > 1 else "bad")
                f.write(f"<tr><td>プロフィットファクター</td><td class='{pf_class}'>{pf:.2f}</td></tr>\n")
                
                # 最大ドローダウン
                max_dd = basic_stats.get("最大ドローダウン", 0)
                max_dd_class = "good" if max_dd < 10 else ("neutral" if max_dd < 20 else "bad")
                f.write(f"<tr><td>最大ドローダウン</td><td class='{max_dd_class}'>{max_dd:.2f}%</td></tr>\n")
                
                f.write("</table></div>\n")
                
                # トレード統計ボックス
                f.write("""
                <div class="stat-box">
                    <h3>トレード統計</h3>
                    <table>
                """)
                
                # トレード数
                f.write(f"<tr><td>総トレード数</td><td>{basic_stats.get('トレード総数', 0)}</td></tr>\n")
                f.write(f"<tr><td>勝ちトレード</td><td>{basic_stats.get('勝ちトレード数', 0)}</td></tr>\n")
                f.write(f"<tr><td>負けトレード</td><td>{basic_stats.get('負けトレード数', 0)}</td></tr>\n")
                
                # 平均リターン
                avg_return = basic_stats.get("平均リターン", 0) * 100
                avg_return_class = "good" if avg_return > 0 else "bad"
                f.write(f"<tr><td>平均リターン</td><td class='{avg_return_class}'>{avg_return:.2f}%</td></tr>\n")
                
                # 平均勝ちリターン
                avg_win = basic_stats.get("平均勝ちリターン", 0) * 100
                f.write(f"<tr><td>平均勝ちリターン</td><td class='good'>{avg_win:.2f}%</td></tr>\n")
                
                # 平均負けリターン
                avg_loss = basic_stats.get("平均負けリターン", 0) * 100
                f.write(f"<tr><td>平均負けリターン</td><td class='bad'>{avg_loss:.2f}%</td></tr>\n")
                
                f.write("</table></div>\n")
                
                # 保有期間統計（あれば）
                if "平均保有期間" in basic_stats:
                    f.write("""
                    <div class="stat-box">
                        <h3>保有期間</h3>
                        <table>
                    """)
                    
                    f.write(f"<tr><td>平均保有期間</td><td>{basic_stats.get('平均保有期間', 0):.1f}日</td></tr>\n")
                    f.write(f"<tr><td>最長保有期間</td><td>{basic_stats.get('最長保有期間', 0)}日</td></tr>\n")
                    f.write(f"<tr><td>最短保有期間</td><td>{basic_stats.get('最短保有期間', 0)}日</td></tr>\n")
                    
                    f.write("</table></div>\n")
                
                f.write("</div>\n")  # stats-container終了
            
            # トレード特性分析
            trade_char = results.get("trade_characteristics", {})
            if trade_char:
                f.write("<h2>トレード特性分析</h2>\n")
                
                # 連勝・連敗情報
                if "最大連勝" in trade_char:
                    f.write("""
                    <div class="stats-container">
                        <div class="stat-box">
                            <h3>連勝・連敗</h3>
                            <table>
                    """)
                    
                    f.write(f"<tr><td>最大連勝</td><td class='good'>{trade_char.get('最大連勝', 0)}</td></tr>\n")
                    f.write(f"<tr><td>最大連敗</td><td class='bad'>{trade_char.get('最大連敗', 0)}</td></tr>\n")
                    f.write(f"<tr><td>平均連勝</td><td>{trade_char.get('平均連勝', 0):.1f}</td></tr>\n")
                    f.write(f"<tr><td>平均連敗</td><td>{trade_char.get('平均連敗', 0):.1f}</td></tr>\n")
                    
                    f.write("</table></div>\n")
                
                # 曜日別パフォーマンス
                if "曜日別パフォーマンス" in trade_char:
                    weekday_perf = trade_char["曜日別パフォーマンス"]
                    
                    f.write("""
                        <div class="stat-box">
                            <h3>曜日別パフォーマンス</h3>
                            <table>
                                <tr>
                                    <th>曜日</th>
                                    <th>平均リターン</th>
                                    <th>取引数</th>
                                </tr>
                    """)
                    
                    for day, data in weekday_perf.items():
                        if isinstance(data, dict):
                            mean_return = data.get("mean", 0) * 100
                            count = data.get("count", 0)
                            return_class = "good" if mean_return > 0 else "bad"
                            f.write(f"<tr><td>{day}</td><td class='{return_class}'>{mean_return:.2f}%</td><td>{count}</td></tr>\n")
                    
                    f.write("</table></div>\n")
                
                f.write("</div>\n")  # stats-container終了
            
            # 時系列分析
            time_series = results.get("time_series", {})
            if time_series:
                f.write("<h2>時系列分析</h2>\n")
                
                f.write("""
                <div class="stats-container">
                    <div class="stat-box">
                        <h3>月次パフォーマンス</h3>
                        <table>
                """)
                
                # 月次リターン
                monthly_return = time_series.get("月次平均リターン", 0) * 100
                monthly_return_class = "good" if monthly_return > 0 else "bad"
                f.write(f"<tr><td>月次平均リターン</td><td class='{monthly_return_class}'>{monthly_return:.2f}%</td></tr>\n")
                
                # 月次標準偏差
                f.write(f"<tr><td>月次標準偏差</td><td>{time_series.get('月次リターン標準偏差', 0) * 100:.2f}%</td></tr>\n")
                
                # 勝ち月/負け月
                win_months = time_series.get("勝ち月数", 0)
                loss_months = time_series.get("負け月数", 0)
                monthly_win_rate = time_series.get("月次勝率", 0) * 100
                monthly_win_rate_class = "good" if monthly_win_rate > 60 else ("neutral" if monthly_win_rate > 50 else "bad")
                
                f.write(f"<tr><td>勝ち月/負け月</td><td>{win_months}/{loss_months}</td></tr>\n")
                f.write(f"<tr><td>月次勝率</td><td class='{monthly_win_rate_class}'>{monthly_win_rate:.1f}%</td></tr>\n")
                
                f.write("</table></div>\n")
                
                # 日次パフォーマンス
                f.write("""
                    <div class="stat-box">
                        <h3>日次パフォーマンス</h3>
                        <table>
                """)
                
                # 日次リターン
                daily_return = time_series.get("日次平均リターン", 0) * 100
                daily_return_class = "good" if daily_return > 0 else "bad"
                f.write(f"<tr><td>日次平均リターン</td><td class='{daily_return_class}'>{daily_return:.3f}%</td></tr>\n")
                
                # 日次標準偏差
                f.write(f"<tr><td>日次標準偏差</td><td>{time_series.get('日次リターン標準偏差', 0) * 100:.3f}%</td></tr>\n")
                
                # カルマー比率
                calmar = time_series.get("カルマー比率", 0)
                calmar_class = "good" if calmar > 1 else ("neutral" if calmar > 0.5 else "bad")
                f.write(f"<tr><td>カルマー比率</td><td class='{calmar_class}'>{calmar:.2f}</td></tr>\n")
                
                # 勝ち日/負け日
                win_days = time_series.get("勝ち日数", 0)
                loss_days = time_series.get("負け日数", 0)
                daily_win_rate = time_series.get("日次勝率", 0) * 100
                daily_win_rate_class = "good" if daily_win_rate > 55 else ("neutral" if daily_win_rate > 50 else "bad")
                
                f.write(f"<tr><td>勝ち日/負け日</td><td>{win_days}/{loss_days}</td></tr>\n")
                f.write(f"<tr><td>日次勝率</td><td class='{daily_win_rate_class}'>{daily_win_rate:.1f}%</td></tr>\n")
                
                f.write("</table></div>\n")
                
                f.write("</div>\n")  # stats-container終了
            
            f.write("""
                <h2>レポートまとめ</h2>
                <p>このレポートは、戦略のバックテスト結果の詳細分析を提供します。主要なパフォーマンス指標をもとに、戦略の長所と短所を評価し、改善点を特定するのに役立てることができます。</p>
                
                <h3>評価基準</h3>
                <ul>
                    <li><span class="good">緑色</span>: 優れたパフォーマンスを示している指標</li>
                    <li><span class="neutral">オレンジ色</span>: 普通のパフォーマンスを示している指標</li>
                    <li><span class="bad">赤色</span>: 改善が必要なパフォーマンスを示している指標</li>
                </ul>
            """)
            
            # レポート生成日時
            f.write(f"<p><small>レポート生成: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</small></p>\n")
            
            f.write("</body>\n</html>")
        
        return html_file

"""
最適化ダッシュボードを作成するためのスタンドアロンスクリプト
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import argparse
import json
import webbrowser
import logging
from typing import Dict, List, Any, Optional, Tuple

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("optimization_dashboard")


def create_optimization_dashboard(results_file: str, output_dir: str = None) -> str:
    """
    最適化結果のCSVファイルからダッシュボードを生成
    
    Parameters:
        results_file (str): 最適化結果のCSVファイルパス
        output_dir (str, optional): 出力ディレクトリ
        
    Returns:
        str: 生成されたHTMLダッシュボードのファイルパス
    """
    # 出力ディレクトリのデフォルト設定
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    # 出力ディレクトリがなければ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイル名から戦略名を推測
    strategy_name = os.path.basename(results_file).split('_')[0]
    
    # CSVファイルを読み込む
    try:
        results_df = pd.read_csv(results_file)
        logger.info(f"{len(results_df)}件の最適化結果データを読み込みました")
    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {str(e)}")
        return ""
    
    if results_df.empty:
        logger.warning("最適化結果データが空です")
        return ""
    
    # タイムスタンプの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # HTMLダッシュボードファイル名
    html_file = os.path.join(output_dir, f"{strategy_name}_optimization_dashboard_{timestamp}.html")
    
    # グラフのベースディレクトリ（HTML内で使用するパス）
    graphs_dir = f"graphs_{timestamp}"
    full_graphs_dir = os.path.join(output_dir, graphs_dir)
    os.makedirs(full_graphs_dir, exist_ok=True)
    
    # パラメータリストの作成
    param_columns = [col for col in results_df.columns if col not in ['score', 'error', 'trades']]
    
    # 各パラメータの影響度グラフを作成
    param_importance_data = {}
    param_graph_files = []
    
    for param in param_columns:
        if len(results_df[param].unique()) <= 15:  # 値の異なりが15以下のパラメータのみ可視化
            try:
                # 箱ひげ図の作成
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=param, y='score', data=results_df[results_df['score'] > -np.inf])
                plt.title(f"{param}の値とスコアの関係", fontsize=14)
                plt.xlabel(param, fontsize=12)
                plt.ylabel('スコア', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # グラフファイルの保存
                graph_file = f"{param}_impact.png"
                plt.savefig(os.path.join(full_graphs_dir, graph_file), dpi=100)
                plt.close()
                
                # グラフファイルリストに追加
                param_graph_files.append((param, os.path.join(graphs_dir, graph_file)))
                
                # パラメータの影響度データ計算
                param_impact = results_df.groupby(param)['score'].mean().reset_index()
                best_value = param_impact.loc[param_impact['score'].idxmax()][param]
                param_importance_data[param] = {
                    'best_value': best_value,
                    'unique_values': len(results_df[param].unique()),
                    'impact': param_impact['score'].max() - param_impact['score'].min()
                }
                
                logger.info(f"パラメータ {param} のグラフを生成しました")
                
            except Exception as e:
                logger.error(f"パラメータ {param} のグラフ生成エラー: {str(e)}")
    
    # パラメータの影響度でソート
    sorted_params = sorted(param_importance_data.items(), key=lambda x: x[1]['impact'], reverse=True)
    
    # スコア分布のヒストグラム
    try:
        plt.figure(figsize=(10, 6))
        valid_scores = results_df[results_df['score'] > -np.inf]['score']
        sns.histplot(valid_scores, kde=True, bins=20)
        plt.title('最適化スコア分布', fontsize=14)
        plt.xlabel('スコア', fontsize=12)
        plt.ylabel('頻度', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        score_hist_file = 'score_distribution.png'
        plt.savefig(os.path.join(full_graphs_dir, score_hist_file), dpi=100)
        plt.close()
        
        logger.info("スコア分布ヒストグラムを生成しました")
    except Exception as e:
        logger.error(f"スコア分布ヒストグラム生成エラー: {str(e)}")
        score_hist_file = None
    
    # 最適パラメータの取得
    best_params = {}
    if not results_df.empty:
        best_row = results_df.nlargest(1, 'score').iloc[0]
        for param in param_columns:
            best_params[param] = best_row[param]
    
    # HTMLダッシュボードの作成
    try:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{strategy_name} 最適化ダッシュボード</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    h1, h2, h3 {{
                        color: #2c3e50;
                    }}
                    .dashboard-header {{
                        background-color: #2c3e50;
                        color: white;
                        padding: 20px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                        text-align: center;
                    }}
                    .dashboard-header h1 {{
                        color: white;
                        margin: 0;
                    }}
                    .dashboard-container {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    .dashboard-panel {{
                        background-color: white;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        padding: 20px;
                        margin-bottom: 20px;
                    }}
                    .full-width {{
                        width: 100%;
                    }}
                    .half-width {{
                        width: calc(50% - 10px);
                    }}
                    @media (max-width: 768px) {{
                        .half-width {{
                            width: 100%;
                        }}
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 12px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    tr:nth-child(even) {{
                        background-color: #f9f9f9;
                    }}
                    .param-impact {{
                        background-color: #e8f4f8;
                        border-left: 4px solid #3498db;
                        padding: 10px;
                        margin-bottom: 10px;
                    }}
                    .graph-container {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                        justify-content: center;
                    }}
                    .graph-item {{
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 10px;
                        background-color: white;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        max-width: 500px;
                        margin-bottom: 10px;
                    }}
                    .graph-item img {{
                        max-width: 100%;
                        height: auto;
                    }}
                    .graph-item h4 {{
                        margin-top: 0;
                        margin-bottom: 10px;
                        text-align: center;
                        color: #2c3e50;
                    }}
                </style>
            </head>
            <body>
                <div class="dashboard-header">
                    <h1>{strategy_name} 最適化ダッシュボード</h1>
                    <p>生成日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>
                </div>
                
                <div class="dashboard-container">
                    <div class="dashboard-panel half-width">
                        <h2>最適化サマリー</h2>
                        <table>
                            <tr>
                                <td>最適化結果数</td>
                                <td>{len(results_df)}</td>
                            </tr>
                            <tr>
                                <td>有効なスコア数</td>
                                <td>{len(results_df[results_df['score'] > -np.inf])}</td>
                            </tr>
                            <tr>
                                <td>最高スコア</td>
                                <td>{results_df['score'].max():.4f}</td>
                            </tr>
                            <tr>
                                <td>平均スコア</td>
                                <td>{results_df[results_df['score'] > -np.inf]['score'].mean():.4f}</td>
                            </tr>
                            <tr>
                                <td>パラメータ組み合わせ数</td>
                                <td>{len(param_columns)}</td>
                            </tr>
                        </table>
                    </div>
            """)
            
            # スコア分布ヒストグラム
            if score_hist_file:
                f.write(f"""
                    <div class="dashboard-panel half-width">
                        <h2>スコア分布</h2>
                        <img src="{os.path.join(graphs_dir, score_hist_file)}" alt="Score Distribution" style="width:100%;">
                    </div>
                """)
            
            # 最適パラメータ
            f.write("""
                </div>
                
                <div class="dashboard-panel full-width">
                    <h2>最適パラメータ</h2>
                    <table>
                        <tr>
                            <th>パラメータ</th>
                            <th>最適値</th>
                        </tr>
            """)
            
            for param, value in best_params.items():
                f.write(f"<tr><td>{param}</td><td>{value}</td></tr>\n")
            
            f.write("""
                    </table>
                </div>
                
            """)
            
            # パラメータの重要度
            f.write("""
                <div class="dashboard-panel full-width">
                    <h2>パラメータ影響度ランキング</h2>
                    <p>各パラメータがスコアに与える影響度を分析した結果です。</p>
            """)
            
            if sorted_params:
                f.write("""
                    <table>
                        <tr>
                            <th>ランク</th>
                            <th>パラメータ</th>
                            <th>影響度スコア</th>
                            <th>最適値</th>
                            <th>異なる値の数</th>
                        </tr>
                """)
                
                for i, (param, data) in enumerate(sorted_params, 1):
                    f.write(f"""
                        <tr>
                            <td>{i}</td>
                            <td>{param}</td>
                            <td>{data['impact']:.4f}</td>
                            <td>{data['best_value']}</td>
                            <td>{data['unique_values']}</td>
                        </tr>
                    """)
                
                f.write("</table>")
                
                # 上位3つのパラメータについてのコメント
                f.write("<h3>主要パラメータの分析</h3>")
                for i, (param, data) in enumerate(sorted_params[:3], 1):
                    f.write(f"""
                        <div class="param-impact">
                            <h4>{i}. {param}</h4>
                            <p>影響度スコア: <strong>{data['impact']:.4f}</strong></p>
                            <p>最適値: <strong>{data['best_value']}</strong></p>
                            <p>このパラメータは全体のパフォーマンスに大きな影響を与えています。
                            最適化プロセスを改善するため、このパラメータに特に注意を払うことをお勧めします。</p>
                        </div>
                    """)
            else:
                f.write("<p>パラメータ影響度の分析に十分なデータがありません。</p>")
            
            f.write("</div>")
            
            # パラメータ影響グラフ
            if param_graph_files:
                f.write("""
                    <div class="dashboard-panel full-width">
                        <h2>パラメータ影響グラフ</h2>
                        <p>各パラメータの値とスコアの関係を可視化したグラフです。</p>
                        <div class="graph-container">
                """)
                
                for param, graph_file in param_graph_files:
                    f.write(f"""
                        <div class="graph-item">
                            <h4>{param}</h4>
                            <img src="{graph_file}" alt="{param} Impact Graph">
                        </div>
                    """)
                
                f.write("</div></div>")
            
            # 上位パラメータ組み合わせ
            f.write("""
                <div class="dashboard-panel full-width">
                    <h2>上位パラメータ組み合わせ</h2>
                    <p>スコアが最も高かった上位5つのパラメータ組み合わせです。</p>
                    <table>
                        <tr>
                            <th>ランク</th>
            """)
            
            for param in param_columns:
                f.write(f"<th>{param}</th>")
            
            f.write("<th>スコア</th></tr>")
            
            top_results = results_df.nlargest(5, 'score')
            for i, (_, row) in enumerate(top_results.iterrows(), 1):
                f.write(f"<tr><td>{i}</td>")
                for param in param_columns:
                    f.write(f"<td>{row[param]}</td>")
                f.write(f"<td>{row['score']:.4f}</td></tr>")
            
            f.write("</table></div>")
            
            # 推奨事項
            f.write("""
                <div class="dashboard-panel full-width">
                    <h2>推奨事項</h2>
                    <ul>
            """)
            
            # 影響度の高いパラメータに関する推奨事項
            for param, data in sorted_params[:3]:
                if data['best_value'] in [results_df[param].min(), results_df[param].max()]:
                    f.write(f"<li><strong>{param}</strong>: 探索範囲の拡大を推奨します。最適値が現在の範囲の境界にあります。</li>\n")
                else:
                    f.write(f"<li><strong>{param}</strong>: 最適値 {data['best_value']} の周辺でより細かい探索を検討してください。</li>\n")
            
            # ダッシュボードのフッター
            f.write("""
                    <li>最適パラメータは特定のテストデータに最適化されている可能性があります。他の期間や銘柄でもテストすることをお勧めします。</li>
                    <li>高い影響度を持つパラメータを中心に、より詳細な感度分析を行うことを検討してください。</li>
                </ul>
                </div>
                
                <p><small>このダッシュボードは最適化結果の分析を自動化したものです。戦略の最終判断には人間の判断が必要です。</small></p>
                <p><small>生成日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</small></p>
            </body>
            </html>
            """.format(datetime=datetime, os=os, graphs_dir=graphs_dir))
        
        logger.info(f"最適化ダッシュボードを生成しました: {html_file}")
        return html_file
    
    except Exception as e:
        logger.exception(f"ダッシュボード生成中にエラーが発生: {str(e)}")
        return ""


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='最適化結果からダッシュボードを生成します')
    parser.add_argument('results_file', type=str, help='CSVフォーマットの最適化結果ファイルパス')
    parser.add_argument('--output-dir', type=str, default=None, help='出力ディレクトリ（デフォルトは結果ファイルと同じ場所）')
    parser.add_argument('--open-browser', action='store_true', help='生成後にブラウザで開く')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        logger.error(f"ファイルが見つかりません: {args.results_file}")
        return 1
    
    dashboard_file = create_optimization_dashboard(args.results_file, args.output_dir)
    
    if dashboard_file and args.open_browser:
        # ダッシュボードをブラウザで開く
        webbrowser.open('file://' + os.path.abspath(dashboard_file))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
シンプルExcel出力システム - Phase 2.3 強化版
File: simple_excel_exporter.py
Description: 
  Phase 2.3: データ品質最適化対応版
  - data_extraction_enhancer.py統合
  - 正確な取引データ抽出
  - Excel品質向上

Author: GitHub Copilot  
Created: 2025-09-04
Version: 2.3 (Enhanced Data Quality)

Purpose:
  - 正確なトレード抽出とパフォーマンス計算
  - Excel出力品質向上
  - データ品質検証強化
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import warnings

# 新規追加: データ抽出エンハンサー
from .data_extraction_enhancer import MainDataExtractor, extract_and_analyze_main_data

# 警告を抑制
warnings.filterwarnings('ignore')

class ExcelDataProcessor:
    """Excel出力用データ処理クラス - Phase 2.3拡張"""
    
    def __init__(self):
        self.extractor = MainDataExtractor()
    
    def process_main_data(self, stock_data: pd.DataFrame, ticker: str = "UNKNOWN") -> Dict[str, Any]:
        """
        main.pyデータを正確に処理
        
        Args:
            stock_data: main.pyから渡されるDataFrame
            ticker: 銘柄コード
            
        Returns:
            Dict: 処理済みデータ
        """
        if stock_data is None or stock_data.empty:
            return self._get_empty_data(ticker)
        
        # データ抽出エンハンサーで処理
        analyzed_data = extract_and_analyze_main_data(stock_data, ticker)
        
        # Excel用に再構造化
        return {
            'metadata': {
                'ticker': analyzed_data['ticker'],
                'analysis_date': analyzed_data['extraction_timestamp'],
                'data_quality': analyzed_data['data_quality'],
                'period_start': analyzed_data['period']['start_date'],
                'period_end': analyzed_data['period']['end_date'],
                'total_days': analyzed_data['period']['total_days']
            },
            'summary': analyzed_data['performance'],
            'trades': analyzed_data['trades'],
            'raw_data': stock_data
        }
    
    def _get_empty_data(self, ticker: str) -> Dict[str, Any]:
        """空データの場合のデフォルト構造"""
        return {
            'metadata': {
                'ticker': ticker,
                'analysis_date': datetime.now(),
                'data_quality': 'empty',
                'period_start': datetime.now(),
                'period_end': datetime.now(),
                'total_days': 0
            },
            'summary': {
                'final_portfolio_value': 1000000.0,
                'total_return': 0.0,
                'total_pnl': 0.0,
                'num_trades': 0,
                'win_rate': 0.0
            },
            'trades': [],
            'raw_data': pd.DataFrame()
        }

def save_backtest_results_simple(
    stock_data: Union[Dict[str, Any], pd.DataFrame, Any] = None,
    results: Union[Dict[str, Any], Any] = None,
    ticker: str = "UNKNOWN",
    filename: Optional[str] = None,
    output_dir: str = "backtest_results/improved_results"
) -> str:
    """
    バックテスト結果をExcel形式で保存 - Phase 2.3強化版
    
    データ抽出エンハンサーを使用して正確な解析結果を出力
    
    Args:
        stock_data: 株価DataFrame（main.pyから渡される）
        results: バックテスト結果データ（互換性のため）
        ticker: 銘柄コード
        filename: 出力ファイル名（Noneの場合は自動生成）
        output_dir: 出力ディレクトリ
    
    Returns:
        str: 保存されたファイルパス
    
    Raises:
        Exception: Excel出力に失敗した場合（フォールバック処理実行）
    """
    
    try:
        # 1. ファイル名生成
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_backtest_{ticker}_{timestamp}.xlsx"
        
        # 2. 出力ディレクトリ確保
        if not output_dir:
            output_dir = "backtest_results/improved_results"  # デフォルト値を設定
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # 3. データ処理（Phase 2.3新機能）
        processor = ExcelDataProcessor()
        
        # stock_dataを優先して使用
        if isinstance(stock_data, pd.DataFrame) and not stock_data.empty:
            normalized_data = processor.process_main_data(stock_data, ticker)
        else:
            # フォールバック: 従来の処理
            target_data = results if results is not None else stock_data
            normalized_data = _normalize_results_data(target_data)
        
        # 4. Excel出力実行
        _create_excel_output(normalized_data, filepath)
        
        print(f"✅ Phase 2.3 Enhanced Excel出力完了: {filepath}")
        print(f"📊 最終ポートフォリオ価値: {normalized_data['summary'].get('final_portfolio_value', 0):,.0f}円")
        print(f"🎯 総取引数: {normalized_data['summary'].get('num_trades', 0)}件")
        return filepath
        
    except Exception as e:
        print(f"⚠️ Excel出力エラー: {e}")
        # フォールバック: CSV出力
        target_data = stock_data if stock_data is not None else results
        fallback_path = _create_fallback_output(target_data, output_dir, filename)
        print(f"📄 フォールバックCSV出力: {fallback_path}")
        return fallback_path


def _normalize_results_data(results: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    結果データを正規化して統一形式に変換
    
    Args:
        results: 入力データ（様々な形式に対応）
    
    Returns:
        Dict[str, Any]: 正規化されたデータ
    """
    
    normalized = {
        'metadata': {},
        'summary': {},
        'trades': [],
        'daily_pnl': [],
        'switches': []
    }
    
    try:
        # 辞書形式の場合
        if isinstance(results, dict):
            normalized.update(results)
        
        # オブジェクト形式の場合
        elif hasattr(results, '__dict__'):
            normalized.update(vars(results))
        
        # DataFrameの場合（main.pyから渡される戦略適用後データ）
        elif hasattr(results, 'to_dict'):
            import pandas as pd
            
            if isinstance(results, pd.DataFrame):
                # DataFrameを辞書形式に変換
                normalized['data'] = results.to_dict('records')
                normalized['metadata'] = {
                    'total_rows': len(results),
                    'columns': list(results.columns),
                    'date_range': {
                        'start': str(results.index[0]) if not results.empty else 'N/A',
                        'end': str(results.index[-1]) if not results.empty else 'N/A'
                    }
                }
                
                # Entry_SignalとExit_Signalからトレード抽出
                if 'Entry_Signal' in results.columns and 'Exit_Signal' in results.columns:
                    trades = _extract_trades_from_signals(results)
                    normalized['trades'] = trades
                    
                    # トレードからサマリー計算
                    summary = _calculate_summary_from_trades(trades, results)
                    normalized['summary'] = summary
                
                # 日次PnL計算（可能な場合）
                if 'Close' in results.columns:
                    daily_pnl = _calculate_daily_pnl_from_dataframe(results)
                    normalized['daily_pnl'] = daily_pnl
            else:
                normalized['data'] = results.to_dict()
        
        # その他の場合
        else:
            normalized['raw_data'] = str(results)
            
        # メタデータ補完
        if 'timestamp' not in normalized['metadata']:
            normalized['metadata']['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        if 'version' not in normalized['metadata']:
            normalized['metadata']['version'] = "simple_excel_exporter_v1.0"
            
    except Exception as e:
        print(f"⚠️ データ正規化エラー: {e}")
        normalized['raw_data'] = str(results)
        normalized['error'] = str(e)
    
    return normalized


def _create_excel_output(data: Dict[str, Any], filepath: str) -> None:
    """
    正規化されたデータからExcelファイルを作成
    
    Args:
        data: 正規化されたデータ
        filepath: 出力ファイルパス
    """
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        
        # 1. サマリーシート作成
        summary_data = _create_summary_data(data)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='サマリー', index=False)
        
        # 2. 取引履歴シート作成（データがある場合）
        if data.get('trades') and len(data['trades']) > 0:
            trades_df = _create_trades_dataframe(data['trades'])
            trades_df.to_excel(writer, sheet_name='取引履歴', index=False)
        
        # 3. 日次損益シート作成（データがある場合）
        if data.get('daily_pnl') and len(data['daily_pnl']) > 0:
            pnl_df = _create_pnl_dataframe(data['daily_pnl'])
            pnl_df.to_excel(writer, sheet_name='日次損益', index=False)
        
        # 4. メタデータシート作成
        metadata_data = [
            ['項目', '値'],
            ['出力日時', data['metadata'].get('timestamp', 'N/A')],
            ['バージョン', data['metadata'].get('version', 'N/A')],
            ['データソース', 'DSSMS Backtester'],
            ['処理ステータス', '正常']
        ]
        metadata_df = pd.DataFrame(metadata_data[1:], columns=metadata_data[0])
        metadata_df.to_excel(writer, sheet_name='メタデータ', index=False)


def _create_summary_data(data: Dict[str, Any]) -> List[List[Any]]:
    """
    サマリーデータ作成
    
    Args:
        data: 正規化されたデータ
    
    Returns:
        List[List[Any]]: サマリーテーブルデータ
    """
    
    summary = data.get('summary', {})
    
    # 基本的なサマリー項目
    summary_items = [
        ['項目', '値'],
        ['実行日時', data['metadata'].get('timestamp', 'N/A')],
        ['バックテスト期間', summary.get('period', 'N/A')],
        ['初期資本', f"{summary.get('initial_capital', 1000000):,.0f}円"],
        ['最終ポートフォリオ価値', f"{summary.get('final_value', 0):,.0f}円"],
        ['総リターン', f"{summary.get('total_return', 0):.2%}"],
        ['年率リターン', f"{summary.get('annual_return', 0):.2%}"],
        ['最大ドローダウン', f"{summary.get('max_drawdown', 0):.2%}"],
        ['シャープレシオ', f"{summary.get('sharpe_ratio', 0):.3f}"],
        ['', ''],  # 空行
        ['DSSMS固有指標', ''],
        ['銘柄切替回数', f"{summary.get('switch_count', 0)}回"],
        ['切替成功率', f"{summary.get('switch_success_rate', 0):.2%}"],
        ['平均保有期間', f"{summary.get('avg_holding_period', 0):.1f}時間"],
        ['切替コスト合計', f"{summary.get('switch_cost', 0):,.0f}円"]
    ]
    
    return summary_items


def _create_trades_dataframe(trades_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    取引履歴DataFrameを作成
    
    Args:
        trades_data: 取引データのリスト
    
    Returns:
        pd.DataFrame: 取引履歴DataFrame
    """
    
    if not trades_data:
        return pd.DataFrame(columns=['日付', '銘柄', '売買', '数量', '価格', '金額', '手数料'])
    
    # データ形式の正規化
    normalized_trades = []
    for trade in trades_data:
        if isinstance(trade, dict):
            normalized_trades.append(trade)
        else:
            # オブジェクトの場合は辞書に変換
            normalized_trades.append(vars(trade) if hasattr(trade, '__dict__') else {'data': str(trade)})
    
    df = pd.DataFrame(normalized_trades)
    
    # 基本列の確保
    required_columns = ['日付', '銘柄', '売買', '数量', '価格', '金額', '手数料']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 'N/A'
    
    return df[required_columns]


def _create_pnl_dataframe(pnl_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    日次損益DataFrameを作成
    
    Args:
        pnl_data: 日次損益データのリスト
    
    Returns:
        pd.DataFrame: 日次損益DataFrame
    """
    
    if not pnl_data:
        return pd.DataFrame(columns=['日付', '日次損益', '累積損益', '保有銘柄'])
    
    df = pd.DataFrame(pnl_data)
    
    # 基本列の確保
    required_columns = ['日付', '日次損益', '累積損益', '保有銘柄']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 'N/A'
    
    return df[required_columns]


def _create_fallback_output(results: Any, output_dir: str, filename: Optional[str]) -> str:
    """
    Excel出力失敗時のフォールバックCSV出力
    
    Args:
        results: 結果データ
        output_dir: 出力ディレクトリ
        filename: ファイル名
    
    Returns:
        str: フォールバックファイルパス
    """
    
    try:
        # output_dirがNoneの場合のデフォルト値設定
        if not output_dir:
            output_dir = "backtest_results/improved_results"
        
        # CSV用ファイル名生成
        if filename:
            csv_filename = filename.replace('.xlsx', '_fallback.csv')
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"simple_backtest_fallback_{timestamp}.csv"
        
        # ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        csv_filepath = os.path.join(output_dir, csv_filename)
        
        # 基本的なCSVデータ作成
        fallback_data = [
            ['項目', '値'],
            ['出力日時', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ['データ型', str(type(results))],
            ['データ内容', str(results)[:1000]],  # 最初の1000文字
            ['ステータス', 'フォールバック出力'],
            ['注意', 'Excel出力に失敗したため、CSV形式で保存されました']
        ]
        
        df = pd.DataFrame(fallback_data[1:], columns=fallback_data[0])
        df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
        
        return csv_filepath
        
    except Exception as e:
        # 最終フォールバック: テキストファイル
        if not output_dir:
            output_dir = "output"
        
        txt_filename = f"simple_backtest_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs(output_dir, exist_ok=True)
        txt_filepath = os.path.join(output_dir, txt_filename)
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(f"緊急出力ファイル\n")
            f.write(f"作成日時: {datetime.now()}\n")
            f.write(f"エラー: {e}\n")
            f.write(f"データ: {str(results)}\n")
        
        return txt_filepath


# 追加のユーティリティ関数

def validate_results_data(results: Any) -> bool:
    """
    結果データの妥当性チェック
    
    Args:
        results: チェック対象データ
    
    Returns:
        bool: 妥当性（True=有効, False=無効）
    """
    
    if results is None:
        return False
    
    # 基本的な妥当性チェック
    try:
        if isinstance(results, (dict, list)):
            return len(results) > 0
        elif hasattr(results, '__dict__'):
            return True
        elif hasattr(results, '__len__'):
            return len(results) > 0
        else:
            return True
    except:
        return False


def get_summary_from_results(results: Any) -> Dict[str, Any]:
    """
    結果データから基本サマリーを抽出
    
    Args:
        results: 結果データ
    
    Returns:
        Dict[str, Any]: 抽出されたサマリー
    """
    
    summary = {}
    
    try:
        if isinstance(results, dict):
            # 一般的なキーを探索
            summary.update({
                'total_return': results.get('total_return', results.get('リターン', 0)),
                'final_value': results.get('final_value', results.get('最終価値', 1000000)),
                'switch_count': results.get('switch_count', results.get('切替回数', 0)),
                'max_drawdown': results.get('max_drawdown', results.get('最大ドローダウン', 0))
            })
        
        # デフォルト値で補完
        default_summary = {
            'initial_capital': 1000000,
            'final_value': 1000000,
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'switch_count': 0,
            'switch_success_rate': 0.0,
            'avg_holding_period': 0.0,
            'switch_cost': 0.0
        }
        
        for key, default_value in default_summary.items():
            if key not in summary:
                summary[key] = default_value
                
    except Exception as e:
        print(f"⚠️ サマリー抽出エラー: {e}")
        summary = default_summary
    
    return summary


# メイン実行部分（テスト用）
if __name__ == "__main__":
    # テストデータ
    test_results = {
        'summary': {
            'initial_capital': 1000000,
            'final_value': 1150000,
            'total_return': 0.15,
            'switch_count': 5
        },
        'trades': [
            {'日付': '2023-01-01', '銘柄': '7203', '売買': '買', '数量': 100, '価格': 1000, '金額': 100000}
        ]
    }
    
    # テスト実行
    output_path = save_backtest_results_simple(test_results)
    print(f"テスト出力完了: {output_path}")


def _extract_trades_from_signals(df):
    """
    DataFrameのEntry_SignalとExit_Signalからトレード情報を抽出
    
    Args:
        df: 戦略シグナルが追加されたDataFrame
    
    Returns:
        List[Dict]: トレード履歴のリスト
    """
    import pandas as pd
    
    trades = []
    
    try:
        if 'Entry_Signal' not in df.columns or 'Exit_Signal' not in df.columns:
            return trades
        
        # エントリーシグナルのある日付を抽出
        entry_dates = df[df['Entry_Signal'] == 1].index.tolist()
        exit_dates = df[df['Exit_Signal'] == 1].index.tolist()
        
        # エントリーとエグジットのペアを作成
        for i, entry_date in enumerate(entry_dates):
            # 対応するエグジット日付を検索
            exit_date = None
            entry_price = df.loc[entry_date, 'Close'] if 'Close' in df.columns else 0
            
            # エントリー日より後のエグジット日を検索
            for exit_dt in exit_dates:
                if exit_dt > entry_date:
                    exit_date = exit_dt
                    break
            
            if exit_date:
                exit_price = df.loc[exit_date, 'Close'] if 'Close' in df.columns else 0
                pnl = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
                
                trades.append({
                    'trade_id': i + 1,
                    'entry_date': str(entry_date),
                    'exit_date': str(exit_date),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_amount': (exit_price - entry_price) * 100,  # 仮に100株とする
                    'holding_days': (exit_date - entry_date).days if hasattr((exit_date - entry_date), 'days') else 1
                })
        
    except Exception as e:
        print(f"トレード抽出エラー: {e}")
    
    return trades


def _calculate_summary_from_trades(trades, df):
    """
    トレード履歴からサマリー情報を計算
    
    Args:
        trades: トレード履歴のリスト
        df: 元のDataFrame
    
    Returns:
        Dict: サマリー情報
    """
    
    summary = {
        'total_trades': len(trades),
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0.0,
        'win_rate': 0.0,
        'avg_pnl': 0.0,
        'max_profit': 0.0,
        'max_loss': 0.0,
        'avg_holding_days': 0.0,
        'total_return': 0.0,
        'annual_return': 0.0,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0
    }
    
    try:
        if trades:
            pnls = [trade.get('pnl_amount', 0) for trade in trades]
            summary['total_pnl'] = sum(pnls)
            summary['avg_pnl'] = summary['total_pnl'] / len(trades)
            summary['winning_trades'] = len([pnl for pnl in pnls if pnl > 0])
            summary['losing_trades'] = len([pnl for pnl in pnls if pnl < 0])
            summary['win_rate'] = summary['winning_trades'] / len(trades) * 100
            summary['max_profit'] = max(pnls) if pnls else 0
            summary['max_loss'] = min(pnls) if pnls else 0
            
            holding_days = [trade.get('holding_days', 0) for trade in trades]
            summary['avg_holding_days'] = sum(holding_days) / len(holding_days) if holding_days else 0
            
            # 初期資本を仮定してリターン計算
            initial_capital = 1000000
            summary['total_return'] = summary['total_pnl'] / initial_capital * 100
            
            # 年率リターン（簡易計算）
            if len(df) > 0:
                days = len(df)
                years = days / 252  # 営業日ベース
                if years > 0:
                    summary['annual_return'] = (1 + summary['total_return']/100) ** (1/years) - 1
                    summary['annual_return'] *= 100
    
    except Exception as e:
        print(f"サマリー計算エラー: {e}")
    
    return summary


def _calculate_daily_pnl_from_dataframe(df):
    """
    DataFrameから日次PnLを計算
    
    Args:
        df: 価格データを含むDataFrame
    
    Returns:
        List[Dict]: 日次PnL情報のリスト
    """
    
    daily_pnl = []
    
    try:
        if 'Close' in df.columns and len(df) > 1:
            df_copy = df.copy()
            df_copy['daily_return'] = df_copy['Close'].pct_change()
            df_copy['cumulative_return'] = (1 + df_copy['daily_return']).cumprod() - 1
            
            for date, row in df_copy.iterrows():
                daily_pnl.append({
                    'date': str(date),
                    'close_price': row.get('Close', 0),
                    'daily_return': row.get('daily_return', 0),
                    'cumulative_return': row.get('cumulative_return', 0),
                    'daily_pnl': row.get('daily_return', 0) * 1000000  # 仮に100万円の資本
                })
    
    except Exception as e:
        print(f"日次PnL計算エラー: {e}")
    
    return daily_pnl

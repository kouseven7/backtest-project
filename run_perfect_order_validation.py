"""
パーフェクトオーダー検証スクリプト

6銘柄で固定銘柄モードのDSSMSバックテストを実行し、
パーフェクトオーダー前提の有効性を検証する。

主な機能:
- 6銘柄（8306, 5713, 4063, 4004, 2768, 7013）の自動実行
- 2014-2025年の11年間バックテスト
- GC戦略のみ使用（take_profit=15%, trailing_stop=5%, stop_loss=3%）
- 各銘柄の結果をCSV形式で集計
- パーフェクトオーダー前提の有効性判定

統合コンポーネント:
- src.dssms.dssms_integrated_main: 固定銘柄モードのバックテスト実行
- hierarchical_ranking_system: パーフェクトオーダー判定

セーフティ機能/注意事項:
- 各銘柄の実行時間は約30-60分（11年間のデータ処理）
- エラー発生時も次の銘柄の実行を継続
- 最終結果はdocs/To make profits from system trading/に保存

Author: Backtest Project Team
Created: 2026-01-19
Last Modified: 2026-01-19
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

def run_single_symbol_backtest(symbol: str, start_date: str, end_date: str) -> dict:
    """
    固定銘柄モードでDSSMSバックテストを実行
    
    Args:
        symbol: 銘柄コード（例: 8306.T）
        start_date: 開始日（YYYY-MM-DD）
        end_date: 終了日（YYYY-MM-DD）
    
    Returns:
        dict: 実行結果（成功/失敗、出力ディレクトリ）
    """
    print(f"\n{'='*80}")
    print(f"銘柄 {symbol} のバックテスト開始")
    print(f"期間: {start_date} ~ {end_date}")
    print(f"{'='*80}\n")
    
    cmd = [
        "python", "-m", "src.dssms.dssms_integrated_main",
        "--start-date", start_date,
        "--end-date", end_date,
        "--fixed-symbol", symbol
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='cp932',
            errors='replace',
            timeout=7200  # 2時間タイムアウト
        )
        
        print(f"[STDOUT]\n{result.stdout}")
        
        if result.returncode == 0:
            print(f"\n[SUCCESS] 銘柄 {symbol} のバックテスト成功")
            
            # 出力ディレクトリを探索
            output_base = Path("output/dssms_integration")
            latest_dir = max(output_base.glob("dssms_*"), key=lambda p: p.name, default=None)
            
            return {
                'symbol': symbol,
                'success': True,
                'output_dir': str(latest_dir) if latest_dir else None,
                'return_code': result.returncode
            }
        else:
            print(f"\n[ERROR] 銘柄 {symbol} のバックテスト失敗")
            print(f"[STDERR]\n{result.stderr}")
            
            return {
                'symbol': symbol,
                'success': False,
                'error': result.stderr[:500],
                'return_code': result.returncode
            }
    
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] 銘柄 {symbol} のバックテストがタイムアウト（2時間超過）")
        return {
            'symbol': symbol,
            'success': False,
            'error': 'Timeout (2 hours)',
            'return_code': -1
        }
    
    except Exception as e:
        print(f"\n[EXCEPTION] 銘柄 {symbol} の実行中にエラー: {e}")
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e),
            'return_code': -1
        }

def parse_comprehensive_report(output_dir: Path) -> dict:
    """
    comprehensive_report.txtから結果を抽出
    
    Args:
        output_dir: 出力ディレクトリパス
    
    Returns:
        dict: バックテスト結果の主要指標
    """
    report_path = output_dir / "comprehensive_report.txt"
    
    if not report_path.exists():
        return {'error': 'Report not found'}
    
    try:
        with open(report_path, 'r', encoding='cp932') as f:
            content = f.read()
        
        # 結果抽出（簡易的な文字列検索）
        result = {'symbol': output_dir.name}
        
        # 総取引回数
        if '総取引回数:' in content:
            line = [l for l in content.split('\n') if '総取引回数:' in l][0]
            result['total_trades'] = int(line.split(':')[1].strip())
        
        # 総リターン
        if '総リターン:' in content:
            line = [l for l in content.split('\n') if '総リターン:' in l][0]
            return_str = line.split(':')[1].strip().replace('%', '')
            result['total_return'] = float(return_str)
        
        # 勝率
        if '勝率:' in content:
            line = [l for l in content.split('\n') if '勝率:' in l][0]
            winrate_str = line.split(':')[1].strip().replace('%', '')
            result['win_rate'] = float(winrate_str)
        
        # 純利益
        if '純利益:' in content:
            line = [l for l in content.split('\n') if '純利益:' in l][0]
            pnl_str = line.split(':')[1].strip().replace('¥', '').replace(',', '')
            result['net_profit'] = float(pnl_str)
        
        return result
    
    except Exception as e:
        return {'error': f'Parse error: {e}'}

def main():
    """
    パーフェクトオーダー検証メイン処理
    """
    print("\n" + "="*80)
    print("パーフェクトオーダー検証スクリプト")
    print("="*80)
    print("\n目的: パーフェクトオーダー銘柄を選択すれば利益が出るのか検証")
    print("方式: 固定銘柄モード + 毎日パーフェクトオーダーチェック")
    print("期間: 2014-01-01 ~ 2025-12-31（11年間）")
    print("戦略: GC戦略のみ（take_profit=15%, trailing_stop=5%, stop_loss=3%）\n")
    
    # テスト銘柄リスト
    test_symbols = [
        "8306.T",  # 三菱UFJフィナンシャル・グループ
        "5713.T",  # 住友金属鉱山
        "4063.T",  # 信越化学工業
        "4004.T",  # 昭和電工
        "2768.T",  # 双日
        "7013.T"   # IHI
    ]
    
    # テスト期間
    start_date = "2014-01-01"
    end_date = "2025-12-31"
    
    print(f"テスト銘柄: {len(test_symbols)}銘柄")
    for symbol in test_symbols:
        print(f"  - {symbol}")
    print()
    
    # 実行確認
    response = input("バックテストを開始しますか？ (y/n): ")
    if response.lower() != 'y':
        print("キャンセルしました。")
        return
    
    # 各銘柄でバックテスト実行
    results = []
    start_time = datetime.now()
    
    for i, symbol in enumerate(test_symbols, 1):
        print(f"\n[{i}/{len(test_symbols)}] 銘柄 {symbol} 処理中...")
        
        result = run_single_symbol_backtest(symbol, start_date, end_date)
        results.append(result)
        
        # 成功した場合は結果を抽出
        if result['success'] and result.get('output_dir'):
            output_dir = Path(result['output_dir'])
            backtest_result = parse_comprehensive_report(output_dir)
            result.update(backtest_result)
    
    # 実行時間
    elapsed_time = datetime.now() - start_time
    
    # 結果集計
    print("\n" + "="*80)
    print("バックテスト実行完了")
    print("="*80)
    print(f"\n総実行時間: {elapsed_time}")
    print(f"成功: {sum(1 for r in results if r['success'])}/{len(results)}銘柄\n")
    
    # 結果テーブル表示
    print("結果サマリー:")
    print("-"*80)
    print(f"{'銘柄':<12} {'総取引':<10} {'総リターン':<12} {'勝率':<10} {'純利益':<15} {'ステータス':<10}")
    print("-"*80)
    
    successful_symbols = 0
    for result in results:
        symbol = result['symbol']
        if result['success']:
            trades = result.get('total_trades', '-')
            total_return = result.get('total_return', '-')
            win_rate = result.get('win_rate', '-')
            net_profit = result.get('net_profit', '-')
            status = '成功'
            
            # プラスリターン判定
            if isinstance(total_return, (int, float)) and total_return > 0:
                successful_symbols += 1
                status = '✅成功'
        else:
            trades = '-'
            total_return = '-'
            win_rate = '-'
            net_profit = '-'
            status = '❌失敗'
        
        print(f"{symbol:<12} {str(trades):<10} {str(total_return) + '%' if isinstance(total_return, (int, float)) else '-':<12} "
              f"{str(win_rate) + '%' if isinstance(win_rate, (int, float)) else '-':<10} "
              f"{'¥' + f'{net_profit:,.0f}' if isinstance(net_profit, (int, float)) else '-':<15} {status:<10}")
    
    print("-"*80)
    
    # パーフェクトオーダー前提の判定
    success_rate = successful_symbols / len(results) * 100 if results else 0
    print(f"\nプラスリターン銘柄: {successful_symbols}/{len(results)}銘柄 ({success_rate:.1f}%)")
    
    if success_rate >= 66.7:
        print("\n✅ パーフェクトオーダー前提は有効です（66%以上の銘柄でプラス）")
    else:
        print("\n❌ パーフェクトオーダー前提は無効です（50%以下の銘柄でプラス）")
    
    # 結果をCSVに保存
    output_dir = Path("docs/To make profits from system trading")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"perfect_order_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding='cp932')
    
    print(f"\n結果をCSVに保存: {csv_path}")
    print("\n調査レポートの更新:")
    print("  - docs/To make profits from system trading/System_Trading_Profit_Improvement_Investigation.md")
    print("  - Cycle 3の実行結果を追記してください")

if __name__ == "__main__":
    main()

"""
max_positions制約2ヶ月検証スクリプト

2026-02-15作成
- 2024-01-01~02-28の2ヶ月バックテスト実行
- 全日のポジション数を記録し、max_positions=2違反がないか検証
"""
import logging
import sys
import re
from datetime import datetime
from io import StringIO
from src.dssms.dssms_integrated_main import DSSMSIntegratedBacktester

# ログキャプチャ用ハンドラ
log_buffer = StringIO()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(log_buffer)
    ]
)

def main():
    print("=" * 80)
    print("max_positions制約 2ヶ月検証バックテスト")
    print("=" * 80)
    print(f"期間: 2024-01-01 ~ 2024-02-28")
    print(f"設定: max_positions=2")
    print("=" * 80)
    
    config = {
        'initial_capital': 1000000,
        'target_symbols': ['9101.T', '9104.T', '9107.T', '5802.T', '8802.T', '6301.T', '6703.T'],
        'dssms_backtest_start_date': '2024-01-01',
        'dssms_backtest_end_date': '2024-02-28'
    }
    
    backtester = DSSMSIntegratedBacktester(config=config)
    
    start_date = datetime.strptime('2024-01-01', '%Y-%m-%d')
    end_date = datetime.strptime('2024-02-28', '%Y-%m-%d')
    target_symbols = config['target_symbols']
    
    print("\nバックテスト実行中...")
    results = backtester.run_dynamic_backtest(start_date, end_date, target_symbols)
    print("\nバックテスト完了")
    
    # ログからPOSITION_DEBUGを抽出してmax_positions違反チェック
    log_content = log_buffer.getvalue()
    
    position_debug_lines = [line for line in log_content.split('\n') if 'POSITION_DEBUG' in line]
    switch_debug_lines = [line for line in log_content.split('\n') if 'len(positions)=' in line]
    force_close_lines = [line for line in log_content.split('\n') if 'FORCE_CLOSE' in line and 'FIFO' in line]
    safety_check_lines = [line for line in log_content.split('\n') if 'SAFETY_CHECK' in line]
    
    print("\n" + "=" * 80)
    print("検証結果")
    print("=" * 80)
    
    # ポジション数の最大値チェック
    max_observed = 0
    violations = []
    
    for line in position_debug_lines:
        match = re.search(r'len\(positions\)=(\d+)', line)
        if match:
            count = int(match.group(1))
            if count > max_observed:
                max_observed = count
            if count > 2:
                violations.append(line.strip())
    
    # 切替判定ログからもチェック
    for line in switch_debug_lines:
        match = re.search(r'len\(positions\)=(\d+)', line)
        if match:
            count = int(match.group(1))
            if count > max_observed:
                max_observed = count
            if count > 2:
                violations.append(line.strip())
    
    print(f"最大同時保有数: {max_observed}")
    print(f"FORCE_CLOSE実行回数: {len(force_close_lines)}")
    print(f"SAFETY_CHECK発動回数: {len(safety_check_lines)}")
    print(f"max_positions制約違反: {len(violations)}件")
    
    if violations:
        print("\n[NG] 制約違反検出:")
        for v in violations[:10]:
            print(f"  {v}")
    else:
        print("\n[OK] max_positions=2制約を全日で遵守")
    
    # POSITION_DEBUGサマリ
    print(f"\nBUY実行回数: {len(position_debug_lines)}")
    for line in position_debug_lines:
        # 行を簡潔に表示
        cleaned = re.sub(r'^.*POSITION_DEBUG\] ', '', line.strip())
        print(f"  {cleaned}")
    
    # FORCE_CLOSEサマリ
    if force_close_lines:
        print(f"\nFORCE_CLOSE履歴:")
        for line in force_close_lines[:20]:
            cleaned = re.sub(r'^.*FORCE_CLOSE\] ', '', line.strip())
            print(f"  {cleaned}")
    
    print("\n" + "=" * 80)
    print(f"総取引数: {results.get('total_trades', 0)}")
    print(f"総損益: {results.get('total_pnl', 0):,.0f}円")
    print("=" * 80)

if __name__ == "__main__":
    main()

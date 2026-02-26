import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parents[1]))

from stoploss_optimizer.utils.backtest_runner import run_backtest
from stoploss_optimizer.utils.result_analyzer import collect_results

result = run_backtest('2024-01-01', '2024-01-31', -0.05)
print(f'成功: {result["success"]}')
print(f'出力先: {result["output_dir"]}')

if result['success']:
    metrics = collect_results(result['output_dir'])
    print(f'PF: {metrics["pf"]}')
    print(f'取引数: {metrics["trades"]}')
else:
    print(f'エラー: {result["error"]}')
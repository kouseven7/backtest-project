"""
戦略クラスインポートテスト
"""

import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

strategy_imports = [
    ("VWAPBreakoutStrategy", "strategies.VWAP_Breakout"),
    ("MomentumInvestingStrategy", "strategies.Momentum_Investing"), 
    ("BreakoutStrategy", "strategies.Breakout"),
    ("VWAPBounceStrategy", "strategies.VWAP_Bounce"),
    ("OpeningGapStrategy", "strategies.Opening_Gap"),
    ("ContrarianStrategy", "strategies.contrarian_strategy"),
    ("GCStrategy", "strategies.gc_strategy_signal")
]

for strategy_name, module_path in strategy_imports:
    try:
        print(f"インポート中: {strategy_name} from {module_path}")
        module = __import__(module_path, fromlist=[strategy_name])
        strategy_class = getattr(module, strategy_name)
        print(f"✅ {strategy_name} インポート成功")
    except Exception as e:
        print(f"❌ {strategy_name} インポートエラー: {e}")
        import traceback
        traceback.print_exc()
        print()

print("戦略インポートテスト完了")

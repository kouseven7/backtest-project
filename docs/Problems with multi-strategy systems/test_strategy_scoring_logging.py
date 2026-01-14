"""
戦略スコアリングとMarket Analyzer分析結果のログ取得テスト

目的:
- DynamicStrategySelectorのスコアリング結果を取得
- MarketAnalyzerの市場環境判定結果を取得
- なぜGC戦略のみ選択されるか特定

Author: Backtest Project Team
Created: 2026-01-12
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 一時的にデバッグログ出力を有効化
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# main_new.pyを直接実行
from main_new import main

print("\n" + "="*80)
print("戦略スコアリング・市場環境判定ログ取得テスト")
print("="*80)
print("\n[INFO] main_new.pyを実行してログを収集します...")
print("[INFO] 以下のログを確認してください:")
print("  - [SCORE_DETAIL] Strategy scores calculated")
print("  - [SELECTION_RESULT] Selected strategies")
print("  - Market analysis completed - Regime")
print("="*80 + "\n")

# 実行
results = main()

print("\n" + "="*80)
print("実行完了")
print("="*80)

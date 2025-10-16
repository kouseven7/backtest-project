"""
main_system - Main.py統合システム
動的戦略選択・実行・リスク管理・パフォーマンス計算の統合モジュール群

このパッケージはmain.pyから固定優先度システムを動的戦略選択システムに変換するための
統合候補モジュール群を含みます。

構成:
- data_acquisition: データ取得・キャッシュ・エラーハンドリング
- market_analysis: トレンド判定・相場分析・Perfect Order検出
- strategy_selection: 戦略選択・重み計算・切替システム
- execution_control: 戦略実行管理・バッチテスト・マルチストラテジー制御
- risk_management: リスク制御・ドローダウン管理・拡張リスク管理
- performance: パフォーマンス計算・データ抽出・取引分析
- reporting: レポート生成・可視化ダッシュボード
"""

__version__ = "1.0.0"
__author__ = "backtest-project integration team"

# 主要モジュールのインポート
from . import data_acquisition
from . import market_analysis
from . import strategy_selection
from . import execution_control
from . import risk_management
from . import performance
from . import reporting

__all__ = [
    'data_acquisition',
    'market_analysis', 
    'strategy_selection',
    'execution_control',
    'risk_management',
    'performance',
    'reporting'
]
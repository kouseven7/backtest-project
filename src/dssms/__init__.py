# DSSMS __init__.py - パフォーマンス最適化版
# 
# TODO-PERF-004: 自動インポート除去
# 1932モジュール自動ロード問題の解決のため、すべての自動インポートを無効化
# 必要なコンポーネントは明示的にインポートすること
#
# 最適化効果: 2871ms → 目標 < 50ms
# 
# 注意: この変更により、`from src.dssms import Component` 形式は使用不可
#       明示的インポート `from src.dssms.component_file import Component` を使用

__version__ = "2.2.0"
__author__ = "AI Assistant"

# 自動インポート無効化
# すべてのコンポーネントは必要時に明示的インポートすること

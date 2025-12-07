"""
直接実行成功の理由調査
python src/dssms/dssms_integrated_main.py が成功する理由を確認
"""
import sys
import os

print("=" * 80)
print("dssms_integrated_main.py を直接実行した際のsys.path")
print("=" * 80)

# dssms_integrated_main.pyと同じロジックを再現
__file__ = "src/dssms/dssms_integrated_main.py"
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
print(f"__file__: {__file__}")
print(f"project_root: {project_root}")
print(f"Absolute project_root: {os.path.abspath(project_root)}")

# 直接実行時のsys.path[0]
direct_exec_path = os.path.dirname(os.path.abspath("src/dssms/dssms_integrated_main.py"))
print(f"\n直接実行時のsys.path[0]: {direct_exec_path}")

# dssms_backtester_v3.py の場所
dss_v3_path = os.path.join(direct_exec_path, "dssms_backtester_v3.py")
print(f"dssms_backtester_v3.py: {dss_v3_path}")
print(f"Exists: {os.path.exists(dss_v3_path)}")

print("\n" + "=" * 80)
print("結論")
print("=" * 80)
print("直接実行: python src/dssms/dssms_integrated_main.py")
print(f"  → sys.path[0] = {direct_exec_path} (スクリプトのディレクトリ)")
print(f"  → import dssms_backtester_v3 は {direct_exec_path} で検索")
print(f"  → 同じフォルダにあるため成功")
print("\nモジュール実行: python -m src.dssms.dssms_integrated_main")
print(f"  → sys.path[0] = プロジェクトルート")
print(f"  → import dssms_backtester_v3 はプロジェクトルートで検索")
print(f"  → 見つからない (src/dssms/にあるため失敗)")

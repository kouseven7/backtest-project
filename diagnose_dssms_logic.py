import sys
import json
from pathlib import Path

def diagnose_switching_logic():
    """DSSMS切り替えロジックの診断"""
    
    print("[SEARCH] DSSMS切り替えロジック診断")
    print("=" * 50)
    
    # 1. 設定ファイルの切り替え判定基準確認
    config_files = {
        "intelligent_switch": "config/dssms/intelligent_switch_config.json",
        "ranking": "config/dssms/ranking_config.json", 
        "dssms_main": "config/dssms/dssms_config.json"
    }
    
    for name, file_path in config_files.items():
        path = Path(file_path)
        if path.exists():
            print(f"\n[LIST] {name} 設定確認:")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 切り替え関連の設定を探す
                switching_keys = [
                    'switching_threshold', 'minimum_improvement', 'evaluation_frequency',
                    'min_holding_period', 'switch_criteria', 'threshold', 'confidence'
                ]
                
                found_keys = []
                for key in switching_keys:
                    if key in config:
                        found_keys.append(f"{key}: {config[key]}")
                    
                    # ネストした設定も確認
                    for sub_key, sub_value in config.items():
                        if isinstance(sub_value, dict) and key in sub_value:
                            found_keys.append(f"{sub_key}.{key}: {sub_value[key]}")
                
                if found_keys:
                    for key_info in found_keys:
                        print(f"  [OK] {key_info}")
                else:
                    print(f"  [WARNING]  切り替え基準が見つかりません")
                    print(f"  📝 主要キー: {list(config.keys())}")
                    
            except Exception as e:
                print(f"  [ERROR] 読み込みエラー: {e}")
        else:
            print(f"  [ERROR] {file_path} が見つかりません")

def analyze_switching_frequency_cause():
    """286回切り替えの原因分析"""
    
    print("\n[TARGET] 286回切り替えの原因分析")
    print("-" * 30)
    
    # 基本計算
    total_days = 365  # 2023年
    switches = 286
    avg_switches_per_day = switches / total_days
    
    print(f"年間日数: {total_days}日")
    print(f"総切り替え: {switches}回") 
    print(f"1日平均: {avg_switches_per_day:.2f}回")
    
    if avg_switches_per_day > 0.7:
        print("[ALERT] ほぼ毎日切り替えています！")
        print("   → 評価頻度が daily である可能性が高い")
        print("   → 切り替え閾値が低すぎる可能性")
    
    # 予想される問題パターン
    print("\n[SEARCH] 予想される問題パターン:")
    patterns = [
        "毎日評価 + 閾値なし切り替え",
        "短期的な価格変動への過反応", 
        "パーフェクトオーダー判定の不安定性",
        "スコアリング基準の感度が高すぎる"
    ]
    
    for i, pattern in enumerate(patterns, 1):
        print(f"{i}. {pattern}")

def check_backtest_logic_files():
    """バックテストロジックファイルの存在確認"""
    
    print("\n📁 ロジックファイル確認")
    print("-" * 30)
    
    logic_files = [
        "src/dssms/dssms_backtester.py",
        "src/dssms/intelligent_switch_manager.py", 
        "src/dssms/hierarchical_ranking_system.py",
        "src/dssms/comprehensive_scoring_engine.py",
        "src/dssms/perfect_order_detector.py"
    ]
    
    for file_path in logic_files:
        path = Path(file_path)
        if path.exists():
            print(f"[OK] {file_path}")
            # ファイルサイズで複雑さを推測
            size = path.stat().st_size
            print(f"   サイズ: {size:,} bytes")
        else:
            print(f"[ERROR] {file_path}")

if __name__ == "__main__":
    diagnose_switching_logic()
    analyze_switching_frequency_cause() 
    check_backtest_logic_files()
    
    print("\n[TARGET] 次のステップ:")
    print("1. 上記のロジックファイルの内容確認")
    print("2. 切り替え判定基準の特定")
    print("3. 評価頻度の確認（daily vs weekly vs monthly）") 
    print("4. 閾値パラメータの調整または追加")
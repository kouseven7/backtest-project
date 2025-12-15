"""実際に読み込まれる銘柄数の確認"""
from pathlib import Path
import json

# 設定ファイルからの読み込みをシミュレート
config_file = "nikkei225_components.json"
config_path = Path(__file__).parent / "config" / "dssms" / config_file

print(f"=== 実際の動作シミュレーション ===\n")
print(f"設定ファイルパス: {config_path}")
print(f"ファイル存在: {config_path.exists()}\n")

if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        symbols = data.get('symbols', [])
        
        print(f"【Layer 1: 設定ファイルからの読み込み】")
        print(f"総銘柄数: {len(symbols)}")
        print(f"ユニーク銘柄数: {len(set(symbols))}")
        print(f"重複数: {len(symbols) - len(set(symbols))}")
        
        if symbols and len(symbols) > 20:
            print(f"\n条件 'len(symbols) > 20' を満たす: True")
            print(f"→ この{len(symbols)}個の銘柄リストが返される")
            print(f"→ Layer 2（約40銘柄）とLayer 3（50銘柄）は実行されない\n")
            
            print(f"【結論】")
            print(f"現在のシステムは {len(symbols)} 個（ユニーク {len(set(symbols))} 個）の銘柄から")
            print(f"フィルタリングを開始しています。")
            print(f"\n50個ではなく、{len(symbols)}個からフィルタリングしています。")
        else:
            print(f"\n条件を満たさない → Layer 2へ")
else:
    print(f"【Layer 1失敗】")
    print(f"設定ファイルが存在しない → Layer 2の約40銘柄リストへ")

print(f"\n" + "="*50)
print(f"補足: Layer 2とLayer 3は設定ファイルが存在しない場合のフォールバックです")

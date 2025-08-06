"""
フェーズ3B リアルタイムデータ接続システム 依存関係インストーラー
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """必要な依存関係をインストール"""
    
    # フェーズ3B用の追加依存関係
    requirements = [
        "yfinance>=0.2.18",         # Yahoo Finance データ
        "requests>=2.31.0",         # HTTP リクエスト
        "pandas>=2.0.0",            # データ処理
        "numpy>=1.24.0",            # 数値計算
        "aiohttp>=3.8.0",           # 非同期HTTP（将来の拡張用）
        "asyncio",                  # 非同期処理（標準ライブラリ）
        "dataclasses",              # データクラス（Python 3.7+標準）
        "threading",                # マルチスレッド（標準ライブラリ）
        "sqlite3",                  # SQLite（標準ライブラリ）
        "pickle",                   # シリアライゼーション（標準ライブラリ）
        "hashlib",                  # ハッシュ（標準ライブラリ）
        "json",                     # JSON（標準ライブラリ）
        "pathlib",                  # パス操作（標準ライブラリ）
        "datetime",                 # 日時（標準ライブラリ）
        "time",                     # 時間（標準ライブラリ）
        "collections",              # コレクション（標準ライブラリ）
        "enum",                     # 列挙型（標準ライブラリ）
        "abc",                      # 抽象基底クラス（標準ライブラリ）
        "typing",                   # 型ヒント（標準ライブラリ）
        "traceback",               # トレースバック（標準ライブラリ）
        "signal"                    # シグナル（標準ライブラリ）
    ]
    
    # pip でインストールが必要なパッケージ
    pip_packages = [
        "yfinance>=0.2.18",
        "requests>=2.31.0", 
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "aiohttp>=3.8.0"
    ]
    
    print("フェーズ3B: リアルタイムデータ接続システム")
    print("依存関係をインストール中...")
    print("=" * 50)
    
    for package in pip_packages:
        try:
            print(f"インストール中: {package}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ {package} インストール完了")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ {package} インストール失敗: {e}")
            print(f"エラー出力: {e.stderr}")
            
        except Exception as e:
            print(f"✗ {package} インストール中にエラー: {e}")
    
    print("=" * 50)
    print("依存関係インストール完了")
    
    # インストール確認
    print("\\nインストール確認中...")
    
    import_tests = [
        ("yfinance", "yfinance"),
        ("requests", "requests"),
        ("pandas", "pandas"),
        ("numpy", "numpy")
    ]
    
    for module_name, import_name in import_tests:
        try:
            __import__(import_name)
            print(f"✓ {module_name} インポート成功")
        except ImportError as e:
            print(f"✗ {module_name} インポート失敗: {e}")
    
    print("\\n依存関係確認完了")

if __name__ == "__main__":
    install_requirements()

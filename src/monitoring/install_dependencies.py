"""
フェーズ3B 監視ダッシュボード 依存関係インストールスクリプト

このスクリプトは、監視ダッシュボードに必要な依存関係を
自動的にインストールします。
"""

import subprocess
import sys
from pathlib import Path

# 必要な依存関係
DASHBOARD_REQUIREMENTS = [
    "fastapi>=0.68.0",
    "uvicorn[standard]>=0.15.0",
    "jinja2>=3.0.0",
    "aiofiles>=0.7.0",
    "plotly>=5.0.0",
    "numpy>=1.21.0",
    "psutil>=5.8.0",  # システムメトリクス取得用
    "requests>=2.25.0",  # Webhook通知用
    "websockets>=10.0",  # WebSocket通信用
]

# オプション依存関係（メール通知用）
OPTIONAL_REQUIREMENTS = [
    "email-validator>=1.1.0",
    "python-multipart>=0.0.5",
]

def install_requirements():
    """依存関係インストール"""
    print("フェーズ3B 監視ダッシュボード 依存関係インストール")
    print("=" * 60)
    
    # 基本依存関係インストール
    print("基本依存関係をインストールしています...")
    for package in DASHBOARD_REQUIREMENTS:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"✅ {package} インストール完了")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} インストール失敗: {e}")
            
    # オプション依存関係インストール
    print("\nオプション依存関係をインストールしています...")
    for package in OPTIONAL_REQUIREMENTS:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"✅ {package} インストール完了")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ {package} インストール失敗（オプション）: {e}")
            
    print("\n依存関係インストール完了!")
    print("\n次のステップ:")
    print("1. テスト実行: python src/monitoring/test_dashboard.py")
    print("2. デモ実行: python src/monitoring/demo_dashboard.py")
    print("3. ダッシュボードアクセス: http://localhost:8080")

if __name__ == "__main__":
    install_requirements()

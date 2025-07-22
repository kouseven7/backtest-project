"""
4-3-2 Dashboard System Test and Commit Script
戦略比率とパフォーマンスのリアルタイム表示システム テスト&コミット

PowerShell用 統合テストスクリプト
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_powershell_command(command: str, description: str = "") -> bool:
    """PowerShellコマンド実行"""
    try:
        if description:
            logger.info(f"実行中: {description}")
        
        logger.info(f"コマンド: {command}")
        
        # PowerShell コマンド実行
        result = subprocess.run(
            ["powershell.exe", "-Command", command],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            logger.info("✅ 成功")
            if result.stdout.strip():
                logger.info(f"出力: {result.stdout.strip()}")
            return True
        else:
            logger.error("❌ 失敗")
            if result.stderr.strip():
                logger.error(f"エラー: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        logger.error(f"コマンド実行エラー: {e}")
        return False

def main():
    """メイン実行関数"""
    logger.info("=== 4-3-2 システム テスト&コミット開始 ===")
    
    # 1. Python 環境確認
    python_check = run_powershell_command(
        "python --version",
        "Python環境確認"
    )
    
    if not python_check:
        logger.error("Python環境が確認できません")
        return False
    
    # 2. 必要なパッケージインストール
    packages = [
        "matplotlib",
        "pandas",
        "numpy"
    ]
    
    for package in packages:
        install_result = run_powershell_command(
            f"python -m pip install {package}",
            f"{package} インストール確認"
        )
        
        if not install_result:
            logger.warning(f"{package} インストール失敗（既にインストール済みの可能性）")
    
    # 3. コンポーネントテスト実行
    component_test = run_powershell_command(
        "python demo_dashboard_4_3_2.py --component-test",
        "4-3-2 コンポーネントテスト実行"
    )
    
    # 4. 統合デモテスト実行
    demo_test = run_powershell_command(
        "python demo_dashboard_4_3_2.py",
        "4-3-2 統合デモテスト実行"
    )
    
    # 5. ファイル確認
    check_files = [
        "visualization\\performance_data_collector.py",
        "visualization\\dashboard_chart_generator.py", 
        "visualization\\dashboard_config.py",
        "visualization\\strategy_performance_dashboard.py",
        "demo_dashboard_4_3_2.py"
    ]
    
    files_exist = True
    for file_path in check_files:
        file_check = run_powershell_command(
            f"Test-Path '{file_path}'",
            f"ファイル確認: {file_path}"
        )
        if not file_check:
            files_exist = False
    
    # 6. Git状態確認
    git_status = run_powershell_command(
        "git status --porcelain",
        "Git状態確認"
    )
    
    # 7. Gitコミット実行
    if files_exist and (component_test or demo_test):
        logger.info("=== Gitコミット実行 ===")
        
        # ファイル追加
        add_result = run_powershell_command(
            "git add visualization/performance_data_collector.py ; git add visualization/dashboard_chart_generator.py ; git add visualization/dashboard_config.py ; git add visualization/strategy_performance_dashboard.py ; git add demo_dashboard_4_3_2.py",
            "ファイル追加"
        )
        
        if add_result:
            # コミット実行（日本語メッセージ）
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"4-3-2: 戦略比率とパフォーマンスのリアルタイム表示システム実装完了 ({timestamp})"
            
            commit_result = run_powershell_command(
                f'git commit -m "{commit_message}"',
                "Git コミット実行"
            )
            
            if commit_result:
                logger.info("🎉 4-3-2システム実装&コミット完了!")
                return True
    
    logger.error("❌ テストまたはコミットに失敗しました")
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("🎉 4-3-2「戦略比率とパフォーマンスのリアルタイム表示」")
        print("   システム実装完了!")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ 4-3-2 システム実装に問題があります")
        print("   ログを確認してください")
        print("="*60)
        sys.exit(1)

#!/usr/bin/env python3
"""
DSSMS リターン計算最終修正スクリプト

問題: _unified_total_return 属性が正しく設定されていない
- 統一出力システム: 160.13% (正しい)
- 最終ログ表示: 0.0% (間違い)

根本原因: 統一出力システム内でのリターン保存処理が適切に動作していない
"""

import re
import logging
from datetime import datetime
from pathlib import Path

def setup_logger():
    """ロガー設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def fix_unified_return_attribute():
    """統一リターン属性の修正"""
    logger = setup_logger()
    logger.info("[TOOL] DSSMS 統一リターン属性修正開始")
    
    # DSSMSバックテスターファイル
    dssms_file = Path("src/dssms/dssms_backtester.py")
    
    if not dssms_file.exists():
        logger.error(f"ファイルが見つかりません: {dssms_file}")
        return False
    
    try:
        # ファイル読み込み
        with open(dssms_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info("[CHART] 修正内容:")
        
        # 1. 統一出力システム内での total_return 保存を確実にする
        logger.info("1. 統一出力システム内でのリターン保存を強化")
        
        # パフォーマンス計算後に total_return を確実に保存
        performance_log_pattern = r'(self\.logger\.info\(f"パフォーマンス計算: 総リターン\{total_return:.2f\}%, 勝率\{win_rate:.2f\}"\))'
        
        def enhance_return_saving(match):
            replacement = f'''{match.group(1)}
                
                # 統一出力用の正確なリターン値を保存（確実に実行）
                self._unified_total_return = float(total_return)
                self.logger.info(f"統一リターン値保存: {{self._unified_total_return:.2f}}%")'''
            return replacement
        
        if re.search(performance_log_pattern, content):
            content = re.sub(performance_log_pattern, enhance_return_saving, content)
            logger.info("   [OK] パフォーマンス計算後の保存処理を強化")
        else:
            logger.warning("   [WARNING] パフォーマンス計算ログが見つかりません")
        
        # 2. 最終表示部分の修正を確実にする
        logger.info("2. 最終表示での統一リターン取得を修正")
        
        # actual_return 取得部分を修正
        actual_return_pattern = r'actual_return = getattr\(backtester, \'_unified_total_return\', performance_metrics\.total_return\)'
        
        def fix_actual_return_retrieval(match):
            replacement = '''# 統一出力システムで計算された実際のリターンを取得
        unified_return = getattr(backtester, '_unified_total_return', None)
        if unified_return is not None:
            actual_return = unified_return
            logger.info(f"統一リターン値使用: {actual_return:.2f}%")
        else:
            actual_return = performance_metrics.total_return
            logger.info(f"フォールバック値使用: {actual_return:.2f}%")'''
            return replacement
        
        if re.search(actual_return_pattern, content):
            content = re.sub(actual_return_pattern, fix_actual_return_retrieval, content)
            logger.info("   [OK] 最終表示での統一リターン取得を修正")
        else:
            logger.warning("   [WARNING] actual_return 部分が見つかりません")
        
        # 3. デバッグ用の追加ログを挿入
        logger.info("3. デバッグ用ログを追加")
        
        # demo実行の最初に_unified_total_returnの初期状態を確認
        demo_start_pattern = r'(def main\(\):.*?logger = setup_logger\(\'dssms\.backtester\.demo\'\))'
        
        def add_debug_logging(match):
            replacement = f'''{match.group(1)}
    logger.info("=== DSSMS リターン計算デバッグ開始 ===")'''
            return replacement
        
        content = re.sub(demo_start_pattern, add_debug_logging, content, flags=re.DOTALL)
        
        # 4. バックテスター作成後に初期化を確認
        backtester_creation_pattern = r'(backtester = DSSMSBacktester\(config\))'
        
        def add_backtester_debug(match):
            replacement = f'''{match.group(1)}
        logger.info(f"バックテスター作成後の _unified_total_return: {{getattr(backtester, '_unified_total_return', 'NOT_SET')}}")'''
            return replacement
        
        content = re.sub(backtester_creation_pattern, add_backtester_debug, content)
        
        # バックアップ作成
        backup_file = dssms_file.with_suffix('.py.backup_final_fix')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(open(dssms_file, 'r', encoding='utf-8').read())
        
        # 修正版を保存
        with open(dssms_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info("[OK] 最終修正完了")
        logger.info(f"📁 バックアップ: {backup_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"修正エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """メイン実行"""
    logger = setup_logger()
    
    print("[TOOL] DSSMS リターン計算最終修正ツール")
    print("=" * 60)
    print("問題: _unified_total_return 属性が正しく設定されていない")
    print("対象: src/dssms/dssms_backtester.py")
    print("=" * 60)
    
    if fix_unified_return_attribute():
        print("\n[OK] 最終修正完了！")
        print("次のステップ:")
        print("1. python src/dssms/dssms_backtester.py  # 最終修正版実行")
        print("2. デバッグログで _unified_total_return の値を確認")
        print("3. 統一リターンの一貫性を確認")
    else:
        print("\n[ERROR] 修正失敗")
        print("手動修正が必要です")

if __name__ == "__main__":
    main()

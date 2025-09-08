#!/usr/bin/env python3
"""
DSSMS リターン計算修正スクリプト

問題: DSSMSバックテスターで2つの異なる総リターン計算が存在
- 統一出力システム: 169.55% (正しい)
- パフォーマンス指標: 4.30% (間違い)

修正内容:
- DSSMSPerformanceMetrics作成時の total_return を正しい値に統一
- 統一出力システムで計算された実際の総リターンを使用
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

def fix_return_calculation_consistency():
    """リターン計算の一貫性を修正"""
    logger = setup_logger()
    logger.info("🔧 DSSMS リターン計算修正開始")
    
    # DSSMSバックテスターファイル
    dssms_file = Path("src/dssms/dssms_backtester.py")
    
    if not dssms_file.exists():
        logger.error(f"ファイルが見つかりません: {dssms_file}")
        return False
    
    try:
        # ファイル読み込み
        with open(dssms_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 問題のある箇所を特定・修正
        modifications = []
        
        # 1. _convert_to_unified_format内のtotal_returnを正確に取得
        logger.info("📊 修正1: 統一出力システムでの total_return 取得方法を修正")
        
        # 統一出力システム内でのリターン計算を正確に実行するようにパッチ
        unified_format_pattern = r'def _convert_to_unified_format\(self\) -> Dict\[str, Any\]:(.*?)(?=def\s+\w+|class\s+\w+|\Z)'
        
        def fix_unified_format_method(match):
            method_content = match.group(1)
            
            # パフォーマンス計算部分で実際のリターンを保存
            if 'self.logger.info(f"パフォーマンス計算: 総リターン{total_return:.2f}%, 勝率{win_rate:.2f}")' in method_content:
                # 統一出力システム内での正確なリターン値を変数として保存
                enhanced_content = method_content.replace(
                    'self.logger.info(f"パフォーマンス計算: 総リターン{total_return:.2f}%, 勝率{win_rate:.2f}")',
                    '''self.logger.info(f"パフォーマンス計算: 総リターン{total_return:.2f}%, 勝率{win_rate:.2f}")
                
                # 統一出力用の正確なリターン値を保存
                self._unified_total_return = total_return'''
                )
                modifications.append("統一出力システム内でのリターン値保存を追加")
                return f'def _convert_to_unified_format(self) -> Dict[str, Any]:{enhanced_content}'
            
            return match.group(0)
        
        content = re.sub(unified_format_pattern, fix_unified_format_method, content, flags=re.DOTALL)
        
        # 2. calculate_performance_metrics内でのtotal_return計算を修正
        logger.info("📈 修正2: パフォーマンス指標計算での total_return を統一")
        
        performance_pattern = r'(performance_metrics = DSSMSPerformanceMetrics\(\s*total_return=float\(total_return\),)'
        
        def fix_performance_metrics(match):
            # 統一出力システムで計算された値を使用
            replacement = '''# 統一出力システムで計算された正確な値を使用
            unified_return = getattr(self, '_unified_total_return', total_return)
            performance_metrics = DSSMSPerformanceMetrics(
                total_return=float(unified_return),'''
            modifications.append("パフォーマンス指標での total_return を統一出力値に修正")
            return replacement
        
        content = re.sub(performance_pattern, fix_performance_metrics, content)
        
        # 3. 初期化時に _unified_total_return 属性を追加
        logger.info("🔧 修正3: 初期化時の属性追加")
        
        init_pattern = r'(self\.logger = setup_logger\(\'dssms\.backtester\'\))'
        
        def add_unified_return_attribute(match):
            replacement = f'''{match.group(1)}
        
        # 統一出力システムとの一貫性確保用
        self._unified_total_return = 0.0'''
            modifications.append("初期化時に _unified_total_return 属性を追加")
            return replacement
        
        content = re.sub(init_pattern, add_unified_return_attribute, content)
        
        # 4. デモ実行部分での最終表示を修正
        logger.info("📋 修正4: 最終ログ表示の修正")
        
        demo_pattern = r'logger\.info\(f"総リターン: \{performance_metrics\.total_return:.2%\}"\)'
        
        def fix_demo_log(match):
            replacement = '''# 統一出力システムで計算された実際のリターンを表示
            actual_return = getattr(backtester, '_unified_total_return', performance_metrics.total_return)
            logger.info(f"総リターン: {actual_return:.2%} (実際の計算値)")
            logger.info(f"参考: パフォーマンス指標値: {performance_metrics.total_return:.2%}")'''
            modifications.append("最終ログ表示を実際の計算値に修正")
            return replacement
        
        content = re.sub(demo_pattern, fix_demo_log, content)
        
        # バックアップ作成
        backup_file = dssms_file.with_suffix('.py.backup_return_fix')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(open(dssms_file, 'r', encoding='utf-8').read())
        
        # 修正版を保存
        with open(dssms_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info("✅ 修正完了:")
        for i, mod in enumerate(modifications, 1):
            logger.info(f"  {i}. {mod}")
        
        logger.info(f"📁 バックアップ: {backup_file}")
        logger.info("🚀 修正版DSSMS準備完了")
        
        return True
        
    except Exception as e:
        logger.error(f"修正エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """メイン実行"""
    logger = setup_logger()
    
    print("🔧 DSSMS リターン計算修正ツール")
    print("=" * 60)
    print("問題: ログ表示と実際の計算値の不整合")
    print("対象: src/dssms/dssms_backtester.py")
    print("=" * 60)
    
    if fix_return_calculation_consistency():
        print("\n✅ 修正完了！")
        print("次のステップ:")
        print("1. python src/dssms/dssms_backtester.py  # 修正版実行")
        print("2. 総リターンの一貫性を確認")
        print("3. Excel/テキスト出力とログ表示の一致を確認")
    else:
        print("\n❌ 修正失敗")
        print("手動修正が必要です")

if __name__ == "__main__":
    main()

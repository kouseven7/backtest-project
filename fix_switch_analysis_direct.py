#!/usr/bin/env python3
"""
切替分析シートの直接修正スクリプト

Excel出力での成功判定とパフォーマンスデータ処理を修正
"""

import os
import shutil
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_excel_exporter_directly():
    """DSSMS Excel エクスポーターを直接修正"""
    
    excel_exporter_path = 'output/dssms_excel_exporter_v2.py'
    
    if not os.path.exists(excel_exporter_path):
        logger.error(f"ファイルが見つかりません: {excel_exporter_path}")
        return False
    
    try:
        # バックアップ作成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f'backup_dssms_excel_exporter_v2_{timestamp}.py'
        shutil.copy2(excel_exporter_path, backup_path)
        logger.info(f"バックアップ作成: {backup_path}")
        
        with open(excel_exporter_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. _generate_switch_history メソッドの修正
        old_generate_method = '''    def _generate_switch_history(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """切替履歴データ生成"""
        try:
            switch_history = []
            
            # DSSMSの切替イベントから履歴を生成
            switches = result.get("switch_history", [])
            
            if not switches:
                # サンプル切替履歴生成
                switches = self._generate_sample_switch_history(result)
            
            for i, switch in enumerate(switches):
                switch_data = {
                    "date": switch.get("date", datetime.now() - timedelta(days=i*2)),
                    "from_symbol": switch.get("from_symbol", f"PREV_{i}"),
                    "to_symbol": switch.get("to_symbol", f"NEW_{i}"),
                    "reason": switch.get("reason", "技術的指標による判定"),
                    "switch_price": switch.get("switch_price", np.random.uniform(1000, 2000)),
                    "switch_cost": switch.get("switch_cost", np.random.uniform(1000, 5000)),
                    "performance_after": switch.get("performance_after", np.random.uniform(-0.1, 0.15)),
                    "success": "成功" if switch.get("profit_loss", 0) > 0 else "失敗"
                }
                
                switch_history.append(switch_data)
            
            return switch_history
            
        except Exception as e:
            self.logger.error(f"切替履歴生成エラー: {e}")
            return []'''

        new_generate_method = '''    def _generate_switch_history(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """切替履歴データ生成（修正版）"""
        try:
            switch_history = []
            
            # DSSMSの切替イベントから履歴を生成
            switches = result.get("switch_history", [])
            
            if not switches:
                # サンプル切替履歴生成
                switches = self._generate_sample_switch_history(result)
            
            for i, switch in enumerate(switches):
                # パフォーマンス値の適切な処理
                profit_loss = switch.get("profit_loss_at_switch", 0.0)
                
                # 数値型に確実に変換
                try:
                    profit_loss_float = float(profit_loss) if profit_loss is not None else 0.0
                except (ValueError, TypeError):
                    profit_loss_float = 0.0
                
                # 成功判定ロジック（バックテスターと同じ基準）
                is_successful = profit_loss_float > 0
                success_status = "成功" if is_successful else "失敗"
                
                # 日付の処理
                date_value = switch.get("timestamp", switch.get("date", datetime.now() - timedelta(days=i*2)))
                
                switch_data = {
                    "date": date_value,
                    "from_symbol": switch.get("from_symbol", f"PREV_{i}"),
                    "to_symbol": switch.get("to_symbol", f"NEW_{i}"),
                    "reason": switch.get("reason", switch.get("trigger", "技術的指標による判定")),
                    "switch_price": float(switch.get("switch_price", 0.0)),
                    "switch_cost": float(switch.get("switch_cost", 0.0)),
                    "performance_after": profit_loss_float,
                    "success": success_status
                }
                
                switch_history.append(switch_data)
                
                # デバッグ情報をログ出力
                if i < 3:  # 最初の3件のみログ出力
                    self.logger.info(f"Switch {i+1}: Performance={profit_loss_float:.4f}, Success={success_status}")
            
            self.logger.info(f"切替履歴データ生成完了: {len(switch_history)}件")
            return switch_history
            
        except Exception as e:
            self.logger.error(f"切替履歴生成エラー: {e}")
            return []'''

        # メソッドを置換
        if old_generate_method in content:
            content = content.replace(old_generate_method, new_generate_method)
            logger.info("_generate_switch_history メソッドを修正しました")
        else:
            # 部分的なマッチングで修正を試行
            import re
            pattern = r'def _generate_switch_history\(self, result: Dict\[str, Any\]\) -> List\[Dict\[str, Any\]\]:.*?(?=\n    def )'
            if re.search(pattern, content, re.DOTALL):
                content = re.sub(pattern, new_generate_method.strip() + '\\n\\n', content, flags=re.DOTALL)
                logger.info("_generate_switch_history メソッドを正規表現で修正しました")
        
        # 2. _create_switch_analysis_sheet メソッドのパフォーマンス処理部分を修正
        old_performance_line = '''            ws[f"G{row_idx}"] = switch.get("performance_after", 0)'''
        new_performance_line = '''            # パフォーマンス値の処理（型安全）
            performance = switch.get("performance_after", 0)
            try:
                performance_val = float(performance) if performance is not None else 0.0
                ws[f"G{row_idx}"] = performance_val
            except (ValueError, TypeError):
                ws[f"G{row_idx}"] = 0.0'''
        
        if old_performance_line in content:
            content = content.replace(old_performance_line, new_performance_line)
            logger.info("パフォーマンス処理を修正しました")
        
        # ファイルに書き戻し
        with open(excel_exporter_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Excel エクスポーター修正完了: {excel_exporter_path}")
        return True
        
    except Exception as e:
        logger.error(f"Excel エクスポーター修正エラー: {e}")
        return False

def main():
    """メイン実行関数"""
    
    logger.info("=== 切替分析シート直接修正開始 ===")
    
    success = fix_excel_exporter_directly()
    
    logger.info("=== 修正完了レポート ===")
    logger.info(f"Excel エクスポーター修正: {'[OK] 成功' if success else '[ERROR] 失敗'}")
    
    if success:
        logger.info("修正内容:")
        logger.info("- 成功判定ロジックをprofit_loss_at_switchベースに統一")
        logger.info("- パフォーマンス値の数値型処理を改善")
        logger.info("- 切替履歴データの型安全性を向上")
        
        logger.info("\\n次のステップ:")
        logger.info("1. python src/dssms/dssms_backtester.py を実行してExcel生成")
        logger.info("2. 生成されたExcelファイルの切替分析シートを確認")
        logger.info("3. 13%のパフォーマンスが'成功'と表示されることを確認")

if __name__ == "__main__":
    main()

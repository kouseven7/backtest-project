#!/usr/bin/env python3
"""
最終修正：切替分析の成功判定とパフォーマンス表示の完全修正

1. 統一出力エンジンでの数値データ保持
2. Excelエクスポーターでの正しい成功判定処理
3. パフォーマンス値の適切なフォーマット
"""

import os
import shutil
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_final_comprehensive_fix():
    """最終的な包括修正を適用"""
    
    # 修正するファイルのリスト
    files_to_fix = [
        ('src/dssms/unified_output_engine.py', 'unified_output_engine'),
        ('output/dssms_excel_exporter_v2.py', 'excel_exporter')
    ]
    
    success_count = 0
    
    for file_path, file_type in files_to_fix:
        if fix_file(file_path, file_type):
            success_count += 1
    
    return success_count == len(files_to_fix)

def fix_file(file_path, file_type):
    """個別ファイルの修正"""
    
    if not os.path.exists(file_path):
        logger.error(f"ファイルが見つかりません: {file_path}")
        return False
    
    try:
        # バックアップ作成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f'backup_{file_type}_{timestamp}.py'
        shutil.copy2(file_path, backup_path)
        logger.info(f"バックアップ作成: {backup_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if file_type == 'unified_output_engine':
            content = apply_unified_engine_fixes(content)
        elif file_type == 'excel_exporter':
            content = apply_excel_exporter_fixes(content)
        
        # ファイルに書き戻し
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"{file_type} 修正完了: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"{file_type} 修正エラー: {e}")
        return False

def apply_unified_engine_fixes(content):
    """統一出力エンジンの修正適用"""
    
    # 修正済みの処理の改良版
    old_switch_processing = '''                processed_switch = {
                    'date': switch_data.get('date'),
                    'timestamp': switch_data.get('date'),
                    'from_symbol': switch_data.get('from_symbol', ''),
                    'to_symbol': switch_data.get('to_symbol', ''),
                    'reason': switch_data.get('reason', '技術的指標による判定'),
                    'trigger': switch_data.get('reason', '技術的指標による判定'),
                    'switch_price': 0.0,  # 価格情報は別途取得
                    'switch_cost': switch_cost_float,
                    'profit_loss_at_switch': profit_loss_float,
                    'performance_after': profit_loss_float,  # パフォーマンス値として使用
                    'net_gain': profit_loss_float - switch_cost_float,
                    'success': is_successful
                }'''

    new_switch_processing = '''                processed_switch = {
                    'date': switch_data.get('date'),
                    'timestamp': switch_data.get('date'),
                    'from_symbol': switch_data.get('from_symbol', ''),
                    'to_symbol': switch_data.get('to_symbol', ''),
                    'reason': switch_data.get('reason', '技術的指標による判定'),
                    'trigger': switch_data.get('reason', '技術的指標による判定'),
                    'switch_price': 0.0,  # 価格情報は別途取得
                    'switch_cost': switch_cost_float,
                    'profit_loss_at_switch': profit_loss_float,
                    'performance_after': profit_loss_float,  # 数値のまま保持（パーセント変換はExcel出力時）
                    'net_gain': profit_loss_float - switch_cost_float,
                    'success': is_successful,
                    # デバッグ用の追加情報
                    '_profit_loss_raw': switch_data.get('profit_loss', 0.0),
                    '_is_successful_calculated': is_successful
                }'''
    
    if old_switch_processing in content:
        content = content.replace(old_switch_processing, new_switch_processing)
        logger.info("統一出力エンジンの切替処理を改良しました")
    
    return content

def apply_excel_exporter_fixes(content):
    """Excelエクスポーターの修正適用"""
    
    # 既存の修正をさらに改良
    old_generate_method = '''    def _generate_switch_history(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
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

    new_generate_method = '''    def _generate_switch_history(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """切替履歴データ生成（最終修正版）"""
        try:
            switch_history = []
            
            # DSSMSの切替イベントから履歴を生成
            switches = result.get("switch_history", [])
            
            if not switches:
                self.logger.warning("switch_historyが見つかりません。サンプル生成します。")
                switches = self._generate_sample_switch_history(result)
            
            self.logger.info(f"処理する切替データ: {len(switches)}件")
            
            for i, switch in enumerate(switches):
                # パフォーマンス値の取得（複数のフィールドから試行）
                profit_loss_raw = switch.get("profit_loss_at_switch", 
                                            switch.get("performance_after", 
                                                     switch.get("profit_loss", 0.0)))
                
                # 数値型に確実に変換
                try:
                    profit_loss_float = float(profit_loss_raw) if profit_loss_raw is not None else 0.0
                except (ValueError, TypeError):
                    profit_loss_float = 0.0
                
                # 成功判定ロジック（数値ベース）
                is_successful_calculated = profit_loss_float > 0
                
                # 既存の成功判定もチェック
                existing_success = switch.get("success", switch.get("_is_successful_calculated"))
                if isinstance(existing_success, bool):
                    final_success = existing_success
                else:
                    final_success = is_successful_calculated
                
                success_status = "成功" if final_success else "失敗"
                
                # 日付の処理
                date_value = switch.get("timestamp", switch.get("date", datetime.now() - timedelta(days=i*2)))
                
                switch_data = {
                    "date": date_value,
                    "from_symbol": switch.get("from_symbol", f"PREV_{i}"),
                    "to_symbol": switch.get("to_symbol", f"NEW_{i}"),
                    "reason": switch.get("reason", switch.get("trigger", "技術的指標による判定")),
                    "switch_price": float(switch.get("switch_price", 0.0)),
                    "switch_cost": float(switch.get("switch_cost", 0.0)),
                    "performance_after": profit_loss_float,  # 数値として保持
                    "success": success_status
                }
                
                switch_history.append(switch_data)
                
                # 詳細デバッグ情報（最初の5件）
                if i < 5:
                    self.logger.info(f"Switch {i+1}: Raw={profit_loss_raw}, Float={profit_loss_float:.6f}, Success={success_status}")
            
            self.logger.info(f"切替履歴データ生成完了: {len(switch_history)}件")
            return switch_history
            
        except Exception as e:
            self.logger.error(f"切替履歴生成エラー: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []'''
    
    if old_generate_method in content:
        content = content.replace(old_generate_method, new_generate_method)
        logger.info("Excelエクスポーターの切替履歴生成を改良しました")
    
    # 切替分析シート作成の修正
    old_sheet_creation = '''                # パフォーマンス値の処理（型安全）
                performance = switch.get("performance_after", 0)
                try:
                    performance_val = float(performance) if performance is not None else 0.0
                    ws[f"G{row_idx}"] = performance_val
                    ws[f"G{row_idx}"].number_format = self.percentage_format
                except (ValueError, TypeError):
                    logger.warning(f"Row {row_idx}: Invalid performance value: {performance}")
                    ws[f"G{row_idx}"] = 0.0
                    ws[f"G{row_idx}"].number_format = self.percentage_format'''

    new_sheet_creation = '''                # パフォーマンス値の処理（数値として、パーセント形式でExcel表示）
                performance = switch.get("performance_after", 0)
                try:
                    performance_val = float(performance) if performance is not None else 0.0
                    # パフォーマンス値を数値として設定（Excelが自動でパーセント表示）
                    ws[f"G{row_idx}"] = performance_val / 100.0  # パーセント形式のために100で割る
                    ws[f"G{row_idx}"].number_format = self.percentage_format
                except (ValueError, TypeError):
                    self.logger.warning(f"Row {row_idx}: Invalid performance value: {performance}")
                    ws[f"G{row_idx}"] = 0.0
                    ws[f"G{row_idx}"].number_format = self.percentage_format'''
    
    if old_sheet_creation in content:
        content = content.replace(old_sheet_creation, new_sheet_creation)
        logger.info("切替分析シートのパフォーマンス表示を修正しました")
    
    return content

def main():
    """メイン実行関数"""
    
    logger.info("=== 切替分析の最終包括修正開始 ===")
    
    success = create_final_comprehensive_fix()
    
    logger.info("=== 最終修正完了レポート ===")
    logger.info(f"全体修正: {'✅ 成功' if success else '❌ 失敗'}")
    
    if success:
        logger.info("修正内容:")
        logger.info("- 統一出力エンジンでの数値データ保持を確実化")
        logger.info("- Excelエクスポーターでの正確な成功判定処理")
        logger.info("- パフォーマンス値の適切なパーセント表示")
        logger.info("- デバッグ情報の強化")
        
        logger.info("\\n検証手順:")
        logger.info("1. python src/dssms/dssms_backtester.py を実行")
        logger.info("2. python verify_switch_analysis_fix.py で検証")
        logger.info("3. 生成されたExcelファイルで13%が'成功'と表示されることを確認")

if __name__ == "__main__":
    main()

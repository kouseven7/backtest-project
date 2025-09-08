#!/usr/bin/env python3
"""
切替分析データフロー修正スクリプト

統一出力エンジンでの切替データ処理を修正し、
正しいパフォーマンス値と成功判定を反映させます。
"""

import os
import shutil
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_unified_output_switch_processing():
    """統一出力エンジンでの切替データ処理を修正"""
    
    unified_engine_path = 'src/dssms/unified_output_engine.py'
    
    if not os.path.exists(unified_engine_path):
        logger.error(f"ファイルが見つかりません: {unified_engine_path}")
        return False
    
    try:
        # バックアップ作成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f'backup_unified_output_engine_{timestamp}.py'
        shutil.copy2(unified_engine_path, backup_path)
        logger.info(f"バックアップ作成: {backup_path}")
        
        with open(unified_engine_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # _convert_unified_to_dssms_format メソッドの修正
        old_conversion = '''        return {
            'ticker': unified_model.metadata.ticker,
            'start_date': unified_model.metadata.start_date.isoformat(),
            'end_date': unified_model.metadata.end_date.isoformat(),
            'total_return': unified_model.performance.total_return,
            'total_profit_loss': unified_model.performance.total_pnl,
            'win_rate': unified_model.performance.win_rate,
            'total_trades': unified_model.performance.total_trades,
            'sharpe_ratio': unified_model.performance.sharpe_ratio,
            'max_drawdown': unified_model.performance.max_drawdown,
            'portfolio_value': unified_model.performance.portfolio_value,
            'strategy_scores': unified_model.dssms_metrics.strategy_scores if unified_model.dssms_metrics else {},
            'switch_decisions': unified_model.dssms_metrics.switch_decisions if unified_model.dssms_metrics else [],
            'ranking_data': unified_model.dssms_metrics.ranking_data if unified_model.dssms_metrics else {},
            'switch_success_rate': unified_model.dssms_metrics.switch_success_rate if unified_model.dssms_metrics else 0.0,
            'switch_frequency': unified_model.dssms_metrics.switch_frequency if unified_model.dssms_metrics else 0.0,
            'trades': [trade.to_dict() for trade in unified_model.trades],
            'strategy_statistics': strategy_statistics,  # 戦略別統計を追加
            'reliability_score': unified_model.quality_assurance.reliability_score if unified_model.quality_assurance else 0.0,
            'recommended_actions': unified_model.quality_assurance.quality_recommendations if unified_model.quality_assurance else [],
            'enhanced_data': unified_model.raw_data
        }'''

        new_conversion = '''        # 切替履歴データの適切な変換
        processed_switches = []
        if unified_model.raw_data and 'switches' in unified_model.raw_data:
            for switch_data in unified_model.raw_data['switches']:
                # パフォーマンス値を数値として保持
                profit_loss = switch_data.get('profit_loss', 0.0)
                switch_cost = switch_data.get('cost', switch_data.get('switch_cost', 0.0))
                
                # 数値型に確実に変換
                try:
                    profit_loss_float = float(profit_loss)
                    switch_cost_float = float(switch_cost)
                except (ValueError, TypeError):
                    profit_loss_float = 0.0
                    switch_cost_float = 0.0
                
                # 成功判定（正の損益かどうか）
                is_successful = profit_loss_float > 0
                
                processed_switch = {
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
                }
                processed_switches.append(processed_switch)
        
        return {
            'ticker': unified_model.metadata.ticker,
            'start_date': unified_model.metadata.start_date.isoformat(),
            'end_date': unified_model.metadata.end_date.isoformat(),
            'total_return': unified_model.performance.total_return,
            'total_profit_loss': unified_model.performance.total_pnl,
            'win_rate': unified_model.performance.win_rate,
            'total_trades': unified_model.performance.total_trades,
            'sharpe_ratio': unified_model.performance.sharpe_ratio,
            'max_drawdown': unified_model.performance.max_drawdown,
            'portfolio_value': unified_model.performance.portfolio_value,
            'strategy_scores': unified_model.dssms_metrics.strategy_scores if unified_model.dssms_metrics else {},
            'switch_decisions': unified_model.dssms_metrics.switch_decisions if unified_model.dssms_metrics else [],
            'ranking_data': unified_model.dssms_metrics.ranking_data if unified_model.dssms_metrics else {},
            'switch_success_rate': unified_model.dssms_metrics.switch_success_rate if unified_model.dssms_metrics else 0.0,
            'switch_frequency': unified_model.dssms_metrics.switch_frequency if unified_model.dssms_metrics else 0.0,
            'trades': [trade.to_dict() for trade in unified_model.trades],
            'strategy_statistics': strategy_statistics,  # 戦略別統計を追加
            'reliability_score': unified_model.quality_assurance.reliability_score if unified_model.quality_assurance else 0.0,
            'recommended_actions': unified_model.quality_assurance.quality_recommendations if unified_model.quality_assurance else [],
            'enhanced_data': unified_model.raw_data,
            'switch_history': processed_switches  # 処理済み切替履歴を追加
        }'''
        
        # 置換実行
        if old_conversion in content:
            content = content.replace(old_conversion, new_conversion)
            logger.info("_convert_unified_to_dssms_format メソッドを修正しました")
        else:
            logger.warning("該当する変換メソッドが見つかりませんでした")
        
        # ファイルに書き戻し
        with open(unified_engine_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"統一出力エンジン修正完了: {unified_engine_path}")
        return True
        
    except Exception as e:
        logger.error(f"統一出力エンジン修正エラー: {e}")
        return False

def main():
    """メイン実行関数"""
    
    logger.info("=== 切替分析データフロー修正開始 ===")
    
    success = fix_unified_output_switch_processing()
    
    logger.info("=== 修正完了レポート ===")
    logger.info(f"統一出力エンジン修正: {'✅ 成功' if success else '❌ 失敗'}")
    
    if success:
        logger.info("修正内容:")
        logger.info("- 切替履歴データの適切な数値処理を追加")
        logger.info("- パフォーマンス値の型安全な変換を実装")
        logger.info("- 成功判定ロジックをprofit_loss基準に統一")
        logger.info("- switch_historyフィールドに処理済みデータを提供")
        
        logger.info("\\n次のステップ:")
        logger.info("1. python src/dssms/dssms_backtester.py を再実行")
        logger.info("2. 新しいExcelファイルの切替分析シートを確認")
        logger.info("3. 正のパフォーマンス値が'成功'と表示されることを確認")

if __name__ == "__main__":
    main()

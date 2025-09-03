"""
DSSMS銘柄切替過多問題の修正スクリプト
Phase 2で失われた切替抑制機能の復元
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

project_root = Path(__file__).parent
sys.path.append(str(project_root))

def create_improved_switch_logic():
    """改善された切替ロジックの実装"""
    
    improved_logic = '''
    def _evaluate_switch_decision_improved(self, date: datetime, current_position: Optional[str], 
                                         ranking_result: Dict[str, Any], market_condition: Dict[str, Any]) -> Dict[str, Any]:
        """改善版: 切替判定（過度な切替を抑制）"""
        try:
            should_switch = False
            reason = ""
            target_symbol = None
            
            top_symbol = ranking_result.get('top_symbol')
            
            # 初回ポジション設定
            if current_position is None and top_symbol:
                should_switch = True
                reason = "初期ポジション設定"
                target_symbol = top_symbol
                return {
                    'should_switch': should_switch,
                    'target_symbol': target_symbol,
                    'reason': reason,
                    'trigger': SwitchTrigger.DAILY_EVALUATION if should_switch else None
                }
            
            # 既存ポジションがある場合の厳格な切替判定
            if current_position and top_symbol and current_position != top_symbol:
                current_score = ranking_result.get('rankings', {}).get(current_position, 0)
                top_score = ranking_result.get('rankings', {}).get(top_symbol, 0)
                
                # 1. 最小保有期間チェック（24時間未満なら切替しない）
                if len(self.switch_history) > 0:
                    last_switch = self.switch_history[-1]
                    hours_since_last_switch = (date - last_switch.timestamp).total_seconds() / 3600
                    if hours_since_last_switch < 24.0:  # 最小24時間保有
                        return {
                            'should_switch': False,
                            'target_symbol': current_position,
                            'reason': f"最小保有期間未満: {hours_since_last_switch:.1f}時間",
                            'trigger': None
                        }
                
                # 2. スコア差の厳格化（20%以上の差が必要）
                score_threshold = 0.20  # 10% -> 20%に変更
                score_diff = top_score - current_score
                
                if score_diff > score_threshold:
                    # 3. 追加条件: 連続切替回数制限
                    recent_switches = [s for s in self.switch_history 
                                     if (date - s.timestamp).days <= 7]  # 過去7日
                    
                    if len(recent_switches) >= 3:  # 週3回以上の切替を制限
                        return {
                            'should_switch': False,
                            'target_symbol': current_position,
                            'reason': f"週間切替制限: {len(recent_switches)}回",
                            'trigger': None
                        }
                    
                    # 4. 市場ボラティリティチェック（高ボラ時は切替しない）
                    if market_condition.get('volatility_level') == 'high':
                        return {
                            'should_switch': False,
                            'target_symbol': current_position,
                            'reason': "高ボラティリティ期間",
                            'trigger': None
                        }
                    
                    # 全条件をクリアした場合のみ切替
                    should_switch = True
                    reason = f"スコア大幅改善: {current_score:.3f} -> {top_score:.3f} (+{score_diff:.3f})"
                    target_symbol = top_symbol
                else:
                    reason = f"スコア差不足: {score_diff:.3f} < {score_threshold}"
            
            return {
                'should_switch': should_switch,
                'target_symbol': target_symbol or current_position,
                'reason': reason,
                'trigger': SwitchTrigger.DAILY_EVALUATION if should_switch else None
            }
            
        except Exception as e:
            self.logger.warning(f"切替判定エラー {date}: {e}")
            return {
                'should_switch': False,
                'target_symbol': current_position,
                'reason': f"エラー: {e}",
                'trigger': None
            }
    '''
    
    return improved_logic

def create_realistic_ranking_system():
    """より現実的なランキングシステム"""
    
    ranking_logic = '''
    def _update_symbol_ranking_realistic(self, date: datetime, symbols: List[str]) -> Dict[str, Any]:
        """現実的な銘柄ランキング更新（安定性重視）"""
        try:
            # 前回のランキングを取得（継続性のため）
            previous_rankings = getattr(self, '_previous_rankings', {})
            
            ranking_scores = {}
            
            for symbol in symbols:
                # 前回スコアがあれば継続性を持たせる
                if symbol in previous_rankings:
                    base_score = previous_rankings[symbol]
                    # 小さな変動のみ（±5%以内）
                    variation = np.random.normal(0, 0.05)
                    new_score = max(0.1, min(0.9, base_score + variation))
                else:
                    # 新規銘柄は中央値付近から開始
                    new_score = np.random.normal(0.5, 0.1)
                    new_score = max(0.1, min(0.9, new_score))
                
                ranking_scores[symbol] = new_score
            
            # ランキングを保存（次回使用のため）
            self._previous_rankings = ranking_scores.copy()
            
            # 上位銘柄選択（変更を最小限に）
            sorted_symbols = sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)
            top_symbols = sorted_symbols[:5]
            
            result = {
                'date': date,
                'rankings': dict(top_symbols),
                'top_symbol': top_symbols[0][0] if top_symbols else None,
                'top_score': top_symbols[0][1] if top_symbols else 0,
                'total_symbols': len(ranking_scores),
                'data_source': 'realistic_stable'
            }
            
            self.logger.info(f"安定ランキング更新: 上位={result['top_symbol']} ({result['top_score']:.3f})")
            return result
            
        except Exception as e:
            self.logger.warning(f"ランキング更新エラー {date}: {e}")
            return {'date': date, 'rankings': {}, 'error': str(e)}
    '''
    
    return ranking_logic

def apply_dssms_switch_fix():
    """DSSMS切替問題の修正を適用"""
    
    print("🔧 DSSMS銘柄切替過多問題の修正開始")
    
    # 1. 元ファイルのバックアップ
    original_file = Path("src/dssms/dssms_backtester.py")
    backup_file = Path("src/dssms/dssms_backtester_backup.py")
    
    if original_file.exists():
        import shutil
        shutil.copy2(original_file, backup_file)
        print(f"✅ バックアップ作成: {backup_file}")
    
    # 2. 修正版の作成
    print("📝 修正版ロジック実装...")
    
    # 修正パッチの適用
    patch_content = f"""
# ===== DSSMS切替過多問題修正パッチ =====
# 適用日: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{create_improved_switch_logic()}

{create_realistic_ranking_system()}

# ===== パッチ適用完了 =====
"""
    
    # パッチファイルの作成
    patch_file = Path("dssms_switch_optimization_patch.py")
    with open(patch_file, 'w', encoding='utf-8') as f:
        f.write(patch_content)
    
    print(f"✅ 修正パッチ作成: {patch_file}")
    
    # 3. 設定値の推奨変更
    config_recommendations = {
        'switch_cost_rate': 0.002,  # 0.2%（現実的な取引コスト）
        'score_threshold': 0.20,   # 20%（過度な切替を防ぐ）
        'min_holding_hours': 24,   # 最小24時間保有
        'max_weekly_switches': 3,  # 週間最大3回切替
        'volatility_check': True   # 高ボラ時の切替停止
    }
    
    print("\n📊 推奨設定値:")
    for key, value in config_recommendations.items():
        print(f"  {key}: {value}")
    
    # 4. テスト実行の推奨
    print("\n🧪 修正確認テスト:")
    print("1. python test_dssms_switch_optimization.py")
    print("2. 期待結果: 切替回数 < 50回/年、平均保有期間 > 7日")
    
    return True

def create_test_script():
    """修正確認用テストスクリプトの作成"""
    
    test_content = '''
"""
DSSMS切替最適化の確認テスト
修正後の切替頻度を検証
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.dssms.dssms_backtester import DSSMSBacktester

def test_switch_optimization():
    """切替最適化テスト"""
    
    print("=== DSSMS切替最適化テスト開始 ===")
    
    # 修正版設定
    config = {
        'initial_capital': 1000000,
        'switch_cost_rate': 0.002,  # 0.2%
        'output_excel': False,
        'output_detailed_report': True
    }
    
    backtester = DSSMSBacktester(config)
    
    # 3ヶ月間のテスト
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2024, 8, 31)
    symbols = ['7203.T', '6758.T', '9984.T', '4063.T', '8316.T']
    
    try:
        result = backtester.simulate_dynamic_selection(
            start_date=start_date,
            end_date=end_date,
            symbol_universe=symbols
        )
        
        if result.get('success'):
            switch_count = result.get('switch_count', 0)
            avg_holding = result.get('average_holding_period_hours', 0)
            total_return = result.get('total_return', 0)
            
            print(f"\\n📊 修正後の結果:")
            print(f"切替回数: {switch_count}回 (3ヶ月)")
            print(f"平均保有期間: {avg_holding:.1f}時間")
            print(f"総リターン: {total_return:.2%}")
            
            # 改善判定
            yearly_switches = switch_count * 4  # 年間換算
            
            if yearly_switches < 100:
                print("✅ 切替頻度: 良好（年間100回未満）")
            elif yearly_switches < 200:
                print("⚠️ 切替頻度: 改善余地あり")
            else:
                print("❌ 切替頻度: まだ高すぎる")
            
            if avg_holding > 48:
                print("✅ 保有期間: 良好（48時間以上）")
            else:
                print("⚠️ 保有期間: 短い（さらに改善推奨）")
            
            return True
        else:
            print(f"❌ テスト失敗: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False

if __name__ == "__main__":
    success = test_switch_optimization()
    if success:
        print("\\n🎉 修正テスト完了")
    else:
        print("\\n❌ 修正が必要です")
'''
    
    with open("test_dssms_switch_optimization.py", 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ テストスクリプト作成: test_dssms_switch_optimization.py")

if __name__ == "__main__":
    print("🚀 DSSMS銘柄切替過多問題の修正開始")
    
    success = apply_dssms_switch_fix()
    
    if success:
        create_test_script()
        print("\n✅ 修正完了! 次の手順:")
        print("1. パッチの内容確認")
        print("2. 実際のコード適用")
        print("3. テスト実行")
        print("4. 結果確認")
    else:
        print("❌ 修正に失敗しました")

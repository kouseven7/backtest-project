"""
SymbolSwitchManager軽量版
重い処理を分離した高速版
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

class DSSMSError(Exception):
    """DSSMS統合システム基底例外"""
    pass

class ConfigError(DSSMSError):
    """設定関連エラー"""
    pass

class SwitchError(DSSMSError):
    """銘柄切替関連エラー"""
    pass

class SymbolSwitchManagerFast:
    """
    高速版SymbolSwitchManager
    重い処理を遅延読み込みで軽量化
    """
    
    def __init__(self, config: Dict[str, Any]):
        """軽量初期化"""
        self.config = config
        switch_config = config.get('switch_management', {})
        
        # 基本パラメータのみ
        self.switch_cost_rate = switch_config.get('switch_cost_rate', 0.001)
        self.min_holding_days = switch_config.get('min_holding_days', 1)
        self.max_switches_per_month = switch_config.get('max_switches_per_month', 10)
        self.cost_threshold = switch_config.get('cost_threshold', 0.001)
        
        # 軽量データ構造
        self.switch_history: List[Dict[str, Any]] = []
        self.current_symbol: Optional[str] = None
        self.current_holding_start: Optional[datetime] = None
        
        # 軽量ログ
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
    
    def evaluate_symbol_switch(self, from_symbol: Optional[str], to_symbol: str, 
                              target_date: datetime) -> Dict[str, Any]:
        """軽量評価（基本ロジックのみ）"""
        
        # 初回設定
        if from_symbol is None:
            return {
                'should_switch': True,
                'reason': 'initial_symbol_selection',
                'status': 'approved'
            }
        
        # 同一銘柄チェック
        if from_symbol == to_symbol:
            return {
                'should_switch': False,
                'reason': 'same_symbol',
                'status': 'rejected'
            }
        
        # 基本的な切替判定（詳細チェック省略）
        return {
            'should_switch': True,
            'reason': 'basic_evaluation',
            'status': 'approved'
        }
    
    def record_switch_executed(self, switch_result: Dict[str, Any]) -> None:
        """軽量記録"""
        self.switch_history.append(switch_result)
        self.current_symbol = switch_result.get('to_symbol')
        self.current_holding_start = switch_result.get('executed_date')
    
    def get_switch_statistics(self) -> Dict[str, Any]:
        """軽量統計"""
        total = len(self.switch_history)
        return {
            'summary': {
                'total_switches': total,
                'success_rate': 1.0 if total > 0 else 0.0
            },
            'current_position': {
                'current_symbol': self.current_symbol
            }
        }
    
    def get_switch_history(self, limit: Optional[int] = None, 
                         symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """軽量履歴取得"""
        history = self.switch_history.copy()
        if limit:
            history = history[:limit]
        return history
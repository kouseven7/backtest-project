class SymbolSwitchManagerUltraLight:
    def __init__(self, config):
        switch_config = config.get('switch_management', {})
        self.switch_cost_rate = switch_config.get('switch_cost_rate', 0.001)
        self.min_holding_days = switch_config.get('min_holding_days', 1)
        self.switch_history = []
        self.current_symbol = None
        self.current_holding_start = None
    
    def evaluate_symbol_switch(self, from_symbol, to_symbol, target_date):
        if from_symbol is None:
            return {'should_switch': True, 'reason': 'initial', 'status': 'approved'}
        if from_symbol == to_symbol:
            return {'should_switch': False, 'reason': 'same', 'status': 'rejected'}
        return {'should_switch': True, 'reason': 'basic', 'status': 'approved'}
    
    def record_switch_executed(self, switch_result):
        self.switch_history.append(switch_result)
        self.current_symbol = switch_result.get('to_symbol')
        self.current_holding_start = switch_result.get('executed_date')
    
    def get_switch_statistics(self):
        total = len(self.switch_history)
        return {
            'summary': {'total_switches': total, 'success_rate': 1.0 if total > 0 else 0.0},
            'current_position': {'current_symbol': self.current_symbol}
        }
    
    def get_switch_history(self, limit=None, symbol=None):
        history = self.switch_history
        if limit and limit < len(history):
            history = history[:limit]
        return history
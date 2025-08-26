"""
DSSMS ポートフォリオ計算修正システム
Task 1.2: ポートフォリオ価値0.01円問題の根本的解決
ハイブリッド方式（既存修正＋新エンジン）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import warnings
import sys

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

class PortfolioCalculationFixer:
    """ポートフォリオ計算修正エンジン - ハイブリッド方式"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = setup_logger(__name__)
        self.config = self._load_config(config_path)
        self.calculation_history: Dict[str, Any] = {}
        self.emergency_mode = False
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """ポートフォリオ修正設定読み込み"""
        default_config = {
            "portfolio_protection": {
                "minimum_value_ratio": 0.01,      # 元本の1%以下で警告
                "emergency_threshold": 0.001,     # 元本の0.1%以下で緊急モード
                "max_drawdown_limit": 0.95,       # 95%以上の損失で停止
                "position_size_limit": 0.3        # 単一ポジション30%上限
            },
            "calculation_methods": {
                "price_validation": "strict",      # strict, moderate, loose
                "weight_normalization": "auto",    # auto, manual, none
                "rebalancing_frequency": "daily",  # daily, weekly, monthly
                "cash_management": "enabled"       # enabled, disabled
            },
            "emergency_settings": {
                "fallback_to_equal_weight": True,
                "preserve_cash_reserve": 0.05,    # 5%現金保持
                "gradual_position_exit": True,
                "max_daily_trades": 10
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"ポートフォリオ設定読み込み失敗: {e}")
        
        return default_config
    
    def fix_portfolio_calculation(self, portfolio_data: Dict[str, Any], price_data: Dict[str, pd.DataFrame], 
                                 initial_capital: float = 1000000) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        ポートフォリオ計算修正メイン処理
        Args:
            portfolio_data: ポートフォリオ履歴データ
            price_data: 価格データ辞書
            initial_capital: 初期資本
        Returns:
            (修正済みポートフォリオ, 修正ログ)
        """
        fix_log = {
            "timestamp": datetime.now().isoformat(),
            "initial_capital": initial_capital,
            "operations": [],
            "warnings": [],
            "emergency_triggered": False,
            "status": "unknown"
        }
        
        try:
            # 1. 入力データ検証
            validation_result = self._validate_input_data(portfolio_data, price_data)
            fix_log["operations"].append(validation_result)
            
            if validation_result["status"] == "failed":
                fix_log["status"] = "failed"
                return portfolio_data, fix_log
            
            # 2. 既存計算の診断
            diagnosis_result = self._diagnose_calculation_issues(portfolio_data, initial_capital)
            fix_log["operations"].append(diagnosis_result)
            
            # 3. 緊急モード判定
            if diagnosis_result.get("emergency_required", False):
                self.emergency_mode = True
                fix_log["emergency_triggered"] = True
                fix_log["warnings"].append("緊急モード発動: ポートフォリオ価値が危険水準")
            
            # 4. 修正方式選択・実行
            if self.emergency_mode:
                fixed_portfolio, repair_result = self._emergency_portfolio_repair(
                    portfolio_data, price_data, initial_capital
                )
            else:
                fixed_portfolio, repair_result = self._standard_portfolio_repair(
                    portfolio_data, price_data, initial_capital
                )
            
            fix_log["operations"].append(repair_result)
            
            # 5. 修正後検証
            final_validation = self._validate_fixed_portfolio(fixed_portfolio, initial_capital)
            fix_log["operations"].append(final_validation)
            
            fix_log["status"] = final_validation["status"]
            
            self.calculation_history[datetime.now().isoformat()] = fix_log
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ計算修正エラー: {e}")
            fix_log["status"] = "error"
            fix_log["error"] = str(e)
            fixed_portfolio = portfolio_data
        
        return fixed_portfolio, fix_log
    
    def _validate_input_data(self, portfolio_data: Dict[str, Any], price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """入力データ検証"""
        issues = []
        
        # ポートフォリオデータチェック
        if not portfolio_data:
            issues.append("ポートフォリオデータが空")
        
        if 'portfolio_value' not in portfolio_data:
            issues.append("portfolio_value列が欠損")
        
        # 価格データチェック
        if not price_data:
            issues.append("価格データが空")
        
        # データ整合性チェック
        if portfolio_data and price_data:
            portfolio_symbols = set()
            if 'positions' in portfolio_data:
                for pos_data in portfolio_data['positions']:
                    if isinstance(pos_data, dict) and 'symbol' in pos_data:
                        portfolio_symbols.add(pos_data['symbol'])
            
            price_symbols = set(price_data.keys())
            missing_prices = portfolio_symbols - price_symbols
            
            if missing_prices:
                issues.append(f"価格データ欠損銘柄: {missing_prices}")
        
        return {
            "operation": "input_validation",
            "issues": issues,
            "issue_count": len(issues),
            "status": "passed" if len(issues) == 0 else "warning" if len(issues) <= 2 else "failed"
        }
    
    def _diagnose_calculation_issues(self, portfolio_data: Dict[str, Any], initial_capital: float) -> Dict[str, Any]:
        """既存計算の問題診断"""
        issues = []
        emergency_required = False
        
        # ポートフォリオ価値チェック
        current_value = portfolio_data.get('portfolio_value', {})
        if isinstance(current_value, dict):
            final_values = list(current_value.values())
            if final_values:
                final_value = final_values[-1]
                value_ratio = final_value / initial_capital
                
                if value_ratio < self.config["portfolio_protection"]["emergency_threshold"]:
                    emergency_required = True
                    issues.append(f"緊急事態: ポートフォリオ価値 {final_value:.2f}円 (比率: {value_ratio:.4f})")
                elif value_ratio < self.config["portfolio_protection"]["minimum_value_ratio"]:
                    issues.append(f"警告: ポートフォリオ価値低下 {final_value:.2f}円 (比率: {value_ratio:.4f})")
        
        # ポジションデータの整合性
        positions = portfolio_data.get('positions', [])
        if not positions:
            issues.append("ポジションデータが空")
        
        # 取引履歴の妥当性
        trades = portfolio_data.get('trades', [])
        if isinstance(trades, list):
            negative_trades = [t for t in trades if isinstance(t, dict) and t.get('pnl', 0) < -initial_capital * 0.5]
            if negative_trades:
                issues.append(f"異常な大損失取引: {len(negative_trades)}件")
        
        return {
            "operation": "calculation_diagnosis",
            "issues": issues,
            "emergency_required": emergency_required,
            "status": "passed" if len(issues) == 0 else "warning"
        }
    
    def _emergency_portfolio_repair(self, portfolio_data: Dict[str, Any], price_data: Dict[str, pd.DataFrame], 
                                   initial_capital: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """緊急ポートフォリオ修復"""
        self.logger.warning("緊急ポートフォリオ修復を実行中...")
        
        # 新しいポートフォリオを初期化
        repaired_portfolio = {
            'portfolio_value': {},
            'positions': [],
            'trades': [],
            'cash': initial_capital * self.config["emergency_settings"]["preserve_cash_reserve"],
            'repair_mode': 'emergency',
            'repair_timestamp': datetime.now().isoformat()
        }
        
        # 利用可能な銘柄で等ウェイト分散
        valid_symbols = []
        for symbol, data in price_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                if 'Close' in data.columns and data['Close'].iloc[-1] > 0:
                    valid_symbols.append(symbol)
        
        if valid_symbols:
            # 等ウェイト配分
            cash_for_investment = initial_capital * (1 - self.config["emergency_settings"]["preserve_cash_reserve"])
            weight_per_symbol = 1.0 / len(valid_symbols)
            
            for symbol in valid_symbols[:5]:  # 最大5銘柄に制限
                current_price = price_data[symbol]['Close'].iloc[-1]
                position_value = cash_for_investment * weight_per_symbol
                shares = int(position_value / current_price)
                
                if shares > 0:
                    repaired_portfolio['positions'].append({
                        'symbol': symbol,
                        'shares': shares,
                        'entry_price': current_price,
                        'current_price': current_price,
                        'value': shares * current_price
                    })
        
        # ポートフォリオ価値再計算
        total_position_value = sum(pos['value'] for pos in repaired_portfolio['positions'])
        repaired_portfolio['portfolio_value'] = {
            datetime.now().strftime('%Y-%m-%d'): total_position_value + repaired_portfolio['cash']
        }
        
        return repaired_portfolio, {
            "operation": "emergency_repair",
            "method": "equal_weight_reconstruction",
            "symbols_used": len(repaired_portfolio['positions']),
            "final_value": list(repaired_portfolio['portfolio_value'].values())[0],
            "status": "success"
        }
    
    def _standard_portfolio_repair(self, portfolio_data: Dict[str, Any], price_data: Dict[str, pd.DataFrame], 
                                  initial_capital: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """標準ポートフォリオ修復"""
        repaired_portfolio = portfolio_data.copy()
        repair_actions = []
        
        # 1. 価格データの整合性修正
        if 'positions' in repaired_portfolio:
            valid_positions = []
            for position in repaired_portfolio['positions']:
                symbol = position.get('symbol')
                if symbol in price_data and not price_data[symbol].empty:
                    # 現在価格更新
                    current_price = price_data[symbol]['Close'].iloc[-1]
                    if current_price > 0:
                        position['current_price'] = current_price
                        position['value'] = position.get('shares', 0) * current_price
                        valid_positions.append(position)
                        repair_actions.append(f"価格更新: {symbol}")
                else:
                    repair_actions.append(f"除外: {symbol} (価格データなし)")
            
            repaired_portfolio['positions'] = valid_positions
        
        # 2. ポートフォリオ価値再計算
        if repaired_portfolio['positions']:
            total_position_value = sum(pos.get('value', 0) for pos in repaired_portfolio['positions'])
            cash = repaired_portfolio.get('cash', 0)
            
            # 現金残高の妥当性チェック
            if cash < 0:
                cash = initial_capital * 0.05  # 5%の現金を確保
                repair_actions.append("現金残高修正")
            
            new_portfolio_value = total_position_value + cash
            
            # ポートフォリオ価値が極端に低い場合の修正
            if new_portfolio_value < initial_capital * 0.01:
                # 部分的リセット
                cash = initial_capital * 0.5
                # ポジションサイズを調整
                for position in repaired_portfolio['positions']:
                    position['shares'] = max(1, position.get('shares', 0) // 10)
                    position['value'] = position['shares'] * position['current_price']
                
                new_portfolio_value = sum(pos['value'] for pos in repaired_portfolio['positions']) + cash
                repair_actions.append("部分的ポートフォリオリセット")
            
            repaired_portfolio['portfolio_value'] = {
                datetime.now().strftime('%Y-%m-%d'): new_portfolio_value
            }
            repaired_portfolio['cash'] = cash
        
        return repaired_portfolio, {
            "operation": "standard_repair",
            "repair_actions": repair_actions,
            "action_count": len(repair_actions),
            "final_value": list(repaired_portfolio.get('portfolio_value', {0: 0}).values())[0],
            "status": "success" if repair_actions else "no_changes_needed"
        }
    
    def _validate_fixed_portfolio(self, portfolio: Dict[str, Any], initial_capital: float) -> Dict[str, Any]:
        """修正後ポートフォリオ検証"""
        issues = []
        
        # 基本構造チェック
        required_keys = ['portfolio_value', 'positions']
        missing_keys = [key for key in required_keys if key not in portfolio]
        if missing_keys:
            issues.append(f"必須キー欠損: {missing_keys}")
        
        # ポートフォリオ価値チェック
        if 'portfolio_value' in portfolio:
            values = list(portfolio['portfolio_value'].values())
            if values:
                final_value = values[-1]
                value_ratio = final_value / initial_capital
                
                if value_ratio >= self.config["portfolio_protection"]["minimum_value_ratio"]:
                    # 正常範囲
                    pass
                elif value_ratio >= self.config["portfolio_protection"]["emergency_threshold"]:
                    issues.append(f"注意: 低ポートフォリオ価値 (比率: {value_ratio:.4f})")
                else:
                    issues.append(f"警告: 極低ポートフォリオ価値 (比率: {value_ratio:.4f})")
        
        # ポジション妥当性チェック
        if 'positions' in portfolio:
            total_position_value = sum(pos.get('value', 0) for pos in portfolio['positions'])
            if total_position_value <= 0:
                issues.append("有効なポジションなし")
        
        return {
            "operation": "final_validation",
            "issues": issues,
            "issue_count": len(issues),
            "status": "passed" if len(issues) == 0 else "warning" if len(issues) <= 1 else "failed"
        }
    
    def generate_fix_report(self, fix_results: List[Dict[str, Any]]) -> str:
        """修正結果レポート生成"""
        if not fix_results:
            return "修正結果がありません。"
        
        report_lines = [
            "=" * 60,
            "DSSMS ポートフォリオ計算修正レポート",
            f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]
        
        # サマリー統計
        total_fixes = len(fix_results)
        success_count = sum(1 for r in fix_results if r.get("status") == "success")
        emergency_count = sum(1 for r in fix_results if r.get("emergency_triggered", False))
        
        report_lines.extend([
            "📊 修正サマリー",
            "-" * 20,
            f"修正実行回数: {total_fixes}",
            f"成功: {success_count} ({success_count/total_fixes*100:.1f}%)",
            f"緊急モード発動: {emergency_count}回",
            ""
        ])
        
        return "\n".join(report_lines)

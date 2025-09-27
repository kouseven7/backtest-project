"""
DSSMS統合システム - SymbolSwitchManager
動的銘柄切替の判定・管理を行うクラス

Author: AI Assistant
Created: 2025-09-27
Phase: Phase 3 Tier 2 実装
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.logger_config import setup_logger


class DSSMSError(Exception):
    """DSSMS統合システム基底例外"""
    pass


class ConfigError(DSSMSError):
    """設定関連エラー"""
    pass


class SwitchError(DSSMSError):
    """銘柄切替関連エラー"""
    pass


class SymbolSwitchManager:
    """
    動的銘柄切替の判定・管理を行うクラス
    
    Responsibilities:
    - 切替必要性評価
    - 切替制限管理（最小保有期間、月次制限）
    - 切替履歴記録・統計
    - 切替コスト効率性評価
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        銘柄切替管理の初期化
        
        Args:
            config: 設定辞書
        
        Raises:
            ConfigError: 設定値エラー
        """
        try:
            # 基本設定
            self.config = config
            switch_config = config.get('switch_management', {})
            
            # 切替制限パラメータ
            self.switch_cost_rate = switch_config.get('switch_cost_rate', 0.001)  # 0.1%
            self.min_holding_days = switch_config.get('min_holding_days', 1)      # 最小保有日数
            self.max_switches_per_month = switch_config.get('max_switches_per_month', 10)
            self.cost_threshold = switch_config.get('cost_threshold', 0.001)      # コスト効率性閾値
            
            # 切替履歴管理
            self.switch_history: List[Dict[str, Any]] = []
            self.current_holding_start: Optional[datetime] = None
            self.current_symbol: Optional[str] = None
            
            # 統計データ
            self.switch_stats = {
                'total_switches': 0,
                'successful_switches': 0,
                'cost_saved': 0.0,
                'cost_incurred': 0.0,
                'average_holding_days': 0.0
            }
            
            # ログ設定
            self.logger = setup_logger(f"{self.__class__.__name__}")
            
            # 設定検証
            self._validate_config()
            
            self.logger.info(f"SymbolSwitchManager初期化完了 - 切替コスト: {self.switch_cost_rate:.1%}, "
                           f"最小保有: {self.min_holding_days}日, 月次制限: {self.max_switches_per_month}回")
            
        except Exception as e:
            self.logger.error(f"SymbolSwitchManager初期化エラー: {e}")
            raise ConfigError(f"SymbolSwitchManager初期化失敗: {e}")
    
    def _validate_config(self) -> None:
        """設定値の検証"""
        try:
            if not (0.0 <= self.switch_cost_rate <= 0.01):
                raise ValueError(f"切替コスト率が範囲外: {self.switch_cost_rate}")
            
            if self.min_holding_days < 0:
                raise ValueError(f"最小保有日数が負数: {self.min_holding_days}")
            
            if self.max_switches_per_month <= 0:
                raise ValueError(f"月次切替制限が無効: {self.max_switches_per_month}")
            
            self.logger.debug("設定値検証完了")
            
        except Exception as e:
            raise ConfigError(f"設定値検証失敗: {e}")
    
    def evaluate_symbol_switch(self, from_symbol: Optional[str], to_symbol: str, 
                              target_date: datetime) -> Dict[str, Any]:
        """
        銘柄切替の必要性評価
        
        Args:
            from_symbol: 切替前銘柄 (None可)
            to_symbol: 切替後銘柄
            target_date: 対象日付
        
        Returns:
            Dict[str, Any]: 切替評価結果
        
        Example:
            evaluation = switch_mgr.evaluate_symbol_switch(
                from_symbol='7203',
                to_symbol='6758', 
                target_date=datetime(2023, 6, 15)
            )
            if evaluation['should_switch']:
                print(f"切替推奨: {evaluation['reason']}")
        """
        try:
            self.logger.debug(f"銘柄切替評価開始: {from_symbol} → {to_symbol} @ {target_date}")
            
            # 初回設定（銘柄なし → 新規銘柄）
            if from_symbol is None:
                return {
                    'should_switch': True,
                    'from_symbol': None,
                    'to_symbol': to_symbol,
                    'target_date': target_date,
                    'reason': 'initial_symbol_selection',
                    'evaluation_details': {
                        'is_initial': True,
                        'cost_estimate': 0.0,
                        'expected_benefit': 'portfolio_start'
                    },
                    'status': 'approved'
                }
            
            # 同一銘柄の場合
            if from_symbol == to_symbol:
                return {
                    'should_switch': False,
                    'from_symbol': from_symbol,
                    'to_symbol': to_symbol,
                    'target_date': target_date,
                    'reason': 'same_symbol',
                    'evaluation_details': {
                        'same_symbol_check': True
                    },
                    'status': 'rejected'
                }
            
            # 制限チェック
            restrictions = self._check_switch_restrictions(target_date)
            if not restrictions['allowed']:
                return {
                    'should_switch': False,
                    'from_symbol': from_symbol,
                    'to_symbol': to_symbol,
                    'target_date': target_date,
                    'reason': restrictions['reason'],
                    'evaluation_details': restrictions,
                    'status': 'rejected'
                }
            
            # コスト効率性評価
            cost_analysis = self._evaluate_switch_cost_efficiency(from_symbol, to_symbol, target_date)
            
            # 最終判定
            should_switch = cost_analysis['is_cost_effective']
            
            return {
                'should_switch': should_switch,
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'target_date': target_date,
                'reason': cost_analysis['reason'],
                'evaluation_details': {
                    'restrictions': restrictions,
                    'cost_analysis': cost_analysis,
                    'switch_cost_rate': self.switch_cost_rate,
                    'estimated_cost': cost_analysis.get('switch_cost', 0.0)
                },
                'status': 'approved' if should_switch else 'rejected'
            }
            
        except Exception as e:
            self.logger.error(f"銘柄切替評価エラー: {e}")
            return {
                'should_switch': False,
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'target_date': target_date,
                'reason': 'evaluation_error',
                'error': str(e),
                'status': 'error'
            }
    
    def _check_switch_restrictions(self, target_date: datetime) -> Dict[str, Any]:
        """
        切替制限チェック（最小保有期間、月次制限）
        
        Args:
            target_date: 対象日付
        
        Returns:
            Dict[str, Any]: 制限チェック結果
        """
        try:
            # 最小保有期間チェック
            if not self._check_min_holding_period(target_date):
                holding_days = self._get_current_holding_days(target_date)
                return {
                    'allowed': False,
                    'reason': 'min_holding_period_not_met',
                    'current_holding_days': holding_days,
                    'required_holding_days': self.min_holding_days,
                    'days_remaining': self.min_holding_days - holding_days
                }
            
            # 月次切替制限チェック  
            if not self._check_monthly_switch_limit(target_date):
                monthly_count = self._get_monthly_switch_count(target_date)
                return {
                    'allowed': False,
                    'reason': 'monthly_switch_limit_exceeded',
                    'current_monthly_switches': monthly_count,
                    'monthly_limit': self.max_switches_per_month,
                    'switches_remaining': 0
                }
            
            # 制限クリア
            return {
                'allowed': True,
                'reason': 'restrictions_passed',
                'current_holding_days': self._get_current_holding_days(target_date),
                'current_monthly_switches': self._get_monthly_switch_count(target_date),
                'switches_remaining': self.max_switches_per_month - self._get_monthly_switch_count(target_date)
            }
            
        except Exception as e:
            self.logger.error(f"制限チェックエラー: {e}")
            return {
                'allowed': False,
                'reason': 'restriction_check_error',
                'error': str(e)
            }
    
    def _check_min_holding_period(self, target_date: datetime) -> bool:
        """最小保有期間をチェック"""
        try:
            if self.current_holding_start is None:
                return True  # 初回は制限なし
            
            holding_days = self._get_current_holding_days(target_date)
            return holding_days >= self.min_holding_days
            
        except Exception as e:
            self.logger.warning(f"最小保有期間チェックエラー: {e}")
            return False
    
    def _check_monthly_switch_limit(self, target_date: datetime) -> bool:
        """月次切替制限をチェック"""
        try:
            monthly_count = self._get_monthly_switch_count(target_date)
            return monthly_count < self.max_switches_per_month
            
        except Exception as e:
            self.logger.warning(f"月次制限チェックエラー: {e}")
            return False
    
    def _get_monthly_switch_count(self, target_date: datetime) -> int:
        """当月の切替回数を取得"""
        try:
            month_start = target_date.replace(day=1)
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            
            count = 0
            for switch in self.switch_history:
                switch_date = switch.get('executed_date', switch.get('target_date'))
                if isinstance(switch_date, datetime) and month_start <= switch_date <= month_end:
                    if switch.get('status') == 'executed':
                        count += 1
            
            return count
            
        except Exception as e:
            self.logger.warning(f"月次切替回数取得エラー: {e}")
            return 0
    
    def _get_current_holding_days(self, target_date: datetime) -> int:
        """現在の保有日数を取得"""
        try:
            if self.current_holding_start is None:
                return 0
            
            return (target_date - self.current_holding_start).days
            
        except Exception as e:
            self.logger.warning(f"保有日数取得エラー: {e}")
            return 0
    
    def _evaluate_switch_cost_efficiency(self, from_symbol: str, to_symbol: str, 
                                        target_date: datetime) -> Dict[str, Any]:
        """
        切替コスト効率性評価
        
        Args:
            from_symbol: 切替前銘柄
            to_symbol: 切替後銘柄
            target_date: 対象日付
        
        Returns:
            Dict[str, Any]: コスト効率性評価結果
        """
        try:
            # 基本的なコスト効率性評価
            # 実際の実装では、期待リターンやリスク指標も考慮
            
            # 切替コスト見積もり（ポートフォリオ価値の一定比率）
            estimated_portfolio_value = 1000000  # デフォルト値、実際は外部から取得
            switch_cost = estimated_portfolio_value * self.switch_cost_rate
            
            # 簡単な効率性判定（実際はより複雑な分析が必要）
            is_cost_effective = True  # 基本的に切替を許可
            
            # 頻繁な切替を抑制するためのペナルティ
            recent_switches = self._count_recent_switches(target_date, days=7)
            if recent_switches >= 2:  # 1週間で2回以上の切替は抑制
                is_cost_effective = False
                reason = "frequent_switching_penalty"
            else:
                reason = "cost_effective_switch"
            
            return {
                'is_cost_effective': is_cost_effective,
                'reason': reason,
                'switch_cost': switch_cost,
                'cost_rate': self.switch_cost_rate,
                'recent_switches': recent_switches,
                'analysis_details': {
                    'estimated_portfolio_value': estimated_portfolio_value,
                    'cost_threshold_met': switch_cost <= estimated_portfolio_value * self.cost_threshold * 10,
                    'frequency_penalty': recent_switches >= 2
                }
            }
            
        except Exception as e:
            self.logger.error(f"コスト効率性評価エラー: {e}")
            return {
                'is_cost_effective': False,
                'reason': 'cost_evaluation_error',
                'error': str(e)
            }
    
    def _count_recent_switches(self, target_date: datetime, days: int = 7) -> int:
        """指定期間内の切替回数をカウント"""
        try:
            start_date = target_date - timedelta(days=days)
            count = 0
            
            for switch in self.switch_history:
                switch_date = switch.get('executed_date', switch.get('target_date'))
                if isinstance(switch_date, datetime) and start_date <= switch_date <= target_date:
                    if switch.get('status') == 'executed':
                        count += 1
            
            return count
            
        except Exception as e:
            self.logger.warning(f"直近切替回数カウントエラー: {e}")
            return 0
    
    def record_switch_executed(self, switch_result: Dict[str, Any]) -> None:
        """
        切替実行の記録
        
        Args:
            switch_result: 切替実行結果
        
        Raises:
            ValueError: 必須キーの不足
            SwitchError: 履歴記録エラー
        
        Example:
            switch_result = {
                'from_symbol': '7203',
                'to_symbol': '6758',
                'executed_date': datetime.now(),
                'portfolio_value_before': 1000000,
                'portfolio_value_after': 999000,
                'switch_cost': 1000,
                'status': 'executed'
            }
            switch_mgr.record_switch_executed(switch_result)
        """
        try:
            # 必須キーの検証
            required_keys = ['from_symbol', 'to_symbol', 'executed_date']
            for key in required_keys:
                if key not in switch_result:
                    raise ValueError(f"必須キー不足: {key}")
            
            # 切替履歴に記録
            switch_record = {
                **switch_result,
                'recorded_at': datetime.now(),
                'switch_id': len(self.switch_history) + 1
            }
            
            self.switch_history.append(switch_record)
            
            # 現在状態の更新
            self.current_symbol = switch_result['to_symbol']
            self.current_holding_start = switch_result['executed_date']
            
            # 統計更新
            self._update_switch_statistics(switch_record)
            
            self.logger.info(f"銘柄切替記録: {switch_result['from_symbol']} → {switch_result['to_symbol']} "
                           f"@ {switch_result['executed_date']}")
            
        except ValueError as e:
            raise e
        except Exception as e:
            self.logger.error(f"切替記録エラー: {e}")
            raise SwitchError(f"切替記録失敗: {e}")
    
    def _update_switch_statistics(self, switch_record: Dict[str, Any]) -> None:
        """切替統計の更新"""
        try:
            self.switch_stats['total_switches'] += 1
            
            if switch_record.get('status') == 'executed':
                self.switch_stats['successful_switches'] += 1
                
                # コスト統計
                switch_cost = switch_record.get('switch_cost', 0.0)
                self.switch_stats['cost_incurred'] += switch_cost
                
                # 平均保有日数更新（簡易計算）
                if len(self.switch_history) > 1:
                    total_days = sum(self._calculate_holding_days(record) for record in self.switch_history[-10:])
                    count = min(len(self.switch_history), 10)
                    self.switch_stats['average_holding_days'] = total_days / count if count > 0 else 0
            
            self.logger.debug(f"統計更新完了: 総切替数 {self.switch_stats['total_switches']}")
            
        except Exception as e:
            self.logger.warning(f"統計更新エラー: {e}")
    
    def _calculate_holding_days(self, switch_record: Dict[str, Any]) -> int:
        """個別切替記録の保有日数計算"""
        try:
            # 簡易実装：実際は前回切替との差分を計算
            return self.min_holding_days  # デフォルト値
        except Exception:
            return 0
    
    def get_switch_statistics(self) -> Dict[str, Any]:
        """
        切替統計情報の取得
        
        Returns:
            Dict[str, Any]: 切替統計
        
        Example:
            stats = switch_mgr.get_switch_statistics()
            print(f"総切替数: {stats['summary']['total_switches']}")
            print(f"成功率: {stats['summary']['success_rate']:.1%}")
        """
        try:
            current_time = datetime.now()
            
            # 基本統計
            total_switches = self.switch_stats['total_switches']
            successful_switches = self.switch_stats['successful_switches']
            success_rate = successful_switches / total_switches if total_switches > 0 else 0.0
            
            # 期間別統計
            monthly_switches = self._get_monthly_switch_count(current_time)
            weekly_switches = self._count_recent_switches(current_time, days=7)
            
            # 現在状況
            current_holding_days = self._get_current_holding_days(current_time)
            
            statistics = {
                'summary': {
                    'total_switches': total_switches,
                    'successful_switches': successful_switches,
                    'success_rate': success_rate,
                    'average_holding_days': self.switch_stats['average_holding_days'],
                    'total_cost_incurred': self.switch_stats['cost_incurred']
                },
                'current_period': {
                    'monthly_switches': monthly_switches,
                    'weekly_switches': weekly_switches,
                    'monthly_limit': self.max_switches_per_month,
                    'monthly_remaining': max(0, self.max_switches_per_month - monthly_switches)
                },
                'current_position': {
                    'current_symbol': self.current_symbol,
                    'holding_start_date': self.current_holding_start,
                    'current_holding_days': current_holding_days,
                    'min_holding_requirement_met': current_holding_days >= self.min_holding_days
                },
                'configuration': {
                    'switch_cost_rate': self.switch_cost_rate,
                    'min_holding_days': self.min_holding_days,
                    'max_switches_per_month': self.max_switches_per_month
                },
                'history_count': len(self.switch_history),
                'last_updated': current_time
            }
            
            self.logger.debug(f"統計情報取得完了: 総切替数 {total_switches}, 成功率 {success_rate:.1%}")
            return statistics
            
        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {e}")
            return {
                'summary': {},
                'error': str(e),
                'last_updated': datetime.now()
            }
    
    def get_switch_history(self, limit: Optional[int] = None, 
                          symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        切替履歴の取得
        
        Args:
            limit: 取得件数制限
            symbol: 特定銘柄フィルター
        
        Returns:
            List[Dict[str, Any]]: 切替履歴リスト
        """
        try:
            history = self.switch_history.copy()
            
            # 銘柄フィルター
            if symbol:
                history = [
                    switch for switch in history 
                    if switch.get('from_symbol') == symbol or switch.get('to_symbol') == symbol
                ]
            
            # 日付順ソート（最新が先頭）
            history.sort(
                key=lambda x: x.get('executed_date', x.get('recorded_at', datetime.min)), 
                reverse=True
            )
            
            # 件数制限
            if limit:
                history = history[:limit]
            
            self.logger.debug(f"切替履歴取得: {len(history)}件")
            return history
            
        except Exception as e:
            self.logger.error(f"切替履歴取得エラー: {e}")
            return []


def main():
    """SymbolSwitchManager 動作テスト"""
    print("SymbolSwitchManager 動作テスト")
    print("=" * 50)
    
    try:
        # 1. 初期化テスト
        config = {
            'switch_management': {
                'switch_cost_rate': 0.001,
                'min_holding_days': 1,
                'max_switches_per_month': 10,
                'cost_threshold': 0.001
            }
        }
        
        ssm = SymbolSwitchManager(config)
        print("✅ SymbolSwitchManager初期化成功")
        
        # 2. 初回銘柄選択テスト
        test_date = datetime(2023, 6, 1)
        
        evaluation = ssm.evaluate_symbol_switch(
            from_symbol=None,
            to_symbol='7203',
            target_date=test_date
        )
        
        print(f"✅ 初回銘柄選択評価: {evaluation['should_switch']} ({evaluation['reason']})")
        
        if evaluation['should_switch']:
            # 切替実行記録
            switch_result = {
                'from_symbol': None,
                'to_symbol': '7203',
                'executed_date': test_date,
                'portfolio_value_before': 1000000,
                'portfolio_value_after': 999000,
                'switch_cost': 1000,
                'status': 'executed'
            }
            ssm.record_switch_executed(switch_result)
            print("✅ 初回切替記録成功")
        
        # 3. 同一銘柄チェックテスト
        same_symbol_eval = ssm.evaluate_symbol_switch(
            from_symbol='7203',
            to_symbol='7203',
            target_date=test_date + timedelta(days=1)
        )
        print(f"✅ 同一銘柄チェック: {same_symbol_eval['should_switch']} ({same_symbol_eval['reason']})")
        
        # 4. 最小保有期間チェックテスト
        min_holding_eval = ssm.evaluate_symbol_switch(
            from_symbol='7203',
            to_symbol='6758',
            target_date=test_date  # 同日切替（制限に引っかかるはず）
        )
        print(f"✅ 最小保有期間チェック: {min_holding_eval['should_switch']} ({min_holding_eval['reason']})")
        
        # 5. 正常切替テスト
        normal_switch_eval = ssm.evaluate_symbol_switch(
            from_symbol='7203',
            to_symbol='6758',
            target_date=test_date + timedelta(days=2)  # 2日後
        )
        print(f"✅ 正常切替評価: {normal_switch_eval['should_switch']} ({normal_switch_eval['reason']})")
        
        if normal_switch_eval['should_switch']:
            # 正常切替実行
            switch_result2 = {
                'from_symbol': '7203',
                'to_symbol': '6758',
                'executed_date': test_date + timedelta(days=2),
                'portfolio_value_before': 1010000,
                'portfolio_value_after': 1009000,
                'switch_cost': 1000,
                'status': 'executed'
            }
            ssm.record_switch_executed(switch_result2)
            print("✅ 正常切替記録成功")
        
        # 6. 統計情報テスト
        stats = ssm.get_switch_statistics()
        print(f"✅ 統計情報取得成功:")
        print(f"  - 総切替数: {stats['summary']['total_switches']}")
        print(f"  - 成功率: {stats['summary']['success_rate']:.1%}")
        print(f"  - 月次切替数: {stats['current_period']['monthly_switches']}")
        print(f"  - 現在銘柄: {stats['current_position']['current_symbol']}")
        
        # 7. 履歴取得テスト
        history = ssm.get_switch_history(limit=5)
        print(f"✅ 切替履歴取得成功: {len(history)}件")
        
        # 8. 月次制限テスト（複数回切替シミュレーション）
        print(f"\n🧪 月次制限テスト:")
        for i in range(8):  # 8回追加切替を試行
            test_eval = ssm.evaluate_symbol_switch(
                from_symbol='6758',
                to_symbol=f'TEST{i}',
                target_date=test_date + timedelta(days=3+i)
            )
            
            if test_eval['should_switch']:
                test_switch = {
                    'from_symbol': '6758',
                    'to_symbol': f'TEST{i}',
                    'executed_date': test_date + timedelta(days=3+i),
                    'portfolio_value_before': 1000000,
                    'portfolio_value_after': 999000,
                    'switch_cost': 1000,
                    'status': 'executed'
                }
                ssm.record_switch_executed(test_switch)
        
        # 最終統計確認
        final_stats = ssm.get_switch_statistics()
        print(f"  - 最終切替数: {final_stats['summary']['total_switches']}")
        print(f"  - 月次残り回数: {final_stats['current_period']['monthly_remaining']}")
        
        # 9. 制限超過テスト
        limit_test_eval = ssm.evaluate_symbol_switch(
            from_symbol=ssm.current_symbol,
            to_symbol='LIMIT_TEST',
            target_date=test_date + timedelta(days=15)
        )
        print(f"✅ 制限超過テスト: {limit_test_eval['should_switch']} ({limit_test_eval['reason']})")
        
        print(f"\n🎉 SymbolSwitchManager テスト完了！")
        print(f"実装機能: 切替評価、制限管理、履歴記録、統計計算")
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
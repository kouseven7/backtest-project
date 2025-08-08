"""
データ正規化・整合システム
フェーズ4A3: バックテストvs実運用比較分析器
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

class DataAligner:
    """データ正規化・整合システム"""
    
    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.filtering_criteria = config.get('analysis_settings', {}).get('filtering_criteria', {})
    
    async def align_datasets(self, backtest_data: Dict, live_data: Dict) -> Dict:
        """データセット整合処理"""
        try:
            self.logger.info("データセット整合処理開始...")
            
            # 1. 共通戦略抽出
            common_strategies = self._find_common_strategies(backtest_data, live_data)
            self.logger.info(f"共通戦略: {len(common_strategies)}件 - {list(common_strategies)}")
            
            if not common_strategies:
                self.logger.warning("共通戦略が見つかりません")
                return {}
            
            # 2. データフィルタリング
            filtered_bt_data = self._filter_strategy_data(backtest_data, common_strategies)
            filtered_live_data = self._filter_strategy_data(live_data, common_strategies)
            
            # 3. メトリクス正規化
            normalized_bt_data = self._normalize_metrics(filtered_bt_data)
            normalized_live_data = self._normalize_metrics(filtered_live_data)
            
            # 4. データ品質チェック
            alignment_quality = self._assess_alignment_quality(normalized_bt_data, normalized_live_data)
            
            aligned_data = {
                "backtest": normalized_bt_data,
                "live": normalized_live_data,
                "common_strategies": list(common_strategies),
                "alignment_quality": alignment_quality,
                "alignment_timestamp": datetime.now()
            }
            
            self.logger.info(f"データ整合完了 - 品質: {alignment_quality}")
            return aligned_data
            
        except Exception as e:
            self.logger.error(f"データ整合エラー: {e}")
            return {}
    
    def _find_common_strategies(self, backtest_data: Dict, live_data: Dict) -> set:
        """共通戦略抽出"""
        try:
            bt_strategies = set(backtest_data.get('strategies', {}).keys())
            live_strategies = set(live_data.get('strategies', {}).keys())
            
            # 直接マッチング
            direct_common = bt_strategies.intersection(live_strategies)
            
            # 類似名マッチング（部分マッチ）
            fuzzy_common = set()
            for bt_strategy in bt_strategies:
                for live_strategy in live_strategies:
                    if self._is_similar_strategy_name(bt_strategy, live_strategy):
                        fuzzy_common.add(bt_strategy)
                        break
            
            common_strategies = direct_common.union(fuzzy_common)
            
            # 最小取引数フィルタリング
            min_trades = self.filtering_criteria.get('min_trades', 10)
            filtered_common = set()
            
            for strategy in common_strategies:
                bt_trades = backtest_data.get('strategies', {}).get(strategy, {}).get('basic_metrics', {}).get('total_trades', 0)
                live_trades = live_data.get('strategies', {}).get(strategy, {}).get('basic_metrics', {}).get('total_trades', 0)
                
                if bt_trades >= min_trades or live_trades >= min_trades:
                    filtered_common.add(strategy)
            
            return filtered_common
            
        except Exception as e:
            self.logger.warning(f"共通戦略抽出エラー: {e}")
            return set()
    
    def _is_similar_strategy_name(self, name1: str, name2: str) -> bool:
        """戦略名類似性判定"""
        try:
            # 正規化
            norm1 = name1.lower().replace('_', '').replace('-', '').replace(' ', '')
            norm2 = name2.lower().replace('_', '').replace('-', '').replace(' ', '')
            
            # 完全一致
            if norm1 == norm2:
                return True
            
            # 部分一致（長い方の70%以上）
            longer = max(norm1, norm2, key=len)
            shorter = min(norm1, norm2, key=len)
            
            if len(shorter) >= len(longer) * 0.7:
                if shorter in longer or longer in shorter:
                    return True
            
            # キーワードマッチング
            keywords1 = set(norm1.split())
            keywords2 = set(norm2.split())
            
            if keywords1.intersection(keywords2):
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"戦略名類似性判定エラー: {e}")
            return False
    
    def _filter_strategy_data(self, data: Dict, target_strategies: set) -> Dict:
        """戦略データフィルタリング"""
        try:
            filtered_strategies = {}
            
            for strategy_name in target_strategies:
                if strategy_name in data.get('strategies', {}):
                    strategy_data = data['strategies'][strategy_name]
                    
                    # 最小取引数チェック
                    total_trades = strategy_data.get('basic_metrics', {}).get('total_trades', 0)
                    min_trades = self.filtering_criteria.get('min_trades', 10)
                    
                    if total_trades >= min_trades:
                        filtered_strategies[strategy_name] = strategy_data
                    else:
                        self.logger.info(f"戦略 {strategy_name} を取引数不足でフィルタ ({total_trades} < {min_trades})")
            
            filtered_data = data.copy()
            filtered_data['strategies'] = filtered_strategies
            
            return filtered_data
            
        except Exception as e:
            self.logger.warning(f"戦略データフィルタリングエラー: {e}")
            return data
    
    def _normalize_metrics(self, data: Dict) -> Dict:
        """メトリクス正規化"""
        try:
            normalized_data = data.copy()
            
            for strategy_name, strategy_data in normalized_data.get('strategies', {}).items():
                # 基本メトリクス正規化
                basic_metrics = strategy_data.get('basic_metrics', {})
                normalized_basic = {}
                
                for metric, value in basic_metrics.items():
                    if isinstance(value, (int, float)) and pd.notna(value):
                        normalized_basic[metric] = float(value)
                    else:
                        normalized_basic[metric] = 0.0
                
                # win_rate計算（存在しない場合）
                if 'win_rate' not in normalized_basic or normalized_basic['win_rate'] == 0:
                    total_trades = normalized_basic.get('total_trades', 0)
                    winning_trades = normalized_basic.get('winning_trades', 0)
                    if total_trades > 0:
                        normalized_basic['win_rate'] = winning_trades / total_trades
                
                # リスクメトリクス正規化
                risk_metrics = strategy_data.get('risk_metrics', {})
                normalized_risk = {}
                
                for metric, value in risk_metrics.items():
                    if isinstance(value, (int, float)) and pd.notna(value):
                        normalized_risk[metric] = float(value)
                    else:
                        normalized_risk[metric] = 0.0
                
                # 正規化されたデータを更新
                strategy_data['basic_metrics'] = normalized_basic
                strategy_data['risk_metrics'] = normalized_risk
                
                # 派生メトリクス計算
                strategy_data['derived_metrics'] = self._calculate_derived_metrics(normalized_basic, normalized_risk)
            
            return normalized_data
            
        except Exception as e:
            self.logger.warning(f"メトリクス正規化エラー: {e}")
            return data
    
    def _calculate_derived_metrics(self, basic_metrics: Dict, risk_metrics: Dict) -> Dict:
        """派生メトリクス計算"""
        try:
            derived = {}
            
            # Profit Factor
            winning_trades = basic_metrics.get('winning_trades', 0)
            total_trades = basic_metrics.get('total_trades', 0)
            losing_trades = total_trades - winning_trades
            total_pnl = basic_metrics.get('total_pnl', 0)
            
            if total_trades > 0 and losing_trades > 0:
                avg_win = total_pnl / winning_trades if winning_trades > 0 else 0
                avg_loss = abs(total_pnl) / losing_trades if losing_trades > 0 else 0
                
                derived['avg_win'] = avg_win
                derived['avg_loss'] = avg_loss
                derived['profit_factor'] = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Risk-adjusted return
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            
            if max_drawdown != 0:
                derived['return_to_drawdown'] = total_pnl / abs(max_drawdown)
            
            # Trade frequency (per day)
            derived['trade_frequency'] = total_trades / 30  # 30日あたり
            
            return derived
            
        except Exception as e:
            self.logger.warning(f"派生メトリクス計算エラー: {e}")
            return {}
    
    def _assess_alignment_quality(self, bt_data: Dict, live_data: Dict) -> str:
        """整合品質評価"""
        try:
            quality_scores = []
            
            # 戦略数の一致度
            bt_strategies = len(bt_data.get('strategies', {}))
            live_strategies = len(live_data.get('strategies', {}))
            
            if bt_strategies > 0 and live_strategies > 0:
                strategy_score = min(bt_strategies, live_strategies) / max(bt_strategies, live_strategies)
                quality_scores.append(strategy_score)
            
            # データ完全性スコア
            for strategy_name in bt_data.get('strategies', {}):
                if strategy_name in live_data.get('strategies', {}):
                    bt_basic = bt_data['strategies'][strategy_name].get('basic_metrics', {})
                    live_basic = live_data['strategies'][strategy_name].get('basic_metrics', {})
                    
                    # 必須メトリクスの存在確認
                    required_metrics = ['total_trades', 'total_pnl', 'win_rate']
                    bt_completeness = sum(1 for metric in required_metrics if metric in bt_basic and bt_basic[metric] != 0) / len(required_metrics)
                    live_completeness = sum(1 for metric in required_metrics if metric in live_basic and live_basic[metric] != 0) / len(required_metrics)
                    
                    avg_completeness = (bt_completeness + live_completeness) / 2
                    quality_scores.append(avg_completeness)
            
            overall_score = np.mean(quality_scores) if quality_scores else 0
            
            if overall_score >= 0.8:
                return "high"
            elif overall_score >= 0.6:
                return "medium"
            elif overall_score >= 0.4:
                return "low"
            else:
                return "poor"
                
        except Exception as e:
            self.logger.warning(f"整合品質評価エラー: {e}")
            return "unknown"

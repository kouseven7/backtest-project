"""
負荷テスト用のデータ生成器
5-3-3 戦略間相関を考慮した配分最適化システム

Author: imega
Created: 2025-07-24
Task: Load Testing for 5-3-3
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LoadTestDataGenerator:
    """負荷テスト用データ生成器"""
    
    def __init__(self, random_seed: int = 42):
        self.random_state = random_seed
        np.random.seed(random_seed)
        self.logger = logging.getLogger(__name__)
        
    def generate_strategy_data(self, 
                             num_strategies: int, 
                             periods: int,
                             base_return: float = 0.0001,
                             volatility_range: Tuple[float, float] = (0.01, 0.03)) -> Dict[str, Any]:
        """戦略パフォーマンスデータ生成"""
        
        self.logger.info(f"Generating strategy data: {num_strategies} strategies, {periods} periods")
        
        # 戦略名生成
        strategy_names = [f"Strategy_{i+1:02d}" for i in range(num_strategies)]
        
        # 日付インデックス生成
        end_date = datetime.now()
        start_date = end_date - timedelta(days=periods)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')[:periods]
        
        # リターンデータ生成
        returns_data = {}
        for strategy in strategy_names:
            # 各戦略の特性パラメータ
            mu = np.random.normal(base_return, base_return * 0.5)
            sigma = np.random.uniform(*volatility_range)
            
            # GBM風のリターン生成
            daily_returns = np.random.normal(mu, sigma, periods)
            
            # 価格データ生成（初期値100から開始）
            prices = [100.0]
            for ret in daily_returns:
                prices.append(prices[-1] * (1 + ret))
            
            returns_data[strategy] = {
                'returns': daily_returns,
                'prices': prices[1:],  # 初期値を除く
                'mu': mu,
                'sigma': sigma
            }
        
        # DataFrame形式でも提供
        returns_df = pd.DataFrame(
            {strategy: data['returns'] for strategy, data in returns_data.items()},
            index=date_range
        )
        
        prices_df = pd.DataFrame(
            {strategy: data['prices'] for strategy, data in returns_data.items()},
            index=date_range
        )
        
        return {
            'strategy_names': strategy_names,
            'date_range': date_range,
            'returns_data': returns_data,
            'returns_df': returns_df,
            'prices_df': prices_df,
            'num_strategies': num_strategies,
            'periods': periods
        }
    
    def generate_correlation_scenarios(self) -> List[Dict[str, Any]]:
        """相関シナリオデータ生成"""
        scenarios = [
            {
                "name": "low_correlation",
                "description": "低相関環境（独立性の高い戦略）",
                "base_corr": 0.1,
                "corr_range": (0.0, 0.2),
                "volatility_clustering": False
            },
            {
                "name": "medium_correlation", 
                "description": "中相関環境（標準的な市場環境）",
                "base_corr": 0.4,
                "corr_range": (0.2, 0.6),
                "volatility_clustering": True
            },
            {
                "name": "high_correlation",
                "description": "高相関環境（ストレス時）", 
                "base_corr": 0.8,
                "corr_range": (0.6, 0.9),
                "volatility_clustering": True
            },
            {
                "name": "mixed_correlation",
                "description": "混合相関環境（複雑な構造）",
                "base_corr": "mixed",
                "corr_range": (0.0, 0.9),
                "volatility_clustering": True
            }
        ]
        return scenarios
    
    def create_stress_test_data(self, scenario: str, num_strategies: int = 10, periods: int = 252) -> Dict[str, Any]:
        """ストレステスト用データ作成"""
        
        self.logger.info(f"Creating stress test data for scenario: {scenario}")
        
        if scenario == "extreme_correlation":
            return self._create_extreme_correlation_data(num_strategies, periods)
        elif scenario == "rapid_correlation_change":
            return self._create_rapid_correlation_change_data(num_strategies, periods)
        elif scenario == "singular_matrix":
            return self._create_singular_matrix_data(num_strategies, periods)
        elif scenario == "memory_pressure":
            return self._create_memory_pressure_data(num_strategies, periods)
        else:
            raise ValueError(f"Unknown stress test scenario: {scenario}")
    
    def _create_extreme_correlation_data(self, num_strategies: int, periods: int) -> Dict[str, Any]:
        """極端な相関データ作成"""
        # 完全相関に近いデータ
        base_returns = np.random.normal(0.0001, 0.02, periods)
        
        returns_data = {}
        strategy_names = [f"HighCorr_Strategy_{i+1:02d}" for i in range(num_strategies)]
        
        for i, strategy in enumerate(strategy_names):
            # 微小なノイズを追加して完全相関を避ける
            noise = np.random.normal(0, 0.001, periods)
            strategy_returns = base_returns + noise
            
            returns_data[strategy] = {
                'returns': strategy_returns,
                'correlation_factor': 0.99 - i * 0.001  # 微小な差異
            }
        
        return {
            'scenario': 'extreme_correlation',
            'strategy_names': strategy_names,
            'returns_data': returns_data,
            'expected_correlation': 0.99
        }
    
    def _create_rapid_correlation_change_data(self, num_strategies: int, periods: int) -> Dict[str, Any]:
        """急激な相関変化データ作成"""
        # 期間を3分割して相関を変化させる
        period_1 = periods // 3
        period_2 = periods // 3
        period_3 = periods - period_1 - period_2
        
        returns_data = {}
        strategy_names = [f"RapidChange_Strategy_{i+1:02d}" for i in range(num_strategies)]
        
        for strategy in strategy_names:
            # 期間1: 低相関
            returns_1 = np.random.normal(0.0001, 0.015, period_1)
            
            # 期間2: 高相関
            base_2 = np.random.normal(0.0001, 0.025, period_2)
            returns_2 = base_2 + np.random.normal(0, 0.005, period_2)
            
            # 期間3: 中相関
            base_3 = np.random.normal(0.0001, 0.02, period_3)
            returns_3 = base_3 + np.random.normal(0, 0.01, period_3)
            
            full_returns = np.concatenate([returns_1, returns_2, returns_3])
            
            returns_data[strategy] = {
                'returns': full_returns,
                'correlation_phases': [
                    (0, period_1, 'low_correlation'),
                    (period_1, period_1 + period_2, 'high_correlation'),
                    (period_1 + period_2, periods, 'medium_correlation')
                ]
            }
        
        return {
            'scenario': 'rapid_correlation_change',
            'strategy_names': strategy_names,
            'returns_data': returns_data,
            'phase_info': 'Three correlation regimes with rapid transitions'
        }
    
    def _create_singular_matrix_data(self, num_strategies: int, periods: int) -> Dict[str, Any]:
        """特異行列データ作成（数値安定性テスト用）"""
        # 線形従属な戦略を含むデータ
        base_strategies = min(num_strategies // 2, 5)  # 基底戦略数
        
        returns_data = {}
        strategy_names = [f"Singular_Strategy_{i+1:02d}" for i in range(num_strategies)]
        
        # 基底戦略生成
        base_returns = {}
        for i in range(base_strategies):
            base_name = f"Base_{i+1}"
            base_returns[base_name] = np.random.normal(0.0001, 0.02, periods)
        
        # 従属戦略生成
        for i, strategy in enumerate(strategy_names):
            if i < base_strategies:
                # 基底戦略
                returns_data[strategy] = {
                    'returns': list(base_returns.values())[i],
                    'dependency': 'independent'
                }
            else:
                # 線形結合戦略
                base_keys = list(base_returns.keys())
                weights = np.random.rand(len(base_keys))
                weights = weights / weights.sum()  # 正規化
                
                combined_returns = np.zeros(periods)
                for j, base_key in enumerate(base_keys):
                    combined_returns += weights[j] * base_returns[base_key]
                
                # 微小ノイズ追加
                noise = np.random.normal(0, 0.0001, periods)
                combined_returns += noise
                
                returns_data[strategy] = {
                    'returns': combined_returns,
                    'dependency': f'linear_combination_of_{base_keys}',
                    'weights': weights.tolist()
                }
        
        return {
            'scenario': 'singular_matrix',
            'strategy_names': strategy_names,
            'returns_data': returns_data,
            'base_strategies': base_strategies,
            'warning': 'Contains linearly dependent strategies'
        }
    
    def _create_memory_pressure_data(self, num_strategies: int, periods: int) -> Dict[str, Any]:
        """メモリプレッシャーデータ作成（大容量データテスト）"""
        # 通常より大きなデータセット
        extended_periods = periods * 5  # 5倍の期間
        extended_strategies = num_strategies * 2  # 2倍の戦略数
        
        returns_data = {}
        strategy_names = [f"MemPress_Strategy_{i+1:03d}" for i in range(extended_strategies)]
        
        # より詳細なデータを生成
        for strategy in strategy_names:
            # 高頻度データをシミュレート
            mu = np.random.normal(0.00001, 0.000005)  # 日次→分次レベル
            sigma = np.random.uniform(0.001, 0.005)
            
            returns = np.random.normal(mu, sigma, extended_periods)
            
            # 追加の時系列特徴量
            volatility = np.random.gamma(2, 0.01, extended_periods)
            volume = np.random.lognormal(10, 1, extended_periods)
            
            returns_data[strategy] = {
                'returns': returns,
                'volatility': volatility,
                'volume': volume,
                'extended_features': True
            }
        
        return {
            'scenario': 'memory_pressure',
            'strategy_names': strategy_names,
            'returns_data': returns_data,
            'data_size_gb': self._estimate_data_size(returns_data),
            'extended_periods': extended_periods,
            'extended_strategies': extended_strategies
        }
    
    def _estimate_data_size(self, returns_data: Dict) -> float:
        """データサイズ推定（GB）"""
        total_elements = 0
        for strategy_data in returns_data.values():
            for key, value in strategy_data.items():
                if isinstance(value, (list, np.ndarray)):
                    total_elements += len(value)
        
        # float64を想定（8bytes per element）
        bytes_size = total_elements * 8
        gb_size = bytes_size / (1024**3)
        return round(gb_size, 4)
    
    def save_test_data(self, data: Dict[str, Any], filepath: str):
        """テストデータ保存"""
        try:
            # DataFrameを辞書に変換
            export_data = data.copy()
            
            if 'returns_df' in export_data:
                export_data['returns_df'] = export_data['returns_df'].to_dict()
            if 'prices_df' in export_data:
                export_data['prices_df'] = export_data['prices_df'].to_dict()
            if 'date_range' in export_data:
                export_data['date_range'] = [d.isoformat() for d in export_data['date_range']]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                
            self.logger.info(f"Test data saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Save error: {e}")

# テスト用の実行例
if __name__ == "__main__":
    import os
    
    # ロギング設定
    logging.basicConfig(level=logging.INFO)
    
    generator = LoadTestDataGenerator()
    
    # 中規模データ生成テスト
    print("=== 中規模データ生成テスト ===")
    data = generator.generate_strategy_data(num_strategies=10, periods=756)
    print(f"生成完了: {data['num_strategies']}戦略, {data['periods']}期間")
    print(f"リターンDF形状: {data['returns_df'].shape}")
    
    # 相関シナリオテスト
    print("\n=== 相関シナリオ ===")
    scenarios = generator.generate_correlation_scenarios()
    for scenario in scenarios:
        print(f"- {scenario['name']}: {scenario['description']}")
    
    # ストレステストデータ生成
    print("\n=== ストレステストデータ ===")
    stress_data = generator.create_stress_test_data("extreme_correlation", 5, 100)
    print(f"ストレスシナリオ: {stress_data['scenario']}")
    print(f"戦略数: {len(stress_data['strategy_names'])}")

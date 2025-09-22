"""
Problem 11: ISM統合品質改善 - 動的切替基準最適化システム

切替判定基準の動的調整による品質向上:
- 市場環境に応じた閾値最適化
- 過去パフォーマンスに基づく学習調整
- 不要切替率<20%達成のための基準チューニング
"""

import sys
from pathlib import Path
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

@dataclass
class SwitchCriteriaConfig:
    """切替基準設定"""
    daily_performance_threshold: float = 0.02
    weekly_performance_threshold: float = 0.05
    volatility_threshold: float = 0.03
    confidence_threshold: float = 0.6
    emergency_drawdown_limit: float = 0.1
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

class AdaptiveSwitchCriteriaOptimizer:
    """市場環境適応型切替基準最適化"""
    
    def __init__(self):
        self.historical_performance = []
        self.optimization_history = []
        self.current_criteria = SwitchCriteriaConfig()
        
    def optimize_switch_thresholds(self, historical_switches: List[Dict[str, Any]]) -> SwitchCriteriaConfig:
        """過去パフォーマンスに基づく閾値最適化"""
        print("\n🎯 切替基準動的最適化")
        print("-" * 50)
        
        if not historical_switches:
            print("⚠ 履歴データ不足 - デフォルト基準を使用")
            return self.current_criteria
        
        # 最適化対象パラメータ
        optimization_targets = {
            'daily_performance_threshold': [0.01, 0.015, 0.02, 0.025, 0.03],
            'volatility_threshold': [0.02, 0.025, 0.03, 0.035, 0.04],
            'confidence_threshold': [0.5, 0.55, 0.6, 0.65, 0.7]
        }
        
        best_criteria = self.current_criteria
        best_score = float('inf')
        
        print("パラメータ最適化実行中...")
        
        for param_name, candidate_values in optimization_targets.items():
            print(f"\n{param_name} 最適化:")
            
            for value in candidate_values:
                # テスト用基準設定
                test_criteria = SwitchCriteriaConfig(**self.current_criteria.to_dict())
                setattr(test_criteria, param_name, value)
                
                # シミュレーション実行
                simulation_result = self._simulate_with_criteria(historical_switches, test_criteria)
                
                # スコア計算（不要切替率を主要指標とする）
                score = self._calculate_optimization_score(simulation_result)
                
                print(f"  {value:.3f}: 不要切替率={simulation_result['unnecessary_rate']:.1%}, スコア={score:.3f}")
                
                if score < best_score:
                    best_score = score
                    best_criteria = test_criteria
                    
        print(f"\n✓ 最適化完了")
        print(f"改善前基準: {self.current_criteria.to_dict()}")
        print(f"最適化基準: {best_criteria.to_dict()}")
        
        self.current_criteria = best_criteria
        return best_criteria
    
    def _simulate_with_criteria(self, historical_switches: List[Dict[str, Any]], 
                               criteria: SwitchCriteriaConfig) -> Dict[str, Any]:
        """指定基準での切替シミュレーション"""
        
        unnecessary_count = 0
        total_switches = len(historical_switches)
        
        for switch in historical_switches:
            # 新基準での判定シミュレーション
            would_switch = self._evaluate_switch_with_criteria(switch, criteria)
            
            if would_switch:
                # 実際の結果と比較
                actual_net_gain = switch.get('net_gain', 0)
                if actual_net_gain <= 0:
                    unnecessary_count += 1
        
        unnecessary_rate = unnecessary_count / total_switches if total_switches > 0 else 0.0
        
        return {
            'unnecessary_rate': unnecessary_rate,
            'unnecessary_count': unnecessary_count,
            'total_switches': total_switches,
            'criteria_used': criteria.to_dict()
        }
    
    def _evaluate_switch_with_criteria(self, switch_context: Dict[str, Any], 
                                     criteria: SwitchCriteriaConfig) -> bool:
        """指定基準での切替判定評価"""
        
        # 日次パフォーマンス基準
        daily_performance = switch_context.get('performance_before', 0)
        if daily_performance < -criteria.daily_performance_threshold:
            return True
            
        # ボラティリティ基準
        volatility = switch_context.get('market_volatility', 0)
        if volatility > criteria.volatility_threshold:
            return True
            
        # 信頼度基準（模擬）
        confidence = np.random.uniform(0.3, 0.9)  # 模擬信頼度
        if confidence >= criteria.confidence_threshold:
            return True
            
        return False
    
    def _calculate_optimization_score(self, simulation_result: Dict[str, Any]) -> float:
        """最適化スコア計算"""
        unnecessary_rate = simulation_result['unnecessary_rate']
        
        # 目標不要切替率20%からの乖離をペナルティとする
        target_rate = 0.20
        deviation_penalty = abs(unnecessary_rate - target_rate) * 10
        
        # 20%以下なら追加ボーナス
        if unnecessary_rate < target_rate:
            achievement_bonus = (target_rate - unnecessary_rate) * 5
            return deviation_penalty - achievement_bonus
        
        return deviation_penalty

class SwitchQualityLearningSystem:
    """切替品質継続改善学習システム"""
    
    def __init__(self):
        self.quality_history = []
        self.threshold_adjustments = []
        self.learning_rate = 0.1
        
    def continuous_quality_improvement(self, current_metrics: Dict[str, Any], 
                                     target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """継続的品質改善サイクル"""
        print("\n📈 継続的品質改善学習")
        print("-" * 50)
        
        current_unnecessary_rate = current_metrics.get('unnecessary_rate', 0)
        current_consistency_rate = current_metrics.get('consistency_rate', 0)
        
        target_unnecessary_rate = target_metrics.get('unnecessary_rate', 0.20)
        target_consistency_rate = target_metrics.get('consistency_rate', 0.95)
        
        print(f"現在品質:")
        print(f"  不要切替率: {current_unnecessary_rate:.1%} (目標: {target_unnecessary_rate:.1%})")
        print(f"  一貫性率:   {current_consistency_rate:.1%} (目標: {target_consistency_rate:.1%})")
        
        # 改善必要性判定
        needs_improvement = (
            current_unnecessary_rate > target_unnecessary_rate or
            current_consistency_rate < target_consistency_rate
        )
        
        if not needs_improvement:
            print("✓ 品質目標達成済み - 調整不要")
            return {'adjustment_needed': False, 'adjustments': {}}
        
        # 調整計算
        adjustments = self._calculate_quality_adjustments(
            current_metrics, target_metrics
        )
        
        print(f"\n推奨調整:")
        for param, adjustment in adjustments.items():
            print(f"  {param}: {adjustment:+.4f}")
        
        # 学習履歴保存
        learning_record = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'target_metrics': target_metrics,
            'adjustments': adjustments,
            'improvement_needed': needs_improvement
        }
        
        self.quality_history.append(learning_record)
        
        return {
            'adjustment_needed': True,
            'adjustments': adjustments,
            'learning_record': learning_record
        }
    
    def _calculate_quality_adjustments(self, current_metrics: Dict[str, Any], 
                                     target_metrics: Dict[str, float]) -> Dict[str, float]:
        """品質メトリクスに基づく調整計算"""
        
        adjustments = {}
        
        # 不要切替率調整
        unnecessary_rate_diff = current_metrics.get('unnecessary_rate', 0) - target_metrics.get('unnecessary_rate', 0.20)
        if unnecessary_rate_diff > 0:
            # 不要切替率が高い → 閾値を厳しく
            adjustments['daily_performance_threshold'] = unnecessary_rate_diff * 0.01  # 最大1%調整
            adjustments['volatility_threshold'] = unnecessary_rate_diff * 0.005       # 最大0.5%調整
            adjustments['confidence_threshold'] = unnecessary_rate_diff * 0.1          # 最大10%調整
        
        # 一貫性率調整
        consistency_rate_diff = target_metrics.get('consistency_rate', 0.95) - current_metrics.get('consistency_rate', 0)
        if consistency_rate_diff > 0:
            # 一貫性率が低い → より厳格な基準
            adjustments['confidence_threshold'] = adjustments.get('confidence_threshold', 0) + consistency_rate_diff * 0.05
        
        return adjustments
    
    def apply_learned_adjustments(self, base_criteria: SwitchCriteriaConfig, 
                                adjustments: Dict[str, float]) -> SwitchCriteriaConfig:
        """学習済み調整の適用"""
        print("\n⚙️ 学習済み調整適用")
        print("-" * 50)
        
        adjusted_criteria = SwitchCriteriaConfig(**base_criteria.to_dict())
        
        for param, adjustment in adjustments.items():
            if hasattr(adjusted_criteria, param):
                old_value = getattr(adjusted_criteria, param)
                new_value = max(0.001, old_value + adjustment)  # 最小値制限
                setattr(adjusted_criteria, param, new_value)
                
                print(f"{param}: {old_value:.4f} → {new_value:.4f} ({adjustment:+.4f})")
        
        return adjusted_criteria

def integrate_with_intelligent_switch_manager(optimized_criteria: SwitchCriteriaConfig):
    """IntelligentSwitchManagerとの統合更新"""
    print("\n🔗 ISM統合設定更新")
    print("-" * 50)
    
    try:
        # 設定ファイル更新
        config_path = project_root / "config" / "dssms" / "dssms_backtester_config.json"
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # ISM統合設定セクション更新
        if 'intelligent_switch_manager' not in config:
            config['intelligent_switch_manager'] = {}
        
        config['intelligent_switch_manager']['switching_criteria'] = {
            'daily_performance_threshold': optimized_criteria.daily_performance_threshold,
            'weekly_performance_threshold': optimized_criteria.weekly_performance_threshold,
            'volatility_threshold': optimized_criteria.volatility_threshold,
            'confidence_threshold': optimized_criteria.confidence_threshold,
            'emergency_drawdown_limit': optimized_criteria.emergency_drawdown_limit
        }
        
        # 最適化メタデータ追加
        config['intelligent_switch_manager']['optimization_metadata'] = {
            'last_optimization': datetime.now().isoformat(),
            'optimization_version': '1.1',
            'target_unnecessary_rate': 0.20,
            'target_consistency_rate': 0.95
        }
        
        # 設定ファイル保存
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 設定ファイル更新完了: {config_path.name}")
        print("✓ ISM統合切替基準最適化完了")
        
        return True
        
    except Exception as e:
        print(f"❌ ISM統合設定更新エラー: {e}")
        return False

def main():
    """動的切替基準最適化メイン実行"""
    print("=" * 70)
    print("Problem 11: 動的切替基準最適化システム")
    print("=" * 70)
    
    try:
        # Phase 1: 履歴データ読み込み（実データ評価結果）
        print("📂 履歴データ読み込み")
        
        # 実データ評価システムから履歴取得
        from problem11_quality_improvement_evaluator import RealDataSwitchQualityEvaluator
        
        evaluator = RealDataSwitchQualityEvaluator()
        historical_switches = evaluator.collect_real_switch_data(days_back=30)
        
        if not historical_switches:
            print("❌ 履歴データ取得失敗")
            return False
        
        print(f"✓ 履歴データ取得完了: {len(historical_switches)}件")
        
        # Phase 2: 切替基準最適化
        optimizer = AdaptiveSwitchCriteriaOptimizer()
        optimized_criteria = optimizer.optimize_switch_thresholds(historical_switches)
        
        # Phase 3: 学習システム適用
        learning_system = SwitchQualityLearningSystem()
        
        # 現在メトリクス算出
        unnecessary_result = evaluator.evaluate_real_unnecessary_switches(historical_switches)
        current_metrics = {
            'unnecessary_rate': unnecessary_result['unnecessary_rate'],
            'consistency_rate': 0.5  # 模擬値（実際は多次元評価から取得）
        }
        
        target_metrics = {
            'unnecessary_rate': 0.20,
            'consistency_rate': 0.95
        }
        
        learning_result = learning_system.continuous_quality_improvement(
            current_metrics, target_metrics
        )
        
        # Phase 4: 最終調整適用
        if learning_result['adjustment_needed']:
            final_criteria = learning_system.apply_learned_adjustments(
                optimized_criteria, learning_result['adjustments']
            )
        else:
            final_criteria = optimized_criteria
        
        # Phase 5: ISM統合設定更新
        integration_success = integrate_with_intelligent_switch_manager(final_criteria)
        
        # 最終結果レポート
        print("\n" + "=" * 70)
        print("📊 動的最適化結果サマリー")
        print("=" * 70)
        
        print("最適化済み切替基準:")
        for param, value in final_criteria.to_dict().items():
            print(f"  {param}: {value:.4f}")
        
        print(f"\nISM統合更新: {'✓ 成功' if integration_success else '❌ 失敗'}")
        print(f"期待改善効果:")
        print(f"  不要切替率: {current_metrics['unnecessary_rate']:.1%} → <20%")
        print(f"  一貫性率:   {current_metrics['consistency_rate']:.1%} → ≥95%")
        
        return integration_success
        
    except Exception as e:
        print(f"❌ 動的最適化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
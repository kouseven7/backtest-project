"""
Module: Adaptive Learning Engine
File: adaptive_learning.py
Description: 
  5-2-2「トレンド判定精度の自動補正」
  適応学習エンジン - 継続的学習による補正パラメータの最適化

Author: imega
Created: 2025-07-22
Modified: 2025-07-22
"""

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

class AdaptiveLearningEngine:
    """適応学習システム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # 学習設定
        self.algorithm = config.get('algorithm', 'adam')
        self.learning_rate = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.beta1 = config.get('beta1', 0.9)
        self.beta2 = config.get('beta2', 0.999)
        self.epsilon = config.get('epsilon', 1e-8)
        self.enable_continuous_learning = config.get('enable_continuous_learning', True)
        
        # 学習アルゴリズムの選択
        self.learning_algorithms = {
            'gradient_descent': self._gradient_descent_update,
            'momentum': self._momentum_update,
            'adam': self._adam_update
        }
        
        # 学習状態
        self._velocity = {}
        self._m = {}  # Adam: first moment estimates
        self._v = {}  # Adam: second moment estimates
        self._t = 0   # Adam: time step
        
        self.logger.info(f"AdaptiveLearningEngine initialized with {self.algorithm} algorithm")
    
    def continuous_learning_update(self,
                                 correction_engine: Any,
                                 new_feedback: List[Any]) -> Dict[str, Any]:
        """継続的学習による更新"""
        
        try:
            if not self.enable_continuous_learning or not new_feedback:
                return {'message': 'No updates applied'}
            
            # 1. フィードバックデータの分析
            feedback_analysis = self._analyze_feedback(new_feedback)
            
            # 2. 学習パラメータの更新
            parameter_updates = self._calculate_parameter_updates(feedback_analysis)
            
            # 3. 信頼度較正の更新
            calibration_updates = self._calculate_calibration_updates(feedback_analysis)
            
            # 4. 更新の適用
            self._apply_updates(correction_engine, parameter_updates, calibration_updates)
            
            # 5. 学習メトリクスの計算
            learning_metrics = self._calculate_learning_metrics(feedback_analysis)
            
            self.logger.info(f"Applied continuous learning updates: {len(parameter_updates)} parameters, "
                           f"{len(calibration_updates)} calibration updates")
            
            return {
                'parameters_updated': len(parameter_updates),
                'calibration_updated': len(calibration_updates),
                'feedback_processed': len(new_feedback),
                'learning_metrics': learning_metrics,
                'update_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to apply continuous learning update: {e}")
            return {'error': str(e)}
    
    def _analyze_feedback(self, feedback_data: List[Any]) -> Dict[str, Any]:
        """フィードバックデータを分析"""
        
        try:
            if not feedback_data:
                return {'total_feedback': 0}
            
            # 精度データの抽出
            accuracies = []
            confidence_errors = []
            method_performance = {}
            
            for record in feedback_data:
                if hasattr(record, 'accuracy') and record.accuracy is not None:
                    accuracies.append(record.accuracy)
                
                if hasattr(record, 'confidence_score') and hasattr(record, 'accuracy'):
                    if record.accuracy is not None:
                        error = abs(record.confidence_score - record.accuracy)
                        confidence_errors.append(error)
                
                # メソッド別パフォーマンス
                if hasattr(record, 'method') and hasattr(record, 'accuracy'):
                    method = record.method
                    if method not in method_performance:
                        method_performance[method] = []
                    if record.accuracy is not None:
                        method_performance[method].append(record.accuracy)
            
            # 統計計算
            analysis = {
                'total_feedback': len(feedback_data),
                'avg_accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
                'accuracy_std': float(np.std(accuracies)) if accuracies else 0.0,
                'avg_confidence_error': float(np.mean(confidence_errors)) if confidence_errors else 0.0,
                'method_performance': {
                    method: {
                        'count': len(accs),
                        'avg_accuracy': float(np.mean(accs)),
                        'std_accuracy': float(np.std(accs))
                    }
                    for method, accs in method_performance.items()
                },
                'performance_trend': self._calculate_performance_trend(accuracies)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze feedback: {e}")
            return {'total_feedback': 0, 'error': str(e)}
    
    def _calculate_performance_trend(self, accuracies: List[float]) -> str:
        """パフォーマンストレンドを計算"""
        
        try:
            if len(accuracies) < 10:
                return 'insufficient_data'
            
            # 最近の半分と古い半分を比較
            mid_point = len(accuracies) // 2
            recent_half = accuracies[mid_point:]
            older_half = accuracies[:mid_point]
            
            recent_avg = np.mean(recent_half)
            older_avg = np.mean(older_half)
            
            improvement = recent_avg - older_avg
            
            if improvement > 0.05:
                return 'improving'
            elif improvement < -0.05:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            self.logger.error(f"Failed to calculate performance trend: {e}")
            return 'unknown'
    
    def _calculate_parameter_updates(self, feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """パラメータ更新を計算"""
        
        try:
            updates = {}
            
            avg_accuracy = feedback_analysis.get('avg_accuracy', 0.5)
            performance_trend = feedback_analysis.get('performance_trend', 'stable')
            
            # 精度に基づく補正強度の調整
            if avg_accuracy < 0.6:
                # 精度が低い場合は補正を強化
                updates['correction_strength_adjustment'] = 0.1
            elif avg_accuracy > 0.8:
                # 精度が高い場合は補正を緩和
                updates['correction_strength_adjustment'] = -0.05
            
            # パフォーマンストレンドに基づく学習率の調整
            if performance_trend == 'declining':
                updates['learning_rate_adjustment'] = 0.2  # 学習率を上げる
            elif performance_trend == 'improving':
                updates['learning_rate_adjustment'] = -0.1  # 学習率を下げる（安定化）
            
            # メソッド別の調整
            method_performance = feedback_analysis.get('method_performance', {})
            for method, perf in method_performance.items():
                if perf['count'] >= 5:  # 十分なサンプルがある場合
                    method_key = f'method_{method}_weight'
                    if perf['avg_accuracy'] > 0.7:
                        updates[method_key] = 0.05  # 重みを増加
                    elif perf['avg_accuracy'] < 0.5:
                        updates[method_key] = -0.1  # 重みを減少
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Failed to calculate parameter updates: {e}")
            return {}
    
    def _calculate_calibration_updates(self, feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """較正更新を計算"""
        
        try:
            updates = {}
            
            confidence_error = feedback_analysis.get('avg_confidence_error', 0.0)
            
            # 信頼度誤差に基づく較正強度の調整
            if confidence_error > 0.2:
                updates['calibration_strength'] = 0.1
            elif confidence_error < 0.05:
                updates['calibration_strength'] = -0.05
            
            # 較正方法の選択
            method_performance = feedback_analysis.get('method_performance', {})
            if method_performance:
                # 最も性能の良いメソッドの較正パラメータを強化
                best_method = max(
                    method_performance.items(), 
                    key=lambda x: x[1]['avg_accuracy']
                )[0]
                updates[f'calibration_weight_{best_method}'] = 0.1
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Failed to calculate calibration updates: {e}")
            return {}
    
    def _apply_updates(self, 
                     correction_engine: Any,
                     parameter_updates: Dict[str, Any],
                     calibration_updates: Dict[str, Any]):
        """更新を適用"""
        
        try:
            # 選択された学習アルゴリズムを使用
            learning_algorithm = self.learning_algorithms.get(self.algorithm, self._adam_update)
            
            # パラメータ更新の適用
            for param_name, gradient in parameter_updates.items():
                if hasattr(correction_engine, param_name.replace('_adjustment', '')):
                    current_value = getattr(correction_engine, param_name.replace('_adjustment', ''))
                    
                    # 学習アルゴリズムによる更新
                    updated_value = learning_algorithm(param_name, current_value, gradient)
                    
                    # 値の制限
                    if 'correction_strength' in param_name:
                        updated_value = max(0.1, min(0.8, updated_value))
                    elif 'learning_rate' in param_name:
                        updated_value = max(0.001, min(0.1, updated_value))
                    
                    # 値を設定
                    setattr(correction_engine, param_name.replace('_adjustment', ''), updated_value)
                    self.logger.debug(f"Updated {param_name}: {current_value} -> {updated_value}")
            
            # 較正更新の適用
            if hasattr(correction_engine, 'confidence_calibrator'):
                calibrator = correction_engine.confidence_calibrator
                for calib_name, adjustment in calibration_updates.items():
                    if hasattr(calibrator, calib_name.replace('calibration_', '')):
                        current_value = getattr(calibrator, calib_name.replace('calibration_', ''))
                        updated_value = learning_algorithm(calib_name, current_value, adjustment)
                        setattr(calibrator, calib_name.replace('calibration_', ''), updated_value)
                        self.logger.debug(f"Updated calibration {calib_name}: {current_value} -> {updated_value}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply updates: {e}")
    
    def _gradient_descent_update(self, param_name: str, current_value: float, gradient: float) -> float:
        """勾配降下法による更新"""
        return current_value - self.learning_rate * gradient
    
    def _momentum_update(self, param_name: str, current_value: float, gradient: float) -> float:
        """モメンタム法による更新"""
        if param_name not in self._velocity:
            self._velocity[param_name] = 0.0
        
        self._velocity[param_name] = self.momentum * self._velocity[param_name] - self.learning_rate * gradient
        return current_value + self._velocity[param_name]
    
    def _adam_update(self, param_name: str, current_value: float, gradient: float) -> float:
        """Adam最適化による更新"""
        if param_name not in self._m:
            self._m[param_name] = 0.0
            self._v[param_name] = 0.0
        
        self._t += 1
        
        # Biased first moment estimate
        self._m[param_name] = self.beta1 * self._m[param_name] + (1 - self.beta1) * gradient
        
        # Biased second raw moment estimate  
        self._v[param_name] = self.beta2 * self._v[param_name] + (1 - self.beta2) * (gradient ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self._m[param_name] / (1 - self.beta1 ** self._t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self._v[param_name] / (1 - self.beta2 ** self._t)
        
        # Update parameter
        return current_value - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def _calculate_learning_metrics(self, feedback_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """学習メトリクスを計算"""
        
        try:
            return {
                'algorithm': self.algorithm,
                'learning_rate': self.learning_rate,
                'feedback_quality': feedback_analysis.get('avg_accuracy', 0.0),
                'confidence_calibration_error': feedback_analysis.get('avg_confidence_error', 0.0),
                'performance_trend': feedback_analysis.get('performance_trend', 'unknown'),
                'learning_steps': self._t,
                'velocity_params': len(self._velocity),
                'adam_params': len(self._m)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate learning metrics: {e}")
            return {'error': str(e)}
    
    def reset_learning_state(self):
        """学習状態をリセット"""
        try:
            self._velocity.clear()
            self._m.clear()
            self._v.clear()
            self._t = 0
            self.logger.info("Learning state reset")
        except Exception as e:
            self.logger.error(f"Failed to reset learning state: {e}")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """学習ステータスを取得"""
        
        try:
            return {
                'algorithm': self.algorithm,
                'learning_rate': self.learning_rate,
                'momentum': self.momentum,
                'continuous_learning_enabled': self.enable_continuous_learning,
                'learning_steps': self._t,
                'tracked_parameters': {
                    'velocity': len(self._velocity),
                    'adam_m': len(self._m),
                    'adam_v': len(self._v)
                },
                'optimizer_settings': {
                    'beta1': self.beta1,
                    'beta2': self.beta2,
                    'epsilon': self.epsilon
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get learning status: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # テスト用コード
    print("AdaptiveLearningEngine モジュールが正常にロードされました")

"""
Module: DSSMS Simulation Quality Manager
File: dssms_simulation_quality_manager.py
Description: 
  Task 1.2: シミュレーション品質の包括的管理システム
  シミュレーション異常検出、データ整合性補正、リアリズム強化

Author: GitHub Copilot
Created: 2025-08-25
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import sys
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from config.logger_config import setup_logger
except ImportError:
    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class DSSMSSimulationQualityManager:
    """シミュレーション品質の包括的管理システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初期化"""
        self.logger = setup_logger(__name__)
        
        # デフォルト設定
        self.default_config = {
            'anomaly_detection': {
                'max_daily_return': 0.2,  # 最大日次リターン
                'max_portfolio_change': 0.15,  # 最大ポートフォリオ変化
                'min_data_points': 5,  # 最小データポイント数
                'volatility_threshold': 0.05  # ボラティリティ閾値
            },
            'consistency_checks': {
                'price_continuity': True,  # 価格連続性チェック
                'volume_validation': True,  # 出来高検証
                'trend_coherence': True,  # トレンド一貫性
                'correlation_limits': 0.95  # 相関上限
            },
            'realism_factors': {
                'market_impact': 0.001,  # マーケットインパクト
                'slippage_factor': 0.0005,  # スリッページ
                'transaction_cost': 0.001,  # 取引コスト
                'liquidity_penalty': 0.002  # 流動性ペナルティ
            }
        }
        
        # 設定読み込み
        self.config = self._load_config(config_path)
        
        # 品質メトリクス
        self.quality_metrics = {
            'anomalies_detected': 0,
            'corrections_applied': 0,
            'realism_adjustments': 0,
            'data_quality_score': 1.0
        }
        
        # 異常履歴
        self.anomaly_history: List[Dict[str, Any]] = []
        
        self.logger.info("DSSMS シミュレーション品質管理システムを初期化しました")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom_config = json.load(f)
                
                # デフォルト設定とマージ
                config = self.default_config.copy()
                config.update(custom_config)
                
                self.logger.info(f"設定ファイル読み込み完了: {config_path}")
                return config
                
            except Exception as e:
                self.logger.warning(f"設定ファイル読み込み失敗: {e}")
        
        self.logger.info("デフォルト設定を使用")
        return self.default_config.copy()
    
    def detect_simulation_anomalies(self, simulation_state: Dict[str, Any]) -> Dict[str, Any]:
        """シミュレーション異常検出"""
        try:
            anomalies = []
            severity_scores = []
            
            # 1. ポートフォリオ価値異常
            portfolio_anomalies = self._detect_portfolio_value_anomalies(simulation_state)
            if portfolio_anomalies:
                anomalies.extend(portfolio_anomalies)
                severity_scores.extend([a.get('severity', 0.5) for a in portfolio_anomalies])
            
            # 2. リターン異常
            return_anomalies = self._detect_return_anomalies(simulation_state)
            if return_anomalies:
                anomalies.extend(return_anomalies)
                severity_scores.extend([a.get('severity', 0.5) for a in return_anomalies])
            
            # 3. データ整合性異常
            consistency_anomalies = self._detect_consistency_anomalies(simulation_state)
            if consistency_anomalies:
                anomalies.extend(consistency_anomalies)
                severity_scores.extend([a.get('severity', 0.5) for a in consistency_anomalies])
            
            # 4. トレンド異常
            trend_anomalies = self._detect_trend_anomalies(simulation_state)
            if trend_anomalies:
                anomalies.extend(trend_anomalies)
                severity_scores.extend([a.get('severity', 0.5) for a in trend_anomalies])
            
            # 異常統計更新
            self.quality_metrics['anomalies_detected'] += len(anomalies)
            
            # 総合異常スコア
            overall_severity = np.mean(severity_scores) if severity_scores else 0.0
            
            result = {
                'anomalies_found': len(anomalies),
                'anomaly_details': anomalies,
                'overall_severity': overall_severity,
                'detection_timestamp': datetime.now().isoformat(),
                'simulation_quality': self._calculate_simulation_quality(anomalies, simulation_state)
            }
            
            # 異常履歴記録
            if anomalies:
                self.anomaly_history.extend(anomalies)
                self.logger.warning(f"シミュレーション異常検出: {len(anomalies)}件 (重要度: {overall_severity:.3f})")
            else:
                self.logger.debug("シミュレーション異常なし")
            
            return result
            
        except Exception as e:
            self.logger.error(f"異常検出エラー: {e}")
            return {
                'anomalies_found': 0,
                'anomaly_details': [],
                'overall_severity': 0.0,
                'error': str(e)
            }
    
    def _detect_portfolio_value_anomalies(self, simulation_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ポートフォリオ価値異常検出"""
        anomalies = []
        
        try:
            portfolio_history = simulation_state.get('portfolio_history', [])
            if len(portfolio_history) < 2:
                return anomalies
            
            # 連続する価値変化をチェック
            for i in range(1, len(portfolio_history)):
                current = portfolio_history[i].get('portfolio_value', 0)
                previous = portfolio_history[i-1].get('portfolio_value', 0)
                
                if previous > 0:
                    change_rate = abs((current / previous) - 1)
                    
                    # 異常な変化率
                    if change_rate > self.config['anomaly_detection']['max_portfolio_change']:
                        anomalies.append({
                            'type': 'portfolio_value_spike',
                            'description': f'異常なポートフォリオ価値変化: {change_rate:.2%}',
                            'severity': min(1.0, change_rate * 3),
                            'timestamp': portfolio_history[i].get('date'),
                            'values': {'previous': previous, 'current': current, 'change_rate': change_rate}
                        })
                
                # ゼロまたは負の価値
                if current <= 0:
                    anomalies.append({
                        'type': 'invalid_portfolio_value',
                        'description': f'無効なポートフォリオ価値: {current}',
                        'severity': 1.0,
                        'timestamp': portfolio_history[i].get('date'),
                        'values': {'current': current}
                    })
            
        except Exception as e:
            self.logger.warning(f"ポートフォリオ価値異常検出エラー: {e}")
        
        return anomalies
    
    def _detect_return_anomalies(self, simulation_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """リターン異常検出"""
        anomalies = []
        
        try:
            daily_returns = simulation_state.get('performance_history', {}).get('daily_returns', [])
            if len(daily_returns) < 5:
                return anomalies
            
            # 統計的異常検出
            returns_array = np.array(daily_returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            # 外れ値検出（3シグマルール）
            for i, ret in enumerate(daily_returns):
                if abs(ret - mean_return) > 3 * std_return:
                    anomalies.append({
                        'type': 'return_outlier',
                        'description': f'異常なリターン: {ret:.4f} (平均: {mean_return:.4f}, 標準偏差: {std_return:.4f})',
                        'severity': min(1.0, abs(ret - mean_return) / (3 * std_return)),
                        'index': i,
                        'values': {'return': ret, 'mean': mean_return, 'std': std_return}
                    })
                
                # 極端なリターン
                if abs(ret) > self.config['anomaly_detection']['max_daily_return']:
                    anomalies.append({
                        'type': 'extreme_return',
                        'description': f'極端なリターン: {ret:.2%}',
                        'severity': min(1.0, abs(ret) / self.config['anomaly_detection']['max_daily_return']),
                        'index': i,
                        'values': {'return': ret}
                    })
            
        except Exception as e:
            self.logger.warning(f"リターン異常検出エラー: {e}")
        
        return anomalies
    
    def _detect_consistency_anomalies(self, simulation_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """データ整合性異常検出"""
        anomalies = []
        
        try:
            # ポジション履歴の整合性
            positions = simulation_state.get('performance_history', {}).get('positions', [])
            timestamps = simulation_state.get('performance_history', {}).get('timestamps', [])
            
            if len(positions) != len(timestamps):
                anomalies.append({
                    'type': 'data_length_mismatch',
                    'description': f'データ長不一致: positions={len(positions)}, timestamps={len(timestamps)}',
                    'severity': 0.8,
                    'values': {'positions_len': len(positions), 'timestamps_len': len(timestamps)}
                })
            
            # 時系列の順序性
            if len(timestamps) > 1:
                for i in range(1, len(timestamps)):
                    if isinstance(timestamps[i-1], datetime) and isinstance(timestamps[i], datetime):
                        if timestamps[i] <= timestamps[i-1]:
                            anomalies.append({
                                'type': 'timestamp_order_error',
                                'description': f'時系列順序エラー: {timestamps[i-1]} -> {timestamps[i]}',
                                'severity': 0.9,
                                'index': i,
                                'values': {'prev': timestamps[i-1], 'current': timestamps[i]}
                            })
            
        except Exception as e:
            self.logger.warning(f"整合性異常検出エラー: {e}")
        
        return anomalies
    
    def _detect_trend_anomalies(self, simulation_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """トレンド異常検出"""
        anomalies = []
        
        try:
            daily_returns = simulation_state.get('performance_history', {}).get('daily_returns', [])
            if len(daily_returns) < 10:
                return anomalies
            
            # トレンドの急激な変化検出
            window_size = 5
            for i in range(window_size, len(daily_returns) - window_size):
                before_window = daily_returns[i-window_size:i]
                after_window = daily_returns[i:i+window_size]
                
                before_trend = np.mean(before_window)
                after_trend = np.mean(after_window)
                
                trend_change = abs(after_trend - before_trend)
                
                # 急激なトレンド変化
                if trend_change > 0.02:  # 2%
                    anomalies.append({
                        'type': 'trend_discontinuity',
                        'description': f'急激なトレンド変化: {before_trend:.4f} -> {after_trend:.4f}',
                        'severity': min(1.0, trend_change * 25),
                        'index': i,
                        'values': {'before_trend': before_trend, 'after_trend': after_trend, 'change': trend_change}
                    })
            
        except Exception as e:
            self.logger.warning(f"トレンド異常検出エラー: {e}")
        
        return anomalies
    
    def correct_data_inconsistencies(self, portfolio_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """データ整合性補正"""
        try:
            corrected_history = portfolio_history.copy()
            corrections_applied = []
            
            # 1. 無効なポートフォリオ価値補正
            corrections_applied.extend(
                self._correct_invalid_portfolio_values(corrected_history)
            )
            
            # 2. 時系列順序補正
            corrections_applied.extend(
                self._correct_timestamp_order(corrected_history)
            )
            
            # 3. 異常なジャンプ補正
            corrections_applied.extend(
                self._correct_value_jumps(corrected_history)
            )
            
            # 統計更新
            self.quality_metrics['corrections_applied'] += len(corrections_applied)
            
            result = {
                'corrected_history': corrected_history,
                'corrections_count': len(corrections_applied),
                'correction_details': corrections_applied,
                'correction_timestamp': datetime.now().isoformat()
            }
            
            if corrections_applied:
                self.logger.info(f"データ整合性補正完了: {len(corrections_applied)}件")
            else:
                self.logger.debug("データ整合性補正: 修正不要")
            
            return result
            
        except Exception as e:
            self.logger.error(f"データ整合性補正エラー: {e}")
            return {
                'corrected_history': portfolio_history,
                'corrections_count': 0,
                'correction_details': [],
                'error': str(e)
            }
    
    def _correct_invalid_portfolio_values(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """無効なポートフォリオ価値補正"""
        corrections = []
        
        for i, record in enumerate(history):
            value = record.get('portfolio_value', 0)
            
            # ゼロまたは負の価値
            if value <= 0:
                # 前後の値から推定
                corrected_value = self._estimate_portfolio_value(history, i)
                record['portfolio_value'] = corrected_value
                
                corrections.append({
                    'type': 'invalid_value_correction',
                    'index': i,
                    'original_value': value,
                    'corrected_value': corrected_value,
                    'method': 'interpolation'
                })
        
        return corrections
    
    def _estimate_portfolio_value(self, history: List[Dict[str, Any]], index: int) -> float:
        """ポートフォリオ価値推定"""
        # 前後の有効な値から線形補間
        before_value = None
        after_value = None
        
        # 前の有効値探索
        for i in range(index - 1, -1, -1):
            val = history[i].get('portfolio_value', 0)
            if val > 0:
                before_value = val
                break
        
        # 後の有効値探索
        for i in range(index + 1, len(history)):
            val = history[i].get('portfolio_value', 0)
            if val > 0:
                after_value = val
                break
        
        # 推定値計算
        if before_value and after_value:
            return (before_value + after_value) / 2
        elif before_value:
            return before_value * 0.999  # 僅かに減少
        elif after_value:
            return after_value * 1.001  # 僅かに増加
        else:
            return 1000000  # デフォルト値
    
    def _correct_timestamp_order(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """時系列順序補正"""
        corrections = []
        
        # 日付でソート
        try:
            history.sort(key=lambda x: x.get('date', datetime.min))
            corrections.append({
                'type': 'timestamp_reorder',
                'description': '時系列順序を修正',
                'method': 'sort_by_date'
            })
        except Exception as e:
            self.logger.warning(f"時系列順序補正エラー: {e}")
        
        return corrections
    
    def _correct_value_jumps(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """価値ジャンプ補正"""
        corrections = []
        
        for i in range(1, len(history)):
            current = history[i].get('portfolio_value', 0)
            previous = history[i-1].get('portfolio_value', 0)
            
            if previous > 0:
                change_rate = abs((current / previous) - 1)
                
                # 異常な変化率を制限
                if change_rate > 0.2:  # 20%以上
                    direction = 1 if current > previous else -1
                    corrected_value = previous * (1 + direction * 0.1)  # 10%に制限
                    
                    history[i]['portfolio_value'] = corrected_value
                    
                    corrections.append({
                        'type': 'value_jump_correction',
                        'index': i,
                        'original_value': current,
                        'corrected_value': corrected_value,
                        'original_change_rate': change_rate,
                        'corrected_change_rate': 0.1
                    })
        
        return corrections
    
    def enhance_realism_factors(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """リアリズム要因の強化"""
        try:
            enhanced_data = market_data.copy()
            
            # 1. マーケットインパクト適用
            enhanced_data = self._apply_market_impact(enhanced_data)
            
            # 2. スリッページ適用
            enhanced_data = self._apply_slippage(enhanced_data)
            
            # 3. 取引コスト適用
            enhanced_data = self._apply_transaction_costs(enhanced_data)
            
            # 4. 流動性ペナルティ適用
            enhanced_data = self._apply_liquidity_penalty(enhanced_data)
            
            # 統計更新
            self.quality_metrics['realism_adjustments'] += 1
            
            result = {
                'enhanced_data': enhanced_data,
                'realism_factors_applied': [
                    'market_impact',
                    'slippage',
                    'transaction_costs',
                    'liquidity_penalty'
                ],
                'enhancement_timestamp': datetime.now().isoformat()
            }
            
            self.logger.debug("リアリズム要因強化完了")
            return result
            
        except Exception as e:
            self.logger.error(f"リアリズム要因強化エラー: {e}")
            return {
                'enhanced_data': market_data,
                'realism_factors_applied': [],
                'error': str(e)
            }
    
    def _apply_market_impact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """マーケットインパクト適用"""
        impact_factor = self.config['realism_factors']['market_impact']
        
        # 取引サイズに応じたインパクト
        if 'transaction_size' in data:
            size_ratio = data['transaction_size'] / 1000000  # 100万円基準
            impact = impact_factor * np.sqrt(size_ratio)
            
            if 'execution_price' in data:
                data['execution_price'] *= (1 + impact)
        
        return data
    
    def _apply_slippage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """スリッページ適用"""
        slippage_factor = self.config['realism_factors']['slippage_factor']
        
        if 'execution_price' in data:
            # ランダムスリッページ
            slippage = np.random.normal(0, slippage_factor)
            data['execution_price'] *= (1 + slippage)
            data['slippage_applied'] = slippage
        
        return data
    
    def _apply_transaction_costs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """取引コスト適用"""
        cost_factor = self.config['realism_factors']['transaction_cost']
        
        if 'transaction_value' in data:
            cost = data['transaction_value'] * cost_factor
            data['transaction_cost'] = cost
            data['net_value'] = data['transaction_value'] - cost
        
        return data
    
    def _apply_liquidity_penalty(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """流動性ペナルティ適用"""
        penalty_factor = self.config['realism_factors']['liquidity_penalty']
        
        # 市場時間外や低流動性銘柄に対するペナルティ
        if data.get('market_hours', True) is False:
            if 'execution_price' in data:
                data['execution_price'] *= (1 - penalty_factor)
                data['liquidity_penalty_applied'] = penalty_factor
        
        return data
    
    def _calculate_simulation_quality(self, anomalies: List[Dict[str, Any]], 
                                    simulation_state: Dict[str, Any]) -> float:
        """シミュレーション品質スコア計算"""
        try:
            base_score = 1.0
            
            # 異常による減点
            if anomalies:
                severity_penalty = np.mean([a.get('severity', 0.5) for a in anomalies])
                anomaly_count_penalty = min(0.5, len(anomalies) * 0.1)
                base_score -= (severity_penalty * 0.3 + anomaly_count_penalty * 0.2)
            
            # データ完全性ボーナス
            portfolio_history = simulation_state.get('portfolio_history', [])
            daily_returns = simulation_state.get('performance_history', {}).get('daily_returns', [])
            
            if len(portfolio_history) > 10 and len(daily_returns) > 10:
                base_score += 0.1  # データ完全性ボーナス
            
            return max(0.0, min(1.0, base_score))
            
        except Exception:
            return 0.5
    
    def get_quality_report(self) -> Dict[str, Any]:
        """品質レポート生成"""
        return {
            'quality_metrics': self.quality_metrics.copy(),
            'anomaly_count': len(self.anomaly_history),
            'recent_anomalies': self.anomaly_history[-5:] if self.anomaly_history else [],
            'config_summary': {
                'anomaly_thresholds': self.config['anomaly_detection'],
                'realism_factors': self.config['realism_factors']
            },
            'report_timestamp': datetime.now().isoformat()
        }

def demo_simulation_quality_manager():
    """シミュレーション品質管理システムデモ"""
    print("=== DSSMS シミュレーション品質管理システムデモ ===")
    
    try:
        # システム初期化
        manager = DSSMSSimulationQualityManager()
        
        # テストデータ作成
        test_simulation_state = {
            'portfolio_history': [
                {'date': datetime.now() - timedelta(days=i), 'portfolio_value': 1000000 * (1 + np.random.normal(0, 0.01))}
                for i in range(10, 0, -1)
            ],
            'performance_history': {
                'daily_returns': [np.random.normal(0.001, 0.02) for _ in range(10)],
                'positions': ['7203.T'] * 10,
                'timestamps': [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
            }
        }
        
        # 異常データを意図的に追加
        test_simulation_state['portfolio_history'][5]['portfolio_value'] = 1500000  # 50%のジャンプ
        test_simulation_state['performance_history']['daily_returns'][3] = 0.25  # 25%のリターン
        
        # 異常検出テスト
        print(f"\n🔍 異常検出テスト")
        anomaly_result = manager.detect_simulation_anomalies(test_simulation_state)
        
        print(f"   検出された異常: {anomaly_result['anomalies_found']}件")
        print(f"   総合重要度: {anomaly_result['overall_severity']:.3f}")
        print(f"   シミュレーション品質: {anomaly_result['simulation_quality']:.3f}")
        
        for anomaly in anomaly_result['anomaly_details'][:3]:  # 最初の3件表示
            print(f"   - {anomaly['type']}: {anomaly['description']}")
        
        # データ整合性補正テスト
        print(f"\n🔧 データ整合性補正テスト")
        correction_result = manager.correct_data_inconsistencies(test_simulation_state['portfolio_history'])
        
        print(f"   適用された補正: {correction_result['corrections_count']}件")
        
        for correction in correction_result['correction_details']:
            print(f"   - {correction['type']}: {correction.get('description', 'N/A')}")
        
        # リアリズム要因強化テスト
        print(f"\n🎯 リアリズム要因強化テスト")
        test_market_data = {
            'execution_price': 1000,
            'transaction_size': 2000000,
            'transaction_value': 1000000,
            'market_hours': False
        }
        
        realism_result = manager.enhance_realism_factors(test_market_data)
        
        print(f"   適用された要因: {len(realism_result['realism_factors_applied'])}種類")
        
        enhanced = realism_result['enhanced_data']
        print(f"   実行価格: {test_market_data['execution_price']} -> {enhanced['execution_price']:.2f}")
        if 'transaction_cost' in enhanced:
            print(f"   取引コスト: {enhanced['transaction_cost']:.0f}円")
        
        # 品質レポート表示
        print(f"\n📊 品質レポート")
        quality_report = manager.get_quality_report()
        
        for key, value in quality_report['quality_metrics'].items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ デモエラー: {e}")
        return False

if __name__ == "__main__":
    success = demo_simulation_quality_manager()
    if success:
        print("\n✅ シミュレーション品質管理システムデモ完了")
    else:
        print("\n❌ シミュレーション品質管理システムデモ失敗")

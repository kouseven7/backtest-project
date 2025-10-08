"""
DSSMS Phase 2 Task 2.2: ポートフォリオ価値追跡システム
Portfolio Value Tracker - 高精度な価値追跡とリアルタイム監視

主要機能:
1. ポートフォリオ価値のリアルタイム追跡
2. 異常値の即座検出と修正
3. 価値変動パターンの分析
4. 履歴データの効率的管理
5. DSSMSPerformanceCalculatorV2との統合

Author: GitHub Copilot Agent
Created: 2025-01-22
Task: Phase 2 Task 2.2 - パフォーマンス計算エンジン修正
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
import json
from collections import deque
import threading
import time

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

# 警告を抑制
warnings.filterwarnings('ignore')

class ValueStatus(Enum):
    """価値ステータス"""
    NORMAL = "normal"
    ANOMALY_DETECTED = "anomaly_detected"
    CORRECTION_APPLIED = "correction_applied"
    DATA_MISSING = "data_missing"
    CALCULATION_ERROR = "calculation_error"

class TrackingMode(Enum):
    """追跡モード"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    HYBRID = "hybrid"

@dataclass
class ValueSnapshot:
    """価値スナップショット"""
    timestamp: datetime
    portfolio_value: float
    cash_value: float
    position_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    status: ValueStatus = ValueStatus.NORMAL
    anomalies: List[str] = field(default_factory=list)
    data_source: str = "unknown"
    confidence_score: float = 1.0

@dataclass
class TrackingConfiguration:
    """追跡設定"""
    tracking_mode: TrackingMode = TrackingMode.HYBRID
    update_interval_seconds: float = 1.0
    anomaly_detection_window: int = 20
    value_change_threshold: float = 0.10  # 10%の変動で異常とみなす
    auto_correction_enabled: bool = True
    historical_retention_days: int = 365
    cache_size_limit: int = 10000

class PortfolioValueTracker:
    """
    ポートフォリオ価値追跡システム
    高精度な価値追跡とリアルタイム監視機能
    """
    
    def __init__(self, config: Optional[TrackingConfiguration] = None):
        """
        Args:
            config: 追跡設定
        """
        self.logger = setup_logger(__name__)
        self.config = config or TrackingConfiguration()
        
        # 価値履歴の管理
        self.value_history: deque = deque(maxlen=self.config.cache_size_limit)
        self.current_snapshot: Optional[ValueSnapshot] = None
        
        # 異常値検出用の統計データ
        self.recent_values: deque = deque(maxlen=self.config.anomaly_detection_window)
        self.value_statistics = {
            'mean': 0.0,
            'std': 0.0,
            'min': float('inf'),
            'max': float('-inf')
        }
        
        # スレッド安全性のためのロック
        self.lock = threading.Lock()
        
        # リアルタイム追跡用
        self.is_tracking = False
        self.tracking_thread = None
        
        self.logger.info("PortfolioValueTracker初期化完了")
    
    def start_tracking(self, data_source_callback: callable):
        """
        リアルタイム追跡開始
        
        Args:
            data_source_callback: データ取得コールバック関数
        """
        if self.is_tracking:
            self.logger.warning("既に追跡中です")
            return
        
        self.is_tracking = True
        self.data_source_callback = data_source_callback
        
        if self.config.tracking_mode in [TrackingMode.REAL_TIME, TrackingMode.HYBRID]:
            self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self.tracking_thread.start()
            self.logger.info("リアルタイム追跡開始")
    
    def stop_tracking(self):
        """リアルタイム追跡停止"""
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5.0)
        self.logger.info("リアルタイム追跡停止")
    
    def _tracking_loop(self):
        """追跡ループ（別スレッドで実行）"""
        while self.is_tracking:
            try:
                # データ源からの値取得
                value_data = self.data_source_callback()
                if value_data:
                    self.update_value(value_data)
                
                time.sleep(self.config.update_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"追跡ループエラー: {e}")
                time.sleep(1.0)  # エラー時は短い間隔で再試行
    
    def update_value(self, value_data: Dict[str, Any]) -> ValueSnapshot:
        """
        価値データの更新
        
        Args:
            value_data: 価値データ辞書
            
        Returns:
            更新された価値スナップショット
        """
        with self.lock:
            try:
                # 価値スナップショットの作成
                snapshot = self._create_snapshot(value_data)
                
                # 異常値検出
                anomalies = self._detect_value_anomalies(snapshot)
                if anomalies:
                    snapshot.anomalies = anomalies
                    snapshot.status = ValueStatus.ANOMALY_DETECTED
                    
                    # 自動修正が有効な場合
                    if self.config.auto_correction_enabled:
                        snapshot = self._apply_value_corrections(snapshot)
                
                # 履歴の更新
                self.value_history.append(snapshot)
                self.current_snapshot = snapshot
                
                # 統計の更新
                self._update_statistics(snapshot.portfolio_value)
                
                self.logger.debug(f"価値更新: ¥{snapshot.portfolio_value:,.0f} (ステータス: {snapshot.status.value})")
                return snapshot
                
            except Exception as e:
                self.logger.error(f"価値更新エラー: {e}")
                return self._create_error_snapshot(value_data)
    
    def _create_snapshot(self, value_data: Dict[str, Any]) -> ValueSnapshot:
        """価値スナップショットの作成"""
        try:
            timestamp = value_data.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            # 基本的な価値データの抽出
            portfolio_value = float(value_data.get('portfolio_value', 0.0))
            cash_value = float(value_data.get('cash_value', 0.0))
            position_value = float(value_data.get('position_value', 0.0))
            
            # P&Lの計算
            unrealized_pnl = float(value_data.get('unrealized_pnl', 0.0))
            realized_pnl = float(value_data.get('realized_pnl', 0.0))
            total_pnl = unrealized_pnl + realized_pnl
            
            # データソースの識別
            data_source = value_data.get('data_source', 'unknown')
            
            # 信頼性スコアの計算
            confidence_score = self._calculate_confidence_score(value_data)
            
            return ValueSnapshot(
                timestamp=timestamp,
                portfolio_value=portfolio_value,
                cash_value=cash_value,
                position_value=position_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_pnl=total_pnl,
                data_source=data_source,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"スナップショット作成エラー: {e}")
            return ValueSnapshot(
                timestamp=datetime.now(),
                portfolio_value=0.0,
                cash_value=0.0,
                position_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                total_pnl=0.0,
                status=ValueStatus.CALCULATION_ERROR,
                anomalies=[f"スナップショット作成エラー: {str(e)}"]
            )
    
    def _calculate_confidence_score(self, value_data: Dict[str, Any]) -> float:
        """信頼性スコアの計算"""
        score = 1.0
        
        # 必要なフィールドの存在チェック
        required_fields = ['portfolio_value', 'timestamp']
        missing_fields = [field for field in required_fields if field not in value_data]
        score -= len(missing_fields) * 0.2
        
        # データの妥当性チェック
        portfolio_value = value_data.get('portfolio_value', 0)
        if portfolio_value <= 0:
            score *= 0.5
        
        # データソースの信頼性
        data_source = value_data.get('data_source', 'unknown')
        if data_source == 'unknown':
            score *= 0.8
        elif data_source in ['test', 'simulation']:
            score *= 0.9
        
        return max(0.0, min(1.0, score))
    
    def _detect_value_anomalies(self, snapshot: ValueSnapshot) -> List[str]:
        """価値異常値の検出"""
        anomalies = []
        
        try:
            # 基本的な値チェック
            if snapshot.portfolio_value <= 0:
                anomalies.append("ポートフォリオ価値がゼロ以下")
            
            if np.isnan(snapshot.portfolio_value) or np.isinf(snapshot.portfolio_value):
                anomalies.append("ポートフォリオ価値が無効な数値")
            
            # 前回値との比較（急激な変動の検出）
            if self.current_snapshot and len(self.recent_values) > 0:
                previous_value = self.current_snapshot.portfolio_value
                if previous_value > 0:
                    change_ratio = abs(snapshot.portfolio_value - previous_value) / previous_value
                    if change_ratio > self.config.value_change_threshold:
                        anomalies.append(f"急激な価値変動: {change_ratio:.2%}")
            
            # 統計的異常値の検出
            if len(self.recent_values) >= 5:
                mean_value = np.mean(list(self.recent_values))
                std_value = np.std(list(self.recent_values))
                
                if std_value > 0:
                    z_score = abs(snapshot.portfolio_value - mean_value) / std_value
                    if z_score > 3.0:  # 3σ以上の偏差
                        anomalies.append(f"統計的異常値: Z-score={z_score:.2f}")
            
            # 構成要素の整合性チェック
            calculated_total = snapshot.cash_value + snapshot.position_value
            if abs(calculated_total - snapshot.portfolio_value) > snapshot.portfolio_value * 0.01:  # 1%以上の差異
                anomalies.append("構成要素の合計値が不整合")
            
        except Exception as e:
            anomalies.append(f"異常値検出エラー: {str(e)}")
        
        return anomalies
    
    def _apply_value_corrections(self, snapshot: ValueSnapshot) -> ValueSnapshot:
        """価値修正の適用"""
        corrected_snapshot = snapshot
        corrections_applied = []
        
        try:
            # ゼロ以下の値の修正
            if snapshot.portfolio_value <= 0:
                if len(self.recent_values) > 0:
                    corrected_snapshot.portfolio_value = list(self.recent_values)[-1]
                    corrections_applied.append("ゼロ以下の値を前回値で補正")
                else:
                    corrected_snapshot.portfolio_value = 1000000.0  # デフォルト値
                    corrections_applied.append("ゼロ以下の値をデフォルト値で補正")
            
            # 無効な数値の修正
            if np.isnan(corrected_snapshot.portfolio_value) or np.isinf(corrected_snapshot.portfolio_value):
                if len(self.recent_values) > 0:
                    corrected_snapshot.portfolio_value = np.mean(list(self.recent_values))
                    corrections_applied.append("無効な数値を平均値で補正")
                else:
                    corrected_snapshot.portfolio_value = 1000000.0
                    corrections_applied.append("無効な数値をデフォルト値で補正")
            
            # 急激な変動の修正
            if self.current_snapshot and len(self.recent_values) > 0:
                previous_value = self.current_snapshot.portfolio_value
                if previous_value > 0:
                    change_ratio = abs(corrected_snapshot.portfolio_value - previous_value) / previous_value
                    if change_ratio > self.config.value_change_threshold:
                        # 変動を閾値内に制限
                        max_change = previous_value * self.config.value_change_threshold
                        if corrected_snapshot.portfolio_value > previous_value:
                            corrected_snapshot.portfolio_value = previous_value + max_change
                        else:
                            corrected_snapshot.portfolio_value = previous_value - max_change
                        corrections_applied.append(f"急激な変動を{self.config.value_change_threshold:.1%}以内に制限")
            
            if corrections_applied:
                corrected_snapshot.status = ValueStatus.CORRECTION_APPLIED
                corrected_snapshot.anomalies.extend(corrections_applied)
                self.logger.info(f"価値修正適用: {', '.join(corrections_applied)}")
            
        except Exception as e:
            self.logger.error(f"価値修正エラー: {e}")
            corrected_snapshot.anomalies.append(f"修正処理エラー: {str(e)}")
        
        return corrected_snapshot
    
    def _update_statistics(self, value: float):
        """統計データの更新"""
        try:
            # 最近の値リストに追加
            self.recent_values.append(value)
            
            # 統計の再計算
            if len(self.recent_values) > 0:
                values_list = list(self.recent_values)
                self.value_statistics['mean'] = np.mean(values_list)
                self.value_statistics['std'] = np.std(values_list)
                self.value_statistics['min'] = min(self.value_statistics['min'], value)
                self.value_statistics['max'] = max(self.value_statistics['max'], value)
        
        except Exception as e:
            self.logger.error(f"統計更新エラー: {e}")
    
    def _create_error_snapshot(self, value_data: Dict[str, Any]) -> ValueSnapshot:
        """エラー時スナップショットの作成"""
        return ValueSnapshot(
            timestamp=datetime.now(),
            portfolio_value=0.0,
            cash_value=0.0,
            position_value=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            status=ValueStatus.CALCULATION_ERROR,
            anomalies=["価値データの処理中にエラーが発生しました"],
            confidence_score=0.0
        )
    
    def get_value_history(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[ValueSnapshot]:
        """
        価値履歴の取得
        
        Args:
            start_time: 開始時刻
            end_time: 終了時刻
            
        Returns:
            指定期間の価値履歴
        """
        with self.lock:
            history = list(self.value_history)
            
            if start_time or end_time:
                filtered_history = []
                for snapshot in history:
                    if start_time and snapshot.timestamp < start_time:
                        continue
                    if end_time and snapshot.timestamp > end_time:
                        continue
                    filtered_history.append(snapshot)
                return filtered_history
            
            return history
    
    def get_current_value(self) -> Optional[ValueSnapshot]:
        """現在の価値取得"""
        with self.lock:
            return self.current_snapshot
    
    def get_value_statistics(self) -> Dict[str, float]:
        """価値統計の取得"""
        with self.lock:
            return self.value_statistics.copy()
    
    def export_to_dataframe(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        DataFrameへのエクスポート
        
        Args:
            start_time: 開始時刻
            end_time: 終了時刻
            
        Returns:
            価値履歴のDataFrame
        """
        history = self.get_value_history(start_time, end_time)
        
        if not history:
            return pd.DataFrame()
        
        data = []
        for snapshot in history:
            data.append({
                'timestamp': snapshot.timestamp,
                'portfolio_value': snapshot.portfolio_value,
                'cash_value': snapshot.cash_value,
                'position_value': snapshot.position_value,
                'unrealized_pnl': snapshot.unrealized_pnl,
                'realized_pnl': snapshot.realized_pnl,
                'total_pnl': snapshot.total_pnl,
                'status': snapshot.status.value,
                'confidence_score': snapshot.confidence_score,
                'anomaly_count': len(snapshot.anomalies)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def generate_tracking_report(self) -> Dict[str, Any]:
        """追跡レポートの生成"""
        with self.lock:
            current_time = datetime.now()
            history_count = len(self.value_history)
            
            # 異常値統計
            anomaly_count = sum(1 for snapshot in self.value_history if snapshot.anomalies)
            correction_count = sum(1 for snapshot in self.value_history if snapshot.status == ValueStatus.CORRECTION_APPLIED)
            
            # パフォーマンス統計
            if history_count > 1:
                first_value = list(self.value_history)[0].portfolio_value
                last_value = list(self.value_history)[-1].portfolio_value
                total_return = (last_value / first_value - 1) if first_value > 0 else 0.0
            else:
                total_return = 0.0
            
            report = {
                'tracking_summary': {
                    'is_active': self.is_tracking,
                    'tracking_mode': self.config.tracking_mode.value,
                    'update_interval': self.config.update_interval_seconds,
                    'history_count': history_count,
                    'current_time': current_time.isoformat()
                },
                'value_statistics': self.value_statistics.copy(),
                'quality_metrics': {
                    'anomaly_rate': anomaly_count / history_count if history_count > 0 else 0.0,
                    'correction_rate': correction_count / history_count if history_count > 0 else 0.0,
                    'average_confidence': np.mean([s.confidence_score for s in self.value_history]) if history_count > 0 else 0.0
                },
                'performance_summary': {
                    'total_return': total_return,
                    'current_value': self.current_snapshot.portfolio_value if self.current_snapshot else 0.0
                }
            }
            
            return report

def main():
    """メイン実行関数"""
    print("DSSMS Task 2.2: ポートフォリオ価値追跡システム")
    print("=" * 50)
    
    try:
        # 設定の作成
        config = TrackingConfiguration(
            tracking_mode=TrackingMode.BATCH,  # テスト用はバッチモード
            anomaly_detection_window=10,
            value_change_threshold=0.15,
            auto_correction_enabled=True
        )
        
        # 価値追跡システムの初期化
        tracker = PortfolioValueTracker(config)
        
        # サンプルデータでのテスト
        print("\n[UP] サンプルデータでの価値追跡テスト:")
        
        sample_values = [
            {'timestamp': '2024-01-01 09:00:00', 'portfolio_value': 1000000, 'cash_value': 200000, 'position_value': 800000},
            {'timestamp': '2024-01-01 10:00:00', 'portfolio_value': 1005000, 'cash_value': 200000, 'position_value': 805000},
            {'timestamp': '2024-01-01 11:00:00', 'portfolio_value': 995000, 'cash_value': 200000, 'position_value': 795000},
            {'timestamp': '2024-01-01 12:00:00', 'portfolio_value': 0.01, 'cash_value': 0, 'position_value': 0.01},  # 異常値
            {'timestamp': '2024-01-01 13:00:00', 'portfolio_value': 1002000, 'cash_value': 200000, 'position_value': 802000},
        ]
        
        for i, value_data in enumerate(sample_values, 1):
            snapshot = tracker.update_value(value_data)
            print(f"  {i}. {snapshot.timestamp.strftime('%H:%M')} - ¥{snapshot.portfolio_value:,.0f}")
            print(f"     ステータス: {snapshot.status.value}, 信頼度: {snapshot.confidence_score:.2f}")
            if snapshot.anomalies:
                print(f"     異常値: {', '.join(snapshot.anomalies)}")
        
        # 統計情報の表示
        stats = tracker.get_value_statistics()
        print(f"\n[CHART] 統計情報:")
        print(f"  平均値: ¥{stats['mean']:,.0f}")
        print(f"  標準偏差: ¥{stats['std']:,.0f}")
        print(f"  最小値: ¥{stats['min']:,.0f}")
        print(f"  最大値: ¥{stats['max']:,.0f}")
        
        # レポート生成
        report = tracker.generate_tracking_report()
        print(f"\n[LIST] 追跡レポート:")
        print(f"  履歴数: {report['tracking_summary']['history_count']}")
        print(f"  異常値率: {report['quality_metrics']['anomaly_rate']:.2%}")
        print(f"  修正率: {report['quality_metrics']['correction_rate']:.2%}")
        print(f"  平均信頼度: {report['quality_metrics']['average_confidence']:.2f}")
        
        # DataFrameエクスポートのテスト
        df = tracker.export_to_dataframe()
        print(f"\n📁 データ出力: {len(df)}行のDataFrame生成")
        
        print(f"\n[OK] ポートフォリオ価値追跡システム: 正常動作確認")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] エラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

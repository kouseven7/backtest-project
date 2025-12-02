"""
最適化履歴管理システム

ベイジアン最適化の履歴とパフォーマンス追跡
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pickle
import logging
from pathlib import Path

@dataclass
class OptimizationSession:
    """最適化セッション"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_iterations: int
    best_score: float
    convergence_achieved: bool
    initial_weights: Dict[str, float]
    final_weights: Dict[str, float]
    learning_mode: str
    
@dataclass
class PerformanceRecord:
    """パフォーマンス記録"""
    timestamp: datetime
    weights: Dict[str, float]
    performance_score: float
    individual_metrics: Dict[str, float]
    iteration: int
    session_id: str

class OptimizationHistoryManager:
    """
    最適化履歴管理システム
    
    ベイジアン最適化の全プロセスを記録し、
    パフォーマンス追跡、トレンド分析、最適化効率の監視を行う。
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_history_days: int = 365
    ):
        """
        初期化
        
        Args:
            storage_path: 履歴保存パス
            max_history_days: 履歴保持日数
        """
        self.logger = self._setup_logger()
        self.storage_path = Path(storage_path) if storage_path else Path("optimization_history")
        self.max_history_days = max_history_days
        
        # 履歴データ
        self.sessions = []
        self.performance_records = []
        
        # 現在のセッション
        self.current_session = None
        
        # 統計キャッシュ
        self._statistics_cache = {}
        self._cache_timestamp = None
        
        # ストレージの初期化
        self._initialize_storage()
        
        # 既存履歴の読み込み
        self._load_existing_history()
        
        self.logger.info("OptimizationHistoryManager initialized")
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(f"{__name__}.OptimizationHistoryManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _initialize_storage(self) -> None:
        """ストレージの初期化"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # サブディレクトリの作成
        (self.storage_path / "sessions").mkdir(exist_ok=True)
        (self.storage_path / "performance").mkdir(exist_ok=True)
        (self.storage_path / "exports").mkdir(exist_ok=True)
        
    def _load_existing_history(self) -> None:
        """既存履歴の読み込み"""
        try:
            # セッション履歴の読み込み
            sessions_file = self.storage_path / "sessions.json"
            if sessions_file.exists():
                with open(sessions_file, 'r', encoding='utf-8') as f:
                    sessions_data = json.load(f)
                    
                self.sessions = [
                    OptimizationSession(**session_data) 
                    for session_data in sessions_data
                ]
                
            # パフォーマンス記録の読み込み
            performance_file = self.storage_path / "performance_records.json"
            if performance_file.exists():
                with open(performance_file, 'r', encoding='utf-8') as f:
                    records_data = json.load(f)
                    
                self.performance_records = [
                    PerformanceRecord(
                        timestamp=datetime.fromisoformat(record['timestamp']),
                        weights=record['weights'],
                        performance_score=record['performance_score'],
                        individual_metrics=record['individual_metrics'],
                        iteration=record['iteration'],
                        session_id=record['session_id']
                    )
                    for record in records_data
                ]
                
            # 古い履歴のクリーンアップ
            self._cleanup_old_history()
            
            self.logger.info(f"Loaded {len(self.sessions)} sessions and {len(self.performance_records)} records")
            
        except Exception as e:
            self.logger.warning(f"Error loading existing history: {e}")
            
    def start_optimization_session(
        self,
        initial_weights: Dict[str, float],
        learning_mode: str
    ) -> str:
        """
        最適化セッションの開始
        
        Args:
            initial_weights: 初期重み
            learning_mode: 学習モード
            
        Returns:
            セッションID
        """
        session_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = OptimizationSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            total_iterations=0,
            best_score=-np.inf,
            convergence_achieved=False,
            initial_weights=initial_weights.copy(),
            final_weights={},
            learning_mode=learning_mode
        )
        
        self.logger.info(f"Started optimization session: {session_id}")
        return session_id
        
    def record_iteration(
        self,
        weights: Dict[str, float],
        performance_score: float,
        individual_metrics: Dict[str, float],
        iteration: int
    ) -> None:
        """
        反復結果の記録
        
        Args:
            weights: 現在の重み
            performance_score: パフォーマンススコア
            individual_metrics: 個別指標
            iteration: 反復回数
        """
        if self.current_session is None:
            self.logger.warning("No active session - starting new session")
            self.start_optimization_session(weights, "unknown")
            
        # パフォーマンス記録の作成
        record = PerformanceRecord(
            timestamp=datetime.now(),
            weights=weights.copy(),
            performance_score=performance_score,
            individual_metrics=individual_metrics.copy(),
            iteration=iteration,
            session_id=self.current_session.session_id
        )
        
        self.performance_records.append(record)
        
        # セッション情報の更新
        self.current_session.total_iterations = iteration
        
        if performance_score > self.current_session.best_score:
            self.current_session.best_score = performance_score
            self.current_session.final_weights = weights.copy()
            
        # 統計キャッシュの無効化
        self._invalidate_cache()
        
        self.logger.debug(f"Recorded iteration {iteration} with score {performance_score:.4f}")
        
    def end_optimization_session(self, convergence_achieved: bool = False) -> None:
        """
        最適化セッションの終了
        
        Args:
            convergence_achieved: 収束達成フラグ
        """
        if self.current_session is None:
            self.logger.warning("No active session to end")
            return
            
        self.current_session.end_time = datetime.now()
        self.current_session.convergence_achieved = convergence_achieved
        
        # セッション履歴に追加
        self.sessions.append(self.current_session)
        
        # ストレージに保存
        self._save_history()
        
        self.logger.info(
            f"Ended optimization session: {self.current_session.session_id}, "
            f"iterations: {self.current_session.total_iterations}, "
            f"best score: {self.current_session.best_score:.4f}, "
            f"converged: {convergence_achieved}"
        )
        
        self.current_session = None
        
    def _save_history(self) -> None:
        """履歴の保存"""
        try:
            # セッション履歴の保存
            sessions_data = []
            for session in self.sessions:
                session_dict = asdict(session)
                # datetimeオブジェクトを文字列に変換
                session_dict['start_time'] = session.start_time.isoformat()
                if session.end_time:
                    session_dict['end_time'] = session.end_time.isoformat()
                sessions_data.append(session_dict)
                
            with open(self.storage_path / "sessions.json", 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, indent=2, ensure_ascii=False)
                
            # パフォーマンス記録の保存
            records_data = []
            for record in self.performance_records:
                record_dict = {
                    'timestamp': record.timestamp.isoformat(),
                    'weights': record.weights,
                    'performance_score': record.performance_score,
                    'individual_metrics': record.individual_metrics,
                    'iteration': record.iteration,
                    'session_id': record.session_id
                }
                records_data.append(record_dict)
                
            with open(self.storage_path / "performance_records.json", 'w', encoding='utf-8') as f:
                json.dump(records_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error saving history: {e}")
            
    def _cleanup_old_history(self) -> None:
        """古い履歴のクリーンアップ"""
        cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
        
        # 古いセッションの削除
        self.sessions = [
            session for session in self.sessions
            if session.start_time >= cutoff_date
        ]
        
        # 古いパフォーマンス記録の削除
        self.performance_records = [
            record for record in self.performance_records
            if record.timestamp >= cutoff_date
        ]
        
    def get_optimization_statistics(
        self,
        lookback_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        最適化統計の取得
        
        Args:
            lookback_days: 遡及日数
            
        Returns:
            統計情報
        """
        # キャッシュチェック
        cache_key = f"stats_{lookback_days or 'all'}"
        if (self._cache_timestamp and 
            (datetime.now() - self._cache_timestamp).seconds < 300 and
            cache_key in self._statistics_cache):
            return self._statistics_cache[cache_key]
            
        # データの絞り込み
        if lookback_days:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            sessions = [s for s in self.sessions if s.start_time >= cutoff_date]
            records = [r for r in self.performance_records if r.timestamp >= cutoff_date]
        else:
            sessions = self.sessions
            records = self.performance_records
            
        if not sessions and not records:
            return {}
            
        # 基本統計
        stats = {
            'total_sessions': len(sessions),
            'total_iterations': sum(s.total_iterations for s in sessions),
            'total_records': len(records),
            'average_session_duration': self._calculate_average_session_duration(sessions),
            'convergence_rate': sum(1 for s in sessions if s.convergence_achieved) / len(sessions) if sessions else 0
        }
        
        # パフォーマンス統計
        if records:
            scores = [r.performance_score for r in records]
            stats.update({
                'best_performance': max(scores),
                'worst_performance': min(scores),
                'average_performance': np.mean(scores),
                'performance_std': np.std(scores),
                'performance_trend': self._calculate_performance_trend(records)
            })
            
        # 学習モード統計
        mode_counts = {}
        for session in sessions:
            mode_counts[session.learning_mode] = mode_counts.get(session.learning_mode, 0) + 1
        stats['learning_mode_distribution'] = mode_counts
        
        # 効率性統計
        stats['optimization_efficiency'] = self._calculate_optimization_efficiency(sessions, records)
        
        # キャッシュ更新
        self._statistics_cache[cache_key] = stats
        self._cache_timestamp = datetime.now()
        
        return stats
        
    def _calculate_average_session_duration(self, sessions: List[OptimizationSession]) -> float:
        """平均セッション時間の計算"""
        durations = []
        for session in sessions:
            if session.end_time:
                duration = (session.end_time - session.start_time).total_seconds()
                durations.append(duration)
                
        return np.mean(durations) if durations else 0.0
        
    def _calculate_performance_trend(self, records: List[PerformanceRecord]) -> str:
        """パフォーマンストレンドの計算"""
        if len(records) < 10:
            return "insufficient_data"
            
        # 最近の記録でトレンド分析
        recent_records = sorted(records, key=lambda x: x.timestamp)[-50:]
        scores = [r.performance_score for r in recent_records]
        
        # 線形回帰による傾き計算
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.001:
            return "improving"
        elif slope < -0.001:
            return "deteriorating"
        else:
            return "stable"
            
    def _calculate_optimization_efficiency(
        self,
        sessions: List[OptimizationSession],
        records: List[PerformanceRecord]
    ) -> Dict[str, float]:
        """最適化効率の計算"""
        if not sessions or not records:
            return {}
            
        # 反復あたりの改善度
        improvements_per_iteration = []
        
        for session in sessions:
            session_records = [r for r in records if r.session_id == session.session_id]
            if len(session_records) > 1:
                session_records.sort(key=lambda x: x.iteration)
                initial_score = session_records[0].performance_score
                final_score = session_records[-1].performance_score
                iterations = len(session_records)
                
                if iterations > 0:
                    improvement_rate = (final_score - initial_score) / iterations
                    improvements_per_iteration.append(improvement_rate)
                    
        # 収束速度
        convergence_speeds = []
        for session in sessions:
            if session.convergence_achieved and session.total_iterations > 0:
                convergence_speeds.append(session.total_iterations)
                
        return {
            'average_improvement_per_iteration': np.mean(improvements_per_iteration) if improvements_per_iteration else 0,
            'average_convergence_speed': np.mean(convergence_speeds) if convergence_speeds else 0,
            'efficiency_score': self._calculate_efficiency_score(improvements_per_iteration, convergence_speeds)
        }
        
    def _calculate_efficiency_score(
        self,
        improvements: List[float],
        convergence_speeds: List[float]
    ) -> float:
        """効率スコアの計算"""
        if not improvements or not convergence_speeds:
            return 0.0
            
        # 改善率と収束速度の正規化スコア
        improvement_score = np.mean(improvements) * 100  # スケール調整
        speed_score = 1 / (np.mean(convergence_speeds) + 1)  # 速いほど高スコア
        
        return min(1.0, max(0.0, (improvement_score + speed_score) / 2))
        
    def get_weight_evolution(
        self,
        weight_names: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        重みの進化の取得
        
        Args:
            weight_names: 対象重み名
            session_id: セッションID
            
        Returns:
            重み進化のDataFrame
        """
        # データの絞り込み
        records = self.performance_records
        
        if session_id:
            records = [r for r in records if r.session_id == session_id]
            
        if not records:
            return pd.DataFrame()
            
        # DataFrame作成
        data = []
        for record in records:
            row = {
                'timestamp': record.timestamp,
                'session_id': record.session_id,
                'iteration': record.iteration,
                'performance_score': record.performance_score
            }
            
            # 重みデータの追加
            for weight_name, weight_value in record.weights.items():
                if weight_names is None or weight_name in weight_names:
                    row[weight_name] = weight_value
                    
            data.append(row)
            
        df = pd.DataFrame(data)
        df = df.sort_values(['session_id', 'iteration'])
        
        return df
        
    def get_performance_comparison(self) -> Dict[str, Any]:
        """パフォーマンス比較の取得"""
        if not self.sessions:
            return {}
            
        # セッション間の比較
        session_performance = []
        for session in self.sessions:
            session_records = [r for r in self.performance_records if r.session_id == session.session_id]
            
            if session_records:
                initial_score = min(r.performance_score for r in session_records)
                final_score = max(r.performance_score for r in session_records)
                improvement = final_score - initial_score
                
                session_performance.append({
                    'session_id': session.session_id,
                    'learning_mode': session.learning_mode,
                    'initial_score': initial_score,
                    'final_score': final_score,
                    'improvement': improvement,
                    'iterations': session.total_iterations,
                    'converged': session.convergence_achieved
                })
                
        # 学習モード別の比較
        mode_performance = {}
        for perf in session_performance:
            mode = perf['learning_mode']
            if mode not in mode_performance:
                mode_performance[mode] = {
                    'sessions': 0,
                    'total_improvement': 0,
                    'total_iterations': 0,
                    'convergence_count': 0
                }
                
            mode_performance[mode]['sessions'] += 1
            mode_performance[mode]['total_improvement'] += perf['improvement']
            mode_performance[mode]['total_iterations'] += perf['iterations']
            if perf['converged']:
                mode_performance[mode]['convergence_count'] += 1
                
        # 平均値の計算
        for mode, stats in mode_performance.items():
            if stats['sessions'] > 0:
                stats['average_improvement'] = stats['total_improvement'] / stats['sessions']
                stats['average_iterations'] = stats['total_iterations'] / stats['sessions']
                stats['convergence_rate'] = stats['convergence_count'] / stats['sessions']
                
        return {
            'session_performance': session_performance,
            'mode_performance': mode_performance,
            'best_session': max(session_performance, key=lambda x: x['improvement']) if session_performance else None
        }
        
    def _invalidate_cache(self) -> None:
        """統計キャッシュの無効化"""
        self._statistics_cache.clear()
        self._cache_timestamp = None
        
    def export_history(
        self,
        export_format: str = "csv",
        include_weights: bool = True
    ) -> str:
        """
        履歴のエクスポート
        
        Args:
            export_format: エクスポート形式 ('csv', 'json', 'excel')
            include_weights: 重み情報を含むか
            
        Returns:
            エクスポートファイルのパス
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format == "csv":
            filename = f"optimization_history_{timestamp}.csv"
            filepath = self.storage_path / "exports" / filename
            
            # DataFrameの作成
            records = []
            for record in self.performance_records:
                row = {
                    'timestamp': record.timestamp,
                    'session_id': record.session_id,
                    'iteration': record.iteration,
                    'performance_score': record.performance_score
                }
                
                # 個別指標の追加
                row.update(record.individual_metrics)
                
                # 重み情報の追加
                if include_weights:
                    for weight_name, weight_value in record.weights.items():
                        row[f'weight_{weight_name}'] = weight_value
                        
                records.append(row)
                
            df = pd.DataFrame(records)
            df.to_csv(filepath, index=False)
            
        elif export_format == "json":
            filename = f"optimization_history_{timestamp}.json"
            filepath = self.storage_path / "exports" / filename
            
            export_data = {
                'sessions': [asdict(session) for session in self.sessions],
                'performance_records': [
                    {
                        'timestamp': record.timestamp.isoformat(),
                        'session_id': record.session_id,
                        'iteration': record.iteration,
                        'performance_score': record.performance_score,
                        'individual_metrics': record.individual_metrics,
                        'weights': record.weights if include_weights else {}
                    }
                    for record in self.performance_records
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
                
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
            
        self.logger.info(f"History exported to {filepath}")
        return str(filepath)

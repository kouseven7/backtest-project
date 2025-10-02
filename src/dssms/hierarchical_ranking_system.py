"""
DSSMS階層的銘柄ランキングシステム
Phase 2 Task 2.1: 階層的銘柄ランキングシステム実装
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta

# Phase 1のモジュールをインポート
from .perfect_order_detector import PerfectOrderDetector
from .fundamental_analyzer import FundamentalAnalyzer
from .dssms_data_manager import DSSMSDataManager
from .nikkei225_screener import Nikkei225Screener

class PriorityLevel(Enum):
    """優先度レベル定義"""
    LEVEL_1 = 1  # 全軸パーフェクトオーダー
    LEVEL_2 = 2  # 月週軸パーフェクトオーダー
    LEVEL_3 = 3  # その他

@dataclass
class RankingScore:
    """ランキングスコア詳細情報"""
    symbol: str
    total_score: float
    perfect_order_score: float
    fundamental_score: float
    technical_score: float
    volume_score: float
    volatility_score: float
    priority_group: int
    confidence_level: float
    affordability_penalty: float
    last_updated: datetime

@dataclass
class SelectionResult:
    """銘柄選択結果"""
    primary_candidate: Optional[str]
    backup_candidates: List[str]
    selection_reason: str
    available_fund_ratio: float
    total_candidates_evaluated: int
    priority_distribution: Dict[int, int]

class HierarchicalRankingSystem:
    """優先度ベースの階層的ランキングシステム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        
        # コンポーネント初期化
        self.perfect_order_detector = PerfectOrderDetector()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.data_manager = DSSMSDataManager()
        self.screener = Nikkei225Screener()
        
        # スコア重み設定
        self.scoring_weights = config.get('ranking_system', {}).get('scoring_weights', {
            "fundamental": 0.40,
            "technical": 0.30,
            "volume": 0.20,
            "volatility": 0.10
        })
        
        # キャッシュ設定
        self.ranking_cache = {}
        self.cache_duration = timedelta(minutes=30)
        
    def _setup_logger(self):
        """ロガー設定"""
        logger = logging.getLogger("dssms.hierarchical_ranking")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def categorize_by_perfect_order_priority(self, symbols: List[str]) -> Dict[int, List[str]]:
        """
        パーフェクトオーダー状況による優先度分類
        
        Args:
            symbols: 分析対象銘柄リスト
            
        Returns:
            優先度レベル別の銘柄分類辞書
        """
        priority_groups = {1: [], 2: [], 3: []}
        
        self.logger.info(f"優先度分類開始: {len(symbols)}銘柄")
        
        for symbol in symbols:
            try:
                # マルチタイムフレームデータ取得
                data_dict = self.data_manager.get_multi_timeframe_data(symbol)
                
                # パーフェクトオーダー検出
                po_result = self.perfect_order_detector.check_multi_timeframe_perfect_order(symbol, data_dict)
                
                if po_result is None:
                    priority_groups[3].append(symbol)
                    continue
                
                # 優先度判定
                priority = self._determine_priority_level(po_result)
                priority_groups[priority].append(symbol)
                
            except Exception as e:
                self.logger.warning(f"銘柄 {symbol} の優先度分類エラー: {e}")
                priority_groups[3].append(symbol)
        
        # 結果ログ
        for level, group in priority_groups.items():
            self.logger.info(f"優先度レベル{level}: {len(group)}銘柄")
        
        return priority_groups
    
    def rank_within_priority_group(self, symbols: List[str]) -> List[Tuple[str, float]]:
        """
        同一優先度グループ内での詳細ランキング
        
        Args:
            symbols: 同一優先度グループの銘柄リスト
            
        Returns:
            (銘柄コード, 総合スコア)のタプルリスト（降順ソート）
        """
        ranking_scores = []
        
        self.logger.info(f"グループ内ランキング開始: {len(symbols)}銘柄")
        
        for symbol in symbols:
            try:
                # 各スコア計算
                score_data = self._calculate_comprehensive_score(symbol)
                
                if score_data:
                    ranking_scores.append((symbol, score_data.total_score))
                    
            except Exception as e:
                self.logger.warning(f"銘柄 {symbol} のスコア計算エラー: {e}")
        
        # スコア降順でソート
        ranking_scores.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"ランキング完了: トップ銘柄 {ranking_scores[0][0] if ranking_scores else 'なし'}")
        
        return ranking_scores
    
    def get_top_candidate(self, available_funds: float) -> Optional[str]:
        """
        利用可能資金を考慮した最適候補銘柄選択
        
        Args:
            available_funds: 利用可能資金
            
        Returns:
            最適銘柄コード（購入不可能な場合はNone）
        """
        self.logger.info(f"最適候補選択開始: 利用可能資金 {available_funds:,.0f}円")
        
        # 全銘柄の優先度分類
        screener_result = self._get_screened_symbols(available_funds)
        priority_groups = self.categorize_by_perfect_order_priority(screener_result)
        
        # 優先度順に最適候補を探索
        for priority_level in [1, 2, 3]:
            group_symbols = priority_groups.get(priority_level, [])
            
            if not group_symbols:
                continue
            
            # グループ内ランキング
            ranked_symbols = self.rank_within_priority_group(group_symbols)
            
            # 購入可能性チェック
            for symbol, score in ranked_symbols:
                if self._check_affordability(symbol, available_funds):
                    self.logger.info(f"最適候補決定: {symbol} (優先度レベル{priority_level}, スコア{score:.3f})")
                    return symbol
        
        self.logger.warning("購入可能な最適候補が見つかりませんでした")
        return None
    
    def get_backup_candidates(self, n: int = 5) -> List[str]:
        """
        バックアップ候補銘柄リスト生成
        
        Args:
            n: 取得するバックアップ候補数
            
        Returns:
            バックアップ候補銘柄リスト
        """
        self.logger.info(f"バックアップ候補生成開始: {n}銘柄")
        
        # 全銘柄の優先度分類とランキング
        screener_result = self._get_screened_symbols()
        priority_groups = self.categorize_by_perfect_order_priority(screener_result)
        
        backup_candidates = []
        
        # 優先度順にバックアップ候補を収集
        for priority_level in [1, 2, 3]:
            group_symbols = priority_groups.get(priority_level, [])
            
            if not group_symbols:
                continue
            
            ranked_symbols = self.rank_within_priority_group(group_symbols)
            
            # 上位銘柄をバックアップ候補に追加
            for symbol, score in ranked_symbols:
                if len(backup_candidates) >= n:
                    break
                backup_candidates.append(symbol)
            
            if len(backup_candidates) >= n:
                break
        
        self.logger.info(f"バックアップ候補生成完了: {len(backup_candidates)}銘柄")
        return backup_candidates[:n]
    
    def get_selection_result(self, available_funds: float, backup_count: int = 5) -> SelectionResult:
        """
        統合選択結果生成
        
        Args:
            available_funds: 利用可能資金
            backup_count: バックアップ候補数
            
        Returns:
            選択結果の詳細情報
        """
        # 基本選択処理
        primary = self.get_top_candidate(available_funds)
        backups = self.get_backup_candidates(backup_count)
        
        # 統計情報生成
        screener_result = self._get_screened_symbols(available_funds)
        priority_groups = self.categorize_by_perfect_order_priority(screener_result)
        
        priority_distribution = {
            level: len(symbols) for level, symbols in priority_groups.items()
        }
        
        # 選択理由生成
        reason = self._generate_selection_reason(primary, priority_groups)
        
        return SelectionResult(
            primary_candidate=primary,
            backup_candidates=backups,
            selection_reason=reason,
            available_fund_ratio=self._calculate_fund_utilization(primary, available_funds),
            total_candidates_evaluated=len(screener_result),
            priority_distribution=priority_distribution
        )
    
    # === プライベートメソッド ===
    
    def _determine_priority_level(self, po_result: Dict[str, Any]) -> int:
        """パーフェクトオーダー結果から優先度レベル決定"""
        if not po_result:
            return 3
        
        daily_po = po_result.daily_result.is_perfect_order if po_result.daily_result else False
        weekly_po = po_result.weekly_result.is_perfect_order if po_result.weekly_result else False
        monthly_po = po_result.monthly_result.is_perfect_order if po_result.monthly_result else False
        
        if daily_po and weekly_po and monthly_po:
            return 1  # 全軸パーフェクトオーダー
        elif weekly_po and monthly_po:
            return 2  # 月週軸パーフェクトオーダー
        else:
            return 3  # その他
    
    def _calculate_comprehensive_score(self, symbol: str) -> Optional[RankingScore]:
        """総合スコア計算"""
        try:
            # 各種スコア取得
            fundamental_score = self.fundamental_analyzer.calculate_fundamental_score(symbol)
            technical_score = self._calculate_technical_score(symbol)
            volume_score = self._calculate_volume_score(symbol)
            volatility_score = self._calculate_volatility_score(symbol)
            perfect_order_score = self._calculate_perfect_order_score(symbol)
            
            # 加重平均による総合スコア
            total_score = (
                fundamental_score * self.scoring_weights['fundamental'] +
                technical_score * self.scoring_weights['technical'] +
                volume_score * self.scoring_weights['volume'] +
                volatility_score * self.scoring_weights['volatility']
            )
            
            # 信頼度計算
            confidence = self._calculate_confidence_level(
                fundamental_score, technical_score, volume_score, volatility_score
            )
            
            return RankingScore(
                symbol=symbol,
                total_score=total_score,
                perfect_order_score=perfect_order_score,
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                volume_score=volume_score,
                volatility_score=volatility_score,
                priority_group=0,  # 後で設定
                confidence_level=confidence,
                affordability_penalty=0.0,  # 後で計算
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"銘柄 {symbol} の総合スコア計算エラー: {e}")
            return None
    
    def _calculate_technical_score(self, symbol: str) -> float:
        """テクニカルスコア計算"""
        try:
            data = self.data_manager.get_daily_data(symbol)
            if data is None or len(data) < 50:
                return 0.5  # デフォルトスコア
            
            # RSI計算
            rsi = self._calculate_rsi(data['Close'])
            rsi_score = self._normalize_rsi_score(rsi.iloc[-1])
            
            # MACD計算
            macd_line, signal_line = self._calculate_macd(data['Close'])
            macd_score = 1.0 if macd_line.iloc[-1] > signal_line.iloc[-1] else 0.3
            
            # モメンタム計算
            momentum = data['Close'].pct_change(10).iloc[-1]
            momentum_score = min(max((momentum + 0.1) / 0.2, 0), 1)
            
            return (rsi_score * 0.4 + macd_score * 0.4 + momentum_score * 0.2)
            
        except Exception as e:
            self.logger.warning(f"テクニカルスコア計算エラー {symbol}: {e}")
            return 0.5
    
    def _calculate_volume_score(self, symbol: str) -> float:
        """出来高スコア計算"""
        try:
            data = self.data_manager.get_daily_data(symbol)
            if data is None or len(data) < 20:
                return 0.5
            
            # 平均出来高比較
            recent_volume = data['Volume'].tail(5).mean()
            avg_volume = data['Volume'].tail(20).mean()
            
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 正規化（1.0-3.0の範囲を0.0-1.0にマップ）
            normalized_score = min(max((volume_ratio - 1.0) / 2.0, 0), 1)
            
            return normalized_score
            
        except Exception as e:
            self.logger.warning(f"出来高スコア計算エラー {symbol}: {e}")
            return 0.5
    
    def _calculate_volatility_score(self, symbol: str) -> float:
        """ボラティリティスコア計算"""
        try:
            data = self.data_manager.get_daily_data(symbol)
            if data is None or len(data) < 20:
                return 0.5
            
            # 20日ボラティリティ
            returns = data['Close'].pct_change().dropna()
            volatility = returns.tail(20).std() * np.sqrt(252)  # 年率換算
            
            # 適正ボラティリティ範囲: 15%-35%
            if 0.15 <= volatility <= 0.35:
                return 1.0
            elif volatility < 0.15:
                return 0.7  # 低ボラティリティ
            else:
                return max(0.1, 1.0 - (volatility - 0.35) / 0.5)  # 高ボラティリティペナルティ
                
        except Exception as e:
            self.logger.warning(f"ボラティリティスコア計算エラー {symbol}: {e}")
            return 0.5
    
    def _calculate_perfect_order_score(self, symbol: str) -> float:
        """パーフェクトオーダー強度スコア"""
        try:
            # マルチタイムフレームデータ取得
            data_dict = self.data_manager.get_multi_timeframe_data(symbol)
            
            po_result = self.perfect_order_detector.check_multi_timeframe_perfect_order(symbol, data_dict)
            if not po_result:
                return 0.0
            
            # 各時間軸のパーフェクトオーダー強度を統合
            daily_strength = po_result.daily_result.strength_score if po_result.daily_result else 0.0
            weekly_strength = po_result.weekly_result.strength_score if po_result.weekly_result else 0.0
            monthly_strength = po_result.monthly_result.strength_score if po_result.monthly_result else 0.0
            
            # 加重平均（長期時間軸を重視）
            weighted_score = (
                daily_strength * 0.3 +
                weekly_strength * 0.4 +
                monthly_strength * 0.3
            )
            
            return min(weighted_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"パーフェクトオーダースコア計算エラー {symbol}: {e}")
            return 0.0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """MACD計算"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9).mean()
        return macd_line, signal_line
    
    def _normalize_rsi_score(self, rsi_value: float) -> float:
        """RSI値の正規化スコア"""
        if 30 <= rsi_value <= 70:
            return 1.0  # 適正範囲
        elif rsi_value < 30:
            return 0.8  # 売られ過ぎ（やや有利）
        else:
            return 0.2  # 買われ過ぎ（不利）
    
    def _calculate_confidence_level(self, *scores) -> float:
        """信頼度レベル計算"""
        scores_array = np.array(scores)
        mean_score = np.mean(scores_array)
        score_variance = np.var(scores_array)
        
        # 分散が小さく、平均スコアが高いほど信頼度が高い
        base_confidence = mean_score
        variance_penalty = min(score_variance * 2, 0.3)
        
        return max(base_confidence - variance_penalty, 0.1)
    
    def _check_affordability(self, symbol: str, available_funds: float) -> bool:
        """購入可能性チェック"""
        try:
            data = self.data_manager.get_latest_price(symbol)
            if data is None:
                return False
            
            current_price = data.get('Close', 0)
            min_investment = current_price * 100  # 100株単位
            
            # 利用可能資金の80%以内で購入可能
            return min_investment <= (available_funds * 0.8)
            
        except Exception as e:
            self.logger.warning(f"購入可能性チェックエラー {symbol}: {e}")
            return False
    
    def _get_screened_symbols(self, available_funds: float = 10_000_000) -> List[str]:
        """スクリーニング済み銘柄取得"""
        try:
            # キャッシュチェック
            cache_key = f"screened_symbols_{available_funds}"
            if (cache_key in self.ranking_cache and 
                datetime.now() - self.ranking_cache[cache_key]['timestamp'] < self.cache_duration):
                return self.ranking_cache[cache_key]['data']
            
            # Nikkei225Screenerから取得
            symbols = self.screener.get_filtered_symbols(available_funds)
            
            # キャッシュ更新
            self.ranking_cache[cache_key] = {
                'data': symbols,
                'timestamp': datetime.now()
            }
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"スクリーニング済み銘柄取得エラー: {e}")
            # フォールバック：Phase 1のテスト用銘柄を使用
            return ["7203", "6758", "9984", "8035", "9432"]
    
    def _generate_selection_reason(self, primary: Optional[str], priority_groups: Dict) -> str:
        """選択理由生成"""
        if not primary:
            return "購入可能な適切な候補が見つかりませんでした"
        
        # 優先度レベル判定
        for level, symbols in priority_groups.items():
            if primary in symbols:
                if level == 1:
                    return f"全時間軸パーフェクトオーダー銘柄 {primary} を選択"
                elif level == 2:
                    return f"月週軸パーフェクトオーダー銘柄 {primary} を選択"
                else:
                    return f"その他条件による最適銘柄 {primary} を選択"
        
        return f"総合スコア最優秀銘柄 {primary} を選択"
    
    def _calculate_fund_utilization(self, symbol: Optional[str], available_funds: float) -> float:
        """資金利用率計算"""
        if not symbol:
            return 0.0
        
        try:
            data = self.data_manager.get_latest_price(symbol)
            if data is None:
                return 0.0
            
            current_price = data.get('Close', 0)
            min_investment = current_price * 100
            
            return min_investment / available_funds if available_funds > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"資金利用率計算エラー {symbol}: {e}")
            return 0.0


class DSSMSRankingIntegrator:
    """DSSMSランキングシステムの統合インターフェース"""
    
    def __init__(self, config_path: str = "config/dssms/ranking_config.json"):
        self.logger = self._setup_logger()  # ロガーを最初に初期化
        self.config = self._load_config(config_path)
        self.ranking_system = HierarchicalRankingSystem(self.config)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"設定ファイルが見つかりません: {config_path}")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "ranking_system": {
                "scoring_weights": {
                    "fundamental": 0.40,
                    "technical": 0.30,
                    "volume": 0.20,
                    "volatility": 0.10
                }
            }
        }
    
    def _setup_logger(self):
        """ロガー設定"""
        logger = logging.getLogger("dssms.ranking_integrator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def execute_full_ranking_process(self, available_funds: float) -> SelectionResult:
        """完全ランキングプロセス実行"""
        self.logger.info("DSSMSランキングプロセス開始")
        
        try:
            result = self.ranking_system.get_selection_result(
                available_funds=available_funds,
                backup_count=5
            )
            
            self.logger.info(f"ランキング完了: 主候補={result.primary_candidate}")
            return result
            
        except Exception as e:
            self.logger.error(f"ランキングプロセスエラー: {e}")
            raise
    
    def get_ranking_summary(self, available_funds: float) -> Dict[str, Any]:
        """ランキング結果サマリー"""
        result = self.execute_full_ranking_process(available_funds)
        
        return {
            "execution_timestamp": datetime.now().isoformat(),
            "primary_candidate": result.primary_candidate,
            "backup_candidates": result.backup_candidates,
            "selection_reason": result.selection_reason,
            "available_funds": available_funds,
            "fund_utilization_ratio": result.available_fund_ratio,
            "total_evaluated": result.total_candidates_evaluated,
            "priority_distribution": result.priority_distribution,
            "system_status": "SUCCESS"
        }

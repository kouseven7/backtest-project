"""
市場状況の定義とエニュムレーション
A→B段階的市場分類システムのベース定義
"""
from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass


class SimpleMarketCondition(Enum):
    """シンプル市場分類（既存システムとの互換性維持）"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    RECOVERY = "recovery"


class DetailedMarketCondition(Enum):
    """詳細市場分類（7カテゴリ）"""
    STRONG_BULL = "strong_bull"          # 強気上昇トレンド
    MODERATE_BULL = "moderate_bull"      # 中程度上昇トレンド
    SIDEWAYS_BULL = "sideways_bull"      # 上方向レンジ
    NEUTRAL_SIDEWAYS = "neutral_sideways" # 中立レンジ
    SIDEWAYS_BEAR = "sideways_bear"      # 下方向レンジ
    MODERATE_BEAR = "moderate_bear"      # 中程度下降トレンド
    STRONG_BEAR = "strong_bear"          # 強気下降トレンド


@dataclass
class MarketMetrics:
    """市場メトリクス情報"""
    trend_strength: float           # トレンド強度 (-1.0 to 1.0)
    volatility: float              # ボラティリティ (0.0 to 1.0+)
    momentum: float                # モメンタム (-1.0 to 1.0)
    volume_trend: float            # 出来高トレンド (-1.0 to 1.0)
    price_momentum: float          # 価格モメンタム (-1.0 to 1.0)
    risk_level: float              # リスクレベル (0.0 to 1.0)
    
    # 追加メトリクス
    rsi: Optional[float] = None
    ma_slope: Optional[float] = None
    atr_ratio: Optional[float] = None
    volume_ratio: Optional[float] = None


@dataclass
class ClassificationResult:
    """分類結果の詳細情報"""
    simple_condition: SimpleMarketCondition
    detailed_condition: DetailedMarketCondition
    confidence: float               # 分類信頼度 (0.0 to 1.0)
    metrics: MarketMetrics
    timestamp: str
    symbol: str
    
    # 分類根拠
    classification_reason: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            'simple_condition': self.simple_condition.value,
            'detailed_condition': self.detailed_condition.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'metrics': {
                'trend_strength': self.metrics.trend_strength,
                'volatility': self.metrics.volatility,
                'momentum': self.metrics.momentum,
                'volume_trend': self.metrics.volume_trend,
                'price_momentum': self.metrics.price_momentum,
                'risk_level': self.metrics.risk_level,
                'rsi': self.metrics.rsi,
                'ma_slope': self.metrics.ma_slope,
                'atr_ratio': self.metrics.atr_ratio,
                'volume_ratio': self.metrics.volume_ratio
            },
            'classification_reason': self.classification_reason
        }


class MarketConditions:
    """市場状況管理クラス"""
    
    # A→B変換マッピング（シンプル→詳細）
    SIMPLE_TO_DETAILED_MAPPING = {
        SimpleMarketCondition.TRENDING_BULL: [
            DetailedMarketCondition.STRONG_BULL,
            DetailedMarketCondition.MODERATE_BULL
        ],
        SimpleMarketCondition.TRENDING_BEAR: [
            DetailedMarketCondition.STRONG_BEAR,
            DetailedMarketCondition.MODERATE_BEAR
        ],
        SimpleMarketCondition.SIDEWAYS: [
            DetailedMarketCondition.SIDEWAYS_BULL,
            DetailedMarketCondition.NEUTRAL_SIDEWAYS,
            DetailedMarketCondition.SIDEWAYS_BEAR
        ],
        SimpleMarketCondition.VOLATILE: [
            DetailedMarketCondition.NEUTRAL_SIDEWAYS,  # 高ボラティリティレンジ
        ],
        SimpleMarketCondition.RECOVERY: [
            DetailedMarketCondition.MODERATE_BULL,
            DetailedMarketCondition.SIDEWAYS_BULL
        ]
    }
    
    # B→A変換マッピング（詳細→シンプル）
    DETAILED_TO_SIMPLE_MAPPING = {
        DetailedMarketCondition.STRONG_BULL: SimpleMarketCondition.TRENDING_BULL,
        DetailedMarketCondition.MODERATE_BULL: SimpleMarketCondition.TRENDING_BULL,
        DetailedMarketCondition.SIDEWAYS_BULL: SimpleMarketCondition.SIDEWAYS,
        DetailedMarketCondition.NEUTRAL_SIDEWAYS: SimpleMarketCondition.SIDEWAYS,
        DetailedMarketCondition.SIDEWAYS_BEAR: SimpleMarketCondition.SIDEWAYS,
        DetailedMarketCondition.MODERATE_BEAR: SimpleMarketCondition.TRENDING_BEAR,
        DetailedMarketCondition.STRONG_BEAR: SimpleMarketCondition.TRENDING_BEAR,
    }
    
    @classmethod
    def get_simple_from_detailed(cls, detailed: DetailedMarketCondition) -> SimpleMarketCondition:
        """詳細分類からシンプル分類に変換"""
        return cls.DETAILED_TO_SIMPLE_MAPPING[detailed]
    
    @classmethod
    def get_possible_detailed_from_simple(cls, simple: SimpleMarketCondition) -> list[DetailedMarketCondition]:
        """シンプル分類から可能な詳細分類のリストを取得"""
        return cls.SIMPLE_TO_DETAILED_MAPPING[simple]
    
    @classmethod
    def is_compatible(cls, simple: SimpleMarketCondition, detailed: DetailedMarketCondition) -> bool:
        """シンプルと詳細分類の互換性をチェック"""
        return detailed in cls.SIMPLE_TO_DETAILED_MAPPING[simple]

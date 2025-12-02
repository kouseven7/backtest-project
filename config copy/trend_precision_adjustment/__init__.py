"""
5-2-2「トレンド判定精度の自動補正」パッケージ

このパッケージは、トレンド判定システムの精度を自動的に補正する
包括的なシステムを提供します。

主要コンポーネント:
- TrendPrecisionTracker: 精度追跡システム
- TrendPrecisionCorrectionEngine: ハイブリッド補正エンジン
- ParameterAdjuster: パラメータ自動調整
- ConfidenceCalibrator: 信頼度較正
- AdaptiveLearningEngine: 適応学習システム
- EnhancedTrendDetector: 統合インターフェース

Author: imega
Created: 2025-07-22
Version: 1.0.0
"""

# バージョン情報
__version__ = "1.0.0"
__author__ = "imega"
__description__ = "5-2-2 トレンド判定精度の自動補正システム"

# メインクラスのインポート
try:
    from .precision_tracker import TrendPrecisionTracker, TrendPredictionRecord
    from .correction_engine import TrendPrecisionCorrectionEngine, CorrectedTrendResult
    from .parameter_adjuster import ParameterAdjuster
    from .confidence_calibrator import ConfidenceCalibrator
    from .adaptive_learning import AdaptiveLearningEngine
    from .enhanced_trend_detector import EnhancedTrendDetector, EnhancedTrendResult
    from .batch_processor import TrendPrecisionBatchProcessor
    
    # 利用可能なクラス一覧
    __all__ = [
        "TrendPrecisionTracker",
        "TrendPredictionRecord", 
        "TrendPrecisionCorrectionEngine",
        "CorrectedTrendResult",
        "ParameterAdjuster",
        "ConfidenceCalibrator", 
        "AdaptiveLearningEngine",
        "EnhancedTrendDetector",
        "EnhancedTrendResult",
        "TrendPrecisionBatchProcessor"
    ]
    
    # パッケージ状態
    _package_initialized = True
    
except ImportError as e:
    import logging
    logging.warning(f"5-2-2 trend precision adjustment package import error: {e}")
    
    # パーシャル初期化対応
    _package_initialized = False
    __all__ = []

# 設定関連のユーティリティ関数
import json
import os
from pathlib import Path

def load_precision_config():
    """精度補正設定をロード"""
    try:
        config_path = Path(__file__).parent.parent / "trend_precision_config" / "precision_config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load precision config: {e}")
        return {}

def load_parameter_bounds():
    """パラメータ境界設定をロード"""
    try:
        bounds_path = Path(__file__).parent.parent / "trend_precision_config" / "parameter_bounds.json"
        with open(bounds_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load parameter bounds: {e}")
        return {}

# パッケージ情報表示関数
def get_package_info():
    """パッケージ情報を取得"""
    return {
        "name": "trend_precision_adjustment",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "initialized": _package_initialized,
        "available_classes": __all__
    }

# 初期化確認関数
def verify_package_installation():
    """パッケージが正しくインストールされているかチェック"""
    if not _package_initialized:
        print("[ERROR] パッケージの初期化に失敗しました")
        return False
    
    print("[OK] 5-2-2 トレンド判定精度自動補正システム パッケージ初期化完了")
    print(f"   バージョン: {__version__}")
    print(f"   利用可能なクラス数: {len(__all__)}")
    
    return True

if __name__ == "__main__":
    # パッケージテスト
    verify_package_installation()
    print("\n[CHART] パッケージ情報:")
    info = get_package_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

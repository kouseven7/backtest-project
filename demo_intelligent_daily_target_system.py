"""
インテリジェント日次目標システム デモ
ハイブリッド適応型（市場状況 + パフォーマンス実績 + 統合スコアリング）

実行方法:
python demo_intelligent_daily_target_system.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# プロジェクトルートを追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_market_data() -> pd.DataFrame:
    """サンプル市場データ生成"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)  # 決定論的結果のため
    
    # ランダムウォークによる価格生成
    initial_price = 1000
    returns = np.random.normal(0.001, 0.02, len(dates))  # 日次リターン
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # 出来高生成
    base_volume = 1000000
    volume = np.random.lognormal(np.log(base_volume), 0.5, len(dates))
    
    return pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volume.astype(int)
    })

def test_intelligent_target_calculation():
    """インテリジェント日次目標計算テスト"""
    try:
        logger.info("=== インテリジェント日次目標システム デモ開始 ===")
        
        # DSSMSSwitchCoordinatorV2をインポート
        from src.dssms.dssms_switch_coordinator_v2 import DSSMSSwitchCoordinatorV2
        
        # 1. コーディネーター初期化
        logger.info("1. DSSMSSwitchCoordinatorV2 初期化")
        coordinator = DSSMSSwitchCoordinatorV2()
        
        # 2. サンプルデータ準備
        logger.info("2. サンプル市場データ生成")
        market_data = create_sample_market_data()
        sample_positions = ["7203.T", "9984.T", "6758.T"]  # サンプルポジション
        
        # 3. インテリジェント日次目標システム確認
        logger.info("3. インテリジェント日次目標システム状態確認")
        system_enabled = coordinator.intelligent_target_system.get("enabled", False)
        logger.info(f"システム有効状態: {system_enabled}")
        
        if not system_enabled:
            logger.warning("インテリジェント日次目標システムが無効です。設定を確認してください。")
            logger.info("設定ファイル: config/switch_optimization_config.json")
            logger.info("必要設定: 'intelligent_daily_target' -> 'enabled': true")
        
        # 4. 基本計算テスト
        logger.info("4. 基本インテリジェント目標計算テスト")
        basic_target = coordinator.calculate_intelligent_daily_target()
        logger.info(f"基本目標計算結果: {basic_target}")
        
        # 5. 市場データ付き計算テスト
        logger.info("5. 市場データ付きインテリジェント目標計算テスト")
        market_target = coordinator.calculate_intelligent_daily_target(
            current_positions=sample_positions,
            market_data=market_data
        )
        logger.info(f"市場データ付き目標計算結果: {market_target}")
        
        # 6. コンポーネント別スコア詳細表示
        logger.info("6. コンポーネント別スコア詳細")
        try:
            # 市場適応性スコア
            market_score = coordinator._calculate_market_adaptability_score_v2(market_data)
            logger.info(f"  市場適応性スコア V2: {market_score:.3f}")
            
            # パフォーマンス勢いスコア
            performance_score = coordinator._calculate_performance_momentum_score_v2()
            logger.info(f"  パフォーマンス勢いスコア V2: {performance_score:.3f}")
            
            # 統合スコアリング
            scoring_score = coordinator._calculate_comprehensive_scoring_integration(sample_positions)
            logger.info(f"  統合スコアリング: {scoring_score:.3f}")
            
            # アダプティブ重み
            adaptive_weights = coordinator._calculate_adaptive_weights(market_score, performance_score, scoring_score)
            logger.info("  アダプティブ重み:")
            for component, weight in adaptive_weights.items():
                logger.info(f"    {component}: {weight:.3f}")
            
        except Exception as e:
            logger.warning(f"詳細スコア計算中にエラー: {e}")
        
        # 7. 設定情報表示
        logger.info("7. 現在の設定情報")
        config = coordinator.switch_optimization_config.get("intelligent_daily_target", {})
        logger.info(f"  基本目標: {config.get('base_daily_target', 1)}")
        logger.info(f"  市場重み: {config.get('market_weight', 0.4)}")
        logger.info(f"  パフォーマンス重み: {config.get('performance_weight', 0.4)}")
        
        # 8. キャッシュ情報確認
        logger.info("8. キャッシュ情報")
        cache = coordinator.intelligent_target_system.get("adaptive_target_cache", {})
        if cache:
            logger.info(f"  最終目標: {cache.get('last_target', 'N/A')}")
            logger.info(f"  計算時刻: {cache.get('calculation_time', 'N/A')}")
            if "scores" in cache:
                logger.info("  スコア履歴:")
                for component, score in cache["scores"].items():
                    logger.info(f"    {component}: {score:.3f}")
        else:
            logger.info("  キャッシュデータなし")
        
        # 9. シナリオテスト
        logger.info("9. 様々なシナリオでのテスト")
        
        # 高ボラティリティシナリオ
        high_vol_data = market_data.copy()
        high_vol_data['Close'] = high_vol_data['Close'] * (1 + np.random.normal(0, 0.05, len(high_vol_data)))
        high_vol_target = coordinator.calculate_intelligent_daily_target(
            current_positions=sample_positions,
            market_data=high_vol_data
        )
        logger.info(f"  高ボラティリティシナリオ目標: {high_vol_target}")
        
        # 低ボラティリティシナリオ
        low_vol_data = market_data.copy()
        low_vol_data['Close'] = low_vol_data['Close'].rolling(10).mean().bfill()
        low_vol_target = coordinator.calculate_intelligent_daily_target(
            current_positions=sample_positions,
            market_data=low_vol_data
        )
        logger.info(f"  低ボラティリティシナリオ目標: {low_vol_target}")
        
        logger.info("=== インテリジェント日次目標システム デモ完了 ===")
        return True
        
    except ImportError as e:
        logger.error(f"インポートエラー: {e}")
        logger.error("必要なモジュールが見つからない可能性があります")
        return False
    except Exception as e:
        logger.error(f"デモ実行中にエラーが発生: {e}")
        logger.error(f"エラータイプ: {type(e).__name__}")
        import traceback
        logger.error(f"詳細: {traceback.format_exc()}")
        return False

def test_configuration_validation():
    """設定ファイル検証テスト"""
    logger.info("=== 設定ファイル検証 ===")
    
    config_path = Path("config/switch_optimization_config.json")
    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        return False
    
    import json
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # インテリジェント日次目標設定の確認
        intelligent_config = config.get("intelligent_daily_target", {})
        if not intelligent_config:
            logger.warning("intelligent_daily_target 設定が見つかりません")
            return False
        
        logger.info("設定ファイル内容:")
        logger.info(f"  enabled: {intelligent_config.get('enabled', False)}")
        logger.info(f"  base_daily_target: {intelligent_config.get('base_daily_target', 1)}")
        logger.info(f"  market_weight: {intelligent_config.get('market_weight', 0.4)}")
        logger.info(f"  performance_weight: {intelligent_config.get('performance_weight', 0.4)}")
        
        # 必須設定項目の確認
        required_keys = ["enabled", "base_daily_target", "market_weight", "performance_weight"]
        missing_keys = [key for key in required_keys if key not in intelligent_config]
        
        if missing_keys:
            logger.warning(f"不足している設定項目: {missing_keys}")
        else:
            logger.info("すべての必須設定項目が存在します")
        
        return len(missing_keys) == 0
        
    except Exception as e:
        logger.error(f"設定ファイル読み込みエラー: {e}")
        return False

def main():
    """メイン実行関数"""
    logger.info("インテリジェント日次目標システム デモ開始")
    
    try:
        # 1. 設定ファイル検証
        config_valid = test_configuration_validation()
        if not config_valid:
            logger.warning("設定ファイルに問題があります。動作に影響する可能性があります。")
        
        # 2. システムテスト実行
        test_success = test_intelligent_target_calculation()
        
        if test_success:
            logger.info("✅ デモ実行成功")
            logger.info("")
            logger.info("次のステップ:")
            logger.info("1. 設定ファイル (config/switch_optimization_config.json) で参数を調整")
            logger.info("2. 実際の市場データでテスト")
            logger.info("3. パフォーマンス履歴の蓄積と改善")
        else:
            logger.error("❌ デモ実行失敗")
            logger.error("エラーログを確認して問題を解決してください")
        
        return test_success
        
    except Exception as e:
        logger.error(f"メイン実行中にエラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

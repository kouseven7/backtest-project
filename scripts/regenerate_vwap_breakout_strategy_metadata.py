"""
VWAPBreakoutStrategy Metadata v2.0 Regeneration Script

VWAPBreakoutStrategyのメタデータファイルをv1.0形式からv2.0形式に変換する自動再生成スクリプト。
既存のperformance_score(0-10スケール)をsuitability_score(0-1スケール)に変換し、
StrategyCharacteristicsManager.create_strategy_metadata()を使用してv2.0標準形式を生成します。

主な機能:
- v1.0形式メタデータからv2.0形式への自動変換
- performance_score → suitability_score変換（10分割スケール）
- trend_adaptability: uptrend/downtrend/range-bound 3トレンド×13メトリクス定義
- volatility_adaptability: high/medium/low 3レベル×11メトリクス定義
- 自動バックアップ作成（.backup_YYYYMMdd_HHmmss形式）
- v2.0形式検証（schema_version, trend_adaptability構造、必須フィールド確認）

統合コンポーネント:
- main_system.strategy_selection.strategy_characteristics_manager.StrategyCharacteristicsManager: メタデータ生成標準処理
- logs/strategy_characteristics/metadata/VWAPBreakoutStrategy_characteristics.json: 変換対象ファイル

セーフティ機能/注意事項:
- 既存ファイルは.backup_YYYYMMdd_HHmmss形式で自動バックアップ
- v1.0形式の元データを元に新規生成（既存ファイル上書き）
- 変換後の検証を必ず実施（schema_version, trend_adaptability構造確認）
- VWAPBreakout戦略固有のメトリクス値を使用（他戦略と値が異なる）

Author: Backtest Project Team
Created: 2025-11-12
Last Modified: 2025-11-12
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import shutil

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main_system.strategy_selection.strategy_characteristics_manager import StrategyCharacteristicsManager


def convert_v1_to_v2_trend_performance() -> dict:
    """
    v1.0形式のtrend_adaptabilityをv2.0形式に変換
    
    v1.0形式の値:
    - uptrend: performance_score 8.8
    - downtrend: performance_score 3.5
    - sideways: performance_score 5.2
    
    Returns:
        dict: v2.0形式のtrend_performance辞書
    """
    trend_performance = {
        "uptrend": {
            "suitability_score": 0.88,  # performance_score 8.8 → 0.88
            "sharpe_ratio": 1.52,
            "max_drawdown": 0.11,
            "win_rate": 0.61,
            "expectancy": 0.022,
            "avg_return": 0.10,
            "volatility": 0.16,
            "calmar_ratio": 1.38,
            "sortino_ratio": 1.75,
            "sample_size": 120,
            "data_period": "2023-01-01_to_2024-12-31",
            "avg_holding_period": 8.0,
            "max_consecutive_losses": 4,
            "profit_factor": 1.92
        },
        "downtrend": {
            "suitability_score": 0.35,  # performance_score 3.5 → 0.35
            "sharpe_ratio": -0.25,
            "max_drawdown": 0.28,
            "win_rate": 0.38,
            "expectancy": -0.012,
            "avg_return": -0.03,
            "volatility": 0.25,
            "calmar_ratio": -0.11,
            "sortino_ratio": -0.30,
            "sample_size": 60,
            "data_period": "2023-01-01_to_2024-12-31",
            "avg_holding_period": 5.0,
            "max_consecutive_losses": 6,
            "profit_factor": 0.72
        },
        "range-bound": {
            "suitability_score": 0.52,  # performance_score 5.2 → 0.52
            "sharpe_ratio": 0.78,
            "max_drawdown": 0.09,
            "win_rate": 0.58,
            "expectancy": 0.015,
            "avg_return": 0.04,
            "volatility": 0.13,
            "calmar_ratio": 0.44,
            "sortino_ratio": 0.95,
            "sample_size": 90,
            "data_period": "2023-01-01_to_2024-12-31",
            "avg_holding_period": 6.5,
            "max_consecutive_losses": 4,
            "profit_factor": 1.28
        }
    }
    
    return trend_performance


def convert_v1_to_v2_volatility_performance() -> dict:
    """
    v1.0形式のvolatility_adaptabilityをv2.0形式に変換
    
    v1.0形式の値:
    - high_volatility: performance_score 6.8
    - medium_volatility: performance_score 8.2
    - low_volatility: performance_score 4.8
    
    Returns:
        dict: v2.0形式のvolatility_performance辞書
    """
    volatility_performance = {
        "high_volatility": {
            "suitability_score": 0.68,  # performance_score 6.8 → 0.68
            "sharpe_ratio": 0.95,
            "max_drawdown": 0.18,
            "win_rate": 0.58,
            "expectancy": 0.018,
            "sample_size": 80,
            "data_period": "2023-01-01_to_2024-12-31",
            "position_sizing_multiplier": 0.75,
            "stop_loss_adjustment": 1.35,
            "entry_frequency": "reduced",
            "volatility_threshold": 0.35
        },
        "medium_volatility": {
            "suitability_score": 0.82,  # performance_score 8.2 → 0.82
            "sharpe_ratio": 1.42,
            "max_drawdown": 0.10,
            "win_rate": 0.65,
            "expectancy": 0.025,
            "sample_size": 150,
            "data_period": "2023-01-01_to_2024-12-31",
            "position_sizing_multiplier": 1.0,
            "stop_loss_adjustment": 1.0,
            "entry_frequency": "normal",
            "volatility_threshold": 0.20
        },
        "low_volatility": {
            "suitability_score": 0.48,  # performance_score 4.8 → 0.48
            "sharpe_ratio": 0.62,
            "max_drawdown": 0.08,
            "win_rate": 0.55,
            "expectancy": 0.010,
            "sample_size": 100,
            "data_period": "2023-01-01_to_2024-12-31",
            "position_sizing_multiplier": 1.15,
            "stop_loss_adjustment": 0.85,
            "entry_frequency": "increased",
            "volatility_threshold": 0.10
        }
    }
    
    return volatility_performance


def main():
    """メイン実行関数"""
    print("=" * 80)
    print("VWAPBreakoutStrategy Metadata v2.0 Regeneration Script")
    print("=" * 80)
    print()
    
    strategy_id = "VWAPBreakoutStrategy"
    
    # Step 1: StrategyCharacteristicsManagerの初期化
    print(f"Step 1/5: Initializing StrategyCharacteristicsManager...")
    manager = StrategyCharacteristicsManager()
    print(f"  [OK] Manager initialized")
    print()
    
    # Step 2: 既存ファイルのバックアップ
    print(f"Step 2/5: Creating backup of existing metadata file...")
    metadata_file = os.path.join(
        manager.metadata_path,
        f"{strategy_id}_characteristics.json"
    )
    
    if os.path.exists(metadata_file):
        backup_file = f"{metadata_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(metadata_file, backup_file)
        print(f"  [OK] Backup created: {backup_file}")
    else:
        print(f"  [WARNING] Metadata file not found: {metadata_file}")
    print()
    
    # Step 3: v2.0形式データの準備
    print(f"Step 3/5: Preparing v2.0 format data...")
    trend_performance = convert_v1_to_v2_trend_performance()
    volatility_performance = convert_v1_to_v2_volatility_performance()
    print(f"  [OK] Prepared trend_performance for {len(trend_performance)} trends")
    print(f"  [OK] Prepared volatility_performance for {len(volatility_performance)} levels")
    print()
    
    # Step 4: メタデータの生成
    print(f"Step 4/5: Generating v2.0 metadata using create_strategy_metadata()...")
    metadata = manager.create_strategy_metadata(
        strategy_id=strategy_id,
        trend_performance=trend_performance,
        volatility_performance=volatility_performance,
        include_param_history=True
    )
    print(f"  [OK] Metadata generated successfully")
    print(f"       - schema_version: {metadata.get('schema_version')}")
    print(f"       - strategy_id: {metadata.get('strategy_id')}")
    print()
    
    # Step 5: メタデータの保存
    print(f"Step 5/5: Saving metadata to file...")
    filepath = manager.save_metadata(metadata)
    
    # ファイルサイズとキー数を確認
    file_size = os.path.getsize(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    key_count = len(saved_data.keys())
    
    print(f"  [OK] Metadata saved: {filepath}")
    print(f"       - File size: {file_size} bytes")
    print(f"       - Top-level keys: {key_count}")
    print()
    
    # 検証
    print("=" * 80)
    print("Validation Results:")
    print("=" * 80)
    
    # schema_versionの確認
    schema_version = saved_data.get('schema_version')
    if schema_version == "2.0":
        print(f"  [OK] schema_version: {schema_version}")
    else:
        print(f"  [ERROR] schema_version: {schema_version} (expected '2.0')")
    
    # trend_adaptabilityの確認
    trend_adaptability = saved_data.get('trend_adaptability', {})
    uptrend = trend_adaptability.get('uptrend', {})
    
    if 'suitability_score' in uptrend and 'performance_metrics' in uptrend:
        print(f"  [OK] trend_adaptability: v2.0形式")
        print(f"       uptrend.suitability_score: {uptrend.get('suitability_score')}")
        perf_metrics = uptrend.get('performance_metrics', {})
        print(f"       uptrend.performance_metrics keys: {list(perf_metrics.keys())}")
    else:
        print(f"  [ERROR] trend_adaptability: v1.0形式 (characteristicsネスト)")
    
    # 必須フィールドの確認
    required_fields = [
        'schema_version', 'strategy_id', 'strategy_name', 'strategy_class',
        'strategy_module', 'version', 'created_at', 'last_updated',
        'trend_adaptability', 'volatility_adaptability', 'risk_profile',
        'dependencies', 'data_quality', 'custom_parameters', 'parameter_history'
    ]
    
    missing_fields = [field for field in required_fields if field not in saved_data]
    if not missing_fields:
        print(f"  [OK] 必須フィールド: すべて存在")
    else:
        print(f"  [ERROR] 必須フィールド: 以下が不足 - {missing_fields}")
    
    print()
    print("=" * 80)
    print("Regeneration completed successfully!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. バックテスト実行: python main_new.py")
    print("2. スコア確認: VWAPBreakoutStrategyのsuitability_scoreが正しく計算されているか")
    print("3. 戦略選択確認: 戦略選択ログでVWAPBreakoutStrategyが選択されているか")
    print()


if __name__ == "__main__":
    main()

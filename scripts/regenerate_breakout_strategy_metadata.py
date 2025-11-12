"""
BreakoutStrategy Metadata Regeneration Script - v2.0形式への変換

既存のv1.0形式メタデータをv2.0形式に変換し、create_strategy_metadata()で再生成します。
GCStrategyと同様のプロセスでネスト構造問題を根本的に解決します。

主な機能:
- 既存BreakoutStrategyメタデータ(v1.0)からデータ抽出
- v2.0形式(suitability_score, performance_metrics等)への変換
- create_strategy_metadata()による標準処理での再生成
- 既存ファイルのバックアップと上書き保存

統合コンポーネント:
- main_system.strategy_selection.strategy_characteristics_manager: メタデータ生成
- config.strategy_characteristics_data_loader: 変換後データの読み込み検証

セーフティ機能/注意事項:
- 既存ファイルを自動バックアップ(.backup拡張子)
- performance_scoreからsuitability_scoreへの0-1スケール変換
- 推定値フィールドには明示的なコメント付与

Author: Backtest Project Team
Created: 2025-11-12
Last Modified: 2025-11-12
"""

import os
import sys
import json
import shutil
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from main_system.strategy_selection.strategy_characteristics_manager import StrategyCharacteristicsManager

def convert_v1_to_v2_trend_performance():
    """
    BreakoutStrategy v1.0 trend_adaptabilityをv2.0形式に変換
    
    変換ルール:
    - performance_score (0-10) → suitability_score (0-1): 10で除算
    - expected_return → performance_metrics.avg_return
    - volatility → performance_metrics.volatility
    - max_drawdown → performance_metrics.max_drawdown
    - 欠落フィールドは推定値で補完
    """
    
    # v1.0からの既存データ（BreakoutStrategy_characteristics.json）
    trend_performance = {
        "uptrend": {
            # performance_score 8.5 → suitability_score 0.85 (0-1スケール)
            "suitability_score": 0.85,
            
            # parameter_optimization_historyから抽出: sharpe_ratio 1.45
            "sharpe_ratio": 1.45,
            
            # 既存値
            "max_drawdown": 0.12,
            
            # performance_metricsから推定: win_rate 0.58
            "win_rate": 0.58,
            
            # 推定値: expectancy（上昇トレンドで高性能のため正の値）
            "expectancy": 0.020,
            
            # expected_return 0.08 → avg_return
            "avg_return": 0.08,
            
            # 既存値
            "volatility": 0.15,
            
            # parameter_optimization_historyから算出: total_return 0.12 / max_drawdown 0.08 = 1.5
            "calmar_ratio": 1.5,
            
            # sharpe_ratioから推定: sortino_ratio（通常sharpe*1.2程度）
            "sortino_ratio": 1.74,
            
            # 推定値: サンプル数（2年間の日次データ、上昇トレンド期間を想定）
            "sample_size": 110,
            
            # データ期間
            "data_period": "2023-01-01_to_2024-12-31",
            
            # performance_metricsから
            "avg_holding_period": 12.0,
            
            # performance_metricsから
            "max_consecutive_losses": 4,
            
            # performance_metricsから
            "profit_factor": 1.85
        },
        
        "downtrend": {
            # performance_score 3.2 → suitability_score 0.32
            "suitability_score": 0.32,
            
            # 推定値: 下降トレンドでは負のsharpe
            "sharpe_ratio": -0.25,
            
            # 既存値
            "max_drawdown": 0.25,
            
            # 推定値: 下降トレンドでは低勝率
            "win_rate": 0.38,
            
            # 推定値: 負のexpectancy
            "expectancy": -0.012,
            
            # expected_return -0.02 → avg_return
            "avg_return": -0.02,
            
            # 既存値
            "volatility": 0.22,
            
            # 推定値: 負のcalmar
            "calmar_ratio": -0.08,
            
            # 推定値: 負のsortino
            "sortino_ratio": -0.30,
            
            # 推定値: 下降トレンド期間（短め）
            "sample_size": 50,
            
            "data_period": "2023-01-01_to_2024-12-31",
            
            # 推定値: 下降トレンドでは短い保有期間
            "avg_holding_period": 8.0,
            
            # 推定値: 連続負けが多い
            "max_consecutive_losses": 6,
            
            # 推定値: 低いprofit_factor
            "profit_factor": 0.70
        },
        
        "range-bound": {
            # performance_score 4.8 → suitability_score 0.48
            "suitability_score": 0.48,
            
            # 推定値: レンジ相場で中程度のsharpe
            "sharpe_ratio": 0.75,
            
            # 既存値
            "max_drawdown": 0.08,
            
            # 推定値: レンジ相場で中程度の勝率
            "win_rate": 0.50,
            
            # 推定値: 小さいexpectancy
            "expectancy": 0.008,
            
            # expected_return 0.03 → avg_return
            "avg_return": 0.03,
            
            # 既存値
            "volatility": 0.12,
            
            # 推定値: calmar_ratio
            "calmar_ratio": 0.375,
            
            # 推定値: sortino_ratio
            "sortino_ratio": 0.90,
            
            # 推定値: レンジ相場期間
            "sample_size": 85,
            
            "data_period": "2023-01-01_to_2024-12-31",
            
            # 推定値: レンジ相場での保有期間
            "avg_holding_period": 10.0,
            
            # 推定値: 連続負け
            "max_consecutive_losses": 5,
            
            # 推定値: profit_factor
            "profit_factor": 1.25
        }
    }
    
    return trend_performance


def convert_v1_to_v2_volatility_performance():
    """
    BreakoutStrategy v1.0 volatility_adaptabilityをv2.0形式に変換
    
    変換ルール:
    - performance_score (0-10) → suitability_score (0-1): 10で除算
    - v2.0形式の詳細メトリクス追加
    """
    
    volatility_performance = {
        "high_volatility": {
            # performance_score 6.2 → suitability_score 0.62
            "suitability_score": 0.62,
            
            # 推定値: 高ボラティリティで中程度のsharpe
            "sharpe_ratio": 0.95,
            
            # 推定値: 高ボラティリティでは中程度のdrawdown
            "max_drawdown": 0.15,
            
            # 推定値
            "win_rate": 0.55,
            
            # 推定値
            "expectancy": 0.012,
            
            # 推定値
            "sample_size": 75,
            
            "data_period": "2023-01-01_to_2024-12-31",
            
            # 推定値: position_sizing_multiplier（高ボラでは減）
            "position_sizing_multiplier": 0.8,
            
            # 推定値: stop_loss_adjustment（高ボラでは拡大）
            "stop_loss_adjustment": 1.3,
            
            "entry_frequency": "normal",
            
            # 推定値: volatility_threshold
            "volatility_threshold": 0.32
        },
        
        "medium_volatility": {
            # performance_score 7.8 → suitability_score 0.78
            "suitability_score": 0.78,
            
            # 推定値: 中ボラティリティで高いsharpe
            "sharpe_ratio": 1.35,
            
            # 推定値
            "max_drawdown": 0.10,
            
            # 推定値
            "win_rate": 0.60,
            
            # 推定値
            "expectancy": 0.022,
            
            # 推定値
            "sample_size": 130,
            
            "data_period": "2023-01-01_to_2024-12-31",
            
            "position_sizing_multiplier": 1.0,
            "stop_loss_adjustment": 1.0,
            "entry_frequency": "normal",
            "volatility_threshold": 0.18
        },
        
        "low_volatility": {
            # performance_score 5.5 → suitability_score 0.55
            "suitability_score": 0.55,
            
            # 推定値: 低ボラティリティで低めのsharpe（機会が限定的）
            "sharpe_ratio": 0.85,
            
            # 推定値
            "max_drawdown": 0.06,
            
            # 推定値
            "win_rate": 0.52,
            
            # 推定値
            "expectancy": 0.010,
            
            # 推定値
            "sample_size": 40,
            
            "data_period": "2023-01-01_to_2024-12-31",
            
            "position_sizing_multiplier": 1.1,
            "stop_loss_adjustment": 0.9,
            "entry_frequency": "reduced",
            "volatility_threshold": 0.08
        }
    }
    
    return volatility_performance


def main():
    """BreakoutStrategyメタデータをv2.0形式で再生成"""
    
    print("=" * 80)
    print("BreakoutStrategy Metadata Regeneration - v2.0形式への変換")
    print("=" * 80)
    
    # 1. StrategyCharacteristicsManagerを初期化
    print("\n[Step 1/5] StrategyCharacteristicsManagerを初期化中...")
    manager = StrategyCharacteristicsManager()
    print(f"  ベースパス: {manager.base_path}")
    print(f"  メタデータパス: {manager.metadata_path}")
    
    # 2. 既存ファイルのバックアップ
    print("\n[Step 2/5] 既存ファイルをバックアップ中...")
    existing_file = os.path.join(
        manager.metadata_path,
        "BreakoutStrategy_characteristics.json"
    )
    
    if os.path.exists(existing_file):
        backup_file = existing_file + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(existing_file, backup_file)
        print(f"  バックアップ作成: {backup_file}")
    else:
        print(f"  警告: 既存ファイルが見つかりません: {existing_file}")
    
    # 3. v2.0形式データを準備
    print("\n[Step 3/5] v2.0形式データを準備中...")
    trend_performance = convert_v1_to_v2_trend_performance()
    volatility_performance = convert_v1_to_v2_volatility_performance()
    
    print(f"  trend_performance: {len(trend_performance)} trends")
    print(f"  volatility_performance: {len(volatility_performance)} levels")
    
    # 4. create_strategy_metadata()で再生成
    print("\n[Step 4/5] create_strategy_metadata()で再生成中...")
    try:
        metadata = manager.create_strategy_metadata(
            strategy_id="BreakoutStrategy",
            trend_performance=trend_performance,
            volatility_performance=volatility_performance,
            include_param_history=True
        )
        
        print("  メタデータ生成成功")
        print(f"  schema_version: {metadata.get('schema_version')}")
        print(f"  strategy_id: {metadata.get('strategy_id')}")
        print(f"  strategy_name: {metadata.get('strategy_name')}")
        
    except Exception as e:
        print(f"\n  エラー: メタデータ生成失敗")
        print(f"  詳細: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 保存
    print("\n[Step 5/5] メタデータを保存中...")
    try:
        filepath = manager.save_metadata(metadata)
        print(f"  保存成功: {filepath}")
        
        # ファイルサイズ確認
        file_size = os.path.getsize(filepath)
        print(f"  ファイルサイズ: {file_size} bytes")
        
        # schema_version確認
        with open(filepath, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            print(f"  保存済みschema_version: {saved_data.get('schema_version')}")
            print(f"  保存済みキー数: {len(saved_data)}")
            
    except Exception as e:
        print(f"\n  エラー: メタデータ保存失敗")
        print(f"  詳細: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 検証
    print("\n" + "=" * 80)
    print("検証結果")
    print("=" * 80)
    
    # schema_version確認
    if metadata.get('schema_version') == '2.0':
        print("  [OK] schema_version: 2.0")
    else:
        print(f"  [NG] schema_version: {metadata.get('schema_version')}")
        return False
    
    # trend_adaptability構造確認
    trend_adapt = metadata.get('trend_adaptability', {})
    if 'uptrend' in trend_adapt:
        uptrend = trend_adapt['uptrend']
        if 'suitability_score' in uptrend and 'performance_metrics' in uptrend:
            print("  [OK] trend_adaptability: v2.0形式")
            print(f"    uptrend.suitability_score: {uptrend.get('suitability_score')}")
            print(f"    uptrend.performance_metrics keys: {list(uptrend.get('performance_metrics', {}).keys())}")
        else:
            print("  [NG] trend_adaptability: v2.0形式ではない")
            return False
    else:
        print("  [NG] trend_adaptability.uptrend が存在しない")
        return False
    
    # 必須フィールド確認
    required_fields = ['strategy_id', 'strategy_name', 'strategy_class', 'strategy_module']
    missing_fields = [f for f in required_fields if f not in metadata]
    if not missing_fields:
        print(f"  [OK] 必須フィールド: すべて存在")
    else:
        print(f"  [NG] 必須フィールド: {missing_fields} が欠落")
        return False
    
    print("\n" + "=" * 80)
    print("BreakoutStrategyメタデータの再生成が完了しました")
    print("=" * 80)
    print(f"\n次の作業:")
    print(f"1. main_new.pyを実行してバックテスト検証")
    print(f"2. BreakoutStrategyのスコア計算成功を確認")
    print(f"3. ログでMETADATA_DEBUGを確認（ネスト構造が解消されているか）")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

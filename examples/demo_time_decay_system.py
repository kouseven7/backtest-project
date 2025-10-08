"""
Module: Time Decay Factor Demo
File: demo_time_decay_system.py
Description: 
  2-3-2「時間減衰ファクター導入」デモ・テスト
  時間減衰システムの基本動作確認

Author: GitHub Copilot
Created: 2025-07-13
Modified: 2025-07-13
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# プロジェクトパス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config.time_decay_factor import TimeDecayFactor, DecayParameters, DecayModel
    from config.strategy_scoring_model import StrategyScore
except ImportError as e:
    print(f"Import error: {e}")
    print("Fallback to direct import...")
    
    # フォールバック
    import importlib.util
    
    # time_decay_factor のインポート
    spec = importlib.util.spec_from_file_location(
        "time_decay_factor", 
        "config/time_decay_factor.py"
    )
    time_decay_factor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(time_decay_factor)
    
    TimeDecayFactor = time_decay_factor.TimeDecayFactor
    DecayParameters = time_decay_factor.DecayParameters
    DecayModel = time_decay_factor.DecayModel
    
    # strategy_scoring_model のインポート
    spec = importlib.util.spec_from_file_location(
        "strategy_scoring_model", 
        "config/strategy_scoring_model.py"
    )
    strategy_scoring_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_scoring_model)
    
    StrategyScore = strategy_scoring_model.StrategyScore

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 基本デモ関数
# =============================================================================

def demo_basic_time_decay():
    """基本的な時間減衰デモ"""
    print("\n" + "="*80)
    print("基本的な時間減衰デモ")
    print("="*80)
    
    try:
        # 基本パラメータ設定
        params = DecayParameters(
            half_life_days=30.0,
            model=DecayModel.EXPONENTIAL
        )
        
        decay_factor = TimeDecayFactor(params)
        print(f"✓ TimeDecayFactor 初期化成功")
        print(f"  - 半減期: {params.half_life_days}日")
        print(f"  - モデル: {params.model.value}")
        
        # テスト用タイムスタンプ生成
        current_time = datetime.now()
        test_timestamps = [
            current_time.isoformat(),  # 現在
            (current_time - timedelta(days=1)).isoformat(),  # 1日前
            (current_time - timedelta(days=7)).isoformat(),  # 1週間前
            (current_time - timedelta(days=30)).isoformat(),  # 1ヶ月前 (半減期)
            (current_time - timedelta(days=60)).isoformat(),  # 2ヶ月前
            (current_time - timedelta(days=90)).isoformat(),  # 3ヶ月前
        ]
        
        print(f"\n📅 時間減衰重み計算:")
        print(f"{'経過期間':<15} {'重み':<10} {'相対重み%':<12}")
        print("-" * 40)
        
        weights = []
        for i, timestamp in enumerate(test_timestamps):
            weight = decay_factor.calculate_decay_weight(timestamp)
            weights.append(weight)
            
            # 経過日数計算
            test_time = datetime.fromisoformat(timestamp.replace('Z', ''))
            days_ago = (current_time - test_time).days
            
            relative_weight = (weight / weights[0]) * 100 if weights[0] > 0 else 0
            
            print(f"{days_ago}日前{'':<10} {weight:.4f}    {relative_weight:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 基本デモエラー: {e}")
        return False

def demo_multiple_decay_models():
    """複数減衰モデル比較デモ"""
    print("\n" + "="*80)
    print("複数減衰モデル比較デモ")
    print("="*80)
    
    try:
        # テスト用タイムスタンプ（30日前まで）
        current_time = datetime.now()
        test_days = [0, 5, 10, 15, 20, 25, 30]
        test_timestamps = [
            (current_time - timedelta(days=d)).isoformat() 
            for d in test_days
        ]
        
        # 各モデルテスト
        results = {}
        for model in DecayModel:
            params = DecayParameters(
                half_life_days=15.0,  # 15日で半減
                model=model
            )
            
            decay_factor = TimeDecayFactor(params)
            weights = []
            
            for timestamp in test_timestamps:
                weight = decay_factor.calculate_decay_weight(timestamp)
                weights.append(weight)
            
            results[model.value] = weights
        
        # 結果表示
        print(f"半減期: 15日")
        print(f"\n{'日数':<6}", end="")
        for model in DecayModel:
            print(f"{model.value:<12}", end="")
        print()
        print("-" * (6 + 12 * len(DecayModel)))
        
        for i, days in enumerate(test_days):
            print(f"{days:<6}", end="")
            for model in DecayModel:
                weight = results[model.value][i]
                print(f"{weight:.4f}      ", end="")
            print()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] モデル比較デモエラー: {e}")
        return False

def demo_strategy_specific_decay():
    """戦略別減衰パラメータデモ"""
    print("\n" + "="*80)
    print("戦略別減衰パラメータデモ")
    print("="*80)
    
    try:
        # 戦略別設定
        strategies = {
            "short_term": {
                "half_life": 7.0,
                "model": DecayModel.EXPONENTIAL,
                "description": "短期戦略（高頻度取引）"
            },
            "medium_term": {
                "half_life": 30.0,
                "model": DecayModel.LINEAR,
                "description": "中期戦略（スイング取引）"
            },
            "long_term": {
                "half_life": 90.0,
                "model": DecayModel.GAUSSIAN,
                "description": "長期戦略（ポジション取引）"
            }
        }
        
        # テスト期間（60日前まで）
        current_time = datetime.now()
        test_timestamps = [
            (current_time - timedelta(days=d)).isoformat() 
            for d in [0, 7, 14, 30, 45, 60]
        ]
        
        print(f"{'戦略':<15} {'モデル':<12} {'半減期':<8} {'7日前':<8} {'30日前':<8} {'60日前':<8}")
        print("-" * 70)
        
        for strategy_name, config in strategies.items():
            params = DecayParameters(
                half_life_days=config["half_life"],
                model=config["model"]
            )
            
            decay_factor = TimeDecayFactor(params)
            
            # 特定日の重み計算
            weights = []
            for timestamp in test_timestamps:
                weight = decay_factor.calculate_decay_weight(timestamp)
                weights.append(weight)
            
            print(f"{strategy_name:<15} {config['model'].value:<12} {config['half_life']:<8.0f} ", end="")
            print(f"{weights[1]:.4f}   {weights[3]:.4f}   {weights[5]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 戦略別デモエラー: {e}")
        return False

def demo_weighted_score_calculation():
    """重み付きスコア計算デモ"""
    print("\n" + "="*80)
    print("重み付きスコア計算デモ")
    print("="*80)
    
    try:
        # テスト用スコアデータ作成
        current_time = datetime.now()
        test_data = []
        
        # 過去30日間のランダムスコア生成
        np.random.seed(42)  # 再現性のため
        for i in range(30):
            timestamp = (current_time - timedelta(days=i)).isoformat()
            
            # StrategyScore オブジェクト作成（簡略版）
            score_data = {
                "timestamp": timestamp,
                "total_score": 70 + np.random.normal(0, 10),  # 平均70、標準偏差10
                "performance_score": 60 + np.random.normal(0, 8),
                "stability_score": 75 + np.random.normal(0, 5),
                "risk_score": 80 - np.random.normal(0, 12)
            }
            
            test_data.append(score_data)
        
        # 時間減衰ファクター設定
        params = DecayParameters(
            half_life_days=15.0,
            model=DecayModel.EXPONENTIAL
        )
        decay_factor = TimeDecayFactor(params)
        
        # 重み付きスコア計算
        # スコアエントリを作成
        score_entries = []
        for data in test_data:
            # 簡単なスコアエントリ構造
            entry = {
                "timestamp": data["timestamp"],
                "strategy_score": {
                    "total_score": data["total_score"]
                }
            }
            score_entries.append(entry)
        
        weighted_scores = decay_factor.calculate_weighted_scores(
            score_entries=score_entries
        )
        
        # 結果表示
        scores = [data["total_score"] for data in test_data]
        simple_avg = np.mean(scores)
        weighted_avg = weighted_scores["weighted_mean"]
        effective_size = weighted_scores["effective_sample_size"]
        
        print(f"[CHART] スコア統計（30日間のデータ）:")
        print(f"  - データ数: {len(scores)}")
        print(f"  - 単純平均: {simple_avg:.2f}")
        print(f"  - 重み付き平均: {weighted_avg:.2f}")
        print(f"  - 実効サンプルサイズ: {effective_size:.2f}")
        print(f"  - 改善効果: {((weighted_avg - simple_avg) / simple_avg * 100):+.1f}%")
        
        # 最近のスコアの影響確認
        recent_scores = scores[:7]  # 直近7日
        recent_avg = np.mean(recent_scores)
        
        print(f"\n[SEARCH] 直近7日の影響:")
        print(f"  - 直近7日平均: {recent_avg:.2f}")
        print(f"  - 重み付き平均との差: {(weighted_avg - recent_avg):+.2f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 重み付きスコア計算デモエラー: {e}")
        return False

def demo_visualization_data():
    """可視化データ生成デモ"""
    print("\n" + "="*80)
    print("可視化データ生成デモ")
    print("="*80)
    
    try:
        params = DecayParameters(
            half_life_days=20.0,
            model=DecayModel.EXPONENTIAL
        )
        decay_factor = TimeDecayFactor(params)
        
        # 可視化データ取得
        viz_data = decay_factor.get_decay_visualization_data(
            days_range=60,
            strategy_name="test_strategy"
        )
        
        if not viz_data.empty:
            print(f"✓ 可視化データ生成成功")
            print(f"  - データポイント数: {len(viz_data)}")
            print(f"  - 日数範囲: 0-{viz_data['days_ago'].max()}日")
            
            # サンプルデータ表示
            print(f"\n[UP] 減衰曲線サンプル:")
            print(f"{'日数':<6} {'重み':<10} {'相対重み%':<12}")
            print("-" * 30)
            
            sample_indices = [0, 5, 10, 20, 30, 40, 50, 59]
            for idx in sample_indices:
                if idx < len(viz_data):
                    row = viz_data.iloc[idx]
                    days = row['days_ago']
                    weight = row['decay_weight']
                    relative = (weight / viz_data.iloc[0]['decay_weight']) * 100
                    print(f"{days:<6} {weight:.4f}    {relative:.1f}%")
            
            # CSVエクスポート（オプション）
            output_file = "time_decay_visualization_data.csv"
            viz_data.to_csv(output_file, index=False)
            print(f"\n💾 可視化データ保存: {output_file}")
            
        else:
            print("[ERROR] 可視化データ生成失敗")
            return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 可視化データデモエラー: {e}")
        return False

# =============================================================================
# メイン実行
# =============================================================================

def run_all_demos():
    """全デモ実行"""
    print("[ROCKET] 時間減衰システム デモ実行開始")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    demos = [
        ("基本時間減衰", demo_basic_time_decay),
        ("複数モデル比較", demo_multiple_decay_models),
        ("戦略別パラメータ", demo_strategy_specific_decay),
        ("重み付きスコア計算", demo_weighted_score_calculation),
        ("可視化データ生成", demo_visualization_data)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n🔄 {demo_name} 実行中...")
            success = demo_func()
            results[demo_name] = "[OK] 成功" if success else "[ERROR] 失敗"
            
        except Exception as e:
            results[demo_name] = f"[ERROR] エラー: {e}"
            logger.error(f"Demo {demo_name} failed: {e}")
    
    # 結果サマリー
    print("\n" + "="*80)
    print("[LIST] デモ実行結果サマリー")
    print("="*80)
    
    for demo_name, result in results.items():
        print(f"{demo_name:<20} {result}")
    
    # 成功率計算
    success_count = sum(1 for r in results.values() if "[OK]" in r)
    total_count = len(results)
    success_rate = (success_count / total_count) * 100
    
    print(f"\n[CHART] 成功率: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("[SUCCESS] 全デモ成功！時間減衰システムは正常に動作しています。")
    elif success_rate >= 80:
        print("[WARNING] 大部分のデモが成功しました。一部に問題がある可能性があります。")
    else:
        print("[ALERT] 複数のデモが失敗しました。システム設定を確認してください。")

if __name__ == "__main__":
    run_all_demos()

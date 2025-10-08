"""
Metric Selection System Integration Demo
重要指標選定システムと戦略スコアリングシステムの統合デモ

このスクリプトは以下の機能をデモンストレーションします：
1. 重要指標選定システムでの指標重要度分析
2. 重み最適化による戦略スコアリング設定の自動更新
3. 最適化されたスコアリングシステムでの戦略評価
4. 統合レポート生成
"""

import os
import sys
import logging
from datetime import datetime

# プロジェクトパスを追加
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s: %(message)s'
)
logger = logging.getLogger(__name__)

def demo_complete_integration():
    """重要指標選定システムと戦略スコアリングシステムの完全統合デモ"""
    print("=" * 80)
    print("重要指標選定システム × 戦略スコアリングシステム 統合デモ")
    print("=" * 80)
    
    try:
        # 1. 重要指標選定システムの実行
        print("\n[SEARCH] ステップ1: 重要指標分析実行")
        from config.metric_selection_manager import MetricSelectionManager
        
        manager = MetricSelectionManager()
        result = manager.run_complete_analysis(
            target_metric="sharpe_ratio",
            optimization_method="balanced_approach"
        )
        
        print(f"  ✓ 分析完了 - 信頼度: {result['confidence_level']}")
        print(f"  ✓ 推奨指標数: {len(result['recommended_metrics'])}")
        print(f"  ✓ 重み改善スコア: {result['weight_optimization']['improvement_score']:.3f}")
        
        # 2. 戦略スコアリングシステムでの評価（最適化前）
        print("\n[CHART] ステップ2: 最適化前の戦略スコア評価")
        from config.strategy_scoring_model import StrategyScoreManager
        
        scoring_manager = StrategyScoreManager()
        
        # サンプル戦略データ
        sample_strategies = [
            {
                'strategy_name': 'MomentumInvestingStrategy',
                'ticker': '^N225',
                'performance_metrics': {
                    'sharpe_ratio': 1.45,
                    'sortino_ratio': 1.62,
                    'max_drawdown': 0.12,
                    'win_rate': 0.65,
                    'profit_factor': 1.85,
                    'total_return': 0.15,
                    'volatility': 0.14,
                    'expectancy': 0.08,
                    'consistency_ratio': 0.78
                }
            },
            {
                'strategy_name': 'VWAPBounceStrategy',
                'ticker': '^N225',
                'performance_metrics': {
                    'sharpe_ratio': 1.28,
                    'sortino_ratio': 1.48,
                    'max_drawdown': 0.09,
                    'win_rate': 0.72,
                    'profit_factor': 2.12,
                    'total_return': 0.12,
                    'volatility': 0.11,
                    'expectancy': 0.06,
                    'consistency_ratio': 0.85
                }
            }
        ]
        
        original_scores = []
        for strategy in sample_strategies:
            score = scoring_manager.calculate_composite_score(
                strategy['strategy_name'],
                strategy['ticker'],
                strategy['performance_metrics']
            )
            original_scores.append(score)
            print(f"  {strategy['strategy_name']}: {score.composite_score:.3f}")
        
        # 3. 最適化された重みでの評価
        print("\n[TARGET] ステップ3: 最適化後の戦略スコア評価")
        
        # 最適化された重みを適用
        optimized_weights = result['weight_optimization']['optimized_weights']
        
        # 新しい重みでスコア計算
        from config.strategy_scoring_model import ScoreWeights
        new_weights = ScoreWeights(
            performance=optimized_weights['performance'],
            stability=optimized_weights['stability'],
            risk_adjusted=optimized_weights['risk_adjusted'],
            trend_adaptation=optimized_weights['trend_adaptation'],
            reliability=optimized_weights['reliability']
        )
        
        optimized_scores = []
        for strategy in sample_strategies:
            score = scoring_manager.calculate_composite_score(
                strategy['strategy_name'],
                strategy['ticker'],
                strategy['performance_metrics'],
                weights=new_weights
            )
            optimized_scores.append(score)
            print(f"  {strategy['strategy_name']}: {score.composite_score:.3f}")
        
        # 4. 改善効果の比較
        print("\n[UP] ステップ4: 最適化効果の評価")
        print("\n最適化による変化:")
        for i, strategy in enumerate(sample_strategies):
            improvement = optimized_scores[i].composite_score - original_scores[i].composite_score
            print(f"  {strategy['strategy_name']}: {improvement:+.3f}")
        
        avg_improvement = sum(optimized_scores[i].composite_score - original_scores[i].composite_score 
                            for i in range(len(sample_strategies))) / len(sample_strategies)
        print(f"\n平均改善スコア: {avg_improvement:+.3f}")
        
        # 5. 推奨指標の影響分析
        print("\n🔬 ステップ5: 推奨指標の分析")
        top_metrics = result['recommended_metrics'][:3]
        print(f"\n上位推奨指標:")
        for i, metric in enumerate(top_metrics, 1):
            print(f"  {i}. {metric['metric_name']} (重要度: {metric['importance_score']:.3f})")
        
        # 6. 統合レポートの生成
        print("\n📝 ステップ6: 統合レポートの生成")
        report_data = {
            'original_scores': original_scores,
            'optimized_scores': optimized_scores,
            'improvement': avg_improvement,
            'analysis_result': result,
            'strategies': sample_strategies
        }
        
        report_path = generate_integration_report(report_data)
        print(f"  ✓ レポート生成完了: {report_path}")
        
        print("\n" + "=" * 80)
        print("[SUCCESS] 統合デモが成功しました！")
        print(f"[CHART] 平均スコア改善: {avg_improvement:+.3f}")
        print(f"[UP] 重み最適化による改善: {result['weight_optimization']['improvement_score']:.3f}")
        print(f"[LIST] 詳細レポート: {report_path}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"統合デモでエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_integration_report(data):
    """統合レポートの生成"""
    from pathlib import Path
    
    # レポート保存ディレクトリ
    report_dir = Path("logs/metric_selection_system/integration_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"integration_demo_report_{timestamp}.md"
    
    # レポート内容の生成
    content = f"""# 重要指標選定システム統合デモレポート

**実行日時**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**統合テスト**: [OK] 成功  

## 統合結果サマリー

### スコア改善効果

| 戦略名 | 最適化前 | 最適化後 | 改善効果 |
|--------|----------|----------|----------|
"""
    
    for i, strategy in enumerate(data['strategies']):
        original = data['original_scores'][i].composite_score
        optimized = data['optimized_scores'][i].composite_score
        improvement = optimized - original
        content += f"| {strategy['strategy_name']} | {original:.3f} | {optimized:.3f} | {improvement:+.3f} |\n"
    
    content += f"""
### 重み最適化結果

- **平均スコア改善**: {data['improvement']:+.3f}
- **重み最適化による改善**: {data['analysis_result']['weight_optimization']['improvement_score']:.3f}
- **分析信頼度**: {data['analysis_result']['confidence_level']}

### 推奨指標ランキング

"""
    
    for i, metric in enumerate(data['analysis_result']['recommended_metrics'][:5], 1):
        content += f"{i}. **{metric['metric_name']}** (重要度: {metric['importance_score']:.3f})\n"
    
    content += f"""
### 重み変更詳細

"""
    
    weights = data['analysis_result']['weight_optimization']['optimized_weights']
    original_weights = data['analysis_result']['weight_optimization'].get('original_weights', {})
    
    for category, new_weight in weights.items():
        original = original_weights.get(category, 0.0)
        change = new_weight - original
        content += f"- **{category}**: {original:.3f} → {new_weight:.3f} ({change:+.3f})\n"
    
    content += f"""
## 推奨事項

1. **重み最適化の効果が確認されました** - 平均スコア改善: {data['improvement']:+.3f}
2. **推奨指標の活用** - 上位指標を重点的に監視することを推奨
3. **継続的な分析** - 定期的な重要指標分析により最適化を継続

---
*レポート生成: 重要指標選定システム統合デモ*
"""
    
    # ファイルに保存
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(report_path)

def demo_quick_analysis():
    """クイック分析デモ"""
    print("=" * 60)
    print("重要指標選定システム クイック分析デモ")
    print("=" * 60)
    
    try:
        from config.metric_selection_manager import MetricSelectionManager
        
        manager = MetricSelectionManager()
        
        # 分析実行
        print("\n[CHART] 重要指標分析を実行中...")
        result = manager.run_complete_analysis(
            target_metric="sharpe_ratio",
            optimization_method="importance_based"
        )
        
        print(f"✓ 分析完了")
        print(f"  - 推奨指標数: {len(result.recommended_metrics)}")
        print(f"  - 信頼度: {result.confidence_level}")
        if result.weight_optimization_result:
            print(f"  - 改善スコア: {result.weight_optimization_result.improvement_score:.3f}")
        
        # 上位推奨指標を表示
        print("\n🏆 上位推奨指標:")
        for i, metric in enumerate(result.recommended_metrics[:5], 1):
            print(f"  {i}. {metric['feature']} (重要度: {metric['importance_score']:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"クイック分析でエラーが発生しました: {e}")
        return False

if __name__ == "__main__":
    print("[ROCKET] 重要指標選定システム 統合デモを開始します\n")
    
    # メニュー選択
    print("実行するデモを選択してください:")
    print("1. 完全統合デモ（推奨）")
    print("2. クイック分析デモ")
    
    try:
        choice = input("\n選択 (1-2): ").strip()
        
        if choice == "1":
            success = demo_complete_integration()
        elif choice == "2":
            success = demo_quick_analysis()
        else:
            print("無効な選択です。デフォルトでクイック分析を実行します。")
            success = demo_quick_analysis()
        
        if success:
            print("\n[OK] デモが正常に完了しました！")
        else:
            print("\n[ERROR] デモ実行中にエラーが発生しました。")
            
    except KeyboardInterrupt:
        print("\n\n[WARNING] デモが中断されました。")
    except Exception as e:
        print(f"\n[ERROR] 予期しないエラーが発生しました: {e}")

"""
オーバーフィッティング検出機能（モメンタム戦略用）
"""
import pandas as pd
import numpy as np
from typing import Dict, List

class OverfittingDetector:
    def __init__(self):
        self.momentum_thresholds = {
            "sharpe_ratio_max": 3.0,           # 異常に高いシャープレシオ
            "parameter_count_max": 10,         # パラメータ数上限
            "win_rate_max": 0.85,             # 異常に高い勝率
            "complexity_score_max": 15,        # 複雑性スコア上限
            "total_return_max": 2.0,           # 異常に高い総リターン（200%）
            "sortino_ratio_max": 4.0           # 異常に高いソルティノレシオ
        }
    
    def detect_overfitting(self, optimization_results: pd.DataFrame) -> Dict:
        """モメンタム戦略のオーバーフィッティング検出"""
        warnings = []
        best_result = optimization_results.iloc[0]
        
        # 1. 異常に高いパフォーマンス指標をチェック
        sharpe_ratio = best_result.get('sharpe_ratio', 0)
        if sharpe_ratio > self.momentum_thresholds['sharpe_ratio_max']:
            warnings.append(f"異常に高いシャープレシオ: {sharpe_ratio:.2f}")
        
        sortino_ratio = best_result.get('sortino_ratio', 0)
        if sortino_ratio > self.momentum_thresholds['sortino_ratio_max']:
            warnings.append(f"異常に高いソルティノレシオ: {sortino_ratio:.2f}")
        
        total_return = best_result.get('total_return', 0)
        if total_return > self.momentum_thresholds['total_return_max']:
            warnings.append(f"異常に高い総リターン: {total_return:.1%}")
        
        win_rate = best_result.get('win_rate', 0)
        if win_rate > self.momentum_thresholds['win_rate_max']:
            warnings.append(f"異常に高い勝率: {win_rate:.1%}")
        
        # 2. パラメータの複雑性チェック
        param_columns = [col for col in optimization_results.columns 
                        if col not in ['sharpe_ratio', 'sortino_ratio', 'total_return', 
                                     'max_drawdown', 'win_rate', 'expectancy', 'score']]
        param_count = len(param_columns)
        
        if param_count > self.momentum_thresholds['parameter_count_max']:
            warnings.append(f"パラメータ数が多すぎます: {param_count}個")
        
        # 3. 結果の一貫性チェック
        if len(optimization_results) > 1:
            top_results = optimization_results.head(3)
            sharpe_std = top_results['sharpe_ratio'].std() if 'sharpe_ratio' in top_results.columns else 0
            if sharpe_std > 0.5:  # 上位結果のばらつきが大きい
                warnings.append(f"上位結果のパフォーマンスにばらつきが大きい: std={sharpe_std:.2f}")
        
        # 4. リスク調整後リターンの妥当性チェック
        max_drawdown = abs(best_result.get('max_drawdown', 0))
        if max_drawdown < 0.02:  # 最大ドローダウンが2%未満は異常
            warnings.append(f"最大ドローダウンが異常に小さい: {max_drawdown:.1%}")
        
        # 5. リスクレベル判定
        risk_level = "low"
        if len(warnings) >= 3:
            risk_level = "high"
        elif len(warnings) >= 1:
            risk_level = "medium"
        
        return {
            "overfitting_risk": risk_level,
            "warnings": warnings,
            "parameter_count": param_count,
            "recommendations": self._generate_recommendations(warnings, risk_level),
            "analysis_details": {
                "best_sharpe_ratio": sharpe_ratio,
                "best_total_return": total_return,
                "best_win_rate": win_rate,
                "parameter_complexity": param_count
            }
        }
    
    def _generate_recommendations(self, warnings: List[str], risk_level: str) -> List[str]:
        """推奨事項を生成"""
        recommendations = []
        
        if risk_level == "high":
            recommendations.extend([
                "⚠️ 高いオーバーフィッティングリスクが検出されました",
                "📊 アウトオブサンプル期間でのテストを必須で実施してください",
                "🔧 パラメータ数を大幅に減らすことを強く推奨します",
                "📈 より長期間のデータでの検証を実施してください",
                "🚫 このパラメータセットの実運用は推奨されません"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "⚡ 中程度のオーバーフィッティングリスクが検出されました", 
                "🧪 追加のバックテスト期間での検証を推奨します",
                "📉 パラメータの簡素化を検討してください",
                "🔍 ウォークフォワード分析を実施してください"
            ])
        else:
            recommendations.extend([
                "✅ オーバーフィッティングリスクは低いです",
                "📋 定期的なパフォーマンス監視を継続してください"
            ])
        
        return recommendations
    
    def generate_validation_report(self, analysis_result: Dict) -> str:
        """検証レポートを生成"""
        report = f"""
=== オーバーフィッティング検証レポート ===
🎯 リスクレベル: {analysis_result['overfitting_risk'].upper()}
📊 パラメータ数: {analysis_result['parameter_count']}

📈 パフォーマンス分析:
  - シャープレシオ: {analysis_result['analysis_details']['best_sharpe_ratio']:.2f}
  - 総リターン: {analysis_result['analysis_details']['best_total_return']:.1%}
  - 勝率: {analysis_result['analysis_details']['best_win_rate']:.1%}

⚠️ 検出された警告:
"""
        for warning in analysis_result['warnings']:
            report += f"  - {warning}\n"
        
        if not analysis_result['warnings']:
            report += "  なし\n"
        
        report += "\n💡 推奨事項:\n"
        for rec in analysis_result['recommendations']:
            report += f"  {rec}\n"
        
        return report

"""
対話式パラメータレビューシステム

このモジュールは最適化されたパラメータセットのレビューと承認を管理します。
オーバーフィッティング検出結果と妥当性検証結果を表示し、
人間のレビュアーが情報に基づいた承認判断を行えるよう支援します。
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# プロジェクトルートを追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.optimized_parameters import OptimizedParameterManager
from optimization.overfitting_detector import OverfittingDetector
from validation.parameter_validator import ParameterValidator


@dataclass
class ReviewDecision:
    """レビュー決定を格納するデータクラス"""
    approved: bool
    reviewer_id: str
    review_date: str
    notes: str
    confidence_level: int  # 1-5の信頼度
    risk_acceptance: str   # 'low', 'medium', 'high'


class ParameterReviewer:
    """
    最適化されたパラメータの対話式レビューシステム
    
    機能:
    - 最適化結果の詳細表示
    - オーバーフィッティング検出結果の可視化
    - パラメータ妥当性検証結果の表示
    - 対話式承認プロセス
    - レビュー履歴の管理
    """
    
    def __init__(self, reviewer_id: str = "default_reviewer"):
        self.parameter_manager = OptimizedParameterManager()
        self.overfitting_detector = OverfittingDetector()
        self.parameter_validator = ParameterValidator()
        self.reviewer_id = reviewer_id
        
    def start_review_session(self, strategy_name: str) -> None:
        """レビューセッションを開始"""
        print(f"\n{'='*60}")
        print(f"パラメータレビューシステム - {strategy_name}")
        print(f"レビュアー: {self.reviewer_id}")
        print(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
        # 利用可能なパラメータセットを表示
        available_sets = self.parameter_manager.list_parameter_sets(strategy_name)
        
        if not available_sets:
            print(f"❌ {strategy_name}の最適化結果が見つかりません。")
            return
        
        print(f"📊 {strategy_name}の利用可能なパラメータセット:")
        for i, param_set in enumerate(available_sets):
            status = param_set.get('status', 'pending')
            sharpe = param_set.get('sharpe_ratio', 'N/A')
            total_return = param_set.get('total_return', 'N/A')
            print(f"  {i+1}. ID: {param_set['parameter_id']} | "
                  f"Sharpe: {sharpe:.4f} | Return: {total_return:.2%} | "
                  f"Status: {status}")
        
        # レビューするパラメータセットを選択
        while True:
            try:
                choice = input(f"\nレビューするパラメータセットを選択 (1-{len(available_sets)}, 'q'で終了): ")
                if choice.lower() == 'q':
                    return
                
                index = int(choice) - 1
                if 0 <= index < len(available_sets):
                    selected_set = available_sets[index]
                    self._review_parameter_set(strategy_name, selected_set)
                    
                    # 続行するかどうか確認
                    if input("\n他のパラメータセットをレビューしますか？ (y/n): ").lower() != 'y':
                        break
                else:
                    print("❌ 無効な選択です。")
            except ValueError:
                print("❌ 数値を入力してください。")
    
    def _review_parameter_set(self, strategy_name: str, param_set: Dict) -> None:
        """個別パラメータセットのレビュー"""
        param_id = param_set['parameter_id']
        
        print(f"\n{'='*60}")
        print(f"パラメータセット詳細レビュー - ID: {param_id}")
        print(f"{'='*60}")
        
        # 1. 基本情報表示
        self._display_basic_info(param_set)
        
        # 2. パフォーマンス指標表示
        self._display_performance_metrics(param_set)
        
        # 3. オーバーフィッティング検出結果
        overfitting_result = self._analyze_overfitting(param_set)
        
        # 4. パラメータ妥当性検証結果
        validation_result = self._validate_parameters(strategy_name, param_set)
        
        # 5. 総合リスク評価
        overall_risk = self._calculate_overall_risk(overfitting_result, validation_result)
        
        # 6. 対話式承認プロセス
        decision = self._interactive_approval(param_set, overfitting_result, 
                                            validation_result, overall_risk)
        
        # 7. 決定を保存
        self._save_review_decision(strategy_name, param_id, decision)
    
    def _display_basic_info(self, param_set: Dict) -> None:
        """基本情報の表示"""
        print("\n📋 基本情報:")
        print(f"  作成日時: {param_set.get('created_at', 'N/A')}")
        print(f"  最適化期間: {param_set.get('optimization_period', 'N/A')}")
        print(f"  データ期間: {param_set.get('data_start_date', 'N/A')} - {param_set.get('data_end_date', 'N/A')}")
        print(f"  現在のステータス: {param_set.get('status', 'pending')}")
    
    def _display_performance_metrics(self, param_set: Dict) -> None:
        """パフォーマンス指標の表示"""
        print("\n📈 パフォーマンス指標:")
        metrics = param_set.get('performance_metrics', {})
        
        key_metrics = [
            ('sharpe_ratio', 'シャープレシオ', '.4f'),
            ('total_return', 'トータルリターン', '.2%'),
            ('max_drawdown', '最大ドローダウン', '.2%'),
            ('win_rate', '勝率', '.2%'),
            ('profit_factor', 'プロフィットファクター', '.4f'),
            ('volatility', 'ボラティリティ', '.4f')
        ]
        
        for key, label, fmt in key_metrics:
            value = metrics.get(key, param_set.get(key, 'N/A'))
            if isinstance(value, (int, float)):
                print(f"  {label}: {value:{fmt}}")
            else:
                print(f"  {label}: {value}")
    
    def _analyze_overfitting(self, param_set: Dict) -> Dict:
        """オーバーフィッティング分析"""
        print("\n🔍 オーバーフィッティング検出:")
        
        # パフォーマンス指標からデータを抽出
        performance_data = {
            'sharpe_ratio': param_set.get('sharpe_ratio', 0),
            'total_return': param_set.get('total_return', 0),
            'max_drawdown': param_set.get('max_drawdown', 0),
            'win_rate': param_set.get('win_rate', 0.5),
            'volatility': param_set.get('volatility', 0.1)
        }
        
        parameters = param_set.get('parameters', {})
        
        # オーバーフィッティング検出実行
        result = self.overfitting_detector.detect_overfitting(performance_data, parameters)
        
        # 結果表示
        print(f"  🎯 総合リスクレベル: {result['overall_risk_level']}")
        print(f"  📊 リスクスコア: {result['risk_score']:.2f}")
        
        print("\n  詳細検出結果:")
        for detection in result['detections']:
            risk_icon = "🔴" if detection['risk_level'] == 'high' else "🟡" if detection['risk_level'] == 'medium' else "🟢"
            print(f"    {risk_icon} {detection['type']}: {detection['risk_level']}")
            print(f"       理由: {detection['reason']}")
        
        if result['recommendations']:
            print("\n  💡 推奨事項:")
            for rec in result['recommendations']:
                print(f"    • {rec}")
        
        return result
    
    def _validate_parameters(self, strategy_name: str, param_set: Dict) -> Dict:
        """パラメータ妥当性検証"""
        print("\n✅ パラメータ妥当性検証:")
        
        parameters = param_set.get('parameters', {})
        result = self.parameter_validator.validate_parameters(strategy_name, parameters)
        
        # 結果表示
        print(f"  📋 検証結果: {'✅ 合格' if result['is_valid'] else '❌ 不合格'}")
        print(f"  📊 信頼度スコア: {result['confidence_score']:.2f}")
        
        if result['errors']:
            print("\n  ❌ エラー:")
            for error in result['errors']:
                print(f"    • {error}")
        
        if result['warnings']:
            print("\n  ⚠️ 警告:")
            for warning in result['warnings']:
                print(f"    • {warning}")
        
        if result['recommendations']:
            print("\n  💡 推奨事項:")
            for rec in result['recommendations']:
                print(f"    • {rec}")
        
        return result
    
    def _calculate_overall_risk(self, overfitting_result: Dict, validation_result: Dict) -> str:
        """総合リスクレベルの計算"""
        risk_levels = ['low', 'medium', 'high']
        
        # オーバーフィッティングリスク
        overfitting_risk = overfitting_result.get('overall_risk_level', 'medium')
        
        # 妥当性検証リスク
        validation_risk = 'low' if validation_result.get('is_valid', False) else 'high'
        if validation_result.get('warnings', []):
            validation_risk = 'medium' if validation_risk == 'low' else validation_risk
        
        # より高いリスクレベルを採用
        overall_risk_index = max(
            risk_levels.index(overfitting_risk),
            risk_levels.index(validation_risk)
        )
        
        return risk_levels[overall_risk_index]
    
    def _interactive_approval(self, param_set: Dict, overfitting_result: Dict, 
                             validation_result: Dict, overall_risk: str) -> ReviewDecision:
        """対話式承認プロセス"""
        print(f"\n{'='*60}")
        print("📝 レビュー決定")
        print(f"{'='*60}")
        
        print(f"\n🎯 総合リスク評価: {overall_risk.upper()}")
        
        # リスクレベルに応じた推奨アクション
        risk_recommendations = {
            'low': "✅ 承認推奨 - リスクは低く、本番環境での使用に適しています",
            'medium': "⚠️ 慎重検討 - 追加検証や制限付き運用を検討してください", 
            'high': "❌ 承認非推奨 - 高リスクのため、さらなる最適化が必要です"
        }
        
        print(f"💡 推奨アクション: {risk_recommendations[overall_risk]}")
        
        # 承認決定
        while True:
            decision = input("\n決定を選択してください (approve/reject/defer): ").lower()
            if decision in ['approve', 'reject', 'defer']:
                break
            print("❌ 'approve', 'reject', 'defer'のいずれかを入力してください。")
        
        approved = decision == 'approve'
        
        # 追加情報収集
        notes = input("レビューノート（任意）: ") or "なし"
        
        while True:
            try:
                confidence = int(input("信頼度レベル (1-5, 5が最高): "))
                if 1 <= confidence <= 5:
                    break
                print("❌ 1-5の範囲で入力してください。")
            except ValueError:
                print("❌ 数値を入力してください。")
        
        while True:
            risk_acceptance = input("リスク受容レベル (low/medium/high): ").lower()
            if risk_acceptance in ['low', 'medium', 'high']:
                break
            print("❌ 'low', 'medium', 'high'のいずれかを入力してください。")
        
        return ReviewDecision(
            approved=approved,
            reviewer_id=self.reviewer_id,
            review_date=datetime.now().isoformat(),
            notes=notes,
            confidence_level=confidence,
            risk_acceptance=risk_acceptance
        )
    
    def _save_review_decision(self, strategy_name: str, param_id: str, decision: ReviewDecision) -> None:
        """レビュー決定を保存"""
        # パラメータセットのステータス更新
        new_status = 'approved' if decision.approved else 'rejected'
        self.parameter_manager.update_parameter_status(strategy_name, param_id, new_status)
        
        # レビュー履歴の保存
        review_data = {
            'parameter_id': param_id,
            'strategy_name': strategy_name,
            'decision': 'approved' if decision.approved else 'rejected',
            'reviewer_id': decision.reviewer_id,
            'review_date': decision.review_date,
            'notes': decision.notes,
            'confidence_level': decision.confidence_level,
            'risk_acceptance': decision.risk_acceptance
        }
        
        self._save_review_history(review_data)
        
        # 結果表示
        status_icon = "✅" if decision.approved else "❌"
        print(f"\n{status_icon} レビュー決定が保存されました:")
        print(f"  決定: {new_status}")
        print(f"  パラメータID: {param_id}")
        print(f"  信頼度: {decision.confidence_level}/5")
    
    def _save_review_history(self, review_data: Dict) -> None:
        """レビュー履歴をファイルに保存"""
        history_dir = os.path.join(project_root, 'config', 'review_history')
        os.makedirs(history_dir, exist_ok=True)
        
        history_file = os.path.join(history_dir, f"{review_data['strategy_name']}_reviews.json")
        
        # 既存履歴の読み込み
        reviews = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    reviews = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                reviews = []
        
        # 新しいレビューを追加
        reviews.append(review_data)
        
        # 保存
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)
    
    def show_review_history(self, strategy_name: str) -> None:
        """レビュー履歴の表示"""
        history_file = os.path.join(project_root, 'config', 'review_history', f"{strategy_name}_reviews.json")
        
        if not os.path.exists(history_file):
            print(f"❌ {strategy_name}のレビュー履歴が見つかりません。")
            return
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("❌ レビュー履歴の読み込みに失敗しました。")
            return
        
        print(f"\n📚 {strategy_name}のレビュー履歴:")
        print(f"{'='*60}")
        
        for review in sorted(reviews, key=lambda x: x['review_date'], reverse=True):
            status_icon = "✅" if review['decision'] == 'approved' else "❌"
            print(f"\n{status_icon} パラメータID: {review['parameter_id']}")
            print(f"   決定: {review['decision']}")
            print(f"   レビュアー: {review['reviewer_id']}")
            print(f"   日時: {review['review_date']}")
            print(f"   信頼度: {review['confidence_level']}/5")
            print(f"   リスク受容: {review['risk_acceptance']}")
            if review['notes'] != "なし":
                print(f"   ノート: {review['notes']}")


def main():
    """メイン関数 - コマンドライン実行用"""
    if len(sys.argv) < 2:
        print("使用方法: python parameter_reviewer.py <strategy_name> [reviewer_id]")
        print("例: python parameter_reviewer.py MomentumInvestingStrategy john_doe")
        return
    
    strategy_name = sys.argv[1]
    reviewer_id = sys.argv[2] if len(sys.argv) > 2 else "default_reviewer"
    
    reviewer = ParameterReviewer(reviewer_id)
    
    while True:
        print(f"\n{'='*60}")
        print("パラメータレビューシステム")
        print(f"{'='*60}")
        print("1. レビューセッション開始")
        print("2. レビュー履歴表示")
        print("3. 終了")
        
        choice = input("\n選択してください (1-3): ")
        
        if choice == '1':
            reviewer.start_review_session(strategy_name)
        elif choice == '2':
            reviewer.show_review_history(strategy_name)
        elif choice == '3':
            print("レビューシステムを終了します。")
            break
        else:
            print("❌ 無効な選択です。")


if __name__ == "__main__":
    main()

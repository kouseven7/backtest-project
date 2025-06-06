"""
最適化パラメータのレビューツール（修正版）
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Any

# プロジェクトルートをPythonパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.optimized_parameters import OptimizedParameterManager
from validation.parameter_validator import ParameterValidator


class ParameterReviewer:
    def __init__(self):
        self.parameter_manager = OptimizedParameterManager()
        self.validator = ParameterValidator()
        self.review_log = []
          # 戦略名マッピング（短縮名から正式名への変換）
        self.strategy_mapping = {
            'momentum': 'MomentumInvestingStrategy',
            'breakout': 'BreakoutStrategy',
            'contrarian': 'ContrarianStrategy',
            'vwap': 'VWAPStrategy'
        }
    
    def _normalize_strategy_name(self, strategy_name: str) -> str:
        """戦略名を正規化（短縮名を正式名に変換）"""
        return self.strategy_mapping.get(strategy_name.lower(), strategy_name)
    
    def start_review_session(self, strategy_name: str = "momentum"):
        """レビューセッションを開始"""
        # 戦略名を正規化
        normalized_strategy_name = self._normalize_strategy_name(strategy_name)
        
        print(f"\n🔍 {strategy_name}戦略のパラメータレビューを開始します...")
        if strategy_name != normalized_strategy_name:
            print(f"   ({strategy_name} → {normalized_strategy_name})")
        
        # pending_reviewのファイルを取得
        available_configs = self.parameter_manager.list_available_configs(
            strategy_name=normalized_strategy_name,
            status="pending_review"
        )
        
        if not available_configs:
            print(f"❌ レビュー待ちの{strategy_name}戦略設定はありません。")
            self._show_available_files(normalized_strategy_name)
            return
        
        print(f"📋 レビュー対象: {len(available_configs)}件")
        
        # 各ファイルをレビュー
        for i, config in enumerate(available_configs, 1):
            print(f"\n{'='*60}")
            print(f"📁 ファイル {i}/{len(available_configs)}: {config['filename']}")
            
            review_result = self._review_single_config(config)
            self.review_log.append(review_result)
            
            if review_result['action'] == 'quit':
                break
        
        # レビュー結果のサマリー
        self._show_review_summary()
    
    def _show_available_files(self, strategy_name: str):
        """利用可能なファイルを表示"""
        all_configs = self.parameter_manager.list_available_configs(strategy_name=strategy_name)
        
        if all_configs:
            print(f"\n📂 {strategy_name}戦略の利用可能なファイル:")
            status_emoji = {
                "approved": "✅",
                "pending_review": "⏳", 
                "rejected": "❌"
            }
            
            for config in all_configs[:10]:  # 最大10件表示
                emoji = status_emoji.get(config.get('status'), "❓")
                print(f"  {emoji} {config['filename']} ({config.get('status', 'unknown')})")
            
            if len(all_configs) > 10:
                print(f"  ... 他 {len(all_configs) - 10}件")
        else:
            print(f"❌ {strategy_name}戦略のファイルが見つかりません。")
    
    def _review_single_config(self, config: Dict) -> Dict:
        """単一設定ファイルのレビュー"""
        print(f"📊 銘柄: {config.get('ticker', 'N/A')}")
        print(f"📅 最適化日: {config.get('optimization_date', 'N/A')}")
        
        # パフォーマンス指標表示
        self._display_performance_metrics(config.get('performance_metrics', {}))
        
        # パラメータ表示
        params = config.get('parameters', {})
        self._display_parameters(params)
        
        # 検証結果表示
        self._display_validation_info(config.get('validation_info', {}))
        
        # パラメータ妥当性の再検証
        if params:
            validation_result = self.validator.validate_momentum_parameters(params)
            self._display_revalidation_result(validation_result)
        
        # レビュー判定
        return self._get_review_decision(config)
    
    def _display_performance_metrics(self, metrics: Dict):
        """パフォーマンス指標を表示"""
        print(f"\n📊 パフォーマンス指標:")
        
        metric_display = {
            'sharpe_ratio': ('シャープレシオ', ''),
            'sortino_ratio': ('ソルティノレシオ', ''),
            'total_return': ('総リターン', '%'),
            'max_drawdown': ('最大ドローダウン', '%'),
            'win_rate': ('勝率', '%'),
            'total_trades': ('総取引数', '回'),
            'profit_factor': ('プロフィットファクター', '')
        }
        
        for key, (label, unit) in metric_display.items():
            value = metrics.get(key, 'N/A')
            if value != 'N/A' and unit == '%':
                if isinstance(value, (int, float)):
                    value = f"{value:.1%}" if abs(value) < 1 else f"{value:.1f}%"
            elif value != 'N/A' and isinstance(value, float):
                value = f"{value:.3f}"
            print(f"  {label}: {value}")
    
    def _display_parameters(self, params: Dict):
        """パラメータを表示"""
        print(f"\n⚙️ 最適化パラメータ:")
        
        if not params:
            print("  パラメータが見つかりません")
            return
        
        # パラメータをカテゴリ別に整理
        categories = {
            'テクニカル指標': ['sma_short', 'sma_long', 'rsi_period', 'rsi_lower', 'rsi_upper'],
            'リスク管理': ['take_profit', 'stop_loss', 'trailing_stop'],
            'その他': []
        }
        
        # カテゴリに属さないパラメータを「その他」に追加
        categorized_params = set()
        for cat_params in categories.values():
            categorized_params.update(cat_params)
        
        for param_name in params:
            if param_name not in categorized_params:
                categories['その他'].append(param_name)
        
        # カテゴリ別に表示
        for category, param_names in categories.items():
            category_params = {k: v for k, v in params.items() if k in param_names}
            if category_params:
                print(f"  📋 {category}:")
                for k, v in category_params.items():
                    print(f"    {k}: {v}")
    
    def _display_validation_info(self, validation_info: Dict):
        """検証情報を表示"""
        if not validation_info:
            return
        
        print(f"\n🔍 検証結果:")
        
        # オーバーフィッティングリスク
        overfitting_risk = validation_info.get('overfitting_risk', 'N/A')
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(overfitting_risk, "❓")
        print(f"  オーバーフィッティングリスク: {risk_emoji} {overfitting_risk}")
        
        # パラメータ検証
        param_validation = validation_info.get('parameter_validation')
        if param_validation is not None:
            validation_emoji = "✅" if param_validation else "❌"
            status_text = "通過" if param_validation else "不合格"
            print(f"  パラメータ検証: {validation_emoji} {status_text}")
    
    def _display_revalidation_result(self, validation_result: Dict):
        """再検証結果を表示"""
        print(f"\n🔍 パラメータ再検証:")
        print(f"  結果: {'✅ 合格' if validation_result['valid'] else '❌ 不合格'}")
        
        if validation_result.get('errors'):
            print(f"  ❌ エラー ({len(validation_result['errors'])}件):")
            for error in validation_result['errors'][:3]:  # 最大3件表示
                print(f"    • {error}")
            if len(validation_result['errors']) > 3:
                print(f"    ... 他 {len(validation_result['errors']) - 3}件")
        
        if validation_result.get('warnings'):
            print(f"  ⚠️ 警告 ({len(validation_result['warnings'])}件):")
            for warning in validation_result['warnings'][:3]:  # 最大3件表示
                print(f"    • {warning}")
            if len(validation_result['warnings']) > 3:
                print(f"    ... 他 {len(validation_result['warnings']) - 3}件")
    
    def _get_review_decision(self, config: Dict) -> Dict:
        """レビュー判定を取得"""
        while True:
            print(f"\n👤 レビュー判定:")
            print("  a = 承認 (approved)")
            print("  r = 却下 (rejected)")
            print("  s = スキップ")
            print("  d = 詳細表示")
            print("  q = レビュー終了")
            
            choice = input("選択 (a/r/s/d/q): ").lower().strip()
            
            if choice == 'a':
                return self._approve_config(config)
            elif choice == 'r':
                return self._reject_config(config)
            elif choice == 's':
                print("⏭️ スキップしました。")
                return {'action': 'skip', 'config': config['filename']}
            elif choice == 'd':
                self._show_detailed_info(config)
                continue
            elif choice == 'q':
                print("🚪 レビューを終了します。")
                return {'action': 'quit', 'config': config['filename']}
            else:
                print("❌ 無効な選択です。a, r, s, d, q のいずれかを入力してください。")
    
    def _approve_config(self, config: Dict) -> Dict:
        """設定を承認"""
        config['status'] = 'approved'
        config['approved_by'] = 'default_reviewer'
        config['approved_at'] = datetime.now().isoformat()
        
        # ファイルを更新
        filepath = os.path.join(self.parameter_manager.config_dir, config['filename'])
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("✅ 承認しました。")
        return {'action': 'approve', 'config': config['filename']}
    
    def _reject_config(self, config: Dict) -> Dict:
        """設定を却下"""
        reason = input("却下理由を入力してください: ").strip()
        
        config['status'] = 'rejected'
        config['rejected_by'] = 'default_reviewer'
        config['rejected_at'] = datetime.now().isoformat()
        config['rejection_reason'] = reason
        
        # ファイルを更新
        filepath = os.path.join(self.parameter_manager.config_dir, config['filename'])
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("❌ 却下しました。")
        return {'action': 'reject', 'config': config['filename'], 'reason': reason}
    
    def _show_detailed_info(self, config: Dict):
        """詳細情報を表示"""
        print(f"\n📋 詳細情報:")
        print(f"ファイルパス: {config.get('filename', 'N/A')}")
        print(f"作成日時: {config.get('created_at', 'N/A')}")
        
        # パラメータ詳細
        params = config.get('parameters', {})
        if params:
            validation_result = self.validator.validate_momentum_parameters(params)
            detailed_report = self.validator.generate_validation_report(validation_result)
            print(detailed_report)
    
    def _show_review_summary(self):
        """レビュー結果のサマリーを表示"""
        if not self.review_log:
            return
        
        print(f"\n{'='*60}")
        print(f"📊 レビューセッション完了")
        print(f"{'='*60}")
        
        # 統計
        actions = [log['action'] for log in self.review_log]
        approve_count = actions.count('approve')
        reject_count = actions.count('reject')
        skip_count = actions.count('skip')
        
        print(f"✅ 承認: {approve_count}件")
        print(f"❌ 却下: {reject_count}件") 
        print(f"⏭️ スキップ: {skip_count}件")
        
        # 却下理由（あれば）
        reject_logs = [log for log in self.review_log if log['action'] == 'reject']
        if reject_logs:
            print(f"\n❌ 却下理由:")
            for log in reject_logs:
                print(f"  • {log['config']}: {log.get('reason', '理由なし')}")
    
    def show_review_history(self):
        """レビュー履歴を表示"""
        print(f"\n📜 レビュー履歴表示機能（未実装）")
        print("この機能は今後のバージョンで実装予定です。")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='パラメータレビューシステム')
    parser.add_argument('--strategy', '-s', default='momentum', 
                       help='レビューする戦略名 (デフォルト: momentum)')
    parser.add_argument('--auto-mode', action='store_true',
                       help='自動モードでレビューセッションを直接開始')
    
    args = parser.parse_args()
    
    reviewer = ParameterReviewer()
    
    # 自動モードの場合は直接レビューセッション開始
    if args.auto_mode:
        print(f"\n{'='*60}")
        print(f"パラメータレビューシステム - {args.strategy}")
        print(f"レビュアー: default_reviewer")
        print(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        reviewer.start_review_session(args.strategy)
        return
    
    # 対話モード
    while True:
        print(f"\n{'='*60}")
        print(f"📋 パラメータレビューシステム")
        print(f"{'='*60}")
        print("1. レビューセッション開始")
        print("2. レビュー履歴表示")
        print("3. 終了")
        
        choice = input("\n選択してください (1-3): ").strip()
        
        if choice == '1':
            strategy_name = input(f"戦略名を入力してください (デフォルト: {args.strategy}): ").strip()
            if not strategy_name:
                strategy_name = args.strategy
            
            print(f"\n{'='*60}")
            print(f"パラメータレビューシステム - {strategy_name}")
            print(f"レビュアー: default_reviewer")
            print(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            reviewer.start_review_session(strategy_name)
            
        elif choice == '2':
            reviewer.show_review_history()
            
        elif choice == '3':
            print("👋 レビューシステムを終了します。")
            break
            
        else:
            print("❌ 無効な選択です。1-3の数字を入力してください。")


if __name__ == "__main__":
    main()

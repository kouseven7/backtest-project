"""
DSSMS Phase 3 Task 3.1 市場全体監視システム デモンストレーション
実装完了後の動作確認デモ
"""

import sys
from pathlib import Path
import logging

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dssms_market_demo')

def demonstrate_market_monitoring():
    """市場監視システムのデモンストレーション"""
    
    print("=" * 70)
    print("[TARGET] DSSMS Phase 3 Task 3.1 市場全体監視システム デモ")
    print("=" * 70)
    
    try:
        from src.dssms.market_condition_monitor import MarketConditionMonitor, DSSMSMarketMonitorIntegrator
        
        # システム初期化
        print("\n[CHART] システム初期化中...")
        monitor = MarketConditionMonitor()
        integrator = DSSMSMarketMonitorIntegrator()
        print("[OK] 初期化完了")
        
        # 1. 日経225総合分析
        print("\n" + "─" * 50)
        print("[UP] 日経225 総合市場分析")
        print("─" * 50)
        
        trend_analysis = monitor.analyze_nikkei225_trend()
        
        if "error" not in trend_analysis:
            print(f"[SEARCH] トレンド方向: {trend_analysis['trend_direction']}")
            print(f"💪 強度スコア: {trend_analysis['strength_score']:.3f}")
            print(f"[CHART] ボラティリティ: {trend_analysis['volatility_level']}")
            print(f"📦 出来高状況: {trend_analysis['volume_profile']}")
            print(f"[TARGET] パーフェクトオーダー: {trend_analysis['perfect_order_status']}")
            
            # 詳細分析表示
            details = trend_analysis.get('analysis_details', {})
            if details:
                print(f"  [CHART] 現在価格: ¥{details.get('current_price', 'N/A'):,.0f}")
                print(f"  [UP] 1日変化率: {details.get('daily_change', 0)*100:.2f}%")
                print(f"  [DOWN] ボラティリティ: {details.get('volatility_value', 0)*100:.2f}%")
        else:
            print(f"[ERROR] 分析エラー: {trend_analysis['error']}")
        
        # 2. パーフェクトオーダー詳細
        print("\n" + "─" * 50)
        print("[TARGET] SBI準拠 パーフェクトオーダー判定")
        print("─" * 50)
        
        perfect_order = monitor.check_market_perfect_order()
        status_emoji = "[OK]" if perfect_order else "[ERROR]"
        print(f"{status_emoji} パーフェクトオーダー状態: {perfect_order}")
        
        # 3. 市場健全性スコア
        print("\n" + "─" * 50)
        print("🏥 市場健全性総合スコア")
        print("─" * 50)
        
        health_score = monitor.get_market_health_score()
        
        # スコア評価
        if health_score >= 0.7:
            health_status = "🟢 健全"
        elif health_score >= 0.5:
            health_status = "🟡 注意"
        elif health_score >= 0.3:
            health_status = "🟠 警戒"
        else:
            health_status = "🔴 危険"
        
        print(f"[CHART] 総合スコア: {health_score:.3f} ({health_status})")
        
        # 4. 売買停止判定
        print("\n" + "─" * 50)
        print("🚦 売買停止判定システム")
        print("─" * 50)
        
        halt_flag, reason = monitor.should_halt_trading()
        
        if halt_flag:
            print("🔴 売買停止推奨")
            print(f"[LIST] 理由: {reason}")
        else:
            print("🟢 売買継続可能")
            print(f"[LIST] 状況: {reason}")
        
        # 5. 統合管理インターフェース
        print("\n" + "─" * 50)
        print("🔗 統合管理システム")
        print("─" * 50)
        
        # 取引許可確認
        trading_permission = integrator.get_trading_permission()
        permission_emoji = "🟢" if trading_permission['trading_allowed'] else "🔴"
        print(f"{permission_emoji} 取引許可: {trading_permission['trading_allowed']}")
        print(f"[LIST] 判定理由: {trading_permission['reason']}")
        print(f"[CHART] 健全性: {trading_permission['health_score']:.3f}")
        
        # 市場サマリー
        market_summary = integrator.get_market_summary()
        print(f"📡 監視状況: {market_summary['status']}")
        print(f"🕐 最終確認: {market_summary['last_check']}")
        
        # 6. リアルタイム監視設定
        print("\n" + "─" * 50)
        print("⚙️ 監視設定情報")
        print("─" * 50)
        
        print("📅 監視間隔: 15分")
        print("🕘 監視時間: 09:00-11:30, 12:30-15:00")
        print("[TARGET] 監視対象: 日経225 (^N225)")
        print("🔔 アラート: 4段階 (normal/warning/caution/halt)")
        
        # 7. システム推奨事項
        print("\n" + "─" * 50)
        print("[IDEA] システム推奨事項")
        print("─" * 50)
        
        recommendations: list[str] = []
        
        if health_score < 0.5:
            recommendations.append("[CHART] 市場健全性が低下しています。慎重な取引を推奨します。")
        
        if halt_flag:
            recommendations.append("[ALERT] 緊急事態が検出されました。直ちに取引を停止してください。")
        
        if not perfect_order:
            recommendations.append("[TARGET] パーフェクトオーダーが崩れています。トレンド転換に注意してください。")
        
        if trend_analysis.get('volatility_level') == 'high':
            recommendations.append("[CHART] ボラティリティが高騰しています。リスク管理を強化してください。")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("[OK] 現在、特別な推奨事項はありません。")
        
        print("\n" + "=" * 70)
        print("[SUCCESS] DSSMS Phase 3 Task 3.1 デモ完了")
        print("[OK] 市場全体監視システムが正常に稼働中")
        print("=" * 70)
        
    except Exception as e:
        print(f"[ERROR] デモ実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demonstrate_market_monitoring()

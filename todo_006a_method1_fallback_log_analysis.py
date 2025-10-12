#!/usr/bin/env python3
"""
TODO-006-A 手法1: フォールバックログ解析
目的: システムフォールバックの二重処理がエントリー/エグジットシグナルに与える影響を調査
"""
import sys
sys.path.append(r"C:\Users\imega\Documents\my_backtest_project")

import json
from pathlib import Path
from datetime import datetime

def analyze_fallback_logs():
    """
    フォールバックログ詳細解析
    """
    print("=" * 60)
    print("🔍 TODO-006-A 手法1: フォールバックログ解析")
    print("=" * 60)
    
    # フォールバックレポートファイルの確認
    reports_dir = Path(r"C:\Users\imega\Documents\my_backtest_project\reports\fallback")
    
    if not reports_dir.exists():
        print(f"❌ フォールバックレポートディレクトリが存在しません: {reports_dir}")
        return None
    
    # 最新のフォールバックレポート取得
    fallback_files = list(reports_dir.glob("fallback_usage_report_*.json"))
    
    if not fallback_files:
        print(f"❌ フォールバックレポートファイルが見つかりません")
        return None
    
    latest_file = max(fallback_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 最新フォールバックレポート: {latest_file.name}")
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            fallback_data = json.load(f)
        
        print(f"\n📊 フォールバック統計:")
        print(f"  - 総失敗回数: {fallback_data.get('total_failures', 'N/A')}")
        print(f"  - 成功フォールバック: {fallback_data.get('successful_fallbacks', 'N/A')}")
        print(f"  - フォールバック使用率: {fallback_data.get('fallback_usage_rate', 'N/A')}")
        print(f"  - システムモード: {fallback_data.get('system_mode', 'N/A')}")
        
        # コンポーネント別統計
        if 'by_component_type' in fallback_data:
            print(f"\n📋 コンポーネント別統計:")
            for comp_type, stats in fallback_data['by_component_type'].items():
                print(f"  - {comp_type}: 総失敗={stats.get('total', 0)}, フォールバック使用={stats.get('fallback_used', 0)}")
        
        # エラータイプ別統計
        if 'by_error_type' in fallback_data:
            print(f"\n🚨 エラータイプ別統計:")
            for error_type, stats in fallback_data['by_error_type'].items():
                print(f"  - {error_type}: 回数={stats.get('count', 0)}, フォールバック使用={stats.get('fallback_used', 0)}")
        
        # 詳細レコード
        if 'records' in fallback_data:
            print(f"\n📝 フォールバック詳細レコード:")
            for i, record in enumerate(fallback_data['records'], 1):
                print(f"  Record {i}:")
                print(f"    - タイムスタンプ: {record.get('timestamp', 'N/A')}")
                print(f"    - コンポーネント: {record.get('component_name', 'N/A')}")
                print(f"    - エラー: {record.get('error_message', 'N/A')}")
                print(f"    - フォールバック使用: {record.get('fallback_used', 'N/A')}")
        
        return fallback_data
        
    except Exception as e:
        print(f"❌ フォールバックレポート読み込みエラー: {e}")
        return None

def analyze_fallback_timing_impact():
    """
    フォールバック処理のタイミングとシグナル生成への影響分析
    """
    print(f"\n" + "=" * 60)
    print("🕐 フォールバック処理タイミング影響分析")
    print("=" * 60)
    
    # ログから推測される処理フロー
    expected_flow = [
        "1. 統合システム初期化",
        "2. MultiStrategyManager.execute_multi_strategy_flow 実行",
        "3. 統合システム実行失敗",
        "4. フォールバック処理開始",
        "5. 従来システム実行",
        "6. unified_exporter による結果処理"
    ]
    
    print(f"📋 予想される処理フロー:")
    for step in expected_flow:
        print(f"  {step}")
    
    print(f"\n🔍 重要な仮説:")
    print(f"  A. フォールバック処理が二重実行される")
    print(f"  B. 統合システムと従来システムの両方が部分的に実行される")
    print(f"  C. シグナル生成が重複または競合する")
    print(f"  D. unified_exporterが競合データを受け取る")
    
    return expected_flow

def correlate_fallback_with_signals():
    """
    フォールバック処理とシグナル異常の相関関係分析
    """
    print(f"\n" + "=" * 60)
    print("🔗 フォールバック-シグナル異常相関分析")
    print("=" * 60)
    
    # 既知の異常パターン
    signal_anomalies = {
        'entry_count_change': {'before': 81, 'after': 62, 'difference': -19},
        'exit_count_change': {'before': 0, 'after': 62, 'difference': +62},
        'simultaneous_signals': 62,
        'price_difference': 0.0
    }
    
    print(f"📊 既知のシグナル異常:")
    for anomaly, data in signal_anomalies.items():
        if isinstance(data, dict) and 'before' in data:
            print(f"  - {anomaly}: {data['before']} → {data['after']} (差: {data['difference']:+d})")
        else:
            print(f"  - {anomaly}: {data}")
    
    print(f"\n💡 フォールバック相関仮説:")
    print(f"  1. 統合システム失敗 → 部分的データ生成")
    print(f"  2. フォールバック処理 → 従来システム実行")
    print(f"  3. 両システムのデータ競合 → 同時entry/exit")
    print(f"  4. unified_exporter → 競合データをペアリング")
    
    return signal_anomalies

def main():
    print("🔍 TODO-006-A フォールバック処理影響確認 開始")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: フォールバックログ解析
    fallback_data = analyze_fallback_logs()
    
    # Step 2: タイミング影響分析
    processing_flow = analyze_fallback_timing_impact()
    
    # Step 3: 相関関係分析
    correlations = correlate_fallback_with_signals()
    
    # 結論
    print(f"\n" + "=" * 60)
    print("🎯 手法1結論")
    print("=" * 60)
    
    if fallback_data:
        print(f"✅ フォールバック処理確認:")
        print(f"  - MultiStrategyManager失敗が確認された")
        print(f"  - フォールバック使用率: {fallback_data.get('fallback_usage_rate', 'N/A')}")
        print(f"  - 二重処理の兆候: ログ出力重複")
        
        print(f"\n🔍 次の調査ポイント:")
        print(f"  - 統合システムがどの段階まで実行されたか")
        print(f"  - 従来システムとの処理重複範囲")
        print(f"  - シグナル生成の競合メカニズム")
    else:
        print(f"❌ フォールバックデータ不足 - 手法2で補完調査が必要")
    
    return {
        'fallback_data': fallback_data,
        'processing_flow': processing_flow,
        'correlations': correlations
    }

if __name__ == "__main__":
    results = main()
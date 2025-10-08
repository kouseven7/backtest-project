#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Simple Ranking System - Problem 19診断用
目的: DSSMSの複雑性を排除して切替の基本動作を確認
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# ログイン設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraSimpleRanking:
    """Problem 19専用・超シンプル版ランキングシステム
    
    特徴:
    - top_symbol=None不可能な設計
    - 決定論的動作（常に同じ結果）
    - 外部依存なし（yfinance等不使用）
    - 複雑な統合ロジックなし
    """
    
    def __init__(self, symbols: List[str]):
        """初期化
        
        Args:
            symbols: 銘柄リスト（例: ['6758', '7203', '8306', '9984']）
        """
        if not symbols:
            raise ValueError("銘柄リストが空です")
            
        self.symbols = symbols
        self.current_index = 0  # 現在選択中の銘柄インデックス
        self.switch_history = []  # 切替履歴
        
        logger.info(f"UltraSimpleRanking初期化: {len(symbols)}銘柄")
        logger.info(f"銘柄: {symbols}")
        
    def get_current_symbol(self) -> str:
        """現在の銘柄を取得（必ずNone以外を返す）"""
        current = self.symbols[self.current_index % len(self.symbols)]
        return current
        
    def should_switch(self, day: int) -> bool:
        """切替判定（10日ごとに切替）
        
        Args:
            day: 経過日数（1から開始）
            
        Returns:
            True: 切替する, False: 切替しない
        """
        # Day 10, 20, 30, ... で切替
        return day % 10 == 0
        
    def switch_to_next(self, day: int) -> str:
        """次の銘柄に切替
        
        Args:
            day: 現在の日数
            
        Returns:
            切替後の銘柄（必ずNone以外）
        """
        old_symbol = self.get_current_symbol()
        self.current_index = (self.current_index + 1) % len(self.symbols)
        new_symbol = self.get_current_symbol()
        
        # 切替履歴記録
        switch_record = {
            'day': day,
            'from_symbol': old_symbol,
            'to_symbol': new_symbol,
            'reason': f'{day}日目・定期切替'
        }
        self.switch_history.append(switch_record)
        
        logger.info(f"Day {day}: {old_symbol} → {new_symbol}")
        return new_symbol
        
    def simulate_switches(self, total_days: int = 100) -> Dict[str, Any]:
        """指定日数分の切替をシミュレート
        
        Args:
            total_days: シミュレーション日数（1からtotal_daysまで）
            
        Returns:
            シミュレーション結果
        """
        logger.info(f"=== {total_days}日間シミュレーション開始 ===")
        
        results = {
            'total_days': total_days,
            'switches': [],
            'daily_symbols': [],
            'switch_count': 0,
            'success': True
        }
        
        try:
            # Day 1 から total_days まで（30日なら1-30）
            for day in range(1, total_days + 1):
                # 切替判定
                if self.should_switch(day):
                    new_symbol = self.switch_to_next(day)
                    results['switches'].append({
                        'day': day,
                        'symbol': new_symbol
                    })
                    results['switch_count'] += 1
                
                # 日次銘柄記録
                current = self.get_current_symbol()
                results['daily_symbols'].append({
                    'day': day,
                    'symbol': current
                })
                
                # None値チェック（絶対に発生してはいけない）
                if current is None:
                    logger.error(f"CRITICAL: Day {day}でNone値検出！")
                    results['success'] = False
                    break
                    
        except Exception as e:
            logger.error(f"シミュレーション失敗: {e}")
            results['success'] = False
            results['error'] = str(e)
            
        logger.info(f"=== シミュレーション完了 ===")
        logger.info(f"総切替回数: {results['switch_count']}")
        logger.info(f"成功: {results['success']}")
        
        return results

def test_ultra_simple_ranking():
    """UltraSimpleRankingの基本動作テスト"""
    
    print("[TOOL] Ultra Simple Ranking Test 開始")
    print("=" * 50)
    
    # テスト用銘柄リスト（DSSMSと同じ）
    symbols = ['6758', '7203', '8306', '9984']
    
    try:
        # Step 1: 初期化テスト
        print("Step 1: 初期化テスト")
        ranking = UltraSimpleRanking(symbols)
        initial_symbol = ranking.get_current_symbol()
        print(f"  初期銘柄: {initial_symbol}")
        assert initial_symbol is not None, "初期銘柄がNoneです！"
        print("  [OK] 初期化成功")
        
        # Step 2: 基本切替テスト（短期間）
        print("\nStep 2: 基本切替テスト（30日間）")
        short_results = ranking.simulate_switches(30)
        print(f"  期待切替回数: 3回 (Day 10,20,30)")
        print(f"  実際切替回数: {short_results['switch_count']}回")
        print(f"  成功: {short_results['success']}")
        assert short_results['success'], "短期間テスト失敗"
        assert short_results['switch_count'] == 3, f"期待3回、実際{short_results['switch_count']}回"
        print("  [OK] 基本切替テスト成功")
        
        # Step 3: 長期間テスト（DSSMS相当）
        print("\nStep 3: 長期間テスト（100日間）")
        ranking2 = UltraSimpleRanking(symbols)  # 新しいインスタンス
        long_results = ranking2.simulate_switches(100)
        print(f"  期待切替回数: 10回 (Day 10,20,30,...,100)")
        print(f"  実際切替回数: {long_results['switch_count']}回")
        print(f"  成功: {long_results['success']}")
        assert long_results['success'], "長期間テスト失敗"
        assert long_results['switch_count'] == 10, f"期待10回、実際{long_results['switch_count']}回"
        print("  [OK] 長期間テスト成功")
        
        # Step 4: None値検証
        print("\nStep 4: None値絶対阻止テスト")
        none_count = 0
        for daily in long_results['daily_symbols']:
            if daily['symbol'] is None:
                none_count += 1
                
        print(f"  None値検出回数: {none_count}回")
        assert none_count == 0, f"None値が{none_count}回検出されました！"
        print("  [OK] None値阻止テスト成功")
        
        # Step 5: 決定論的動作テスト
        print("\nStep 5: 決定論的動作テスト（再現性確認）")
        ranking3 = UltraSimpleRanking(symbols)
        repeat_results = ranking3.simulate_switches(100)
        
        # 結果比較
        same_switch_count = (long_results['switch_count'] == repeat_results['switch_count'])
        same_pattern = True
        for i, (original, repeat) in enumerate(zip(long_results['daily_symbols'], repeat_results['daily_symbols'])):
            if original['symbol'] != repeat['symbol']:
                same_pattern = False
                print(f"    Day {i}: {original['symbol']} vs {repeat['symbol']}")
                break
                
        print(f"  切替回数一致: {same_switch_count}")
        print(f"  銘柄パターン一致: {same_pattern}")
        assert same_switch_count and same_pattern, "決定論的動作が保証されていません"
        print("  [OK] 決定論的動作テスト成功")
        
        print("\n" + "=" * 50)
        print("[SUCCESS] Ultra Simple Ranking Test 全テスト成功！")
        print("=" * 50)
        
        # 結果サマリー
        print(f"\n[CHART] テスト結果サマリー:")
        print(f"  - 基本動作: [OK] 正常")
        print(f"  - 切替機能: [OK] 正常（期待通り10回）")
        print(f"  - None値阻止: [OK] 正常（0回検出）")
        print(f"  - 決定論的動作: [OK] 正常（完全再現）")
        print(f"  - 外部依存: [OK] なし")
        
        return {
            'test_success': True,
            'basic_switch_works': True,
            'none_prevention_works': True,
            'deterministic_works': True,
            'switch_count_accurate': True
        }
        
    except AssertionError as e:
        print(f"\n[ERROR] テスト失敗: {e}")
        return {
            'test_success': False,
            'error': str(e)
        }
    except Exception as e:
        print(f"\n💥 予期しないエラー: {e}")
        return {
            'test_success': False,
            'error': str(e),
            'unexpected_error': True
        }

if __name__ == "__main__":
    # メインテスト実行
    print("Problem 19 - Ultra Simple Ranking Diagnosis")
    print("目的: 切替の基本動作確認・問題の深度測定")
    print("日時:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    test_results = test_ultra_simple_ranking()
    
    print(f"\n[TARGET] 診断結果:")
    if test_results['test_success']:
        print("[OK] 超シンプル版は正常動作")
        print("➡️  問題はDSSMSの複雑性に起因する可能性が高い")
        print("➡️  局所的修正で解決可能と判断")
    else:
        print("[ERROR] 超シンプル版でも問題発生")
        print("➡️  より根本的な問題が存在")
        print("➡️  環境・Python・OS レベルの調査が必要")
        
    print(f"\n次のステップ:")
    if test_results['test_success']:
        print("1. DSSMSとの差分特定")
        print("2. 問題箇所の局所修正")
        print("3. 段階的統合テスト")
    else:
        print("1. 実行環境の詳細調査")
        print("2. Python・ライブラリ依存関係確認")
        print("3. より基本的なシステム診断")
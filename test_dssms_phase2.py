"""
DSSMS Phase 2 (階層的ランキングシステム) テストスイート
Task 2.1: 階層的銘柄ランキングシステムのユニットテスト
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import tempfile
import os

# テスト対象モジュール
from src.dssms.hierarchical_ranking_system import (
    HierarchicalRankingSystem,
    DSSMSRankingIntegrator,
    PriorityLevel,
    RankingScore,
    SelectionResult
)

class TestHierarchicalRankingSystem(unittest.TestCase):
    """階層的ランキングシステムのテスト"""
    
    def setUp(self):
        """テスト準備"""
        self.test_config = {
            "ranking_system": {
                "scoring_weights": {
                    "fundamental": 0.40,
                    "technical": 0.30,
                    "volume": 0.20,
                    "volatility": 0.10
                },
                "priority_classification": {
                    "adaptive_priority_levels": 3,
                    "perfect_order_weight": 0.35
                },
                "affordability_penalty": {
                    "high_price_threshold": 5000,
                    "penalty_rate": 0.15
                }
            }
        }
        
        # モックのセットアップ
        with patch('src.dssms.hierarchical_ranking_system.PerfectOrderDetector'), \
             patch('src.dssms.hierarchical_ranking_system.FundamentalAnalyzer'), \
             patch('src.dssms.hierarchical_ranking_system.DSSMSDataManager'), \
             patch('src.dssms.hierarchical_ranking_system.Nikkei225Screener'):
            
            self.ranking_system = HierarchicalRankingSystem(self.test_config)
        
        # テスト用データ
        self.test_symbols = ["7203", "6758", "9984", "8035", "9432"]
        self.test_funds = 10_000_000  # 1000万円
    
    def test_categorize_by_perfect_order_priority(self):
        """パーフェクトオーダー優先度分類テスト"""
        # モック設定
        mock_po_results = {
            "7203": {  # レベル1: 全軸パーフェクトオーダー
                'daily_perfect_order': True,
                'weekly_perfect_order': True,
                'monthly_perfect_order': True
            },
            "6758": {  # レベル2: 月週軸パーフェクトオーダー
                'daily_perfect_order': False,
                'weekly_perfect_order': True,
                'monthly_perfect_order': True
            },
            "9984": {  # レベル3: その他
                'daily_perfect_order': True,
                'weekly_perfect_order': False,
                'monthly_perfect_order': False
            },
            "8035": None,  # レベル3: データなし
            "9432": {  # レベル3: パーフェクトオーダーなし
                'daily_perfect_order': False,
                'weekly_perfect_order': False,
                'monthly_perfect_order': False
            }
        }
        
        self.ranking_system.perfect_order_detector.check_multi_timeframe_perfect_order.side_effect = \
            lambda symbol: mock_po_results.get(symbol)
        
        # テスト実行
        priority_groups = self.ranking_system.categorize_by_perfect_order_priority(self.test_symbols)
        
        # 結果検証
        self.assertEqual(priority_groups[1], ["7203"])  # レベル1
        self.assertEqual(priority_groups[2], ["6758"])  # レベル2
        self.assertEqual(set(priority_groups[3]), {"9984", "8035", "9432"})  # レベル3
    
    def test_rank_within_priority_group(self):
        """同一優先度グループ内ランキングテスト"""
        # 総合スコア計算のモック
        mock_scores = {
            "7203": self._create_mock_ranking_score("7203", 0.85),
            "6758": self._create_mock_ranking_score("6758", 0.78),
            "9984": self._create_mock_ranking_score("9984", 0.72)
        }
        
        self.ranking_system._calculate_comprehensive_score = \
            lambda symbol: mock_scores.get(symbol)
        
        # テスト実行
        ranked_symbols = self.ranking_system.rank_within_priority_group(list(mock_scores.keys()))
        
        # 結果検証（スコア降順）
        self.assertEqual(ranked_symbols[0][0], "7203")  # 最高スコア
        self.assertEqual(ranked_symbols[1][0], "6758")  # 2番目
        self.assertEqual(ranked_symbols[2][0], "9984")  # 3番目
        
        # スコア値検証
        self.assertAlmostEqual(ranked_symbols[0][1], 0.85, places=2)
        self.assertAlmostEqual(ranked_symbols[1][1], 0.78, places=2)
        self.assertAlmostEqual(ranked_symbols[2][1], 0.72, places=2)
    
    def test_get_top_candidate_success(self):
        """最適候補選択成功テスト"""
        # モックセットアップ
        self._setup_comprehensive_mocks()
        
        # 購入可能性をTrue
        self.ranking_system._check_affordability = Mock(return_value=True)
        
        # テスト実行
        result = self.ranking_system.get_top_candidate(self.test_funds)
        
        # 結果検証
        self.assertIsNotNone(result)
        self.assertIn(result, self.test_symbols)
    
    def test_get_top_candidate_affordability_check(self):
        """購入可能性チェックテスト"""
        # モックセットアップ
        self._setup_comprehensive_mocks()
        
        # 高優先度銘柄は購入不可、低優先度銘柄は購入可能
        def mock_affordability(symbol, funds):
            high_priority = ["7203", "6758"]  # 購入不可
            return symbol not in high_priority
        
        self.ranking_system._check_affordability = mock_affordability
        
        # テスト実行
        result = self.ranking_system.get_top_candidate(self.test_funds)
        
        # 結果検証（低優先度から選択される）
        self.assertIsNotNone(result)
        self.assertNotIn(result, ["7203", "6758"])
    
    def test_get_backup_candidates(self):
        """バックアップ候補生成テスト"""
        # モックセットアップ
        self._setup_comprehensive_mocks()
        
        # テスト実行
        backup_count = 3
        backup_candidates = self.ranking_system.get_backup_candidates(backup_count)
        
        # 結果検証
        self.assertLessEqual(len(backup_candidates), backup_count)
        self.assertTrue(all(symbol in self.test_symbols for symbol in backup_candidates))
        
        # 重複チェック
        self.assertEqual(len(backup_candidates), len(set(backup_candidates)))
    
    def test_get_selection_result_comprehensive(self):
        """統合選択結果テスト"""
        # モックセットアップ
        self._setup_comprehensive_mocks()
        self.ranking_system._check_affordability = Mock(return_value=True)
        
        # テスト実行
        result = self.ranking_system.get_selection_result(
            available_funds=self.test_funds,
            backup_count=3
        )
        
        # 結果検証
        self.assertIsInstance(result, SelectionResult)
        self.assertIsNotNone(result.primary_candidate)
        self.assertLessEqual(len(result.backup_candidates), 3)
        self.assertGreater(result.total_candidates_evaluated, 0)
        self.assertIsInstance(result.priority_distribution, dict)
        self.assertGreater(len(result.selection_reason), 0)
    
    def test_technical_score_calculation(self):
        """テクニカルスコア計算テスト"""
        # テスト用価格データ作成
        test_data = self._create_test_price_data()
        
        # データマネージャーモック
        self.ranking_system.data_manager.get_daily_data = Mock(return_value=test_data)
        
        # テスト実行
        technical_score = self.ranking_system._calculate_technical_score("7203")
        
        # 結果検証
        self.assertGreaterEqual(technical_score, 0.0)
        self.assertLessEqual(technical_score, 1.0)
        self.assertIsInstance(technical_score, float)
    
    def test_volume_score_calculation(self):
        """出来高スコア計算テスト"""
        # テスト用出来高データ
        test_data = self._create_test_price_data()
        test_data['Volume'] = np.random.randint(100000, 1000000, len(test_data))
        
        self.ranking_system.data_manager.get_daily_data = Mock(return_value=test_data)
        
        # テスト実行
        volume_score = self.ranking_system._calculate_volume_score("7203")
        
        # 結果検証
        self.assertGreaterEqual(volume_score, 0.0)
        self.assertLessEqual(volume_score, 1.0)
    
    def test_volatility_score_calculation(self):
        """ボラティリティスコア計算テスト"""
        # テスト用データ
        test_data = self._create_test_price_data()
        
        self.ranking_system.data_manager.get_daily_data = Mock(return_value=test_data)
        
        # テスト実行
        volatility_score = self.ranking_system._calculate_volatility_score("7203")
        
        # 結果検証
        self.assertGreaterEqual(volatility_score, 0.0)
        self.assertLessEqual(volatility_score, 1.0)
    
    def test_perfect_order_score_calculation(self):
        """パーフェクトオーダースコア計算テスト"""
        # モックデータ
        mock_po_result = {
            'daily_strength': 0.8,
            'weekly_strength': 0.9,
            'monthly_strength': 0.7
        }
        
        self.ranking_system.perfect_order_detector.check_multi_timeframe_perfect_order = \
            Mock(return_value=mock_po_result)
        
        # テスト実行
        po_score = self.ranking_system._calculate_perfect_order_score("7203")
        
        # 結果検証
        self.assertGreaterEqual(po_score, 0.0)
        self.assertLessEqual(po_score, 1.0)
        
        # 加重平均の検証
        expected_score = 0.8 * 0.3 + 0.9 * 0.4 + 0.7 * 0.3
        self.assertAlmostEqual(po_score, expected_score, places=2)
    
    def test_affordability_check(self):
        """購入可能性チェックテスト"""
        # 価格データモック
        mock_price_data = {"Close": 3000}  # 3000円
        self.ranking_system.data_manager.get_latest_price = Mock(return_value=mock_price_data)
        
        # テストケース1: 購入可能
        available_funds = 1_000_000  # 100万円
        result = self.ranking_system._check_affordability("7203", available_funds)
        self.assertTrue(result)  # 3000円×100株=30万円 < 100万円×0.8
        
        # テストケース2: 購入不可
        available_funds = 200_000  # 20万円
        result = self.ranking_system._check_affordability("7203", available_funds)
        self.assertFalse(result)  # 30万円 > 20万円×0.8
    
    def test_confidence_level_calculation(self):
        """信頼度計算テスト"""
        # テストケース1: 高信頼度（スコアが高く、分散が小さい）
        high_confidence = self.ranking_system._calculate_confidence_level(0.8, 0.85, 0.9, 0.82)
        
        # テストケース2: 低信頼度（スコアが低く、分散が大きい）
        low_confidence = self.ranking_system._calculate_confidence_level(0.3, 0.8, 0.1, 0.9)
        
        # 結果検証
        self.assertGreater(high_confidence, low_confidence)
        self.assertGreaterEqual(high_confidence, 0.1)
        self.assertLessEqual(high_confidence, 1.0)
    
    def test_rsi_calculation(self):
        """RSI計算テスト"""
        # テスト用価格データ
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 
                           111, 110, 112, 114, 113, 115, 117, 116, 118, 120])
        
        # RSI計算
        rsi = self.ranking_system._calculate_rsi(prices, period=14)
        
        # 結果検証
        self.assertFalse(rsi.isna().all())  # NaNでない
        last_rsi = rsi.iloc[-1]
        self.assertGreaterEqual(last_rsi, 0)
        self.assertLessEqual(last_rsi, 100)
    
    def test_macd_calculation(self):
        """MACD計算テスト"""
        # テスト用価格データ
        prices = pd.Series(np.random.randn(50).cumsum() + 100)
        
        # MACD計算
        macd_line, signal_line = self.ranking_system._calculate_macd(prices)
        
        # 結果検証
        self.assertEqual(len(macd_line), len(prices))
        self.assertEqual(len(signal_line), len(prices))
        self.assertFalse(macd_line.isna().all())
        self.assertFalse(signal_line.isna().all())
    
    # === ヘルパーメソッド ===
    
    def _create_mock_ranking_score(self, symbol: str, total_score: float) -> RankingScore:
        """モック用ランキングスコア作成"""
        return RankingScore(
            symbol=symbol,
            total_score=total_score,
            perfect_order_score=0.8,
            fundamental_score=0.7,
            technical_score=0.6,
            volume_score=0.5,
            volatility_score=0.4,
            priority_group=1,
            confidence_level=0.75,
            affordability_penalty=0.0,
            last_updated=datetime.now()
        )
    
    def _create_test_price_data(self, days: int = 100) -> pd.DataFrame:
        """テスト用価格データ作成"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        # ランダムウォークベースの価格データ
        np.random.seed(42)  # 再現性のため
        returns = np.random.normal(0.001, 0.02, days)
        prices = 1000 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, days)),
            'High': prices * (1 + np.abs(np.random.normal(0.01, 0.005, days))),
            'Low': prices * (1 - np.abs(np.random.normal(0.01, 0.005, days))),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, days)
        }, index=dates)
        
        return data
    
    def _setup_comprehensive_mocks(self):
        """包括的モックセットアップ"""
        # スクリーニング結果
        self.ranking_system._get_screened_symbols = Mock(return_value=self.test_symbols)
        
        # パーフェクトオーダー結果
        mock_po_results = {
            "7203": {'daily_perfect_order': True, 'weekly_perfect_order': True, 'monthly_perfect_order': True},
            "6758": {'daily_perfect_order': False, 'weekly_perfect_order': True, 'monthly_perfect_order': True},
            "9984": {'daily_perfect_order': True, 'weekly_perfect_order': False, 'monthly_perfect_order': False},
            "8035": {'daily_perfect_order': False, 'weekly_perfect_order': False, 'monthly_perfect_order': False},
            "9432": {'daily_perfect_order': False, 'weekly_perfect_order': False, 'monthly_perfect_order': False}
        }
        self.ranking_system.perfect_order_detector.check_multi_timeframe_perfect_order.side_effect = \
            lambda symbol: mock_po_results.get(symbol)
        
        # 総合スコア
        mock_scores = {
            symbol: self._create_mock_ranking_score(symbol, 0.8 - i * 0.1)
            for i, symbol in enumerate(self.test_symbols)
        }
        self.ranking_system._calculate_comprehensive_score = \
            lambda symbol: mock_scores.get(symbol)


class TestDSSMSRankingIntegrator(unittest.TestCase):
    """DSSMSランキング統合システムテスト"""
    
    def setUp(self):
        """テスト準備"""
        # 一時設定ファイル作成
        self.temp_config = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        
        test_config = {
            "ranking_system": {
                "scoring_weights": {
                    "fundamental": 0.40,
                    "technical": 0.30,
                    "volume": 0.20,
                    "volatility": 0.10
                }
            }
        }
        
        json.dump(test_config, self.temp_config)
        self.temp_config.close()
        
        # 統合システム初期化
        with patch('src.dssms.hierarchical_ranking_system.HierarchicalRankingSystem'):
            self.integrator = DSSMSRankingIntegrator(self.temp_config.name)
    
    def tearDown(self):
        """テスト後処理"""
        os.unlink(self.temp_config.name)
    
    def test_config_loading(self):
        """設定ファイル読み込みテスト"""
        # 設定が正しく読み込まれているか確認
        self.assertIn('ranking_system', self.integrator.config)
        self.assertIn('scoring_weights', self.integrator.config['ranking_system'])
    
    def test_execute_full_ranking_process(self):
        """完全ランキングプロセステスト"""
        # モック結果
        mock_result = SelectionResult(
            primary_candidate="7203",
            backup_candidates=["6758", "9984"],
            selection_reason="全時間軸パーフェクトオーダー銘柄 7203 を選択",
            available_fund_ratio=0.3,
            total_candidates_evaluated=5,
            priority_distribution={1: 1, 2: 1, 3: 3}
        )
        
        self.integrator.ranking_system.get_selection_result = Mock(return_value=mock_result)
        
        # テスト実行
        result = self.integrator.execute_full_ranking_process(10_000_000)
        
        # 結果検証
        self.assertEqual(result.primary_candidate, "7203")
        self.assertEqual(len(result.backup_candidates), 2)
    
    def test_get_ranking_summary(self):
        """ランキングサマリーテスト"""
        # モック結果
        mock_result = SelectionResult(
            primary_candidate="7203",
            backup_candidates=["6758", "9984"],
            selection_reason="テスト選択",
            available_fund_ratio=0.3,
            total_candidates_evaluated=5,
            priority_distribution={1: 1, 2: 1, 3: 3}
        )
        
        self.integrator.ranking_system.get_selection_result = Mock(return_value=mock_result)
        
        # テスト実行
        summary = self.integrator.get_ranking_summary(10_000_000)
        
        # 結果検証
        required_keys = [
            'execution_timestamp', 'primary_candidate', 'backup_candidates',
            'selection_reason', 'available_funds', 'fund_utilization_ratio',
            'total_evaluated', 'priority_distribution', 'system_status'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary)
        
        self.assertEqual(summary['system_status'], 'SUCCESS')
        self.assertEqual(summary['primary_candidate'], '7203')
    
    def test_default_config_fallback(self):
        """デフォルト設定フォールバックテスト"""
        # 存在しない設定ファイルで初期化
        with patch('src.dssms.hierarchical_ranking_system.HierarchicalRankingSystem'):
            integrator = DSSMSRankingIntegrator("nonexistent_config.json")
        
        # デフォルト設定が使用されているか確認
        self.assertIn('ranking_system', integrator.config)
        default_weights = integrator.config['ranking_system']['scoring_weights']
        self.assertAlmostEqual(default_weights['fundamental'], 0.40)
        self.assertAlmostEqual(default_weights['technical'], 0.30)


class TestDataClassesAndEnums(unittest.TestCase):
    """データクラスと列挙型のテスト"""
    
    def test_priority_level_enum(self):
        """優先度レベル列挙型テスト"""
        self.assertEqual(PriorityLevel.LEVEL_1.value, 1)
        self.assertEqual(PriorityLevel.LEVEL_2.value, 2)
        self.assertEqual(PriorityLevel.LEVEL_3.value, 3)
    
    def test_ranking_score_dataclass(self):
        """ランキングスコアデータクラステスト"""
        score = RankingScore(
            symbol="7203",
            total_score=0.85,
            perfect_order_score=0.9,
            fundamental_score=0.8,
            technical_score=0.7,
            volume_score=0.6,
            volatility_score=0.5,
            priority_group=1,
            confidence_level=0.75,
            affordability_penalty=0.05,
            last_updated=datetime.now()
        )
        
        self.assertEqual(score.symbol, "7203")
        self.assertAlmostEqual(score.total_score, 0.85)
        self.assertEqual(score.priority_group, 1)
    
    def test_selection_result_dataclass(self):
        """選択結果データクラステスト"""
        result = SelectionResult(
            primary_candidate="7203",
            backup_candidates=["6758", "9984"],
            selection_reason="テスト選択",
            available_fund_ratio=0.3,
            total_candidates_evaluated=5,
            priority_distribution={1: 1, 2: 1, 3: 3}
        )
        
        self.assertEqual(result.primary_candidate, "7203")
        self.assertEqual(len(result.backup_candidates), 2)
        self.assertIn("6758", result.backup_candidates)


if __name__ == '__main__':
    # テストスイート実行
    unittest.main(verbosity=2)

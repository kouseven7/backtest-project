"""
Data Aggregation System for 4-3-1 Trend Strategy Time Series Visualization

既存システムからのデータ集約・前処理・品質チェック機能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import sys
import os

# プロジェクトルートを取得してパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class VisualizationDataAggregator:
    """可視化用データ集約クラス"""
    
    def __init__(self, symbol: str = "USDJPY", period_days: int = 30):
        self.symbol = symbol
        self.period_days = period_days
        self.logger = self._setup_logger()
        
        # データソース参照
        self.data_sources: Dict[str, Optional[pd.DataFrame]] = {
            'price_data': None,
            'trend_data': None,
            'strategy_data': None,
            'confidence_data': None,
            'volume_data': None
        }
        
        # 集約済みデータ
        self.aggregated_data = None
        self.data_quality_score = 0.0
    
    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger(f"DataAggregator_{self.symbol}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_price_data(self) -> pd.DataFrame:
        """価格データ読み込み"""
        try:
            # 既存のdata_fetcher.pyを使用（パス調整）
            try:
                from data_fetcher import get_parameters_and_data
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.period_days + 10)  # 余裕をもたせる
                
                _, _, _, stock_data, _ = get_parameters_and_data(
                    ticker=self.symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if stock_data is not None and not stock_data.empty:
                    # 期間でフィルタリング
                    data = stock_data.tail(self.period_days)
                    self.data_sources['price_data'] = data
                    self.logger.info(f"価格データ読み込み成功: {len(data)}レコード")
                    return data
                else:
                    raise ValueError("価格データが空です")
                    
            except ImportError:
                self.logger.warning("data_fetcher モジュールが見つかりません")
                raise ValueError("データソースが利用できません")
                
        except Exception as e:
            self.logger.warning(f"価格データ読み込み失敗: {e}")
            return self._generate_synthetic_price_data()
    
    def load_trend_data(self) -> pd.DataFrame:
        """トレンドデータ読み込み"""
        try:
            # 既存のunified_trend_detector.pyを使用を試行
            try:
                from indicators.unified_trend_detector import UnifiedTrendDetector
                
                detector = UnifiedTrendDetector()
                
                # 価格データが必要
                if self.data_sources['price_data'] is None:
                    self.load_price_data()
                
                price_data = self.data_sources['price_data']
                if price_data is not None:
                    trend_results = detector.analyze_comprehensive_trend(price_data)
                    
                    if trend_results and 'trend_periods' in trend_results:
                        trend_df = self._convert_trend_to_dataframe(
                            trend_results['trend_periods'], 
                            price_data.index
                        )
                        self.data_sources['trend_data'] = trend_df
                        self.logger.info(f"トレンドデータ処理成功: {len(trend_df)}レコード")
                        return trend_df
                    else:
                        raise ValueError("トレンド分析結果が無効です")
                else:
                    raise ValueError("価格データが無効です")
                    
            except ImportError:
                self.logger.warning("unified_trend_detector モジュールが見つかりません")
                raise ValueError("トレンド検出器が利用できません")
                
        except Exception as e:
            self.logger.warning(f"トレンドデータ読み込み失敗: {e}")
            return self._generate_synthetic_trend_data()
    
    def load_strategy_data(self) -> pd.DataFrame:
        """戦略データ読み込み"""
        try:
            # 既存のmulti_strategy_coordination_manager.pyを使用を試行
            try:
                from config.multi_strategy_coordination_manager import MultiStrategyCoordinationManager
                
                manager = MultiStrategyCoordinationManager()
                
                # 価格データが必要
                if self.data_sources['price_data'] is None:
                    self.load_price_data()
                
                price_data = self.data_sources['price_data']
                if price_data is not None:
                    strategy_results = []
                    for i, row in price_data.iterrows():
                        # 戦略選択を実行
                        strategy_decision = manager.select_optimal_strategy({
                            'price': row['Close'],
                            'timestamp': i,
                            'market_data': row.to_dict()
                        })
                        
                        strategy_results.append({
                            'timestamp': i,
                            'strategy': strategy_decision.get('selected_strategy', 'unknown'),
                            'confidence': strategy_decision.get('confidence', 0.5),
                            'reason': strategy_decision.get('reason', 'N/A')
                        })
                    
                    strategy_df = pd.DataFrame(strategy_results)
                    strategy_df.set_index('timestamp', inplace=True)
                    
                    self.data_sources['strategy_data'] = strategy_df
                    self.logger.info(f"戦略データ処理成功: {len(strategy_df)}レコード")
                    return strategy_df
                else:
                    raise ValueError("価格データが無効です")
                    
            except ImportError:
                self.logger.warning("multi_strategy_coordination_manager モジュールが見つかりません")
                raise ValueError("戦略管理システムが利用できません")
            
        except Exception as e:
            self.logger.warning(f"戦略データ読み込み失敗: {e}")
            return self._generate_synthetic_strategy_data()
    
    def load_confidence_data(self) -> pd.DataFrame:
        """信頼度データ読み込み"""
        try:
            # 既存のconfidence_threshold_manager.pyを使用を試行
            try:
                from config.confidence_threshold_manager import ConfidenceThresholdManager
                
                manager = ConfidenceThresholdManager()
                
                # 戦略データが必要
                if self.data_sources['strategy_data'] is None:
                    self.load_strategy_data()
                
                strategy_data = self.data_sources['strategy_data']
                if strategy_data is not None:
                    confidence_results = []
                    for i, row in strategy_data.iterrows():
                        confidence_level = manager.calculate_confidence_level({
                            'strategy': row['strategy'],
                            'market_conditions': 'normal',  # 簡略化
                            'timestamp': i
                        })
                        
                        confidence_results.append({
                            'timestamp': i,
                            'confidence_score': confidence_level.get('score', 0.5),
                            'confidence_level': confidence_level.get('level', 'medium'),
                            'factors': confidence_level.get('factors', [])
                        })
                    
                    confidence_df = pd.DataFrame(confidence_results)
                    confidence_df.set_index('timestamp', inplace=True)
                    
                    self.data_sources['confidence_data'] = confidence_df
                    self.logger.info(f"信頼度データ処理成功: {len(confidence_df)}レコード")
                    return confidence_df
                else:
                    raise ValueError("戦略データが無効です")
                    
            except ImportError:
                self.logger.warning("confidence_threshold_manager モジュールが見つかりません")
                raise ValueError("信頼度管理システムが利用できません")
            
        except Exception as e:
            self.logger.warning(f"信頼度データ読み込み失敗: {e}")
            return self._generate_synthetic_confidence_data()
    
    def load_volume_data(self) -> pd.DataFrame:
        """ボリュームデータ読み込み"""
        try:
            # 価格データからボリューム情報を取得
            if self.data_sources['price_data'] is None:
                self.load_price_data()
            
            price_data = self.data_sources['price_data']
            
            if 'Volume' in price_data.columns:
                volume_df = price_data[['Volume']].copy()
                volume_df['volume_ma'] = volume_df['Volume'].rolling(window=5).mean()
                volume_df['volume_ratio'] = volume_df['Volume'] / volume_df['volume_ma']
            else:
                # ボリュームデータがない場合は合成データ
                volume_df = self._generate_synthetic_volume_data()
            
            self.data_sources['volume_data'] = volume_df
            self.logger.info(f"ボリュームデータ処理成功: {len(volume_df)}レコード")
            return volume_df
            
        except Exception as e:
            self.logger.warning(f"ボリュームデータ読み込み失敗: {e}")
            return self._generate_synthetic_volume_data()
    
    def aggregate_all_data(self) -> pd.DataFrame:
        """全データを集約"""
        try:
            # 各データソースを読み込み
            price_df = self.load_price_data()
            trend_df = self.load_trend_data()
            strategy_df = self.load_strategy_data()
            confidence_df = self.load_confidence_data()
            volume_df = self.load_volume_data()
            
            # データをマージ
            merged_df = price_df.copy()
            
            # ボリュームデータがある場合は価格データから削除
            if 'Volume' in merged_df.columns and 'Volume' in volume_df.columns:
                merged_df = merged_df.drop(columns=['Volume'])
            
            # トレンドデータをマージ
            merged_df = merged_df.join(trend_df, how='left')
            
            # 戦略データをマージ  
            merged_df = merged_df.join(strategy_df, how='left')
            
            # 信頼度データをマージ
            merged_df = merged_df.join(confidence_df, how='left')
            
            # ボリュームデータをマージ
            merged_df = merged_df.join(volume_df, how='left')
            
            # NaN値の処理
            merged_df = self._clean_merged_data(merged_df)
            
            # データ品質評価
            self.data_quality_score = self._evaluate_data_quality(merged_df)
            
            self.aggregated_data = merged_df
            self.logger.info(f"データ集約完了: {len(merged_df)}レコード, 品質スコア: {self.data_quality_score:.2f}")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"データ集約失敗: {e}")
            # フォールバック：完全合成データ
            return self._generate_complete_synthetic_data()
    
    def _convert_trend_to_dataframe(self, trend_periods: List, index: pd.DatetimeIndex) -> pd.DataFrame:
        """トレンド期間データをDataFrameに変換"""
        trend_series = pd.Series('sideways', index=index, name='trend_type')
        
        for period in trend_periods:
            start_idx = period.get('start_index', 0)
            end_idx = period.get('end_index', len(index) - 1)
            trend_type = period.get('trend_type', 'sideways')
            
            if start_idx < len(index) and end_idx < len(index):
                trend_series.iloc[start_idx:end_idx+1] = trend_type
        
        trend_df = pd.DataFrame({
            'trend_type': trend_series,
            'trend_strength': np.random.uniform(0.3, 0.9, len(index))  # 簡略化
        })
        
        return trend_df
    
    def _clean_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """マージされたデータのクリーニング"""
        # 基本的なNaN補完
        if 'trend_type' in df.columns:
            df['trend_type'].fillna('sideways', inplace=True)
        
        if 'strategy' in df.columns:
            df['strategy'].fillna('trend_following', inplace=True)
        
        if 'confidence_score' in df.columns:
            df['confidence_score'].fillna(0.5, inplace=True)
        
        if 'confidence_level' in df.columns:
            df['confidence_level'].fillna('medium', inplace=True)
        
        # 数値列の補完
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _evaluate_data_quality(self, df: pd.DataFrame) -> float:
        """データ品質評価（0-1のスコア）"""
        if df.empty:
            return 0.0
        
        quality_factors = []
        
        # 1. データ完整性（NaN割合）
        nan_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        completeness_score = max(0, 1 - nan_ratio)
        quality_factors.append(completeness_score * 0.3)
        
        # 2. データ期間の適切性
        expected_records = self.period_days
        actual_records = len(df)
        coverage_score = min(1.0, actual_records / expected_records)
        quality_factors.append(coverage_score * 0.2)
        
        # 3. 必須列の存在
        required_columns = ['Close', 'trend_type', 'strategy', 'confidence_score']
        present_columns = sum(1 for col in required_columns if col in df.columns)
        column_score = present_columns / len(required_columns)
        quality_factors.append(column_score * 0.3)
        
        # 4. データ値の妥当性
        validity_score = 1.0  # 簡略化
        if 'Close' in df.columns:
            if df['Close'].min() <= 0:
                validity_score *= 0.5
        quality_factors.append(validity_score * 0.2)
        
        return sum(quality_factors)
    
    def _generate_synthetic_price_data(self) -> pd.DataFrame:
        """合成価格データ生成（フォールバック用）"""
        self.logger.info("合成価格データを生成中...")
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=self.period_days),
            periods=self.period_days,
            freq='D'
        )
        
        # ランダムウォーク価格生成
        np.random.seed(42)  # 再現性のため
        base_price = 150.0 if self.symbol == "USDJPY" else 1.0
        
        returns = np.random.normal(0, 0.01, self.period_days)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'Open': [p * np.random.uniform(0.999, 1.001) for p in prices],
            'High': [p * np.random.uniform(1.002, 1.005) for p in prices],
            'Low': [p * np.random.uniform(0.995, 0.998) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, self.period_days)
        }, index=dates)
        
        return df
    
    def _generate_synthetic_trend_data(self) -> pd.DataFrame:
        """合成トレンドデータ生成"""
        self.logger.info("合成トレンドデータを生成中...")
        
        if self.data_sources['price_data'] is None:
            index = pd.date_range(
                start=datetime.now() - timedelta(days=self.period_days),
                periods=self.period_days,
                freq='D'
            )
        else:
            index = self.data_sources['price_data'].index
        
        # トレンドパターン生成
        trends = np.random.choice(['uptrend', 'downtrend', 'sideways'], 
                                  size=len(index), 
                                  p=[0.3, 0.3, 0.4])
        
        df = pd.DataFrame({
            'trend_type': trends,
            'trend_strength': np.random.uniform(0.3, 0.9, len(index))
        }, index=index)
        
        return df
    
    def _generate_synthetic_strategy_data(self) -> pd.DataFrame:
        """合成戦略データ生成"""
        self.logger.info("合成戦略データを生成中...")
        
        if self.data_sources['price_data'] is None:
            index = pd.date_range(
                start=datetime.now() - timedelta(days=self.period_days),
                periods=self.period_days,
                freq='D'
            )
        else:
            index = self.data_sources['price_data'].index
        
        strategies = np.random.choice([
            'trend_following', 'mean_reversion', 'momentum', 'breakout', 'hybrid'
        ], size=len(index), p=[0.25, 0.25, 0.2, 0.15, 0.15])
        
        df = pd.DataFrame({
            'strategy': strategies,
            'confidence': np.random.uniform(0.3, 0.9, len(index)),
            'reason': ['synthetic_data'] * len(index)
        }, index=index)
        
        return df
    
    def _generate_synthetic_confidence_data(self) -> pd.DataFrame:
        """合成信頼度データ生成"""
        self.logger.info("合成信頼度データを生成中...")
        
        if self.data_sources['strategy_data'] is None:
            index = pd.date_range(
                start=datetime.now() - timedelta(days=self.period_days),
                periods=self.period_days,
                freq='D'
            )
        else:
            index = self.data_sources['strategy_data'].index
        
        confidence_scores = np.random.uniform(0.2, 0.9, len(index))
        confidence_levels = ['low' if s < 0.4 else 'medium' if s < 0.7 else 'high' 
                             for s in confidence_scores]
        
        df = pd.DataFrame({
            'confidence_score': confidence_scores,
            'confidence_level': confidence_levels,
            'factors': [['synthetic'] for _ in range(len(index))]
        }, index=index)
        
        return df
    
    def _generate_synthetic_volume_data(self) -> pd.DataFrame:
        """合成ボリュームデータ生成"""
        self.logger.info("合成ボリュームデータを生成中...")
        
        if self.data_sources['price_data'] is None:
            index = pd.date_range(
                start=datetime.now() - timedelta(days=self.period_days),
                periods=self.period_days,
                freq='D'
            )
        else:
            index = self.data_sources['price_data'].index
        
        volumes = np.random.randint(1000, 10000, len(index))
        volume_ma = pd.Series(volumes).rolling(window=5).mean()
        
        df = pd.DataFrame({
            'Volume': volumes,
            'volume_ma': volume_ma,
            'volume_ratio': volumes / volume_ma
        }, index=index)
        
        return df
    
    def _generate_complete_synthetic_data(self) -> pd.DataFrame:
        """完全合成データ生成（最終フォールバック）"""
        self.logger.info("完全合成データを生成中...")
        
        # 全てのデータを合成生成
        price_df = self._generate_synthetic_price_data()
        trend_df = self._generate_synthetic_trend_data()
        strategy_df = self._generate_synthetic_strategy_data()
        confidence_df = self._generate_synthetic_confidence_data()
        volume_df = self._generate_synthetic_volume_data()
        
        # Volumeカラムの重複を避けるために、price_dfからVolumeを削除してvolume_dfを使用
        price_df_clean = price_df.drop(columns=['Volume'], errors='ignore')
        
        # マージ
        merged_df = price_df_clean.join(trend_df).join(strategy_df).join(confidence_df).join(volume_df)
        
        self.data_quality_score = 0.5  # 合成データのスコア
        
        return merged_df
    
    def get_data_summary(self) -> Dict[str, Any]:
        """データサマリー取得"""
        if self.aggregated_data is None:
            return {"status": "no_data", "message": "データが集約されていません"}
        
        df = self.aggregated_data
        
        return {
            "status": "success",
            "symbol": self.symbol,
            "period_days": self.period_days,
            "record_count": len(df),
            "date_range": {
                "start": df.index.min().strftime('%Y-%m-%d'),
                "end": df.index.max().strftime('%Y-%m-%d')
            },
            "data_quality_score": self.data_quality_score,
            "columns": list(df.columns),
            "trend_distribution": df['trend_type'].value_counts().to_dict() if 'trend_type' in df.columns else {},
            "strategy_distribution": df['strategy'].value_counts().to_dict() if 'strategy' in df.columns else {},
            "confidence_stats": {
                "mean": float(df['confidence_score'].mean()) if 'confidence_score' in df.columns else 0,
                "std": float(df['confidence_score'].std()) if 'confidence_score' in df.columns else 0
            }
        }
    
    def export_aggregated_data(self, filepath: str) -> bool:
        """集約データのエクスポート"""
        try:
            if self.aggregated_data is None:
                self.logger.warning("エクスポートするデータがありません")
                return False
            
            self.aggregated_data.to_csv(filepath)
            self.logger.info(f"データエクスポート成功: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"データエクスポート失敗: {e}")
            return False

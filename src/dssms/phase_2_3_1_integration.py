"""
DSSMS Phase 2.3 Task 2.3.1 統合インターフェース
バックテストデータ収集最適化 - main.py統合用

Purpose:
  - main.pyからの呼び出しインターフェース
  - 既存システムとの完全互換性
  - 品質保証システムの透明な統合
  - エラーフリーな実行環境

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Usage Example:
    from src.dssms.phase_2_3_1_integration import apply_dssms_quality_enhancement
    
    # main.pyからの呼び出し
    enhanced_result = apply_dssms_quality_enhancement(
        stock_data=your_dataframe,
        strategy_name="VWAPBreakoutStrategy",
        enable_quality_assurance=True
    )
"""

import pandas as pd
from typing import Dict, Any, Optional, Union
from pathlib import Path
import sys
import warnings

# 警告抑制
warnings.filterwarnings('ignore')

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger

# 条件付きインポート（品質保証システム）
try:
    from src.dssms.integration_manager import DSSMSIntegrationManager
    QUALITY_ASSURANCE_AVAILABLE = True
except ImportError as e:
    QUALITY_ASSURANCE_AVAILABLE = False
    import_error = str(e)


def apply_dssms_quality_enhancement(
    stock_data: pd.DataFrame,
    strategy_name: Optional[str] = None,
    enable_quality_assurance: bool = True,
    custom_config: Optional[Dict[str, Any]] = None,
    fallback_on_error: bool = True,
    initial_capital: float = 1000000.0
) -> Dict[str, Any]:
    """
    DSSMS品質保証システムの統合適用
    
    main.pyから呼び出し可能な統合インターフェース
    既存システムとの互換性を保ちながら品質向上を実現
    
    Args:
        stock_data: 株価データ（main.pyから渡される）
        strategy_name: 戦略名
        enable_quality_assurance: 品質保証有効化フラグ
        custom_config: カスタム設定
        fallback_on_error: エラー時のフォールバック有効化
        initial_capital: 初期資本金
    
    Returns:
        Dict[str, Any]: 品質保証適用済み結果または通常結果
    """
    logger = setup_logger(__name__)
    
    try:
        # 品質保証システムが利用可能かチェック
        if enable_quality_assurance and QUALITY_ASSURANCE_AVAILABLE:
            logger.info(f"DSSMS品質保証システム適用開始: {strategy_name or 'Unknown'}")
            
            # Integration Manager初期化
            integration_manager = DSSMSIntegrationManager(
                initial_capital=initial_capital,
                auto_export=custom_config.get('auto_export', False) if custom_config else False
            )
            
            # 品質保証付き実行
            enhanced_result = integration_manager.execute_dssms_with_qa(
                stock_data=stock_data,
                strategy_name=strategy_name,
                custom_config=custom_config
            )
            
            logger.info(f"品質保証システム適用完了: 信頼性スコア {enhanced_result.get('reliability_score', 0):.3f}")
            return enhanced_result
            
        elif enable_quality_assurance and not QUALITY_ASSURANCE_AVAILABLE:
            logger.warning(f"品質保証システム利用不可: {import_error}")
            if fallback_on_error:
                return _create_fallback_result(stock_data, strategy_name, initial_capital, logger)
            else:
                raise ImportError(f"品質保証システムインポートエラー: {import_error}")
        
        else:
            logger.info("品質保証システム無効化 - 通常処理実行")
            return _create_fallback_result(stock_data, strategy_name, initial_capital, logger)
            
    except Exception as e:
        logger.error(f"品質保証システム適用中にエラー: {e}")
        
        if fallback_on_error:
            logger.info("フォールバック処理に切り替え")
            return _create_fallback_result(stock_data, strategy_name, initial_capital, logger, error=str(e))
        else:
            raise


def apply_multi_strategy_dssms_quality_enhancement(
    strategy_data_dict: Dict[str, pd.DataFrame],
    enable_quality_assurance: bool = True,
    custom_config: Optional[Dict[str, Any]] = None,
    fallback_on_error: bool = True,
    initial_capital: float = 1000000.0
) -> Dict[str, Any]:
    """
    複数戦略DSSMS品質保証システムの統合適用
    
    Args:
        strategy_data_dict: 戦略別データ辞書
        enable_quality_assurance: 品質保証有効化フラグ
        custom_config: カスタム設定
        fallback_on_error: エラー時のフォールバック有効化
        initial_capital: 初期資本金
    
    Returns:
        Dict[str, Any]: 品質保証適用済み複数戦略結果
    """
    logger = setup_logger(__name__)
    
    try:
        if enable_quality_assurance and QUALITY_ASSURANCE_AVAILABLE:
            logger.info(f"複数戦略DSSMS品質保証システム適用開始: {len(strategy_data_dict)}戦略")
            
            # Integration Manager初期化
            integration_manager = DSSMSIntegrationManager(
                initial_capital=initial_capital,
                auto_export=custom_config.get('auto_export', False) if custom_config else False
            )
            
            # 複数戦略品質保証付き実行
            enhanced_result = integration_manager.execute_multi_strategy_dssms(
                strategy_data_dict=strategy_data_dict,
                custom_config=custom_config
            )
            
            logger.info("複数戦略品質保証システム適用完了")
            return enhanced_result
            
        else:
            logger.info("複数戦略品質保証システム無効化 - 個別処理実行")
            return _create_multi_strategy_fallback_result(strategy_data_dict, initial_capital, logger)
            
    except Exception as e:
        logger.error(f"複数戦略品質保証システム適用中にエラー: {e}")
        
        if fallback_on_error:
            logger.info("複数戦略フォールバック処理に切り替え")
            return _create_multi_strategy_fallback_result(strategy_data_dict, initial_capital, logger, error=str(e))
        else:
            raise


def _create_fallback_result(
    stock_data: pd.DataFrame,
    strategy_name: Optional[str],
    initial_capital: float,
    logger,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """フォールバック結果の作成"""
    try:
        # 基本的な取引抽出（output/data_extraction_enhancer.pyを使用）
        try:
            from output.data_extraction_enhancer import MainDataExtractor
            
            extractor = MainDataExtractor(initial_capital)
            trades = extractor.extract_accurate_trades(stock_data)
            
            # 基本統計計算
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
            total_profit_loss = sum(t.get('profit_loss', 0) for t in trades) * initial_capital
            
            result = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_profit_loss': total_profit_loss,
                'trades': trades,
                'data': stock_data,
                'strategy': strategy_name or 'Unknown',
                'execution_mode': 'fallback',
                'quality_assurance_applied': False
            }
            
            if error:
                result['fallback_reason'] = error
            
            logger.info(f"フォールバック処理完了: {total_trades}取引, 勝率{win_rate:.3f}")
            return result
            
        except ImportError:
            logger.warning("MainDataExtractor利用不可 - 最小限結果を返却")
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit_loss': 0.0,
                'trades': [],
                'data': stock_data,
                'strategy': strategy_name or 'Unknown',
                'execution_mode': 'minimal_fallback',
                'quality_assurance_applied': False,
                'fallback_reason': error or 'システム制限'
            }
            
    except Exception as e:
        logger.error(f"フォールバック処理中にエラー: {e}")
        return {
            'error': str(e),
            'total_trades': 0,
            'win_rate': 0.0,
            'total_profit_loss': 0.0,
            'execution_mode': 'error_fallback',
            'quality_assurance_applied': False
        }


def _create_multi_strategy_fallback_result(
    strategy_data_dict: Dict[str, pd.DataFrame],
    initial_capital: float,
    logger,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """複数戦略フォールバック結果の作成"""
    try:
        strategy_results = {}
        
        for strategy_name, strategy_data in strategy_data_dict.items():
            try:
                strategy_results[strategy_name] = _create_fallback_result(
                    strategy_data, strategy_name, initial_capital, logger, error
                )
            except Exception as e:
                logger.error(f"戦略{strategy_name}のフォールバック処理中にエラー: {e}")
                strategy_results[strategy_name] = {
                    'error': str(e),
                    'execution_mode': 'error_fallback'
                }
        
        return {
            'strategy_results': strategy_results,
            'overall_qa_metrics': {
                'total_strategies': len(strategy_data_dict),
                'successful_strategies': len([r for r in strategy_results.values() if 'error' not in r]),
                'execution_mode': 'multi_strategy_fallback'
            },
            'quality_assurance_applied': False,
            'fallback_reason': error or 'システム制限'
        }
        
    except Exception as e:
        logger.error(f"複数戦略フォールバック処理中にエラー: {e}")
        return {
            'error': str(e),
            'strategy_results': {},
            'execution_mode': 'multi_strategy_error_fallback'
        }


def check_quality_assurance_availability() -> Dict[str, Any]:
    """品質保証システムの利用可能性チェック"""
    return {
        'available': QUALITY_ASSURANCE_AVAILABLE,
        'import_error': import_error if not QUALITY_ASSURANCE_AVAILABLE else None,
        'components': {
            'integration_manager': QUALITY_ASSURANCE_AVAILABLE,
            'data_quality_enhancer': QUALITY_ASSURANCE_AVAILABLE,
            'output_validator': QUALITY_ASSURANCE_AVAILABLE,
            'enhanced_backtester': QUALITY_ASSURANCE_AVAILABLE
        }
    }


def get_recommended_quality_config() -> Dict[str, Any]:
    """推奨品質設定の取得"""
    return {
        'quality_config': {
            'min_price_threshold': 0.01,
            'max_price_change_ratio': 0.30,
            'signal_consistency_check': True,
            'auto_correction_enabled': True,
            'quality_threshold': 0.85,
            'fallback_to_previous': True
        },
        'validation_config': {
            'min_trade_count': 1,
            'max_drawdown_threshold': 0.50,
            'min_win_rate': 0.20,
            'min_sharpe_ratio': -2.0,
            'max_single_loss': 0.10,
            'consistency_check': True,
            'statistical_validation': True,
            'auto_correction': True
        },
        'quality_standards': {
            'minimum_quality_score': 0.75,
            'minimum_validation_score': 0.70,
            'minimum_reliability_score': 0.80,
            'maximum_allowed_errors': 5
        },
        'auto_export': False,
        'output_dir': 'backtest_results/dssms_enhanced'
    }


if __name__ == "__main__":
    # テスト実行
    logger = setup_logger(__name__)
    
    # 利用可能性チェック
    availability = check_quality_assurance_availability()
    logger.info(f"品質保証システム利用可能性: {availability['available']}")
    
    if availability['available']:
        # テストデータでの実行
        import pandas as pd
        from datetime import datetime, timedelta
        
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
        test_data = pd.DataFrame({
            'Close': [100, 102, 98, 105, 103, 107, 109, 106],
            'Entry_Signal': [1, 0, 0, 1, 0, 0, 0, 0],
            'Exit_Signal': [0, 1, 0, 0, 1, 0, 1, 0],
            'Strategy': ['TestStrategy'] * 8
        }, index=dates[:8])
        
        # 統合インターフェーステスト
        result = apply_dssms_quality_enhancement(
            stock_data=test_data,
            strategy_name="TestStrategy",
            enable_quality_assurance=True,
            custom_config={'auto_export': False}
        )
        
        logger.info(f"統合インターフェーステスト完了: 信頼性スコア {result.get('reliability_score', 0):.3f}")
    else:
        logger.warning(f"品質保証システム利用不可: {availability['import_error']}")

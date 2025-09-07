"""
DSSMS Unified Output Engine Demo
Phase 2.3 Task 2.3.2: 多形式出力エンジン構築

Purpose:
  - 統一出力エンジンの動作デモ
  - 既存システム統合テスト
  - 品質保証機能テスト
  - 多形式出力テスト

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Demo Scenarios:
  1. 基本的な統一出力テスト
  2. DSSMS データ対応テスト
  3. 品質向上機能テスト
  4. テンプレートシステムテスト
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config.logger_config import setup_logger
from src.dssms.unified_output_engine import UnifiedOutputEngine
from src.dssms.output_data_model import UnifiedDataModelConverter
from src.dssms.template_manager import TemplateManager
from src.dssms.unified_output_config import get_config, validate_config


def create_sample_basic_data():
    """基本的なサンプルデータ作成"""
    return {
        'metadata': {
            'ticker': 'DEMO_BASIC',
            'period_start': '2024-01-01',
            'period_end': '2024-12-31'
        },
        'summary': {
            'total_return': 0.185,
            'total_pnl': 185000,
            'win_rate': 0.65,
            'num_trades': 25,
            'sharpe_ratio': 1.45,
            'max_drawdown': -0.12,
            'final_portfolio_value': 1185000
        },
        'trades': [
            {
                'strategy': 'VWAPBreakout',
                'entry_date': '2024-01-15',
                'exit_date': '2024-01-22',
                'entry_price': 2500.0,
                'exit_price': 2650.0,
                'shares': 100,
                'profit_loss': 15000,
                'profit_loss_pct': 0.06,
                'duration_days': 7
            },
            {
                'strategy': 'MeanReversion',
                'entry_date': '2024-02-05',
                'exit_date': '2024-02-12',
                'entry_price': 2750.0,
                'exit_price': 2680.0,
                'shares': 100,
                'profit_loss': -7000,
                'profit_loss_pct': -0.025,
                'duration_days': 7
            },
            {
                'strategy': 'TrendFollowing',
                'entry_date': '2024-03-10',
                'exit_date': '2024-03-20',
                'entry_price': 2800.0,
                'exit_price': 2950.0,
                'shares': 100,
                'profit_loss': 15000,
                'profit_loss_pct': 0.054,
                'duration_days': 10
            }
        ],
        'raw_data': pd.DataFrame({
            'Date': pd.date_range('2024-01-01', '2024-12-31', freq='D'),
            'Close': [2500 + i * 0.5 + (i % 10 - 5) * 10 for i in range(366)],
            'Volume': [1000000 + i * 1000 for i in range(366)]
        })
    }


def create_sample_dssms_data():
    """DSSMSサンプルデータ作成"""
    return {
        'ticker': 'DEMO_DSSMS',
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'total_return': 0.225,
        'total_profit_loss': 0.225,  # パーセンテージとして
        'win_rate': 0.72,
        'total_trades': 35,
        'sharpe_ratio': 1.68,
        'max_drawdown': -0.095,
        'portfolio_value': 1225000,
        'strategy_scores': {
            'VWAP_Breakout': 0.85,
            'Mean_Reversion': 0.72,
            'Trend_Following': 0.91,
            'Momentum': 0.68,
            'Contrarian': 0.55,
            'Swing_Trading': 0.78,
            'Day_Trading': 0.62
        },
        'switch_decisions': [
            {
                'date': '2024-01-15',
                'from_strategy': 'VWAP_Breakout',
                'to_strategy': 'Trend_Following',
                'confidence': 0.82,
                'market_condition': 'bullish'
            },
            {
                'date': '2024-02-20',
                'from_strategy': 'Trend_Following',
                'to_strategy': 'Mean_Reversion',
                'confidence': 0.75,
                'market_condition': 'sideways'
            }
        ],
        'ranking_data': {
            'current_ranking': ['Trend_Following', 'VWAP_Breakout', 'Swing_Trading', 'Mean_Reversion'],
            'confidence_scores': [0.91, 0.85, 0.78, 0.72],
            'last_update': '2024-12-31'
        },
        'switch_success_rate': 0.78,
        'switch_frequency': 0.25,
        'trades': [
            {
                'strategy': 'DSSMS_Adaptive',
                'entry_date': '2024-01-10',
                'exit_date': '2024-01-18',
                'entry_price': 2450.0,
                'exit_price': 2580.0,
                'profit_loss': 0.053,  # パーセンテージ
                'shares': 100
            },
            {
                'strategy': 'DSSMS_Adaptive',
                'entry_date': '2024-02-15',
                'exit_date': '2024-02-22',
                'entry_price': 2720.0,
                'exit_price': 2845.0,
                'profit_loss': 0.046,
                'shares': 100
            }
        ],
        'quality_assurance': {
            'data_quality_score': 0.88,
            'validation_score': 0.92,
            'enhancement_applied': True,
            'validation_errors': []
        },
        'reliability_score': 0.87,
        'recommended_actions': [
            'Trend_Following戦略の重みを増加',
            'Contrarian戦略のパラメータ調整を検討'
        ]
    }


def create_sample_enhanced_data():
    """品質向上データサンプル作成"""
    base_data = create_sample_basic_data()
    
    # MainDataExtractor形式に拡張
    enhanced_data = {
        'ticker': 'DEMO_ENHANCED',
        'period': {
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        },
        'extraction_timestamp': datetime.now().isoformat(),
        'performance': {
            'total_return_pct': 0.195,
            'total_profit_loss': 195000,
            'win_rate': 0.68,
            'total_trades': 28,
            'winning_trades': 19,
            'losing_trades': 9,
            'average_win': 25500,
            'average_loss': -8900,
            'profit_factor': 1.85,
            'sharpe_ratio': 1.52,
            'max_drawdown': -0.108,
            'final_portfolio_value': 1195000,
            'initial_capital': 1000000
        },
        'trades': [
            {
                'strategy': 'Enhanced_VWAP',
                'entry_date': '2024-01-12',
                'exit_date': '2024-01-19',
                'entry_price': 2480.0,
                'exit_price': 2620.0,
                'shares': 100,
                'profit_loss_amount': 14000,
                'profit_loss': 0.056,
                'duration_days': 7
            },
            {
                'strategy': 'Enhanced_MeanRev',
                'entry_date': '2024-02-08',
                'exit_date': '2024-02-14',
                'entry_price': 2730.0,
                'exit_price': 2850.0,
                'shares': 100,
                'profit_loss_amount': 12000,
                'profit_loss': 0.044,
                'duration_days': 6
            }
        ],
        'data_quality': {
            'overall_score': 0.91,
            'completeness': 0.95,
            'accuracy': 0.89,
            'consistency': 0.88
        }
    }
    
    return enhanced_data


def demo_basic_unified_output():
    """基本的な統一出力デモ"""
    print("=" * 60)
    print("Demo 1: 基本的な統一出力テスト")
    print("=" * 60)
    
    try:
        # 統一出力エンジンの初期化
        engine = UnifiedOutputEngine("demo_output/basic")
        
        # サンプルデータの作成
        sample_data = create_sample_basic_data()
        
        print(f"サンプルデータ作成完了:")
        print(f"  銘柄: {sample_data['metadata']['ticker']}")
        print(f"  取引数: {len(sample_data['trades'])}")
        print(f"  総リターン: {sample_data['summary']['total_return']:.2%}")
        
        # 統一出力の生成
        output_files = engine.generate_unified_output(
            data=sample_data,
            output_formats=['excel', 'json', 'text', 'html'],
            output_prefix='demo_basic_unified'
        )
        
        print(f"\n出力ファイル生成完了:")
        for format_type, filepath in output_files.items():
            print(f"  {format_type.upper()}: {filepath}")
        
        # 出力履歴の確認
        history = engine.get_output_history()
        print(f"\n出力履歴エントリー数: {len(history)}")
        
        return True
        
    except Exception as e:
        print(f"基本出力デモ中にエラー: {e}")
        return False


def demo_dssms_output():
    """DSSMS対応出力デモ"""
    print("\n" + "=" * 60)
    print("Demo 2: DSSMS データ対応テスト")
    print("=" * 60)
    
    try:
        # 統一出力エンジンの初期化
        engine = UnifiedOutputEngine("demo_output/dssms")
        
        # DSSMSサンプルデータの作成
        dssms_data = create_sample_dssms_data()
        
        print(f"DSSMSデータ作成完了:")
        print(f"  銘柄: {dssms_data['ticker']}")
        print(f"  戦略数: {len(dssms_data['strategy_scores'])}")
        print(f"  切り替え成功率: {dssms_data['switch_success_rate']:.2%}")
        print(f"  信頼性スコア: {dssms_data['reliability_score']:.2%}")
        
        # DSSMS統一出力の生成
        output_files = engine.generate_unified_output(
            data=dssms_data,
            output_formats=['excel', 'json', 'text', 'html'],
            output_prefix='demo_dssms_unified'
        )
        
        print(f"\nDSSMS出力ファイル生成完了:")
        for format_type, filepath in output_files.items():
            print(f"  {format_type.upper()}: {filepath}")
        
        # 統一データモデル変換テスト
        converter = UnifiedDataModelConverter()
        unified_model = converter.convert_from_dssms_data(dssms_data)
        
        print(f"\n統一データモデル変換結果:")
        print(f"  データソース: {unified_model.metadata.data_source}")
        print(f"  DSSMSメトリクス: {'あり' if unified_model.dssms_metrics else 'なし'}")
        print(f"  品質保証情報: {'あり' if unified_model.quality_assurance else 'なし'}")
        
        # モデル妥当性検証
        validation_result = engine.validate_unified_model(unified_model)
        print(f"\nモデル妥当性検証:")
        print(f"  有効: {validation_result['is_valid']}")
        print(f"  品質スコア: {validation_result['quality_score']:.3f}")
        if validation_result['warnings']:
            print(f"  警告: {len(validation_result['warnings'])} 件")
        
        return True
        
    except Exception as e:
        print(f"DSSMS出力デモ中にエラー: {e}")
        return False


def demo_quality_enhancement():
    """品質向上機能デモ"""
    print("\n" + "=" * 60)
    print("Demo 3: 品質向上機能テスト")
    print("=" * 60)
    
    try:
        # 統一出力エンジンの初期化（品質向上強制有効）
        engine = UnifiedOutputEngine("demo_output/enhanced")
        
        # 品質向上サンプルデータの作成
        enhanced_data = create_sample_enhanced_data()
        
        print(f"品質向上データ作成完了:")
        print(f"  銘柄: {enhanced_data['ticker']}")
        print(f"  データ品質スコア: {enhanced_data['data_quality']['overall_score']:.3f}")
        print(f"  抽出タイムスタンプ: {enhanced_data['extraction_timestamp']}")
        
        # 品質向上有効での統一出力生成
        output_files = engine.generate_unified_output(
            data=enhanced_data,
            output_formats=['excel', 'json', 'text'],
            output_prefix='demo_enhanced_unified',
            force_enhanced_extraction=True
        )
        
        print(f"\n品質向上出力ファイル生成完了:")
        for format_type, filepath in output_files.items():
            print(f"  {format_type.upper()}: {filepath}")
        
        # 品質向上なしでの比較出力
        output_files_basic = engine.generate_unified_output(
            data=enhanced_data,
            output_formats=['json'],
            output_prefix='demo_enhanced_comparison',
            force_enhanced_extraction=False
        )
        
        print(f"\n比較用基本出力:")
        for format_type, filepath in output_files_basic.items():
            print(f"  {format_type.upper()}: {filepath}")
        
        return True
        
    except Exception as e:
        print(f"品質向上デモ中にエラー: {e}")
        return False


def demo_template_system():
    """テンプレートシステムデモ"""
    print("\n" + "=" * 60)
    print("Demo 4: テンプレートシステムテスト")
    print("=" * 60)
    
    try:
        # テンプレートマネージャーの初期化
        template_manager = TemplateManager("demo_output/templates")
        
        # 統一データモデル変換器の初期化
        converter = UnifiedDataModelConverter()
        
        # サンプルデータをDSSMS形式で作成
        dssms_data = create_sample_dssms_data()
        unified_model = converter.convert_from_dssms_data(dssms_data)
        
        print(f"テンプレートシステムテスト:")
        print(f"  銘柄: {unified_model.metadata.ticker}")
        print(f"  データソース: {unified_model.metadata.data_source}")
        
        # HTMLテンプレートレンダリング
        html_output = template_manager.render_html_template(unified_model)
        html_path = Path("demo_output/templates/demo_template_test.html")
        html_path.parent.mkdir(parents=True, exist_ok=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
        print(f"  HTMLテンプレート出力: {html_path}")
        
        # テキストテンプレートレンダリング
        text_output = template_manager.render_text_template(unified_model)
        text_path = Path("demo_output/templates/demo_template_test.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_output)
        print(f"  テキストテンプレート出力: {text_path}")
        
        # Excel設定テンプレート取得
        excel_config = template_manager.get_excel_template_config()
        print(f"  Excel設定テンプレート: {len(excel_config)} セクション")
        
        # JSON スキーマ検証
        test_json_data = unified_model.to_dict()
        validation_result = template_manager.validate_json_schema(test_json_data)
        print(f"  JSON スキーマ検証: {'成功' if validation_result['is_valid'] else '失敗'}")
        
        return True
        
    except Exception as e:
        print(f"テンプレートシステムデモ中にエラー: {e}")
        return False


def demo_config_validation():
    """設定検証デモ"""
    print("\n" + "=" * 60)
    print("Demo 5: 設定検証テスト")
    print("=" * 60)
    
    try:
        # 基本設定の取得と検証
        config = get_config()
        validation_result = validate_config(config)
        
        print(f"設定検証結果:")
        print(f"  設定有効: {validation_result['is_valid']}")
        print(f"  エラー数: {len(validation_result['errors'])}")
        print(f"  警告数: {len(validation_result['warnings'])}")
        
        if validation_result['errors']:
            print(f"  エラー詳細:")
            for error in validation_result['errors']:
                print(f"    - {error}")
        
        if validation_result['warnings']:
            print(f"  警告詳細:")
            for warning in validation_result['warnings']:
                print(f"    - {warning}")
        
        # 主要設定の表示
        print(f"\n主要設定:")
        print(f"  エンジン名: {config['engine_info']['name']}")
        print(f"  バージョン: {config['engine_info']['version']}")
        print(f"  有効出力形式: {config['output_formats']['enabled_formats']}")
        print(f"  品質保証有効: {config['quality_assurance']['enabled']}")
        print(f"  統合システム数: {len(config['integration']['existing_systems'])}")
        
        return validation_result['is_valid']
        
    except Exception as e:
        print(f"設定検証デモ中にエラー: {e}")
        return False


def run_comprehensive_demo():
    """包括的なデモ実行"""
    print("🚀 DSSMS統一出力エンジン包括デモ開始")
    print("=" * 80)
    
    # ログ設定
    logger = setup_logger(__name__)
    logger.info("統一出力エンジンデモを開始します")
    
    # デモ結果追跡
    demo_results = {}
    
    try:
        # Demo 1: 基本的な統一出力
        demo_results['basic_output'] = demo_basic_unified_output()
        
        # Demo 2: DSSMS対応出力
        demo_results['dssms_output'] = demo_dssms_output()
        
        # Demo 3: 品質向上機能
        demo_results['quality_enhancement'] = demo_quality_enhancement()
        
        # Demo 4: テンプレートシステム
        demo_results['template_system'] = demo_template_system()
        
        # Demo 5: 設定検証
        demo_results['config_validation'] = demo_config_validation()
        
        # 結果サマリー
        print("\n" + "=" * 80)
        print("🎯 デモ実行結果サマリー")
        print("=" * 80)
        
        total_demos = len(demo_results)
        successful_demos = sum(demo_results.values())
        
        print(f"実行デモ数: {total_demos}")
        print(f"成功デモ数: {successful_demos}")
        print(f"成功率: {(successful_demos/total_demos)*100:.1f}%")
        
        print(f"\n詳細結果:")
        for demo_name, result in demo_results.items():
            status = "✅ 成功" if result else "❌ 失敗"
            print(f"  {demo_name}: {status}")
        
        # 出力ディレクトリの確認
        output_base = Path("demo_output")
        if output_base.exists():
            print(f"\n📁 出力ディレクトリ構造:")
            for subdir in output_base.iterdir():
                if subdir.is_dir():
                    file_count = len(list(subdir.glob("*")))
                    print(f"  {subdir.name}/: {file_count} ファイル")
        
        logger.info(f"デモ完了: {successful_demos}/{total_demos} 成功")
        
        if successful_demos == total_demos:
            print(f"\n🎉 全デモが正常に完了しました！")
            print(f"Phase 2.3 Task 2.3.2 「多形式出力エンジン構築」実装完了")
        else:
            print(f"\n⚠️ 一部デモに問題がありました。ログを確認してください。")
        
        return successful_demos == total_demos
        
    except Exception as e:
        print(f"\n❌ デモ実行中に予期しないエラーが発生しました: {e}")
        logger.error(f"デモ実行エラー: {e}")
        return False


if __name__ == "__main__":
    success = run_comprehensive_demo()
    sys.exit(0 if success else 1)

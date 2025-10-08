"""
DSSMS Unified Output Engine Configuration
Phase 2.3 Task 2.3.2: 多形式出力エンジン構築

Purpose:
  - 統一出力エンジンの設定管理
  - 出力形式・テンプレート設定
  - 品質保証設定
  - システム統合設定

Author: GitHub Copilot Agent
Created: 2025-01-24
Version: 1.0

Configuration Categories:
  - output_formats: 出力形式設定
  - template_settings: テンプレート設定
  - quality_assurance: 品質保証設定
  - integration: 既存システム統合設定
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# 統一出力エンジン設定
UNIFIED_OUTPUT_CONFIG = {
    "engine_info": {
        "name": "DSSMS統一出力エンジン",
        "version": "1.0.0",
        "phase": "2.3",
        "task": "2.3.2",
        "author": "GitHub Copilot Agent",
        "created": "2025-01-24"
    },
    
    "output_formats": {
        "enabled_formats": ["excel", "json", "text", "html"],
        "default_formats": ["excel", "json"],
        "priority_order": ["excel", "json", "text", "html"],
        
        "excel": {
            "enabled": True,
            "use_existing_exporters": True,
            "simple_exporter_priority": True,
            "dssms_exporter_for_dssms_data": True,
            "enhanced_extraction": True,
            "template_based": True,
            "file_extension": ".xlsx",
            "default_sheets": ["Summary", "Trades", "DSSMS_Analysis", "Quality_Assurance"]
        },
        
        "json": {
            "enabled": True,
            "pretty_print": True,
            "ensure_ascii": False,
            "indent": 2,
            "schema_validation": True,
            "file_extension": ".json",
            "include_metadata": True,
            "datetime_format": "iso"
        },
        
        "text": {
            "enabled": True,
            "encoding": "utf-8",
            "line_separator": "\n",
            "section_separator": "=" * 40,
            "border_line": "=" * 80,
            "file_extension": ".txt",
            "max_trades_display": 10,
            "use_existing_reporter": True
        },
        
        "html": {
            "enabled": True,
            "template_based": True,
            "responsive_design": True,
            "include_css": True,
            "file_extension": ".html",
            "chart_support": False,  # Phase 2.4で実装予定
            "max_trades_display": 20
        }
    },
    
    "template_settings": {
        "template_directory": "src/dssms/templates",
        "cache_templates": True,
        "auto_create_missing": True,
        "template_validation": True,
        
        "excel_template": {
            "config_file": "excel_template_config.json",
            "styling": {
                "header_font_bold": True,
                "header_font_color": "FFFFFF",
                "header_fill_color": "366092",
                "positive_color": "27AE60",
                "negative_color": "E74C3C",
                "neutral_color": "34495E"
            }
        },
        
        "html_template": {
            "file": "html_report_template.html",
            "responsive": True,
            "modern_design": True,
            "color_scheme": {
                "primary": "#667eea",
                "secondary": "#764ba2",
                "positive": "#27ae60",
                "negative": "#e74c3c",
                "neutral": "#34495e"
            }
        },
        
        "text_template": {
            "file": "text_report_template.txt",
            "variable_markers": ["{{", "}}"],
            "section_styling": True
        },
        
        "json_schema": {
            "file": "json_schema_template.json",
            "validation_enabled": True,
            "strict_mode": False
        }
    },
    
    "quality_assurance": {
        "enabled": True,
        "force_enhanced_extraction": True,
        "data_validation": True,
        "output_validation": True,
        "quality_scoring": True,
        
        "enhancement_settings": {
            "use_main_data_extractor": True,
            "fallback_on_error": True,
            "min_reliability_score": 0.6,
            "enhancement_required_conditions": [
                "missing_trades_data",
                "incomplete_performance_metrics",
                "low_data_quality_score"
            ]
        },
        
        "validation_rules": {
            "required_fields": ["metadata.ticker", "performance.total_trades"],
            "range_checks": {
                "win_rate": [0, 1],
                "total_trades": [0, None],
                "reliability_score": [0, 1]
            },
            "date_validation": True,
            "trade_consistency_check": True
        },
        
        "scoring_weights": {
            "data_completeness": 0.3,
            "validation_passed": 0.3,
            "enhancement_applied": 0.2,
            "existing_system_compatibility": 0.2
        }
    },
    
    "integration": {
        "existing_systems": {
            "simple_excel_exporter": {
                "enabled": True,
                "module": "simple_excel_exporter",
                "class": "ExcelDataProcessor",
                "priority": 1,
                "condition": "always"
            },
            
            "dssms_excel_exporter_v2": {
                "enabled": True,
                "module": "dssms_excel_exporter_v2",
                "class": "DSSMSExcelExporterV2",
                "priority": 2,
                "condition": "dssms_data_present"
            },
            
            "main_text_reporter": {
                "enabled": True,
                "module": "main_text_reporter",
                "class": "MainTextReporter",
                "priority": 3,
                "condition": "text_output_requested"
            },
            
            "data_extraction_enhancer": {
                "enabled": True,
                "module": "data_extraction_enhancer",
                "class": "MainDataExtractor",
                "priority": 0,  # 最高優先度
                "condition": "quality_enhancement_required"
            }
        },
        
        "wrapper_strategy": {
            "use_wrapper_pattern": True,
            "preserve_existing_interfaces": True,
            "enhance_before_processing": True,
            "unified_output_directory": "output",
            "maintain_output_history": True
        },
        
        "compatibility": {
            "main_py_integration": True,
            "existing_config_respect": True,
            "output_directory_structure": "preserve",
            "filename_convention": "timestamp_based"
        }
    },
    
    "output_management": {
        "base_directory": "output",
        "subdirectories": {
            "unified_reports": "unified_reports",
# TODO(tag:excel_deprecated, rationale:Excel output eliminated 2025-10-08) # BACKTEST_IMPACT: Trading data output affected
# ORIGINAL: "excel_outputs": "excel_outputs",
            "json_outputs": "json_outputs",
            "text_outputs": "text_outputs",
            "html_outputs": "html_outputs"
        },
        
        "filename_patterns": {
            "unified": "{prefix}_{timestamp}",
            "excel": "{prefix}_{timestamp}.xlsx",
            "json": "{prefix}_{timestamp}.json",
            "text": "{prefix}_{timestamp}.txt",
            "html": "{prefix}_{timestamp}.html"
        },
        
        "history_management": {
            "maintain_history": True,
            "history_file": "output_history.json",
            "max_history_entries": 1000,
            "cleanup_old_files": False
        },
        
        "file_permissions": {
            "create_directories": True,
            "overwrite_existing": True,
            "backup_on_overwrite": False
        }
    },
    
    "performance": {
        "caching": {
            "template_cache": True,
            "data_cache": False,
            "cache_timeout": 3600
        },
        
        "optimization": {
            "parallel_processing": False,  # Phase 2.4で検討
            "lazy_loading": True,
            "memory_management": True
        },
        
        "limits": {
            "max_trades_in_memory": 10000,
            "max_output_file_size": "100MB",
            "timeout_seconds": 300
        }
    },
    
    "debugging": {
        "log_level": "INFO",
        "detailed_error_messages": True,
        "save_debug_info": False,
        "validation_verbose": False,
        
        "fallback_behavior": {
            "on_template_error": "use_built_in",
            "on_integration_error": "skip_enhancement",
            "on_validation_error": "warn_and_continue",
            "on_critical_error": "return_empty_result"
        }
    },
    
    "experimental": {
        "features": {
            "auto_chart_generation": False,  # Phase 2.4
            "pdf_output": False,            # Phase 2.4
            "cloud_storage": False,         # Phase 2.5
            "real_time_updates": False      # Phase 2.5
        }
    }
}

# 設定バリデーション関数
def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    設定の妥当性検証
    
    Args:
        config: 検証対象の設定
        
    Returns:
        Dict[str, Any]: 検証結果
    """
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    try:
        # 必須セクションの存在確認
        required_sections = ["engine_info", "output_formats", "template_settings", "integration"]
        for section in required_sections:
            if section not in config:
                validation_result["errors"].append(f"必須セクション '{section}' が見つかりません")
                validation_result["is_valid"] = False
        
        # 出力形式の妥当性確認
        if "output_formats" in config:
            enabled_formats = config["output_formats"].get("enabled_formats", [])
            supported_formats = ["excel", "json", "text", "html"]
            
            for fmt in enabled_formats:
                if fmt not in supported_formats:
                    validation_result["warnings"].append(f"未サポートの出力形式: {fmt}")
                
                if fmt not in config["output_formats"]:
                    validation_result["errors"].append(f"出力形式 '{fmt}' の設定が見つかりません")
                    validation_result["is_valid"] = False
        
        # 品質保証設定の確認
        if "quality_assurance" in config:
            qa_config = config["quality_assurance"]
            if qa_config.get("enabled", False):
                if "enhancement_settings" not in qa_config:
                    validation_result["warnings"].append("品質保証が有効ですが、enhancement_settingsが見つかりません")
        
        # 統合設定の確認
        if "integration" in config:
            existing_systems = config["integration"].get("existing_systems", {})
            for system_name, system_config in existing_systems.items():
                if not isinstance(system_config, dict):
                    validation_result["errors"].append(f"システム '{system_name}' の設定が無効です")
                    validation_result["is_valid"] = False
                elif "module" not in system_config:
                    validation_result["warnings"].append(f"システム '{system_name}' にモジュール名が指定されていません")
        
    except Exception as e:
        validation_result["errors"].append(f"設定検証中にエラー: {e}")
        validation_result["is_valid"] = False
    
    return validation_result

# 設定取得関数
def get_config() -> Dict[str, Any]:
    """統一出力エンジン設定の取得"""
    return UNIFIED_OUTPUT_CONFIG.copy()

def get_output_format_config(format_type: str) -> Dict[str, Any]:
    """特定出力形式の設定取得"""
    return UNIFIED_OUTPUT_CONFIG.get("output_formats", {}).get(format_type, {})

def get_template_config() -> Dict[str, Any]:
    """テンプレート設定の取得"""
    return UNIFIED_OUTPUT_CONFIG.get("template_settings", {})

def get_quality_config() -> Dict[str, Any]:
    """品質保証設定の取得"""
    return UNIFIED_OUTPUT_CONFIG.get("quality_assurance", {})

def get_integration_config() -> Dict[str, Any]:
    """統合設定の取得"""
    return UNIFIED_OUTPUT_CONFIG.get("integration", {})

# 設定保存・読み込み関数
def save_config_to_file(config: Dict[str, Any], filepath: str):
    """設定をファイルに保存"""
    try:
        config_path = Path(filepath)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        raise Exception(f"設定保存中にエラー: {e}")

def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """ファイルから設定を読み込み"""
    try:
        config_path = Path(filepath)
        
        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {filepath}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except Exception as e:
        raise Exception(f"設定読み込み中にエラー: {e}")

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """設定のマージ（override_configがbase_configを上書き）"""
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    return deep_merge(base_config, override_config)

# 実行時設定の動的更新
def update_config_at_runtime(section: str, key: str, value: Any):
    """実行時の設定更新"""
    try:
        if section in UNIFIED_OUTPUT_CONFIG:
            if isinstance(UNIFIED_OUTPUT_CONFIG[section], dict):
                UNIFIED_OUTPUT_CONFIG[section][key] = value
            else:
                raise ValueError(f"セクション '{section}' は辞書型ではありません")
        else:
            UNIFIED_OUTPUT_CONFIG[section] = {key: value}
            
    except Exception as e:
        raise Exception(f"実行時設定更新中にエラー: {e}")

def get_effective_config(user_config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    実効設定の取得（デフォルト + ユーザー設定）
    
    Args:
        user_config_path: ユーザー設定ファイルパス
        
    Returns:
        Dict[str, Any]: 実効設定
    """
    effective_config = get_config()
    
    if user_config_path:
        try:
            user_config = load_config_from_file(user_config_path)
            effective_config = merge_configs(effective_config, user_config)
        except Exception as e:
            # ユーザー設定の読み込みに失敗してもデフォルト設定で継続
            print(f"ユーザー設定読み込み警告: {e}")
    
    return effective_config


if __name__ == "__main__":
    # 設定テスト
    print("=== 統一出力エンジン設定テスト ===\n")
    
    # 基本設定の表示
    config = get_config()
    print(f"エンジン名: {config['engine_info']['name']}")
    print(f"バージョン: {config['engine_info']['version']}")
    print(f"フェーズ: {config['engine_info']['phase']}")
    
    # 出力形式設定の表示
    print(f"\n有効な出力形式: {config['output_formats']['enabled_formats']}")
    print(f"デフォルト出力形式: {config['output_formats']['default_formats']}")
    
    # 品質保証設定の表示
    qa_config = config['quality_assurance']
    print(f"\n品質保証有効: {qa_config['enabled']}")
    print(f"強制品質向上: {qa_config['force_enhanced_extraction']}")
    
    # 設定検証のテスト
    print("\n=== 設定検証テスト ===")
    validation_result = validate_config(config)
    print(f"設定妥当性: {validation_result['is_valid']}")
    if validation_result['errors']:
        print(f"エラー: {validation_result['errors']}")
    if validation_result['warnings']:
        print(f"警告: {validation_result['warnings']}")
    
    # 特定設定の取得テスト
    print("\n=== 特定設定取得テスト ===")
    excel_config = get_output_format_config('excel')
    print(f"Excel設定 - 既存エクスポーター使用: {excel_config.get('use_existing_exporters', False)}")
    print(f"Excel設定 - 拡張抽出: {excel_config.get('enhanced_extraction', False)}")
    
    template_config = get_template_config()
    print(f"テンプレートディレクトリ: {template_config.get('template_directory', '')}")
    
    # 実行時設定更新のテスト
    print("\n=== 実行時設定更新テスト ===")
    original_log_level = config['debugging']['log_level']
    update_config_at_runtime('debugging', 'log_level', 'DEBUG')
    updated_config = get_config()
    print(f"ログレベル更新: {original_log_level} -> {updated_config['debugging']['log_level']}")
    
    print("\n=== 設定テスト完了 ===")

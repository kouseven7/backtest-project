"""
Dashboard Configuration Manager for 4-3-2
ダッシュボード設定管理システム
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """ダッシュボード設定管理クラス"""
    
    # 更新設定
    update_interval_minutes: int = 15
    low_frequency_interval_minutes: int = 60
    enable_interval_switching: bool = True
    
    # データ保持設定
    data_retention_days: int = 90  # 3ヶ月
    chart_retention_days: int = 7
    
    # 表示設定
    enable_alerts: bool = True
    max_alerts_display: int = 5
    enable_risk_dashboard: bool = True
    
    # パフォーマンス設定
    max_strategies_display: int = 10
    enable_caching: bool = True
    cache_timeout_minutes: int = 5
    
    # 出力設定
    generate_html_reports: bool = True
    generate_chart_images: bool = True
    generate_text_summaries: bool = True
    
    # ファイルパス設定
    base_output_dir: str = "logs/dashboard"
    template_dir: str = "visualization/dashboard_templates"
    config_dir: str = "config"
    
    # アラート閾値
    risk_score_threshold_low: float = 50.0
    sharpe_ratio_threshold_low: float = 0.5
    concentration_risk_threshold_high: float = 50.0
    
    # 追加設定
    additional_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初期化後処理"""
        # ディレクトリ作成
        self._ensure_directories()
        
    def _ensure_directories(self):
        """必要ディレクトリの作成"""
        paths = [
            self.base_output_dir,
            f"{self.base_output_dir}/performance_data",
            f"{self.base_output_dir}/dashboard_reports", 
            f"{self.base_output_dir}/chart_images",
            f"{self.base_output_dir}/data_archive"
        ]
        
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def load_from_file(cls, config_file: str) -> 'DashboardConfig':
        """設定ファイルからの読み込み"""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_file}, using defaults")
                return cls()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # データクラスフィールドと一致するもののみ適用
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
            filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}
            
            # additional_settingsに残りを格納
            additional = {k: v for k, v in config_data.items() if k not in valid_fields}
            if additional:
                filtered_data['additional_settings'] = additional
            
            return cls(**filtered_data)
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            return cls()
    
    def save_to_file(self, config_file: str) -> bool:
        """設定ファイルへの保存"""
        try:
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # データクラスを辞書に変換
            config_dict = {
                'update_interval_minutes': self.update_interval_minutes,
                'low_frequency_interval_minutes': self.low_frequency_interval_minutes,
                'enable_interval_switching': self.enable_interval_switching,
                'data_retention_days': self.data_retention_days,
                'chart_retention_days': self.chart_retention_days,
                'enable_alerts': self.enable_alerts,
                'max_alerts_display': self.max_alerts_display,
                'enable_risk_dashboard': self.enable_risk_dashboard,
                'max_strategies_display': self.max_strategies_display,
                'enable_caching': self.enable_caching,
                'cache_timeout_minutes': self.cache_timeout_minutes,
                'generate_html_reports': self.generate_html_reports,
                'generate_chart_images': self.generate_chart_images,
                'generate_text_summaries': self.generate_text_summaries,
                'base_output_dir': self.base_output_dir,
                'template_dir': self.template_dir,
                'config_dir': self.config_dir,
                'risk_score_threshold_low': self.risk_score_threshold_low,
                'sharpe_ratio_threshold_low': self.sharpe_ratio_threshold_low,
                'concentration_risk_threshold_high': self.concentration_risk_threshold_high
            }
            
            # 追加設定をマージ
            config_dict.update(self.additional_settings)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Config saved to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to {config_file}: {e}")
            return False
    
    def get_alert_thresholds(self) -> Dict[str, float]:
        """アラート閾値の取得"""
        return {
            'risk_score_low': self.risk_score_threshold_low,
            'sharpe_ratio_low': self.sharpe_ratio_threshold_low,
            'concentration_risk_high': self.concentration_risk_threshold_high
        }
    
    def get_output_paths(self) -> Dict[str, Path]:
        """出力パス一覧の取得"""
        base = Path(self.base_output_dir)
        return {
            'base': base,
            'performance_data': base / 'performance_data',
            'dashboard_reports': base / 'dashboard_reports',
            'chart_images': base / 'chart_images',
            'data_archive': base / 'data_archive',
            'templates': Path(self.template_dir)
        }
    
    def is_low_frequency_mode(self, current_time: Optional[int] = None) -> bool:
        """低頻度モードの判定"""
        if not self.enable_interval_switching:
            return False
        
        # 簡易実装：平日営業時間外は低頻度
        from datetime import datetime
        
        now = datetime.now() if current_time is None else datetime.fromtimestamp(current_time)
        
        # 土日は低頻度
        if now.weekday() >= 5:
            return True
            
        # 営業時間外（夜間・早朝）は低頻度
        if now.hour < 8 or now.hour > 20:
            return True
            
        return False
    
    def get_current_update_interval(self) -> int:
        """現在の更新間隔を取得"""
        if self.is_low_frequency_mode():
            return self.low_frequency_interval_minutes
        return self.update_interval_minutes
    
    def validate_config(self) -> List[str]:
        """設定値の検証"""
        errors = []
        
        # 数値範囲チェック
        if not (1 <= self.update_interval_minutes <= 1440):
            errors.append("update_interval_minutes must be between 1 and 1440")
        
        if not (1 <= self.data_retention_days <= 365):
            errors.append("data_retention_days must be between 1 and 365")
        
        if not (1 <= self.max_strategies_display <= 50):
            errors.append("max_strategies_display must be between 1 and 50")
        
        if not (0.0 <= self.risk_score_threshold_low <= 100.0):
            errors.append("risk_score_threshold_low must be between 0.0 and 100.0")
        
        # ディレクトリ存在チェック
        required_dirs = [self.base_output_dir, self.config_dir]
        for dir_path in required_dirs:
            if not Path(dir_path).parent.exists():
                errors.append(f"Parent directory does not exist: {dir_path}")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """設定サマリーの取得"""
        return {
            'update_settings': {
                'interval_minutes': self.update_interval_minutes,
                'low_frequency_minutes': self.low_frequency_interval_minutes,
                'interval_switching': self.enable_interval_switching
            },
            'data_retention': {
                'data_days': self.data_retention_days,
                'chart_days': self.chart_retention_days
            },
            'display_features': {
                'alerts': self.enable_alerts,
                'risk_dashboard': self.enable_risk_dashboard,
                'max_strategies': self.max_strategies_display
            },
            'output_generation': {
                'html_reports': self.generate_html_reports,
                'chart_images': self.generate_chart_images,
                'text_summaries': self.generate_text_summaries
            },
            'thresholds': self.get_alert_thresholds()
        }

# ファクトリー関数
def load_dashboard_config(config_file: Optional[str] = None) -> DashboardConfig:
    """ダッシュボード設定読み込み"""
    if config_file is None:
        config_file = "config/dashboard_config.json"
    
    return DashboardConfig.load_from_file(config_file)

def create_default_config_file(config_file: str = "config/dashboard_config.json") -> bool:
    """デフォルト設定ファイルの作成"""
    try:
        default_config = DashboardConfig()
        return default_config.save_to_file(config_file)
    except Exception as e:
        logger.error(f"Failed to create default config: {e}")
        return False

if __name__ == "__main__":
    # テスト・デモ実行
    print("=== Dashboard Configuration Test ===")
    
    # デフォルト設定作成
    config = DashboardConfig()
    print("Default config created")
    print(f"Update interval: {config.update_interval_minutes} minutes")
    print(f"Data retention: {config.data_retention_days} days")
    
    # 設定検証
    errors = config.validate_config()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("Configuration is valid")
    
    # 設定サマリー
    summary = config.get_config_summary()
    print(f"Configuration summary: {json.dumps(summary, indent=2)}")
    
    # ファイル保存テスト
    test_config_file = "test_dashboard_config.json"
    if config.save_to_file(test_config_file):
        print(f"Config saved to {test_config_file}")
        
        # 読み込みテスト
        loaded_config = DashboardConfig.load_from_file(test_config_file)
        print(f"Config loaded, interval: {loaded_config.update_interval_minutes}")
        
        # クリーンアップ
        Path(test_config_file).unlink(missing_ok=True)

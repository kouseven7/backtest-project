"""
設定ファイル - VaR計算システム

デフォルト設定値の定義
"""

# VaR計算設定
VAR_CALCULATION_CONFIG = {
    # 基本設定
    "lookback_period": 252,  # 1年
    "confidence_levels": [0.95, 0.99],
    "monte_carlo_simulations": 10000,
    
    # ハイブリッド計算重み
    "hybrid_weights": {
        "parametric": 0.3,
        "historical": 0.4,
        "monte_carlo": 0.3
    },
    
    # 市場レジーム設定
    "regime_detection": {
        "volatility_window": 60,  # 2ヶ月
        "high_vol_threshold": 0.02,  # 2%
        "low_vol_threshold": 0.01   # 1%
    },
    
    # 適応的時間ウィンドウ
    "adaptive_windows": {
        "min_window": 60,
        "max_window": 252,
        "regime_based_adjustment": True
    }
}

# 監視システム設定
MONITORING_CONFIG = {
    # 基本監視設定
    "monitoring_interval": 300,  # 5分（秒）
    
    # VaR閾値
    "thresholds": {
        "var_95": 0.05,   # 5%
        "var_99": 0.08,   # 8%
        "warning_ratio": 0.8,    # 閾値の80%で警告
        "critical_ratio": 1.2    # 閾値の120%でクリティカル
    },
    
    # アラート設定
    "alerts": {
        "enable_email": False,
        "enable_log": True,
        "enable_system_integration": True
    },
    
    # データ保持設定
    "data_retention": {
        "monitoring_history_days": 90,
        "alert_history_days": 30
    }
}

# バックテスト設定
BACKTEST_CONFIG = {
    # 基本設定
    "lookback_window": 252,
    "rolling_window": 60,
    "min_observations": 30,
    
    # 統計検定設定
    "statistical_tests": {
        "confidence_level": 0.05,
        "enable_kupiec": True,
        "enable_christoffersen": True
    },
    
    # 性能評価
    "performance_evaluation": {
        "enable_model_comparison": True,
        "enable_regime_analysis": True
    }
}

# 統合ブリッジ設定
INTEGRATION_CONFIG = {
    # レガシーシステム統合
    "legacy_integration": {
        "enable_comparison": True,
        "comparison_tolerance": 0.1,  # 10%
        "prefer_advanced": True
    },
    
    # ログ設定
    "logging": {
        "log_comparisons": True,
        "log_discrepancies": True
    },
    
    # ハイブリッド統合
    "hybrid_integration": {
        "merge_calculation_methods": True,
        "use_hybrid_recommendations": True
    }
}

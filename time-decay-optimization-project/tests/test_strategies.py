my_backtest_project/
│
├── config/
│   ├── logger_config.py
│   ├── strategy_characteristics_data_loader.py
│   └── strategy_data_persistence.py
│
├── data/
│   └── (data files)
│
├── indicators/
│   ├── trend_analysis.py
│   └── volume_indicators.py
│
├── metrics/
│   ├── performance_metrics.py
│   └── performance_metrics_util.py
│
├── optimization/
│   ├── optimize_vwap_bounce_strategy.py
│   ├── optimize_vwap_breakout_strategy.py
│   ├── parallel_optimizer.py
│   └── time_decay_optimizer.py
│
├── strategies/
│   ├── VWAP_Bounce.py
│   └── VWAP_Breakout.py
│
├── tests/
│   ├── test_time_decay_optimizer.py
│   └── test_strategy_data_loader.py
│
└── main.py
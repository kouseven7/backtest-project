# DSSMS出力システム詳細設計書

**作成日**: 2025-12-03  
**フェーズ**: Phase 1 - 詳細設計  
**ステータス**: 設計完了、実装待ち

---

## 1. 設計概要

### 1.1 目的

DSSMS（Dynamic Symbol Selection Multi-Strategy）バックテスト実行時に、単一フォルダ内に包括的なレポートを生成するシステムを設計する。

### 1.2 確定事項

以下の仕様がユーザーにより確定済み（2025-12-03）:

1. **銘柄切替履歴CSV**: 必要
   - カラム: 切替日、前の銘柄、新銘柄、切替理由、コスト
2. **命名規則**: `dssms_{timestamp}` (例: `dssms_20251203_131025`)
3. **アプローチ**: A（ComprehensiveReporter再利用）
4. **複数銘柄対応**: タイムスタンプ使用
5. **出力制御**:
   - DSSMS実行時: `main_new.py`のレポート生成を呼び出さない
   - `main_new.py`単体実行時: 従来通り`output/comprehensive_reports`に出力

---

## 2. データ構造分析

### 2.1 DSSMS側データ構造

#### 2.1.1 `daily_result` (Line 429-498)

```python
{
    'date': '2024-01-04',                    # str
    'symbol': '5803.T',                      # str
    'success': True,                         # bool
    'portfolio_value_start': 1000000,        # float
    'portfolio_value_end': 1050000,          # float
    'portfolio_value': 1050000,              # float (互換性用)
    'daily_return': 50000,                   # float (金額)
    'daily_return_rate': 0.05,               # float (比率)
    'strategy_results': {...},               # Dict (main_new.py実行結果変換後)
    'switch_executed': False,                # bool
    'execution_time_ms': 123.45,             # float
    'errors': []                             # List[str]
}
```

**データソース**: `dssms_integrated_main.py` Line 429-500  
**データ収集**: `self.daily_results.append(daily_result)` (Line 381)

#### 2.1.2 `switch_result` (Line 1383-1423)

```python
{
    'date': '2024-01-04',                       # str
    'from_symbol': '5803.T',                    # str
    'to_symbol': '9101.T',                      # str
    'switch_executed': True,                    # bool
    'switch_cost': 1000,                        # float (金額)
    'reason': 'dss_optimization',               # str
    'portfolio_value_after_switch': 999000,     # float
    'executed_date': datetime(2024, 1, 4),      # datetime
    'close_result': {...},                      # Dict
    'open_result': {...}                        # Dict
}
```

**データソース**: `dssms_integrated_main.py` Line 1383-1423  
**データ収集**: `self.switch_history.append(switch_result)` (Line 453)

#### 2.1.3 `final_results` (Line 2055-2189)

```python
{
    'status': 'success',
    'execution_time_sec': 45.67,
    'total_trading_days': 12,
    'successful_days': 10,
    'daily_results': [daily_result1, daily_result2, ...],
    'switch_history': [switch_result1, switch_result2, ...],
    'summary': {
        'total_return': 903800,
        'return_rate': 0.9038,
        'max_drawdown': -0.05,
        ...
    },
    ...
}
```

**データソース**: `dssms_integrated_main.py` Line 2055-2189

---

### 2.2 main_new.py側データ構造

#### 2.2.1 `execution_results` (IntegratedExecutionManager出力)

```python
{
    'status': 'SUCCESS',                        # str
    'total_strategies': 3,                      # int
    'successful_strategies': 2,                 # int
    'failed_strategies': 1,                     # int
    'weighted_performance': 0.15,               # float
    'total_portfolio_value': 1050000,           # float
    'execution_results': [                      # List[Dict]
        {
            'strategy_name': 'VWAPBreakoutStrategy',
            'weight': 0.5,
            'success': True,
            'execution_details': [              # List[Dict]
                {
                    'date': '2024-01-04',
                    'action': 'BUY',
                    'price': 1234.5,
                    'shares': 100,
                    'status': 'executed',
                    ...
                },
                ...
            ],
            ...
        },
        ...
    ],
    'execution_details': [...],                 # 全戦略統合 (Phase 5-B-6)
    'strategy_weights': {'VWAPBreakoutStrategy': 0.5, ...},
    'equity_recorder': <EquityCurveRecorder>
}
```

**データソース**: `integrated_execution_manager.py` Line 451-550

---

## 3. データ変換ロジック設計

### 3.1 `_convert_to_execution_format()` I/O仕様

#### 3.1.1 関数シグネチャ

```python
def _convert_to_execution_format(
    self,
    final_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    DSSMS final_resultsをmain_new.py execution_results形式に変換
    
    Args:
        final_results: DSSMSバックテストの最終結果
            - daily_results: List[Dict] (日次実行結果)
            - switch_history: List[Dict] (銘柄切替履歴)
            - summary: Dict (サマリー統計)
    
    Returns:
        execution_results: ComprehensiveReporter互換形式
            - status: str
            - execution_results: List[Dict] (戦略別結果)
            - execution_details: List[Dict] (全取引詳細)
            - strategy_weights: Dict[str, float]
            - total_portfolio_value: float
    """
```

#### 3.1.2 変換マッピング

| DSSMS (入力) | main_new.py (出力) | 変換ロジック |
|---|---|---|
| `final_results['status']` | `execution_results['status']` | 'success' → 'SUCCESS'<br>'error' → 'ERROR' |
| `final_results['daily_results']` | `execution_results['execution_details']` | 各daily_resultから取引詳細を抽出<br>（後述の3.1.3参照） |
| `final_results['summary']['total_return']` | `execution_results['total_portfolio_value']` | `initial_capital + total_return` |
| N/A (DSSMS単一戦略想定) | `execution_results['strategy_weights']` | `{'DSSMS_MultiStrategy': 1.0}` |
| N/A | `execution_results['execution_results']` | 単一戦略として構築（後述の3.1.4参照） |

#### 3.1.3 `execution_details`抽出ロジック

**問題**: `daily_results`には`strategy_results`（main_new.pyの結果変換後）が含まれるが、`execution_details`は保持されていない。

**解決策**: 
1. `_execute_multi_strategies()`で`main_new.py`の`execution_results`を保持
2. `_convert_main_new_result()`で`execution_details`を抽出して`daily_result`に追加
3. `_convert_to_execution_format()`で全日次の`execution_details`を統合

**実装詳細**:

```python
# dssms_integrated_main.py Line 1527-1623の修正
def _convert_main_new_result(self, main_new_result, symbol, target_date):
    """main_new.pyの結果をDSSMS形式に変換（修正版）"""
    
    execution_results = main_new_result.get('execution_results', {})
    
    # CRITICAL: execution_detailsを保持
    execution_details = execution_results.get('execution_details', [])
    
    dssms_result = {
        'status': 'success',
        'symbol': symbol,
        'date': target_date.strftime('%Y-%m-%d'),
        'execution_details': execution_details,  # 追加: 取引詳細保持
        'strategy_results': {...},
        'position_update': {...}
    }
    
    return dssms_result
```

```python
# _convert_to_execution_format()の実装
def _convert_to_execution_format(self, final_results):
    """DSSMS結果をmain_new.py形式に変換"""
    
    daily_results = final_results.get('daily_results', [])
    
    # 全日次のexecution_detailsを統合
    all_execution_details = []
    for daily_result in daily_results:
        execution_details = daily_result.get('execution_details', [])
        all_execution_details.extend(execution_details)
    
    return {
        'status': 'SUCCESS',
        'execution_details': all_execution_details,  # 統合済み取引詳細
        'total_portfolio_value': final_results['summary']['final_portfolio_value'],
        'execution_results': [{  # 単一戦略として構築
            'strategy_name': 'DSSMS_MultiStrategy',
            'weight': 1.0,
            'success': True,
            'execution_details': all_execution_details
        }],
        'strategy_weights': {'DSSMS_MultiStrategy': 1.0}
    }
```

#### 3.1.4 EquityCurve再構築ロジック

**課題**: ComprehensiveReporterは`equity_recorder`を使用して`portfolio_equity_curve.csv`を生成する。

**解決策**: `daily_results`から日次ポートフォリオデータを抽出し、EquityCurveRecorder互換形式で再構築。

```python
def _rebuild_equity_curve(self, daily_results):
    """daily_resultsからequity_curve DataFrameを再構築"""
    
    equity_data = []
    for daily_result in daily_results:
        equity_data.append({
            'date': pd.Timestamp(daily_result['date']),
            'portfolio_value': daily_result['portfolio_value_end'],
            'cash_balance': daily_result.get('cash_balance', 0),  # 要追加
            'position_value': daily_result.get('position_value', 0),  # 要追加
            'peak_value': daily_result.get('peak_value', daily_result['portfolio_value_end']),  # 要追加
            'drawdown_pct': daily_result.get('drawdown_pct', 0),  # 要追加
            'cumulative_pnl': daily_result.get('cumulative_pnl', 0),  # 要追加
            'daily_pnl': daily_result['daily_return'],
            'total_trades': daily_result.get('total_trades', 0),  # 要追加
            'active_positions': daily_result.get('active_positions', 0),  # 要追加
            'risk_status': daily_result.get('risk_status', 'Normal'),  # 要追加
            'blocked_trades': daily_result.get('blocked_trades', 0),  # 要追加
            'risk_action': daily_result.get('risk_action', ''),  # 要追加
            'snapshot_type': 'daily'
        })
    
    return pd.DataFrame(equity_data).set_index('date')
```

**重要**: 上記の「要追加」カラムは現在`daily_result`に含まれていないため、`_process_daily_trading()`の修正が必要。

---

## 4. CSV仕様確定

### 4.1 基本CSV（ComprehensiveReporter互換）

#### 4.1.1 `dssms_portfolio_equity_curve.csv` (13カラム)

| カラム名 | 型 | 説明 | データソース |
|---|---|---|---|
| date | datetime | 日付 | `daily_result['date']` |
| portfolio_value | float | ポートフォリオ総額 | `daily_result['portfolio_value_end']` |
| cash_balance | float | 現金残高 | **要追加** |
| position_value | float | ポジション評価額 | **要追加** |
| peak_value | float | ピーク値 | **要追加** |
| drawdown_pct | float | ドローダウン率 | **要追加** |
| cumulative_pnl | float | 累積損益 | **要追加** |
| daily_pnl | float | 日次損益 | `daily_result['daily_return']` |
| total_trades | int | 総取引数 | **要追加** |
| active_positions | int | アクティブポジション数 | **要追加** |
| risk_status | str | リスクステータス | **要追加** |
| blocked_trades | int | ブロックされた取引数 | **要追加** |
| risk_action | str | リスクアクション | **要追加** |
| snapshot_type | str | スナップショットタイプ | 'daily' (固定) |

**実装優先度**: 高（ComprehensiveReporter再利用のため必須）

#### 4.1.2 `dssms_trades.csv` (13カラム、DSSMS拡張版)

| カラム名 | 型 | 説明 | データソース |
|---|---|---|---|
| **symbol** | str | **銘柄コード** | **daily_result['symbol']** (DSSMS拡張) |
| entry_date | datetime | エントリー日 | `execution_detail['date']` (action='BUY') |
| exit_date | datetime | エグジット日 | `execution_detail['date']` (action='SELL') |
| entry_price | float | エントリー価格 | `execution_detail['price']` (BUY) |
| exit_price | float | エグジット価格 | `execution_detail['price']` (SELL) |
| shares | int | 株数 | `execution_detail['shares']` |
| pnl | float | 損益 | `(exit_price - entry_price) * shares` |
| return_pct | float | リターン率 | `pnl / (entry_price * shares)` |
| holding_period_days | int | 保有期間 | `(exit_date - entry_date).days` |
| strategy | str | 戦略名 | `execution_detail['strategy']` |
| position_value | float | ポジション価値 | `entry_price * shares` |
| is_forced_exit | bool | 強制決済フラグ | `execution_detail['status'] == 'force_closed'` |
| is_executed_trade | bool | 実行済み取引フラグ | `execution_detail['status'] == 'executed'` |

**実装優先度**: 高（取引履歴の基本）

#### 4.1.3 `dssms_performance_summary.csv` (14メトリクス)

| メトリクス名 | 型 | 説明 | データソース |
|---|---|---|---|
| initial_capital | float | 初期資本 | `config['initial_capital']` (デフォルト: 1,000,000) |
| final_portfolio_value | float | 最終ポートフォリオ価値 | `final_results['summary']['final_portfolio_value']` |
| total_return | float | 総リターン | `final_results['summary']['total_return']` |
| win_rate | float | 勝率 | `final_results['summary']['win_rate']` |
| winning_trades | int | 勝ちトレード数 | `final_results['summary']['winning_trades']` |
| losing_trades | int | 負けトレード数 | `final_results['summary']['losing_trades']` |
| avg_profit | float | 平均利益 | `final_results['summary']['avg_profit']` |
| avg_loss | float | 平均損失 | `final_results['summary']['avg_loss']` |
| max_profit | float | 最大利益 | `final_results['summary']['max_profit']` |
| max_loss | float | 最大損失 | `final_results['summary']['max_loss']` |
| total_profit | float | 総利益 | `final_results['summary']['total_profit']` |
| total_loss | float | 総損失 | `final_results['summary']['total_loss']` |
| net_profit | float | 純利益 | `final_results['summary']['net_profit']` |
| profit_factor | float | プロフィットファクター | `final_results['summary']['profit_factor']` |

**実装優先度**: 中（既存のComprehensiveReporterロジック流用可能）

---

### 4.2 DSSMS特有CSV

#### 4.2.1 `dssms_switch_history.csv` (5カラム + 追加3カラム)

| カラム名 | 型 | 説明 | データソース |
|---|---|---|---|
| switch_date | datetime | 切替日 | `switch_result['date']` |
| from_symbol | str | 前の銘柄 | `switch_result['from_symbol']` |
| to_symbol | str | 新銘柄 | `switch_result['to_symbol']` |
| reason | str | 切替理由 | `switch_result['reason']` |
| switch_cost | float | コスト（金額） | `switch_result['switch_cost']` |
| **ranking_score** | float | **DSS選択スコア** | **要追加** (optional) |
| **portfolio_value_before** | float | **切替前ポートフォリオ価値** | **要追加** (optional) |
| **portfolio_value_after** | float | **切替後ポートフォリオ価値** | `switch_result['portfolio_value_after_switch']` |

**ユーザー確定カラム** (2025-12-03):
1. 切替日 → `switch_date`
2. 前の銘柄 → `from_symbol`
3. 新銘柄 → `to_symbol`
4. 切替理由 → `reason`
5. コスト → `switch_cost`

**追加推奨カラム**:
- `ranking_score`: DSS Coreの選択スコア（詳細分析用）
- `portfolio_value_before/after`: 切替前後の資産推移追跡用

**実装優先度**: 高（ユーザー要求の必須機能）

---

### 4.3 `daily_result`への追加カラム設計

`_process_daily_trading()`（Line 418-500）の修正が必要:

```python
def _process_daily_trading(self, target_date, target_symbols):
    """日次取引処理（修正版）"""
    
    # ... 既存処理 ...
    
    daily_result = {
        'date': target_date.strftime('%Y-%m-%d'),
        'symbol': self.current_symbol,
        'success': False,
        'portfolio_value_start': self.portfolio_value,
        'daily_return': 0,
        'daily_return_rate': 0,
        'strategy_results': {},
        'switch_executed': False,
        'errors': [],
        
        # === Phase 1追加: equity_curve再構築用カラム ===
        'cash_balance': 0,                   # 要実装
        'position_value': 0,                 # 要実装
        'peak_value': self.peak_value,       # 要実装: self.peak_valueを追跡
        'drawdown_pct': 0,                   # 要実装
        'cumulative_pnl': 0,                 # 要実装: self.cumulative_pnlを追跡
        'total_trades': 0,                   # 要実装
        'active_positions': 0,               # 要実装
        'risk_status': 'Normal',             # 要実装
        'blocked_trades': 0,                 # 要実装
        'risk_action': '',                   # 要実装
        
        # === Phase 1追加: execution_details保持 ===
        'execution_details': []              # _convert_main_new_result()で設定
    }
    
    # ... 既存処理 ...
    
    # ポートフォリオ価値更新時にpeak_value/drawdown計算
    if strategy_result.get('position_update'):
        # ... 既存処理 ...
        
        # peak_value更新
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        # drawdown計算
        daily_result['peak_value'] = self.peak_value
        daily_result['drawdown_pct'] = (self.peak_value - self.portfolio_value) / self.peak_value
        
        # cumulative_pnl更新
        self.cumulative_pnl += position_return
        daily_result['cumulative_pnl'] = self.cumulative_pnl
    
    # ... 既存処理 ...
    
    return daily_result
```

**実装優先度**: 最高（equity_curve再構築の基盤）

---

## 5. 出力制御設計

### 5.1 main_new.pyレポート生成抑制

**課題**: DSSMS実行時に`main_new.py`が呼び出すComprehensiveReporterが日次でフォルダを生成する。

**解決策**: `main_new.py`に`suppress_report_generation`フラグを追加し、DSSMS経由の呼び出し時にレポート生成をスキップ。

#### 5.1.1 MainSystemController修正

```python
# main_new.py Line 49
class MainSystemController:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.suppress_report_generation = self.config.get('suppress_report_generation', False)
        # ... 既存処理 ...
```

```python
# main_new.py Line 330-333
# 7. 包括的レポート生成（抑制制御追加）
if not self.suppress_report_generation:
    self.logger.info(f"[STEP 7/7] 包括的レポート生成")
    report_results = self.reporter.generate_full_backtest_report(
        execution_results, data_for_analysis, ticker, config=None
    )
else:
    self.logger.info(f"[STEP 7/7] レポート生成スキップ（suppress_report_generation=True）")
    report_results = {
        'status': 'SUPPRESSED',
        'message': 'Report generation suppressed by caller'
    }
```

#### 5.1.2 dssms_integrated_main.py修正

```python
# dssms_integrated_main.py Line 1458-1477
config = {
    'execution': {
        'execution_mode': 'simple',
        'broker': {
            'initial_cash': self.config.get('initial_capital', 1000000),
            'commission_per_trade': 1.0
        }
    },
    'risk_management': {
        'use_enhanced_risk': False,
        'max_drawdown_threshold': 0.15
    },
    'performance': {
        'use_aggregator': False
    },
    'suppress_report_generation': True  # DSSMS経由の呼び出し時はレポート生成抑制
}
```

**実装優先度**: 最高（ユーザー要求の必須機能）

---

### 5.2 main_new.py単体実行時の通常動作保証

**確認事項**: `suppress_report_generation`のデフォルト値を`False`に設定することで、main_new.py単体実行時は従来通りレポート生成される。

**検証方法**:
1. `python main_new.py`を実行
2. `output/comprehensive_reports/{ticker}_{timestamp}/`フォルダが生成されることを確認
3. 8ファイルが正常に出力されることを確認

**実装優先度**: 高（既存機能の後方互換性保証）

---

## 6. 実装計画

### Phase 2: 実装（次ステップ）

#### 6.1 データ変換処理実装

**ファイル**: `src/dssms/dssms_integrated_main.py`

**実装タスク**:
1. `_convert_main_new_result()`修正: `execution_details`を`daily_result`に保持
2. `_convert_to_execution_format()`新規実装: DSSMS結果をmain_new.py形式に変換
3. `_rebuild_equity_curve()`新規実装: equity_curve DataFrameを再構築

**推定工数**: 4時間

---

#### 6.2 daily_result拡張実装

**ファイル**: `src/dssms/dssms_integrated_main.py`

**実装タスク**:
1. `__init__()`に`self.peak_value`, `self.cumulative_pnl`追加
2. `_process_daily_trading()`修正: 13カラム追加
3. ポートフォリオ更新ロジック修正: peak_value/drawdown計算

**推定工数**: 3時間

---

#### 6.3 CSV生成処理実装（ComprehensiveReporter再利用）

**ファイル**: `src/dssms/dssms_integrated_main.py`

**実装タスク**:
1. `_generate_outputs()`修正:
   - `_convert_to_execution_format()`を呼び出し
   - ComprehensiveReporterを初期化（DSSMS専用出力先設定）
   - `generate_full_backtest_report()`を呼び出し
2. 出力先ディレクトリ: `output/dssms_integration/dssms_{timestamp}/`

**推定工数**: 2時間

---

#### 6.4 DSSMS特有ファイル（switch_history.csv）実装

**ファイル**: `src/dssms/dssms_integrated_main.py`

**実装タスク**:
1. `_generate_switch_history_csv()`新規実装:
   - `self.switch_history`からCSV生成
   - カラム: switch_date, from_symbol, to_symbol, reason, switch_cost
   - オプショナルカラム: ranking_score, portfolio_value_before, portfolio_value_after
2. `_generate_outputs()`に組み込み

**推定工数**: 2時間

---

#### 6.5 出力制御実装

**ファイル**: `main_new.py`, `src/dssms/dssms_integrated_main.py`

**実装タスク**:
1. MainSystemController修正: `suppress_report_generation`フラグ追加
2. レポート生成ステップ修正: 条件分岐追加
3. dssms_integrated_main.py修正: `suppress_report_generation=True`設定

**推定工数**: 1時間

---

**Phase 2合計推定工数**: 12時間

---

### Phase 3: テスト（最終ステップ）

#### 3.1 単体テスト

**テスト対象**:
1. `_convert_to_execution_format()`: 入出力データ型検証
2. `_rebuild_equity_curve()`: DataFrame構造検証
3. `_generate_switch_history_csv()`: CSV出力検証

**推定工数**: 3時間

---

#### 3.2 統合テスト

**テストシナリオ**:
1. DSSMS 1ヶ月バックテスト実行
2. 出力フォルダ確認: `output/dssms_integration/dssms_{timestamp}/`
3. 10ファイル生成確認:
   - CSV: 4ファイル（equity_curve, trades, performance, switch_history）
   - TXT: 2ファイル（SUMMARY, comprehensive_report）
   - JSON: 4ファイル（trade_analysis, performance_metrics, execution_results, comprehensive_analysis）

**検証項目**:
- [ ] フォルダ数: 1個のみ（84個→1個に削減）
- [ ] ファイル数: 10個
- [ ] 各CSVのレコード数が正常
- [ ] switch_history.csvに銘柄切替履歴が記録されている
- [ ] `output/comprehensive_reports`に日次フォルダが生成されていない

**推定工数**: 2時間

---

#### 3.3 比較テスト

**テストシナリオ**:
1. main_new.py単体実行
2. 出力先確認: `output/comprehensive_reports/{ticker}_{timestamp}/`
3. 8ファイル生成確認

**検証項目**:
- [ ] フォルダ生成先が従来通り
- [ ] ファイル数: 8個（DSSMS特有の2ファイルなし）
- [ ] ファイル内容が従来と一致

**推定工数**: 1時間

---

**Phase 3合計推定工数**: 6時間

---

## 7. 要追加実装の詳細

### 7.1 `__init__()`修正

```python
# dssms_integrated_main.py Line 114-169
def __init__(self, config: Optional[Dict[str, Any]] = None):
    # ... 既存処理 ...
    
    # Phase 1追加: ポートフォリオ追跡変数
    self.peak_value = self.portfolio_value  # ピーク値追跡
    self.cumulative_pnl = 0.0               # 累積損益追跡
```

---

### 7.2 `_convert_main_new_result()`修正

```python
# dssms_integrated_main.py Line 1527-1623
def _convert_main_new_result(self, main_new_result, symbol, target_date):
    """main_new.pyの結果をDSSMS形式に変換（Phase 1修正版）"""
    
    # ... 既存処理 ...
    
    execution_results = main_new_result.get('execution_results', {})
    
    # CRITICAL: execution_detailsを保持
    execution_details = execution_results.get('execution_details', [])
    
    # CRITICAL: main_new.pyのexecution_resultsを完全保持
    main_execution_results_full = execution_results  # ComprehensiveReporter再利用用
    
    dssms_result = {
        'status': 'success',
        'symbol': symbol,
        'date': target_date.strftime('%Y-%m-%d'),
        'execution_details': execution_details,              # Phase 1追加
        'main_execution_results': main_execution_results_full,  # Phase 1追加
        'strategy_results': {...},
        'position_update': {...}
    }
    
    return dssms_result
```

---

### 7.3 `_generate_outputs()`修正

```python
# dssms_integrated_main.py Line 2231-2247
def _generate_outputs(self, final_results: Dict[str, Any]) -> None:
    """出力ファイル生成（Phase 1実装版）"""
    
    try:
        # タイムスタンプ生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 出力ディレクトリ作成
        output_dir = Path("output/dssms_integration") / f"dssms_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"[DSSMS_OUTPUT] 出力先: {output_dir}")
        
        # 1. DSSMS結果をmain_new.py形式に変換
        execution_results = self._convert_to_execution_format(final_results)
        
        # 2. ComprehensiveReporter初期化（DSSMS専用出力先設定）
        from main_system.reporting.comprehensive_reporter import ComprehensiveReporter
        reporter = ComprehensiveReporter(output_base_dir=str(output_dir.parent))
        
        # 3. stock_data再構築（equity_curve生成用）
        stock_data = self._rebuild_stock_data_for_reporting(final_results)
        
        # 4. ComprehensiveReporterでレポート生成
        # 注: tickerは"DSSMS"固定（複数銘柄を扱うため個別銘柄名は不適切）
        reporter.generate_full_backtest_report(
            execution_results=execution_results,
            stock_data=stock_data,
            ticker="DSSMS",
            config=None
        )
        
        # 5. DSSMS特有ファイル生成
        self._generate_switch_history_csv(final_results, output_dir)
        
        self.logger.info(f"[DSSMS_OUTPUT] 出力完了: {output_dir}")
        
    except Exception as e:
        self.logger.error(f"出力ファイル生成エラー: {e}", exc_info=True)
```

---

## 8. リスク・課題

### 8.1 データ不足リスク

**リスク**: `daily_result`に13カラム追加が必要だが、現在のDSSMSでは追跡していないデータがある。

**対策**:
1. `_process_daily_trading()`修正で全カラム実装
2. 初期値/デフォルト値を設定（実装優先度に応じて段階的実装可能）

---

### 8.2 パフォーマンスリスク

**リスク**: ComprehensiveReporterの再利用により、DSSMS実行時間が増加する可能性。

**対策**:
1. レポート生成は最終段階のみ（日次ではない）
2. パフォーマンス測定を統合テストで実施

---

### 8.3 後方互換性リスク

**リスク**: main_new.py修正により既存のバックテストに影響する可能性。

**対策**:
1. `suppress_report_generation`のデフォルト値を`False`に設定
2. 比較テストで従来の動作を検証

---

## 9. 次のステップ

**Phase 1完了**: 詳細設計書作成完了

**Phase 2開始条件**: ユーザー承認

**Phase 2実装順序**:
1. 出力制御実装（最優先、1時間）
2. daily_result拡張実装（3時間）
3. データ変換処理実装（4時間）
4. CSV生成処理実装（2時間）
5. DSSMS特有ファイル実装（2時間）

**Phase 3開始条件**: Phase 2完了

---

## 10. 参照

- **前回調査報告**: `docs/DSSMS_OUTPUT_INVESTIGATION_REPORT.md`（2025-12-03作成済み想定）
- **copilot-instructions.md**: `.github/copilot-instructions.md`
- **関連ファイル**:
  - `src/dssms/dssms_integrated_main.py`
  - `main_new.py`
  - `main_system/reporting/comprehensive_reporter.py`
  - `main_system/execution_control/integrated_execution_manager.py`

---

**設計書作成完了日時**: 2025-12-03  
**次回アクション**: ユーザー承認待ち → Phase 2実装開始

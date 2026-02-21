# エグジット戦略ゼロベース再設計ガイド v2.0

**作成日**: 2026年1月22日  
**目的**: Phase 3-6の失敗を踏まえ、カーブフィッティングを回避した新規エグジット戦略開発プロトコルを確立  
**対象読者**: バックテスト実装者、戦略開発者  
**関連ドキュメント**: PHASE6_PARAMETER_REFITTING_ANALYSIS.md, EXIT_STRATEGY_VALIDATION_REPORT.md

---

## 📋 目次

1. [Phase 3-6失敗の総括](#phase-3-6失敗の総括)
2. [カーブフィッティング回避の7原則](#カーブフィッティング回避の7原則)
3. [シンプルエグジット戦略の定義](#シンプルエグジット戦略の定義)
4. [段階的検証プロトコル](#段階的検証プロトコル)
5. [再利用可能なスクリプト](#再利用可能なスクリプト)
6. [推奨パラメータ空間](#推奨パラメータ空間)
7. [実装例](#実装例)

---

## 🚨 Phase 3-6失敗の総括

### 根本的な設計ミス

| 項目 | Phase 3-6の問題 | 影響 |
|------|------------------|------|
| **単一銘柄最適化** | 7203.T（トヨタ自動車）のみで最適化 | 銘柄依存、汎化性能ゼロ |
| **短期データ最適化** | 2022-2024（2年間）のみ | 市場環境依存、過学習 |
| **異常値PFの採用** | PF=121.07を「最優秀」と判定 | 2025年で-98.9%崩壊 |
| **パラメータ選択バイアス** | min_hold/max_hold/confidenceのみ検証 | trailing_stop/take_profit等を無視 |
| **Win Rate軽視** | PF最大化のみ追求 | エントリー品質（Win Rate 22.4%）を放置 |

### 失敗の証拠

**Aggressive(1/30/0.4)の性能推移**:
```
2022-2024（7203.T単一銘柄）: PF=121.07, Win Rate=95.5%, 取引数=44
2025年（3銘柄平均）      : PF=1.09,   Win Rate=22.4%, 取引数=34
→ -98.9%劣化（損益分岐点付近まで崩壊）
```

**教訓**: **極端に高いPF（> 3.0）は過学習の証拠**

---

## ✅ カーブフィッティング回避の7原則

### 原則1: 多銘柄同時検証（10銘柄以上）

**理由**: 単一銘柄最適化は銘柄依存を生む

**実装**:
```python
# 日経225から業種分散10銘柄選定
VALIDATION_TICKERS = [
    "7203.T",  # トヨタ自動車（自動車）
    "9984.T",  # ソフトバンクグループ（通信）
    "8306.T",  # 三菱UFJ FG（銀行）
    "6758.T",  # ソニーグループ（電機）
    "9983.T",  # ファーストリテイリング（小売）
    "6501.T",  # 日立製作所（電機）
    "8001.T",  # 伊藤忠商事（商社）
    "4063.T",  # 信越化学工業（化学）
    "4502.T",  # 武田薬品工業（医薬品）
    "6861.T"   # キーエンス（電機）
]
```

**成功基準**: 10銘柄中8銘柄以上でPF > 1.0

---

### 原則2: 長期データ検証（5年以上）

**理由**: 短期データは市場環境依存を生む

**実装**:
```python
# 2020-2024（5年間）でIn-Sample検証
BACKTEST_START = "2020-01-01"
BACKTEST_END = "2024-12-31"

# 2025年（1年間）でOut-of-Sample検証
OOS_START = "2025-01-01"
OOS_END = "2025-12-31"
```

**成功基準**: 5年間で年次PF分散 < 0.5（安定性）

---

### 原則3: PF上限制約（3.0以下）

**理由**: PF > 3.0は過学習の強い兆候

**実装**:
```python
def filter_overfit_params(results_df: pd.DataFrame, max_pf: float = 3.0) -> pd.DataFrame:
    """
    過学習パラメータをフィルタリング
    
    Args:
        results_df: バックテスト結果
        max_pf: PF上限（デフォルト3.0）
    
    Returns:
        フィルタリング後の結果
    """
    return results_df[
        (results_df['profit_factor'] > 1.0) &
        (results_df['profit_factor'] <= max_pf)
    ]
```

**成功基準**: 採用パラメータは全てPF 1.0-3.0の範囲内

---

### 原則4: Win Rate最低基準（40%以上）

**理由**: エントリー品質が低いとエグジット最適化は無意味

**実装**:
```python
# Win Rate < 40%のパラメータは除外
MINIMUM_WIN_RATE = 0.40

filtered = results_df[results_df['win_rate'] >= MINIMUM_WIN_RATE]
```

**成功基準**: 採用パラメータは全てWin Rate ≥ 40%

---

### 原則5: 統計的有意性確保（20取引以上）

**理由**: 少ない取引数では偶然の影響が大きい

**実装**:
```python
# 取引数 < 20のパラメータは除外
MINIMUM_TRADES = 20

filtered = results_df[results_df['num_trades'] >= MINIMUM_TRADES]
```

**成功基準**: 採用パラメータは全て取引数 ≥ 20

---

### 原則6: シンプルなパラメータ空間

**理由**: パラメータ数が多いと過学習リスク増大

**実装**:
```python
# Phase 1: シンプル2-3パラメータから開始
SIMPLE_PARAM_GRID = {
    'stop_loss_pct': [0.03, 0.05, 0.07],      # 損切3パターン
    'trailing_stop_pct': [0.05, 0.10, 0.15]   # トレーリング3パターン
}
# 合計: 3 × 3 = 9組み合わせ

# Phase 2: 有望な組み合わせのみ拡張
# Phase 3: 最終候補2-3パターンでOut-of-Sample検証
```

**成功基準**: 最終採用パラメータは3つ以下

---

### 原則7: Out-of-Sample検証必須

**理由**: In-Sampleのみの評価は過学習を検出不可

**実装**:
```python
# In-Sample（80%）とOut-of-Sample（20%）分割
IN_SAMPLE_END = "2023-12-31"
OOS_START = "2024-01-01"

# Out-of-Sample劣化率 < 30%を基準
oos_pf = backtest_oos(best_params)
degradation_rate = (in_sample_pf - oos_pf) / in_sample_pf

if degradation_rate > 0.30:
    print("⚠️ 警告: Out-of-Sampleで30%以上劣化、過学習の可能性")
```

**成功基準**: Out-of-Sample PF ≥ In-Sample PF × 0.7

---

## 🎯 シンプルエグジット戦略の定義

### Phase 1: 最小限エグジットルール

**目的**: PF > 1.0達成可能な最小限のルールを発見

**パラメータ**:
```python
PHASE1_PARAMS = {
    'stop_loss_pct': 0.05,          # 固定損切5%
    'trailing_stop_pct': 0.10,      # 固定トレーリング10%
    'take_profit_pct': None,        # 利確なし（トレンドを追う）
    'min_hold_days': None,          # 最小保有期間なし
    'max_hold_days': None           # 最大保有期間なし
}
```

**検証項目**:
- [ ] 10銘柄平均PF > 1.0
- [ ] 10銘柄中8銘柄でPF > 1.0
- [ ] 平均取引数 ≥ 20

**判定**:
- ✅ PASS → Phase 2へ進む
- ❌ FAIL → パラメータ微調整（損切3-7%、トレーリング5-15%で再試行）

---

### Phase 2: パラメータグリッドサーチ

**目的**: Phase 1で有望なルールを最適化

**パラメータ空間**:
```python
PHASE2_PARAM_GRID = {
    'stop_loss_pct': [0.03, 0.05, 0.07, 0.10],      # 損切4パターン
    'trailing_stop_pct': [0.05, 0.08, 0.10, 0.15],  # トレーリング4パターン
    'take_profit_pct': [None, 0.15, 0.20]           # 利確3パターン（なし含む）
}
# 合計: 4 × 4 × 3 = 48組み合わせ
```

**検証項目**:
- [ ] 10銘柄平均PF > 2.0
- [ ] 10銘柄平均Win Rate > 40%
- [ ] 最高PF ≤ 3.0（過学習排除）
- [ ] PF標準偏差 < 0.5（銘柄間安定性）

**成功基準**: TOP 3パラメータセットを選定

---

### Phase 3: Out-of-Sample最終検証

**目的**: TOP 3パラメータの汎化性能確認

**検証内容**:
```python
# In-Sample: 2020-2023（4年間）
# Out-of-Sample: 2024-2025（2年間）

for params in top3_params:
    oos_results = backtest(params, period="2024-2025")
    
    # 劣化率チェック
    degradation = (in_sample_pf - oos_pf) / in_sample_pf
    
    if degradation < 0.30:
        print(f"✅ {params}: Out-of-Sample劣化率{degradation:.1%}（許容範囲）")
    else:
        print(f"❌ {params}: Out-of-Sample劣化率{degradation:.1%}（過学習）")
```

**成功基準**: TOP 3中2パラメータ以上で劣化率 < 30%

---

## 📊 段階的検証プロトコル

### プロトコル全体像

```
Phase 1: シンプルルール検証（1週間）
  ├─ 目的: PF > 1.0達成可能性確認
  ├─ 銘柄: 10銘柄
  ├─ 期間: 5年（2020-2024）
  └─ パラメータ: 固定（損切5%、トレーリング10%）
  
Phase 2: グリッドサーチ（2週間）
  ├─ 目的: 最適パラメータ発見
  ├─ 銘柄: 10銘柄
  ├─ 期間: 4年（2020-2023, In-Sample）
  └─ パラメータ: 48組み合わせ
  
Phase 3: Out-of-Sample検証（1週間）
  ├─ 目的: 汎化性能確認
  ├─ 銘柄: 10銘柄
  ├─ 期間: 2年（2024-2025, Out-of-Sample）
  └─ パラメータ: TOP 3のみ
```

### Phase 1詳細手順

**Step 1-1: データ準備**
```bash
# 10銘柄×5年データ取得（yfinance + CSV cache）
python scripts/fetch_multi_ticker_data.py \
  --tickers 7203.T,9984.T,8306.T,6758.T,9983.T,6501.T,8001.T,4063.T,4502.T,6861.T \
  --start 2020-01-01 \
  --end 2024-12-31
```

**Step 1-2: Phase 1実行**
```bash
# シンプルルール検証
python scripts/validate_exit_simple_v2.py \
  --phase 1 \
  --stop-loss 0.05 \
  --trailing-stop 0.10
```

**Step 1-3: 結果評価**
```python
# 出力: results/phase1_simple_20260122.csv
results = pd.read_csv("results/phase1_simple_20260122.csv")

avg_pf = results['profit_factor'].mean()
pass_rate = (results['profit_factor'] > 1.0).sum() / len(results)

if avg_pf > 1.0 and pass_rate >= 0.8:
    print("✅ Phase 1成功 → Phase 2へ進む")
else:
    print("❌ Phase 1失敗 → パラメータ微調整")
```

---

### Phase 2詳細手順

**Step 2-1: グリッドサーチ実行**
```bash
python scripts/validate_exit_simple_v2.py \
  --phase 2 \
  --grid-search
```

**Step 2-2: TOP 3選定**
```python
# 過学習フィルタリング
filtered = results[
    (results['profit_factor'] > 1.0) &
    (results['profit_factor'] <= 3.0) &
    (results['win_rate'] >= 0.40) &
    (results['num_trades'] >= 20)
]

# 複合スコアで順位付け
filtered['score'] = (
    filtered['profit_factor'] * 0.4 +
    filtered['win_rate'] * 100 * 0.3 +
    (1 / filtered['pf_std']) * 0.3  # 銘柄間安定性
)

top3 = filtered.nlargest(3, 'score')
```

**Step 2-3: TOP 3詳細レポート**
```python
for idx, row in top3.iterrows():
    print(f"\n【候補{idx+1}】")
    print(f"  stop_loss: {row['stop_loss_pct']:.1%}")
    print(f"  trailing_stop: {row['trailing_stop_pct']:.1%}")
    print(f"  平均PF: {row['profit_factor']:.2f}")
    print(f"  平均Win Rate: {row['win_rate']:.1%}")
    print(f"  PF標準偏差: {row['pf_std']:.2f}")
```

---

### Phase 3詳細手順

**Step 3-1: Out-of-Sample実行**
```bash
python scripts/validate_exit_simple_v2.py \
  --phase 3 \
  --params-file results/phase2_top3.json \
  --oos-start 2024-01-01 \
  --oos-end 2025-12-31
```

**Step 3-2: 劣化率分析**
```python
for params in top3_params:
    in_sample_pf = phase2_results[params]['pf']
    oos_pf = phase3_results[params]['pf']
    
    degradation = (in_sample_pf - oos_pf) / in_sample_pf
    
    status = "✅ 合格" if degradation < 0.30 else "❌ 過学習"
    print(f"{params}: In-Sample PF={in_sample_pf:.2f}, "
          f"OOS PF={oos_pf:.2f}, 劣化率={degradation:.1%} {status}")
```

**Step 3-3: 最終推奨**
```python
# 劣化率 < 30%のパラメータを推奨
recommended = [p for p in top3_params if degradation[p] < 0.30]

if len(recommended) >= 2:
    print(f"✅ Phase 3成功: {len(recommended)}パラメータがOut-of-Sample通過")
    print(f"推奨パラメータ: {recommended[0]}")
else:
    print("❌ Phase 3失敗: 汎化性能不足、Phase 2再実施推奨")
```

---

## 💻 再利用可能なスクリプト

### validate_exit_simple_v2.py

**用途**: シンプルエグジット戦略の段階的検証

**機能**:
- Phase 1: 固定パラメータ検証
- Phase 2: グリッドサーチ（48組み合わせ）
- Phase 3: Out-of-Sample検証
- 10銘柄対応
- PF上限3.0制約
- 統計的有意性チェック

**使用例**:
```bash
# Phase 1実行
python scripts/validate_exit_simple_v2.py --phase 1

# Phase 2実行（Full版）
python scripts/validate_exit_simple_v2.py --phase 2 --grid-search

# Phase 3実行
python scripts/validate_exit_simple_v2.py --phase 3 --top3-file results/phase2_top3.json
```

**出力**:
- `results/phase1_simple_YYYYMMDD_HHMMSS.csv`: Phase 1結果
- `results/phase2_grid_YYYYMMDD_HHMMSS.csv`: Phase 2結果
- `results/phase3_oos_YYYYMMDD_HHMMSS.csv`: Phase 3結果
- `results/phase2_top3.json`: TOP 3パラメータ

---

### SimpleExitStrategy クラス

**用途**: エントリー固定、エグジット単体検証用の基底クラス

**実装**:
```python
class SimpleExitStrategy(BaseStrategy):
    """
    シンプルエグジット戦略
    
    エントリー: 固定（GC戦略など）
    エグジット: 損切 + トレーリング + 利確（オプション）
    """
    
    def __init__(self, data: pd.DataFrame, params: dict):
        super().__init__(data, params)
        
        self.stop_loss_pct = params.get('stop_loss_pct', 0.05)
        self.trailing_stop_pct = params.get('trailing_stop_pct', 0.10)
        self.take_profit_pct = params.get('take_profit_pct', None)
    
    def generate_exit_signal(self, idx: int, entry_idx: int = -1) -> int:
        """
        エグジット判定
        
        Args:
            idx: 現在のインデックス
            entry_idx: エントリーインデックス
        
        Returns:
            1（エグジット）または0（保有継続）
        """
        if entry_idx == -1:
            return 0
        
        entry_price = self.data['Adj Close'].iloc[entry_idx]
        current_price = self.data['Adj Close'].iloc[idx]
        
        # 損切チェック
        if current_price <= entry_price * (1 - self.stop_loss_pct):
            self.logger.info(f"損切エグジット: {current_price:.2f} "
                           f"<= {entry_price * (1 - self.stop_loss_pct):.2f}")
            return 1
        
        # トレーリングストップチェック
        max_price = self.data['Adj Close'].iloc[entry_idx:idx+1].max()
        if current_price <= max_price * (1 - self.trailing_stop_pct):
            self.logger.info(f"トレーリングストップ: {current_price:.2f} "
                           f"<= {max_price * (1 - self.trailing_stop_pct):.2f}")
            return 1
        
        # 利確チェック（オプション）
        if self.take_profit_pct is not None:
            if current_price >= entry_price * (1 + self.take_profit_pct):
                self.logger.info(f"利確エグジット: {current_price:.2f} "
                               f">= {entry_price * (1 + self.take_profit_pct):.2f}")
                return 1
        
        return 0
```

---

## 📐 推奨パラメータ空間

### Phase 1推奨値

| パラメータ | 推奨値 | 範囲 | 理由 |
|------------|--------|------|------|
| stop_loss_pct | 0.05 | 3-7% | 一般的な損切水準 |
| trailing_stop_pct | 0.10 | 5-15% | トレンド追従とリスク管理のバランス |
| take_profit_pct | None | - | トレンドを最後まで追う |
| min_hold_days | None | - | エグジット条件のみで判断 |
| max_hold_days | None | - | エグジット条件のみで判断 |

---

### Phase 2推奨グリッド

```python
PHASE2_RECOMMENDED_GRID = {
    # 損切: 保守的→中立→攻撃的
    'stop_loss_pct': [0.03, 0.05, 0.07, 0.10],
    
    # トレーリング: タイト→標準→ルーズ
    'trailing_stop_pct': [0.05, 0.08, 0.10, 0.15],
    
    # 利確: なし（トレンド追従）、標準、保守的
    'take_profit_pct': [None, 0.15, 0.20]
}
```

**組み合わせ数**: 4 × 4 × 3 = 48

**推定実行時間**: 約30-60分（10銘柄×48パラメータ=480検証）

---

### Phase 2拡張グリッド（オプション）

```python
PHASE2_EXTENDED_GRID = {
    'stop_loss_pct': [0.02, 0.03, 0.05, 0.07, 0.10],           # 5パターン
    'trailing_stop_pct': [0.05, 0.07, 0.10, 0.12, 0.15],       # 5パターン
    'take_profit_pct': [None, 0.10, 0.15, 0.20, 0.25],         # 5パターン
    'min_hold_days': [None, 1, 3, 5],                          # 4パターン
    'max_hold_days': [None, 30, 60, 90]                        # 4パターン
}
```

**組み合わせ数**: 5 × 5 × 5 × 4 × 4 = 2,000

**推定実行時間**: 約8-12時間（10銘柄×2,000パラメータ=20,000検証）

**推奨**: Phase 2標準グリッドで有望な範囲を特定後に実施

---

## 🔧 実装例

### 例1: Phase 1最小限実行

```python
from strategies.simple_exit_strategy import SimpleExitStrategy
from data_fetcher import get_parameters_and_data

# データ取得
ticker = "7203.T"
stock_data, _ = get_parameters_and_data(
    ticker=ticker,
    start_date="2020-01-01",
    end_date="2024-12-31",
    warmup_days=150
)

# Phase 1パラメータ
params = {
    'stop_loss_pct': 0.05,
    'trailing_stop_pct': 0.10,
    'take_profit_pct': None
}

# 戦略実行
strategy = SimpleExitStrategy(stock_data, params)
results = strategy.backtest()

# 結果出力
print(f"PF: {results['profit_factor']:.2f}")
print(f"Win Rate: {results['win_rate']:.1%}")
print(f"取引数: {results['num_trades']}")
```

---

### 例2: Phase 2グリッドサーチ

```python
import itertools
import pandas as pd

# パラメータグリッド
param_grid = {
    'stop_loss_pct': [0.03, 0.05, 0.07, 0.10],
    'trailing_stop_pct': [0.05, 0.08, 0.10, 0.15],
    'take_profit_pct': [None, 0.15, 0.20]
}

# 10銘柄
tickers = ["7203.T", "9984.T", "8306.T", "6758.T", "9983.T", 
           "6501.T", "8001.T", "4063.T", "4502.T", "6861.T"]

# グリッドサーチ
results = []
param_combinations = list(itertools.product(*param_grid.values()))

for params_tuple in param_combinations:
    params = dict(zip(param_grid.keys(), params_tuple))
    
    ticker_results = []
    for ticker in tickers:
        stock_data, _ = get_parameters_and_data(ticker, "2020-01-01", "2023-12-31")
        strategy = SimpleExitStrategy(stock_data, params)
        result = strategy.backtest()
        ticker_results.append(result)
    
    # 平均パフォーマンス計算
    avg_pf = np.mean([r['profit_factor'] for r in ticker_results])
    avg_wr = np.mean([r['win_rate'] for r in ticker_results])
    
    results.append({
        **params,
        'avg_pf': avg_pf,
        'avg_win_rate': avg_wr,
        'pf_std': np.std([r['profit_factor'] for r in ticker_results])
    })

# 結果保存
results_df = pd.DataFrame(results)
results_df.to_csv("results/phase2_grid_20260122.csv", index=False)

# 過学習フィルタリング
filtered = results_df[
    (results_df['avg_pf'] > 1.0) &
    (results_df['avg_pf'] <= 3.0) &
    (results_df['avg_win_rate'] >= 0.40)
]

# TOP 3選定
top3 = filtered.nlargest(3, 'avg_pf')
print("\n【TOP 3パラメータ】")
print(top3)
```

---

### 例3: Out-of-Sample検証

```python
# Phase 2 TOP 3パラメータ
top3_params = [
    {'stop_loss_pct': 0.05, 'trailing_stop_pct': 0.10, 'take_profit_pct': None},
    {'stop_loss_pct': 0.03, 'trailing_stop_pct': 0.08, 'take_profit_pct': 0.15},
    {'stop_loss_pct': 0.07, 'trailing_stop_pct': 0.15, 'take_profit_pct': None}
]

# In-Sample結果（Phase 2）
in_sample_results = {
    'params_1': {'pf': 2.35, 'win_rate': 0.52},
    'params_2': {'pf': 2.18, 'win_rate': 0.48},
    'params_3': {'pf': 2.42, 'win_rate': 0.50}
}

# Out-of-Sample検証
oos_results = {}
for i, params in enumerate(top3_params):
    ticker_results = []
    
    for ticker in tickers:
        stock_data, _ = get_parameters_and_data(ticker, "2024-01-01", "2025-12-31")
        strategy = SimpleExitStrategy(stock_data, params)
        result = strategy.backtest()
        ticker_results.append(result)
    
    avg_pf = np.mean([r['profit_factor'] for r in ticker_results])
    oos_results[f'params_{i+1}'] = {'pf': avg_pf}

# 劣化率分析
for key in in_sample_results:
    in_pf = in_sample_results[key]['pf']
    oos_pf = oos_results[key]['pf']
    degradation = (in_pf - oos_pf) / in_pf
    
    status = "✅" if degradation < 0.30 else "❌"
    print(f"{key}: In-Sample={in_pf:.2f}, OOS={oos_pf:.2f}, "
          f"劣化率={degradation:.1%} {status}")
```

---

## ⚠️ よくある失敗パターン

### 失敗1: PF最大化のみ追求

**症状**: Phase 2でPF=5.0超えのパラメータを「最優秀」と判定

**原因**: 過学習、異常値を成功と誤認

**対策**: PF上限3.0制約を必ず適用

---

### 失敗2: 単一銘柄最適化

**症状**: 7203.Tのみでパラメータ最適化 → 他銘柄で崩壊

**原因**: 銘柄依存、汎化性能ゼロ

**対策**: 10銘柄同時検証を必須とする

---

### 失敗3: 短期データ最適化

**症状**: 2022-2024（2年間）でPF=100超え → 2025年でPF=1.0

**原因**: 市場環境依存、上昇トレンドのみで機能

**対策**: 5年以上のデータで検証、Out-of-Sample必須

---

### 失敗4: Win Rate軽視

**症状**: Win Rate=22.4%でもエグジット最適化を継続

**原因**: エントリー品質の低さを無視

**対策**: Win Rate < 40%ならエントリー戦略見直しを優先

---

### 失敗5: 統計的有意性無視

**症状**: 取引数2-3でPF=30を「成功」と判定

**原因**: サンプルサイズ不足、偶然の影響

**対策**: 取引数 < 20のパラメータは除外

---

## 📚 関連ドキュメント

1. **PHASE6_PARAMETER_REFITTING_ANALYSIS.md**: Phase 3-6失敗の詳細分析
2. **EXIT_STRATEGY_VALIDATION_REPORT.md**: Phase 3検証レポート（PF=121.07異常値記録）
3. **EXIT_STRATEGY_SEPARATION_DESIGN.md**: エントリー・エグジット分離設計

---

## 🚀 次のステップ

### ユーザー判断事項

1. **Phase 1実施判断**
   - [ ] Phase 1実行（シンプルルール、10銘柄、5年データ）
   - [ ] Phase 1スキップ、Phase 2直接実施

2. **Phase 2グリッド規模**
   - [ ] 標準グリッド（48組み合わせ、30-60分）
   - [ ] 拡張グリッド（2,000組み合わせ、8-12時間）

3. **Phase 3実施タイミング**
   - [ ] Phase 2完了後すぐ実施
   - [ ] Phase 2 TOP 3確認後ユーザー判断

---

## 📊 期待成果

### Phase 1-3完了時の成果物

1. **推奨パラメータ**: 汎化性能確認済み、PF 2.0-3.0、Win Rate > 40%
2. **10銘柄検証結果**: 銘柄依存性排除、業種分散確認
3. **Out-of-Sample実績**: 劣化率 < 30%、過学習排除
4. **再利用可能なスクリプト**: 新規エグジット戦略開発用テンプレート

### リアルトレード移行判断

**条件**:
- ✅ Phase 3で2パラメータ以上がOut-of-Sample通過
- ✅ 推奨パラメータのPF 2.0-3.0、Win Rate > 40%
- ✅ 10銘柄中8銘柄でPF > 1.0

**移行時期**: 2026年2月以降（Phase 3完了後1ヶ月間のペーパートレード推奨）

---

**作成者**: Backtest Project Team  
**バージョン**: 2.0  
**最終更新**: 2026年1月22日  
**ステータス**: 実装準備完了

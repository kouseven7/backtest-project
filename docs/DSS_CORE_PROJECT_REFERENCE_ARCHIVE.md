# アーカイブ DSS Core Project 開発ガイド
**DSS（Dynamic Stock Selector）バックテスター V3 構築リファレンス**

## プロジェクト概要

**目標**: DSS基本機能（銘柄選択エンジン）に特化したシンプルなバックテスター作成  
**成果物**: `src/dssms/dssms_backtester_v3.py`  
**設計思想**: 毎日のランキング更新 → パーフェクトオーダー1位銘柄選定 → 結果出力

## DSS Core 5段階フロー

### **1. データを取得する**
**概要**: 複数銘柄の市場データを並列取得
```python
# ハードコード銘柄リスト（シンプル開始）
symbol_universe = ['7203', '9984', '6758', '4063', '8306', '6861', '7741', '9432', '8058', '9020']
```
**関連ファイル・パス：**
- `src/dssms/dssms_data_manager.py` - DSSMS専用データ管理
- `config/dssms/nikkei225_components.json` - 日経225銘柄リスト（将来拡張用）
- `data_fetcher.py` (フォールバック用)

**V3実装ポイント**:
- 初期版では10銘柄程度に限定
- yfinanceによる直接取得（DSSMSDataManagerは後回し）
- エラーハンドリングはシンプルに

### **2. データを処理する**
**概要**: 取得データの品質チェックと前処理
**関連ファイル・パス：**
- `src/dssms/data_quality_enhancer.py` - データ品質向上（オプショナル）
- `src/dssms/data_cleaning_engine.py` - データクリーニング（オプショナル）

**V3実装ポイント**:
- 最小限の処理（欠損値チェック程度）
- 移動平均計算（5日、25日、75日）のみ実装

### **3. トレンド判断をする（パーフェクトオーダー）**
**概要**: 移動平均の配置によるトレンド強度判定
**関連ファイル・パス：**
- `src/dssms/perfect_order_detector.py` - パーフェクトオーダー検出

**V3実装ポイント**:
```python
def detect_perfect_order(self, symbol_data: pd.DataFrame) -> float:
    """
    パーフェクトオーダー判定
    Returns: 0.0-1.0 のスコア（1.0が最強の上昇トレンド）
    """
    # MA5 > MA25 > MA75 and 価格 > MA5 = 完璧なパーフェクトオーダー
    # 段階的にスコア化
```

### **4. ランキングする**
**概要**: 各銘柄のパーフェクトオーダースコアで順位付け
**関連ファイル・パス：**
- `src/dssms/hierarchical_ranking_system.py` - 階層的ランキング（参考のみ）
- `src/dssms/comprehensive_scoring_engine.py` - 総合スコアリング（参考のみ）

**V3実装ポイント**:
```python
def rank_symbols(self, symbol_data_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
    """
    銘柄ランキング実行
    Returns: [{'symbol': '7203', 'score': 0.95, 'rank': 1}, ...]
    """
    # シンプルなスコア順ソート
```

### **5. 上位ランキングの銘柄を選択する**
**概要**: ランキング1位の銘柄を抽出
**関連ファイル・パス：**
- `src/dssms/intelligent_switch_manager.py` - インテリジェント切替管理（参考のみ）

**V3実装ポイント**:
```python
def select_top_symbol(self, ranking_result: List[Dict]) -> str:
    """
    最上位銘柄選択
    Returns: 銘柄コード（例: '7203'）
    """
    return ranking_result[0]['symbol']  # 1位を単純選択
```

## dssms_backtester_v3.py 実装仕様

### クラス設計
```python
class DSSBacktesterV3:
    """
    DSS（Dynamic Stock Selector）バックテスター V3
    
    目標: 銘柄選択エンジンとしての機能に特化
    範囲: データ取得 → パーフェクトオーダー判定 → ランキング → 1位選択
    """
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.symbol_universe = ['7203', '9984', '6758', '4063', '8306']  # 初期5銘柄
        
    def run_daily_selection(self, target_date: datetime) -> Dict[str, Any]:
        """日次銘柄選択実行（メイン処理）"""
        
    def fetch_market_data(self, symbols: List[str], date: datetime) -> Dict[str, pd.DataFrame]:
        """市場データ取得"""
        
    def calculate_perfect_order_scores(self, market_data: Dict) -> Dict[str, float]:
        """パーフェクトオーダースコア計算"""
        
    def rank_symbols(self, scores: Dict[str, float]) -> List[Dict]:
        """銘柄ランキング"""
        
    def select_top_symbol(self, ranking: List[Dict]) -> str:
        """1位銘柄選択"""
```

### 実装範囲と制限

[OK] **V3で実装**:
1. **データ取得**: yfinance直接利用（5-10銘柄）
2. **パーフェクトオーダー判定**: MA5>MA25>MA75 判定
3. **ランキング**: スコア順ソート
4. **1位選択**: ランキング[0]の単純選択
5. **結果出力**: JSON/Dict形式

[ERROR] **V3で実装しない**:
- 取引シミュレーション機能
- ポートフォリオ価値計算
- 損益管理
- 複雑なエラーハンドリング
- Excel出力機能
- 決定論モード
- 統計計算エンジン

### 開発順序
1. **Phase 1**: 基本クラス作成 + ダミーデータでのテスト
2. **Phase 2**: yfinanceによる実データ取得
3. **Phase 3**: パーフェクトオーダー判定ロジック
4. **Phase 4**: ランキング・選択機能
5. **Phase 5**: 日次実行ループ + 結果出力

### 注意事項・制約
- **シンプル第一**: 複雑な機能は後回し
- **エラーは最小限**: try-except は必要最小限
- **ログ出力重視**: 各段階の処理状況を明確にログ
- **テスト可能**: 各メソッドが独立してテスト可能
- **将来拡張性**: マルチ戦略システム統合を意識した設計

---

## 既存システムとの関係

### 参考にする既存ファイル
- `src/dssms/dssms_backtester.py` - 複雑すぎるが、一部のロジック参考
- `src/dssms/perfect_order_detector.py` - パーフェクトオーダー判定参考
- `data_fetcher.py` - データ取得方法参考

### 既存ファイルで除外する機能
<details>
<summary>🚫 以下は既存DSSMSの取引実行部分（V3では除外）</summary>

- ~~独自の取引シミュレーション機能~~
- ~~ポートフォリオ価値管理~~
- ~~Entry_Signal/Exit_Signal生成~~
- ~~戦略名ローテーション機能~~
- ~~統計計算エンジン~~
- ~~決定論モード~~

**理由**: DSS Core はランキング1位選択までに特化

</details>

---

## サンプルコード実装例

### メインクラス構造
```python
class DSSBacktesterV3:
    """DSS（Dynamic Stock Selector）バックテスター V3"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        # シンプルな5銘柄で開始
        self.symbol_universe = ['7203', '9984', '6758', '4063', '8306']
        
    def run_daily_selection(self, target_date: datetime) -> Dict[str, Any]:
        """
        日次銘柄選択実行（DSS Core メイン処理）
        
        Returns:
            {
                'date': datetime,
                'selected_symbol': str,
                'ranking': List[Dict],
                'execution_time_ms': float
            }
        """
        start_time = time.time()
        
        # 1. データ取得
        market_data = self.fetch_market_data(self.symbol_universe, target_date)
        
        # 2. パーフェクトオーダースコア計算
        scores = self.calculate_perfect_order_scores(market_data)
        
        # 3. ランキング
        ranking = self.rank_symbols(scores)
        
        # 4. 1位選択
        selected_symbol = self.select_top_symbol(ranking)
        
        execution_time = (time.time() - start_time) * 1000
        
        return {
            'date': target_date,
            'selected_symbol': selected_symbol,
            'ranking': ranking,
            'execution_time_ms': execution_time
        }
```

### パーフェクトオーダー判定例
```python
def calculate_perfect_order_score(self, data: pd.DataFrame) -> float:
    """
    パーフェクトオーダースコア計算
    
    Perfect Order: MA5 > MA25 > MA75 かつ 価格 > MA5
    
    Returns:
        0.0-1.0 のスコア（1.0が完璧なパーフェクトオーダー）
    """
    if len(data) < 75:  # 75日移動平均に必要なデータ不足
        return 0.0
        
    # 移動平均計算
    ma5 = data['Close'].rolling(5).mean().iloc[-1]
    ma25 = data['Close'].rolling(25).mean().iloc[-1]
    ma75 = data['Close'].rolling(75).mean().iloc[-1]
    current_price = data['Close'].iloc[-1]
    
    # パーフェクトオーダー判定
    score = 0.0
    
    if current_price > ma5:
        score += 0.25
    if ma5 > ma25:
        score += 0.25
    if ma25 > ma75:
        score += 0.25
    
    # ボーナス: すべての条件を満たす場合
    if score == 0.75:
        score = 1.0
        
    return score
```

### 実行例
```python
# DSS V3 実行例
def main():
    dss = DSSBacktesterV3()
    
    # 単日実行
    result = dss.run_daily_selection(datetime(2023, 1, 15))
    print(f"選択銘柄: {result['selected_symbol']}")
    print(f"実行時間: {result['execution_time_ms']:.1f}ms")
    
    # 期間実行（将来拡張）
    # results = dss.run_period_selection(start_date, end_date)
```

---

## 開発チェックリスト

### Phase 1: 基本実装
- [ ] クラス構造作成
- [ ] ログシステム設定
- [ ] 銘柄リスト定義
- [ ] ダミーデータでの動作確認

### Phase 2: データ取得
- [ ] yfinance連携
- [ ] データ品質チェック
- [ ] エラーハンドリング

### Phase 3: パーフェクトオーダー
- [ ] 移動平均計算
- [ ] スコア算出ロジック
- [ ] 境界ケース対応

### Phase 4: ランキング・選択
- [ ] スコア順ソート
- [ ] 1位選択機能
- [ ] 結果データ構造定義

### Phase 5: 統合・テスト
- [ ] 全体フロー確認
- [ ] パフォーマンステスト
- [ ] ログ出力検証